from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .types import BBox, TrackedDetection


@dataclass
class PersistentTrack:
    persistent_id: int
    bbox: BBox
    confidence: float
    class_id: int

    byte_track_id: int

    last_frame_id: int
    missed_frames: int

    velocity_x: float = 0.0
    velocity_y: float = 0.0

    appearance: np.ndarray | None = None


class PersistentMOT:
    """
    Persistent identity layer on top of ByteTrack.

    ByteTrack handles short-term motion + IoU tracking.

    This layer maintains application-level persistent IDs
    even when ByteTrack temporarily loses or changes an ID.
    """

    def __init__(
        self,
        max_missed_frames: int = 150,
        appearance_threshold: float = 0.68,
    ) -> None:

        self.max_missed_frames = max_missed_frames
        self.appearance_threshold = appearance_threshold

        self._tracks: Dict[int, PersistentTrack] = {}

        self._next_persistent_id = 0

    # ---------------------------------------------------------
    # PUBLIC
    # ---------------------------------------------------------

    def update(
        self,
        detections: List[TrackedDetection],
        frame: np.ndarray,
        frame_id: int,
    ) -> List[PersistentTrack]:

        if not detections:

            self._update_missing_tracks(frame_id)

            return []

        candidates = []

        for detection in detections:

            appearance = self._extract_appearance(
                frame,
                detection.bbox,
            )

            candidates.append(
                (
                    detection,
                    appearance,
                )
            )

        # -----------------------------------------------------
        # Build possible matches
        # -----------------------------------------------------

        matches = []

        for persistent_id, track in self._tracks.items():

            predicted_bbox = self._predict_bbox(
                track,
                frame_id,
            )

            for detection_index, (
                detection,
                appearance,
            ) in enumerate(candidates):

                score = self._calculate_match_score(
                    track=track,
                    predicted_bbox=predicted_bbox,
                    detection=detection,
                    appearance=appearance,
                    frame_id=frame_id,
                )

                if score is None:
                    continue

                matches.append(
                    (
                        score,
                        persistent_id,
                        detection_index,
                    )
                )

        # Highest score first
        matches.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        matched_tracks = set()
        matched_detections = set()

        # -----------------------------------------------------
        # Greedy association
        # -----------------------------------------------------

        for (
            score,
            persistent_id,
            detection_index,
        ) in matches:

            if persistent_id in matched_tracks:
                continue

            if detection_index in matched_detections:
                continue

            detection, appearance = candidates[
                detection_index
            ]

            track = self._tracks[persistent_id]

            # Additional safety threshold
            if score < 0.40:
                continue

            self._update_existing_track(
                track=track,
                detection=detection,
                appearance=appearance,
                frame_id=frame_id,
            )

            matched_tracks.add(
                persistent_id
            )

            matched_detections.add(
                detection_index
            )

        # -----------------------------------------------------
        # Unmatched detections = new persistent tracks
        # -----------------------------------------------------

        for detection_index, (
            detection,
            appearance,
        ) in enumerate(candidates):

            if detection_index in matched_detections:
                continue

            persistent_id = self._next_persistent_id

            self._next_persistent_id += 1

            self._tracks[persistent_id] = (
                PersistentTrack(
                    persistent_id=persistent_id,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    byte_track_id=detection.track_id,
                    last_frame_id=frame_id,
                    missed_frames=0,
                    appearance=appearance,
                )
            )

        # -----------------------------------------------------
        # Update missing tracks
        # -----------------------------------------------------

        for persistent_id, track in list(
            self._tracks.items()
        ):

            if persistent_id in matched_tracks:
                continue

            # If this track was just created this frame,
            # don't mark it missing.
            if track.last_frame_id == frame_id:
                continue

            track.missed_frames = (
                frame_id - track.last_frame_id
            )

            if (
                track.missed_frames
                > self.max_missed_frames
            ):
                del self._tracks[persistent_id]

        # -----------------------------------------------------
        # Return currently visible persistent tracks
        # -----------------------------------------------------

        visible_tracks = []

        for persistent_id in sorted(
            matched_tracks
        ):

            track = self._tracks.get(
                persistent_id
            )

            if track is not None:
                visible_tracks.append(track)

        # Include newly created tracks
        for detection_index, (
            detection,
            appearance,
        ) in enumerate(candidates):

            if detection_index in matched_detections:
                continue

            # Find the persistent track created above
            for track in self._tracks.values():

                if (
                    track.last_frame_id == frame_id
                    and track.byte_track_id
                    == detection.track_id
                    and track.bbox
                    == detection.bbox
                ):
                    visible_tracks.append(track)
                    break

        return sorted(
            visible_tracks,
            key=lambda x: x.persistent_id,
        )

    def reset(self) -> None:
        self._tracks.clear()
        self._next_persistent_id = 0

    # ---------------------------------------------------------
    # MATCHING
    # ---------------------------------------------------------

    def _calculate_match_score(
        self,
        track: PersistentTrack,
        predicted_bbox: BBox,
        detection: TrackedDetection,
        appearance: np.ndarray,
        frame_id: int,
    ) -> float | None:

        iou = self._iou(
            predicted_bbox,
            detection.bbox,
        )

        appearance_similarity = (
            self._cosine_similarity(
                track.appearance,
                appearance,
            )
        )

        motion_similarity = (
            self._motion_similarity(
                predicted_bbox,
                detection.bbox,
            )
        )

        byte_id_match = (
            track.byte_track_id
            == detection.track_id
        )

        frame_gap = (
            frame_id - track.last_frame_id
        )

        # -----------------------------------------------------
        # Spatial gating
        # -----------------------------------------------------

        predicted_center = self._center(
            predicted_bbox
        )

        detection_center = self._center(
            detection.bbox
        )

        distance = float(
            np.linalg.norm(
                np.asarray(predicted_center)
                - np.asarray(detection_center)
            )
        )

        bbox_width = max(
            1.0,
            predicted_bbox[2] - predicted_bbox[0],
        )

        bbox_height = max(
            1.0,
            predicted_bbox[3] - predicted_bbox[1],
        )

        bbox_diagonal = (
            bbox_width**2 + bbox_height**2
        ) ** 0.5

        max_distance = max(
            150.0,
            bbox_diagonal * 4.0,
        )

        if distance > max_distance:
            # Appearance can still rescue a long-gap track,
            # but only if similarity is very strong.
            if appearance_similarity < 0.82:
                return None

        # -----------------------------------------------------
        # Long disappearance = stricter appearance requirement
        # -----------------------------------------------------

        if frame_gap > 45:
            if appearance_similarity < 0.78:
                return None

        # -----------------------------------------------------
        # Calculate final association score
        # -----------------------------------------------------

        score = (
            0.35 * iou
            + 0.35 * appearance_similarity
            + 0.20 * motion_similarity
            + 0.10 * float(byte_id_match)
        )

        # -----------------------------------------------------
        # Don't allow completely unrelated objects
        # -----------------------------------------------------

        if (
            iou < 0.03
            and appearance_similarity
            < self.appearance_threshold
            and not byte_id_match
        ):
            return None

        return float(score)

    # ---------------------------------------------------------
    # TRACK UPDATE
    # ---------------------------------------------------------

    def _update_existing_track(
        self,
        track: PersistentTrack,
        detection: TrackedDetection,
        appearance: np.ndarray,
        frame_id: int,
    ) -> None:

        old_center = self._center(
            track.bbox
        )

        new_center = self._center(
            detection.bbox
        )

        frame_gap = max(
            1,
            frame_id - track.last_frame_id,
        )

        instant_velocity_x = (
            new_center[0] - old_center[0]
        ) / frame_gap

        instant_velocity_y = (
            new_center[1] - old_center[1]
        ) / frame_gap

        # Smooth velocity
        track.velocity_x = (
            0.70 * track.velocity_x
            + 0.30 * instant_velocity_x
        )

        track.velocity_y = (
            0.70 * track.velocity_y
            + 0.30 * instant_velocity_y
        )

        track.bbox = detection.bbox

        track.confidence = detection.confidence

        track.class_id = detection.class_id

        track.byte_track_id = detection.track_id

        track.last_frame_id = frame_id

        track.missed_frames = 0

        # Exponential moving average for appearance
        if track.appearance is None:
            track.appearance = appearance
        else:
            updated = (
                0.80 * track.appearance
                + 0.20 * appearance
            )

            norm = np.linalg.norm(updated)

            if norm > 1e-8:
                updated = updated / norm

            track.appearance = updated

    # ---------------------------------------------------------
    # MISSING TRACKS
    # ---------------------------------------------------------

    def _update_missing_tracks(
        self,
        frame_id: int,
    ) -> None:

        for persistent_id, track in list(
            self._tracks.items()
        ):

            track.missed_frames = (
                frame_id - track.last_frame_id
            )

            if (
                track.missed_frames
                > self.max_missed_frames
            ):
                del self._tracks[
                    persistent_id
                ]

    # ---------------------------------------------------------
    # MOTION
    # ---------------------------------------------------------

    @staticmethod
    def _predict_bbox(
        track: PersistentTrack,
        frame_id: int,
    ) -> BBox:

        gap = max(
            0,
            frame_id - track.last_frame_id,
        )

        dx = track.velocity_x * gap
        dy = track.velocity_y * gap

        x1, y1, x2, y2 = track.bbox

        return (
            x1 + dx,
            y1 + dy,
            x2 + dx,
            y2 + dy,
        )

    @staticmethod
    def _motion_similarity(
        predicted_bbox: BBox,
        detection_bbox: BBox,
    ) -> float:

        p = np.asarray(
            PersistentMOT._center(
                predicted_bbox
            ),
            dtype=np.float32,
        )

        d = np.asarray(
            PersistentMOT._center(
                detection_bbox
            ),
            dtype=np.float32,
        )

        distance = float(
            np.linalg.norm(p - d)
        )

        width = max(
            1.0,
            predicted_bbox[2]
            - predicted_bbox[0],
        )

        height = max(
            1.0,
            predicted_bbox[3]
            - predicted_bbox[1],
        )

        scale = max(
            30.0,
            (width**2 + height**2) ** 0.5,
        )

        return float(
            np.exp(
                -distance / (scale * 1.5)
            )
        )

    # ---------------------------------------------------------
    # APPEARANCE
    # ---------------------------------------------------------

    @staticmethod
    def _extract_appearance(
        frame: np.ndarray,
        bbox: BBox,
    ) -> np.ndarray:

        height, width = frame.shape[:2]

        x1, y1, x2, y2 = bbox

        x1 = max(
            0,
            min(width - 1, int(x1)),
        )

        y1 = max(
            0,
            min(height - 1, int(y1)),
        )

        x2 = max(
            x1 + 1,
            min(width, int(x2)),
        )

        y2 = max(
            y1 + 1,
            min(height, int(y2)),
        )

        crop = frame[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            return np.zeros(
                64,
                dtype=np.float32,
            )

        # Use the central region to reduce background influence.
        crop_h, crop_w = crop.shape[:2]

        margin_x = int(
            crop_w * 0.15
        )

        margin_y = int(
            crop_h * 0.10
        )

        if (
            crop_w > 2 * margin_x
            and crop_h > 2 * margin_y
        ):
            crop = crop[
                margin_y:crop_h - margin_y,
                margin_x:crop_w - margin_x,
            ]

        hsv = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2HSV,
        )

        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [16, 4],
            [0, 180, 0, 256],
        )

        hist = cv2.normalize(
            hist,
            hist,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_L2,
        )

        feature = hist.flatten().astype(
            np.float32
        )

        norm = np.linalg.norm(feature)

        if norm > 1e-8:
            feature /= norm

        return feature

    @staticmethod
    def _cosine_similarity(
        a: np.ndarray | None,
        b: np.ndarray,
    ) -> float:

        if a is None:
            return 0.0

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denominator <= 1e-8:
            return 0.0

        similarity = float(
            np.dot(a, b)
            / denominator
        )

        return max(
            0.0,
            min(1.0, similarity),
        )

    # ---------------------------------------------------------
    # GEOMETRY
    # ---------------------------------------------------------

    @staticmethod
    def _center(
        bbox: BBox,
    ) -> Tuple[float, float]:

        x1, y1, x2, y2 = bbox

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    @staticmethod
    def _iou(
        a: BBox,
        b: BBox,
    ) -> float:

        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(
            0.0,
            ix2 - ix1,
        )

        ih = max(
            0.0,
            iy2 - iy1,
        )

        intersection = iw * ih

        area_a = max(
            0.0,
            ax2 - ax1,
        ) * max(
            0.0,
            ay2 - ay1,
        )

        area_b = max(
            0.0,
            bx2 - bx1,
        ) * max(
            0.0,
            by2 - by1,
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union