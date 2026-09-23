from __future__ import annotations

from typing import Sequence

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker as _ByteTrackTracker

from .base import BaseTracker
from .config import ByteTrackConfig
from .types import BBox, Detection, TrackedDetection


class ByteTrackTracker(BaseTracker):

    def __init__(self, config: ByteTrackConfig) -> None:
        self.config = config

        self._tracker = _ByteTrackTracker(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
            minimum_iou_threshold=config.minimum_iou_threshold,
            frame_rate=config.frame_rate,
        )

        self._last_frame_id: int | None = None

    def update(
        self,
        detections: Sequence[Detection],
        frame_id: int,
    ) -> list[TrackedDetection]:

        self._validate_frame_id(frame_id)

        supervision_detections = self._to_supervision_detections(
            detections
        )

        tracked = self._tracker.update(
            supervision_detections
        )

        # ---------------------------------------------------------
        # DEBUG
        # ---------------------------------------------------------
        print(
            f"Frame {frame_id} | "
            f"Detections: {len(supervision_detections)} | "
            f"Confidences: {supervision_detections.confidence} | "
            f"Tracker IDs: {tracked.tracker_id}"
        )

        self._last_frame_id = frame_id

        return self._from_supervision_detections(tracked)

    def reset(self) -> None:
        self._tracker.reset()
        self._last_frame_id = None

    def _validate_frame_id(self, frame_id: int) -> None:

        if not isinstance(frame_id, (int, np.integer)):
            raise TypeError(
                "frame_id must be an integer."
            )

        if frame_id < 0:
            raise ValueError(
                "frame_id must be >= 0."
            )

        if (
            self._last_frame_id is not None
            and frame_id < self._last_frame_id
        ):
            raise ValueError(
                "frame_id must be monotonically increasing. "
                f"Received {frame_id} after {self._last_frame_id}."
            )

    @staticmethod
    def _to_supervision_detections(
        detections: Sequence[Detection],
    ) -> sv.Detections:

        if not detections:
            return sv.Detections(
                xyxy=np.empty(
                    (0, 4),
                    dtype=np.float32,
                ),
                confidence=np.empty(
                    (0,),
                    dtype=np.float32,
                ),
                class_id=np.empty(
                    (0,),
                    dtype=np.int32,
                ),
            )

        xyxy = np.asarray(
            [d.bbox for d in detections],
            dtype=np.float32,
        )

        confidence = np.asarray(
            [d.confidence for d in detections],
            dtype=np.float32,
        )

        class_id = np.asarray(
            [d.class_id for d in detections],
            dtype=np.int32,
        )

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

    @staticmethod
    def _from_supervision_detections(
        detections: sv.Detections,
    ) -> list[TrackedDetection]:

        if len(detections) == 0:
            return []

        if detections.tracker_id is None:
            return []

        if detections.confidence is None:
            raise RuntimeError(
                "ByteTrack returned detections without confidence values."
            )

        if detections.class_id is None:
            raise RuntimeError(
                "ByteTrack returned detections without class IDs."
            )

        results: list[TrackedDetection] = []

        for index, track_id in enumerate(
            detections.tracker_id
        ):

            if track_id is None:
                continue

            track_id_int = int(track_id)

            # -1 means the detection has not received
            # a valid persistent track ID.
            if track_id_int < 0:
                continue

            x1 = float(
                detections.xyxy[index][0]
            )
            y1 = float(
                detections.xyxy[index][1]
            )
            x2 = float(
                detections.xyxy[index][2]
            )
            y2 = float(
                detections.xyxy[index][3]
            )

            bbox: BBox = (
                x1,
                y1,
                x2,
                y2,
            )

            results.append(
                TrackedDetection(
                    bbox=bbox,
                    confidence=float(
                        detections.confidence[index]
                    ),
                    class_id=int(
                        detections.class_id[index]
                    ),
                    track_id=track_id_int,
                )
            )

        return results