# -*- coding: utf-8 -*-
"""Pipeline: per-frame processing.

The :class:`PipelineProcessor` glues together all building blocks:

1. ``DeerDetector``       – YOLOv11 detection on the raw frame.
2. ``ByteTrackAdapter``   – assigns persistent track IDs.
3. ``TrajectoryManager``  – stores centre-point history for each track.
4. ``PredictionManager``  – Kalman filter per track + future prediction.
5. ``CollisionEngine``    – evaluates risk based on predicted position.
6. ``VLMManager``         – optional BLIP caption generation (HIGH_RISK, cooldown).

The public method :meth:`process_frame` receives a BGR ``numpy.ndarray`` frame
and its monotonically increasing ``frame_id``.  It returns a dict:

    {
        "frame_id": int,
        "tracks": [
            {
                "track_id": int,
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int,
                "class_name": str,
                "risk": str,          # SAFE | CAUTION | HIGH_RISK | UNKNOWN
                "speed": float,       # px/frame
                "position": [cx, cy], # current centre
                "predicted": [px, py],# predicted centre after horizon
                "trajectory": [[frame_idx, x, y], ...],  # history
                "caption": str | None # BLIP caption if triggered
            },
            ...
        ],
        "annotated_frame": numpy.ndarray  # BGR frame with annotations drawn
    }

No I/O (reading files, writing video, etc.) is performed here.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

from detection.detector import DeerDetector
from bytetrack_adapter.adapter import ByteTrackAdapter
from trajectory.manager import TrajectoryManager
from prediction.manager import PredictionManager
from collision.engine import CollisionEngine, RiskLevel
from vlm.manager import VLMManager
from config.config import get, parse_road_zone

logger = logging.getLogger(__name__)

# ── Annotation colours (BGR) ──────────────────────────────────────────────────
_RISK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "SAFE":      (0, 220, 80),
    "CAUTION":   (0, 200, 255),
    "HIGH_RISK": (0, 60, 230),
    "UNKNOWN":   (160, 160, 160),
}
_TRAJ_COLOR  = (60, 200, 60)
_PRED_COLOR  = (230, 140, 0)
_ZONE_COLOR  = (0, 200, 255)


class PipelineProcessor:
    """Orchestrates the full AI pipeline for a single video frame."""

    def __init__(self) -> None:
        self.device: str = get("YOLO_DEVICE", "cuda")
        self.model_path: str = get("MODEL_PATH", "model/best.pt")
        self.confidence: float = get("YOLO_CONFIDENCE", 0.3)
        self.iou: float = get("YOLO_IOU", 0.5)
        self.img_size: int = get("YOLO_IMG_SIZE", 640)

        logger.info(f"PipelineProcessor init | model={self.model_path} device={self.device}")

        self.detector   = DeerDetector(
            model_path=self.model_path,
            confidence=self.confidence,
            iou=self.iou,
            img_size=self.img_size,
            device=self.device,
        )
        self.tracker    = ByteTrackAdapter()
        self.trajectory = TrajectoryManager()
        self.prediction = PredictionManager()
        self.collision  = CollisionEngine()
        self.vlm        = VLMManager()

    def reset_source_state(self) -> None:
        """Reset per-source tracking state without reloading detection models."""
        self.tracker = ByteTrackAdapter()
        self.trajectory = TrajectoryManager()
        self.prediction = PredictionManager()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _crop_bbox(frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w,     x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h,     y2))
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _annotate(
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        road_zone: List[List[float]],
    ) -> np.ndarray:
        """Draw bounding boxes, IDs, risk, trajectories and road polygon."""
        out = frame.copy()

        # Draw road-zone polygon
        if len(road_zone) >= 3:
            pts = np.array(road_zone, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=True, color=_ZONE_COLOR, thickness=2)
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], color=(*_ZONE_COLOR[:2], 40))
            cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)
            # Label
            cx_zone = int(np.mean([p[0] for p in road_zone]))
            cy_zone = int(np.mean([p[1] for p in road_zone]))
            cv2.putText(out, "ROAD ZONE", (cx_zone - 40, cy_zone),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _ZONE_COLOR, 1)

        for trk in tracks:
            bbox       = list(map(int, trk["bbox"]))
            track_id   = trk["track_id"]
            risk       = trk.get("risk", "UNKNOWN")
            color      = _RISK_COLORS.get(risk, (160, 160, 160))
            traj_pts   = trk.get("trajectory", [])   # [[frame_idx, x, y], ...]
            predicted  = trk.get("predicted")         # [px, py] or None

            # Bounding box
            cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Label
            label = f"#{track_id}  {risk}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (bbox[0], bbox[1] - th - 6), (bbox[0] + tw + 4, bbox[1]), color, -1)
            cv2.putText(out, label, (bbox[0] + 2, bbox[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Trajectory (past points)
            if len(traj_pts) >= 2:
                pts_xy = [(int(p[1]), int(p[2])) for p in traj_pts]
                for i in range(1, len(pts_xy)):
                    cv2.line(out, pts_xy[i - 1], pts_xy[i], _TRAJ_COLOR, 2)
                for pt in pts_xy:
                    cv2.circle(out, pt, 3, _TRAJ_COLOR, -1)

            # Current centre
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            cv2.circle(out, (cx, cy), 5, color, -1)

            # Predicted position
            if predicted is not None:
                px, py = int(predicted[0]), int(predicted[1])
                if len(traj_pts) > 0:
                    last_pt = (int(traj_pts[-1][1]), int(traj_pts[-1][2]))
                    cv2.line(out, last_pt, (px, py), _PRED_COLOR, 1)
                cv2.circle(out, (px, py), 6, _PRED_COLOR, 2)
                cv2.putText(out, "PRED", (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, _PRED_COLOR, 1)

            # Caption (BLIP)
            caption = trk.get("caption")
            if caption:
                short = caption[:50]
                cv2.putText(out, short, (bbox[0], bbox[3] + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 50), 1)

        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, frame_id: int) -> Dict[str, Any]:
        """Run the full pipeline on *frame*.

        Returns a dict with keys:
            ``frame_id``, ``detection_count``, ``tracks``, ``annotated_frame``
        """
        # ── 1. Detection ──────────────────────────────────────────────────────
        detections = self.detector.detect(frame)

        # ── 2. Tracking ───────────────────────────────────────────────────────
        tracked = self.tracker.update(detections, frame_id)
        active_ids = [t["track_id"] for t in tracked]

        # ── 3. Trajectory update ─────────────────────────────────────────────
        for t in tracked:
            self.trajectory.update(t["track_id"], frame_id, t["bbox"])

        # ── 4. Kalman update ─────────────────────────────────────────────────
        for t in tracked:
            self.prediction.update(t["track_id"], t["bbox"])

        # Prune stale state
        self.trajectory.prune_missing(active_ids)
        self.prediction.prune_missing(active_ids)

        # ── 5. Per-track risk + VLM ──────────────────────────────────────────
        road_zone = parse_road_zone()
        output_tracks: List[Dict[str, Any]] = []

        for t in tracked:
            track_id = t["track_id"]

            # Kalman prediction
            try:
                future_pos = self.prediction.predict(track_id)
                speed      = self.prediction.speed(track_id)
                cur_pos    = self.prediction.current_position(track_id)
            except Exception:
                future_pos = (0.0, 0.0)
                speed      = 0.0
                cur_pos    = None

            # Current centre from bbox
            bbox = t["bbox"]
            cx   = (bbox[0] + bbox[2]) / 2.0
            cy   = (bbox[1] + bbox[3]) / 2.0

            risk = self.collision.evaluate(future_pos, speed, current_pos=(cx, cy))

            traj_history = self.trajectory.get_history(track_id)

            track_entry: Dict[str, Any] = {
                "track_id":   track_id,
                "bbox":       bbox,
                "confidence": t["confidence"],
                "class_id":   t["class_id"],
                "class_name": t.get("class_name", "deer"),
                "risk":       risk.value,
                "speed":      round(speed, 2),
                "position":   [round(cx, 1), round(cy, 1)],
                "predicted":  [round(future_pos[0], 1), round(future_pos[1], 1)],
                "trajectory": [[p[0], round(p[1], 1), round(p[2], 1)] for p in traj_history],
            }

            # ── 6. BLIP caption – only for HIGH_RISK ─────────────────────────
            if risk == RiskLevel.HIGH_RISK and self.vlm.enabled:
                crop    = self._crop_bbox(frame, bbox)
                caption = self.vlm.caption(crop, track_id)
                if caption:
                    track_entry["caption"] = caption

            output_tracks.append(track_entry)

        # ── 7. Annotate frame ─────────────────────────────────────────────────
        annotated = self._annotate(frame, output_tracks, road_zone)

        return {
            "frame_id":       frame_id,
            "detection_count": len(detections),
            "tracks":         output_tracks,
            "annotated_frame": annotated,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Quick manual test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    src = get("CAMERA_SOURCE", "videos/deer_test.mp4")
    cap = cv2.VideoCapture(src)
    proc = PipelineProcessor()
    fid = 0
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        result = proc.process_frame(frm, fid)
        print(f"Frame {fid}: {len(result['tracks'])} tracks")
        fid += 1
    cap.release()
    print("Done.")
