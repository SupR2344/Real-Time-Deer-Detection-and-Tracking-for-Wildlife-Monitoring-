# ByteTrack adapter that provides a simple interface used by the pipeline.
# It wraps the existing ByteTrack implementation located in `bytetrack/src`.

from typing import List, Dict, Any

# Import the user‑provided ByteTrack implementation.
# The module path is `bytetrack.src.bytetrack` based on the current layout.
from bytetrack.src.bytetrack import ByteTrackTracker as _ByteTrackTracker
from bytetrack.src.config import ByteTrackConfig

class ByteTrackAdapter:
    """Adapter around the user‑provided ByteTrack implementation.

    The pipeline expects a method ``update(detections, frame_id)`` that returns a
    list of dictionaries with ``track_id``, ``bbox``, ``confidence`` and
    ``class_id``.
    """

    def __init__(self, config: dict | None = None) -> None:
        # Load configuration with sensible defaults if none provided.
        default_cfg = {
            "track_activation_threshold": 0.05,  # Match YOLO_CONFIDENCE=0.05 for 4K sliding-window
            "lost_track_buffer": 60,              # Longer buffer for low-FPS 4K processing
            "minimum_iou_threshold": 0.2,         # Lower IoU for large 4K bboxes
            "frame_rate": 30,
        }
        cfg_dict = config or default_cfg
        cfg = ByteTrackConfig(**cfg_dict)
        self.tracker = _ByteTrackTracker(cfg)
        self._last_frame_id: int | None = None

    def update(self, detections: List[Dict[str, Any]], frame_id: int) -> List[Dict[str, Any]]:
        """Update ByteTrack with detections for the given frame.

        Parameters
        ----------
        detections: List[Dict]
            Each dict must contain ``bbox`` (x1, y1, x2, y2), ``confidence`` and
            ``class_id`` fields, matching the output of :class:`DeerDetector`.
        frame_id: int
            Monotonically increasing frame identifier.
        """
        # Convert to the ``Detection`` objects expected by the upstream library.
        from bytetrack.src.types import Detection

        det_objs = [
            Detection(bbox=tuple(d["bbox"]), confidence=float(d["confidence"]), class_id=int(d["class_id"]))
            for d in detections
        ]
        tracked = self.tracker.update(det_objs, frame_id)
        # Convert back to plain dicts for the pipeline.
        result = []
        for td in tracked:
            result.append({
                "track_id": td.track_id,
                "bbox": list(td.bbox),
                "confidence": td.confidence,
                "class_id": td.class_id,
            })
        return result
