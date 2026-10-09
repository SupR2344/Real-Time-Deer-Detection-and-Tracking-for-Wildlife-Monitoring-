# -*- coding: utf-8 -*-
"""Runtime state shared between the background processing loop and FastAPI endpoints.

Thread-safe via ``threading.Lock``.
"""

import threading
import time
from typing import List, Dict, Any, Optional


class RuntimeState:
    def __init__(self) -> None:
        self._lock           = threading.Lock()
        self._jpeg_bytes:    Optional[bytes] = None
        self._tracks:        List[Dict[str, Any]] = []
        self._detection_count: int = 0
        self._frame_id:      int = -1
        self._camera_status: str = "DISCONNECTED"
        self._source_type: Optional[str] = None
        self._source: Optional[str] = None
        self._frame_width:   int = 0
        self._frame_height:  int = 0
        self._fps:           float = 0.0
        self._events:        List[Dict[str, Any]] = []  # {time, message, level}

    # ── Frame ──────────────────────────────────────────────────────────────────

    def update(
        self,
        jpeg_bytes: Optional[bytes],
        tracks: List[Dict[str, Any]],
        frame_id: int,
        detection_count: Optional[int] = None,
    ) -> None:
        with self._lock:
            self._jpeg_bytes = jpeg_bytes
            self._tracks     = tracks
            self._frame_id   = frame_id
            self._detection_count = len(tracks) if detection_count is None else detection_count

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg_bytes

    def get_tracks(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._tracks)

    def get_detection_count(self) -> int:
        with self._lock:
            return self._detection_count

    def get_frame_id(self) -> int:
        with self._lock:
            return self._frame_id

    # ── Camera ────────────────────────────────────────────────────────────────

    def get_camera_status(self) -> str:
        with self._lock:
            return self._camera_status

    def set_camera_status(self, status: str) -> None:
        with self._lock:
            self._camera_status = status
            if status in ("DISCONNECTED", "ERROR"):
                self._jpeg_bytes = None
                self._tracks = []
                self._detection_count = 0
                self._frame_id = -1
                self._frame_width = 0
                self._frame_height = 0
                self._fps = 0.0

    def set_source(self, source_type: Optional[str], source: Optional[str]) -> None:
        with self._lock:
            self._source_type = source_type
            self._source = source
            self._camera_status = "CONNECTING" if source_type else "DISCONNECTED"
            self._jpeg_bytes = None
            self._tracks = []
            self._detection_count = 0
            self._frame_id = -1
            self._frame_width = 0
            self._frame_height = 0
            self._fps = 0.0

    def get_source(self) -> Dict[str, Optional[str]]:
        with self._lock:
            return {"type": self._source_type, "source": self._source}

    # ── Dimensions ───────────────────────────────────────────────────────────

    def get_frame_dimensions(self):
        with self._lock:
            return (self._frame_width, self._frame_height)

    def set_frame_dimensions(self, width: int, height: int) -> None:
        with self._lock:
            self._frame_width  = width
            self._frame_height = height

    # ── FPS ──────────────────────────────────────────────────────────────────

    def get_fps(self) -> float:
        with self._lock:
            return self._fps

    def set_fps(self, fps: float) -> None:
        with self._lock:
            self._fps = fps

    # ── Events ───────────────────────────────────────────────────────────────

    def add_event(self, message: str, level: str = "info") -> None:
        with self._lock:
            self._events.insert(0, {
                "time":    time.strftime("%H:%M:%S"),
                "message": message,
                "level":   level,
            })
            # Keep at most 50 events
            self._events = self._events[:50]

    def get_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        with self._lock:
            self._jpeg_bytes    = None
            self._tracks        = []
            self._detection_count = 0
            self._frame_id      = -1
            self._camera_status = "DISCONNECTED"
            self._source_type   = None
            self._source        = None
            self._frame_width   = 0
            self._frame_height  = 0
            self._fps           = 0.0
            self._events        = []
