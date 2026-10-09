# -*- coding: utf-8 -*-
"""Unified Camera Manager supporting Video, Webcam, and IP Camera / RTSP sources."""

import logging
import threading
from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np

from camera.base import CameraSource, CameraStatus
from camera.video_source import VideoSource
from camera.webcam_source import WebcamSource
from camera.rtsp_source import RTSPSource, mask_rtsp_url
from config.config import get

logger = logging.getLogger(__name__)


class CameraManager:
    """Manages active camera source lifecycle, reconnection, and frame retrieval."""

    def __init__(self):
        self.current_source: Optional[CameraSource] = None
        self.source_type: str = "none"
        self.raw_source: str = ""
        self.reconnect_delay = int(get("CAMERA_RECONNECT_DELAY", 3))
        self.timeout = int(get("CAMERA_TIMEOUT", 10))
        self._lock = threading.Lock()

    def set_source(self, source_type: str, source_val: str) -> bool:
        """Switch to a new camera source. Safely releases previous source first."""
        with self._lock:
            self._stop_unlocked()
            st = str(source_type).lower().strip()
            self.source_type = st
            self.raw_source = str(source_val).strip()

            if st in ("video", "local_video", "local video"):
                self.current_source = VideoSource(self.raw_source)
            elif st in ("integrated", "integrated_camera", "integrated camera"):
                self.current_source = WebcamSource(self.raw_source if self.raw_source else 0)
            elif st in ("webcam", "usb", "usb_webcam", "usb/webcam"):
                self.current_source = WebcamSource(self.raw_source if self.raw_source else 1)
            elif st in ("rtsp", "ip_camera", "ip camera", "http", "stream"):
                self.current_source = RTSPSource(self.raw_source, timeout=self.timeout)
            else:
                logger.error(f"Unknown camera source type: {source_type}")
                return False

            return self.current_source.connect()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from current source, attempting reconnect if needed."""
        with self._lock:
            if self.current_source is None:
                return False, None

            return self.current_source.read()

    def get_status(self) -> CameraStatus:
        with self._lock:
            if self.current_source is None:
                return CameraStatus.DISCONNECTED
            return self.current_source.status

    def stop(self) -> None:
        """Release current source."""
        with self._lock:
            self._stop_unlocked()

    def _stop_unlocked(self) -> None:
        if self.current_source is not None:
            self.current_source.release()
            self.current_source = None
            self.source_type = "none"
            self.raw_source = ""

    def get_safe_source_info(self) -> dict:
        """Return safe source info without credentials."""
        with self._lock:
            safe_src = mask_rtsp_url(self.raw_source) if "rtsp" in self.raw_source.lower() else self.raw_source
            status_val = self.current_source.status.value if self.current_source else "DISCONNECTED"
            return {
                "type": self.source_type,
                "source": safe_src,
                "status": status_val,
            }

    @staticmethod
    def detect_available_cameras(max_indices: int = 4) -> List[Dict]:
        """Scan for available local OpenCV camera indices (e.g. integrated vs USB)."""
        available = []
        for idx in range(max_indices):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    kind = "Integrated Camera" if idx == 0 else f"USB / External Webcam"
                    available.append({
                        "index": idx,
                        "label": f"{kind} (Index {idx}) — {w}x{h}",
                        "resolution": f"{w}x{h}",
                        "kind": "integrated" if idx == 0 else "webcam"
                    })
        return available

    @staticmethod
    def list_available_videos() -> List[Dict]:
        """Find local test videos in the project."""
        from pathlib import Path
        videos = []
        root = Path(__file__).resolve().parent.parent
        found = set()
        for p in root.glob("*.mp4"):
            if "output" not in p.name.lower() and p.name not in found:
                found.add(p.name)
                videos.append({"name": p.name, "path": p.name})
        vdir = root / "videos"
        if vdir.is_dir():
            for p in vdir.glob("*.mp4"):
                rel = f"videos/{p.name}"
                if rel not in found:
                    found.add(rel)
                    videos.append({"name": rel, "path": rel})
        return videos
