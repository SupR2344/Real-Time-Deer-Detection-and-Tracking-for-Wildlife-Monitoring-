# -*- coding: utf-8 -*-
"""IP Camera / RTSP stream source implementation."""

import logging
import os
import re
import socket
from urllib.parse import urlparse
from typing import Optional, Tuple
import cv2
import numpy as np

from camera.base import CameraSource, CameraStatus

logger = logging.getLogger(__name__)


def mask_rtsp_url(url: str) -> str:
    """Mask credentials in RTSP URL for safe logging/display.
    Example: rtsp://user:pass@192.168.1.1:554/stream -> rtsp://***:***@192.168.1.1:554/stream
    """
    if not url:
        return ""
    return re.sub(r"://([^:@]+):([^@]+)@", r"://***:***@", str(url))


def is_network_stream_reachable(url: str, timeout: float = 1.5) -> bool:
    """Fast check whether the remote host and port can accept a TCP connection."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or (554 if parsed.scheme == "rtsp" else 80)
        if not host:
            return True
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        err = sock.connect_ex((host, port))
        sock.close()
        return err == 0
    except Exception as exc:
        logger.debug(f"Reachability probe failed for {url}: {exc}")
        return False


class RTSPSource(CameraSource):
    """Source reading from an IP camera via RTSP/HTTP/MJPEG/HTTPS."""

    def __init__(self, rtsp_url: str, timeout: int = 5):
        super().__init__(rtsp_url)
        self.timeout = timeout
        self._cap: Optional[cv2.VideoCapture] = None
        self._first_frame: Optional[np.ndarray] = None

    def connect(self) -> bool:
        self.status = CameraStatus.CONNECTING
        safe_url = mask_rtsp_url(self.source_path)
        logger.info(f"Connecting to RTSP source: {safe_url}")

        # Pre-check host reachability to avoid long Windows TCP blocking
        if not is_network_stream_reachable(self.source_path, timeout=1.5):
            self.status = CameraStatus.ERROR
            logger.warning(f"RTSP stream host unreachable: {safe_url}")
            return False

        # Set OpenCV TCP transport flag for RTSP stability
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;tcp|stimeout;{int(self.timeout * 1000000)}"

        try:
            self._cap = cv2.VideoCapture(self.source_path, cv2.CAP_FFMPEG)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.source_path, cv2.CAP_ANY)
        except Exception as exc:
            logger.warning(f"VideoCapture exception on {safe_url}: {exc}")
            self._cap = cv2.VideoCapture(self.source_path)

        if self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                self.status = CameraStatus.CONNECTED
                self._first_frame = frame
                logger.info(f"Connected to RTSP stream: {safe_url} ({frame.shape[1]}x{frame.shape[0]})")
                return True
            else:
                self._cap.release()
                self._cap = None
                self.status = CameraStatus.ERROR
                logger.warning(f"RTSP stream opened but no frame could be decoded: {safe_url}")
                return False
        else:
            self.status = CameraStatus.ERROR
            logger.warning(f"Failed to open RTSP stream: {safe_url}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._first_frame is not None:
            frame = self._first_frame
            self._first_frame = None
            self.status = CameraStatus.STREAMING
            return True, frame

        if self._cap is None or not self._cap.isOpened():
            self.status = CameraStatus.DISCONNECTED
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            self.status = CameraStatus.ERROR
            return False, None

        self.status = CameraStatus.STREAMING
        return True, frame

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        self._first_frame = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.status = CameraStatus.DISCONNECTED
