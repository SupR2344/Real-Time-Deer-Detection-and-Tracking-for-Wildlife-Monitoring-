# -*- coding: utf-8 -*-
"""Local video file source implementation."""

import logging
from typing import Optional, Tuple
from pathlib import Path
import cv2
import numpy as np

from camera.base import CameraSource, CameraStatus

logger = logging.getLogger(__name__)


class VideoSource(CameraSource):
    """Source reading from a local video file, with optional looping."""

    def __init__(self, file_path: str, loop: bool = True):
        super().__init__(str(file_path))
        self.loop = loop
        self._cap: Optional[cv2.VideoCapture] = None

    def connect(self) -> bool:
        self.status = CameraStatus.CONNECTING
        path = Path(self.source_path)
        if not path.is_file():
            logger.warning(f"Video file does not exist: {self.source_path}")
            self.status = CameraStatus.ERROR
            return False

        self._cap = cv2.VideoCapture(str(path))
        if self._cap.isOpened():
            self.status = CameraStatus.CONNECTED
            return True
        else:
            self.status = CameraStatus.ERROR
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if not ret:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if ret:
                    self.status = CameraStatus.STREAMING
                    return True, frame
            self.status = CameraStatus.DISCONNECTED
            return False, None

        self.status = CameraStatus.STREAMING
        return True, frame

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.status = CameraStatus.DISCONNECTED
