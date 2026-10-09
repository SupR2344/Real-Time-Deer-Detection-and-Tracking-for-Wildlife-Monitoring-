# -*- coding: utf-8 -*-
"""USB and Integrated webcam source implementation."""

import logging
from typing import Optional, Tuple
import cv2
import numpy as np

from camera.base import CameraSource, CameraStatus

logger = logging.getLogger(__name__)


class WebcamSource(CameraSource):
    """Source reading from a USB or integrated webcam by device index."""

    def __init__(self, device_index: int | str = 0):
        super().__init__(str(device_index))
        try:
            self.device_index = int(device_index)
        except ValueError:
            self.device_index = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._first_frame: Optional[np.ndarray] = None

    def connect(self) -> bool:
        self.status = CameraStatus.CONNECTING
        logger.info(f"Connecting to webcam at device index {self.device_index}")
        # Try DSHOW backend first on Windows, fallback to CAP_ANY
        self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_ANY)

        if self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                self.status = CameraStatus.CONNECTED
                self._first_frame = frame
                logger.info(f"Webcam index {self.device_index} connected ({frame.shape[1]}x{frame.shape[0]})")
                return True
            else:
                self._cap.release()
                self._cap = None
                self.status = CameraStatus.ERROR
                logger.warning(f"Webcam index {self.device_index} opened but failed to capture frame")
                return False
        else:
            self.status = CameraStatus.ERROR
            logger.warning(f"Failed to open webcam index {self.device_index}")
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
