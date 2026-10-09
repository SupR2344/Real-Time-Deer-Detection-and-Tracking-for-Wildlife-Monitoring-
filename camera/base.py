# -*- coding: utf-8 -*-
"""Base class and common types for camera/video sources."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple
import numpy as np


class CameraStatus(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    STREAMING = "STREAMING"
    ERROR = "ERROR"


class CameraSource(ABC):
    """Abstract interface for all video/camera sources."""

    def __init__(self, source_path: str):
        self.source_path = source_path
        self.status = CameraStatus.DISCONNECTED

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the camera/source."""
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame. Returns (success, frame_bgr)."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if source is currently opened."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release any underlying capture resources."""
        pass

    def reconnect(self) -> bool:
        """Reconnect to the source."""
        self.release()
        return self.connect()
