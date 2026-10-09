from dataclasses import dataclass
from typing import Tuple


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float
    class_id: int


@dataclass(frozen=True)
class TrackedDetection:
    bbox: BBox
    confidence: float
    class_id: int
    track_id: int