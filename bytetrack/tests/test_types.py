import pytest

from src.types import Detection, TrackedDetection


def test_detection_stores_expected_values() -> None:
    detection = Detection(
        bbox=(10.0, 20.0, 100.0, 200.0),
        confidence=0.95,
        class_id=0,
    )

    assert detection.bbox == (10.0, 20.0, 100.0, 200.0)
    assert detection.confidence == 0.95
    assert detection.class_id == 0


def test_tracked_detection_stores_track_id() -> None:
    detection = TrackedDetection(
        bbox=(10.0, 20.0, 100.0, 200.0),
        confidence=0.95,
        class_id=0,
        track_id=7,
    )

    assert detection.bbox == (10.0, 20.0, 100.0, 200.0)
    assert detection.confidence == 0.95
    assert detection.class_id == 0
    assert detection.track_id == 7


def test_detection_is_immutable() -> None:
    detection = Detection(
        bbox=(10.0, 20.0, 100.0, 200.0),
        confidence=0.95,
        class_id=0,
    )

    with pytest.raises(AttributeError):
        setattr(detection, "confidence", 0.5)