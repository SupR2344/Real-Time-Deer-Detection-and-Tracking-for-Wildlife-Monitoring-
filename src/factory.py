from __future__ import annotations

from .base import BaseTracker
from .bytetrack import ByteTrackTracker
from .config import ByteTrackConfig


def create_tracker(
    tracker_type: str,
    config: ByteTrackConfig,
) -> BaseTracker:
    """
    Create a tracker implementation from its configured type.

    Parameters
    ----------
    tracker_type:
        Name of the tracker implementation to create.

    config:
        Configuration passed to the selected tracker.

    Returns
    -------
    BaseTracker:
        Initialized tracker implementation.

    Raises
    ------
    ValueError:
        If the requested tracker type is not supported.
    """

    normalized_type = tracker_type.strip().lower()

    if normalized_type == "bytetrack":
        return ByteTrackTracker(config)

    raise ValueError(
        f"Unsupported tracker type: '{tracker_type}'. "
        "Supported trackers: bytetrack."
    )