# -*- coding: utf-8 -*-
"""Trajectory manager.

Keeps per‑track history of centre points (x, y) together with the frame index.
The length of the history is bounded by ``MAX_HISTORY`` from the configuration.
"""

from collections import deque
from typing import Dict, List, Tuple

from config.config import get

# Type alias for a single point entry: (frame_idx, x, y)
Point = Tuple[int, float, float]

class TrajectoryManager:
    """Manage trajectories for multiple track IDs.

    Internally maps ``track_id`` to a ``deque`` of ``Point`` tuples. ``deque``
    automatically discards the oldest entries when the maximum length is
    exceeded.
    """

    def __init__(self) -> None:
        self.max_len: int = get("MAX_HISTORY", 30)
        self._trajectories: Dict[int, deque[Point]] = {}

    def update(self, track_id: int, frame_idx: int, bbox: List[float]) -> None:
        """Add a new centre point for ``track_id``.

        Parameters
        ----------
        track_id: int
            The persistent ID assigned by ByteTrack.
        frame_idx: int
            Current frame number.
        bbox: List[float]
            Bounding box as ``[x1, y1, x2, y2]``.
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        if track_id not in self._trajectories:
            self._trajectories[track_id] = deque(maxlen=self.max_len)
        self._trajectories[track_id].append((frame_idx, cx, cy))

    def get_history(self, track_id: int) -> List[Point]:
        """Return the ordered list of points for ``track_id`` (oldest → newest)."""
        return list(self._trajectories.get(track_id, []))

    def prune_missing(self, active_ids: List[int]) -> None:
        """Remove trajectories for tracks that are no longer present.

        ``active_ids`` is the list of track IDs returned by the latest ByteTrack
        update. Tracks that disappear for more than ``max_len`` frames are removed
        automatically by the ``deque`` overflow, but we also delete entries that
        are completely missing to free memory.
        """
        to_delete = [tid for tid in self._trajectories if tid not in active_ids]
        for tid in to_delete:
            del self._trajectories[tid]
