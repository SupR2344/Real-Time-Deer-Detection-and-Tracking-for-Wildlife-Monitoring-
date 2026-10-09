# -*- coding: utf-8 -*-
"""Prediction manager that holds a Kalman filter for each track.

The ``PredictionManager`` creates a ``KalmanFilter`` instance the first time a
track ID is seen. On each update it feeds the centre point of the current bbox
into the filter and can predict the future position after ``horizon`` frames.
"""

from typing import Dict, Tuple, List, Optional

from config.config import get
from prediction.kalman_filter import KalmanFilter


class PredictionManager:
    """Manage Kalman filters for multiple tracks.

    ``horizon`` is the number of future frames for which we predict the position.
    """

    def __init__(self) -> None:
        self.horizon: int = get("PREDICTION_HORIZON", 5)
        self._filters: Dict[int, KalmanFilter] = {}

    def _ensure_filter(self, track_id: int) -> KalmanFilter:
        if track_id not in self._filters:
            self._filters[track_id] = KalmanFilter()
        return self._filters[track_id]

    def update(self, track_id: int, bbox: List[float]) -> None:
        """Update the Kalman filter for ``track_id`` with the current ``bbox``.

        ``bbox`` is ``[x1, y1, x2, y2]`` – the centre point is used for the
        measurement.
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        kf = self._ensure_filter(track_id)
        kf.update(cx, cy)

    def predict(self, track_id: int) -> Tuple[float, float]:
        """Predict future centre position after ``horizon`` frames.

        Returns ``(x, y)``. Raises ``KeyError`` if the track has never been seen.
        """
        kf = self._filters[track_id]
        return kf.predict(self.horizon)

    def current_position(self, track_id: int) -> Optional[Tuple[float, float]]:
        """Return the current Kalman-estimated position for ``track_id``.

        Returns ``None`` if the filter is not yet initialised.
        """
        kf = self._filters.get(track_id)
        if kf is None or kf.x is None:
            return None
        return kf.current_position

    def speed(self, track_id: int) -> float:
        """Return scalar speed (pixels per frame) derived from the filter's velocity."""
        kf = self._filters.get(track_id)
        if kf is None or kf.x is None:
            return 0.0
        vx, vy = kf.velocity
        return (vx ** 2 + vy ** 2) ** 0.5

    def prune_missing(self, active_ids: List[int]) -> None:
        """Remove filters for tracks that are no longer active."""
        to_delete = [tid for tid in self._filters if tid not in active_ids]
        for tid in to_delete:
            del self._filters[tid]
