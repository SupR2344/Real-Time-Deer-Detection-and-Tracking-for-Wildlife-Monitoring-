# -*- coding: utf-8 -*-
"""Kalman filter implementation for 2‑D constant‑velocity motion.

Each track has its own ``KalmanFilter`` instance. The state vector is
``[x, y, vx, vy]`` and the measurement vector is ``[x, y]`` (the centre of the
bbox). The filter is initialised on the first observation.
"""

import numpy as np

class KalmanFilter:
    def __init__(self, dt: float = 1.0, process_variance: float = 1e-2, measurement_variance: float = 1e-1):
        # State transition matrix (constant velocity)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        # Observation matrix – we only observe position
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        # Process and measurement covariance matrices
        self.Q = process_variance * np.eye(4, dtype=np.float32)
        self.R = measurement_variance * np.eye(2, dtype=np.float32)
        # Initial state and covariance (set on first update)
        self.x = None  # state vector
        self.P = None  # covariance matrix

    def _initialize(self, cx: float, cy: float):
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1.0

    def update(self, cx: float, cy: float) -> None:
        """Incorporate a new observation (centre of bbox)."""
        if self.x is None:
            self._initialize(cx, cy)
            return
        # Predict step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Measurement update
        z = np.array([cx, cy], dtype=np.float32)
        y = z - (self.H @ self.x)  # residual
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def current_position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    def predict(self, steps: int = 1) -> tuple[float, float]:
        """Predict future position after ``steps`` frames using the current state."""
        if self.x is None:
            raise RuntimeError("Kalman filter not initialised yet")
        x_pred = self.x.copy()
        for _ in range(steps):
            x_pred = self.F @ x_pred
        return float(x_pred[0]), float(x_pred[1])
