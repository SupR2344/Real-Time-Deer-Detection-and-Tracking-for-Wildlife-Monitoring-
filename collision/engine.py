# -*- coding: utf-8 -*-
"""Collision engine for road-zone risk assessment.

Determines the risk level for a tracked deer based on its current position,
predicted future position, speed, and spatial relation to the configured road zone.

Risk levels:
- ``UNKNOWN``   – road polygon missing or invalid (NEVER treated as SAFE).
- ``HIGH_RISK`` – deer inside road zone, or predicted inside road zone, or trajectory
                  directly intersecting road zone with significant velocity, or within
                  high-risk proximity boundary.
- ``CAUTION``   – deer within caution distance boundary of road, or moving toward road.
- ``SAFE``      – deer far from road zone and trajectory not heading into road.
"""

from enum import Enum
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, Point

from config.config import get, parse_road_zone
from collision.geometry import (
    create_polygon,
    point_distance_to_polygon,
    trajectory_intersects_polygon,
)


class RiskLevel(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    HIGH_RISK = "HIGH_RISK"
    UNKNOWN = "UNKNOWN"


class CollisionEngine:
    """Deterministic collision risk evaluator against road-zone geometry."""

    def __init__(self, road_points: Optional[List[List[float]]] = None) -> None:
        pts = road_points if road_points is not None else parse_road_zone()
        self.polygon: Optional[Polygon] = create_polygon(pts)
        self.caution_distance: float = float(get("CAUTION_DISTANCE", 50.0))
        self.high_risk_distance: float = float(get("ROAD_DISTANCE_THRESHOLD", 20.0))
        self.caution_speed: float = float(get("CAUTION_SPEED", 2.0))
        self.high_risk_speed: float = float(get("HIGH_RISK_SPEED", 5.0))

    def evaluate(
        self,
        future_pos: Tuple[float, float],
        speed: float = 0.0,
        current_pos: Optional[Tuple[float, float]] = None,
    ) -> RiskLevel:
        """Evaluate risk for a deer track.

        Parameters
        ----------
        future_pos : (x, y) predicted center after horizon.
        speed : scalar speed (px/frame).
        current_pos : (x, y) current center position.

        Returns
        -------
        RiskLevel (UNKNOWN | HIGH_RISK | CAUTION | SAFE)
        """
        # Rule 1: No road zone polygon -> UNKNOWN. NEVER treat UNKNOWN as SAFE.
        if self.polygon is None:
            return RiskLevel.UNKNOWN

        # Check positions against polygon
        future_dist = point_distance_to_polygon(future_pos, self.polygon)
        cur_dist = point_distance_to_polygon(current_pos, self.polygon) if current_pos else future_dist

        # 1. Deer is already inside road polygon or predicted inside
        if cur_dist == 0.0 or future_dist == 0.0:
            return RiskLevel.HIGH_RISK

        # 2. Deer trajectory line intersects the road polygon
        if current_pos is not None:
            if trajectory_intersects_polygon(current_pos, future_pos, self.polygon):
                return RiskLevel.HIGH_RISK

        # 3. High risk proximity threshold
        if future_dist <= self.high_risk_distance or cur_dist <= self.high_risk_distance:
            if speed >= self.caution_speed:
                return RiskLevel.HIGH_RISK
            return RiskLevel.CAUTION

        # 4. Caution proximity threshold
        if future_dist <= self.caution_distance or cur_dist <= self.caution_distance:
            return RiskLevel.CAUTION

        # 5. Far from road and not heading into it -> SAFE
        return RiskLevel.SAFE

    def batch_evaluate(
        self, predictions: List[Tuple[Tuple[float, float], float]]
    ) -> List[RiskLevel]:
        """Evaluate a list of (future_pos, speed) tuples."""
        return [self.evaluate(pos, spd) for pos, spd in predictions]
