# -*- coding: utf-8 -*-
"""Geometric computations for road-zone polygon and trajectory/predictions."""

from typing import List, Tuple, Optional
from shapely.geometry import Polygon, Point, LineString


def create_polygon(points: List[Tuple[float, float]] | List[List[float]]) -> Optional[Polygon]:
    """Create a Shapely Polygon from a list of (x, y) coordinates.
    Requires at least 3 points. Returns None if invalid or degenerate.
    """
    if not points or len(points) < 3:
        return None
    try:
        poly = Polygon(points)
        if poly.is_valid and poly.area > 0:
            return poly
        # Try to fix invalid polygon (e.g. self-intersection)
        fixed = poly.buffer(0)
        return fixed if (fixed.is_valid and fixed.area > 0) else None
    except Exception:
        return None


def point_distance_to_polygon(point_xy: Tuple[float, float], polygon: Polygon) -> float:
    """Calculate Euclidean distance from a 2D point to the polygon boundary.
    If point is inside the polygon, returns 0.0.
    """
    pt = Point(point_xy)
    if polygon.contains(pt):
        return 0.0
    return float(pt.distance(polygon))


def trajectory_intersects_polygon(
    p_start: Tuple[float, float], p_end: Tuple[float, float], polygon: Polygon
) -> bool:
    """Check if the line segment from p_start to p_end intersects the polygon."""
    line = LineString([p_start, p_end])
    return bool(line.intersects(polygon))
