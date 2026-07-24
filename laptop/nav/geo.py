"""Geodesy helpers for GPS waypoint navigation.

All functions operate on (lat, lon) tuples in WGS84 decimal degrees and
return either metres (distance) or degrees in [0, 360) measured clockwise
from true north (bearing).
"""

from __future__ import annotations
import math

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lon) points, in metres."""
    lat1, lon1 = p1
    lat2, lon2 = p2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_deg(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Initial bearing FROM p1 TO p2, in degrees [0, 360) (0=N, 90=E)."""
    lat1, lon1 = p1
    lat2, lon2 = p2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(phi2)
    x = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlam))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def heading_error_deg(current_heading: float, target_bearing: float) -> float:
    """Signed shortest-arc error from current heading to target bearing.

    Result is in (-180, +180]:
        positive -> turn right (clockwise)
        negative -> turn left  (counter-clockwise)
    """
    err = (target_bearing - current_heading + 540.0) % 360.0 - 180.0
    return err


def pure_pursuit_steer_deg(alpha_deg: float,
                           lookahead_m: float,
                           wheelbase_m: float) -> float:
    """Pure-Pursuit steering law for an Ackermann vehicle.

        delta = atan( 2 * L * sin(alpha) / Ld )

    alpha       signed angle from the vehicle's heading to the lookahead point
                (this is exactly `heading_error_deg`); + = target is to the RIGHT.
    lookahead_m the distance to that point (Ld).
    wheelbase_m front axle to rear axle (L).

    Returns the front-wheel steering angle in degrees, + = steer RIGHT.

    This is the piece the old code was missing: it turns a *continuous* heading
    error into a *continuous* steering angle. The previous controller quantised
    the error to FORWARD/TURN_LEFT/TURN_RIGHT and then to an angle of exactly
    -1.0 / 0.0 / +1.0 — full lock or dead centre, nothing between — which cannot
    track a line.
    """
    # Guard: a tiny Ld makes the law explode (and is meaningless anyway).
    ld = max(float(lookahead_m), 0.5)
    a = math.radians(alpha_deg)
    return math.degrees(math.atan2(2.0 * wheelbase_m * math.sin(a), ld))


# --- Local ENU helpers (good enough at small scales — <1 km) ---
# Treats lat/lon as flat by scaling longitude with cos(lat). For paths under
# ~1 km this gives sub-metre error which is fine for waypoint navigation.

_METRES_PER_DEG_LAT = 111_320.0   # nominal


def _ll_to_xy(origin: tuple[float, float], p: tuple[float, float]) -> tuple[float, float]:
    """Convert (lat, lon) to local east/north metres from `origin`."""
    olat, olon = origin
    plat, plon = p
    east = (plon - olon) * _METRES_PER_DEG_LAT * math.cos(math.radians(olat))
    north = (plat - olat) * _METRES_PER_DEG_LAT
    return east, north


def _xy_to_ll(origin: tuple[float, float], xy: tuple[float, float]) -> tuple[float, float]:
    """Inverse of _ll_to_xy."""
    olat, olon = origin
    east, north = xy
    dlat = north / _METRES_PER_DEG_LAT
    dlon = east / (_METRES_PER_DEG_LAT * math.cos(math.radians(olat)))
    return olat + dlat, olon + dlon


def lookahead_point(path: list[tuple[float, float]],
                    current: tuple[float, float],
                    lookahead_m: float,
                    min_seg: int = 0,
                    ) -> tuple[tuple[float, float], float, int]:
    """Pure Pursuit lookahead.

    Given a polyline `path` and the vehicle's current GPS position, find:
      1. the closest point on the path
      2. a target ("carrot") that is `lookahead_m` metres ahead of that
         closest point along the polyline

    Returns (carrot_latlon, cross_track_m, segment_index_of_closest_point).
    `cross_track_m` is signed: positive means we're to the RIGHT of the path
    direction, negative means LEFT. (Useful if you later add a Stanley term.)
    """
    if not path:
        return current, 0.0, 0

    origin = path[0]
    cur_x, cur_y = _ll_to_xy(origin, current)
    pts_xy = [_ll_to_xy(origin, p) for p in path]

    # 1. Find closest point along the polyline.
    #    `min_seg` forbids searching segments the vehicle has already passed, so
    #    progress along the route can never go backwards (GPS noise, or a shove
    #    from an obstacle manoeuvre, could otherwise snap the projection back to an
    #    earlier segment and make the car re-drive part of the route).
    best_d2 = float("inf")
    best_seg = min(min_seg, max(len(pts_xy) - 2, 0))
    best_t = 0.0
    best_foot = pts_xy[best_seg]
    for i in range(best_seg, len(pts_xy) - 1):
        ax, ay = pts_xy[i]
        bx, by = pts_xy[i + 1]
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-9:
            continue
        # Project current onto segment.
        t = ((cur_x - ax) * dx + (cur_y - ay) * dy) / seg_len2
        t = max(0.0, min(1.0, t))
        fx, fy = ax + t * dx, ay + t * dy
        d2 = (fx - cur_x) ** 2 + (fy - cur_y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_seg = i
            best_t = t
            best_foot = (fx, fy)

    cross_track = math.sqrt(best_d2)
    # Sign cross-track: positive = right of path direction.
    if best_seg < len(pts_xy) - 1:
        ax, ay = pts_xy[best_seg]
        bx, by = pts_xy[best_seg + 1]
        path_dx, path_dy = bx - ax, by - ay
        side = (cur_x - ax) * path_dy - (cur_y - ay) * path_dx
        if side > 0:
            cross_track = -cross_track   # left of path
    # else: at the end, leave unsigned

    # 2. Walk forward along the polyline by lookahead_m from the foot point.
    remaining = lookahead_m
    fx, fy = best_foot
    for j in range(best_seg, len(pts_xy) - 1):
        ax, ay = pts_xy[j]
        bx, by = pts_xy[j + 1]
        # On the current segment, the foot is at (fx, fy); next end is (bx, by).
        seg_len = math.hypot(bx - fx, by - fy)
        if seg_len >= remaining and seg_len > 0:
            ratio = remaining / seg_len
            tx = fx + (bx - fx) * ratio
            ty = fy + (by - fy) * ratio
            return _xy_to_ll(origin, (tx, ty)), cross_track, best_seg
        remaining -= seg_len
        fx, fy = bx, by

    # Past the end — aim at the final waypoint.
    return _xy_to_ll(origin, pts_xy[-1]), cross_track, len(pts_xy) - 2
