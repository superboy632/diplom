#!/usr/bin/env python3
"""
geo_utils.py — Shared geometric utilities for the Astro-Tracker system.

Coordinate-system contract
──────────────────────────
All tile metadata is stored in **projected metres** (equirectangular, Moon
sphere, R = 1 737 400 m).  Conversion to degrees uses the same flat linear
scale for BOTH axes:

    deg = metres / MOON_RADIUS_M * (180 / π)

There is NO cos(lat) correction anywhere in this pipeline.  The crater_db
therefore stores "flat-projected degrees" that are linearly proportional to
projected metres on both axes.

When computing pixel offsets in localize.py the same scale must be used so
hypothesis arithmetic stays consistent with the database:

    Δlat_deg = Δy_m / R * (180 / π)
    Δlon_deg = Δx_m / R * (180 / π)   ← flat projection, no cos(lat)

Y-axis convention (north-up):
    cy = 0.0  →  top of tile    →  lat_max  (most northerly pixel row)
    cy = 1.0  →  bottom of tile →  lat_min  (most southerly pixel row)
"""

import math

import numpy as np

# ── Shared constants ──────────────────────────────────────────────────────────
MOON_RADIUS_M    = 1_737_400.0   # metres (mean lunar radius)
METERS_PER_PIXEL = 100.0         # dataset ground-sample distance
IMAGE_SIZE       = 416           # tile width and height in pixels


# ── Degree / metre conversions ────────────────────────────────────────────────
def metres_to_deg(m: float) -> float:
    """Projected metres → degrees (flat equirectangular, Moon R)."""
    return m / MOON_RADIUS_M * (180.0 / math.pi)


def deg_to_metres(d: float) -> float:
    """Degrees → projected metres (flat equirectangular, Moon R)."""
    return d * MOON_RADIUS_M * math.pi / 180.0


# ── Haversine distance on the Moon ────────────────────────────────────────────
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two (lat, lon) points in degrees."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2.0 * MOON_RADIUS_M * math.asin(math.sqrt(a))


def haversine_vec(
    lat0: float,
    lon0: float,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Vectorised haversine: great-circle distance from (lat0, lon0) to each row."""
    phi0 = math.radians(lat0)
    phi  = np.radians(lats)
    dphi    = phi - phi0
    dlambda = np.radians(lons - lon0)
    a = (np.sin(dphi / 2) ** 2
         + math.cos(phi0) * np.cos(phi) * np.sin(dlambda / 2) ** 2)
    return 2.0 * MOON_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ── Pixel → degree offset (flat equirectangular) ──────────────────────────────
def pixel_offset_deg(cx_norm: float, cy_norm: float) -> tuple[float, float]:
    """
    Return (Δlat_deg, Δlon_deg) of a pixel relative to the tile centre.

    Parameters
    ----------
    cx_norm, cy_norm : float
        Normalised pixel coordinates ∈ [0, 1].
        cy = 0 → top of tile (most northerly); cy = 1 → bottom.

    Returns
    -------
    (dlat_deg, dlon_deg) : tuple[float, float]
        Signed offset in flat-projected degrees from the tile centre.
        Positive dlat → north; positive dlon → east.

    Notes
    -----
    Both axes use the identical flat scale (no cos-lat correction):
        Δdeg = Δmetres / MOON_RADIUS_M * (180 / π)
    This matches how crater_db degrees were derived from projected metres.
    """
    dx_px = (cx_norm - 0.5) * IMAGE_SIZE
    dy_px = (cy_norm - 0.5) * IMAGE_SIZE

    dx_m =  dx_px * METERS_PER_PIXEL   # east  is positive
    dy_m = -dy_px * METERS_PER_PIXEL   # north is positive (y-axis flip)

    dlat = dy_m / MOON_RADIUS_M * (180.0 / math.pi)
    dlon = dx_m / MOON_RADIUS_M * (180.0 / math.pi)   # flat: same scale as lat
    return dlat, dlon


# ── Pixel → absolute lat/lon ──────────────────────────────────────────────────
def pixel_to_latlon(
    cx_norm: float,
    cy_norm: float,
    lat_min_m: float,
    lat_max_m: float,
    lon_min_m: float,
    lon_max_m: float,
) -> tuple[float, float]:
    """
    Normalised pixel coordinate → (lat_deg, lon_deg).

    Tile bounds are given in projected metres.
    cy = 0 → top → lat_max (north-up); cy = 1 → bottom → lat_min.
    """
    lon_m = lon_min_m + cx_norm * (lon_max_m - lon_min_m)
    lat_m = lat_max_m - cy_norm * (lat_max_m - lat_min_m)   # y-axis flip
    return metres_to_deg(lat_m), metres_to_deg(lon_m)


# ── Non-Maximum Suppression ───────────────────────────────────────────────────
def nms(dets: np.ndarray, iou_thresh: float = 0.5) -> np.ndarray:
    """
    Greedy IoU-based Non-Maximum Suppression.

    Parameters
    ----------
    dets       : ndarray of shape (K, 5)
                 Columns: cx_norm, cy_norm, w_norm, h_norm, confidence.
    iou_thresh : float
                 Boxes with IoU > iou_thresh against a kept box are suppressed.

    Returns
    -------
    Filtered ndarray with the same column layout, rows sorted by confidence
    descending (highest-confidence box retained per group).
    """
    if len(dets) == 0:
        return dets
    x1 = dets[:, 0] - dets[:, 2] / 2
    y1 = dets[:, 1] - dets[:, 3] / 2
    x2 = dets[:, 0] + dets[:, 2] / 2
    y2 = dets[:, 1] + dets[:, 3] / 2
    order = dets[:, 4].argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter  = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_j = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
        iou    = inter / (area_i + area_j - inter + 1e-8)
        order  = order[1:][iou < iou_thresh]
    return dets[keep]
