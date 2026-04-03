#!/usr/bin/env python3
"""
localize.py — "Astro-Tracker" geometric lunar navigation.

Algorithm overview
──────────────────
Given a 416×416 query image of unknown location on the Moon:

  Step A  Detect craters with confidence > ANCHOR_CONF.
          Select up to TOP_K anchors by (confidence × radius).

  Step B  Build a local "star-pattern" from inter-anchor distances
          (in metres — scale is fixed: 1 px = 100 m).
          Query the KD-Tree built on crater_db for each anchor candidate.

  Step C  Hypothesis & Voting:
          For every (query anchor, DB candidate) pair, compute the
          implied tile-centre (lat, lon).  Project all other query
          craters to global coords under that hypothesis and count
          how many match DB entries within MATCH_DIST_M.
          Return the hypothesis with the best Navigation Fix Score.

  Output  (lat_est, lon_est, score, matched_count)

Coordinate system:  crater_db stores lat/lon in degrees.
Moon radius used for Haversine: 1 737 400 m.
"""

import argparse
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy.spatial import KDTree

# ── Constants ──────────────────────────────────────────────────────────────────
MOON_RADIUS_M    = 1_737_400.0   # metres
METERS_PER_PIXEL = 100.6
IMAGE_SIZE       = 416
FPN_CHANNELS     = 256

ANCHOR_CONF   = 0.2           # minimum confidence for query anchors
TOP_K         = 10                # number of anchor craters to use
MAX_CANDIDATES = 50              # DB candidates to test per anchor
MATCH_DIST_M  = 1000.0         # spatial tolerance for a "match" (2 km)
RADIUS_TOL    = 0.5              # fractional tolerance for radius match (±50 %)
MIN_MATCHES   = 4

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("localize")


# ── Model & anchors (identical to train_detector.py) ─────────────────────────
def build_model() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )
    feature = backbone.get_layer("block5a_project_bn").output
    x = inp * 255.0
    features = tf.keras.Model(backbone.input, feature)(x)
    x = tf.keras.layers.Conv2D(FPN_CHANNELS, 1, padding="same", activation="relu")(features)
    x = tf.keras.layers.Conv2D(FPN_CHANNELS, 3, padding="same", activation="relu")(x)
    cls_out = tf.keras.layers.Conv2D(9,  3, padding="same")(x)
    box_out = tf.keras.layers.Conv2D(36, 3, padding="same")(x)
    cls_out = tf.keras.layers.Reshape((-1, 1))(cls_out)
    box_out = tf.keras.layers.Reshape((-1, 4))(box_out)
    return tf.keras.Model(inp, [cls_out, box_out])


def build_anchors() -> np.ndarray:
    rows = []
    stride, base = 16, 64
    scales = [1.0, 1.26, 1.59]
    ratios  = [0.5, 1.0, 2.0]
    feat = IMAGE_SIZE // stride
    for r in range(feat):
        for c in range(feat):
            cx = (c + 0.5) * stride / IMAGE_SIZE
            cy = (r + 0.5) * stride / IMAGE_SIZE
            for s in scales:
                for ratio in ratios:
                    area = (base * s) ** 2
                    w = math.sqrt(area / ratio) / IMAGE_SIZE
                    h = math.sqrt(area * ratio) / IMAGE_SIZE
                    rows.append([cx, cy, w, h])
    return np.array(rows, dtype=np.float32)


# ── Decode detections ─────────────────────────────────────────────────────────
def decode_predictions(
    cls_logits: np.ndarray,
    box_deltas: np.ndarray,
    anchors:    np.ndarray,
    conf_thresh: float,
) -> np.ndarray:
    """Return (K, 5) array: cx_norm, cy_norm, w_norm, h_norm, conf."""
    conf = 1.0 / (1.0 + np.exp(-cls_logits[:, 0]))
    mask = conf > conf_thresh
    if mask.sum() == 0:
        return np.empty((0, 5), dtype=np.float32)
    d, a = box_deltas[mask], anchors[mask]
    cx = np.clip(a[:, 0] + d[:, 0] * a[:, 2],               0.0, 1.0)
    cy = np.clip(a[:, 1] + d[:, 1] * a[:, 3],               0.0, 1.0)
    w  = np.clip(a[:, 2] * np.exp(np.clip(d[:, 2], -4, 4)), 0.0, 1.0)
    h  = np.clip(a[:, 3] * np.exp(np.clip(d[:, 3], -4, 4)), 0.0, 1.0)
    return np.stack([cx, cy, w, h, conf[mask]], axis=1).astype(np.float32)


def nms(dets: np.ndarray, iou_thresh: float = 0.5) -> np.ndarray:
    if len(dets) == 0:
        return dets
    x1, y1 = dets[:, 0] - dets[:, 2] / 2, dets[:, 1] - dets[:, 3] / 2
    x2, y2 = dets[:, 0] + dets[:, 2] / 2, dets[:, 1] + dets[:, 3] / 2
    order  = dets[:, 4].argsort()[::-1]
    keep   = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_j = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
        iou = inter / (area_i + area_j - inter + 1e-8)
        order = order[1:][iou < iou_thresh]
    return dets[keep]


# ── Haversine distance on the Moon ────────────────────────────────────────────
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two points (degrees) on the Moon."""
    R = MOON_RADIUS_M
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


def haversine_vec(lat0: float, lon0: float,
                  lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vectorised haversine: distance from (lat0, lon0) to each row in arrays."""
    R = MOON_RADIUS_M
    φ0, φ = math.radians(lat0), np.radians(lats)
    dφ  = φ - φ0
    dλ  = np.radians(lons - lon0)
    a   = np.sin(dφ / 2) ** 2 + math.cos(φ0) * np.cos(φ) * np.sin(dλ / 2) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ── Pixel → relative offset in degrees ───────────────────────────────────────
def pixel_offset_deg(
    cx_norm: float, cy_norm: float, ref_lat_deg: float
) -> tuple[float, float]:
    """
    Return (Δlat_deg, Δlon_deg) of a crater relative to the tile centre
    given normalised pixel coords (cx, cy) ∈ [0, 1].
    cy = 0 → top of tile → most northerly point.
    """
    # pixel offset from tile centre (positive y = south)
    dx_px = (cx_norm - 0.5) * IMAGE_SIZE
    dy_px = (cy_norm - 0.5) * IMAGE_SIZE

    dx_m =  dx_px * METERS_PER_PIXEL   # east  (+)
    dy_m = -dy_px * METERS_PER_PIXEL   # north (+)

    dlat = dy_m / MOON_RADIUS_M * (180.0 / math.pi)

    current_lat_rad = math.radians(ref_lat_deg + dlat)
    cos_lat = math.cos(current_lat_rad)

    cos_lat = max(abs(cos_lat), 0.01)

    dlon = (dx_m / MOON_RADIUS_M) / cos_lat * (180.0 / math.pi)
    return dlat, dlon


# ── Step A: detect craters in one image ──────────────────────────────────────
def detect(
    image_path: str,
    model: tf.keras.Model,
    anchors: np.ndarray,
    conf_thresh: float = ANCHOR_CONF,
) -> np.ndarray:
    """
    Run detector on image_path.
    Returns (K, 5) array: cx_norm, cy_norm, w_norm, h_norm, confidence
    sorted by (confidence × w) descending (largest/best craters first).
    """
    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    img_t = tf.constant(img[np.newaxis])                 # (1, 416, 416, 3)
    cls_out, box_out = model(img_t, training=False)
    dets = decode_predictions(
        cls_out[0].numpy(), box_out[0].numpy(), anchors, conf_thresh
    )
    dets = nms(dets)
    if len(dets) == 0:
        return dets
    # sort by detection quality = confidence × bounding-box width
    quality = dets[:, 4] * dets[:, 2]
    return dets[quality.argsort()[::-1]]


# ── Step B & C: hypothesis search ────────────────────────────────────────────
def localize(
    image_path: str,
    crater_db:  pd.DataFrame,
    db_tree:    KDTree,
    model:      tf.keras.Model,
    anchors:    np.ndarray,
) -> dict:
    """
    Main entry point for one query image.

    Returns a dict with keys:
      lat_est, lon_est, score, matched_count,
      query_craters, all_hypotheses
    """
    log.info(f"Detecting craters in: {image_path}")
    dets = detect(image_path, model, anchors, conf_thresh=ANCHOR_CONF)

    if len(dets) < 2:
        log.warning("Fewer than 2 craters detected — cannot localise.")
        return {"lat_est": None, "lon_est": None, "score": 0.0,
                "matched_count": 0, "query_craters": dets,
                "all_hypotheses": []}

    # Keep top-K anchors
    dets = dets[dets[:, 2] * IMAGE_SIZE > 10]
    # Берем топ-K самых уверенных и крупных
    anchors_q = dets[:TOP_K]

    log.info(f"Using {len(anchors_q)} anchor craters (conf ≥ {ANCHOR_CONF})")

    db_lats = crater_db["lat"].values
    db_lons = crater_db["lon"].values
    db_radii = crater_db["radius_m"].values

    hypotheses = []

    # Step B: For each query anchor, find DB candidates with similar radius
    for qi, (cx_n, cy_n, w_n, h_n, conf_q) in enumerate(anchors_q):
        q_radius_m = (w_n * IMAGE_SIZE / 2.0) * METERS_PER_PIXEL

        # Filter candidates by matching crater radius (±RADIUS_TOL fraction)
        radius_lo = q_radius_m * (1.0 - RADIUS_TOL)
        radius_hi = q_radius_m * (1.0 + RADIUS_TOL)
        r_mask = (db_radii >= radius_lo) & (db_radii <= radius_hi)
        r_candidates = np.where(r_mask)[0]

        if len(r_candidates) == 0:
            # Fallback: no radius filter
            r_candidates = np.arange(len(db_lats))

        # Among radius-filtered candidates, pick top MAX_CANDIDATES by confidence
        cand_conf = crater_db["confidence"].values[r_candidates]
        top_idx   = r_candidates[cand_conf.argsort()[::-1][:MAX_CANDIDATES]]

        # Step C: for each candidate, build hypothesis for tile centre
        for db_idx in top_idx:
            db_lat = db_lats[db_idx]
            db_lon = db_lons[db_idx]

            # Δ from tile centre to this query crater (in degrees)
            # Use db_lat as an approximation for ref_lat
            dlat_q, dlon_q = pixel_offset_deg(cx_n, cy_n, db_lat)

            # Implied tile centre
            lat_hyp = db_lat - dlat_q
            lon_hyp = db_lon - dlon_q

            # Project all query anchors to global coords under this hypothesis
            proj_lats = np.zeros(len(anchors_q))
            proj_lons = np.zeros(len(anchors_q))
            proj_radii = np.zeros(len(anchors_q))
            for j, (cx_j, cy_j, w_j, h_j, _) in enumerate(anchors_q):
                dlat_j, dlon_j = pixel_offset_deg(cx_j, cy_j, lat_hyp)
                proj_lats[j]  = lat_hyp + dlat_j
                proj_lons[j]  = lon_hyp + dlon_j
                proj_radii[j] = (w_j * IMAGE_SIZE / 2.0) * METERS_PER_PIXEL

            # Vote: count how many projected craters match DB entries
            match_count   = 0
            match_dists   = []
            for j in range(len(anchors_q)):
                # Narrow KD-Tree search around projected position
                search_deg = MATCH_DIST_M / MOON_RADIUS_M * (180.0 / math.pi) * 3.0
                nearby_idx = db_tree.query_ball_point(
                    [proj_lats[j], proj_lons[j]], r=search_deg
                )
                if not nearby_idx:
                    continue
                nearby = crater_db.iloc[nearby_idx]
                dists  = haversine_vec(
                    proj_lats[j], proj_lons[j],
                    nearby["lat"].values, nearby["lon"].values,
                )
                min_dist = dists.min()
                if min_dist < MATCH_DIST_M:
                    match_count += 1
                    match_dists.append(min_dist)

            if match_count == 0:
                continue

            if match_count < MIN_MATCHES:
                continue

            mean_dist = np.mean(match_dists)
            rms_dist  = np.sqrt(np.mean(np.array(match_dists) ** 2))
            # Score: more matches + smaller mean distance = better
            score = match_count / (1.0 + mean_dist / MATCH_DIST_M)

            hypotheses.append({
                "lat_est":      lat_hyp,
                "lon_est":      lon_hyp,
                "score":        score,
                "matched_count": match_count,
                "mean_dist_m":  mean_dist,
                "rms_dist_m":   rms_dist,
                "anchor_qi":    qi,
                "anchor_db":    db_idx,
            })

    if not hypotheses:
        log.warning("No hypothesis survived voting.")
        return {"lat_est": None, "lon_est": None, "score": 0.0,
                "matched_count": 0, "query_craters": dets,
                "all_hypotheses": []}

    # Best hypothesis
    best = max(hypotheses, key=lambda h: h["score"])

    log.info(
        f"Navigation Fix: lat={best['lat_est']:.4f}°  "
        f"lon={best['lon_est']:.4f}°  "
        f"score={best['score']:.3f}  "
        f"matches={best['matched_count']}/{len(anchors_q)}"
    )

    return {
        "lat_est":        best["lat_est"],
        "lon_est":        best["lon_est"],
        "score":          best["score"],
        "matched_count":  best["matched_count"],
        "mean_dist_m":    best["mean_dist_m"],
        "rms_dist_m":     best["rms_dist_m"],
        "query_craters":  dets,
        "all_hypotheses": hypotheses,
    }


# ── Loader helpers ────────────────────────────────────────────────────────────
def load_db_and_tree(db_path: str) -> tuple[pd.DataFrame, KDTree]:
    """Load crater_db from .parquet or .csv and build a KD-Tree on (lat, lon)."""
    path = Path(db_path)
    log.info(f"Loading crater DB: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    log.info(f"DB loaded: {len(df):,} craters")
    tree = KDTree(df[["lat", "lon"]].values)
    return df, tree


def load_model(weights_path: str) -> tuple[tf.keras.Model, np.ndarray]:
    log.info("Building model …")
    model = build_model()
    log.info(f"Loading weights: {weights_path}")
    model.load_weights(weights_path)
    return model, build_anchors()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    parser = argparse.ArgumentParser(description="Astro-Tracker lunar localisation")
    parser.add_argument("image",         help="Path to query 416×416 PNG")
    parser.add_argument("--db",          default="crater_db.parquet",
                        help="Path to crater_db (.parquet or .csv)")
    parser.add_argument("--weights",     default="checkpoints/last.weights.h5")
    parser.add_argument("--conf-thresh", type=float, default=ANCHOR_CONF,
                        help="Minimum confidence for anchor craters")
    args = parser.parse_args()

    model, ancs = load_model(args.weights)
    db, tree    = load_db_and_tree(args.db)

    result = localize(args.image, db, tree, model, ancs)

    print("\n── Navigation Fix ──────────────────────────────────────────")
    if result["lat_est"] is not None:
        print(f"  Estimated position : lat={result['lat_est']:.4f}°  "
              f"lon={result['lon_est']:.4f}°")
        print(f"  Fix quality score  : {result['score']:.4f}")
        print(f"  Craters matched    : {result['matched_count']}")
        print(f"  Mean match dist    : {result.get('mean_dist_m', 0):.0f} m")
        print(f"  RMS  match dist    : {result.get('rms_dist_m',  0):.0f} m")
    else:
        print("  Could not determine position (too few detections).")
    print("────────────────────────────────────────────────────────────\n")
