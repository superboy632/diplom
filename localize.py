#!/usr/bin/env python3
"""
localize.py — Astro-Tracker geometric lunar navigation.

Algorithm overview
──────────────────
Given a 416×416 query image of unknown location on the Moon:

  Step A  Detect craters with confidence > ANCHOR_CONF.
          Select up to TOP_K anchors sorted by (confidence × radius).

  Step B  For each query anchor find DB candidates with a matching radius
          (within ±RADIUS_TOL fraction).

  Step C  Hypothesis & Voting:
          For every (query anchor, DB candidate) pair, compute the implied
          tile-centre (lat, lon).  Project all other query craters to global
          coords under that hypothesis and count how many match DB entries
          within MATCH_DIST_M.  Return the hypothesis with the best score.

  Output  dict with lat_est, lon_est, score, matched_count, …

Coordinate system
─────────────────
crater_db stores lat/lon in flat-projected degrees (see geo_utils.py).
Pixel offsets are converted with the same flat scale — no cos(lat).
Moon radius used for Haversine voting: 1 737 400 m.
"""

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from geo_utils import (
    IMAGE_SIZE,
    METERS_PER_PIXEL,
    MOON_RADIUS_M,
    haversine_m,   # re-exported: used by test_navigation.py as loc.haversine_m
    haversine_vec,
    pixel_offset_deg,
)

# ── Tunable constants ─────────────────────────────────────────────────────────
ANCHOR_CONF    = 0.2     # minimum detector confidence for query craters
TOP_K          = 10      # number of anchor craters used for hypothesis search
MAX_CANDIDATES = 50      # DB candidates tested per anchor
MATCH_DIST_M   = 500.0   # spatial tolerance for a "match" in metres
RADIUS_TOL     = 0.5     # fractional radius tolerance (±50 %)
MIN_MATCHES    = 5       # minimum matches required to accept a hypothesis

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("localize")


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(weights_path: str) -> tuple:
    """
    Load YOLOv8 model from weights_path.
    Returns (model, anchors) where anchors is None (not used by YOLO).
    """
    from ultralytics import YOLO
    log.info(f"Loading YOLOv8 model: {weights_path}")
    model = YOLO(weights_path)
    return model, None


# ── Step A: detect craters in one image ──────────────────────────────────────
def detect(
    image_path:  str,
    model,
    anchors,        # ignored for YOLO, kept for API compatibility
    conf_thresh: float = ANCHOR_CONF,
) -> np.ndarray:
    """
    Run YOLOv8 detector on image_path.
    Returns (K, 5) array: cx_norm, cy_norm, w_norm, h_norm, confidence,
    sorted by (confidence × width) descending.
    """
    results = model.predict(
        source=image_path,
        conf=conf_thresh,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return np.empty((0, 5), dtype=np.float32)

    boxes = results[0].boxes
    xywhn = boxes.xywhn.cpu().numpy()   # (K, 4): cx_norm, cy_norm, w_norm, h_norm
    confs = boxes.conf.cpu().numpy()    # (K,)

    dets = np.concatenate([xywhn, confs[:, None]], axis=1).astype(np.float32)

    if len(dets) == 0:
        return dets

    quality = dets[:, 4] * dets[:, 2]
    return dets[quality.argsort()[::-1]]


# ── Steps B & C: hypothesis search ───────────────────────────────────────────
def localize(
    image_path: str,
    crater_db:  pd.DataFrame,
    db_tree:    KDTree,
    model,
    anchors,
) -> dict:
    """
    Estimate tile position from a single 416×416 query image.

    Returns a dict with keys:
        lat_est, lon_est, score, matched_count,
        mean_dist_m, rms_dist_m, query_craters, all_hypotheses
    """
    log.info(f"Detecting craters in: {image_path}")
    dets = detect(image_path, model, anchors, conf_thresh=ANCHOR_CONF)

    if len(dets) < 2:
        log.warning("Fewer than 2 craters detected — cannot localise.")
        return {
            "lat_est": None, "lon_est": None, "score": 0.0,
            "matched_count": 0, "query_craters": dets, "all_hypotheses": [],
        }

    # Discard very small detections and keep top-K by quality
    dets      = dets[dets[:, 2] * IMAGE_SIZE > 10]
    anchors_q = dets[:TOP_K]
    log.info(f"Using {len(anchors_q)} anchor craters (conf ≥ {ANCHOR_CONF})")

    db_lats  = crater_db["lat"].values
    db_lons  = crater_db["lon"].values
    db_radii = crater_db["radius_m"].values

    hypotheses = []

    # ── Step B: for each query anchor find radius-matched DB candidates ───────
    for qi, (cx_n, cy_n, w_n, h_n, conf_q) in enumerate(anchors_q):
        q_radius_m = (w_n * IMAGE_SIZE / 2.0) * METERS_PER_PIXEL
        radius_lo  = q_radius_m * (1.0 - RADIUS_TOL)
        radius_hi  = q_radius_m * (1.0 + RADIUS_TOL)
        r_mask     = (db_radii >= radius_lo) & (db_radii <= radius_hi)
        r_cands    = np.where(r_mask)[0]

        if len(r_cands) == 0:
            r_cands = np.arange(len(db_lats))   # fallback: no radius filter

        cand_conf = crater_db["confidence"].values[r_cands]
        top_idx   = r_cands[cand_conf.argsort()[::-1][:MAX_CANDIDATES]]

        # ── Step C: build and score each hypothesis ───────────────────────────
        for db_idx in top_idx:
            db_lat = db_lats[db_idx]
            db_lon = db_lons[db_idx]

            # Offset from tile centre to this query crater (flat-projected deg)
            dlat_q, dlon_q = pixel_offset_deg(cx_n, cy_n)

            # Implied tile centre
            lat_hyp = db_lat - dlat_q
            lon_hyp = db_lon - dlon_q

            # Project all query craters to global coords under this hypothesis
            proj_lats  = np.empty(len(anchors_q))
            proj_lons  = np.empty(len(anchors_q))
            proj_radii = np.empty(len(anchors_q))
            for j, (cx_j, cy_j, w_j, h_j, _) in enumerate(anchors_q):
                dlat_j, dlon_j  = pixel_offset_deg(cx_j, cy_j)
                proj_lats[j]    = lat_hyp + dlat_j
                proj_lons[j]    = lon_hyp + dlon_j
                proj_radii[j]   = (w_j * IMAGE_SIZE / 2.0) * METERS_PER_PIXEL

            # Vote: count projected craters that match a DB entry
            match_count = 0
            match_dists = []
            search_deg  = MATCH_DIST_M / MOON_RADIUS_M * (180.0 / math.pi) * 3.0

            for j in range(len(anchors_q)):
                nearby_idx = db_tree.query_ball_point(
                    [proj_lats[j], proj_lons[j]], r=search_deg
                )
                if not nearby_idx:
                    continue
                nearby   = crater_db.iloc[nearby_idx]
                dists    = haversine_vec(
                    proj_lats[j], proj_lons[j],
                    nearby["lat"].values, nearby["lon"].values,
                )
                min_dist = dists.min()
                if min_dist < MATCH_DIST_M:
                    match_count += 1
                    match_dists.append(min_dist)

            if match_count < MIN_MATCHES:
                continue

            mean_dist = float(np.mean(match_dists))
            rms_dist  = float(np.sqrt(np.mean(np.array(match_dists) ** 2)))
            score     = match_count / (1.0 + mean_dist / MATCH_DIST_M)

            hypotheses.append({
                "lat_est":       lat_hyp,
                "lon_est":       lon_hyp,
                "score":         score,
                "matched_count": match_count,
                "mean_dist_m":   mean_dist,
                "rms_dist_m":    rms_dist,
                "anchor_qi":     qi,
                "anchor_db":     db_idx,
            })

    if not hypotheses:
        log.warning("No hypothesis survived voting.")
        return {
            "lat_est": None, "lon_est": None, "score": 0.0,
            "matched_count": 0, "query_craters": dets, "all_hypotheses": [],
        }

    if hypotheses:
        scores = sorted([h["score"] for h in hypotheses], reverse=True)[:10]
        lats   = [h["lat_est"] for h in sorted(hypotheses, key=lambda h: h["score"], reverse=True)[:5]]
        lons   = [h["lon_est"] for h in sorted(hypotheses, key=lambda h: h["score"], reverse=True)[:5]]
        log.info(f"Top-10 scores: {[round(s,1) for s in scores]}")
        log.info(f"Top-5 lats:    {[round(l,2) for l in lats]}")
        log.info(f"Top-5 lons:    {[round(l,2) for l in lons]}")

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
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    log.info(f"DB loaded: {len(df):,} craters")
    tree = KDTree(df[["lat", "lon"]].values)
    return df, tree


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Astro-Tracker lunar localisation")
    parser.add_argument("image",         help="Path to query 416×416 PNG")
    parser.add_argument("--db",          default="crater_db_v2.parquet")
    parser.add_argument("--weights",     default="checkpoints/best.pt")
    parser.add_argument("--conf-thresh", type=float, default=ANCHOR_CONF)
    args = parser.parse_args()

    _model, _ancs = load_model(args.weights)
    _db, _tree    = load_db_and_tree(args.db)

    result = localize(args.image, _db, _tree, _model, _ancs)

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