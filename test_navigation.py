#!/usr/bin/env python3
"""
test_navigation.py — Validate Astro-Tracker on randomly sampled tiles.

For each trial:
  1. Pick a random tile from dataset_tiles/ and look up its true (lat, lon)
     from tiles_metadata.csv (tile centre, converted from projected metres).
  2. Run localize() on that tile — without passing the ground-truth coords.
  3. Compare the estimated position to the true tile centre using Haversine.
  4. Print a summary table and aggregate statistics.

Usage:
  python test_navigation.py
  python test_navigation.py --n-trials 20 --db crater_db.parquet
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

import localize as loc   # import all helpers from localize.py

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("test-nav")

MOON_RADIUS_M = 1_737_400.0


# ── Coordinate helpers ────────────────────────────────────────────────────────
def metres_to_deg(m: float) -> float:
    """Projected metres → degrees (equirectangular, Moon R = 1 737 400 m)."""
    return m / MOON_RADIUS_M * (180.0 / math.pi)


def ground_truth_latlon(row: pd.Series) -> tuple[float, float]:
    """
    Extract tile-centre lat/lon in degrees from a metadata row.
    The metadata stores projected metres in lat_center / lon_center columns.
    """
    lat = metres_to_deg(float(row["lat_center"]))
    lon = metres_to_deg(float(row["lon_center"]))
    return lat, lon


# ── Single trial ──────────────────────────────────────────────────────────────
def run_trial(
    tile_path: Path,
    true_lat:  float,
    true_lon:  float,
    db:        pd.DataFrame,
    tree,
    model,
    anchors:   np.ndarray,
    verbose:   bool = True,
) -> dict:
    """Run one localisation trial and return a result dict."""
    result = loc.localize(str(tile_path), db, tree, model, anchors)

    error_m = None
    if result["lat_est"] is not None:
        error_m = loc.haversine_m(
            true_lat, true_lon,
            result["lat_est"], result["lon_est"],
        )

    trial = {
        "tile":          tile_path.name,
        "true_lat":      round(true_lat,  4),
        "true_lon":      round(true_lon,  4),
        "est_lat":       round(result["lat_est"],  4) if result["lat_est"]  is not None else None,
        "est_lon":       round(result["lon_est"],  4) if result["lon_est"]  is not None else None,
        "error_m":       round(error_m, 0)            if error_m is not None else None,
        "score":         round(result["score"], 4),
        "matched":       result["matched_count"],
        "n_detections":  len(result["query_craters"]),
        "success":       error_m is not None and error_m < 50_000,  # 50 km threshold
    }

    if verbose:
        status = "OK" if trial["success"] else ("FAIL" if error_m is not None else "NO-FIX")
        print(
            f"  [{status:6s}]  true=({true_lat:7.3f}°, {true_lon:8.3f}°)  "
            f"est=({trial['est_lat'] or 'N/A':>7}, {trial['est_lon'] or 'N/A':>8})  "
            f"error={f'{error_m/1000:.1f} km' if error_m is not None else 'N/A':>10}  "
            f"score={trial['score']:.3f}  matches={trial['matched']}"
        )
    return trial


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    # Load model, DB, metadata
    model, anchors = loc.load_model(args.weights)
    db, tree       = loc.load_db_and_tree(args.db)

    log.info(f"Loading tile metadata: {args.metadata}")
    meta = pd.read_csv(args.metadata)
    meta["filename"] = meta["filename"].str.strip('"')

    tiles_dir = Path(args.tiles_dir)

    # Restrict to tiles that actually exist on disk
    existing = [
        row for _, row in meta.iterrows()
        if (tiles_dir / row["filename"]).exists()
    ]
    if not existing:
        log.error("No tile PNG files found in %s", tiles_dir)
        return

    log.info(f"Available tiles on disk: {len(existing):,}")

    # Sample randomly
    n = min(args.n_trials, len(existing))
    sample = random.sample(existing, n)

    log.info(f"\nRunning {n} localisation trials …\n")
    print("=" * 100)

    results = []
    for row in sample:
        tile_path = tiles_dir / row["filename"]
        true_lat, true_lon = ground_truth_latlon(row)
        trial = run_trial(tile_path, true_lat, true_lon, db, tree, model, anchors)
        results.append(trial)

    print("=" * 100)

    # Aggregate statistics
    df_res = pd.DataFrame(results)
    attempted  = df_res["error_m"].notna().sum()
    successful = df_res["success"].sum()
    errors     = df_res["error_m"].dropna()

    print(f"\n── Summary ({'N=' + str(n)} trials) ─────────────────────────────────────────────────")
    print(f"  Localisation attempted : {attempted} / {n}")
    print(f"  Success (<50 km error) : {successful} / {attempted}")
    if len(errors) > 0:
        print(f"  Median error           : {errors.median()/1000:.1f} km")
        print(f"  Mean   error           : {errors.mean()/1000:.1f} km")
        print(f"  Min    error           : {errors.min()/1000:.1f} km")
        print(f"  Max    error           : {errors.max()/1000:.1f} km")
        print(f"  P90    error           : {errors.quantile(0.9)/1000:.1f} km")
    print(f"  Mean score             : {df_res['score'].mean():.4f}")
    print(f"  Mean detections/tile   : {df_res['n_detections'].mean():.1f}")
    print("─" * 80)

    # Save detailed results
    out_csv = Path(args.output)
    df_res.to_csv(out_csv, index=False)
    log.info(f"Detailed results saved → {out_csv}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    parser = argparse.ArgumentParser(
        description="Test Astro-Tracker navigation on random tiles"
    )
    parser.add_argument("--n-trials",  type=int,   default=10,
                        help="Number of random tiles to test")
    parser.add_argument("--tiles-dir", default="dataset_tiles",
                        help="Directory with tile PNG files")
    parser.add_argument("--metadata",  default="dataset_tiles/tiles_metadata.csv")
    parser.add_argument("--db",        default="crater_db.parquet",
                        help="Crater database (.parquet or .csv)")
    parser.add_argument("--weights",   default="checkpoints/last.weights.h5")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="Random seed for tile selection")
    parser.add_argument("--output",    default="test_navigation_results.csv",
                        help="CSV file to save per-trial results")
    args = parser.parse_args()

    random.seed(args.seed)
    main(args)
