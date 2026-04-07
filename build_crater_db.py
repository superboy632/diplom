#!/usr/bin/env python3
"""
build_crater_db.py — Run trained YOLOv8 detector on every tile in dataset_tiles/,
convert detections to global lunar coordinates, deduplicate overlapping
detections, and save to crater_db.parquet.

Coordinate note
───────────────
tiles_metadata.csv stores tile bounds in projected metres using a flat
equirectangular projection (Moon sphere, R = 1 737 400 m).
pixel_to_latlon() converts those bounds to degrees with the same linear
scale on both axes — no cos(lat) correction.  See geo_utils.py for details.

Resolution: METERS_PER_PIXEL = 100.0  →  radius_m = (w_px / 2) × 100
"""

import argparse
import logging
from pathlib import Path
from sklearn.neighbors import BallTree

import numpy as np
import pandas as pd
from tqdm import tqdm

from geo_utils import (
    IMAGE_SIZE,
    METERS_PER_PIXEL,
    MOON_RADIUS_M,
    pixel_to_latlon,
)

# ── Constants ──────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.35   # YOLOv8 confidence threshold for DB building
NMS_IOU        = 0.5    # intra-tile NMS IoU threshold (applied after YOLO NMS)
MERGE_DIST_M   = 500.0  # cross-tile deduplication radius in metres

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("crater-db")


# ── Cross-tile deduplication ──────────────────────────────────────────────────
def deduplicate(df: pd.DataFrame, merge_dist_m: float = MERGE_DIST_M) -> pd.DataFrame:
    log.info(f"Deduplicating {len(df):,} detections (threshold = {merge_dist_m} m) …")

    # 1. Sort by confidence (important for NMS)
    df = df.sort_values("confidence", ascending=False).reset_index(drop=True)

    # 2. Convert coordinates to radians for BallTree (haversine metric)
    coords = np.radians(df[["lat", "lon"]].values)

    # 3. Build tree
    tree = BallTree(coords, metric='haversine')

    # Radius in radians
    radius_rad = merge_dist_m / MOON_RADIUS_M

    kept = np.ones(len(df), dtype=bool)

    log.info("Starting fast spatial search...")
    for i in range(len(df)):
        if not kept[i]:
            continue

        indices = tree.query_radius(coords[i:i+1], r=radius_rad)[0]

        for idx in indices:
            if idx > i:
                kept[idx] = False

        if i % 100_000 == 0 and i > 0:
            log.info(f"  dedup progress: {i:,} / {len(df):,}")

    result = df[kept].reset_index(drop=True)
    log.info(f"After fast dedup: {len(result):,} unique craters")
    return result


# ── Main pipeline ─────────────────────────────────────────────────────────────
def build_db(
    tiles_dir:    str,
    metadata_csv: str,
    weights_path: str,
    output_path:  str,
    conf_thresh:  float,
) -> None:
    from ultralytics import YOLO

    log.info(f"Loading YOLOv8 model: {weights_path}")
    model = YOLO(weights_path)

    log.info(f"Loading tile metadata: {metadata_csv}")
    meta = pd.read_csv(metadata_csv)
    meta["filename"] = meta["filename"].str.strip('"')
    log.info(f"Tiles in metadata: {len(meta):,}")

    tiles_dir = Path(tiles_dir)
    records   = []
    filenames = meta["filename"].tolist()

    log.info(f"Running YOLOv8 inference on {len(filenames):,} tiles …")

    for idx, fname in enumerate(tqdm(filenames, desc="Inference")):
        fpath = tiles_dir / fname
        if not fpath.exists():
            continue

        row       = meta.iloc[idx]
        lat_min_m = float(row["lat_min"])
        lat_max_m = float(row["lat_max"])
        lon_min_m = float(row["lon_min"])
        lon_max_m = float(row["lon_max"])

        results = model.predict(
            source=str(fpath),
            conf=conf_thresh,
            iou=NMS_IOU,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            continue

        boxes = results[0].boxes
        if len(boxes) == 0:
            continue

        # xywhn: normalized (cx, cy, w, h) in [0, 1]
        xywhn = boxes.xywhn.cpu().numpy()   # (K, 4)
        confs = boxes.conf.cpu().numpy()    # (K,)

        # Build (K, 5) array: cx_norm, cy_norm, w_norm, h_norm, conf
        dets = np.concatenate([xywhn, confs[:, None]], axis=1).astype(np.float32)

        for cx_n, cy_n, w_n, h_n, conf in dets:
            lat, lon = pixel_to_latlon(
                cx_n, cy_n, lat_min_m, lat_max_m, lon_min_m, lon_max_m
            )
            radius_m = (w_n * IMAGE_SIZE / 2.0) * METERS_PER_PIXEL
            records.append({
                "lat":           round(float(lat),      6),
                "lon":           round(float(lon),      6),
                "radius_m":      round(float(radius_m), 1),
                "confidence":    round(float(conf),     4),
                "tile_filename": fname,
            })

    if not records:
        log.warning("No detections found — check confidence threshold and weights.")
        return

    df = pd.DataFrame(records)
    log.info(f"Total raw detections: {len(df):,}")

    df = deduplicate(df, merge_dist_m=MERGE_DIST_M)
    df.insert(0, "crater_id", range(len(df)))

    out = Path(output_path)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    log.info(f"Saved {len(df):,} craters → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build global lunar crater database")
    parser.add_argument("--tiles-dir",   default="dataset_tiles")
    parser.add_argument("--metadata",    default="dataset_tiles/tiles_metadata.csv")
    parser.add_argument("--weights",     default="checkpoints/best.pt")
    parser.add_argument("--output",      default="crater_db.parquet")
    parser.add_argument("--conf-thresh", type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    build_db(
        tiles_dir=args.tiles_dir,
        metadata_csv=args.metadata,
        weights_path=args.weights,
        output_path=args.output,
        conf_thresh=args.conf_thresh,
    )
