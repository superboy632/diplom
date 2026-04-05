#!/usr/bin/env python3
"""
build_crater_db.py — Run trained detector on every tile in dataset_tiles/,
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
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from geo_utils import (
    IMAGE_SIZE,
    METERS_PER_PIXEL,
    MOON_RADIUS_M,
    metres_to_deg,
    nms,
    pixel_to_latlon,
)

# ── Constants ──────────────────────────────────────────────────────────────────
FPN_CHANNELS   = 256
CONF_THRESHOLD = 0.3    # lowered from 0.6 to populate a denser DB
NMS_IOU        = 0.5    # intra-tile NMS threshold
MERGE_DIST_M   = 500.0  # cross-tile deduplication radius in metres

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("crater-db")


# ── Model (must be identical to train_detector.py) ────────────────────────────
def build_model() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )
    feature  = backbone.get_layer("block5a_project_bn").output
    x        = inp * 255.0
    features = tf.keras.Model(backbone.input, feature)(x)
    x        = tf.keras.layers.Conv2D(FPN_CHANNELS, 1, padding="same", activation="relu")(features)
    x        = tf.keras.layers.Conv2D(FPN_CHANNELS, 3, padding="same", activation="relu")(x)
    cls_out  = tf.keras.layers.Conv2D(9,  3, padding="same")(x)
    box_out  = tf.keras.layers.Conv2D(36, 3, padding="same")(x)
    cls_out  = tf.keras.layers.Reshape((-1, 1))(cls_out)
    box_out  = tf.keras.layers.Reshape((-1, 4))(box_out)
    return tf.keras.Model(inp, [cls_out, box_out])


# ── Anchor generator (must be identical to train_detector.py) ─────────────────
def build_anchors() -> np.ndarray:
    """Return (N, 4) array of (cx, cy, w, h) in normalised [0, 1] coords."""
    import math
    rows = []
    stride, base = 16, 64
    scales = [1.0, 1.26, 1.59]
    ratios = [0.5, 1.0, 2.0]
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


# ── Decode raw model output ───────────────────────────────────────────────────
def decode_predictions(
    cls_logits: np.ndarray,
    box_deltas: np.ndarray,
    anchors:    np.ndarray,
    conf_thresh: float,
) -> np.ndarray:
    """Return (K, 5) array of (cx, cy, w, h, conf) in normalised coords."""
    conf = 1.0 / (1.0 + np.exp(-cls_logits[:, 0]))
    mask = conf > conf_thresh
    if mask.sum() == 0:
        return np.empty((0, 5), dtype=np.float32)
    d  = box_deltas[mask]
    a  = anchors[mask]
    cx = np.clip(a[:, 0] + d[:, 0] * a[:, 2],               0.0, 1.0)
    cy = np.clip(a[:, 1] + d[:, 1] * a[:, 3],               0.0, 1.0)
    w  = np.clip(a[:, 2] * np.exp(np.clip(d[:, 2], -4, 4)), 0.0, 1.0)
    h  = np.clip(a[:, 3] * np.exp(np.clip(d[:, 3], -4, 4)), 0.0, 1.0)
    return np.stack([cx, cy, w, h, conf[mask]], axis=1).astype(np.float32)


# ── Cross-tile deduplication ──────────────────────────────────────────────────
def deduplicate(df: pd.DataFrame, merge_dist_m: float = MERGE_DIST_M) -> pd.DataFrame:
    """
    Remove duplicate craters: keep the highest-confidence detection within
    merge_dist_m of any other.  Processes craters sorted by confidence desc.
    """
    log.info(f"Deduplicating {len(df):,} detections (threshold = {merge_dist_m} m) …")
    df = df.sort_values("confidence", ascending=False).reset_index(drop=True)
    kept = np.ones(len(df), dtype=bool)

    lat_rad = np.radians(df["lat"].values)
    lon_rad = np.radians(df["lon"].values)

    for i in range(0, len(df)):
        if not kept[i]:
            continue
        j_idx = np.where(kept[i + 1:])[0] + i + 1
        if len(j_idx) == 0:
            break
        dlat     = lat_rad[j_idx] - lat_rad[i]
        dlon     = lon_rad[j_idx] - lon_rad[i]
        sin_dlat = np.sin(dlat / 2)
        sin_dlon = np.sin(dlon / 2)
        a    = sin_dlat ** 2 + np.cos(lat_rad[i]) * np.cos(lat_rad[j_idx]) * sin_dlon ** 2
        dist = 2.0 * MOON_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        kept[j_idx[dist < merge_dist_m]] = False
        if i % 50_000 == 0 and i > 0:
            log.info(f"  dedup progress: {i:,} / {len(df):,}")

    result = df[kept].reset_index(drop=True)
    log.info(f"After dedup: {len(result):,} unique craters")
    return result


# ── Main pipeline ─────────────────────────────────────────────────────────────
def build_db(
    tiles_dir:    str,
    metadata_csv: str,
    weights_path: str,
    output_path:  str,
    conf_thresh:  float,
    batch_size:   int,
) -> None:
    log.info("Building model …")
    model = build_model()
    log.info(f"Loading weights: {weights_path}")
    model.load_weights(weights_path)
    anchors = build_anchors()

    log.info(f"Loading tile metadata: {metadata_csv}")
    meta = pd.read_csv(metadata_csv)
    meta["filename"] = meta["filename"].str.strip('"')
    log.info(f"Tiles in metadata: {len(meta):,}")

    tiles_dir = Path(tiles_dir)
    records   = []
    filenames = meta["filename"].tolist()

    log.info(f"Running inference on {len(filenames):,} tiles (batch={batch_size}) …")

    for start in tqdm(range(0, len(filenames), batch_size), desc="Inference"):
        end         = min(start + batch_size, len(filenames))
        batch_files = filenames[start:end]
        batch_meta  = meta.iloc[start:end]

        imgs      = []
        valid_idx = []
        for k, fname in enumerate(batch_files):
            fpath = tiles_dir / fname
            if not fpath.exists():
                continue
            img = np.array(Image.open(fpath).convert("RGB"), dtype=np.float32) / 255.0
            imgs.append(img)
            valid_idx.append(k)

        if not imgs:
            continue

        imgs_t = tf.constant(np.stack(imgs))
        cls_out, box_out = model(imgs_t, training=False)

        for j, k in enumerate(valid_idx):
            row       = batch_meta.iloc[k]
            lat_min_m = float(row["lat_min"])
            lat_max_m = float(row["lat_max"])
            lon_min_m = float(row["lon_min"])
            lon_max_m = float(row["lon_max"])

            dets = decode_predictions(
                cls_out[j].numpy(), box_out[j].numpy(), anchors, conf_thresh
            )
            dets = nms(dets, iou_thresh=NMS_IOU)

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
                    "tile_filename": batch_files[k],
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
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    parser = argparse.ArgumentParser(description="Build global lunar crater database")
    parser.add_argument("--tiles-dir",   default="dataset_tiles")
    parser.add_argument("--metadata",    default="dataset_tiles/tiles_metadata.csv")
    parser.add_argument("--weights",     default="checkpoints/last.weights.h5")
    parser.add_argument("--output",      default="crater_db.parquet")
    parser.add_argument("--conf-thresh", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--batch-size",  type=int,   default=8)
    args = parser.parse_args()

    build_db(
        tiles_dir=args.tiles_dir,
        metadata_csv=args.metadata,
        weights_path=args.weights,
        output_path=args.output,
        conf_thresh=args.conf_thresh,
        batch_size=args.batch_size,
    )
