#!/usr/bin/env python3
"""
train_detector.py — Исправленная стабильная версия
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from build_tf_dataset import create_crater_detection_dataset

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("crater-det")

IMAGE_SIZE = 416
FPN_CHANNELS = 256

LOG_DIR = Path("logs")
CKPT_DIR = Path("checkpoints")
LOG_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)


class AnchorGenerator:
    def __init__(self):
        self._xywh = self._build_anchors()
        cx, cy, w, h = self._xywh.T
        self._xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).clip(0.0, 1.0).astype(np.float32)

    def _build_anchors(self):
        rows = []
        # Используем только один уровень для простоты и согласованности
        stride = 16
        base = 64
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

    @property
    def xywh(self):
        return self._xywh

    @property
    def xyxy(self):
        return self._xyxy

    @property
    def n(self):
        return len(self._xywh)


def pairwise_iou(boxes1, boxes2):
    x1 = np.maximum(boxes1[:, 0][:, None], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1][:, None], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2][:, None], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3][:, None], boxes2[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-8)


def encode_batch_targets(bboxes_px, class_ids, anchors):
    B = bboxes_px.shape[0]
    N = anchors.n
    cls_tgt = np.zeros((B, N), dtype=np.float32)
    box_tgt = np.zeros((B, N, 4), dtype=np.float32)

    for b in range(B):
        valid = class_ids[b] >= 0
        gt_px = bboxes_px[b, valid]
        if len(gt_px) == 0:
            continue
        gt_norm = gt_px / IMAGE_SIZE

        ious = pairwise_iou(anchors.xyxy, gt_norm)
        best_anchors = np.argmax(ious, axis=0)

        for i, a_idx in enumerate(best_anchors):
            cls_tgt[b, a_idx] = 1.0
            gt = gt_norm[i]
            gcx = (gt[0] + gt[2]) / 2
            gcy = (gt[1] + gt[3]) / 2
            gw = gt[2] - gt[0]
            gh = gt[3] - gt[1]

            acx, acy, aw, ah = anchors.xywh[a_idx]
            box_tgt[b, a_idx, 0] = (gcx - acx) / (aw + 1e-8)
            box_tgt[b, a_idx, 1] = (gcy - acy) / (ah + 1e-8)
            box_tgt[b, a_idx, 2] = np.log(gw / (aw + 1e-8) + 1e-8)
            box_tgt[b, a_idx, 3] = np.log(gh / (ah + 1e-8) + 1e-8)

    return cls_tgt, box_tgt


def detection_loss(cls_pred, box_pred, bboxes_px, class_ids, anchors):
    cls_tgt, box_tgt = encode_batch_targets(bboxes_px.numpy(), class_ids.numpy(), anchors)
    cls_tgt = tf.constant(cls_tgt, tf.float32)
    box_tgt = tf.constant(box_tgt, tf.float32)

    # Classification loss
    cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=cls_tgt,
        logits=tf.squeeze(cls_pred, -1)
    )
    cls_loss = tf.reduce_mean(cls_loss)

    # Box loss
    diff = tf.abs(box_pred - box_tgt)
    smooth_l1 = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
    box_loss = tf.reduce_mean(smooth_l1 * tf.expand_dims(cls_tgt, -1))

    total_loss = cls_loss + 5.0 * box_loss
    return total_loss, cls_loss, box_loss


def build_model(backbone_weights='imagenet'):
    inp = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=backbone_weights, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    # Используем только один feature map для согласованности размеров
    feature = backbone.get_layer('block5a_project_bn').output  # ~26x26

    x = inp * 255.0
    features = tf.keras.Model(backbone.input, feature)(x)

    x = tf.keras.layers.Conv2D(FPN_CHANNELS, 1, padding='same', activation='relu')(features)
    x = tf.keras.layers.Conv2D(FPN_CHANNELS, 3, padding='same', activation='relu')(x)

    cls_out = tf.keras.layers.Conv2D(9, 3, padding='same')(x)  # 9 anchors
    box_out = tf.keras.layers.Conv2D(36, 3, padding='same')(x)

    cls_out = tf.keras.layers.Reshape((-1, 1))(cls_out)
    box_out = tf.keras.layers.Reshape((-1, 4))(box_out)

    model = tf.keras.Model(inp, [cls_out, box_out])
    log.info(f"Model built with {model.count_params():,} parameters")
    return model


def train(args):
    train_ds, valid_ds, builder = create_crater_detection_dataset(
        dataset_path="tensorflow_dataset",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=args.batch,
        use_augmentation=True,
        subsample_ratio=args.subsample,
    )

    info = builder.get_dataset_info()
    log.info(f"Dataset: train={info.get('train_samples')} | valid={info.get('valid_samples')}")

    model = build_model(backbone_weights=None if args.weights == "none" else "imagenet")
    optimizer = tf.keras.optimizers.Adam(args.lr)

    anc = AnchorGenerator()
    log.info(f"Anchors generated: {anc.n:,}")

    log.info("Starting training...")

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        steps = 0

        for batch in train_ds:
            images, bboxes, class_ids = batch

            with tf.GradientTape() as tape:
                cls_pred, box_pred = model(images, training=True)
                loss, cls_l, box_l = detection_loss(cls_pred, box_pred, bboxes, class_ids, anc)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss += float(loss)
            steps += 1

            if steps % 50 == 0:
                log.info(f"Epoch {epoch} | Step {steps:4d} | Total Loss: {loss:.4f} | "
                         f"Cls: {float(cls_l):.4f} | Box: {float(box_l):.4f}")

        avg_loss = epoch_loss / steps
        log.info(f"Epoch {epoch}/{args.epochs} completed | Avg Loss: {avg_loss:.5f}")

        model.save_weights(str(CKPT_DIR / "last.weights.h5"))

    log.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weights", type=str, default="imagenet", choices=["imagenet", "none"])
    parser.add_argument("--subsample", type=float, default=0.3)
    args = parser.parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    train(args)