#!/usr/bin/env python3
"""
train_detector.py — Train YOLOv8 crater detector.
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("crater-det")

CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)


def train(args):
    from ultralytics import YOLO

    log.info(f"Initialising YOLOv8 model: {args.model}")
    model = YOLO(args.model)

    log.info(f"Training on: {args.data}")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        project=str(CKPT_DIR),
        name="train",
        exist_ok=True,
        verbose=True,
    )

    # Copy best weights to canonical path expected by other scripts
    best_src = CKPT_DIR / "train" / "weights" / "best.pt"
    best_dst = CKPT_DIR / "best.pt"
    if best_src.exists():
        import shutil
        shutil.copy(best_src, best_dst)
        log.info(f"Best weights saved → {best_dst}")
    else:
        log.warning(f"best.pt not found at {best_src}")

    log.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default="yolov8n.pt",
                        help="YOLOv8 model variant: yolov8n.pt or yolov8s.pt")
    parser.add_argument("--data",   type=str, default="LU3M6TGT_yolo_format/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz",  type=int, default=416)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--lr",     type=float, default=1e-3)
    args = parser.parse_args()

    train(args)
