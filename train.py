"""
src/train.py
─────────────
Trains vehicle detector, plate detector and OCR model in sequence.
Usage:
    python src/train.py --model all           # train everything
    python src/train.py --model vehicle
    python src/train.py --model plate
    python src/train.py --model ocr
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import yaml
import cv2
from keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

from models.vehicle_detection import (
    build_vehicle_detector, compile_vehicle_model
)
from models.plate_detection import (
    build_plate_detector, compile_plate_model
)
from models.plate_recognition import (
    build_char_classifier, compile_ocr_model
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
#  Generic data loader (images → numpy arrays)
# ─────────────────────────────────────────────────────────────

def load_split(split_dir: str,
               target_size: tuple,
               grayscale: bool = False) -> tuple:
    """
    Load images from *split_dir*/images/ and corresponding
    labels.csv.  Returns (X, y) numpy arrays.
    """
    csv_path = os.path.join(split_dir, "labels.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Labels not found: {csv_path}")

    df = pd.read_csv(csv_path)
    images, labels = [], []

    for _, row in df.iterrows():
        img_path = os.path.join(split_dir, "images", row["filename"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[:, :, np.newaxis]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_size[1], target_size[0]))
        images.append(img.astype(np.float32) / 255.0)
        labels.append(row["label"])

    return np.array(images), np.array(labels)


# ─────────────────────────────────────────────────────────────
#  Standard callbacks
# ─────────────────────────────────────────────────────────────

def get_callbacks(model_save_path: str,
                  patience_es: int = 7,
                  patience_lr: int = 3,
                  log_dir: str = "logs") -> list:
    return [
        ModelCheckpoint(model_save_path,
                        save_best_only=True,
                        monitor="val_loss",
                        verbose=1),
        EarlyStopping(patience=patience_es,
                      restore_best_weights=True,
                      verbose=1),
        ReduceLROnPlateau(factor=0.5,
                          patience=patience_lr,
                          verbose=1),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]


# ─────────────────────────────────────────────────────────────
#  Training routines
# ─────────────────────────────────────────────────────────────

def train_vehicle_model(cfg: dict) -> None:
    log.info("── Training Vehicle Detector ──")
    tc     = cfg["training"]
    vc     = cfg["vehicle_detection"]
    paths  = cfg["paths"]

    X_train, y_train = load_split(paths["train"],
                                  vc["input_shape"][:2])
    X_val,   y_val   = load_split(paths["val"],
                                  vc["input_shape"][:2])

    model = build_vehicle_detector(
        input_shape=tuple(vc["input_shape"]),
        num_classes=vc["num_classes"],
        pretrained=vc["pretrained"],
    )
    model = compile_vehicle_model(model, tc["learning_rate"])
    model.summary(print_fn=log.info)

    # Dummy bboxes when not available in labels
    dummy_bbox_train = np.zeros((len(X_train), 4), dtype=np.float32)
    dummy_bbox_val   = np.zeros((len(X_val),   4), dtype=np.float32)

    model.fit(
        X_train,
        {"class_output": y_train, "bbox_output": dummy_bbox_train},
        validation_data=(
            X_val,
            {"class_output": y_val, "bbox_output": dummy_bbox_val},
        ),
        epochs=tc["epochs"],
        batch_size=tc["batch_size"],
        callbacks=get_callbacks(paths["vehicle_model"]),
    )
    log.info("Vehicle model saved → %s", paths["vehicle_model"])


def train_plate_model(cfg: dict) -> None:
    log.info("── Training Plate Detector ──")
    tc    = cfg["training"]
    pc    = cfg["plate_detection"]
    paths = cfg["paths"]

    X_train, _ = load_split(paths["train"], pc["input_shape"][:2])
    X_val,   _ = load_split(paths["val"],   pc["input_shape"][:2])

    # Dummy ground-truth bboxes (replace with real annotations)
    y_train = np.random.rand(len(X_train), 4).astype(np.float32)
    y_val   = np.random.rand(len(X_val),   4).astype(np.float32)

    model = build_plate_detector(tuple(pc["input_shape"]))
    model = compile_plate_model(model, tc["learning_rate"])

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=tc["epochs"],
        batch_size=tc["batch_size"],
        callbacks=get_callbacks(paths["plate_model"]),
    )
    log.info("Plate model saved → %s", paths["plate_model"])


def train_ocr_model(cfg: dict) -> None:
    log.info("── Training OCR Model ──")
    tc    = cfg["training"]
    oc    = cfg["plate_recognition"]
    paths = cfg["paths"]

    input_shape = tuple(oc["input_shape"])  # (H, W, C)
    X_train, y_train = load_split(paths["train"],
                                  input_shape[:2],
                                  grayscale=True)
    X_val,   y_val   = load_split(paths["val"],
                                  input_shape[:2],
                                  grayscale=True)

    model = build_char_classifier(input_shape, oc["num_classes"])
    model = compile_ocr_model(model, tc["learning_rate"])

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=tc["epochs"],
        batch_size=tc["batch_size"],
        callbacks=get_callbacks(paths["ocr_model"]),
    )
    log.info("OCR model saved → %s", paths["ocr_model"])


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train challan system models")
    p.add_argument("--model", default="all",
                   choices=["all", "vehicle", "plate", "ocr"],
                   help="Which model to train")
    p.add_argument("--config", default="config/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)

    if args.model in ("all", "vehicle"):
        train_vehicle_model(cfg)
    if args.model in ("all", "plate"):
        train_plate_model(cfg)
    if args.model in ("all", "ocr"):
        train_ocr_model(cfg)


if __name__ == "__main__":
    main()
