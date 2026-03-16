"""
src/inference.py
─────────────────
Real-time inference pipeline.
  1. Reads frames from camera / video file
  2. Detects vehicles → crops → detects plates → reads text
  3. Tracks objects with SORT
  4. Returns annotated frame + list of active tracks
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import yaml
import tensorflow as tf

from models.vehicle_detection  import detect_vehicles
from models.plate_detection    import detect_plate
from models.plate_recognition  import (
    recognise_plate_cnn,
    recognise_plate_tesseract,
)
from utils.helpers  import centroid_of_bbox, draw_bbox
from utils.ocr      import segment_characters

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  SORT wrapper (lightweight multi-object tracker)
# ─────────────────────────────────────────────────────────────

try:
    from sort_tracker import Sort  # pip install sort-tracker-py
    _SORT_AVAILABLE = True
except ImportError:
    _SORT_AVAILABLE = False
    log.warning("SORT not found – using naive centroid tracker fallback.")


class NaiveTracker:
    """
    Minimal centroid tracker for when SORT is unavailable.
    Not robust; replace with SORT for production use.
    """

    def __init__(self):
        self._next_id  = 0
        self._objects  = {}     # id → centroid

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        detections : (N, 5) array [x1, y1, x2, y2, score]
        Returns    : (N, 5) array [x1, y1, x2, y2, track_id]
        """
        results = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            # Assign new ID (naive: no real matching)
            tid = self._next_id
            self._next_id += 1
            self._objects[tid] = (cx, cy)
            results.append([x1, y1, x2, y2, tid])
        return np.array(results) if results else np.empty((0, 5))


# ─────────────────────────────────────────────────────────────
#  Model loader
# ─────────────────────────────────────────────────────────────

def load_models(cfg: dict) -> dict:
    paths = cfg["paths"]
    models = {}

    for key, path in [
        ("vehicle",  paths["vehicle_model"]),
        ("plate",    paths["plate_model"]),
        ("ocr",      paths["ocr_model"]),
    ]:
        if Path(path).exists():
            models[key] = tf.keras.models.load_model(path)
            log.info("Loaded %s model from %s", key, path)
        else:
            models[key] = None
            log.warning("%s model not found at %s", key, path)

    return models


# ─────────────────────────────────────────────────────────────
#  Per-frame pipeline
# ─────────────────────────────────────────────────────────────

def process_frame(frame_bgr: np.ndarray,
                  models: dict,
                  tracker,
                  cfg: dict) -> tuple:
    """
    Run the full detection + OCR + tracking pipeline on one frame.

    Returns
    -------
    annotated_frame : BGR ndarray with drawn bboxes
    tracks          : list of track dicts
    """
    vc_cfg = cfg["vehicle_detection"]
    pr_cfg = cfg["plate_recognition"]
    h, w   = frame_bgr.shape[:2]

    # ── Normalise frame ─────────────────────────────────────────
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    target    = tuple(vc_cfg["input_shape"][:2])
    frame_resized = cv2.resize(frame_rgb, (target[1], target[0]))
    frame_norm    = frame_resized.astype(np.float32) / 255.0

    # ── 1. Vehicle detection ────────────────────────────────────
    detections = []
    if models["vehicle"] is not None:
        results = detect_vehicles(
            models["vehicle"],
            frame_norm,
            conf_thresh=vc_cfg["confidence_threshold"],
        )
        for r in results:
            x1, y1, x2, y2 = [int(v * d) for v, d in
                               zip(r["bbox"], [w, h, w, h])]
            detections.append([x1, y1, x2, y2, r["confidence"]])

    det_array = np.array(detections) if detections else np.empty((0, 5))

    # ── 2. Tracking ─────────────────────────────────────────────
    tracked = tracker.update(det_array)   # (N, 5): x1,y1,x2,y2,id

    # ── 3. Plate detection + OCR ─────────────────────────────────
    tracks = []
    for row in tracked:
        x1, y1, x2, y2, tid = int(row[0]), int(row[1]), \
                               int(row[2]), int(row[3]), int(row[4])
        vehicle_crop = frame_bgr[max(0, y1):y2, max(0, x1):x2]
        plate_number = "UNKNOWN"

        if vehicle_crop.size > 0 and models["plate"] is not None:
            crop_resized = cv2.resize(vehicle_crop, (320, 320))
            crop_norm    = crop_resized.astype(np.float32) / 255.0
            plate_result = detect_plate(models["plate"], crop_norm)

            if plate_result:
                plate_crop = plate_result["plate_crop"]

                # Try CNN OCR first
                if models["ocr"] is not None:
                    char_imgs = segment_characters(plate_crop)
                    if char_imgs:
                        plate_number = recognise_plate_cnn(
                            models["ocr"], char_imgs
                        )
                # Tesseract fallback
                if (plate_number == "UNKNOWN"
                        and pr_cfg["use_tesseract_fallback"]):
                    plate_number = recognise_plate_tesseract(plate_crop) \
                        or "UNKNOWN"

        centroid = centroid_of_bbox([x1, y1, x2, y2])
        tracks.append({
            "track_id":     tid,
            "bbox":         [x1, y1, x2, y2],
            "centroid":     centroid,
            "plate_number": plate_number,
            "vehicle_class": "vehicle",
            "confidence":   1.0,
        })

        # Draw on frame
        frame_bgr = draw_bbox(
            frame_bgr, [x1, y1, x2, y2],
            label=f"ID:{tid} {plate_number}",
            color=(0, 200, 0),
        )

    return frame_bgr, tracks


# ─────────────────────────────────────────────────────────────
#  Video capture loop (standalone use)
# ─────────────────────────────────────────────────────────────

def run_inference(cfg: dict,
                  on_frame_callback=None) -> None:
    """
    Capture frames from the configured source and run the pipeline.
    *on_frame_callback(annotated_frame, tracks, frame_no)* is called
    for each processed frame if provided.
    """
    models  = load_models(cfg)
    tracker = Sort() if _SORT_AVAILABLE else NaiveTracker()

    source = cfg["input"]["source"]
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    frame_no = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, tracks = process_frame(frame, models,
                                              tracker, cfg)
            if on_frame_callback:
                on_frame_callback(annotated, tracks, frame_no)

            cv2.imshow("Challan System", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_no += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
