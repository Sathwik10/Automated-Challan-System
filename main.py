"""
app/main.py
────────────
Main entry-point for the Automated Challan System.

Usage:
    python app/main.py                    # live webcam
    python app/main.py --source video.mp4 # video file
    python app/main.py --headless         # no display (server mode)
"""

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import yaml

# Ensure project root is on sys.path (works when main.py lives in project root or app/)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))

from inference         import load_models, process_frame, NaiveTracker
from violation_detection import ViolationDetector
from utils.helpers     import (
    ChallanDB, draw_stop_line, save_snapshot,
    init_csv_db, append_challan,
)

try:
    from sort_tracker import Sort
    SORT_AVAILABLE = True
except ImportError:
    SORT_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ChallanSystem")


# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────

def load_config(path: str = None) -> dict:
    default = ROOT / "config" / "config.yaml"
    cfg_path = path or str(default)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
#  Fine amount helper
# ─────────────────────────────────────────────────────────────

FINE_MAP = {
    "overspeed":  "fine_overspeed",
    "red_light":  "fine_red_light",
    "line_cross": "fine_line_cross",
}


def get_fine(cfg: dict, violation_type: str) -> int:
    key = FINE_MAP.get(violation_type, "fine_line_cross")
    return cfg["challan"].get(key, 1000)


# ─────────────────────────────────────────────────────────────
#  Challan issuer
# ─────────────────────────────────────────────────────────────

def issue_challan(event, cfg: dict, db: ChallanDB,
                  snapshot_dir: str) -> dict:
    """
    Persist a ViolationEvent as a challan record and log it.
    Returns the challan dict.
    """
    challan_id = str(uuid.uuid4())[:12].upper()
    fine       = get_fine(cfg, event.violation_type)

    snapshot_path = ""
    if event.frame_snapshot is not None:
        snapshot_path = save_snapshot(
            event.frame_snapshot,
            snapshot_dir,
            prefix=event.violation_type,
        )

    record = {
        "challan_id":     challan_id,
        "timestamp":      event.timestamp,
        "plate_number":   event.plate_number,
        "vehicle_class":  event.vehicle_class,
        "violation_type": event.violation_type,
        "speed_kmph":     event.speed_kmph,
        "fine_inr":       fine,
        "snapshot_path":  snapshot_path,
    }

    db.insert(record)

    # Also write to CSV for easy export
    csv_path = str(ROOT / cfg["paths"]["challan_log"])
    init_csv_db(csv_path)
    append_challan(csv_path, record)

    log.info(
        "🚨 CHALLAN ISSUED | ID: %s | Plate: %s | "
        "Violation: %s | Fine: ₹%d",
        challan_id, event.plate_number,
        event.violation_type, fine,
    )
    return record


# ─────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────

def run(cfg: dict, source=None, headless: bool = False) -> None:
    if source is not None:
        cfg["input"]["source"] = source

    # Ensure output directories exist
    snapshot_dir = str(ROOT / "data" / "snapshots")
    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    Path(ROOT / "data").mkdir(parents=True, exist_ok=True)

    # Initialise components
    models            = load_models(cfg)
    tracker           = Sort() if SORT_AVAILABLE else NaiveTracker()
    violation_det     = ViolationDetector(cfg)
    db                = ChallanDB(str(ROOT / "data" / "challans.db"))
    issued_plates     = set()   # simple dedup: one challan per plate/type

    cap = cv2.VideoCapture(cfg["input"]["source"])
    if not cap.isOpened():
        log.error("Cannot open source: %s", cfg["input"]["source"])
        return

    stop_line_y = cfg["violation"]["line_crossing"]["stop_line_y"]
    frame_no    = 0

    log.info("Challan system running. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.info("End of stream.")
                break

            # Run detection + tracking
            annotated, tracks = process_frame(
                frame, models, tracker, cfg
            )

            # Draw stop line
            annotated = draw_stop_line(annotated, stop_line_y)

            # Check for violations
            violations = violation_det.check(frame, tracks, frame_no)
            for event in violations:
                dedup_key = f"{event.plate_number}:{event.violation_type}"
                if dedup_key not in issued_plates:
                    issue_challan(event, cfg, db, snapshot_dir)
                    issued_plates.add(dedup_key)

            # Overlay stats
            cv2.putText(
                annotated,
                f"Frame: {frame_no}  Tracks: {len(tracks)}  "
                f"Challans: {len(issued_plates)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2,
            )

            if not headless:
                cv2.imshow("Automated Challan System", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_no += 1

    finally:
        cap.release()
        db.close()
        if not headless:
            cv2.destroyAllWindows()

    log.info("Session complete. Total challans issued: %d",
             len(issued_plates))


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Automated Traffic Challan System"
    )
    p.add_argument("--source",   default=None,
                   help="Video file path or camera index (default: config)")
    p.add_argument("--config",   default=None,
                   help="Path to config.yaml")
    p.add_argument("--headless", action="store_true",
                   help="Run without displaying video window")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    run(cfg, source=args.source, headless=args.headless)