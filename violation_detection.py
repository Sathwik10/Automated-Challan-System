"""
src/violation_detection.py
───────────────────────────
Rules for detecting traffic violations:
  • Over-speeding    – pixel displacement between frames → km/h
  • Red-light jump   – vehicle detected inside signal ROI when light is red
  • Stop-line cross  – vehicle bbox crosses a predefined stop-line
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────

@dataclass
class ViolationEvent:
    violation_type: str          # "overspeed" | "red_light" | "line_cross"
    plate_number:   str
    vehicle_class:  str
    confidence:     float
    speed_kmph:     Optional[float] = None
    frame_snapshot: Optional[np.ndarray] = None
    timestamp:      float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "violation_type": self.violation_type,
            "plate_number":   self.plate_number,
            "vehicle_class":  self.vehicle_class,
            "confidence":     round(self.confidence, 3),
            "speed_kmph":     round(self.speed_kmph, 1) if self.speed_kmph else None,
            "timestamp":      self.timestamp,
        }


# ─────────────────────────────────────────────────────────────
#  Speed estimator
# ─────────────────────────────────────────────────────────────

class SpeedEstimator:
    """
    Estimates vehicle speed from centroid displacement over time.

    Parameters
    ----------
    pixels_per_meter : calibration constant (pixels ≡ 1 m in scene)
    fps              : camera frames-per-second
    history_frames   : number of frames kept in the history buffer
    """

    def __init__(self,
                 pixels_per_meter: float = 10.0,
                 fps: float = 30.0,
                 history_frames: int = 30):
        self.ppm            = pixels_per_meter
        self.fps            = fps
        self.history_frames = history_frames
        # track_id → deque of (frame_no, centroid_px)
        self._history: dict[int, list] = {}

    def update(self, track_id: int,
               centroid: tuple,
               frame_no: int) -> Optional[float]:
        """
        Update tracker history and return estimated speed (km/h)
        if enough history is available, otherwise None.
        """
        buf = self._history.setdefault(track_id, [])
        buf.append((frame_no, centroid))

        # Keep only last N frames
        if len(buf) > self.history_frames:
            buf.pop(0)

        if len(buf) < 2:
            return None

        (f0, c0), (f1, c1) = buf[0], buf[-1]
        dt_frames = f1 - f0
        if dt_frames == 0:
            return None

        dist_px = np.linalg.norm(np.array(c1) - np.array(c0))
        dist_m  = dist_px / self.ppm
        dt_s    = dt_frames / self.fps
        speed_ms  = dist_m / dt_s
        speed_kmh = speed_ms * 3.6
        return speed_kmh

    def remove(self, track_id: int) -> None:
        self._history.pop(track_id, None)


# ─────────────────────────────────────────────────────────────
#  Red-light detector
# ─────────────────────────────────────────────────────────────

class RedLightDetector:
    """
    Detects red-signal state via colour analysis of a fixed ROI.

    roi : (x1, y1, x2, y2) in pixel coordinates
    """

    def __init__(self, roi: tuple):
        self.roi = roi

    def is_red(self, frame_bgr: np.ndarray) -> bool:
        x1, y1, x2, y2 = self.roi
        region = frame_bgr[y1:y2, x1:x2]
        hsv    = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Red wraps around hue in HSV
        mask1 = cv2.inRange(hsv, (0,  120, 70), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        red_pixels = cv2.countNonZero(mask1 | mask2)
        total      = region.size // 3          # total pixels

        return (red_pixels / total) > 0.20     # >20 % red → signal is red


# ─────────────────────────────────────────────────────────────
#  Main violation checker
# ─────────────────────────────────────────────────────────────

class ViolationDetector:
    """
    Orchestrates speed, red-light, and stop-line checks.

    Parameters
    ----------
    cfg : the full config dict loaded from config.yaml
    """

    def __init__(self, cfg: dict):
        vc = cfg["violation"]
        tk = cfg["tracker"]

        self.speed_limit   = vc["speed_limit_kmph"]
        self.stop_line_y   = vc["line_crossing"]["stop_line_y"]

        self.speed_est = SpeedEstimator(
            pixels_per_meter=vc["pixels_per_meter"],
            fps=cfg["input"]["fps"],
            history_frames=vc["track_history_frames"],
        )

        roi = vc["red_light_roi"]
        self.red_light_det = RedLightDetector(
            (roi["x1"], roi["y1"], roi["x2"], roi["y2"])
        )

        # track_id → last y-centroid (for line-cross detection)
        self._prev_y: dict[int, float] = {}

    # ── Public API ──────────────────────────────────────────────

    def check(self,
              frame_bgr: np.ndarray,
              tracks: list,
              frame_no: int) -> list:
        """
        Evaluate all tracks for violations.

        Parameters
        ----------
        frame_bgr : current BGR frame
        tracks    : list of dicts from tracker with keys:
                    track_id, centroid, bbox, plate_number,
                    vehicle_class, confidence
        frame_no  : global frame counter

        Returns
        -------
        list of ViolationEvent
        """
        violations = []
        is_red = self.red_light_det.is_red(frame_bgr)

        for t in tracks:
            tid       = t["track_id"]
            centroid  = t["centroid"]
            bbox      = t["bbox"]           # [x1, y1, x2, y2] pixel coords
            plate     = t.get("plate_number", "UNKNOWN")
            v_class   = t.get("vehicle_class", "unknown")
            conf      = t.get("confidence", 1.0)

            # 1. Speed
            speed = self.speed_est.update(tid, centroid, frame_no)
            if speed is not None and speed > self.speed_limit:
                violations.append(ViolationEvent(
                    violation_type="overspeed",
                    plate_number=plate,
                    vehicle_class=v_class,
                    confidence=conf,
                    speed_kmph=speed,
                    frame_snapshot=frame_bgr.copy(),
                ))

            # 2. Red-light jump
            if is_red:
                cy = centroid[1]
                if cy > self.stop_line_y:
                    violations.append(ViolationEvent(
                        violation_type="red_light",
                        plate_number=plate,
                        vehicle_class=v_class,
                        confidence=conf,
                        frame_snapshot=frame_bgr.copy(),
                    ))

            # 3. Stop-line cross (regardless of signal)
            prev_y = self._prev_y.get(tid)
            if prev_y is not None:
                if prev_y < self.stop_line_y <= centroid[1]:
                    violations.append(ViolationEvent(
                        violation_type="line_cross",
                        plate_number=plate,
                        vehicle_class=v_class,
                        confidence=conf,
                        frame_snapshot=frame_bgr.copy(),
                    ))
            self._prev_y[tid] = centroid[1]

        return violations
