# 🚦 Automated Traffic Challan System

An end-to-end deep learning pipeline that detects traffic violations in real time from camera feeds or video files, reads license plates, and automatically issues digital challans (fines).

---

## Architecture

```
Camera / Video
      │
      ▼
 Frame Capture  ──────────────────────────────────────────────────────┐
      │                                                                │
      ▼                                                                │
Vehicle Detector  (MobileNetV2 CNN)                                    │
  → class + bbox                                                       │
      │                                                                │
      ▼                                                                │
SORT Multi-Object Tracker  → track_id + trajectory                    │
      │                                                                │
      ├──────────────────────────────────────────┐                    │
      ▼                                           ▼                    │
Plate Detector (CNN)               Violation Detector                  │
  → plate bbox                       • Over-speeding                   │
      │                               • Red-light jump                 │
      ▼                               • Stop-line cross                │
OCR (CNN chars + Tesseract fallback)          │                        │
  → plate text                                ▼                        │
      │                              Challan Issuer                    │
      └──────────────────────────────────────┘                         │
                                     │                                 │
                                     ▼                                 │
                              SQLite / CSV DB ◄─────────────────────────┘
                              + Snapshot saved
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# Tesseract OCR (optional but recommended)
sudo apt install tesseract-ocr   # Ubuntu/Debian
brew install tesseract            # macOS
```

### 2. Prepare data

Place raw images in `data/raw/` with a `labels.json` mapping filenames to class labels.  
Then run the pre-processing script:

```bash
python src/data_preprocessing.py
```

This will create `data/datasets/train/`, `val/`, and `test/` with `labels.csv` manifests.

### 3. Train models

```bash
# Train all models sequentially
python src/train.py --model all

# Or individually
python src/train.py --model vehicle
python src/train.py --model plate
python src/train.py --model ocr
```

Trained weights are saved to `models/`.

### 4. Run the system

```bash
# Live webcam (default)
python app/main.py

# From a video file
python app/main.py --source path/to/video.mp4

# Headless / server mode (no display)
python app/main.py --headless
```

Press `q` to quit the live display.

---

## Configuration

All hyperparameters, paths, and thresholds live in `config/config.yaml`.

| Key section | Purpose |
|---|---|
| `paths` | Where to find / save models and data |
| `input` | Camera index or video path, resolution, FPS |
| `preprocessing` | Image resize, augmentation settings |
| `vehicle_detection` | Backbone, confidence threshold, num classes |
| `plate_detection` | Input size, confidence |
| `plate_recognition` | OCR charset, Tesseract fallback toggle |
| `training` | Epochs, batch size, learning rate, callbacks |
| `violation` | Speed limit, pixels-per-meter calibration, ROIs |
| `challan` | Fine amounts per violation type |

---

## Project Structure

```
automated_challan_system/
├── app/
│   └── main.py                   ← entry point
├── config/
│   └── config.yaml
├── data/
│   ├── raw/                      ← your images / videos
│   ├── processed/
│   └── datasets/{train,val,test}/
├── models/                       ← saved .h5 weights
├── notebooks/
├── src/
│   ├── data_preprocessing.py
│   ├── inference.py
│   ├── train.py
│   ├── violation_detection.py
│   └── models/
│       ├── vehicle_detection.py
│       ├── plate_detection.py
│       └── plate_recognition.py
├── tests/
│   ├── test_helpers.py
│   └── test_violation_detection.py
├── utils/
│   ├── helpers.py
│   └── ocr.py
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Violation Types & Fines (configurable)

| Violation | Default Fine (₹) |
|---|---|
| Over-speeding | 2,000 |
| Red-light jump | 5,000 |
| Stop-line cross | 1,000 |

---

## Adding Your Own Data

1. Place images in `data/raw/`
2. Create `data/raw/labels.json`:
   ```json
   {
     "frame_000001.jpg": "car",
     "frame_000002.jpg": "truck"
   }
   ```
3. For plate detection, add bounding-box annotations to `labels.csv` (columns: `filename`, `x1`, `y1`, `x2`, `y2`)
4. Re-run `src/data_preprocessing.py` then `src/train.py`

---

## Camera Calibration

Speed estimation requires the `pixels_per_meter` constant in `config.yaml`.  
To calibrate: place a known-length object (e.g., 1 m rod) in the camera's field of view, measure its pixel length, and set `pixels_per_meter` to that value.

---

## License

MIT
