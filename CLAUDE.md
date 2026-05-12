# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end basketball play classification system: simulated video → YOLO+DeepSORT tracking → feature engineering → LSTM classifier. The goal is to distinguish passing plays from random movement using temporal sequences of player/ball tracking data.

## Pipeline Execution Order

Each stage depends on the previous stage's output. Run in sequence:

```bash
# 1. Generate labeled training video
python Passing_Simulation.py          # positive class (organized passing)
python RandomMovement_Simulation.py   # negative class (random movement)

# 2. Detect and track objects frame-by-frame
python Object_Tracking.py

# 3. Extract and normalize features from tracking data
python Feature_Engineering.py

# 4. Train LSTM classifier
python Neural_Network.py

# Optional: persist tracking data to PostgreSQL
python Data_Loading.py
```

Or run all steps via the orchestration script:
```bash
./run_sequence.sh
```

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (no test files exist yet, but pytest is configured)
pytest

# Docker build and run (ARM64 optimized)
docker build -t basketball-analysis .
docker run -v basketball_data:/app/data basketball-analysis

# Lint/format (per project-level preference)
ruff format <file>.py
ruff check --fix <file>.py
```

## Architecture

### Data Flow

```
Simulation (pygame) → MP4/AVI video
  → Object_Tracking.py: YOLO detection + DeepSORT → tracking_data.csv
  → Feature_Engineering.py: encoding, PCA, normalization → features_engineered.csv
  → Neural_Network.py: LSTM + attention → models/best_model.pt
  → Data_Loading.py: SQLAlchemy ORM → PostgreSQL (optional)
```

### Key Files

| File | Role |
|------|------|
| `Passing_Simulation.py` | Positive-class video generation with realistic ball physics (pygame) |
| `RandomMovement_Simulation.py` | Negative-class video generation (independent random movement) |
| `Object_Tracking.py` | YOLO (`assets/YOLOv10s_custom.pt`) + DeepSORT multi-object tracking |
| `Feature_Engineering.py` | One-hot encoding, DeepSORT state extraction, PCA, MinMax/Standard scaling |
| `Neural_Network.py` | Bi-directional LSTM with self-attention, pruning, augmentation |
| `Data_Loading.py` | SQLAlchemy + GeoAlchemy2 ORM for PostgreSQL persistence |
| `utils.py` | Shared CSV I/O and logger setup |
| `deep_sort/` | DeepSORT implementation (Kalman filter, Hungarian assignment, nn matching) |

### Tracking Data Schema

The DeepSORT state vector has 8 dimensions: `[pos_x, pos_y, aspect_ratio, height, vel_x, vel_y, vel_aspect, vel_height]`. Each tracked object per frame also stores: `TrackID`, `ClassID` (Basketball / Team_A / Team_B), 64-element covariance, 128-element appearance feature vector, confidence score, state (tentative/confirmed), hits, and age.

### Model

- **Detection:** Custom YOLOv10s trained on 2,870 images — 99.5% mAP50, 99.6% precision/recall. Confidence thresholds: Basketball ≥ 0.5, Players ≥ 0.6–0.8.
- **Classifier:** Bi-directional LSTM with self-attention. Supports variable-length sequences via custom collate. Includes magnitude-based structured pruning of attention heads post-training.
- **Trained model:** `models/best_model.pt`

### Database

`Data_Loading.py` uses SQLAlchemy ORM with GeoAlchemy2 for PostGIS spatial indexing. The `TrackingData` ORM model maps 30+ feature columns. Requires a running PostgreSQL instance with PostGIS. Connection is configured via `create_sqlalchemy_engine()`.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on push to `main` and `DEV_Code`. It builds the Docker image, executes `run_sequence.sh`, and validates that all steps log "succeeded".
