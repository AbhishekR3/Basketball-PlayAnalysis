# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Pipeline (run order)

The project is a linear ML pipeline; each stage consumes the previous stage's output. Edits to any stage typically need to be considered against the data contract with the next stage:

1. **Passing_Simulation.py** or **RandomMovement_Simulation.py** — Pygame-based generators that render synthetic basketball plays to MP4 in `$VIDEO_DIR`. `Passing_Simulation` produces labeled passing plays (positive class); `RandomMovement_Simulation` produces unstructured movement (negative class). Both use `secrets.SystemRandom()` for randomness and a `cryptographic_normal` helper.
2. **Object_Tracking.py** — Runs the custom YOLOv10s model (`assets/YOLOv10s_custom.pt`, classes: `Basketball`, `Team_A`, `Team_B`) over the simulation video, then feeds detections through the vendored `deep_sort/` Tracker. Writes per-frame tracking CSVs to `$TRACKING_DIR`. Confidence thresholds differ by class (basketball 0.5, players 0.8) — see `filter_lowconfidence`.
3. **Feature_Engineering.py** — Reads the raw tracking CSV and emits engineered features: one-hot ClassID, parsed Kalman `Mean` vector (`pos_x/y`, `aspect_ratio`, `height`, `vel_*`), rolling averages, accelerations, temporal encoding, and PCA. Output CSVs feed both the neural network and the database loader.
4. **Neural_Network.py** — `BasketballPlayDataset` expects a `root_dir` with `train/test/validation` splits, each containing `pass/` and `not-pass/` subdirectories of feature CSVs. Defines a bidirectional LSTM with self-attention, time-warp/jitter augmentation, and structured magnitude-based attention-head pruning. Saves `basketball_lstm_model.pt` and `pruned_model.pt` under `$MODEL_DIR`.
5. **Data_Loading.py** — Loads engineered features into PostgreSQL/PostGIS via SQLAlchemy + GeoAlchemy2. The schema (`TrackingData`) is defined at the top of the file; if you add or rename feature columns in `Feature_Engineering.py`, mirror the change here. The connection string is currently hardcoded to `postgresql://postgres:password@localhost:5432/postgres` (line ~89) — replace before running against a real DB.

`utils.py` holds the only shared helpers: `export_dataframe_to_csv`, `read_dataframe_to_csv`, and `configure_logger`. Every stage script calls `configure_logger(<stage_name>)`. Note `configure_logger` references a module-level `log_dir` that must be set in the caller before invocation (each script sets it from env vars near its `__main__`).

## Environment variables

All stages read paths from env vars with `/app/...` defaults (Docker-oriented). When running locally, either `export` them or accept the defaults and run inside the container. Used variables:

- `ASSETS_DIR` — YOLO weights, court diagram (input)
- `VIDEO_DIR` — simulation MP4s (passed between stages 1 → 2)
- `TRACKING_DIR` — tracking CSVs (passed between stages 2 → 3 → 5)
- `MODEL_DIR` — saved model checkpoints
- `OUTPUT_DIR` — root for any other outputs
- `LOG_DIR` — per-script logs (also where `run_sequence.sh` writes stdout/stderr)
- `DeepSORT_DIR` — only Object_Tracking.py reads this
- `ANTHROPIC_API_KEY` — only `implementation_agent.py` reads this

## Common commands

```bash
# Local run (manual sequence, after pip install -r Requirements.txt)
python Passing_Simulation.py           # or RandomMovement_Simulation.py
python Object_Tracking.py
python Feature_Engineering.py
python Neural_Network.py
python Data_Loading.py                 # requires running Postgres/PostGIS

# Containerized run (preferred; chains stages 1→3 via run_sequence.sh)
docker build -t basketball-analysis .
docker run -v basketball_data:/app/data basketball-analysis

# AI implementation agent (Anthropic SDK; edits codebase from a plain-English idea)
ANTHROPIC_API_KEY=sk-... python implementation_agent.py "Add acceleration features to Feature_Engineering.py"
```

The repo has a pytest suite under `tests/` (config in `pytest.ini`, `testpaths = tests`). Two markers split it: `quick` (fast unit tests, no models / no heavy I/O) and `extensive` (full-pipeline integration against the real YOLO model, ~3-5 min). Behavioral tests for the simulations live in `tests/test_simulations.py`. Several `quick` tests — notably `tests/quick/test_simulation_math.py` — AST-extract the sim `Player`/`Basketball` classes via `tests/conftest.load_callable_from_source` and run them in an isolated namespace; when you add a **module-level constant** that a sim class references (e.g. `WALL_MARGIN`), you must also inject it into that test's `class_extra` namespace or the extracted method raises `NameError`. The `extensive` pipeline/tracking tests need `assets/YOLOv10s_custom.pt` and a real simulation video present; without those assets they fail independently of any code change.

The CI workflow (`.github/workflows/ci.yml`) does **not** run pytest — it only smoke-tests Docker by building a **simplified** Dockerfile inline (not the real `dockerfile`) and grepping for "All scripts completed successfully" in a stubbed log. It does not exercise pipeline code, so run pytest locally; do not rely on CI to catch real regressions.

## Conventions worth following

- **Requirements file capitalization.** The file is `Requirements.txt` (capital R) but `dockerfile` and the README reference lowercase `requirements.txt`. CI copies whichever exists into `requirements.txt`. When editing, keep both names in mind.
- **Module style.** Files use `#%%` cell markers (Spyder/Jupyter-compatible) and large per-section banners. Each function has a docstring with `Objective:` / `Parameters:` / `Returns:` blocks using `[type] name - description`. Match this when adding functions.
- **Logging.** Every function wraps its body in `try/except` and logs via the script's logger before re-raising. Keep this pattern in new code rather than letting exceptions surface naked.
- **Randomness.** Simulations use `secrets.SystemRandom()` (not `random`) — Codacy flags `random` as insecure here. The Neural Network sets `SEED = 42` for `torch` and `numpy`. Don't mix these.
- **deep_sort/** is a vendored fork, imported as `from deep_sort.deep_sort import ...` and `from deep_sort.tools import generate_detections`. Don't replace with the pip package; the Tracker's `Mean` representation is what `Feature_Engineering.py` parses.
- **Schema drift.** `Data_Loading.TrackingData` enumerates every feature column. Adding a column upstream in `Feature_Engineering.py` requires adding it here too, or the DB insert will fail.

## Branching

Active development branches in this repo include `DEV_Code` and feature branches under `claude/...`. CI runs on `main` and `DEV_Code` only. The `assets/ObjectTracking%20Demo.gif` link in the README points at `DEV_Code`.

## License

CC BY-NC 4.0 — non-commercial. The custom YOLOv10s model and Roboflow dataset (see `References/Custom_DetectionModel.txt`) carry the same license.
