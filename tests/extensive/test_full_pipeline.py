'''
Full pipeline integration test: stages 1 -> 2 -> 3 chained.

Runs RandomMovement_Simulation -> Object_Tracking -> Feature_Engineering as
subprocesses with all env vars rooted at tmp_pipeline_dirs. Asserts each
stage's output file exists and feeds the next. Skips if YOLO weights or
DeepSORT encoder aren't available locally.
'''

import os
import subprocess
import sys

import pandas as pd
import pytest


pytestmark = pytest.mark.extensive


def _common_env(repo_root, tmp_pipeline_dirs):
    env = os.environ.copy()
    env.update({
        'SDL_VIDEODRIVER': 'dummy',
        'GITHUB_ACTIONS': 'true',
        'LOG_DIR': str(tmp_pipeline_dirs['LOG_DIR']),
        'OUTPUT_DIR': str(tmp_pipeline_dirs['OUTPUT_DIR']),
        'MODEL_DIR': str(tmp_pipeline_dirs['MODEL_DIR']),
        'TRACKING_DIR': str(tmp_pipeline_dirs['TRACKING_DIR']),
        'VIDEO_DIR': str(tmp_pipeline_dirs['VIDEO_DIR']),
        'ASSETS_DIR': str(repo_root / 'assets'),
        'DeepSORT_DIR': str(repo_root / 'deep_sort'),
        'PYTHONPATH': str(repo_root),
    })
    return env


def _required_assets(repo_root):
    weights = repo_root / 'assets' / 'YOLOv10s_custom.pt'
    encoder_candidates = [
        repo_root / 'deep_sort' / 'model_data' / 'mars-small128.pb',
        repo_root / 'deep_sort' / 'tools' / 'model_data' / 'mars-small128.pb',
    ]
    return weights, encoder_candidates


def test_pipeline_stages_1_to_3_produce_chained_outputs(tmp_pipeline_dirs, repo_root):
    weights, encoder_candidates = _required_assets(repo_root)
    if not weights.exists():
        pytest.skip('YOLOv10s_custom.pt not available (Git LFS not fetched)')
    if not any(e.exists() for e in encoder_candidates):
        pytest.skip('DeepSORT mars-small128.pb encoder weights not available')

    env = _common_env(repo_root, tmp_pipeline_dirs)
    common = dict(cwd=str(repo_root), env=env, capture_output=True, text=True)

    # Stage 1: Simulation
    sim = subprocess.run([sys.executable, 'RandomMovement_Simulation.py'],
                         timeout=120, **common)
    assert sim.returncode == 0, f'Simulation failed:\n{sim.stderr[-2000:]}'
    mp4 = tmp_pipeline_dirs['VIDEO_DIR'] / 'random_movement_video.mp4'
    assert mp4.exists() and mp4.stat().st_size > 0

    # Stage 2: Object Tracking
    track = subprocess.run([sys.executable, 'Object_Tracking.py'],
                          timeout=300, **common)
    assert track.returncode == 0, f'Tracking failed:\n{track.stderr[-2000:]}'
    csv = tmp_pipeline_dirs['TRACKING_DIR'] / 'detected_objects.csv'
    assert csv.exists()
    df = pd.read_csv(csv)
    assert len(df) > 0

    # Stage 3: Feature Engineering
    feat = subprocess.run([sys.executable, 'Feature_Engineering.py'],
                         timeout=300, **common)
    assert feat.returncode == 0, f'Feature engineering failed:\n{feat.stderr[-2000:]}'

    # Feature_Engineering exports a timestamped CSV into TRACKING_DIR
    outputs = [
        p for p in tmp_pipeline_dirs['TRACKING_DIR'].iterdir()
        if p.suffix == '.csv' and p.name != 'detected_objects.csv'
    ]
    assert len(outputs) >= 1, f'no engineered-feature CSV produced in {tmp_pipeline_dirs["TRACKING_DIR"]}'
    final_df = pd.read_csv(outputs[0])
    assert len(final_df) > 0
    assert 'is_Basketball' in final_df.columns
