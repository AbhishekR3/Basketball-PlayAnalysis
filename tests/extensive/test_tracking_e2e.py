'''
End-to-end test for Object_Tracking.py.

Runs the full tracking script on a tiny MP4 produced by RandomMovement_Simulation.
Requires the YOLOv10s_custom.pt weights and the mars-small128.pb DeepSORT
encoder to exist in their expected paths.
'''

import os
import subprocess
import sys

import pandas as pd
import pytest


pytestmark = pytest.mark.extensive


def _run_simulation(repo_root, tmp_pipeline_dirs):
    env = os.environ.copy()
    env.update({
        'SDL_VIDEODRIVER': 'dummy',
        'GITHUB_ACTIONS': 'true',
        'LOG_DIR': str(tmp_pipeline_dirs['LOG_DIR']),
        'VIDEO_DIR': str(tmp_pipeline_dirs['VIDEO_DIR']),
        'ASSETS_DIR': str(repo_root / 'assets'),
        'PYTHONPATH': str(repo_root),
    })
    subprocess.run(
        [sys.executable, 'RandomMovement_Simulation.py'],
        cwd=str(repo_root), env=env, check=True, timeout=120,
        capture_output=True, text=True,
    )


def test_object_tracking_produces_detections_csv(tmp_pipeline_dirs, repo_root):
    weights = repo_root / 'assets' / 'YOLOv10s_custom.pt'
    if not weights.exists():
        pytest.skip('YOLOv10s_custom.pt not available (Git LFS not fetched)')

    encoder = repo_root / 'deep_sort' / 'tools' / 'model_data' / 'mars-small128.pb'
    if not encoder.exists():
        # also try the path Object_Tracking.py constructs from DeepSORT_DIR
        encoder = repo_root / 'deep_sort' / 'model_data' / 'mars-small128.pb'
        if not encoder.exists():
            pytest.skip('DeepSORT mars-small128.pb encoder weights not available')

    _run_simulation(repo_root, tmp_pipeline_dirs)

    env = os.environ.copy()
    env.update({
        'LOG_DIR': str(tmp_pipeline_dirs['LOG_DIR']),
        'VIDEO_DIR': str(tmp_pipeline_dirs['VIDEO_DIR']),
        'TRACKING_DIR': str(tmp_pipeline_dirs['TRACKING_DIR']),
        'ASSETS_DIR': str(repo_root / 'assets'),
        'DeepSORT_DIR': str(repo_root / 'deep_sort'),
        'PYTHONPATH': str(repo_root),
    })

    result = subprocess.run(
        [sys.executable, 'Object_Tracking.py'],
        cwd=str(repo_root), env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f'tracking exited {result.returncode}\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}'
    )

    csv_path = tmp_pipeline_dirs['TRACKING_DIR'] / 'detected_objects.csv'
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    expected_cols = {'TrackID', 'ClassID', 'Mean', 'Co-Variance',
                     'ConfidenceScore', 'State', 'Hits', 'Age', 'Features', 'Frame'}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 0
    assert set(df['ClassID'].dropna().unique()).issubset({'Basketball', 'Team_A', 'Team_B'})
