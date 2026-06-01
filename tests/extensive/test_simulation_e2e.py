'''
End-to-end test for RandomMovement_Simulation.py.

Runs the simulation as a subprocess (the script has heavy module-level side
effects), under SDL dummy driver and the GITHUB_ACTIONS flag that caps the
simulation at ~1 second / 30 frames.
'''

import os
import subprocess
import sys

import pytest


pytestmark = pytest.mark.extensive


def test_random_movement_simulation_produces_mp4(tmp_pipeline_dirs, repo_root):
    """RandomMovement_Simulation.py writes an MP4 with at least 10 frames in CI mode."""
    cv2 = pytest.importorskip('cv2')
    env = os.environ.copy()
    env.update({
        'SDL_VIDEODRIVER': 'dummy',
        'GITHUB_ACTIONS': 'true',
        'LOG_DIR': str(tmp_pipeline_dirs['LOG_DIR']),
        'VIDEO_DIR': str(tmp_pipeline_dirs['VIDEO_DIR']),
        'ASSETS_DIR': str(repo_root / 'assets'),
        'PYTHONPATH': str(repo_root),
    })

    result = subprocess.run(
        [sys.executable, 'RandomMovement_Simulation.py'],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    mp4 = tmp_pipeline_dirs['VIDEO_DIR'] / 'random_movement_video.mp4'
    assert result.returncode == 0, (
        f'simulation exited {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}'
    )
    assert mp4.exists(), f'expected {mp4} to exist'
    assert mp4.stat().st_size > 0

    cap = cv2.VideoCapture(str(mp4))
    try:
        assert cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count >= 10, f'expected >= 10 frames, got {frame_count}'
    finally:
        cap.release()
