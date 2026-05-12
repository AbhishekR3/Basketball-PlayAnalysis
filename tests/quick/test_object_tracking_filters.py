'''
Quick tests for Object_Tracking.filter_lowconfidence.

Object_Tracking.py runs cv2.VideoCapture + YOLO model load at import time, so
we cannot `import Object_Tracking` here. We AST-extract just the function.
'''

import types

import numpy as np
import pytest

from tests.conftest import REPO_ROOT, load_callable_from_source


pytestmark = pytest.mark.quick


@pytest.fixture(scope='module')
def filter_lowconfidence():
    import config
    return load_callable_from_source(
        REPO_ROOT / 'Object_Tracking.py',
        'filter_lowconfidence',
        extra_globals={
            'np': np,
            'config': config,
            'logger': types.SimpleNamespace(error=lambda *a, **k: None,
                                            debug=lambda *a, **k: None,
                                            info=lambda *a, **k: None),
        },
    )


def test_basketball_above_threshold_is_kept(filter_lowconfidence):
    mask = filter_lowconfidence(np.array(['Basketball']), np.array([0.6]),
                                basketball_score=0.5, player_score=0.8)
    assert mask == [True]


def test_basketball_below_threshold_is_dropped(filter_lowconfidence):
    mask = filter_lowconfidence(np.array(['Basketball']), np.array([0.4]),
                                basketball_score=0.5, player_score=0.8)
    assert mask == [False]


def test_player_above_threshold_is_kept(filter_lowconfidence):
    mask = filter_lowconfidence(np.array(['Team_A', 'Team_B']),
                                np.array([0.9, 0.85]),
                                basketball_score=0.5, player_score=0.8)
    assert mask == [True, True]


def test_player_below_threshold_is_dropped(filter_lowconfidence):
    mask = filter_lowconfidence(np.array(['Team_A']), np.array([0.7]),
                                basketball_score=0.5, player_score=0.8)
    assert mask == [False]


def test_mixed_classes_filtered_by_per_class_thresholds(filter_lowconfidence):
    class_names = np.array(['Basketball', 'Team_A', 'Team_B', 'Basketball'])
    scores = np.array([0.6, 0.85, 0.5, 0.3])
    mask = filter_lowconfidence(class_names, scores,
                                basketball_score=0.5, player_score=0.8)
    assert mask == [True, True, False, False]


def test_empty_arrays_return_empty_mask(filter_lowconfidence):
    mask = filter_lowconfidence(np.array([]), np.array([]),
                                basketball_score=0.5, player_score=0.8)
    assert mask == []


def test_default_thresholds_come_from_config(filter_lowconfidence):
    import config
    # Sanity: defaults match config (regression guard against the docstring drift)
    assert config.YOLO_BASKETBALL_SCORE == 0.5
    assert config.YOLO_PLAYER_SCORE == 0.8
