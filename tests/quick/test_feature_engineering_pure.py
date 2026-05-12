'''Quick tests for pure Feature_Engineering transforms.'''

import numpy as np
import pandas as pd
import pytest


pytestmark = pytest.mark.quick


@pytest.fixture(scope='module')
def fe():
    # Feature_Engineering.py touches LOG_DIR/TRACKING_DIR at import; conftest
    # sets those env vars so this import is safe.
    import Feature_Engineering as _fe
    return _fe


def test_one_hot_encode_class_id(fe):
    df = pd.DataFrame({'ClassID': ['Team_A', 'Team_B', 'Basketball', 'Team_A']})
    out = fe.one_hot_encode_class_id(df.copy())
    assert list(out['is_Team_A']) == [True, False, False, True]
    assert list(out['is_Team_B']) == [False, True, False, False]
    assert list(out['is_Basketball']) == [False, False, True, False]
    assert out['is_Team_A'].dtype == bool


def test_extract_mean_values_parses_eight_floats(fe):
    df = pd.DataFrame({
        'Mean': [
            '[1.0 2.0 0.5 50.0 0.1 0.2 0.0 0.0]',
            '[3.0 4.0 0.6 60.0 0.3 0.4 0.0 0.0]',
        ]
    })
    out = fe.extract_mean_values(df.copy())
    expected_cols = ['pos_x', 'pos_y', 'aspect_ratio', 'height',
                     'vel_x', 'vel_y', 'vel_aspect', 'vel_height']
    for col in expected_cols:
        assert col in out.columns
        assert out[col].dtype == float
    assert out.loc[0, 'pos_x'] == 1.0
    assert out.loc[1, 'height'] == 60.0
    assert out.loc[1, 'vel_y'] == pytest.approx(0.4)


def test_transform_state_one_hot(fe):
    df = pd.DataFrame({'State': [1, 2, 1, 2, 0]})
    out = fe.transform_state(df.copy())
    assert list(out['state_tentative']) == [True, False, True, False, False]
    assert list(out['state_confirmed']) == [False, True, False, True, False]


def test_convert_string_array_parses_2d_matrix(fe):
    cov_str = ("[[1.0 0.0 0.0]\n"
               " [0.0 2.0 0.0]\n"
               " [0.0 0.0 3.0]]")
    rows = fe.convert_string_array(cov_str)
    assert len(rows) == 3
    np.testing.assert_array_almost_equal(rows[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(rows[2], [0.0, 0.0, 3.0])


def test_rolling_average_calculation_min_periods_one(fe):
    df = pd.DataFrame({
        'TrackID': [0] * 5,
        'Frame': [0, 1, 2, 3, 4],
        'pos_x': [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    result = fe.rolling_average_calculation(df, window_size=3, rolling_column='pos_x')
    # min_periods=1, so no NaN even at the start
    assert not result.isna().any()
    assert result.iloc[0] == pytest.approx(10.0)
    assert result.iloc[1] == pytest.approx(15.0)
    assert result.iloc[2] == pytest.approx(20.0)  # mean of 10,20,30
    assert result.iloc[4] == pytest.approx(40.0)  # mean of 30,40,50


def test_calculate_feature_stats_returns_four_stats(fe):
    stats = fe.calculate_feature_stats(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert stats['feature_mean'] == pytest.approx(3.0)
    assert stats['feature_min'] == pytest.approx(1.0)
    assert stats['feature_max'] == pytest.approx(5.0)
    assert stats['feature_std'] == pytest.approx(np.std([1, 2, 3, 4, 5]))


def test_covariance_stats_calculation_handles_8x8(fe):
    cov = np.eye(8) * 2.0
    stats = fe.covariance_stats_calculation(cov)
    assert stats[0] == pytest.approx(16.0)  # trace = 8 * 2
    assert stats[2] == pytest.approx(2.0)
    assert stats[7] == pytest.approx(2.0)


def test_process_temporal_features_uses_fps(fe):
    df = pd.DataFrame({
        'TrackID': [0, 0, 0],
        'Frame': [0, 30, 60],
    })
    out = fe.process_temporal_features(df.copy(), fps=30)
    assert list(out['time_since_start']) == [0.0, 1.0, 2.0]
    assert out['delta_time'].iloc[1] == pytest.approx(1.0)
