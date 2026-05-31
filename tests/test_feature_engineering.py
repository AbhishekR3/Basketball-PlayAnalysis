"""Behavioral tests for Feature_Engineering (P0-E)."""

import numpy as np
import pandas as pd
import pytest

import Feature_Engineering as fe


def test_acceleration_has_no_nan_or_inf_with_zero_delta_time():
    """First frame per track has delta_time == 0, and velocities can be 0;
    neither may leak inf/NaN into the acceleration columns."""
    df = pd.DataFrame(
        {
            'TrackID': [1, 1, 1],
            'Frame': [0, 1, 2],
            'vel_x': [0.0, 1.0, 2.0],
            'vel_y': [0.0, 1.0, 2.0],
            'vel_aspect': [0.0, 0.1, 0.2],
            'vel_height': [0.0, 1.0, 0.0],  # zero velocity stresses the denominator
            'delta_time': [0.0, 1 / 30, 1 / 30],  # first frame delta_time == 0
        }
    )

    out = fe.extract_acceleration(df)

    for col in ['accel_x', 'accel_y', 'accel_aspect', 'accel_height']:
        assert np.isfinite(out[col]).all(), f"{col} contains NaN/inf"


def test_schema_validator_rejects_missing_columns():
    incomplete = pd.DataFrame({'Frame': [0], 'TrackID': [1]})
    with pytest.raises(ValueError):
        fe.validate_tracking_schema(incomplete)


def test_schema_validator_accepts_complete_schema():
    complete = pd.DataFrame({col: [0] for col in fe.REQUIRED_TRACKING_COLUMNS})
    assert fe.validate_tracking_schema(complete) is complete


def test_pca_flag_toggles_output_dimensionality():
    """apply_pca_transform (the function the PCA flag invokes) must reduce the
    numeric feature width while preserving identity/categorical columns."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((20, 25)), columns=[f'f{i}' for i in range(25)])
    df['Frame'] = range(20)
    df['is_Team_A'] = True
    df['is_Team_B'] = False
    df['is_Basketball'] = False

    reduced = fe.apply_pca_transform(df, n_components=5)

    assert reduced.shape[1] < df.shape[1]
    assert sum(col.startswith('PC') for col in reduced.columns) == 5
    # preserved columns survive the reduction
    for col in ['Frame', 'is_Team_A', 'is_Team_B', 'is_Basketball']:
        assert col in reduced.columns
