'''
End-to-end test for Feature_Engineering's transform chain.

Drives the pure transforms directly (no subprocess) using the shared
sample_tracking_df fixture. This exercises the feature_extraction() pipeline
end-to-end on synthetic but schema-correct data.
'''

import numpy as np
import pandas as pd
import pytest


pytestmark = pytest.mark.extensive


@pytest.fixture(scope='module')
def fe():
    import Feature_Engineering as _fe
    return _fe


def test_feature_extraction_chain_produces_expected_columns(fe, sample_tracking_df):
    df = sample_tracking_df.copy()

    # Mirror the prep step that the script's main() does
    df['Features'] = df['Features'].apply(fe.convert_string_to_array)

    out = fe.feature_extraction(df)

    # One-hot ClassID
    for col in ['is_Team_A', 'is_Team_B', 'is_Basketball']:
        assert col in out.columns
        assert out[col].dtype == bool

    # State one-hot
    assert 'state_tentative' in out.columns
    assert 'state_confirmed' in out.columns

    # Temporal
    assert 'time_since_start' in out.columns
    assert 'delta_time' in out.columns

    # Feature stats (feature_min is the only stat not absorbed into rolling_avg)
    assert 'feature_min' in out.columns

    # Covariance-derived
    for col in ['cov_trace', 'cov_pos_variance_x', 'cov_pos_variance_y',
                'cov_vel_variance_x', 'cov_vel_variance_y']:
        assert col in out.columns

    # Hits/age derived
    for col in ['OcclusionFrequency', 'DetectionConsistency', 'RecentReliability']:
        assert col in out.columns

    # Temporal encoding
    for col in ['Normalized_Time_Frame', 'Temporal_TrackID']:
        assert col in out.columns

    # Rolling-average versions of position, velocity, accel, and stats
    for col in ['pos_x_rolling_avg', 'pos_y_rolling_avg',
                'vel_x_rolling_avg', 'vel_y_rolling_avg',
                'accel_x_rolling_avg', 'accel_y_rolling_avg',
                'feature_mean_rolling_avg']:
        assert col in out.columns, f'missing rolling-avg column {col}'


def test_optimize_dataset_normalizes_and_drops_columns(fe, sample_tracking_df_with_index):
    df = sample_tracking_df_with_index.copy()
    df['Features'] = df['Features'].apply(fe.convert_string_to_array)
    extracted = fe.feature_extraction(df)
    cleaned = fe.optimize_dataset(extracted)

    # Columns that optimize_dataset drops
    for dropped in ['Mean', 'ConfidenceScore', 'State', 'Features',
                    'ClassID', 'RecentReliability']:
        assert dropped not in cleaned.columns

    # All retained rows must have at least one class flag set
    flags = cleaned[['is_Team_A', 'is_Team_B', 'is_Basketball']]
    assert flags.any(axis=1).all()

    # No duplicate (TrackID, Frame) pairs
    assert not cleaned.duplicated(subset=['TrackID', 'Frame']).any()
