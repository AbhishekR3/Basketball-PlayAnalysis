'''
End-to-end test for Feature_Engineering's transform chain.

Drives the pure transforms directly (no subprocess) using the shared
sample_tracking_df fixture. This exercises the feature_extraction() pipeline
end-to-end on synthetic but schema-correct data.
'''

import pytest


pytestmark = pytest.mark.extensive


@pytest.fixture(scope='module')
def fe():
    """Import Feature_Engineering once for all tests in this module."""
    import Feature_Engineering as _fe  # pylint: disable=import-outside-toplevel
    return _fe


def test_feature_extraction_chain_produces_expected_columns(fe, sample_tracking_df):
    """feature_extraction produces the schema-defined engineered columns."""
    df = sample_tracking_df.copy()
    df['Features'] = df['Features'].apply(fe.convert_string_to_array)

    out = fe.feature_extraction(df)

    for col in ['is_Team_A', 'is_Team_B', 'is_Basketball']:
        assert col in out.columns
        assert out[col].dtype == bool

    assert 'state_tentative' in out.columns
    assert 'state_confirmed' in out.columns

    assert 'time_since_start' in out.columns
    assert 'delta_time' in out.columns

    assert 'feature_min' in out.columns

    for col in ['cov_trace', 'cov_pos_variance_x', 'cov_pos_variance_y',
                'cov_vel_variance_x', 'cov_vel_variance_y']:
        assert col in out.columns

    for col in ['OcclusionFrequency', 'DetectionConsistency', 'RecentReliability']:
        assert col in out.columns

    for col in ['Normalized_Time_Frame', 'Temporal_TrackID']:
        assert col in out.columns

    for col in ['pos_x_rolling_avg', 'pos_y_rolling_avg',
                'vel_x_rolling_avg', 'vel_y_rolling_avg',
                'accel_x_rolling_avg', 'accel_y_rolling_avg',
                'feature_mean_rolling_avg']:
        assert col in out.columns, f'missing rolling-avg column {col}'


def test_optimize_dataset_normalizes_and_drops_columns(fe, sample_tracking_df_with_index):
    """optimize_dataset drops bookkeeping columns and dedupes (TrackID, Frame)."""
    df = sample_tracking_df_with_index.copy()
    df['Features'] = df['Features'].apply(fe.convert_string_to_array)
    extracted = fe.feature_extraction(df)
    cleaned = fe.optimize_dataset(extracted)

    for dropped in ['Mean', 'ConfidenceScore', 'State', 'Features',
                    'ClassID', 'RecentReliability']:
        assert dropped not in cleaned.columns

    flags = cleaned[['is_Team_A', 'is_Team_B', 'is_Basketball']]
    assert flags.any(axis=1).all()

    assert not cleaned.duplicated(subset=['TrackID', 'Frame']).any()
