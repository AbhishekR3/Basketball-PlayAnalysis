'''Quick tests for nn/transforms.py augmentation transforms.'''

import numpy as np
import pytest


pytestmark = pytest.mark.quick


@pytest.fixture(scope='module')
def transforms_mod():
    from nn import transforms as _t
    return _t


def test_compose_chains_transforms_in_order(transforms_mod):
    add_one = lambda x: x + 1
    times_two = lambda x: x * 2
    compose = transforms_mod.ComposeTransforms([add_one, times_two])
    out = compose(np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(out, np.array([[4.0, 6.0], [8.0, 10.0]]))


def test_jitter_transform_preserves_shape(transforms_mod):
    np.random.seed(0)
    jitter = transforms_mod.jitter_transform(intensity=0.1)
    # 50 timesteps x 4 features, each column has > 5 unique values so jitter applies
    features = np.random.rand(50, 4).astype(np.float64)
    out = jitter(features)
    assert out.shape == features.shape
    # at least one column should differ from the original (jitter actually applied)
    assert not np.allclose(out, features)


def test_jitter_transform_skips_invalid_shape(transforms_mod):
    jitter = transforms_mod.jitter_transform(intensity=0.1)
    empty = np.array([])
    assert jitter(empty).size == 0


def test_time_warp_transform_preserves_shape_on_long_seq(transforms_mod):
    np.random.seed(42)
    warp = transforms_mod.time_warp_transform(sigma=0.2, num_knots=4)
    features = np.arange(60, dtype=float).reshape(20, 3)
    out = warp(features)
    assert out.shape == features.shape


def test_time_warp_returns_original_for_short_seq(transforms_mod):
    warp = transforms_mod.time_warp_transform(sigma=0.2, num_knots=4)
    short = np.ones((3, 2))  # seq_len < num_knots + 2
    out = warp(short)
    np.testing.assert_array_equal(out, short)


def test_horizontal_flip_negates_x_position_column(transforms_mod):
    # flip_probability=1.0 guarantees flip
    flip = transforms_mod.horizontal_flip_transform(
        flip_probability=1.0, x_position_col=0, x_velocity_col=1,
    )
    features = np.array([[5.0, 1.0, 99.0], [10.0, -2.0, 88.0]])
    out = flip(features)
    assert out[0, 0] == -5.0
    assert out[1, 0] == -10.0
    assert out[0, 1] == -1.0
    assert out[1, 1] == 2.0
    # untouched column preserved
    assert out[0, 2] == 99.0


def test_horizontal_flip_skipped_when_probability_zero(transforms_mod):
    flip = transforms_mod.horizontal_flip_transform(
        flip_probability=0.0, x_position_col=0,
    )
    features = np.array([[5.0, 1.0], [10.0, -2.0]])
    out = flip(features)
    np.testing.assert_array_equal(out, features)
