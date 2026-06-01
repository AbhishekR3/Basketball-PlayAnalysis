'''Quick tests for nn/transforms.py augmentation transforms.'''

import numpy as np
import pytest


pytestmark = pytest.mark.quick


@pytest.fixture(scope='module')
def transforms_mod():
    """Import the nn.transforms module once for all tests in this module."""
    from nn import transforms as _t  # pylint: disable=import-outside-toplevel
    return _t


def test_compose_chains_transforms_in_order(transforms_mod):
    """ComposeTransforms applies its transforms left-to-right on a copy."""
    add_one = lambda x: x + 1
    times_two = lambda x: x * 2
    compose = transforms_mod.ComposeTransforms([add_one, times_two])
    out = compose(np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(out, np.array([[4.0, 6.0], [8.0, 10.0]]))


def test_jitter_transform_preserves_shape(transforms_mod):
    """jitter_transform returns an array with the same shape as the input."""
    np.random.seed(0)
    jitter = transforms_mod.jitter_transform(intensity=0.1)
    features = np.random.rand(50, 4).astype(np.float64)
    out = jitter(features)
    assert out.shape == features.shape
    assert not np.allclose(out, features)


def test_jitter_transform_skips_invalid_shape(transforms_mod):
    """jitter_transform short-circuits on empty/invalid-shape input."""
    jitter = transforms_mod.jitter_transform(intensity=0.1)
    empty = np.array([])
    assert jitter(empty).size == 0


def test_time_warp_transform_preserves_shape_on_long_seq(transforms_mod):
    """time_warp_transform preserves shape on sequences longer than the knot count."""
    np.random.seed(42)
    warp = transforms_mod.time_warp_transform(sigma=0.2, num_knots=4)
    features = np.arange(60, dtype=float).reshape(20, 3)
    out = warp(features)
    assert out.shape == features.shape


def test_time_warp_returns_original_for_short_seq(transforms_mod):
    """time_warp_transform returns its input unchanged when seq_len <= num_knots + 2."""
    warp = transforms_mod.time_warp_transform(sigma=0.2, num_knots=4)
    short = np.ones((3, 2))
    out = warp(short)
    np.testing.assert_array_equal(out, short)


def test_horizontal_flip_negates_x_position_column(transforms_mod):
    """horizontal_flip_transform negates x-position and x-velocity columns when applied."""
    flip = transforms_mod.horizontal_flip_transform(
        flip_probability=1.0, x_position_col=0, x_velocity_col=1,
    )
    features = np.array([[5.0, 1.0, 99.0], [10.0, -2.0, 88.0]])
    out = flip(features)
    assert out[0, 0] == -5.0
    assert out[1, 0] == -10.0
    assert out[0, 1] == -1.0
    assert out[1, 1] == 2.0
    assert out[0, 2] == 99.0


def test_horizontal_flip_skipped_when_probability_zero(transforms_mod):
    """horizontal_flip_transform returns its input unchanged when probability is 0."""
    flip = transforms_mod.horizontal_flip_transform(
        flip_probability=0.0, x_position_col=0,
    )
    features = np.array([[5.0, 1.0], [10.0, -2.0]])
    out = flip(features)
    np.testing.assert_array_equal(out, features)
