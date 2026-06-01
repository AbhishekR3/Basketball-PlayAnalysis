'''
Quick tests for RandomMovement_Simulation Player/Basketball movement math.

The simulation module runs pygame.init() and a full simulation loop at import
time, so we AST-extract just the classes/helpers we want to test and run them
in an isolated namespace.
'''

import math
import secrets
import time
import types

import numpy as np
import pytest

from tests.conftest import REPO_ROOT, load_callable_from_source


pytestmark = pytest.mark.quick


SIM_PATH = REPO_ROOT / 'RandomMovement_Simulation.py'


@pytest.fixture(scope='module')
def helpers():
    """AST-extract Player/Basketball/cryptographic_normal/reverse_color into a namespace."""
    logger = types.SimpleNamespace(error=lambda *a, **k: None,
                                   debug=lambda *a, **k: None,
                                   info=lambda *a, **k: None,
                                   warning=lambda *a, **k: None)
    extra = {
        'secrets': secrets, 'math': math, 'time': time, 'np': np,
        'logger': logger,
        # P0-H replaced the stdlib secret-random draws with a seeded module-level
        # _rng (np.random.default_rng). AST extraction runs these callables in an
        # isolated namespace, so inject a seeded _rng to satisfy that dependency.
        '_rng': np.random.default_rng(42),
    }
    cryptographic_normal = load_callable_from_source(
        SIM_PATH, 'cryptographic_normal', extra_globals=extra)
    reverse_color = load_callable_from_source(
        SIM_PATH, 'reverse_color', extra_globals=extra)
    class_extra = dict(extra)
    class_extra['SCREEN_WIDTH'] = 470
    class_extra['SCREEN_HEIGHT'] = 500
    class_extra['cryptographic_normal'] = cryptographic_normal
    player_cls = load_callable_from_source(SIM_PATH, 'Player', extra_globals=class_extra)
    basketball_cls = load_callable_from_source(SIM_PATH, 'Basketball', extra_globals=class_extra)
    return types.SimpleNamespace(
        cryptographic_normal=cryptographic_normal,
        reverse_color=reverse_color,
        Player=player_cls,
        Basketball=basketball_cls,
        screen_w=class_extra['SCREEN_WIDTH'],
        screen_h=class_extra['SCREEN_HEIGHT'],
    )


def test_cryptographic_normal_radian_in_reasonable_range(helpers):
    """cryptographic_normal in radian mode produces a near-zero mean and non-zero std."""
    samples = [helpers.cryptographic_normal(0, 1, True) for _ in range(500)]
    arr = np.array(samples)
    assert abs(arr.mean()) < 0.1
    assert arr.std() > 0


def test_cryptographic_normal_degree_mode_returns_finite(helpers):
    """cryptographic_normal in degree mode returns a finite float."""
    val = helpers.cryptographic_normal(90, 5, False)
    assert math.isfinite(val)


def test_reverse_color_inverts_each_channel(helpers):
    """reverse_color flips each RGB channel to its opposite extreme."""
    assert helpers.reverse_color((255, 0, 0)) == (0, 255, 255)
    assert helpers.reverse_color((0, 0, 0)) == (255, 255, 255)
    assert helpers.reverse_color((255, 255, 255)) == (0, 0, 0)


def test_player_move_advances_position_along_angle(helpers):
    """Player.move advances (x, y) by speed * (cos(angle), sin(angle))."""
    p = helpers.Player(x=200.0, y=200.0, radius=10, color=(0, 0, 255))
    p.speed = 1.0
    p.angle = 0.0
    p.last_update_time = time.time()
    p.change_direction_time_limit = 999
    p.move()
    assert p.x == pytest.approx(201.0)
    assert p.y == pytest.approx(200.0)


def test_player_bounces_off_right_wall(helpers):
    """Player.move flips angle when it hits the right wall."""
    p = helpers.Player(x=helpers.screen_w - 16, y=200.0, radius=10, color=(0, 0, 255))
    p.speed = 5.0
    p.angle = 0.0
    p.last_update_time = time.time()
    p.change_direction_time_limit = 999
    p.move()
    assert p.angle == pytest.approx(math.pi)


def test_basketball_move_advances_position_along_angle(helpers):
    """Basketball.move advances (x, y) by speed * (cos(angle), sin(angle))."""
    b = helpers.Basketball(x=100.0, y=100.0, radius=10, color=(255, 165, 0))
    b.speed = 2.0
    b.angle = math.pi / 2
    b.last_update_time = time.time()
    b.change_direction_time_limit = 999
    b.move()
    assert b.x == pytest.approx(100.0, abs=1e-9)
    assert b.y == pytest.approx(102.0)
