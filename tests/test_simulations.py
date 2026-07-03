"""Behavioral tests for the basketball simulation scripts."""

import math

import Passing_Simulation as ps
import RandomMovement_Simulation as rm


def test_dribble_position_anchors_to_dribbling_player():
    """P0-A: new_relative_basketball_position must anchor the ball to the
    dribbling_player argument, not the stale global current_player it used to
    reference."""
    dribbling_player = ps.Player(0.0, 0.0, ps.PLAYER_RADIUS, ps.COLOR_BLUE)

    def call():
        # ball_x, ball_y, randomness, offset, angle, dribbling_player, displacement
        return ps.new_relative_basketball_position(
            300.0, 300.0, True, 0.0, 0.0, dribbling_player, 15
        )

    # With dribbling_player at the origin the function takes the right-side
    # branch: x stays at ball_x, y = dribbling_player.y + displacement = 15.
    ps.current_player = ps.Player(0.0, 100.0, ps.PLAYER_RADIUS, ps.COLOR_RED)
    res_a = call()
    ps.current_player = ps.Player(0.0, 500.0, ps.PLAYER_RADIUS, ps.COLOR_RED)
    res_b = call()

    # Anchored to the dribbling_player ...
    assert res_a == (300.0, 15.0)
    # ... and independent of whatever the global current_player happens to be.
    assert res_a == res_b


# --- P0-H: seeded RNG reproducibility ---------------------------------------

def test_seeded_rng_is_reproducible():
    """Same seed -> identical sequence of random draws (both sims)."""
    for mod in (ps, rm):
        mod.seed_rng(123)
        first = [float(mod._rng.uniform(0, 1)) for _ in range(5)]
        mod.seed_rng(123)
        second = [float(mod._rng.uniform(0, 1)) for _ in range(5)]
        assert first == second


def test_seeded_players_are_identical():
    """Two players built under the same seed have identical random attributes."""
    ps.seed_rng(7)
    p1 = ps.Player(0, 0, ps.PLAYER_RADIUS, ps.COLOR_BLUE)
    ps.seed_rng(7)
    p2 = ps.Player(0, 0, ps.PLAYER_RADIUS, ps.COLOR_BLUE)
    assert p1.speed == p2.speed
    assert p1.angle == p2.angle
    assert p1.next_angle == p2.next_angle


def test_different_seeds_diverge():
    ps.seed_rng(1)
    a = float(ps._rng.uniform(0, 1))
    ps.seed_rng(2)
    b = float(ps._rng.uniform(0, 1))
    assert a != b


# --- P0-G: player edge-stick (#13) and ball-receive jerk (#15) ---------------

def test_player_does_not_stick_to_wall():
    """#13: a player heading into a wall must stay in-bounds every frame and
    escape the edge band instead of orbiting it."""
    ps.seed_rng(0)
    ball = ps.Basketball(0, 0, ps.BALL_RADIUS, ps.COLOR_ORANGE)
    player = ps.Player(0, 0, ps.PLAYER_RADIUS, ps.COLOR_BLUE)

    max_x = ps.SCREEN_WIDTH - ps.WALL_MARGIN - player.radius
    max_y = ps.SCREEN_HEIGHT - ps.WALL_MARGIN - player.radius

    # Start adjacent to the right wall, heading outward (angle 0 -> +x)
    player.x = max_x - 1
    player.y = ps.SCREEN_HEIGHT / 2
    player.angle = 0.0
    player.speed = 2.0

    xs = []
    for _ in range(120):
        player.move(ball)
        xs.append(player.x)
        # Never leaves the valid interior
        assert player.radius <= player.x <= max_x
        assert player.radius <= player.y <= max_y

    # Escaped the right-edge band rather than orbiting it
    assert min(xs) < max_x - 50


def test_ball_approaches_smoothly_with_negative_dx():
    """#15: moving the ball toward a target that requires negative dx must produce
    smooth ~MOVE_SPEED steps (no zero-then-jump), not an instant stall."""
    move_speed = abs(ps.MOVE_SPEED)
    ball = ps.Basketball(300, 100, ps.BALL_RADIUS, ps.COLOR_ORANGE)
    target_x, target_y = 50, 100  # to the left of the ball -> dx < 0

    prev = (ball.x, ball.y)
    steps = []
    for _ in range(60):
        ps.move_basketball_to_location(ball, target_x, target_y)
        steps.append(math.hypot(ball.x - prev[0], ball.y - prev[1]))
        prev = (ball.x, ball.y)
        if math.hypot(target_x - ball.x, target_y - ball.y) <= ps.BALL_SNAP_THRESHOLD:
            break

    # The ball actually started moving (old bug zeroed it because dx < 0)
    assert steps[0] > 0
    # No teleport frames
    assert all(s <= move_speed * 1.5 for s in steps)
    # Each moving step is a steady ~MOVE_SPEED approach (no zero-then-jump)
    nonzero = [s for s in steps if s > 1e-9]
    assert all(abs(s - move_speed) < 1e-6 for s in nonzero)
    # Reached the snap vicinity of the target
    assert math.hypot(target_x - ball.x, target_y - ball.y) <= ps.BALL_SNAP_THRESHOLD


def test_random_player_does_not_stick_to_wall():
    """#13: the RandomMovement player has the same wall handler as Passing and must
    stay in-bounds every frame and escape the edge band instead of orbiting it."""
    rm.seed_rng(0)
    player = rm.Player(0, 0, rm.PLAYER_RADIUS, rm.COLOR_BLUE)

    max_x = rm.SCREEN_WIDTH - rm.WALL_MARGIN - player.radius
    max_y = rm.SCREEN_HEIGHT - rm.WALL_MARGIN - player.radius

    # Start adjacent to the right wall, heading outward (angle 0 -> +x)
    player.x = max_x - 1
    player.y = rm.SCREEN_HEIGHT / 2
    player.angle = 0.0
    player.speed = 2.0

    xs = []
    for _ in range(120):
        player.move()
        xs.append(player.x)
        assert player.radius <= player.x <= max_x
        assert player.radius <= player.y <= max_y

    assert min(xs) < max_x - 50


def test_random_basketball_does_not_stick_to_wall():
    """#13: the RandomMovement basketball uses the same wall handler and must not
    orbit the edge either."""
    rm.seed_rng(0)
    ball = rm.Basketball(0, 0, rm.BALL_RADIUS, rm.COLOR_ORANGE)

    max_x = rm.SCREEN_WIDTH - rm.WALL_MARGIN - ball.radius
    max_y = rm.SCREEN_HEIGHT - rm.WALL_MARGIN - ball.radius

    # Start adjacent to the right wall, heading outward (angle 0 -> +x)
    ball.x = max_x - 1
    ball.y = rm.SCREEN_HEIGHT / 2
    ball.angle = 0.0
    ball.speed = 2.0

    xs = []
    for _ in range(120):
        ball.move()
        xs.append(ball.x)
        assert ball.radius <= ball.x <= max_x
        assert ball.radius <= ball.y <= max_y

    assert min(xs) < max_x - 50
