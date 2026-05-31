"""Behavioral tests for the basketball simulation scripts."""

import Passing_Simulation as ps


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
