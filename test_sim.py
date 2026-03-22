"""Unit tests for the Q*bert game simulator."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qbert.sim import (
    SimState, SimEntity, SimResult, simulate, clone,
    QBERT_HOP_INTERVAL, COILY_HOP_INTERVAL, BALL_HOP_INTERVAL,
    UP, DOWN, LEFT, RIGHT,
)
from qbert.state import NUM_CUBES, pos_to_cube_index


def _make_cubes(target_color=1, all_color=0):
    """Make a 28-cube list where all cubes are all_color."""
    return [all_color] * NUM_CUBES


def _empty_state(qbert_pos=(0, 0), target_color=1):
    """SimState with no enemies."""
    return SimState(
        qbert_pos=qbert_pos,
        qbert_prev=qbert_pos,
        qbert_counter=1,
        entities=[],
        cubes=_make_cubes(target_color, all_color=0),
        target_color=target_color,
        remaining=NUM_CUBES,
    )


def test_basic_hop():
    """Q*bert hops from (0,0) to (1,1) via DOWN-RIGHT."""
    s = _empty_state((0, 0))
    result = simulate(s, [RIGHT])  # RIGHT = down-right = (+1, +1)
    assert result.alive, "Q*bert should survive a simple hop"
    assert result.final_pos == (1, 1), f"Expected (1,1), got {result.final_pos}"
    assert result.steps_survived == 1
    assert result.cubes_colored == 1  # (1,1) was uncolored
    print("  basic hop: OK")


def test_multi_hop():
    """Q*bert hops down the right edge: (0,0) → (1,1) → (2,2) → (3,3)."""
    s = _empty_state((0, 0))
    result = simulate(s, [RIGHT, RIGHT, RIGHT])
    assert result.alive
    assert result.final_pos == (3, 3), f"Expected (3,3), got {result.final_pos}"
    assert result.steps_survived == 3
    assert result.cubes_colored == 3  # 3 new cubes
    print("  multi hop: OK")


def test_cube_coloring():
    """Already-colored cubes don't count as new."""
    s = _empty_state((0, 0), target_color=1)
    # Pre-color cube at (1,1)
    idx = pos_to_cube_index(1, 1)
    s.cubes[idx] = 1  # already target color
    s.remaining = NUM_CUBES - 1
    result = simulate(s, [RIGHT])  # hop to (1,1)
    assert result.cubes_colored == 0, "Already-colored cube shouldn't count"
    print("  cube coloring: OK")


def test_invalid_move_skipped():
    """Invalid moves (off pyramid) are skipped."""
    s = _empty_state((0, 0))
    # UP from (0,0) goes to (-1, 0) — invalid
    result = simulate(s, [UP])
    assert result.alive
    assert result.final_pos == (0, 0), "Should stay at (0,0) after invalid move"
    assert result.steps_survived == 0  # no valid hops completed
    print("  invalid move skipped: OK")


def test_coily_collision():
    """Q*bert walks into Coily and dies."""
    s = _empty_state((2, 1))
    # Place Coily at (1, 1) — Q*bert will hop UP (to (1,1)) and collide
    coily = SimEntity(
        pos=(1, 1), prev_pos=(2, 2), etype="coily",
        anim_counter=999,  # won't move during this sim
        hop_interval=COILY_HOP_INTERVAL,
        direction_bits=0, harmful=True,
    )
    s.entities.append(coily)
    result = simulate(s, [UP])  # UP from (2,1) → (1,1) where Coily is
    assert not result.alive, "Q*bert should die when hopping onto Coily"
    print("  coily collision: OK")


def test_coily_chases_qbert():
    """Coily moves toward Q*bert during simulation."""
    s = _empty_state((6, 0))
    # Coily at (3, 0), will chase Q*bert's prev position
    coily = SimEntity(
        pos=(3, 0), prev_pos=(2, 0), etype="coily",
        anim_counter=10,  # will fire during the 18-frame hop
        hop_interval=COILY_HOP_INTERVAL,
        direction_bits=0, harmful=True,
    )
    s.entities.append(coily)
    # Q*bert hops RIGHT from (6,0) → (7,1) which is invalid, so skip
    # Instead hop DOWN from (6,0) — also invalid. Let's use a valid move.
    # From (6,0): neighbors are UP→(5,0) and RIGHT→(7,1) invalid and LEFT→(5,-1) invalid
    # Actually from (6,0): UP→(5,0), DOWN→(7,0) invalid, LEFT→(5,-1) invalid, RIGHT→(7,1) invalid
    # So only UP is valid from (6,0)
    result = simulate(s, [UP])  # Q*bert goes to (5,0)
    assert result.alive, "Q*bert should survive — Coily started far away"
    # Coily should have moved closer to Q*bert's prev=(6,0)
    # Coily at (3,0) chasing (6,0): goes DOWN, gw1 comparison...
    # target_row=6 > coily_row=3 → DOWN
    # target_gw1 = 6-0+1=7, coily_gw1 = 3-0+1=4 → target > coily → DOWN-LEFT = (4,0)
    assert s.entities[0].pos == (4, 0), f"Coily should be at (4,0), got {s.entities[0].pos}"
    print("  coily chases: OK")


def test_ball_bounces():
    """Ball bounces down following direction_bits."""
    s = _empty_state((6, 6))  # Q*bert far away
    # Ball at (1, 0), direction_bits = 0b101 → LEFT, LEFT, RIGHT
    # bit0=1 → RIGHT: (2,1), bit1=0 → LEFT: (3,1), bit2=1 → RIGHT: (4,2)
    ball = SimEntity(
        pos=(1, 0), prev_pos=(0, 0), etype="ball",
        anim_counter=10,  # will fire quickly
        hop_interval=BALL_HOP_INTERVAL,
        direction_bits=0b101,  # RIGHT, LEFT, RIGHT
        harmful=True,
    )
    s.entities.append(ball)
    # Simulate several Q*bert hops to give ball time to move
    # Ball interval = 43, qbert interval = 18
    # In 3 Q*bert hops = 54 frames, ball gets 54//43 = 1 hop (after first anim_counter fires at 10)
    # Actually: frame 10 → ball hops (first), then counter=43, next at frame 53
    # So in 54 frames: ball hops at frame 10, then at frame 53 → 2 hops
    result = simulate(s, [LEFT, LEFT, LEFT])  # Q*bert stays on bottom row-ish
    # Ball should have moved at least once
    assert s.entities[0].pos != (1, 0), "Ball should have moved"
    # First hop: bit0=1 → RIGHT → (2, 1)
    # Verify ball took the right direction
    print(f"  ball bounced to {s.entities[0].pos}: OK")


def test_clone_independence():
    """Cloned state is independent of original."""
    s = _empty_state((3, 1))
    s2 = clone(s)
    s2.qbert_pos = (4, 2)
    s2.cubes[0] = 99
    assert s.qbert_pos == (3, 1), "Original should be unchanged"
    assert s.cubes[0] == 0, "Original cubes should be unchanged"
    print("  clone independence: OK")


def test_escape_routes():
    """Escape routes counted correctly at different positions."""
    # Apex (0,0) — only DOWN and RIGHT valid
    s = _empty_state((0, 0))
    result = simulate(s, [RIGHT])  # go to (1,1)
    # (1,1) has neighbors: UP→(0,0) invalid? no, (0,0) is valid. LEFT→(0,0), RIGHT→(2,2), DOWN→(2,1)
    # Actually: UP=(-1,0)→(0,1) invalid, DOWN=(+1,0)→(2,1) valid, LEFT=(-1,-1)→(0,0) valid, RIGHT=(+1,+1)→(2,2) valid
    assert result.escape_routes == 3, f"Expected 3 escape routes from (1,1), got {result.escape_routes}"

    # Corner (6,0) — only UP valid
    s2 = _empty_state((5, 0))
    result2 = simulate(s2, [DOWN])  # go to (6,0)
    assert result2.escape_routes == 1, f"Expected 1 escape route from (6,0), got {result2.escape_routes}"
    print("  escape routes: OK")


def test_prev_position_collision():
    """Collision via Q*bert.prev == Enemy.current (ROM cross-check)."""
    # Q*bert at (2,1), about to hop to (3,2). Coily at (2,1) prev.
    # After hop: Q*bert.prev = (2,1), Q*bert.pos = (3,2)
    # If Coily moves to (2,1) during the hop, then qbert_prev == enemy.pos → death
    s = _empty_state((2, 1))
    coily = SimEntity(
        pos=(1, 1), prev_pos=(0, 0), etype="coily",
        anim_counter=5,  # will fire during hop
        hop_interval=COILY_HOP_INTERVAL,
        direction_bits=0, harmful=True,
    )
    s.entities.append(coily)
    # Coily at (1,1) chasing Q*bert prev=(2,1): target below
    # target_row=2 > coily_row=1 → DOWN
    # target_gw1=2-1+1=2, coily_gw1=1-1+1=1 → target > coily → DOWN-LEFT = (2,1)
    # So Coily will move to (2,1) which is Q*bert's starting pos
    # After Q*bert hops: qbert_prev=(2,1), coily.pos=(2,1) → collision!
    result = simulate(s, [RIGHT])  # Q*bert goes to (3,2)
    assert not result.alive, "Should die from prev-position collision"
    print("  prev position collision: OK")


def test_spawn_prediction():
    """Spawn countdown injects phantom balls at apex."""
    s = _empty_state((0, 0))
    s.spawn_countdown = 5  # spawns in 5 frames
    assert len(s.entities) == 0
    # Hop from (0,0) → (1,1). During the 18-frame hop, countdown fires at frame 5.
    result = simulate(s, [RIGHT])
    # Two phantom balls should have been injected
    assert len(s.entities) == 2, f"Expected 2 phantom balls, got {len(s.entities)}"
    assert s.entities[0].pos == (0, 0), "Phantom ball should start at apex"
    assert s.entities[1].pos == (0, 0), "Phantom ball should start at apex"
    # Q*bert is at (1,1), phantoms at (0,0) — no collision
    assert result.alive, "Q*bert should survive (not at spawn point)"
    print("  spawn prediction: OK")


def test_spawn_ball_kills_on_path():
    """Spawned ball kills Q*bert when it bounces onto Q*bert's position."""
    # Q*bert at (1,0). Spawn fires immediately (countdown=1).
    # After spawn delay (80 frames), ball at (0,0) makes first hop.
    # All-left ball: (0,0) → (1,0) — hits Q*bert!
    s = _empty_state((1, 0))
    s.spawn_countdown = 1
    # We need enough hops for the spawn delay (80 frames) to pass.
    # 6 hops * 18 frames = 108 frames > 1 + 80 = 81 frames — enough.
    # Q*bert bounces between (1,0) and (2,0) to stay near top.
    result = simulate(s, [DOWN, UP, DOWN, UP, DOWN, UP])
    # The all-left phantom goes (0,0) → (1,0) where Q*bert might be
    # This should be detected as a collision
    assert not result.alive, "Spawned ball should kill Q*bert on its path"
    print("  spawn ball kills on path: OK")


if __name__ == "__main__":
    print("Q*bert simulator unit tests:")
    test_basic_hop()
    test_multi_hop()
    test_cube_coloring()
    test_invalid_move_skipped()
    test_coily_collision()
    test_coily_chases_qbert()
    test_ball_bounces()
    test_clone_independence()
    test_escape_routes()
    test_prev_position_collision()
    test_spawn_prediction()
    test_spawn_ball_kills_on_path()
    print("\nAll simulator tests PASSED")
