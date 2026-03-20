"""Coily prediction test — unit tests against ROM-verified chase algorithm.

The algorithm at ROM $B6EA is fully deterministic:
  - Compares grid words (gw0=row+1, gw1=row-col+1)
  - target_row > coily_row → go DOWN; compare gw1 for LEFT/RIGHT
  - target_row <= coily_row → go UP; compare gw1 for LEFT/RIGHT
  - Coily chases Q*bert's PREVIOUS position

These test cases are derived directly from the ROM disassembly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qbert.predict import predict_coily, predict_coily_n, predict_ball_path


def test_coily_basic():
    """Test all four direction cases."""
    # Target below-right: DOWN-RIGHT (+1, +1)
    assert predict_coily(2, 1, 4, 3) == (3, 2), "DOWN-RIGHT failed"

    # Target below-left: DOWN-LEFT (+1, 0)
    assert predict_coily(2, 1, 4, 1) == (3, 1), "DOWN-LEFT failed"

    # Target above-right: UP-RIGHT (-1, 0)
    assert predict_coily(4, 2, 2, 2) == (3, 2), "UP-RIGHT failed"

    # Target above-left: UP-LEFT (-1, -1)
    assert predict_coily(4, 2, 2, 0) == (3, 1), "UP-LEFT failed"

    print("  basic directions: OK")


def test_coily_same_row():
    """When on same row, Coily goes UP (row comparison uses <=)."""
    # Same row, target left → UP-LEFT
    assert predict_coily(3, 2, 3, 0) == (2, 1), "same row left failed"

    # Same row, target right → UP-RIGHT
    assert predict_coily(3, 1, 3, 3) == (2, 1), "same row right failed"

    # Same position → UP-LEFT (both comparisons: row<=, gw1>=)
    assert predict_coily(3, 2, 3, 2) == (2, 1), "same position failed"

    print("  same row: OK")


def test_coily_chase_sequence():
    """Simulate Coily chasing a stationary Q*bert, verify the full path."""
    # Coily at (6, 0), Q*bert (prev) at (0, 0)
    # Coily should go straight UP-LEFT to (0,0)
    r, c = 6, 0
    path = []
    for _ in range(6):
        r, c = predict_coily(r, c, 0, 0)
        path.append((r, c))
    assert path == [(5, 0), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0)], \
        f"straight chase failed: {path}"

    # Coily at (6, 6), Q*bert at (0, 0)
    # Should zig-zag up: UP-LEFT each time
    r, c = 6, 6
    path = []
    for _ in range(6):
        r, c = predict_coily(r, c, 0, 0)
        path.append((r, c))
    # gw1 comparison: target_gw1=1, coily_gw1=row-col+1=1 → equal → UP-LEFT
    assert path == [(5, 5), (4, 4), (3, 3), (2, 2), (1, 1), (0, 0)], \
        f"diagonal chase failed: {path}"

    # Coily at (6, 3), Q*bert at (2, 1)
    r, c = 6, 3
    path = []
    for _ in range(4):
        r, c = predict_coily(r, c, 2, 1)
        path.append((r, c))
    # Step 1: target_row=2 < coily_row=6 → UP
    #   target_gw1 = 2-1+1=2, coily_gw1 = 6-3+1=4 → target < coily → UP-RIGHT
    assert path[0] == (5, 3), f"step 1 failed: {path[0]}"
    # Step 2: coily(5,3), target(2,1)
    #   UP, target_gw1=2, coily_gw1=5-3+1=3 → UP-RIGHT
    assert path[1] == (4, 3), f"step 2 failed: {path[1]}"
    # Step 3: coily(4,3), target(2,1)
    #   UP, target_gw1=2, coily_gw1=4-3+1=2 → equal → UP-LEFT
    assert path[2] == (3, 2), f"step 3 failed: {path[2]}"
    # Step 4: coily(3,2), target(2,1)
    #   UP, target_gw1=2, coily_gw1=3-2+1=2 → equal → UP-LEFT
    assert path[3] == (2, 1), f"step 4 failed: {path[3]}"

    print("  chase sequences: OK")


def test_coily_n():
    """Test multi-hop prediction."""
    pos = predict_coily_n((6, 3), (0, 0), 3)
    # (6,3) → (5,3) → (4,3) → (3,3) — all UP-LEFT with gw1 equal
    # Actually: gw1 check: target_gw1=1, coily_gw1=6-3+1=4 → target < coily → UP-RIGHT
    # (6,3)→(5,3)→(4,3)→(3,3)... let me recalc
    # coily(6,3) target(0,0): UP, t_gw1=1, c_gw1=4 → t<c → UP-RIGHT → (5,3)
    # coily(5,3) target(0,0): UP, t_gw1=1, c_gw1=3 → t<c → UP-RIGHT → (4,3)
    # coily(4,3) target(0,0): UP, t_gw1=1, c_gw1=2 → t<c → UP-RIGHT → (3,3)
    assert pos == (3, 3), f"coily_n failed: {pos}"
    print("  multi-hop: OK")


def test_ball_path():
    """Test ball path prediction from direction bits."""
    # All right: bits = 0b1111111 = 127
    path = predict_ball_path(0, 0, 127)
    assert path[:3] == [(1, 1), (2, 2), (3, 3)], f"all-right failed: {path}"

    # All left: bits = 0b0000000 = 0
    path = predict_ball_path(0, 0, 0)
    assert path[:3] == [(1, 0), (2, 0), (3, 0)], f"all-left failed: {path}"

    # Alternating: bits = 0b0101010 = 42 → L,R,L,R,L,R,L
    path = predict_ball_path(0, 0, 42)
    assert path[0] == (1, 0), f"alt step 1: {path[0]}"   # bit0=0 → LEFT
    assert path[1] == (2, 1), f"alt step 2: {path[1]}"   # bit1=1 → RIGHT
    assert path[2] == (3, 1), f"alt step 3: {path[2]}"   # bit2=0 → LEFT

    print("  ball paths: OK")


if __name__ == "__main__":
    print("Coily prediction unit tests (ROM-verified algorithm):")
    test_coily_basic()
    test_coily_same_row()
    test_coily_chase_sequence()
    test_coily_n()
    test_ball_path()
    print("\nAll prediction tests PASSED")
