"""MLE smoke test: start Q*bert, insert coin, verify lives = 3."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mle import MameEnv

ROMS_PATH = "/Users/pat/mame/roms"
RAM = {"lives": 0x0D00, "score": 0x00BE, "qb_gw0": 0x0D66, "qb_gw1": 0x0D67}


def test_smoke():
    print("Starting MLE smoke test...")
    env = MameEnv(ROMS_PATH, "qbert", RAM, render=True, sound=False, throttle=True)

    print("Waiting for boot...")
    env.wait(600)

    print("Inserting coin...")
    env.step_n(":IN1", "Coin 1", 15)
    env.wait(180)

    print("Pressing start...")
    env.step_n(":IN1", "1 Player Start", 5)
    data = env.wait(500)

    lives = data.get("lives", -1)
    print(f"Lives = {lives} (expected 3)")
    assert lives == 3, f"Expected 3 lives, got {lives}"

    gw0 = data.get("qb_gw0", 0)
    gw1 = data.get("qb_gw1", 0)
    row, col = gw0 - 1, gw0 - gw1
    print(f"Q*bert at grid ({row}, {col}), gw=({gw0}, {gw1})")
    assert (row, col) == (0, 0), f"Expected Q*bert at (0,0), got ({row},{col})"

    # Try a move
    print("Moving down-right...")
    env.step_n(":IN4", "P1 Right (Down-Right)", 6)
    data = env.wait(40)
    gw0 = data.get("qb_gw0", 0)
    gw1 = data.get("qb_gw1", 0)
    row, col = gw0 - 1, gw0 - gw1
    print(f"Q*bert now at ({row}, {col}), gw=({gw0}, {gw1})")

    # Test frame capture
    print("Requesting frame...")
    env.request_frame()
    data = env.step()
    has_frame = "frame" in data
    if has_frame:
        print(f"Frame shape: {data['frame'].shape}")
    else:
        print("WARNING: No frame data received")

    print("\nSmoke test PASSED!")
    env.close()


if __name__ == "__main__":
    test_smoke()
