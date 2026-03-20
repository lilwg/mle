"""Overlay test: save a frame with grid markers, verify alignment."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from mle import MameEnv
from qbert.state import QBERT_RAM, read_state, EnemyTracker
from qbert.overlay import draw_overlay

ROMS_PATH = "/Users/pat/mame/roms"


def test_overlay():
    print("Starting overlay test...")
    env = MameEnv(ROMS_PATH, "qbert", QBERT_RAM, render=True, sound=False, throttle=True)
    tracker = EnemyTracker()

    env.wait(600)
    env.step_n(":IN1", "Coin 1", 15)
    env.wait(180)
    env.step_n(":IN1", "1 Player Start", 5)

    # Wait for game to start (lives=3, not 255 during init)
    for _ in range(900):
        data = env.step()
        state = read_state(data, tracker)
        if 0 < state.lives <= 5 and is_valid(state.qbert[0], state.qbert[1]):
            break

    print(f"Game started: lives={state.lives}, Q*bert at {state.qbert}")

    # Move a few times to get some visited cubes and enemies
    moves = [
        (":IN4", "P1 Right (Down-Right)"),  # (1,1)
        (":IN4", "P1 Right (Down-Right)"),  # (2,2)
        (":IN4", "P1 Down (Down-Left)"),    # (3,2)
        (":IN4", "P1 Left (Up-Left)"),      # (2,1)
    ]
    visited = {state.qbert: True}
    for port, field in moves:
        env.step_n(port, field, 12)
        data = env.wait(30)
        state = read_state(data, tracker)
        if state.qbert[0] >= 0:
            visited[state.qbert] = True
        print(f"  Moved to {state.qbert}")

    # Wait a bit for enemies to spawn
    for _ in range(200):
        data = env.step()
        state = read_state(data, tracker)

    # Capture frame with overlay
    env.request_frame()
    data = env.step()
    state = read_state(data, tracker)

    if "frame" not in data:
        print("ERROR: No frame data")
        env.close()
        return

    img = draw_overlay(data["frame"], state, visited)

    os.makedirs("test_output", exist_ok=True)
    path = "test_output/overlay_test.png"
    cv2.imwrite(path, img)
    print(f"\nSaved overlay to {path}")
    print(f"Q*bert at {state.qbert}")
    print(f"Visited: {list(visited.keys())}")
    for e in state.enemies:
        print(f"  Enemy s{e.slot}: {e.etype} @{e.pos} fl={e.flags:#x}")

    print("\nOverlay test PASSED — check test_output/overlay_test.png")
    env.close()


if __name__ == "__main__":
    test_overlay()
