"""Debug script: capture frames and verify RAM state matches pixels."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from mle import MameEnv
from qbert.state import QBERT_RAM, read_state, is_valid, EnemyTracker
from qbert.overlay import grid_to_pixel

ROMS_PATH = "/Users/pat/mame/roms"


def annotate_frame(frame, state):
    """Draw state info directly on the frame."""
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (720, 768), interpolation=cv2.INTER_NEAREST)
    s = 3  # scale factor

    # Q*bert position
    qr, qc = state.qbert
    if is_valid(qr, qc):
        px, py = grid_to_pixel(qr, qc)
        cv2.circle(img, (px * s, py * s), 8 * s, (0, 255, 0), 2)
        cv2.putText(img, f"Q({qr},{qc})", (px * s - 15 * s, py * s - 10 * s),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * s, (0, 255, 0), s)

    # Enemies
    for e in state.enemies:
        if not is_valid(e.pos[0], e.pos[1]):
            continue
        px, py = grid_to_pixel(e.pos[0], e.pos[1])
        if e.etype == "coily":
            color = (0, 0, 255)
            label = f"C{e.slot}"
        elif e.harmless:
            color = (0, 255, 255)
            label = f"S{e.slot}"
        else:
            color = (255, 100, 0)
            label = f"B{e.slot}"
        cv2.circle(img, (px * s, py * s), 6 * s, color, 2)
        cv2.putText(img, f"{label}({e.pos[0]},{e.pos[1]})",
                    (px * s - 15 * s, py * s - 10 * s),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35 * s, color, s)
        # Show st and flags
        cv2.putText(img, f"st={e.state} fl={e.flags:#x} a={e.anim}",
                    (px * s - 20 * s, py * s + 12 * s),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3 * s, color, 1)

    # Info bar
    y = 30
    info_lines = [
        f"Lives={state.lives}  Remaining={state.remaining_cubes}  "
        f"Target={state.target_color}  Spawn={state.spawn_countdown}",
        f"Q*bert=({qr},{qc}) prev={state.qbert_prev}",
    ]
    for e in state.enemies:
        info_lines.append(
            f"  s{e.slot}: {e.etype} @{e.pos} prev={e.prev_pos} "
            f"st={e.state} fl={e.flags:#x} anim={e.anim} dir={e.direction_bits:#x}"
        )
    # Ghost enemies (st=0 but included)
    for e in state.enemies:
        if e.state == 0:
            info_lines.append(f"  ^^ GHOST (st=0)")

    for line in info_lines:
        cv2.putText(img, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 18

    return img


def run_debug():
    env = MameEnv(ROMS_PATH, "qbert", QBERT_RAM, render=True, sound=False,
                  throttle=True)
    tracker = EnemyTracker()
    os.makedirs("debug_frames", exist_ok=True)

    print("Booting...")
    env.wait(700)

    # Coin + start
    env.step_n(":IN1", "Coin 1", 15)
    env.wait(180)
    env.step_n(":IN1", "1 Player Start", 5)

    # Wait for game to start
    for _ in range(900):
        data = env.step()
        state = read_state(data, tracker)
        if 0 < state.lives <= 5 and is_valid(state.qbert[0], state.qbert[1]):
            break

    print(f"Game started: lives={state.lives}, Q*bert at {state.qbert}")

    # Capture initial frame
    env.request_frame()
    data = env.step()
    state = read_state(data, tracker)
    if "frame" in data:
        img = annotate_frame(data["frame"], state)
        cv2.imwrite("debug_frames/00_start.png", img)
        print("Saved debug_frames/00_start.png")

    # Make some moves and capture after each
    moves = [
        (":IN4", "P1 Right (Down-Right)", "dn-rt"),
        (":IN4", "P1 Right (Down-Right)", "dn-rt"),
        (":IN4", "P1 Down (Down-Left)", "dn-lt"),
        (":IN4", "P1 Down (Down-Left)", "dn-lt"),
        (":IN4", "P1 Right (Down-Right)", "dn-rt"),
        (":IN4", "P1 Right (Down-Right)", "dn-rt"),
        (":IN4", "P1 Left (Up-Left)", "up-lt"),
        (":IN4", "P1 Left (Up-Left)", "up-lt"),
        (":IN4", "P1 Down (Down-Left)", "dn-lt"),
        (":IN4", "P1 Down (Down-Left)", "dn-lt"),
        (":IN4", "P1 Right (Down-Right)", "dn-rt"),
        (":IN4", "P1 Right (Down-Right)", "dn-rt"),
        (":IN4", "P1 Down (Down-Left)", "dn-lt"),
        (":IN4", "P1 Down (Down-Left)", "dn-lt"),
        (":IN4", "P1 Down (Down-Left)", "dn-lt"),
        (":IN4", "P1 Down (Down-Left)", "dn-lt"),
        (":IN4", "P1 Down (Down-Left)", "dn-lt"),
        (":IN4", "P1 Down (Down-Left)", "dn-lt"),
        (":IN4", "P1 Right (Down-Right)", "dn-rt"),
        (":IN4", "P1 Right (Down-Right)", "dn-rt"),
    ]

    for i, (port, field, name) in enumerate(moves):
        # Execute hop
        env.step_n(port, field, 6)
        # Wait for hop to complete
        gw_before = (data.get("qb_gw0", 0), data.get("qb_gw1", 0))
        for _ in range(20):
            data = env.step()
            if (data.get("qb_gw0", 0), data.get("qb_gw1", 0)) != gw_before:
                break
        # Read a few more frames for enemy data
        for _ in range(12):
            data = env.step()

        state = read_state(data, tracker)

        # Capture frame
        env.request_frame()
        data = env.step()
        state = read_state(data, tracker)

        if "frame" in data:
            img = annotate_frame(data["frame"], state)
            fname = f"debug_frames/{i+1:02d}_{name}.png"
            cv2.imwrite(fname, img)
            print(f"  #{i+1} {name} → {state.qbert}  saved {fname}")
            # Print all enemy details
            for e in state.enemies:
                ghost = " GHOST" if e.state == 0 else ""
                print(f"    s{e.slot}: {e.etype} @{e.pos} prev={e.prev_pos} "
                      f"st={e.state} fl={e.flags:#x} anim={e.anim} "
                      f"dir={e.direction_bits:#x}{ghost}")
        else:
            print(f"  #{i+1} {name} → {state.qbert} (no frame)")

    print("\nDone! Check debug_frames/ directory.")
    env.close()


if __name__ == "__main__":
    run_debug()
