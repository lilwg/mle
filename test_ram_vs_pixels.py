"""RAM vs Pixels ground truth test.

For each frame: read ALL enemy slot RAM, capture the rendered frame,
and verify that every sprite on screen corresponds to an entity we
detected, and every entity we detected has a sprite on screen.

This exposes filtering bugs — if we can't see an enemy that the
renderer draws, our state reader is wrong.
"""

import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mle import MameEnv
from qbert.state import QBERT_RAM, read_state, is_valid, gw_to_pos, EnemyTracker
from qbert.planner import COIN_BUTTON, START_BUTTON

ROMS_PATH = "/Users/pat/mame/roms"

# Screen geometry: Q*bert uses a 240x256 rotated display.
# Grid position (row, col) maps to screen pixel (x, y).
# We'll derive the mapping empirically from Q*bert's known position.

# From ROM/observation: the pyramid top (0,0) is roughly at screen center-top.
# Each row down moves ~16px down, each col right moves ~16px right,
# and each row down also shifts left by ~8px.
# These are approximate — we'll refine by checking actual pixels.

# Q*bert screen coordinates (from ROM: $B9BA-$B9C2 computes screen pos)
# Screen Y = 0x32 * (gw0) = 50 * row_index (roughly)
# Screen X depends on gw1 and side

# Instead of guessing, let's just check for sprite presence at entity
# screen positions by looking for non-background colored pixels.

# Background colors (the black area around the pyramid)
BG_THRESHOLD = 20  # pixels darker than this are background


def capture_frame(env):
    """Capture one frame and return as numpy array (H, W, 3)."""
    env.request_frame()
    data = env.step()
    if "frame" not in data:
        return None, data
    raw = np.frombuffer(data["frame"], dtype=np.uint8)
    bpp = 3 if len(raw) == 256 * 240 * 3 else 4
    h, w = 256, 240
    pixels = raw[:h * w * bpp].reshape(h, w, bpp)
    if bpp == 4:
        pixels = pixels[:, :, 2::-1]  # BGRA -> RGB
    return pixels, data


def read_all_slots(data):
    """Read ALL 10 enemy slots raw — no filtering."""
    slots = []
    for n in range(10):
        st = data.get(f"e{n}_st", 0)
        flags = data.get(f"e{n}_flags", 0)
        anim = data.get(f"e{n}_anim", 0)
        coll_y = data.get(f"e{n}_coll_y", 0)
        gw0 = data.get(f"e{n}_gw0", 0)
        gw1 = data.get(f"e{n}_gw1", 0)
        dir_bits = data.get(f"e{n}_dir", 0)
        pos = gw_to_pos(gw0, gw1)
        slots.append({
            "slot": n, "st": st, "flags": flags, "anim": anim,
            "coll_y": coll_y, "gw0": gw0, "gw1": gw1,
            "pos": pos, "dir": dir_bits,
        })
    return slots


def find_sprites(frame):
    """Find bright sprite regions on the frame.

    Returns list of (center_y, center_x, color) for each detected sprite cluster.
    Sprites are clusters of bright pixels against the dark background.
    """
    # Convert to grayscale brightness
    brightness = frame.astype(float).max(axis=2)

    # Threshold: anything brighter than background
    mask = brightness > BG_THRESHOLD

    # Simple connected component labeling via flood fill
    from scipy import ndimage
    labeled, num_features = ndimage.label(mask)

    sprites = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        area = len(ys)
        if area < 20:  # too small, noise
            continue
        if area > 5000:  # too big, probably the pyramid itself
            continue
        cy, cx = ys.mean(), xs.mean()
        # Average color
        color = frame[ys, xs].mean(axis=0).astype(int)
        sprites.append({
            "y": int(cy), "x": int(cx),
            "area": area, "color": tuple(color),
        })

    return sprites


def sprite_at_region(frame, y_center, x_center, radius=12):
    """Check if there are bright non-background pixels in a region."""
    h, w = frame.shape[:2]
    y0 = max(0, y_center - radius)
    y1 = min(h, y_center + radius)
    x0 = max(0, x_center - radius)
    x1 = min(w, x_center + radius)
    region = frame[y0:y1, x0:x1]
    bright = region.astype(float).max(axis=2) > BG_THRESHOLD
    return bright.sum()


def estimate_screen_pos(gw0, gw1):
    """Estimate screen pixel position from grid words.

    This is approximate — derived from Q*bert's ROM screen coordinate
    calculation at $B9BA. The exact formula involves cumulative sums.
    """
    # Q*bert's isometric projection (approximate):
    # Base position for row 0 is near top-center of screen
    # Each row down: Y += ~24px, X shifts by ~8px for alignment
    # Each col: X += ~16px

    # These are rough — the test will tell us if they're right
    row = gw0 - 1
    col = gw0 - gw1

    # Screen Y: rows go down
    y = 48 + row * 24
    # Screen X: centered, offset by col
    x = 120 - row * 8 + col * 16

    return int(y), int(x)


def run():
    env = MameEnv(ROMS_PATH, "qbert", QBERT_RAM, render=True, sound=False,
                  throttle=True)
    tracker = EnemyTracker()

    # Start game
    env.wait(700)
    env.step_n(*COIN_BUTTON, 15)
    env.wait(180)
    env.step_n(*START_BUTTON, 5)

    # Wait for game to start
    data = env.step()
    state = read_state(data, tracker)
    for _ in range(900):
        if 0 < state.lives <= 5 and is_valid(state.qbert[0], state.qbert[1]):
            break
        data = env.step()
        state = read_state(data, tracker)

    print(f"Game started: lives={state.lives}, Q*bert at {state.qbert}")
    print()

    # Run for N frames, comparing RAM vs pixels
    mismatches = 0
    total_checks = 0

    for frame_num in range(500):
        # Capture frame + read state
        frame, data = capture_frame(env)
        if frame is None:
            continue
        state = read_state(data, tracker)

        if not is_valid(state.qbert[0], state.qbert[1]):
            continue

        # Read ALL enemy slots (no filtering)
        slots = read_all_slots(data)

        # Q*bert screen position
        qb_gw0 = data.get("qb_gw0", 0)
        qb_gw1 = data.get("qb_gw1", 0)
        qb_sy, qb_sx = estimate_screen_pos(qb_gw0, qb_gw1)

        # For each slot: check if entity is visible on screen
        for s in slots:
            if s["flags"] == 0 and s["st"] == 0:
                continue  # truly empty slot

            sy, sx = estimate_screen_pos(s["gw0"], s["gw1"])

            # Is this entity on-screen?
            if sy < 0 or sy >= 256 or sx < 0 or sx >= 240:
                continue

            # Check pixels at estimated position
            pixel_count = sprite_at_region(frame, sy, sx, radius=14)

            # Is this entity in our parsed state.enemies?
            in_state = any(
                e.slot == s["slot"] for e in state.enemies
            )

            total_checks += 1

            if pixel_count > 30 and not in_state:
                # Sprite on screen but NOT in our state — we're missing it!
                mismatches += 1
                pos = s["pos"]
                print(f"  FRAME {frame_num}: MISSING s{s['slot']} "
                      f"fl={s['flags']:#04x} st={s['st']} pos={pos} "
                      f"a={s['anim']} cy={s['coll_y']} "
                      f"pixels={pixel_count} @({sy},{sx})")

                # Save frame for inspection
                if mismatches <= 10:
                    img = Image.fromarray(frame)
                    # Draw a red box at the missing entity location
                    for dy in range(-14, 15):
                        for dx in [-14, 14]:
                            py, px = sy + dy, sx + dx
                            if 0 <= py < 256 and 0 <= px < 240:
                                img.putpixel((px, py), (255, 0, 0))
                    for dx in range(-14, 15):
                        for dy in [-14, 14]:
                            py, px = sy + dy, sx + dx
                            if 0 <= py < 256 and 0 <= px < 240:
                                img.putpixel((px, py), (255, 0, 0))
                    img.save(f"missing_s{s['slot']}_f{frame_num}.png")

            elif pixel_count <= 10 and in_state:
                # In our state but NO sprite on screen — false positive
                pos = s["pos"]
                if is_valid(pos[0], pos[1]):
                    total_checks += 1  # only count on-grid false positives
                    print(f"  FRAME {frame_num}: GHOST s{s['slot']} "
                          f"fl={s['flags']:#04x} pos={pos} "
                          f"a={s['anim']} cy={s['coll_y']} "
                          f"pixels={pixel_count} — in state but no sprite!")

        # Every 100 frames, print status
        if frame_num % 100 == 0 and frame_num > 0:
            print(f"  ... frame {frame_num}: {mismatches} mismatches "
                  f"/ {total_checks} checks")

    print(f"\nDone: {mismatches} missing entities / {total_checks} checks")
    print(f"Missed detection rate: {mismatches/max(1,total_checks)*100:.1f}%")
    env.close()


if __name__ == "__main__":
    run()
