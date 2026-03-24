"""Capture a game frame and annotate it with entity state from RAM."""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mle import MameEnv
from qbert.state import QBERT_RAM, read_state, is_valid, gw_to_pos, EnemyTracker
from qbert.planner import COIN_BUTTON, START_BUTTON


ROMS_PATH = "/Users/pat/mame/roms"

# Add screen position high bytes (the actual pixel coords used by sprite hardware)
QBERT_RAM["qb_syh"] = 0x0D59  # Q*bert screen Y high byte
QBERT_RAM["qb_sxh"] = 0x0D5B  # Q*bert screen X high byte
for _n in range(10):
    _base = 0x0D70 + _n * 22
    QBERT_RAM[f"e{_n}_syh"] = _base + 5
    QBERT_RAM[f"e{_n}_sxh"] = _base + 7


# Entity type colors for boxes
COLORS = {
    "qbert": (255, 255, 0),    # yellow
    "coily": (255, 0, 255),    # magenta
    "ball": (255, 0, 0),       # red
    "sam": (0, 255, 0),        # green
    "ugg": (255, 128, 0),      # orange
}


def hw_to_screen(hw_y_hi, hw_x_hi):
    """Convert hardware sprite coords (high bytes) to MAME screen pixels.
    Q*bert display is rotated 90°: screen_y = 239 - hw_x, screen_x = hw_y."""
    return 239 - hw_x_hi, hw_y_hi


def capture_frame(env):
    env.request_frame()
    data = env.step()
    if "frame" not in data:
        return None, data
    raw = np.frombuffer(data["frame"], dtype=np.uint8)
    bpp = 3 if len(raw) == 256 * 240 * 3 else 4
    pixels = raw[:256 * 240 * bpp].reshape(256, 240, bpp)
    if bpp == 4:
        pixels = pixels[:, :, 2::-1]
    return pixels, data


def annotate(frame, data, state):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    r = 12

    # Discs — mark with text at estimated edge positions
    # (discs don't have entity RAM, just note their logical position)
    for disc in state.discs:
        # Approximate: disc is at the pyramid edge near its row
        # Use a simple formula since we don't have exact screen coords
        row = disc.row
        if disc.side == "left":
            dx, dy = 8, 48 + row * 24
        else:
            dx, dy = 232, 48 + row * 24
        draw.text((dx, dy), f"DISC r{disc.row}", fill=(0, 255, 255))

    # Q*bert — use hardware screen coordinates
    qy, qx = hw_to_screen(data.get("qb_syh", 0), data.get("qb_sxh", 0))
    draw.rectangle([qx - r, qy - r, qx + r, qy + r], outline=(255, 255, 0), width=2)
    draw.text((qx + r + 2, qy - 5), f"QB {state.qbert}", fill=(255, 255, 0))

    # All enemy slots — read raw, no filtering
    for n in range(10):
        st = data.get(f"e{n}_st", 0)
        flags = data.get(f"e{n}_flags", 0)
        if flags == 0 and st == 0:
            continue

        anim = data.get(f"e{n}_anim", 0)
        coll_y = data.get(f"e{n}_coll_y", 0)
        gw0 = data.get(f"e{n}_gw0", 0)
        gw1 = data.get(f"e{n}_gw1", 0)
        pos = gw_to_pos(gw0, gw1)
        syh = data.get(f"e{n}_syh", 0)
        sxh = data.get(f"e{n}_sxh", 0)
        sy, sx = hw_to_screen(syh, sxh)

        if sy < 0 or sy >= 256 or sx < 0 or sx >= 240:
            continue

        # Find this entity in parsed state
        parsed = None
        for e in state.enemies:
            if e.slot == n:
                parsed = e
                break

        if parsed:
            etype = parsed.etype
            color = COLORS.get(etype, (200, 200, 200))
            harm = "" if not parsed.harmless else " OK"
            label = f"s{n}:{etype}{harm}"
        else:
            color = (128, 128, 128)  # gray = not in parsed state
            label = f"s{n}:UNSEEN"

        draw.rectangle([sx - r, sy - r, sx + r, sy + r], outline=color, width=2)
        detail = f"fl={flags:#x} a={anim} cy={coll_y}"
        draw.text((sx - r, sy + r + 1), label, fill=color)
        draw.text((sx - r, sy + r + 11), detail, fill=color)

    return img


def run():
    env = MameEnv(ROMS_PATH, "qbert", QBERT_RAM, render=True, sound=False,
                  throttle=True)
    tracker = EnemyTracker()

    env.wait(700)
    env.step_n(*COIN_BUTTON, 15)
    env.wait(180)
    env.step_n(*START_BUTTON, 5)

    data = env.step()
    state = read_state(data, tracker)
    for _ in range(900):
        if 0 < state.lives <= 5 and is_valid(state.qbert[0], state.qbert[1]):
            break
        data = env.step()
        state = read_state(data, tracker)

    print(f"Game started: Q*bert at {state.qbert}")

    # Play the game with random inputs to get enemies on screen
    import random
    from qbert.planner import MOVE_BUTTONS
    moves = list(MOVE_BUTTONS.values())
    for _ in range(400):
        data = env.step()

    # Capture and annotate several frames spread over gameplay
    for i in range(10):
        # Make some random hops
        port, field = random.choice(moves)
        env.step_n(port, field, 6)
        for _ in range(60):
            data = env.step()

        frame, data = capture_frame(env)
        if frame is None:
            continue
        state = read_state(data, tracker)

        img = annotate(frame, data, state)
        fname = f"annotated_{i:02d}.png"
        img.save(fname)
        print(f"  Saved {fname}: Q*bert={state.qbert} "
              f"enemies={len(state.enemies)}")

    env.close()
    print("Done. Check annotated_*.png files.")


if __name__ == "__main__":
    run()
