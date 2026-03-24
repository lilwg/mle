"""Calibrate grid-to-screen mapping by marking test points on a captured frame."""

import sys, os, numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mle import MameEnv
from qbert.state import QBERT_RAM, read_state, is_valid, gw_to_pos, pos_to_gw, EnemyTracker
from qbert.planner import COIN_BUTTON, START_BUTTON

ROMS_PATH = "/Users/pat/mame/roms"

def capture_frame(env):
    env.request_frame()
    data = env.step()
    if "frame" not in data:
        return None, data
    raw = np.frombuffer(data["frame"], dtype=np.uint8)
    bpp = 3 if len(raw) == 256 * 240 * 3 else 4
    pixels = raw[:256*240*bpp].reshape(256, 240, bpp)
    if bpp == 4:
        pixels = pixels[:,:,2::-1]
    return pixels, data

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

    # Wait a bit for enemies to appear
    for _ in range(200):
        data = env.step()

    frame, data = capture_frame(env)
    state = read_state(data, tracker)
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Mark ALL 28 cube positions with colored dots using different formulas
    # to find which one matches the actual cube locations on screen

    # The Q*bert screen is 240 wide x 256 tall (rotated from original 256x240)
    # Formula candidates — mark each with a different color

    for row in range(7):
        for col in range(row + 1):
            gw0, gw1 = pos_to_gw(row, col)

            # Formula A: linear mapping (our old guess)
            ya = 48 + row * 24
            xa = 120 - row * 8 + col * 16

            # Formula B: based on ROM cumulative sum, scaled
            yb = 20 + gw0 * 22
            xb = 120 + int((col - row/2) * 16)

            # Formula C: different scale
            yc = 32 + row * 25
            xc = 120 + (2*col - row) * 8

            # Draw small crosses at each candidate
            for dx in range(-2, 3):
                if 0 <= xa+dx < 240 and 0 <= ya < 256:
                    img.putpixel((xa+dx, ya), (255, 0, 0))      # Red = formula A
                if 0 <= ya+dx < 256 and 0 <= xa < 240:
                    img.putpixel((xa, ya+dx), (255, 0, 0))

                if 0 <= xb+dx < 240 and 0 <= yb < 256:
                    img.putpixel((xb+dx, yb), (0, 255, 0))      # Green = formula B
                if 0 <= yb+dx < 256 and 0 <= xb < 240:
                    img.putpixel((xb, yb+dx), (0, 255, 0))

                if 0 <= xc+dx < 240 and 0 <= yc < 256:
                    img.putpixel((xc+dx, yc), (0, 128, 255))    # Blue = formula C
                if 0 <= yc+dx < 256 and 0 <= xc < 240:
                    img.putpixel((xc, yc+dx), (0, 128, 255))

    # Also mark Q*bert's position
    qr, qc = state.qbert
    print(f"Q*bert at grid ({qr},{qc}), gw=({data.get('qb_gw0',0)},{data.get('qb_gw1',0)})")
    print(f"Enemies: {len(state.enemies)}")
    for e in state.enemies:
        print(f"  s{e.slot}: {e.etype} @ {e.pos} fl={e.flags:#x}")

    img.save("calibrate.png")
    print("Saved calibrate.png — Red=A, Green=B, Blue=C")
    env.close()

if __name__ == "__main__":
    run()
