"""Validate score/lives RAM addresses by showing values while you play.

Opens the game, you play normally, and the terminal prints the current
score and lives values every second. Verify they match what's on screen.

Usage:
    python3 validate_addrs.py qbert
    python3 validate_addrs.py galaga
    python3 validate_addrs.py dkong
"""

import sys
import os
import json
import argparse
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mle import MameEnv

ROMS_PATH = "/Users/pat/mame/roms"


def main():
    parser = argparse.ArgumentParser(description="Validate score/lives by playing")
    parser.add_argument("game", help="MAME ROM name")
    parser.add_argument("--score", type=str, default=None)
    parser.add_argument("--lives", type=str, default=None)
    args = parser.parse_args()

    # Load addresses
    score_addrs, lives_addr, encoding = [], None, "raw"
    if args.score:
        score_addrs = [int(a.strip(), 16) for a in args.score.split(",")]
    if args.lives:
        lives_addr = int(args.lives, 16)

    if not score_addrs:
        config_path = os.path.join(os.path.dirname(__file__), "game_configs.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                configs = json.load(f)
            if args.game in configs:
                cfg = configs[args.game]
                score_addrs = [int(a, 16) for a in cfg.get("score_addrs", [])]
                if not lives_addr and "lives_addr" in cfg:
                    lives_addr = int(cfg["lives_addr"], 16)
                encoding = cfg.get("encoding", "raw")

    if not score_addrs:
        print(f"No addresses for {args.game}. Add to game_configs.json or use --score")
        return

    # Build RAM dict
    ram = {}
    for i, addr in enumerate(score_addrs):
        ram[f"s{i}"] = addr
    if lives_addr:
        ram["lives"] = lives_addr

    print(f"[{args.game}] Starting MAME...")
    print(f"  Score: {[f'${a:04X}' for a in score_addrs]} ({encoding})")
    if lives_addr:
        print(f"  Lives: ${lives_addr:04X}")
    print(f"\n  Press 5 = coin, 1 = start. Play normally.")
    print(f"  Values below should match the screen. Ctrl+C to quit.\n")

    env = MameEnv(ROMS_PATH, args.game, ram,
                  render=True, sound=True, throttle=True)

    # Pump thread to keep MAME running + read latest data
    latest = [None]
    running = [True]

    def pump():
        while running[0]:
            try:
                latest[0] = env.step()
            except Exception:
                break

    t = threading.Thread(target=pump, daemon=True)
    t.start()

    # Print values every second
    import time
    prev_score = None
    try:
        while True:
            time.sleep(1)
            data = latest[0]
            if data is None:
                continue

            # Decode score
            vals = [data.get(f"s{i}", 0) for i in range(len(score_addrs))]
            if encoding == "tile":
                score = 0
                for v in vals:
                    score = score * 10 + (v if 0 <= v <= 9 else 0)
            else:
                score = sum(v << (8 * i) for i, v in enumerate(vals))

            lives = data.get("lives", "?")

            # Only print when something changes (less noise)
            if score != prev_score:
                print(f"  score={score:>8d}  lives={lives}  raw={vals}")
                prev_score = score

    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        running[0] = False
        env.close()


if __name__ == "__main__":
    main()
