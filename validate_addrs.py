"""Validate score/lives RAM addresses by playing and showing values.

Plays the game with random inputs while printing the RAM values
alongside a screenshot so you can visually confirm they match
what's on screen.

Usage:
    python3 validate_addrs.py qbert                  # uses game_configs.json
    python3 validate_addrs.py dkong --score 0x6007,0x6008,0x6009 --lives 0x6001
"""

import sys
import os
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mle import MameEnv
from mle_general import (
    ROMS_PATH, COIN_BUTTONS, START_BUTTONS, ALT_DIRECTIONS,
    discover_inputs,
)


def validate(game_id, score_addrs, lives_addr):
    # Build RAM dict
    ram = {"_dummy": 0x0000}
    for i, addr in enumerate(score_addrs):
        ram[f"score{i}"] = addr
    if lives_addr is not None:
        ram["lives"] = lives_addr

    env = MameEnv(ROMS_PATH, game_id, ram, render=True, sound=False, throttle=True)

    # Discover inputs at runtime
    result = env.console.writeln_expect(
        'local r = {}; '
        'for pname, port in pairs(manager.machine.ioport.ports) do '
        'for fname, field in pairs(port.fields) do '
        'r[#r+1] = pname.."|"..fname; '
        'end; end; '
        'print(table.concat(r, ";"))'
    )
    coin, start, actions = None, None, []
    if result:
        for pair in result.split(";"):
            if "|" not in pair:
                continue
            port, field = pair.split("|", 1)
            fl = field.lower()
            if "coin" in fl:
                coin = (port, field)
            elif "1 player" in fl or ("start" in fl and "1" in fl):
                start = (port, field)
            elif "p1" in fl and any(d in fl for d in ["up", "down", "left", "right", "button"]):
                actions.append((port, field))

    # Insert coin + start
    if coin:
        env.step_n(*coin, 15)
    env.wait(120)
    if start:
        env.step_n(*start, 5)
    env.wait(180)

    print(f"\n{'='*60}")
    print(f"Validating {game_id}")
    print(f"Score addresses: {[f'${a:04X}' for a in score_addrs]}")
    print(f"Lives address: {f'${lives_addr:04X}' if lives_addr else 'None'}")
    print(f"Actions: {[f[1] for f in actions]}")
    print(f"{'='*60}")
    print(f"Watch the game window. Values below should match the screen.")
    print(f"Press Ctrl+C to stop.\n")

    import random
    step = 0
    try:
        while True:
            # Random input
            if actions and random.random() < 0.7:
                port, field = random.choice(actions)
                data = env.step(port, field)
            else:
                data = env.step()
            step += 1

            if step % 30 == 0:  # print every 30 frames (~0.5 sec)
                # Read score bytes
                score_vals = []
                for i in range(len(score_addrs)):
                    v = data.get(f"score{i}", 0)
                    score_vals.append(v)

                # Decode as both raw and BCD
                raw_total = sum(v << (8 * i) for i, v in enumerate(score_vals))
                bcd_str = ""
                for v in reversed(score_vals):
                    bcd_str += f"{(v >> 4) & 0xF}{v & 0xF}"
                bcd_str = bcd_str.lstrip("0") or "0"

                lives_val = data.get("lives", "?")

                print(f"  step {step:5d} | "
                      f"score raw={score_vals} hex={raw_total:#06x} bcd={bcd_str} | "
                      f"lives={lives_val}")

    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate score/lives RAM addresses")
    parser.add_argument("game", help="MAME ROM name")
    parser.add_argument("--score", type=str, default=None,
                        help="Score addresses, comma-separated hex")
    parser.add_argument("--lives", type=str, default=None,
                        help="Lives address, hex")
    args = parser.parse_args()

    score_addrs = []
    lives_addr = None

    # Manual override
    if args.score:
        score_addrs = [int(a.strip(), 16) for a in args.score.split(",")]
    if args.lives:
        lives_addr = int(args.lives, 16)

    # Load from config if not specified
    if not score_addrs:
        config_path = os.path.join(os.path.dirname(__file__), "game_configs.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                configs = json.load(f)
            if args.game in configs:
                cfg = configs[args.game]
                score_addrs = [int(a, 16) for a in cfg.get("score_addrs", [])]
                if lives_addr is None and "lives_addr" in cfg:
                    lives_addr = int(cfg["lives_addr"], 16)
                print(f"Loaded config for {args.game}")

    if not score_addrs:
        print(f"No score addresses for {args.game}. Use --score or add to game_configs.json")
        sys.exit(1)

    validate(args.game, score_addrs, lives_addr)
