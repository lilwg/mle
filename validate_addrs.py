"""Find and validate score/lives RAM addresses while you play.

  python3 validate_addrs.py galaga           # find addresses
  python3 validate_addrs.py qbert --validate # verify known addresses

FIND mode: one MAME session, you play, type your score to capture.
Reads ALL RAM via Lua console at capture time (not through the pipe).
"""

import sys
import os
import json
import argparse
import threading
import time
import subprocess as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mle import MameEnv

ROMS_PATH = "/Users/pat/mame/roms"


def get_ram_ranges(game_id):
    import xml.etree.ElementTree as ET
    try:
        result = sp.run(["/opt/homebrew/bin/mame", game_id, "-listxml"],
                        capture_output=True, text=True, timeout=10)
        tree = ET.fromstring(result.stdout)
        cpus = []
        for m in tree.findall('.//machine'):
            if m.get('name') == game_id:
                cpus = [c.get('name', '').lower() for c in m.findall('.//chip')
                        if c.get('type') == 'cpu']
                break
    except Exception:
        cpus = []
    cpu_str = ' '.join(cpus)
    if '8088' in cpu_str or '8086' in cpu_str:
        return [(0x0000, 0x2000)]
    elif 'z80' in cpu_str:
        return [(0x0000, 0x1000), (0x4000, 0x5000),
                (0x6000, 0x7000), (0x8000, 0xA000), (0xC000, 0xD000)]
    else:
        return [(0x0000, 0x2000), (0x8000, 0xA000)]


def bulk_read_ram(env, ranges):
    """Read all RAM in ranges via Lua console. Returns {addr: value}."""
    ram = {}
    for start, end in ranges:
        for chunk in range(start, end, 256):
            chunk_end = min(chunk + 256, end)
            count = chunk_end - chunk
            r = env.console.writeln_expect(
                f'local t={{}};for a={chunk},{chunk_end-1} do '
                f't[#t+1]=mem:read_u8(a) end;print(table.concat(t,","))')
            if r:
                try:
                    vals = r.split(',')
                    for i, v in enumerate(vals[:count]):
                        ram[chunk + i] = int(v.strip())
                except (ValueError, IndexError):
                    pass
    return ram


def byte_matches_score(val, score_val):
    if val == score_val and score_val <= 255:
        return "raw"
    if (val >> 4) <= 9 and (val & 0xF) <= 9:
        bcd = ((val >> 4) * 10) + (val & 0xF)
        if bcd == score_val and score_val <= 99:
            return "BCD"
    if score_val <= 0xFFFF and (score_val & 0xFF) == val:
        return "16bit-lo"
    if score_val > 255 and score_val <= 0xFFFF and ((score_val >> 8) & 0xFF) == val:
        return "16bit-hi"
    return None


def find_mode(game_id):
    ranges = get_ram_ranges(game_id)
    total = sum(e - s for s, e in ranges)

    print(f"[{game_id}] FIND MODE — will scan {total} RAM bytes")
    print(f"  One game session. You play, type your score to capture.")
    print(f"  Enter 3+ different scores, then type 'done'.\n")

    sp.run(["pkill", "-9", "-f", f"mame.*{game_id}"], capture_output=True)
    time.sleep(0.5)

    # Minimal pipe — just keep MAME in sync
    env = MameEnv(ROMS_PATH, game_id, {"_dummy": 0},
                  render=True, sound=False, throttle=True)

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

    print(f"  Game starting. Press 5=coin, 1=start.\n")

    samples = []  # (score_val, {addr: byte_val})

    while True:
        user = input(f"  Score #{len(samples)+1} (or 'done'): ").strip()
        if user.lower() == 'done':
            break
        if user.lower().startswith('q'):
            running[0] = False
            env.close()
            return
        try:
            score_val = int(user)
        except ValueError:
            print("    Enter a number or 'done'")
            continue

        # Read ALL RAM via console while pump keeps running
        # (MAME needs the pipe protocol active to process console commands)
        print(f"    Reading {total} RAM bytes...", end=" ", flush=True)
        ram = bulk_read_ram(env, ranges)
        print(f"got {len(ram)}")
        samples.append((score_val, ram))

    running[0] = False
    env.close()
    sp.run(["pkill", "-9", "-f", f"mame.*{game_id}"], capture_output=True)

    if len(samples) < 2 or len(set(s for s, _ in samples)) < 2:
        print("\nNeed at least 2 captures with different scores.")
        return

    # Find addresses matching ALL scores
    print(f"\n{'='*60}")
    print(f"Analyzing {len(samples)} captures...")

    all_addrs = set()
    for _, ram in samples:
        all_addrs.update(ram.keys())

    consistent = {}
    for addr in sorted(all_addrs):
        matches = []
        for score_val, ram in samples:
            if addr not in ram:
                break
            enc = byte_matches_score(ram[addr], score_val)
            if enc:
                matches.append((score_val, ram[addr], enc))
        if len(matches) == len(samples) and len(set(m[1] for m in matches)) >= 2:
            consistent[addr] = matches

    if consistent:
        print(f"\nAddresses matching all {len(samples)} scores:")
        for addr, matches in sorted(consistent.items()):
            scores_str = " ".join(f"{s}→{v}" for s, v, _ in matches)
            print(f"  ${addr:04X}: {matches[0][2]} [{scores_str}]")

        best = sorted(consistent.keys())[:4]
        result = {"score_addrs": [f"0x{a:04X}" for a in best], "verified": True}

        config_path = os.path.join(os.path.dirname(__file__), "game_configs.json")
        configs = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                configs = json.load(f)
        if game_id in configs and "lives_addr" in configs[game_id]:
            result["lives_addr"] = configs[game_id]["lives_addr"]
        configs[game_id] = result
        with open(config_path, 'w') as f:
            json.dump(configs, f, indent=4)
        print(f"\nSaved to game_configs.json")
    else:
        print("\nNo consistent addresses found.")
        print("Make sure you entered the exact score shown on screen.")


def validate_mode(game_id):
    config_path = os.path.join(os.path.dirname(__file__), "game_configs.json")
    if not os.path.exists(config_path):
        print("No game_configs.json")
        return
    with open(config_path) as f:
        configs = json.load(f)
    if game_id not in configs:
        print(f"No config for {game_id}. Run without --validate to find.")
        return
    cfg = configs[game_id]
    score_addrs = [int(a, 16) for a in cfg.get("score_addrs", [])]
    lives_addr = int(cfg["lives_addr"], 16) if "lives_addr" in cfg else None
    encoding = cfg.get("encoding", "raw")

    ram = {}
    for i, addr in enumerate(score_addrs):
        ram[f"s{i}"] = addr
    if lives_addr:
        ram["lives"] = lives_addr

    sp.run(["pkill", "-9", "-f", f"mame.*{game_id}"], capture_output=True)
    time.sleep(0.5)
    print(f"[{game_id}] VALIDATE")
    if score_addrs:
        print(f"  Score: {[f'${a:04X}' for a in score_addrs]} ({encoding})")
    if lives_addr:
        print(f"  Lives: ${lives_addr:04X}")
    print(f"  5=coin, 1=start. Ctrl+C to quit.\n")

    env = MameEnv(ROMS_PATH, game_id, ram, render=True, sound=False, throttle=True)
    latest = [None]
    running = [True]
    def pump():
        while running[0]:
            try:
                latest[0] = env.step()
            except Exception:
                break
    threading.Thread(target=pump, daemon=True).start()

    prev = None
    try:
        while True:
            time.sleep(0.5)
            data = latest[0]
            if not data:
                continue
            vals = [data.get(f"s{i}", 0) for i in range(len(score_addrs))]
            if encoding == "tile":
                score = 0
                for v in vals:
                    score = score * 10 + (v if 0 <= v <= 9 else 0)
            else:
                score = sum(v << (8 * i) for i, v in enumerate(vals))
            lives = data.get("lives", "?")
            cur = (score, lives)
            if cur != prev:
                print(f"  score={score:>8d}  lives={lives}  raw={vals}")
                prev = cur
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        running[0] = False
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        validate_mode(args.game)
    else:
        find_mode(args.game)
