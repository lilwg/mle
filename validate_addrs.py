"""Find and validate score/lives RAM addresses while you play.

  python3 validate_addrs.py galaga           # find addresses
  python3 validate_addrs.py qbert --validate # verify known addresses
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


def byte_matches_score(val, score_val):
    if val == score_val and score_val <= 255:
        return "raw"
    if (val >> 4) <= 9 and (val & 0xF) <= 9:
        bcd = ((val >> 4) * 10) + (val & 0xF)
        if bcd == score_val and score_val <= 99:
            return "BCD"
        # BCD low pair: e.g. score 1370 → low BCD byte = 0x70 (=70)
        if bcd == score_val % 100:
            return "BCD-lo"
        # BCD high pair: e.g. score 1370 → high BCD byte = 0x13 (=13)
        if bcd == score_val // 100 and score_val >= 100:
            return "BCD-hi"
    if score_val <= 0xFFFF and (score_val & 0xFF) == val:
        return "16bit-lo"
    if score_val > 255 and score_val <= 0xFFFF and ((score_val >> 8) & 0xFF) == val:
        return "16bit-hi"
    # Score divided by 10 (some games store score/10)
    if score_val >= 10 and val == score_val // 10 and score_val // 10 <= 255:
        return "raw/10"
    return None


def find_mode(game_id):
    ranges = get_ram_ranges(game_id)

    # Put ALL addresses in the pipe — tested: 24K works fine
    ram_dict = {}
    for rs, re in ranges:
        for a in range(rs, re):
            ram_dict[f"r{a:04x}"] = a
    total = len(ram_dict)

    print(f"[{game_id}] FIND MODE — reading {total} RAM bytes every frame")
    print(f"  One game. Play, type score 3x, type 'done'.\n")

    sp.run(["pkill", "-9", "-f", f"mame.*{game_id}"], capture_output=True)
    time.sleep(0.5)

    env = MameEnv(ROMS_PATH, game_id, ram_dict,
                  render=True, sound=True, throttle=True)

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

    print(f"  5=coin, 1=start. Play and type your score.\n")

    samples = []
    while True:
        user = input(f"  Score #{len(samples)+1} (or 'done'): ").strip()
        if user.lower() == 'done':
            break
        try:
            score_val = int(user)
        except ValueError:
            print("    Enter a number or 'done'")
            continue

        # Just grab latest data — pump is always running, data is always fresh
        data = latest[0]
        if data:
            ram = {}
            for key, val in data.items():
                if key.startswith("r"):
                    addr = int(key[1:], 16)
                    ram[addr] = val
            samples.append((score_val, ram))
            print(f"    Captured score={score_val} ({len(ram)} bytes)")
        else:
            print("    No data yet, try again")

    running[0] = False
    env.close()
    sp.run(["pkill", "-9", "-f", f"mame.*{game_id}"], capture_output=True)

    if len(samples) < 2 or len(set(s for s, _ in samples)) < 2:
        print("\nNeed 2+ different scores.")
        return

    # Find matching addresses
    print(f"\n{'='*60}")
    print(f"Analyzing {len(samples)} captures...")

    all_addrs = set()
    for _, ram in samples:
        all_addrs.update(ram.keys())

    # Method 1: single-byte match (raw, BCD, 16-bit)
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

    # Method 2: digit match — score stored as individual digits (0-9 per byte)
    # e.g. score 1060 → digits [1, 0, 6, 0] somewhere in RAM
    if not consistent:
        print("  No single-byte match. Trying digit matching...")
        # For each score, build digit list
        score_digits = {}
        for score_val, _ in samples:
            score_digits[score_val] = [int(d) for d in str(score_val)]

        # Find addresses where byte == a digit of the score, and the digit
        # position is consistent across samples
        # For each address, check: does it always hold the Nth digit?
        for pos in range(6):  # up to 6-digit scores
            candidates = {}
            for addr in sorted(all_addrs):
                matches = []
                for score_val, ram in samples:
                    if addr not in ram:
                        break
                    digits = score_digits[score_val]
                    # Pad to same length
                    max_len = max(len(score_digits[s]) for s, _ in samples)
                    padded = [0] * (max_len - len(digits)) + digits
                    if pos < len(padded) and ram[addr] == padded[pos]:
                        matches.append((score_val, ram[addr]))
                if len(matches) == len(samples) and len(set(m[1] for m in matches)) >= 2:
                    candidates[addr] = matches
            if candidates:
                for addr, matches in candidates.items():
                    consistent[addr] = [(s, v, f"digit-pos{pos}") for s, v in matches]

    # Method 3: find bytes that increase with score (any encoding)
    if not consistent:
        print("  No digit match. Finding bytes correlated with score...")
        scores = [s for s, _ in samples]
        for addr in sorted(all_addrs):
            vals = [ram.get(addr, -1) for _, ram in samples]
            if len(set(vals)) < 2:
                continue
            # Check if byte increases when score increases
            increases_match = True
            for i in range(1, len(samples)):
                if scores[i] > scores[i-1] and vals[i] <= vals[i-1]:
                    increases_match = False
                    break
                if scores[i] < scores[i-1] and vals[i] >= vals[i-1]:
                    increases_match = False
                    break
            if increases_match:
                consistent[addr] = [(s, ram.get(addr, 0), "correlated")
                                     for s, ram in samples]

    if consistent:
        print(f"\nCandidate addresses ({len(consistent)} found):")
        for addr, matches in sorted(consistent.items())[:20]:
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
        print("\nNo matches found. Check that scores were exact.")


def validate_mode(game_id):
    config_path = os.path.join(os.path.dirname(__file__), "game_configs.json")
    with open(config_path) as f:
        configs = json.load(f)
    if game_id not in configs:
        print(f"No config for {game_id}. Run without --validate.")
        return
    cfg = configs[game_id]
    score_addrs = [int(a, 16) for a in cfg.get("score_addrs", [])]
    lives_addr = int(cfg["lives_addr"], 16) if "lives_addr" in cfg else None
    encoding = cfg.get("encoding", "raw")
    ram = {}
    for i, a in enumerate(score_addrs):
        ram[f"s{i}"] = a
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
                score = sum(v * (10 ** (len(vals)-1-i)) for i, v in enumerate(vals) if 0 <= v <= 9)
            else:
                score = sum(v << (8*i) for i, v in enumerate(vals))
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
