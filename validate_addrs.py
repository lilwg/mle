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
MAX_PIPE_ADDRS = 400  # safe limit for pipe protocol


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
    if score_val <= 0xFFFF and (score_val & 0xFF) == val:
        return "16bit-lo"
    if score_val > 255 and score_val <= 0xFFFF and ((score_val >> 8) & 0xFF) == val:
        return "16bit-hi"
    return None


def find_mode(game_id):
    ranges = get_ram_ranges(game_id)

    # Split into pipe-safe chunks
    chunks = []
    for rs, re in ranges:
        for cs in range(rs, re, MAX_PIPE_ADDRS):
            chunks.append((cs, min(cs + MAX_PIPE_ADDRS, re)))

    total = sum(e - s for s, e in chunks)
    print(f"[{game_id}] FIND MODE — {total} RAM bytes, {len(chunks)} chunks")
    print(f"  For each chunk: game restarts, you play to a score, type it.")
    print(f"  Need 2+ different scores per chunk. Type 'next' to advance.\n")

    all_results = {}  # addr -> [(score, val, encoding)]

    for ci, (cs, ce) in enumerate(chunks):
        sp.run(["pkill", "-9", "-f", f"mame.*{game_id}"], capture_output=True)
        time.sleep(0.5)

        ram_dict = {f"r{a:04x}": a for a in range(cs, ce)}
        env = MameEnv(ROMS_PATH, game_id, ram_dict,
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

        print(f"--- Chunk {ci+1}/{len(chunks)}: ${cs:04X}-${ce:04X} ---")
        print(f"  5=coin, 1=start. Type score, 'next', or 'done'.")

        chunk_samples = []
        while True:
            user = input("  > ").strip()
            if user.lower() in ('next', 'done'):
                break
            try:
                score_val = int(user)
            except ValueError:
                print("    Number, 'next', or 'done'")
                continue

            data = latest[0]
            if data:
                ram = {a: data.get(f"r{a:04x}", 0) for a in range(cs, ce)}
                chunk_samples.append((score_val, ram))
                print(f"    Got score={score_val}")

        # Analyze this chunk
        if len(chunk_samples) >= 2 and len(set(s for s, _ in chunk_samples)) >= 2:
            for addr in range(cs, ce):
                matches = []
                for score_val, ram in chunk_samples:
                    enc = byte_matches_score(ram.get(addr, 0), score_val)
                    if enc:
                        matches.append((score_val, ram.get(addr, 0), enc))
                if (len(matches) == len(chunk_samples)
                        and len(set(m[1] for m in matches)) >= 2):
                    all_results[addr] = matches

        running[0] = False
        env.close()

        if user.lower() == 'done':
            break

    sp.run(["pkill", "-9", "-f", f"mame.*{game_id}"], capture_output=True)

    if all_results:
        print(f"\n{'='*60}")
        print(f"Addresses where byte matches score:")
        for addr, matches in sorted(all_results.items()):
            scores_str = " ".join(f"{s}→{v}" for s, v, _ in matches)
            print(f"  ${addr:04X}: {matches[0][2]} [{scores_str}]")

        best = sorted(all_results.keys())[:4]
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
        print(f"\nNo matches found in scanned chunks.")


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
