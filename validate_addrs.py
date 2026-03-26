"""Find and validate score/lives RAM addresses while you play.

Two modes:
  python3 validate_addrs.py galaga              # FIND mode: you play, type score, it finds addresses
  python3 validate_addrs.py qbert --validate    # VALIDATE mode: shows known addresses while you play

FIND mode:
  1. Opens the game, you play normally
  2. Type your current score in the terminal
  3. Script searches all RAM for bytes matching that value
  4. After 3+ entries with different scores, shows consistent addresses
  5. Type 'lives' to search for the lives byte too

VALIDATE mode:
  Uses addresses from game_configs.json, prints values as you play.
"""

import sys
import os
import json
import argparse
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mle import MameEnv

ROMS_PATH = "/Users/pat/mame/roms"
SCAN_RANGE = 512  # read 512 bytes per MAME session


def find_mode(game_id):
    """Interactive address finder. You play + type scores."""

    print(f"[{game_id}] FIND MODE")
    print(f"  Game will open. Press 5=coin, 1=start, then play.")
    print(f"  When you want to capture, type your score here and press Enter.")
    print(f"  Type 'lives N' (e.g. 'lives 3') to search for lives address.")
    print(f"  Type 'done' when finished.\n")

    # We'll scan RAM in chunks of SCAN_RANGE per MAME session
    # First session: determine total scan range from CPU type
    import subprocess, xml.etree.ElementTree as ET
    try:
        result = subprocess.run(
            ["/opt/homebrew/bin/mame", game_id, "-listxml"],
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
        ranges = [(0x0000, 0x2000)]
    elif 'z80' in cpu_str:
        ranges = [(0x0000, 0x1000), (0x4000, 0x5000),
                  (0x6000, 0x7000), (0x8000, 0x9000), (0xC000, 0xD000)]
    else:
        ranges = [(0x0000, 0x1000), (0x4000, 0x5000), (0x8000, 0x9000)]

    # Flatten into list of (start, end) chunks of SCAN_RANGE
    chunks = []
    for rs, re in ranges:
        for cs in range(rs, re, SCAN_RANGE):
            chunks.append((cs, min(cs + SCAN_RANGE, re)))

    print(f"  CPU: {cpus}")
    print(f"  Scanning {sum(e-s for s,e in chunks)} RAM bytes across {len(chunks)} chunks")
    print(f"  This requires {len(chunks)} MAME sessions (auto-restarts)\n")

    samples = []  # list of (score_int, {addr: value})
    lives_samples = []  # list of (lives_int, {addr: value})

    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
        ram_dict = {f"r{a:04x}": a for a in range(chunk_start, chunk_end)}
        env = MameEnv(ROMS_PATH, game_id, ram_dict,
                      render=True, sound=True, throttle=True)

        # Pump thread
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

        print(f"--- Chunk {chunk_idx+1}/{len(chunks)}: ${chunk_start:04X}-${chunk_end:04X} ---")
        if chunk_idx == 0:
            print("  Insert coin (5) and start (1). Play and score some points.")

        while True:
            user = input("  Score (or 'next'/'done'/'lives N'): ").strip()
            if user.lower() == 'done':
                running[0] = False
                env.close()
                import subprocess as sp
                sp.run(["pkill", "-9", "-f", f"mame.*{game_id}"], capture_output=True)
                time.sleep(0.5)
                break
            if user.lower() == 'next':
                break
            if user.lower().startswith('lives'):
                parts = user.split()
                if len(parts) >= 2:
                    try:
                        lives_val = int(parts[1])
                        data = latest[0]
                        if data:
                            ram = {a: data.get(f"r{a:04x}", -1)
                                   for a in range(chunk_start, chunk_end)}
                            lives_samples.append((lives_val, ram))
                            print(f"    Recorded lives={lives_val}")
                    except ValueError:
                        print("    Usage: lives 3")
                continue
            try:
                score_val = int(user)
            except ValueError:
                print("    Enter a number, 'next', 'done', or 'lives N'")
                continue

            # Capture RAM snapshot
            running[0] = False
            t.join(timeout=2)
            time.sleep(0.1)
            data = latest[0]
            if data:
                ram = {a: data.get(f"r{a:04x}", -1)
                       for a in range(chunk_start, chunk_end)}
                samples.append((score_val, ram))
                print(f"    Recorded score={score_val}")
            running[0] = True
            t = threading.Thread(target=pump, daemon=True)
            t.start()

        running[0] = False
        try:
            env.close()
        except Exception:
            pass
        import subprocess as sp
        sp.run(["pkill", "-9", "-f", f"mame.*{game_id}"], capture_output=True)
        time.sleep(0.5)

        if user.lower() == 'done':
            break

    # Analyze: find addresses where byte value matches score
    if len(samples) >= 2:
        print(f"\n{'='*60}")
        print(f"Analyzing {len(samples)} score samples...")

        # For each address, check if its value matches the score in each sample
        from find_score_ram import byte_matches_score
        consistent = {}
        all_addrs = set()
        for _, ram in samples:
            all_addrs.update(ram.keys())

        for addr in sorted(all_addrs):
            matches = []
            for score_val, ram in samples:
                if addr not in ram:
                    continue
                val = ram[addr]
                enc = byte_matches_score(val, score_val)
                if enc:
                    matches.append((score_val, val, enc))
            if len(matches) == len(samples):
                # Check values actually differ between samples
                vals = [m[1] for m in matches]
                if len(set(vals)) >= 2:
                    consistent[addr] = matches

        if consistent:
            print(f"\nAddresses matching ALL {len(samples)} samples:")
            for addr, matches in sorted(consistent.items()):
                scores_str = ", ".join(f"{s}={v}" for s, v, _ in matches)
                print(f"  ${addr:04X}: {matches[0][2]} [{scores_str}]")

            best = list(consistent.keys())[:4]
            result = {"score_addrs": [f"0x{a:04X}" for a in best]}

            # Check for lives
            if lives_samples:
                print(f"\nAnalyzing {len(lives_samples)} lives samples...")
                for addr in sorted(all_addrs):
                    if addr in consistent:
                        continue
                    matches = []
                    for lives_val, ram in lives_samples:
                        if addr in ram and ram[addr] == lives_val:
                            matches.append(lives_val)
                    if len(matches) == len(lives_samples):
                        result["lives_addr"] = f"0x{addr:04X}"
                        print(f"  Lives: ${addr:04X}")
                        break

            # Save
            config_path = os.path.join(os.path.dirname(__file__), "game_configs.json")
            configs = {}
            if os.path.exists(config_path):
                with open(config_path) as f:
                    configs = json.load(f)
            configs[game_id] = result
            configs[game_id]["verified"] = True
            with open(config_path, 'w') as f:
                json.dump(configs, f, indent=4)
            print(f"\nSaved to game_configs.json: {json.dumps(result)}")
        else:
            print("No consistent addresses found. Try more samples with different scores.")
    else:
        print(f"\nNeed at least 2 score samples (got {len(samples)})")


def validate_mode(game_id):
    """Show known addresses while you play."""
    config_path = os.path.join(os.path.dirname(__file__), "game_configs.json")
    if not os.path.exists(config_path):
        print("No game_configs.json found")
        return
    with open(config_path) as f:
        configs = json.load(f)
    if game_id not in configs:
        print(f"No config for {game_id}. Run without --validate to find addresses.")
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

    print(f"[{game_id}] VALIDATE MODE")
    print(f"  Score: {[f'${a:04X}' for a in score_addrs]} ({encoding})")
    if lives_addr:
        print(f"  Lives: ${lives_addr:04X}")
    print(f"\n  Press 5=coin, 1=start. Values print when they change. Ctrl+C to quit.\n")

    env = MameEnv(ROMS_PATH, game_id, ram, render=True, sound=True, throttle=True)

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

    prev_score = None
    try:
        while True:
            time.sleep(0.5)
            data = latest[0]
            if data is None:
                continue
            vals = [data.get(f"s{i}", 0) for i in range(len(score_addrs))]
            if encoding == "tile":
                score = 0
                for v in vals:
                    score = score * 10 + (v if 0 <= v <= 9 else 0)
            else:
                score = sum(v << (8 * i) for i, v in enumerate(vals))
            lives = data.get("lives", "?")
            if score != prev_score:
                print(f"  score={score:>8d}  lives={lives}  raw={vals}")
                prev_score = score
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        running[0] = False
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("game", help="MAME ROM name")
    parser.add_argument("--validate", action="store_true",
                        help="Validate known addresses (default: find new ones)")
    parser.add_argument("--score", type=str, default=None)
    parser.add_argument("--lives", type=str, default=None)
    args = parser.parse_args()

    if args.validate or args.score:
        validate_mode(args.game)
    else:
        find_mode(args.game)


if __name__ == "__main__":
    main()
