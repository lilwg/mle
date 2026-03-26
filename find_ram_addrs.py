"""Find score/lives RAM addresses by writing to RAM and watching the screen.

For each candidate address: write a test value, render, check if the
score display changed. No OCR needed — just pixel comparison.

Usage:
    python3 find_ram_addrs.py qbert
    python3 find_ram_addrs.py galaga
    python3 find_ram_addrs.py dkong --validate
"""

import sys
import os
import json
import argparse
import subprocess
import time
import numpy as np
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mle import MameEnv

ROMS_PATH = "/Users/pat/mame/roms"


def get_ram_ranges(game_id):
    """Detect likely RAM address ranges from MAME XML."""
    try:
        result = subprocess.run(
            ["/opt/homebrew/bin/mame", game_id, "-listxml"],
            capture_output=True, text=True, timeout=10)
        tree = ET.fromstring(result.stdout)
        for m in tree.findall('.//machine'):
            if m.get('name') == game_id:
                cpus = [c.get('name', '').lower() for c in m.findall('.//chip')
                        if c.get('type') == 'cpu']
                break
        else:
            cpus = []
    except Exception:
        cpus = []

    # Map CPU types to common RAM ranges
    # Most games have RAM in multiple regions; scan all likely ones
    ranges = []
    cpu_str = ' '.join(cpus)

    if '8088' in cpu_str or '8086' in cpu_str:
        ranges = [(0x0000, 0x1000)]  # Q*bert style
    elif 'z80' in cpu_str:
        if 'namco' in cpu_str or 'mb88' in cpu_str:
            # Namco games (Galaga, Dig Dug, Pac-Man): RAM at $8000+
            ranges = [(0x8000, 0x9000), (0x0000, 0x1000)]
        else:
            # Standard Z80 games (DK, Frogger)
            ranges = [(0x0000, 0x1000), (0x4000, 0x5000), (0x6000, 0x7000),
                      (0x8000, 0x9000), (0xC000, 0xD000)]
    elif '6502' in cpu_str or '6809' in cpu_str:
        ranges = [(0x0000, 0x1000)]
    else:
        # Unknown CPU — scan common ranges
        ranges = [(0x0000, 0x1000), (0x4000, 0x5000), (0x8000, 0x9000)]

    print(f"[{game_id}] CPUs: {cpus}")
    print(f"  RAM ranges to scan: {[f'${s:04X}-${e:04X}' for s, e in ranges]}")
    return ranges


def lua_bulk_read(env, start, count):
    """Read `count` bytes starting at `start` via Lua console."""
    result = env.console.writeln_expect(
        f'local t={{}}; for a={start},{start + count - 1} do '
        f't[#t+1]=mem:read_u8(a) end; print(table.concat(t,","))')
    if not result:
        return [0] * count
    try:
        return [int(x) for x in result.split(',')]
    except ValueError:
        return [0] * count


def scan_range(env, start, end, ref_frame, threshold=0.3):
    """Scan addresses [start, end) by writing test values and comparing frames.

    Uses chunk scanning: test 8 addresses at once, binary search on hits.
    """
    h, w = ref_frame.shape[:2]
    # Score regions to check (top and bottom strips)
    regions = [
        ("top", slice(0, max(1, h // 6)), slice(None)),
        ("bottom", slice(h - h // 8, h), slice(None)),
        ("left", slice(None), slice(0, max(1, w // 6))),
    ]
    ref_regions = {name: ref_frame[ys, xs, :].astype(float)
                   for name, ys, xs in regions}

    # Batch read all originals
    count = end - start
    originals = lua_bulk_read(env, start, count)
    if len(originals) != count:
        print(f"  Warning: read {len(originals)} bytes, expected {count}")
        return []

    hits = []  # (addr, region_name, diff)

    # Chunk scan: test CHUNK_SIZE addresses at once
    CHUNK_SIZE = 32
    for chunk_start in range(0, count, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, count)

        # Write test values via Lua loop
        vals = ",".join(str(0x99 if originals[i] != 0x99 else 0x55)
                        for i in range(chunk_start, chunk_end))
        env.console.writeln(
            f'local v={{{vals}}}; '
            f'for i=1,{chunk_end - chunk_start} do '
            f'mem:write_u8({start + chunk_start}+i-1, v[i]) end')

        # Render
        env.request_frame()
        data = env.step()

        # Restore via Lua loop
        orig_vals = ",".join(str(originals[i])
                             for i in range(chunk_start, chunk_end))
        env.console.writeln(
            f'local v={{{orig_vals}}}; '
            f'for i=1,{chunk_end - chunk_start} do '
            f'mem:write_u8({start + chunk_start}+i-1, v[i]) end')

        if 'frame' not in data:
            continue

        # Check if ANY region changed
        frame = data['frame']
        chunk_hit = False
        for name, ys, xs in regions:
            new_region = frame[ys, xs, :].astype(float)
            diff = np.mean(np.abs(new_region - ref_regions[name]))
            if diff > threshold:
                chunk_hit = True
                break

        if not chunk_hit:
            continue

        # Binary search: test each address individually
        for i in range(chunk_start, chunk_end):
            addr = start + i
            orig = originals[i]
            test = 0x99 if orig != 0x99 else 0x55

            env.console.writeln(f'mem:write_u8({addr}, {test})')
            env.request_frame()
            data = env.step()
            env.console.writeln(f'mem:write_u8({addr}, {orig})')

            if 'frame' not in data:
                continue

            frame = data['frame']
            for name, ys, xs in regions:
                new_region = frame[ys, xs, :].astype(float)
                diff = np.mean(np.abs(new_region - ref_regions[name]))
                if diff > threshold:
                    hits.append((addr, name, diff, orig))

    return hits


def find_lives(env, candidates, ref_frame):
    """Find lives address by writing 0 and checking for game-over behavior."""
    h, w = ref_frame.shape[:2]
    ref_flat = ref_frame.astype(float)
    lives_hits = []

    for addr, orig in candidates:
        if not (1 <= orig <= 6):
            continue

        # Write 0 (no lives = game over?)
        env.console.writeln(f'mem:write_u8({addr}, 0)')
        # Let a few frames pass
        for _ in range(10):
            env.step()
        env.request_frame()
        data = env.step()
        env.console.writeln(f'mem:write_u8({addr}, {orig})')

        if 'frame' not in data:
            continue

        # Big screen change = likely game over
        diff = np.mean(np.abs(data['frame'].astype(float) - ref_flat))
        if diff > 5.0:
            lives_hits.append((addr, diff, orig))

    return lives_hits


def main():
    parser = argparse.ArgumentParser(description="Find score/lives RAM via write-and-observe")
    parser.add_argument("game", help="MAME ROM name")
    parser.add_argument("--validate", action="store_true",
                        help="Validate found addresses visually")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON for programmatic use")
    args = parser.parse_args()

    # Detect RAM ranges
    ram_ranges = get_ram_ranges(args.game)

    # Start MAME
    print(f"\n[{args.game}] Starting MAME...")
    env = MameEnv(ROMS_PATH, args.game, {"_dummy": 0},
                  render=True, sound=False, throttle=False)

    # Discover inputs
    result = env.console.writeln_expect(
        'local r={}; for p,port in pairs(manager.machine.ioport.ports) do '
        'for f,field in pairs(port.fields) do '
        'r[#r+1]=p.."~"..f end end; print(table.concat(r,";"))')
    coin, start_btn, actions = None, None, []
    if result:
        for pair in result.split(';'):
            if '~' in pair:
                port, field = pair.split('~', 1)
                fl = field.lower()
                if 'coin' in fl:
                    coin = (port, field)
                elif '1 player' in fl or ('start' in fl and '1' in fl):
                    start_btn = (port, field)
                elif 'p1' in fl:
                    actions.append((port, field))

    # Insert coin + start
    if coin:
        env.step_n(*coin, 15)
    env.wait(120)
    if start_btn:
        env.step_n(*start_btn, 5)
    env.wait(300)

    # Reference frame
    env.request_frame()
    ref_data = env.step()
    if 'frame' not in ref_data:
        print("ERROR: No frame data. Is render=True?")
        env.close()
        return
    ref_frame = ref_data['frame']
    h, w = ref_frame.shape[:2]
    print(f"  Frame: {w}x{h}")
    print(f"  Scanning for score display addresses...\n")

    # Scan all RAM ranges
    all_hits = []
    for range_start, range_end in ram_ranges:
        count = range_end - range_start
        print(f"  Scanning ${range_start:04X}-${range_end:04X} ({count} bytes)...")
        t0 = time.time()
        hits = scan_range(env, range_start, range_end, ref_frame)
        elapsed = time.time() - t0
        print(f"    {len(hits)} hits in {elapsed:.1f}s")
        all_hits.extend(hits)

    # Group adjacent hits
    if all_hits:
        # Sort by address
        all_hits.sort(key=lambda x: x[0])

        # Cluster adjacent addresses (likely multi-byte score)
        clusters = []
        current_cluster = [all_hits[0]]
        for hit in all_hits[1:]:
            if hit[0] <= current_cluster[-1][0] + 2:  # within 2 bytes
                current_cluster.append(hit)
            else:
                clusters.append(current_cluster)
                current_cluster = [hit]
        clusters.append(current_cluster)

        print(f"\n{'='*60}")
        print(f"Found {len(clusters)} display-affecting regions:")
        print(f"{'='*60}")
        for i, cluster in enumerate(clusters):
            addrs = [h[0] for h in cluster]
            region = cluster[0][1]
            max_diff = max(h[2] for h in cluster)
            addr_range = f"${addrs[0]:04X}" if len(addrs) == 1 else \
                         f"${addrs[0]:04X}-${addrs[-1]:04X}"
            print(f"  {i+1}. {addr_range} ({len(addrs)} bytes, "
                  f"region={region}, max_diff={max_diff:.1f})")

        # Best score candidate: cluster in top region with highest diff
        top_clusters = [c for c in clusters if any(h[1] == 'top' for h in c)]
        if top_clusters:
            best = max(top_clusters, key=lambda c: max(h[2] for h in c))
            score_addrs = [h[0] for h in best]
            print(f"\n  Best score candidate: "
                  f"{[f'${a:04X}' for a in score_addrs]}")
        else:
            score_addrs = [clusters[0][0][0]]
            print(f"\n  Best candidate (non-top): ${score_addrs[0]:04X}")

        # Try to find lives
        print(f"\n  Scanning for lives address...")
        # Candidates: all bytes with value 1-6 in the score region
        lives_candidates = []
        for range_start, range_end in ram_ranges:
            originals = lua_bulk_read(env, range_start, range_end - range_start)
            for i, val in enumerate(originals):
                if 1 <= val <= 6:
                    lives_candidates.append((range_start + i, val))

        lives_hits = find_lives(env, lives_candidates[:50], ref_frame)
        lives_addr = None
        if lives_hits:
            lives_hits.sort(key=lambda x: -x[1])
            lives_addr = lives_hits[0][0]
            print(f"  Lives candidate: ${lives_addr:04X} (was {lives_hits[0][2]})")

        # Output
        result = {
            "score_addrs": [f"0x{a:04X}" for a in score_addrs],
        }
        if lives_addr:
            result["lives_addr"] = f"0x{lives_addr:04X}"

        print(f"\n  Suggested game_configs.json entry:")
        print(f'  "{args.game}": {json.dumps(result, indent=4)}')

        if args.json:
            print(f"\nJSON:{json.dumps(result)}")

        # Auto-save to game_configs.json
        config_path = os.path.join(os.path.dirname(__file__), "game_configs.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                configs = json.load(f)
        else:
            configs = {}
        if args.game not in configs:
            configs[args.game] = result
            with open(config_path, 'w') as f:
                json.dump(configs, f, indent=4)
            print(f"\n  Saved to game_configs.json")
    else:
        print("\nNo display-affecting addresses found.")
        print("Try extending the RAM range or checking if the game is running.")

    env.close()


if __name__ == "__main__":
    main()
