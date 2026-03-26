"""Find score/lives RAM addresses by writing to RAM and watching the screen.

For each candidate address: write a test value, render one frame, check
if the score display region changed. Addresses that cause the biggest
pixel change in the score region are the score bytes.

Usage:
    python3 find_ram_addrs.py qbert
    python3 find_ram_addrs.py galaga
    python3 find_ram_addrs.py dkong
    python3 find_ram_addrs.py frogger --json
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
    """Detect likely RAM ranges from MAME XML CPU info."""
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
        return [(0x0000, 0x1000)]
    elif 'namco' in cpu_str or 'mb88' in cpu_str or 'galaga' in game_id or 'digdug' in game_id:
        return [(0x8000, 0x9000)]
    elif 'z80' in cpu_str:
        return [(0x0000, 0x1000), (0x4000, 0x5000),
                (0x6000, 0x7000), (0x8000, 0x9000), (0xC000, 0xD000)]
    elif '6502' in cpu_str:
        return [(0x0000, 0x0800)]
    else:
        return [(0x0000, 0x1000), (0x8000, 0x9000)]


def discover_inputs(env):
    """Discover coin, start, and P1 actions via Lua ioport."""
    result = env.console.writeln_expect(
        'local r={}; for p,port in pairs(manager.machine.ioport.ports) do '
        'for f,field in pairs(port.fields) do '
        'r[#r+1]=p.."~"..f end end; print(table.concat(r,";"))')
    coin, start, actions = None, None, []
    if result:
        for pair in result.split(';'):
            if '~' not in pair:
                continue
            port, field = pair.split('~', 1)
            fl = field.lower()
            if 'coin' in fl:
                coin = (port, field)
            elif '1 player' in fl or ('start' in fl and '1' in fl):
                start = (port, field)
            elif 'p1' in fl:
                actions.append((port, field))
    return coin, start, actions


def scan_addresses(env, start, end, ref_frame, region_slices):
    """Scan [start, end) by writing test values and comparing score region.

    Two-pass approach:
    1. Coarse pass: test every 4th address, find areas with hits
    2. Fine pass: test every address in hit areas

    Returns list of (addr, diff, original_value).
    """
    h, w = ref_frame.shape[:2]
    ref_regions = [ref_frame[ys, xs, :].astype(int) for ys, xs in region_slices]

    # Read all originals via pipe (already in the RAM dict)
    data = env.step()

    def test_addr(addr):
        key = f'r{addr:04x}'
        orig = data.get(key, 0)
        test = 0x99 if orig != 0x99 else 0x55
        env.console.writeln(f'mem:write_u8({addr}, {test})')
        env.request_frame()
        fd = env.step()
        env.console.writeln(f'mem:write_u8({addr}, {orig})')
        if 'frame' not in fd:
            return 0, orig
        # Count changed pixels (>20 intensity) in each score region
        max_changed = 0
        for i, (ys, xs) in enumerate(region_slices):
            new_region = fd['frame'][ys, xs, :]
            pixel_diff = np.max(np.abs(
                new_region.astype(int) - ref_regions[i].astype(int)), axis=2)
            changed = np.sum(pixel_diff > 20)
            max_changed = max(max_changed, changed)
        return max_changed, orig

    count = end - start

    # Coarse pass: every 4th address
    coarse_hits = set()
    coarse_step = 4
    tested = 0
    for addr in range(start, end, coarse_step):
        diff, orig = test_addr(addr)
        tested += 1
        if diff > 30:
            # Mark this area for fine scanning
            for a in range(max(start, addr - 4), min(end, addr + 5)):
                coarse_hits.add(a)

    print(f"    Coarse: tested {tested}/{count}, {len(coarse_hits)} candidates")

    # Fine pass: test each candidate individually
    hits = []
    for addr in sorted(coarse_hits):
        if addr % coarse_step == (start % coarse_step):
            continue  # already tested in coarse pass, retest for consistency
        diff, orig = test_addr(addr)
        if diff > 30:
            hits.append((addr, diff, orig))

    # Also re-test coarse hits
    for addr in range(start, end, coarse_step):
        diff, orig = test_addr(addr)
        if diff > 30:
            hits.append((addr, diff, orig))

    return hits


def main():
    parser = argparse.ArgumentParser(description="Find score/lives RAM via write-and-observe")
    parser.add_argument("game", help="MAME ROM name")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    ram_ranges = get_ram_ranges(args.game)

    total_addrs = sum(e - s for s, e in ram_ranges)
    print(f"[{args.game}] Scanning {total_addrs} RAM addresses")
    print(f"  Ranges: {[f'${s:04X}-${e:04X}' for s, e in ram_ranges]}")

    # Scan in chunks of 512 to keep pipe data manageable
    CHUNK = 512
    all_hits = []
    ref_frame = None
    ref_data = None
    t0 = time.time()

    for range_start, range_end in ram_ranges:
        for chunk_start in range(range_start, range_end, CHUNK):
            chunk_end = min(chunk_start + CHUNK, range_end)

            # Start MAME with just this chunk's addresses
            ram_dict = {f'r{a:04x}': a for a in range(chunk_start, chunk_end)}
            env = MameEnv(ROMS_PATH, args.game, ram_dict,
                          render=True, sound=False, throttle=False)

            if ref_frame is None:
                # First chunk: discover inputs, coin+start, get reference
                coin, start_btn, actions = discover_inputs(env)
                if coin:
                    env.step_n(*coin, 15)
                env.wait(120)
                if start_btn:
                    env.step_n(*start_btn, 5)
                env.wait(300)

                env.request_frame()
                ref_data = env.step()
                if 'frame' not in ref_data:
                    print("ERROR: No frame data")
                    env.close()
                    return
                ref_frame = ref_data['frame']
                h, w = ref_frame.shape[:2]
                print(f"  Frame: {w}x{h}")
            else:
                # Subsequent chunks: coin+start with same button names
                if coin:
                    env.step_n(*coin, 15)
                env.wait(120)
                if start_btn:
                    env.step_n(*start_btn, 5)
                env.wait(300)
                # Re-capture reference for this session
                env.request_frame()
                rd = env.step()
                if 'frame' in rd:
                    ref_frame = rd['frame']

            h, w = ref_frame.shape[:2]
            region_slices = [
                (slice(0, max(1, h // 12)), slice(None)),
                (slice(h - h // 12, h), slice(None)),
                (slice(0, max(1, h // 8)), slice(0, w // 3)),
                (slice(0, max(1, h // 8)), slice(2 * w // 3, w)),
            ]

            count = chunk_end - chunk_start
            print(f"\n  Scanning ${chunk_start:04X}-${chunk_end:04X} ({count} bytes)...")
            hits = scan_addresses(env, chunk_start, chunk_end, ref_frame, region_slices)
            all_hits.extend(hits)
            elapsed = time.time() - t0
            print(f"    {len(hits)} hits ({elapsed:.0f}s elapsed)")

            env.close()
            time.sleep(0.5)
            subprocess.run(["pkill", "-9", "-f", f"mame.*{args.game}"],
                          capture_output=True)
            time.sleep(0.5)
    total_time = time.time() - t0

    if not all_hits:
        print(f"\nNo display-affecting addresses found in {total_time:.0f}s.")
        return

    # Sort by diff (highest = most likely score)
    all_hits.sort(key=lambda x: -x[1])

    # Compute baseline: median changed-pixel count (background noise)
    diffs = [h[1] for h in all_hits]
    median_diff = sorted(diffs)[len(diffs) // 2]

    # Score candidates: significantly above median (2x)
    score_threshold = max(median_diff * 2, 20)
    score_hits = [(a, d, o) for a, d, o in all_hits if d > score_threshold]

    # Cluster adjacent addresses
    if score_hits:
        score_hits.sort(key=lambda x: x[0])
        clusters = []
        current = [score_hits[0]]
        for hit in score_hits[1:]:
            if hit[0] <= current[-1][0] + 2:
                current.append(hit)
            else:
                clusters.append(current)
                current = [hit]
        clusters.append(current)

        # Best cluster: highest average diff
        best = max(clusters, key=lambda c: sum(h[1] for h in c) / len(c))
        score_addrs = [h[0] for h in best]
    else:
        # Just take the top hit
        score_addrs = [all_hits[0][0]]

    # Lives: find by rescanning near score cluster for bytes with value 1-6
    lives_addr = None
    # Quick scan: start MAME, read bytes near score, look for value 1-6
    score_area_start = max(0, score_addrs[0] - 64)
    score_area_end = score_addrs[-1] + 64
    ram_dict = {f'r{a:04x}': a for a in range(score_area_start, score_area_end)}
    try:
        env2 = MameEnv(ROMS_PATH, args.game, ram_dict,
                       render=True, sound=False, throttle=False)
        if coin: env2.step_n(*coin, 15)
        env2.wait(120)
        if start_btn: env2.step_n(*start_btn, 5)
        env2.wait(300)
        ld = env2.step()
        for addr_offset in range(-64, 65):
            for sa in score_addrs:
                candidate = sa + addr_offset
                if candidate in score_addrs:
                    continue
                key = f'r{candidate:04x}'
                val = ld.get(key)
                if val is not None and 1 <= val <= 6:
                    lives_addr = candidate
                    break
            if lives_addr:
                break
        env2.close()
    except Exception:
        pass
    subprocess.run(["pkill", "-9", "-f", f"mame.*{args.game}"], capture_output=True)

    # Output results
    print(f"\n{'=' * 60}")
    print(f"Results for {args.game} ({total_time:.0f}s)")
    print(f"{'=' * 60}")
    print(f"  Score addresses: {[f'${a:04X}' for a in score_addrs]}")
    print(f"    (diff={max(h[1] for h in best if h[0] in score_addrs):.2f} "
          f"vs noise={median_diff:.2f})")
    if lives_addr:
        print(f"  Lives address: ${lives_addr:04X} "
              f"(value={ref_data.get(f'r{lives_addr:04x}', '?')})")
    else:
        print(f"  Lives address: not found")

    result = {"score_addrs": [f"0x{a:04X}" for a in score_addrs]}
    if lives_addr:
        result["lives_addr"] = f"0x{lives_addr:04X}"

    print(f"\n  game_configs.json entry:")
    print(f'  "{args.game}": {json.dumps(result)}')

    if args.json:
        print(f"\nJSON:{json.dumps(result)}")

    # Auto-save to game_configs.json
    config_path = os.path.join(os.path.dirname(__file__), "game_configs.json")
    configs = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            configs = json.load(f)
    if args.game not in configs:
        configs[args.game] = result
        with open(config_path, 'w') as f:
            json.dump(configs, f, indent=4)
        print(f"  Saved to game_configs.json")


if __name__ == "__main__":
    main()
