"""Auto-find score/lives RAM addresses using OCR + RAM scanning.

Strategy:
1. Play the game, grab screenshots at intervals
2. OCR the score from each screenshot
3. Scan all RAM for byte patterns that match the OCR'd score
4. Intersect candidates across multiple samples → find the address

Usage:
    python3 find_score_ram.py qbert
    python3 find_score_ram.py galaga
    python3 find_score_ram.py dkong
"""

import sys
import os
import re
import time
import argparse
import numpy as np
from PIL import Image
import pytesseract

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mle import MameEnv

ROMS_PATH = "/Users/pat/mame/roms"
SCAN_RANGE = range(0x0000, 0x2000)  # 8KB covers most arcade games


def ocr_score(rgb_frame):
    """OCR digits from the top portion of a game screen.

    Returns list of (value_str, region) for each number found.
    """
    img = Image.fromarray(rgb_frame)
    w, h = img.size

    # Arcade scores are usually in the top 20% of screen
    # Try multiple horizontal strips
    results = []
    for y_frac in [0.0, 0.05]:
        y1 = int(h * y_frac)
        y2 = int(h * (y_frac + 0.15))
        crop = img.crop((0, y1, w, y2))

        # Scale up 3x for better OCR on small arcade fonts
        crop = crop.resize((crop.width * 3, crop.height * 3), Image.NEAREST)

        # OCR with digit-only whitelist
        text = pytesseract.image_to_string(
            crop,
            config='--psm 6 -c tessedit_char_whitelist=0123456789'
        ).strip()

        # Extract all number sequences
        for match in re.finditer(r'\d{2,}', text):
            results.append(match.group())

    return results


def find_matching_ram(ram_snapshot, target_value):
    """Find RAM addresses where the value matches target_value.

    Tries multiple encodings:
    - Single byte (raw)
    - BCD single byte
    - Multi-byte BCD (2-4 adjacent bytes)
    - Little-endian 16-bit
    - Big-endian 16-bit
    """
    matches = {}  # addr -> encoding_description

    for addr in SCAN_RANGE:
        val = ram_snapshot.get(addr, 0)

        # Single byte raw
        if val == target_value and target_value <= 255:
            matches[addr] = f"raw byte ({val})"

        # Single byte BCD: e.g. 0x25 = 25
        bcd_val = ((val >> 4) & 0xF) * 10 + (val & 0xF)
        if (val >> 4) <= 9 and (val & 0xF) <= 9:
            if bcd_val == target_value and target_value <= 99:
                matches[addr] = f"BCD byte ({val:#04x}={bcd_val})"

    # Multi-byte BCD: adjacent bytes form the full score
    # e.g. score 12350 = bytes 0x01 0x23 0x50 (high to low)
    # or bytes 0x50 0x23 0x01 (low to high)
    target_str = str(target_value)
    if len(target_str) >= 2:
        # Pad to even length
        if len(target_str) % 2:
            target_str = "0" + target_str
        n_bytes = len(target_str) // 2
        # Build expected BCD bytes
        bcd_bytes_hi_first = []
        for i in range(0, len(target_str), 2):
            hi = int(target_str[i])
            lo = int(target_str[i + 1])
            bcd_bytes_hi_first.append((hi << 4) | lo)
        bcd_bytes_lo_first = list(reversed(bcd_bytes_hi_first))

        for start_addr in SCAN_RANGE:
            if start_addr + n_bytes > max(SCAN_RANGE):
                break
            actual = [ram_snapshot.get(start_addr + i, -1) for i in range(n_bytes)]

            if actual == bcd_bytes_hi_first:
                addrs = [start_addr + i for i in range(n_bytes)]
                matches[start_addr] = f"BCD {n_bytes}B hi-first {[f'${a:04X}' for a in addrs]}"

            if actual == bcd_bytes_lo_first:
                addrs = [start_addr + i for i in range(n_bytes)]
                matches[start_addr] = f"BCD {n_bytes}B lo-first {[f'${a:04X}' for a in addrs]}"

    # 16-bit little-endian
    if target_value <= 0xFFFF:
        lo = target_value & 0xFF
        hi = (target_value >> 8) & 0xFF
        for addr in SCAN_RANGE:
            if addr + 1 > max(SCAN_RANGE):
                break
            if ram_snapshot.get(addr, -1) == lo and ram_snapshot.get(addr + 1, -1) == hi:
                matches[addr] = f"16-bit LE (${addr:04X}={lo:#04x}, ${addr+1:04X}={hi:#04x})"

    return matches


def read_full_ram(env):
    """Read all bytes in SCAN_RANGE from current MAME state."""
    # We already have them in the data dict from step()
    data = env.step()
    ram = {}
    for addr in SCAN_RANGE:
        key = f"_r{addr:04x}"
        if key in data:
            ram[addr] = data[key]
    return ram, data


def main():
    parser = argparse.ArgumentParser(description="Find score RAM addresses via OCR")
    parser.add_argument("game", help="MAME ROM name")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of score samples to collect")
    parser.add_argument("--headless", action="store_true",
                        help="Run unthrottled (fast, small window)")
    parser.add_argument("--manual", action="store_true",
                        help="You play, script watches and scans RAM")
    args = parser.parse_args()

    # Build RAM dict to read full range
    ram_dict = {f"_r{addr:04x}": addr for addr in SCAN_RANGE}

    headless = getattr(args, 'headless', False)
    print(f"[{args.game}] Starting MAME{'  (headless)' if headless else ''}...")
    # Must use render=True for pixel capture (snapshot_pixels needs rendering)
    # but can use throttle=False for speed
    env = MameEnv(ROMS_PATH, args.game, ram_dict,
                  render=True, sound=False, throttle=not headless)

    # Discover inputs
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
    env.wait(300)

    print(f"[{args.game}] Game started. Playing randomly to generate score...")
    print(f"  Actions: {[f[1] for f in actions]}")
    print()

    # Collect samples: play, screenshot, OCR, scan RAM
    import random
    samples = []  # list of (ocr_numbers_list, ram_snapshot)
    human = getattr(args, 'manual', False)

    if human:
        print(">>> YOU PLAY the game. Press Enter here to capture each sample.")
        print(f">>> Need {args.samples} samples with different scores.\n")

    for sample_idx in range(args.samples):
        if human:
            input(f"  [Sample {sample_idx + 1}/{args.samples}] Press Enter to capture...")
        else:
            # Play randomly for a bit
            n_play = 300 + sample_idx * 150
            for i in range(n_play):
                if actions and random.random() < 0.7:
                    port, field = random.choice(actions)
                    env.step(port, field)
                else:
                    env.step()

            # Re-coin if needed (game over from random play)
            if coin:
                env.step_n(*coin, 15)
                env.wait(30)
            if start:
                env.step_n(*start, 5)
                env.wait(30)

        # Grab frame + RAM
        env.request_frame()
        data = env.step()
        ram = {addr: data.get(f"_r{addr:04x}", 0) for addr in SCAN_RANGE}

        if "frame" not in data:
            print(f"  Sample {sample_idx + 1}: no frame data")
            continue

        frame = data["frame"]
        img = Image.fromarray(frame)
        img.save(f"/tmp/{args.game}_sample{sample_idx}.png")

        scores = ocr_score(frame)
        if not scores:
            print(f"  Sample {sample_idx + 1}: OCR found no numbers")
            continue

        print(f"  Sample {sample_idx + 1}: OCR found {scores}")
        samples.append((scores, ram))

    env.close()

    if len(samples) < 2:
        print("\nNot enough samples. The game may need manual identification.")
        return

    # For each sample, pick the most likely "current score" from OCR values.
    # Strategy: for each address, check if it encodes the OCR'd score in
    # EVERY sample. An address is the score if it consistently matches.

    # Get candidate scores per sample (try each OCR'd number)
    sample_scores = []
    for scores, ram in samples:
        sample_scores.append([int(s) for s in scores])

    # For each RAM address, check: does it match SOME OCR number in EACH sample?
    # And does its value CHANGE between samples?
    consistent_addrs = {}  # addr -> list of (sample_idx, matched_score, encoding)

    def byte_matches_score(val, score_val):
        """Check if a single byte could encode score_val."""
        if val == score_val and score_val <= 255:
            return "raw"
        # BCD
        if (val >> 4) <= 9 and (val & 0xF) <= 9:
            bcd = ((val >> 4) * 10) + (val & 0xF)
            if bcd == score_val and score_val <= 99:
                return "BCD"
        # Low byte of 16-bit
        if score_val <= 0xFFFF and (score_val & 0xFF) == val:
            return "16bit-lo"
        return None

    for addr in SCAN_RANGE:
        addr_matches = []
        for si, (scores, ram) in enumerate(samples):
            val = ram.get(addr, 0)
            matched = None
            for score_val in sample_scores[si]:
                enc = byte_matches_score(val, score_val)
                if enc:
                    matched = (score_val, enc)
                    break
            if matched:
                addr_matches.append((si, matched[0], matched[1]))

        # Must match in at least 2 samples
        if len(addr_matches) >= 2:
            # Value must change (not a constant byte matching a constant OCR number)
            values = [samples[si][1].get(addr, -1) for si, _, _ in addr_matches]
            matched_scores = [sc for _, sc, _ in addr_matches]
            if len(set(values)) >= 2 or len(set(matched_scores)) >= 2:
                consistent_addrs[addr] = addr_matches

    if consistent_addrs:
        # Rank by number of samples matched
        ranked = sorted(consistent_addrs.items(),
                       key=lambda x: -len(x[1]))
        best_addr, best_matches = ranked[0]

        print(f"\n{'='*60}")
        print(f"Consistent RAM addresses ({len(consistent_addrs)} found):")
        print(f"{'='*60}")
        for addr, matches in ranked[:10]:
            scores_str = ", ".join(f"{sc}" for _, sc, _ in matches)
            enc = matches[-1][2]
            print(f"  ${addr:04X}: matched scores [{scores_str}] ({enc})")

        top_addrs = [addr for addr, _ in ranked[:4]]
        addrs_hex = [f'"0x{a:04X}"' for a in top_addrs]
        print(f"\nSuggested game_configs.json entry:")
        print(f'  "{args.game}": {{')
        print(f'    "score_addrs": [{", ".join(addrs_hex)}],')
        print(f'    "lives_addr": "TODO"')
        print(f'  }}')
    else:
        print(f"\nNo consistent score addresses found.")
        print("Try playing the game manually or increase --samples.")


if __name__ == "__main__":
    main()
