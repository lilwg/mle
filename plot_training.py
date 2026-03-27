"""Live training dashboard. Reads SB3 tensorboard logs and plots charts.

Usage: python3 plot_training.py [--refresh 10]
"""

import os
import sys
import time
import glob
import struct
import argparse
from collections import defaultdict

def read_tfevents(path):
    """Read scalar summaries from a TensorFlow events file."""
    results = defaultdict(list)
    try:
        with open(path, 'rb') as f:
            while True:
                # TFRecord format: uint64 length, uint32 crc, bytes data, uint32 crc
                header = f.read(12)
                if len(header) < 12:
                    break
                length = struct.unpack('Q', header[:8])[0]
                data = f.read(length + 4)  # data + footer crc
                if len(data) < length:
                    break

                # Parse the protobuf Event
                # Simple parsing: look for tag strings and float values
                # SB3 writes simple scalar summaries
                try:
                    from tensorboard.compat.proto.event_pb2 import Event
                    event = Event()
                    event.ParseFromString(data[:length])
                    if event.HasField('summary'):
                        for v in event.summary.value:
                            if v.HasField('simple_value'):
                                results[v.tag].append((event.step, v.simple_value))
                except Exception:
                    pass
    except Exception:
        pass
    return dict(results)


def read_from_log(logfile):
    """Fallback: parse ep_rew_mean from stdout log."""
    results = {'rollout/ep_rew_mean': [], 'rollout/ep_len_mean': [],
               'time/fps': []}
    step = 0
    with open(logfile) as f:
        for line in f:
            line = line.strip()
            if 'total_timesteps' in line and '|' in line:
                try:
                    val = line.split('|')[2].strip()
                    step = int(val)
                except (IndexError, ValueError):
                    pass
            elif 'ep_rew_mean' in line and '|' in line:
                try:
                    val = float(line.split('|')[2].strip())
                    results['rollout/ep_rew_mean'].append((step, val))
                except (IndexError, ValueError):
                    pass
            elif 'ep_len_mean' in line and '|' in line:
                try:
                    val = float(line.split('|')[2].strip())
                    results['rollout/ep_len_mean'].append((step, val))
                except (IndexError, ValueError):
                    pass
            elif 'fps' in line and '|' in line and 'time' not in line:
                try:
                    val = float(line.split('|')[2].strip())
                    results['time/fps'].append((step, val))
                except (IndexError, ValueError):
                    pass
    return {k: v for k, v in results.items() if v}


def plot_ascii(title, data, width=60, height=15):
    """Simple ASCII chart."""
    if not data:
        print(f"  {title}: no data")
        return

    steps = [d[0] for d in data]
    values = [d[1] for d in data]

    if len(values) < 2:
        print(f"  {title}: {values[0]:.1f} (1 point)")
        return

    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        max_v = min_v + 1

    print(f"  {title}")
    print(f"  {max_v:>8.1f} ┤", end="")

    # Downsample to width
    if len(values) > width:
        step_size = len(values) / width
        sampled = [values[int(i * step_size)] for i in range(width)]
    else:
        sampled = values

    for row in range(height, -1, -1):
        threshold = min_v + (max_v - min_v) * row / height
        line = ""
        for v in sampled:
            if v >= threshold:
                line += "█"
            else:
                line += " "
        if row == height:
            print(line)
        elif row == 0:
            print(f"  {min_v:>8.1f} ┤{line}")
        else:
            print(f"           │{line}")

    print(f"           └{'─' * len(sampled)}")
    print(f"            0{' ' * (len(sampled)-10)}step {steps[-1]}")
    print(f"  Latest: {values[-1]:.1f}  Best: {max(values):.1f}  "
          f"Points: {len(values)}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", type=int, default=0,
                        help="Auto-refresh every N seconds (0=once)")
    parser.add_argument("--log", type=str, default=None,
                        help="Read from stdout log instead of TB")
    args = parser.parse_args()

    while True:
        os.system('clear' if os.name != 'nt' else 'cls')
        print("=" * 70)
        print("  MLE Training Dashboard")
        print("=" * 70)

        data = {}

        # Try tensorboard logs first
        tb_files = sorted(glob.glob("tb_logs/*/events.out.tfevents.*"))
        if tb_files:
            for tf in tb_files:
                run_name = os.path.basename(os.path.dirname(tf))
                d = read_tfevents(tf)
                if d:
                    print(f"\n  Run: {run_name}")
                    data.update(d)

        # Fallback: parse stdout logs
        if not data:
            logs = sorted(glob.glob("train_*.log"))
            if args.log:
                logs = [args.log]
            for logfile in logs:
                d = read_from_log(logfile)
                if d:
                    print(f"\n  Log: {logfile}")
                    data.update(d)

        if not data:
            print("\n  No training data found.")
            print("  Looking in: tb_logs/ and train_*.log")
        else:
            for key in ['rollout/ep_rew_mean', 'rollout/ep_len_mean', 'time/fps']:
                if key in data:
                    plot_ascii(key, data[key])

        if args.refresh <= 0:
            break
        print(f"\n  Refreshing in {args.refresh}s... (Ctrl+C to stop)")
        try:
            time.sleep(args.refresh)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
