"""General-purpose MAME RL agent. Works with any arcade game ROM.

Usage:
    python3 mle_general.py pacman                        # Train on Pac-Man
    python3 mle_general.py qbert --model dqn             # Train with DQN
    python3 mle_general.py galaga --model a2c             # Train with A2C
    python3 mle_general.py dkong --eval model.zip        # Evaluate saved model
    python3 mle_general.py qbert --detect-score          # Auto-detect score RAM

No game-specific code. Uses pixels for observations, auto-detects
inputs via MAME XML metadata, and can auto-detect score RAM addresses.
"""

import sys
import os
import time
import subprocess
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mle import MameEnv

ROMS_PATH = "/Users/pat/mame/roms"
FRAME_SKIP = 4
OBS_H, OBS_W = 84, 84
STACK_SIZE = 4


def discover_inputs(game_id):
    """Discover game inputs from MAME XML metadata."""
    result = subprocess.run(
        ["/opt/homebrew/bin/mame", game_id, "-listxml"],
        capture_output=True, text=True, timeout=10,
    )
    tree = ET.fromstring(result.stdout)
    machine = tree.find(f'.//machine[@name="{game_id}"]')
    if machine is None:
        # Try as child of root
        for m in tree.findall('.//machine'):
            if m.get('name') == game_id:
                machine = m
                break

    inp = machine.find('input') if machine is not None else None
    n_buttons = 0
    ways = 4
    if inp:
        for ctrl in inp.findall('control'):
            if ctrl.get('player') in (None, '1'):
                b = ctrl.get('buttons')
                if b:
                    n_buttons = max(n_buttons, int(b))
                w = ctrl.get('ways')
                if w:
                    ways = int(w)
    return ways, n_buttons


# Standard MAME input port/field names for P1
DIRECTION_BUTTONS = {
    4: [  # 4-way joystick
        (":IN4", "P1 Up"),
        (":IN4", "P1 Down"),
        (":IN4", "P1 Left"),
        (":IN4", "P1 Right"),
    ],
    2: [  # 2-way (left/right only)
        (":IN0", "P1 Left"),
        (":IN0", "P1 Right"),
    ],
}

# Common port/field names across MAME drivers (tried in order)
ALT_DIRECTIONS = [
    # Q*bert diagonal style (:IN4)
    [(":IN4", "P1 Up (Up-Right)"), (":IN4", "P1 Down (Down-Left)"),
     (":IN4", "P1 Left (Up-Left)"), (":IN4", "P1 Right (Down-Right)")],
    # Standard 4-way on various ports
    [(":IN0", "P1 Up"), (":IN0", "P1 Down"),
     (":IN0", "P1 Left"), (":IN0", "P1 Right")],
    [(":IN1", "P1 Up"), (":IN1", "P1 Down"),
     (":IN1", "P1 Left"), (":IN1", "P1 Right")],
    [(":IN4", "P1 Up"), (":IN4", "P1 Down"),
     (":IN4", "P1 Left"), (":IN4", "P1 Right")],
    # 2-way (left/right only)
    [(":IN0", "P1 Left"), (":IN0", "P1 Right")],
    [(":IN1", "P1 Left"), (":IN1", "P1 Right")],
    # Joystick variants
    [(":IN0", "P1 Joystick Up"), (":IN0", "P1 Joystick Down"),
     (":IN0", "P1 Joystick Left"), (":IN0", "P1 Joystick Right")],
    [(":IN1", "P1 Joystick Up"), (":IN1", "P1 Joystick Down"),
     (":IN1", "P1 Joystick Left"), (":IN1", "P1 Joystick Right")],
]

BUTTON_NAMES = [
    (":IN0", "P1 Button 1"), (":IN1", "P1 Button 1"),
    (":IN2", "P1 Button 1"), (":IN4", "P1 Button 1"),
    (":IN0", "P1 Button1"), (":IN1", "P1 Button1"),
]

COIN_BUTTONS = [
    (":IN0", "Coin 1"), (":IN1", "Coin 1"),
    (":IN5", "Coin 1"), (":IN6", "Coin 1"), (":IN7", "Coin 1"),
    (":IN0", "Coin"), (":IN1", "Coin"),
]

START_BUTTONS = [
    (":IN1", "1 Player Start"), (":IN0", "1 Player Start"),
    (":IN4", "1 Player Start"), (":IN5", "1 Player Start"),
    (":IN6", "1 Player Start"),
    (":IN1", "Start 1"), (":IN0", "Start 1"),
]


def detect_score_ram(game_id, n_frames=600):
    """Auto-detect score and lives RAM addresses by watching what changes.

    Strategy:
    1. Start the game and play randomly for n_frames
    2. Read ALL of RAM each frame (scan a range)
    3. Score bytes: increment monotonically, change when game is active
    4. Lives bytes: start > 0, decrement on death (value 1-5 range)

    Returns (score_addrs, lives_addr) or ([], None) if not found.
    """
    print(f"[{game_id}] Scanning RAM for score/lives addresses...")

    # Read a broad RAM range — most arcade games keep state in $0000-$1FFF
    scan_range = range(0x0000, 0x1000)
    ram_dict = {f"_r{addr:04x}": addr for addr in scan_range}
    ram_dict["_dummy"] = 0x0000  # ensure non-empty

    env = MameEnv(ROMS_PATH, game_id, ram_dict,
                  render=False, sound=False, throttle=False)

    # Skip attract mode, insert coin, start
    env.wait(300)
    # Try common coin/start buttons
    for port, field in COIN_BUTTONS:
        try:
            env.step_n(port, field, 15)
            break
        except Exception:
            continue
    env.wait(60)
    for port, field in START_BUTTONS:
        try:
            env.step_n(port, field, 5)
            break
        except Exception:
            continue
    env.wait(120)

    # Collect RAM snapshots while playing randomly
    snapshots = []
    all_dirs = ALT_DIRECTIONS + [DIRECTION_BUTTONS.get(4, [])]
    working_dirs = None
    for variant in all_dirs:
        try:
            for port, field in variant:
                env.step(port, field)
            working_dirs = variant
            break
        except Exception:
            continue

    for frame in range(n_frames):
        if working_dirs and frame % 8 < 6:
            port, field = working_dirs[frame // 8 % len(working_dirs)]
            data = env.step(port, field)
        else:
            data = env.step()
        if frame % 10 == 0:
            snapshots.append({k: v for k, v in data.items() if k.startswith("_r")})

    env.close()

    if len(snapshots) < 10:
        print("  Not enough data collected")
        return [], None

    # Analyze each byte's behavior over time
    score_candidates = []
    lives_candidates = []

    for addr in scan_range:
        key = f"_r{addr:04x}"
        values = [s.get(key, 0) for s in snapshots]
        if all(v == values[0] for v in values):
            continue  # static — skip

        changes = sum(1 for i in range(1, len(values)) if values[i] != values[i-1])
        increases = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        decreases = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
        first = values[0]
        final = values[-1]
        unique = len(set(values))

        # Score heuristic — relaxed criteria:
        # 1. Changes at least twice
        # 2. More increases than decreases (score goes up mostly)
        # 3. Not a timer/counter that oscillates (unique values > 3)
        # 4. Doesn't look like a position (stays in small range) unless it grows
        if changes >= 2 and increases > decreases and unique >= 3:
            # Prefer bytes that end higher than they start
            growth = final - first if final > first else 0
            score_candidates.append((addr, increases, growth, changes, unique))

        # Lives heuristic — relaxed:
        # Starts 1-6, decreases at least once, values always 0-9
        if 1 <= first <= 6 and all(0 <= v <= 9 for v in values):
            if decreases >= 1 and increases <= 1:
                lives_candidates.append((addr, first, final, decreases))

    # Strategy 1: Find clusters of adjacent score bytes (BCD multi-digit)
    score_candidates.sort(key=lambda x: x[0])
    best_cluster = []
    current_cluster = []
    for entry in score_candidates:
        addr = entry[0]
        if current_cluster and addr == current_cluster[-1][0] + 1:
            current_cluster.append(entry)
        else:
            if len(current_cluster) > len(best_cluster):
                best_cluster = current_cluster
            current_cluster = [entry]
    if len(current_cluster) > len(best_cluster):
        best_cluster = current_cluster

    # Strategy 2: If no cluster, pick single best by growth
    if len(best_cluster) >= 2:
        score_addrs = [e[0] for e in best_cluster]
    else:
        # Sort by growth * increases (best overall score indicator)
        score_candidates.sort(key=lambda x: -(x[2] * x[1]))
        if score_candidates:
            score_addrs = [score_candidates[0][0]]
        else:
            score_addrs = []

    # Pick best lives candidate
    lives_addr = None
    if lives_candidates:
        # Prefer ones that start at common life counts (3, 5)
        lives_candidates.sort(key=lambda x: (
            -(x[3]),                    # most decreases
            -int(x[1] in (3, 5)),       # common starting lives
        ))
        lives_addr = lives_candidates[0][0]

    if score_addrs:
        print(f"  Score addresses: {[f'${a:04X}' for a in score_addrs]}")
    else:
        print("  No score addresses found")
    if lives_addr:
        print(f"  Lives address: ${lives_addr:04X}")
    else:
        print("  No lives address found")

    return score_addrs, lives_addr


class MamePixelEnv(gym.Env):
    """General MAME pixel-based gymnasium environment."""

    def __init__(self, game_id, render=False, throttle=False,
                 score_addrs=None, lives_addr=None):
        super().__init__()
        self.game_id = game_id
        self._render = render
        self._throttle = throttle
        self.env = None
        self._steps = 0
        self._frame_stack = None
        self._prev_frame = None
        self._score_addrs = score_addrs or []
        self._lives_addr = lives_addr
        self._prev_score = 0
        self._prev_lives = 0

        # Discover game controls
        ways, n_buttons = discover_inputs(game_id)
        print(f"[{game_id}] {ways}-way joystick, {n_buttons} buttons")

        # Build action list: directions + buttons + NOOP
        self._actions = []  # (port, field) pairs
        self._action_names = []
        # We'll try all direction variants and find which works at runtime
        self._direction_variants = ALT_DIRECTIONS + [DIRECTION_BUTTONS.get(ways, DIRECTION_BUTTONS[4])]
        self._n_buttons = n_buttons
        # Placeholder — real actions discovered on first reset
        self.action_space = spaces.Discrete(ways + n_buttons + 1)
        self.observation_space = spaces.Box(0, 255, (STACK_SIZE, OBS_H, OBS_W), np.uint8)

    def _start_mame(self):
        subprocess.run(["pkill", "-f", f"mame.*{self.game_id}"],
                       capture_output=True)
        time.sleep(0.5)
        # Build RAM address dict — always include score/lives if detected
        ram = {"_dummy": 0x0000}
        if self._score_addrs:
            for i, addr in enumerate(self._score_addrs):
                ram[f"_score{i}"] = addr
        if self._lives_addr:
            ram["_lives"] = self._lives_addr
        self.env = MameEnv(
            ROMS_PATH, self.game_id, ram,
            render=self._render, sound=False, throttle=self._throttle,
        )
        # Discover which button names work by trying them
        self._discover_working_buttons()

    def _discover_working_buttons(self):
        """Discover game inputs by querying MAME's Lua ioport at runtime."""
        # Query all port:field pairs via Lua console
        result = self.env.console.writeln_expect(
            'local r = {}; '
            'for pname, port in pairs(manager.machine.ioport.ports) do '
            'for fname, field in pairs(port.fields) do '
            'r[#r+1] = pname.."|"..fname; '
            'end; end; '
            'print(table.concat(r, ";"))'
        )

        self._coin = None
        self._start = None
        self._actions = []
        self._action_names = []

        if result:
            for pair in result.split(";"):
                if "|" not in pair:
                    continue
                port, field = pair.split("|", 1)
                fl = field.lower()

                # Player 1 directions
                if "p1" in fl and any(d in fl for d in ["up", "down", "left", "right"]):
                    self._actions.append((port, field))
                    self._action_names.append(field)
                # Player 1 joystick
                elif "p1" in fl and "joystick" in fl:
                    self._actions.append((port, field))
                    self._action_names.append(field)
                # Player 1 buttons
                elif "p1" in fl and "button" in fl:
                    self._actions.append((port, field))
                    self._action_names.append(field)
                # Coin
                elif "coin" in fl and ("1" in field or self._coin is None):
                    self._coin = (port, field)
                # Start
                elif ("1 player" in fl or "start 1" in fl
                      or ("start" in fl and "1" in fl)):
                    self._start = (port, field)

        self.action_space = spaces.Discrete(len(self._actions) + 1)  # +1 NOOP
        print(f"[{self.game_id}] Actions: {self._action_names} + NOOP")
        if self._coin:
            print(f"  Coin: {self._coin[1]}")
        if self._start:
            print(f"  Start: {self._start[1]}")

    def _get_frame(self):
        """Get one frame from MAME."""
        self.env.request_frame()
        data = self.env.step()
        if "frame" in data:
            return data["frame"]
        return None

    def _preprocess(self, rgb):
        """RGB frame → 84x84 grayscale."""
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        img = Image.fromarray(gray).resize((OBS_W, OBS_H), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.env is None:
            self._start_mame()

        # Insert coin + start
        if self._coin:
            self.env.step_n(*self._coin, 15)
        self.env.wait(120)
        if self._start:
            self.env.step_n(*self._start, 5)
        self.env.wait(120)

        frame = self._get_frame()
        if frame is not None:
            processed = self._preprocess(frame)
        else:
            processed = np.zeros((OBS_H, OBS_W), dtype=np.uint8)

        self._frame_stack = np.stack([processed] * STACK_SIZE)
        self._prev_frame = processed.copy()
        self._steps = 0
        self._prev_score = 0
        # Read initial lives
        data = self.env.step()
        self._prev_lives = data.get("_lives", 0)
        return self._frame_stack.copy(), {}

    def step(self, action):
        self._steps += 1
        action = int(action)

        if action < len(self._actions):
            port, field = self._actions[action]
            buttons = (port, field)
        else:
            buttons = None  # NOOP

        # Frame skip: repeat action, grab frame on last step
        for i in range(FRAME_SKIP):
            if i == FRAME_SKIP - 1:
                self.env.request_frame()
            if buttons:
                self.env.step(*buttons)
            else:
                self.env.step()
        # request_frame causes next step to return pixels
        data = self.env.step()
        if "frame" in data:
            processed = self._preprocess(data["frame"])
        else:
            processed = self._prev_frame

        # Compute reward
        reward = 0.0
        terminated = False

        if self._score_addrs:
            # Score-based reward (much better signal)
            score = sum(data.get(f"_score{i}", 0) << (8 * i)
                        for i in range(len(self._score_addrs)))
            delta = score - self._prev_score
            if delta > 0:
                reward += min(delta / 100.0, 10.0)  # cap to avoid huge spikes
            elif delta < -1000:
                # Score wrapped or reset — likely new game
                pass
            self._prev_score = score

        if self._lives_addr:
            # Lives-based death detection
            lives = data.get("_lives", 0)
            if lives < self._prev_lives and self._prev_lives > 0:
                reward -= 5.0  # death penalty
            if lives == 0 and self._prev_lives > 0:
                terminated = True  # game over
            self._prev_lives = lives

        if not self._score_addrs:
            # Fallback: pixel change heuristic
            diff = np.mean(np.abs(
                processed.astype(float) - self._prev_frame.astype(float)
            ))
            if diff > 30:
                reward -= 1.0
            elif diff > 3:
                reward += 0.1

        reward -= 0.01  # small step penalty

        self._prev_frame = processed.copy()
        self._frame_stack = np.roll(self._frame_stack, -1, axis=0)
        self._frame_stack[-1] = processed

        truncated = self._steps > 5000
        return self._frame_stack.copy(), reward, terminated, truncated, {}

    def close(self):
        if self.env:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None


MODELS = {
    "ppo": ("stable_baselines3", "PPO", "CnnPolicy", dict(
        learning_rate=2.5e-4, n_steps=128, batch_size=32, n_epochs=4,
        gamma=0.99, clip_range=0.1, ent_coef=0.01,
    )),
    "dqn": ("stable_baselines3", "DQN", "CnnPolicy", dict(
        learning_rate=1e-4, buffer_size=50000, learning_starts=1000,
        batch_size=32, gamma=0.99, exploration_fraction=0.1,
        exploration_final_eps=0.01,
    )),
    "a2c": ("stable_baselines3", "A2C", "CnnPolicy", dict(
        learning_rate=7e-4, n_steps=5, gamma=0.99, ent_coef=0.01,
    )),
}


def make_model(model_name, env):
    """Create an SB3 model by name."""
    import importlib
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODELS.keys())}")
    module_name, class_name, policy, kwargs = MODELS[model_name]
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(policy, env, verbose=1, device="cpu", **kwargs)


def load_model(model_name, path):
    """Load a saved SB3 model by name."""
    import importlib
    module_name, class_name, _, _ = MODELS[model_name]
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls.load(path)


def train(game_id, model_name, timesteps, save_path, score_addrs=None, lives_addr=None):
    env = MamePixelEnv(game_id, render=False, throttle=False,
                       score_addrs=score_addrs, lives_addr=lives_addr)
    model = make_model(model_name, env)
    reward_src = "score RAM" if score_addrs else "pixel heuristic"
    print(f"Training {game_id} with {model_name.upper()} for {timesteps} steps "
          f"(reward: {reward_src})...")
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"Saved to {save_path}")
    env.close()


def evaluate(game_id, model_name, model_path, episodes, score_addrs=None, lives_addr=None):
    env = MamePixelEnv(game_id, render=True, throttle=True,
                       score_addrs=score_addrs, lives_addr=lives_addr)
    model = load_model(model_name, model_path)
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward, steps = 0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            steps += 1
            if term or trunc:
                break
        print(f"Episode {ep+1}: reward={total_reward:.1f}, steps={steps}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General MAME RL agent (pixel-based)")
    parser.add_argument("game", help="MAME ROM name (e.g. qbert, pacman, dkong)")
    parser.add_argument("--model", choices=["ppo", "dqn", "a2c"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--eval", type=str, default=None, help="Evaluate saved model")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--detect-score", action="store_true",
                        help="Auto-detect score/lives RAM before training")
    parser.add_argument("--score-addr", type=str, default=None,
                        help="Manual score RAM address(es), comma-separated hex (e.g. 0x00BE)")
    parser.add_argument("--lives-addr", type=str, default=None,
                        help="Manual lives RAM address, hex (e.g. 0x0D00)")
    args = parser.parse_args()

    save_path = args.save or f"{args.game}_{args.model}"
    score_addrs, lives_addr = [], None

    if args.score_addr:
        score_addrs = [int(a.strip(), 16) for a in args.score_addr.split(",")]
        print(f"Using manual score addresses: {[f'${a:04X}' for a in score_addrs]}")
    if args.lives_addr:
        lives_addr = int(args.lives_addr, 16)
        print(f"Using manual lives address: ${lives_addr:04X}")

    # Check known game configs
    if not score_addrs:
        import json
        config_path = os.path.join(os.path.dirname(__file__), "game_configs.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                configs = json.load(f)
            if args.game in configs:
                cfg = configs[args.game]
                score_addrs = [int(a, 16) for a in cfg.get("score_addrs", [])]
                if not lives_addr and "lives_addr" in cfg:
                    lives_addr = int(cfg["lives_addr"], 16)
                print(f"[{args.game}] Loaded known config: "
                      f"score={[f'${a:04X}' for a in score_addrs]}, "
                      f"lives=${lives_addr:04X}" if lives_addr else "")

    # Auto-detect as last resort
    if args.detect_score and not score_addrs:
        score_addrs, lives_addr = detect_score_ram(args.game)

    if args.eval:
        evaluate(args.game, args.model, args.eval, args.episodes,
                 score_addrs, lives_addr)
    else:
        train(args.game, args.model, args.timesteps, save_path,
              score_addrs, lives_addr)
