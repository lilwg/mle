"""General-purpose MAME RL agent. Works with any arcade game ROM.

Usage:
    python3 mle_general.py pacman          # Train on Pac-Man
    python3 mle_general.py qbert           # Train on Q*bert
    python3 mle_general.py galaga          # Train on Galaga
    python3 mle_general.py dkong --eval model.zip  # Evaluate saved model

No game-specific code. Uses pixels for observations and auto-detects
inputs via MAME XML metadata.
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

# Common alternate port names (MAME varies per driver)
ALT_DIRECTIONS = [
    # Q*bert style
    [(":IN4", "P1 Up (Up-Right)"), (":IN4", "P1 Down (Down-Left)"),
     (":IN4", "P1 Left (Up-Left)"), (":IN4", "P1 Right (Down-Right)")],
    # Standard :IN0
    [(":IN0", "P1 Up"), (":IN0", "P1 Down"),
     (":IN0", "P1 Left"), (":IN0", "P1 Right")],
    # Standard :IN1
    [(":IN1", "P1 Up"), (":IN1", "P1 Down"),
     (":IN1", "P1 Left"), (":IN1", "P1 Right")],
]

BUTTON_NAMES = [
    (":IN4", "P1 Button 1"), (":IN0", "P1 Button 1"),
    (":IN1", "P1 Button 1"), (":IN2", "P1 Button 1"),
]

COIN_BUTTONS = [
    (":IN5", "Coin 1"), (":IN0", "Coin 1"), (":IN1", "Coin 1"),
    (":IN6", "Coin 1"), (":IN7", "Coin 1"),
]

START_BUTTONS = [
    (":IN5", "1 Player Start"), (":IN0", "1 Player Start"),
    (":IN1", "1 Player Start"), (":IN4", "1 Player Start"),
    (":IN6", "1 Player Start"),
]


class MamePixelEnv(gym.Env):
    """General MAME pixel-based gymnasium environment."""

    def __init__(self, game_id, render=False, throttle=False):
        super().__init__()
        self.game_id = game_id
        self._render = render
        self._throttle = throttle
        self.env = None
        self._steps = 0
        self._frame_stack = None
        self._prev_frame = None

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
        # Need at least one RAM address to keep the pipe protocol working
        self.env = MameEnv(
            ROMS_PATH, self.game_id, {"_dummy": 0x0000},
            render=self._render, sound=False, throttle=self._throttle,
        )
        # Discover which button names work by trying them
        self._discover_working_buttons()

    def _discover_working_buttons(self):
        """Try different port/field combos to find ones that work."""
        # For coin/start, try each until one doesn't error
        self._coin = None
        self._start = None
        for port, field in COIN_BUTTONS:
            try:
                self.env.step(port, field)
                self._coin = (port, field)
                break
            except Exception:
                continue
        for port, field in START_BUTTONS:
            try:
                self.env.step(port, field)
                self._start = (port, field)
                break
            except Exception:
                continue

        # For directions, try each variant set
        self._actions = []
        self._action_names = []
        for variant in self._direction_variants:
            try:
                for port, field in variant:
                    self.env.step(port, field)
                # All worked — use this variant
                self._actions = list(variant)
                self._action_names = [f[1] for f in variant]
                break
            except Exception:
                continue

        # Add buttons
        for port, field in BUTTON_NAMES[:self._n_buttons]:
            try:
                self.env.step(port, field)
                self._actions.append((port, field))
                self._action_names.append(field)
            except Exception:
                continue

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

        # Reward: pixel change magnitude as activity proxy
        diff = np.mean(np.abs(processed.astype(float) - self._prev_frame.astype(float)))
        reward = 0.0
        if diff > 30:
            reward = -1.0  # likely death (big flash/transition)
        elif diff > 3:
            reward = 0.1   # something happening (progress)
        reward -= 0.01     # small step penalty

        self._prev_frame = processed.copy()
        self._frame_stack = np.roll(self._frame_stack, -1, axis=0)
        self._frame_stack[-1] = processed

        truncated = self._steps > 5000
        return self._frame_stack.copy(), reward, False, truncated, {}

    def close(self):
        if self.env:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None


def train(game_id, timesteps, save_path):
    from stable_baselines3 import PPO
    env = MamePixelEnv(game_id, render=False, throttle=False)
    model = PPO("CnnPolicy", env, learning_rate=2.5e-4, n_steps=128,
                batch_size=32, n_epochs=4, gamma=0.99, clip_range=0.1,
                ent_coef=0.01, verbose=1, device="cpu")
    print(f"Training {game_id} for {timesteps} steps...")
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"Saved to {save_path}")
    env.close()


def evaluate(game_id, model_path, episodes):
    from stable_baselines3 import PPO
    env = MamePixelEnv(game_id, render=True, throttle=True)
    model = PPO.load(model_path)
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
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--eval", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    save_path = args.save or f"{args.game}_ppo"
    if args.eval:
        evaluate(args.game, args.eval, args.episodes)
    else:
        train(args.game, args.timesteps, save_path)
