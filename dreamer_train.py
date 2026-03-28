"""Train Dreamer v3 (full implementation via sheeprl) on MAME games.

Requires Python 3.11 venv: source venv311/bin/activate

Usage:
    python dreamer_train.py frogger --timesteps 500000
    python dreamer_train.py galaga --timesteps 1000000
"""

import sys
import os
import json
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our MAME env
from mle import MameEnv
from mle_general import MamePixelEnv, ROMS_PATH, OBS_H, OBS_W, STACK_SIZE

# Register MAME env with gymnasium
class MameGymWrapper(gym.Env):
    """Wrap MamePixelEnv for sheeprl compatibility.
    Returns (H, W, C) RGB observations. Handles pipe robustness.
    """
    metadata = {"render_modes": ["rgb_array"]}
    render_mode = "rgb_array"

    def __init__(self, game_id="frogger", **kwargs):
        super().__init__()
        self.game_id = game_id
        self._game_cfg = {}
        config_path = os.path.join(os.path.dirname(__file__), "game_configs.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                configs = json.load(f)
            if game_id in configs:
                self._game_cfg = configs[game_id]

        # sheeprl wants (H, W, C) RGB
        self.observation_space = spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
        self._obs_shape = (64, 64, 3)
        self.env = None
        self.action_space = spaces.Discrete(5)
        self._keepalive_running = False

    def _make_env(self):
        score_addrs = [int(a, 16) for a in self._game_cfg.get("score_addrs", [])]
        lives_addr = int(self._game_cfg["lives_addr"], 16) if "lives_addr" in self._game_cfg else None
        score_encoding = self._game_cfg.get("encoding")
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        import subprocess, time
        subprocess.run(["pkill", "-9", "-f", f"mame.*{self.game_id}"],
                       capture_output=True)
        time.sleep(0.5)
        self.env = MamePixelEnv(
            self.game_id, render=False, throttle=False,
            score_addrs=score_addrs, lives_addr=lives_addr,
            score_encoding=score_encoding)
        self.action_space = self.env.action_space
        self._keepalive_running = False

    def _to_rgb(self, obs):
        """Convert (C, H, W) grayscale stack to (H, W, 3) RGB for sheeprl."""
        from PIL import Image
        # Take last frame from stack, resize to 64x64, convert to RGB
        frame = obs[-1]  # (84, 84) uint8
        img = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
        gray = np.array(img, dtype=np.uint8)
        return np.stack([gray, gray, gray], axis=-1)  # (64, 64, 3)

    def reset(self, seed=None, options=None):
        self._keepalive_running = False
        if hasattr(self, '_keepalive_thread'):
            self._keepalive_thread.join(timeout=2)
        if self.env is None:
            self._make_env()
        try:
            obs, info = self.env.reset(seed=seed)
        except Exception:
            self._make_env()
            obs, info = self.env.reset(seed=seed)
        self.action_space = self.env.action_space
        self._start_keepalive()
        return self._to_rgb(obs), info

    def step(self, action):
        # Stop keepalive, do step, restart
        self._keepalive_running = False
        if hasattr(self, '_keepalive_thread'):
            self._keepalive_thread.join(timeout=1)
        try:
            obs, reward, term, trunc, info = self.env.step(int(action))
        except Exception:
            self._make_env()
            obs, _ = self.env.reset()
            self._start_keepalive()
            return self._to_rgb(obs), -1.0, True, False, {}
        self._start_keepalive()
        return self._to_rgb(obs), float(reward), bool(term), bool(trunc), info

    def _start_keepalive(self):
        import threading
        self._keepalive_running = True
        def _ka():
            while self._keepalive_running:
                try:
                    self.env.env.step()
                except Exception:
                    break
        self._keepalive_thread = threading.Thread(target=_ka, daemon=True)
        self._keepalive_thread.start()

    def render(self):
        return np.zeros(self._obs_shape, dtype=np.uint8)

    def close(self):
        self._keepalive_running = False
        if self.env:
            try:
                self.env.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("game", help="MAME ROM name")
    parser.add_argument("--timesteps", type=int, default=500000)
    args = parser.parse_args()

    # Register the env
    gym.register(
        id="MameArcade-v0",
        entry_point="dreamer_train:MameGymWrapper",
        kwargs={"game_id": args.game},
    )

    # sheeprl runs via hydra which spawns fresh Python.
    # We need the env registered in THAT process.
    # Solution: write a temp script that registers + runs sheeprl.
    import tempfile, subprocess
    script = f"""
import sys, os
sys.path.insert(0, '{os.path.dirname(os.path.abspath(__file__))}')
import gymnasium as gym
from dreamer_train import MameGymWrapper
gym.register(id="MameArcade-v0", entry_point="dreamer_train:MameGymWrapper",
             kwargs={{"game_id": "{args.game}"}},
             max_episode_steps=1000)

# Now run sheeprl
from sheeprl.cli import run
sys.argv = [
    "sheeprl",
    "exp=dreamer_v3",
    "env=gym",
    "env.id=MameArcade-v0",
    "env.wrapper.id=MameArcade-v0",
    "env.wrapper.render_mode=rgb_array",
    "algo.total_steps={args.timesteps}",
    "algo.replay_ratio=0.5",
    "buffer.size=100000",
    "env.num_envs=1",
    "env.capture_video=false",
    "env.sync_env=true",
    "env.screen_size=64",
    "env.grayscale=false",
    "env.frame_stack=1",
    "env.action_repeat=1",
    "metric.log_every=1000",
    "checkpoint.every=50000",
    "algo.run_test=false",
]
run()
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    print(f"Starting sheeprl Dreamer v3 for {args.game}...")
    subprocess.run([sys.executable, script_path])
    os.unlink(script_path)


if __name__ == "__main__":
    main()
