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
    sheeprl expects obs as (H, W, C) uint8, not (C, H, W).
    """
    metadata = {"render_modes": ["rgb_array"]}
    render_mode = "rgb_array"

    def __init__(self, game_id="frogger", **kwargs):
        super().__init__()
        config_path = os.path.join(os.path.dirname(__file__), "game_configs.json")
        score_addrs, lives_addr, score_encoding = [], None, None
        if os.path.exists(config_path):
            with open(config_path) as f:
                configs = json.load(f)
            if game_id in configs:
                cfg = configs[game_id]
                score_addrs = [int(a, 16) for a in cfg.get("score_addrs", [])]
                if "lives_addr" in cfg:
                    lives_addr = int(cfg["lives_addr"], 16)
                score_encoding = cfg.get("encoding")

        self.env = MamePixelEnv(
            game_id, render=False, throttle=False,
            score_addrs=score_addrs, lives_addr=lives_addr,
            score_encoding=score_encoding)

        n_actions = self.env.action_space.n
        self.action_space = spaces.Discrete(n_actions)
        # sheeprl expects (H, W, C) observations
        self.observation_space = spaces.Box(
            0, 255, (OBS_H, OBS_W, STACK_SIZE), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        # (C, H, W) -> (H, W, C)
        return obs.transpose(1, 2, 0), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return obs.transpose(1, 2, 0), reward, term, trunc, info

    def render(self):
        return np.zeros((OBS_H, OBS_W, 3), dtype=np.uint8)

    def close(self):
        self.env.close()


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
    "env.wrapper.id=MameArcade-v0",
    "env.wrapper.render_mode=rgb_array",
    "algo.total_steps={args.timesteps}",
    "algo.replay_ratio=0.5",
    "buffer.size=100000",
    "env.num_envs=1",
    "metric.log_every=1000",
    "checkpoint.every=50000",
    "env.capture_video=false",
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
