"""Train Dreamer v3 on retro games via stable-retro (FBNeo/Genesis).

Much faster than MAME pipes — emulator runs in-process.

Usage:
    python retro_train.py Frogger-Genesis --timesteps 500000
    python retro_train.py MortalKombat2-Arcade --timesteps 1000000
"""

import sys
import os
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import stable_retro
except ImportError:
    print("Install stable-retro: pip install stable-retro")
    sys.exit(1)


class RetroGymWrapper(gym.Env):
    """Wrap stable-retro env for sheeprl compatibility.
    Returns (64, 64, 3) RGB observations.
    """
    metadata = {"render_modes": ["rgb_array"]}
    render_mode = "rgb_array"

    def __init__(self, game_id="Frogger-Genesis-v0", **kwargs):
        super().__init__()
        self.game_id = game_id
        self.observation_space = spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(126)  # Genesis discrete actions
        self._retro_env = None

    def _make_env(self):
        if self._retro_env is not None:
            try:
                self._retro_env.close()
            except Exception:
                pass
        self._retro_env = stable_retro.make(
            game=self.game_id,
            use_restricted_actions=stable_retro.Actions.DISCRETE,
            render_mode=None
        )

    def _resize(self, obs):
        from PIL import Image
        img = Image.fromarray(obs).resize((64, 64), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        if self._retro_env is None:
            self._make_env()
        result = self._retro_env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        info = result[1] if isinstance(result, tuple) and len(result) > 1 else {}
        return self._resize(obs), info

    def step(self, action):
        result = self._retro_env.step(int(action))
        obs, reward, terminated, truncated, info = result[0], result[1], result[2], result[3], result[4]
        return self._resize(obs), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self):
        if self._retro_env:
            try:
                self._retro_env.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("game", help="stable-retro game name (e.g. Frogger-Genesis)")
    parser.add_argument("--timesteps", type=int, default=500000)
    args = parser.parse_args()

    # Register the env
    gym.register(
        id="RetroArcade-v0",
        entry_point="retro_train:RetroGymWrapper",
        kwargs={"game_id": args.game},
        max_episode_steps=4500,  # 75 seconds at 60fps
    )

    print(f"Starting sheeprl Dreamer v3 for {args.game}...")
    from sheeprl.cli import run
    sys.argv = [
        "sheeprl",
        "exp=dreamer_v3",
        "env=gym",
        "env.id=RetroArcade-v0",
        "env.wrapper.id=RetroArcade-v0",
        "env.wrapper.render_mode=rgb_array",
        f"algo.total_steps={args.timesteps}",
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
        "fabric.accelerator=auto",
    ]
    run()


if __name__ == "__main__":
    main()
