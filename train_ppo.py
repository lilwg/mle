"""Train PPO agent on Q*bert using structured RAM state."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from qbert_gym import QbertEnv


class LogCallback(BaseCallback):
    """Log episode stats during training."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        # Check for episode completion in infos
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                n = len(self.episode_rewards)
                if n % 10 == 0:
                    recent = self.episode_rewards[-10:]
                    print(f"  Episode {n}: avg reward={sum(recent)/len(recent):.1f}, "
                          f"avg length={sum(self.episode_lengths[-10:])/10:.0f}")
        return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps")
    parser.add_argument("--render", action="store_true",
                        help="Render during training (slow)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from saved model")
    parser.add_argument("--save", type=str, default="qbert_ppo",
                        help="Save model path")
    args = parser.parse_args()

    print(f"Training PPO for {args.timesteps} timesteps...")
    env = QbertEnv(render=args.render, throttle=args.render)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=256,          # steps per rollout (shorter for faster updates)
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./tb_logs/",
            device="auto",        # Uses CUDA > MPS > CPU automatically
            policy_kwargs=dict(
                net_arch=[128, 128],
            ),
        )

    callback = LogCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.save)
    print(f"Model saved to {args.save}")
    env.close()


if __name__ == "__main__":
    main()
