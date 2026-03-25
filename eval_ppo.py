"""Evaluate a trained PPO agent on Q*bert."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from qbert_gym import QbertEnv


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    env = QbertEnv(render=not args.no_render, throttle=not args.no_render)
    model = PPO.load(args.model)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        prev_cubes = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            # Print progress every 100 steps
            if steps % 100 == 0:
                s = env.state
                cubes = sum(1 for i in range(28) if s.cube_states[i] == s.target_color)
                print(f"  step {steps}: cubes={cubes}/28 lives={s.lives} level={env.current_level} reward={total_reward:.0f}")
            if terminated or truncated:
                break
        s = env.state
        cubes = sum(1 for i in range(28) if s.cube_states[i] == s.target_color)
        print(f"Episode {ep+1}: reward={total_reward:.1f}, steps={steps}, level={env.current_level}, cubes={cubes}")

    env.close()


if __name__ == "__main__":
    main()
