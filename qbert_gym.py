"""Gymnasium wrapper for Q*bert via MAME Learning Environment.

Uses structured RAM state (~78 dimensions), not pixels.
Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=NOOP
"""

import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mle import MameEnv
from qbert.state import (
    QBERT_RAM, read_state, is_valid, EnemyTracker,
    NUM_CUBES, MAX_ROW, pos_to_cube_index,
)
from qbert.sim import MOVE_DELTAS, UP, DOWN, LEFT, RIGHT
from qbert.planner import neighbors, MOVE_BUTTONS, COIN_BUTTON, START_BUTTON

ROMS_PATH = "/Users/pat/mame/roms"
NOOP = 4
BUTTON_HOLD = 6
NUM_ENEMY_SLOTS = 10
OBS_DIM = 2 + 28 + (NUM_ENEMY_SLOTS * 4) + 4 + 4  # = 78


class QbertEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False, throttle=False):
        super().__init__()
        self.render_mode = "human" if render else None
        self._render = render
        self._throttle = throttle
        self.action_space = spaces.Discrete(5)  # UP, DOWN, LEFT, RIGHT, NOOP
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.env = None
        self.tracker = EnemyTracker()
        self.data = None
        self.state = None
        self.prev_lives = 0
        self.prev_colored = 0
        self.prev_coily_active = False
        self.steps = 0
        self.current_level = 1

    def _start_mame(self):
        """Launch MAME and get to gameplay. Retries on failure."""
        import subprocess, time
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
            time.sleep(0.5)
        # Kill any stale MAME processes
        subprocess.run(["pkill", "-f", "mame.*qbert"], capture_output=True)
        time.sleep(0.5)

        for attempt in range(3):
            try:
                self.env = MameEnv(
                    ROMS_PATH, "qbert", QBERT_RAM,
                    render=self._render, sound=False, throttle=self._throttle,
                )
                # Insert coin and start game
                self.env.wait(600)
                self.env.step_n(*COIN_BUTTON, 15)
                self.env.wait(180)
                self.env.step_n(*START_BUTTON, 5)
                # Wait for game to start
                self.data = self.env.step()
                self.state = read_state(self.data, self.tracker)
                for _ in range(900):
                    if (0 < self.state.lives <= 5
                            and is_valid(self.state.qbert[0], self.state.qbert[1])):
                        return  # success
                    self.data = self.env.step()
                    self.state = read_state(self.data, self.tracker)
                return  # got past wait loop
            except Exception as e:
                print(f"MAME start attempt {attempt+1} failed: {e}")
                try:
                    if self.env:
                        self.env.close()
                except Exception:
                    pass
                self.env = None
                subprocess.run(["pkill", "-f", "mame.*qbert"], capture_output=True)
                time.sleep(1)
        raise RuntimeError("Failed to start MAME after 3 attempts")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tracker = EnemyTracker()

        if self.env is None:
            # First reset — launch MAME
            self._start_mame()
        else:
            # Subsequent resets — just insert coin and start new game
            # (much faster than restarting MAME)
            try:
                self.env.wait(60)
                self.env.step_n(*COIN_BUTTON, 15)
                self.env.wait(60)
                self.env.step_n(*START_BUTTON, 5)
                # Wait for game to start
                self.data = self.env.step()
                self.state = read_state(self.data, self.tracker)
                for _ in range(900):
                    if (0 < self.state.lives <= 5
                            and is_valid(self.state.qbert[0], self.state.qbert[1])):
                        break
                    self.data = self.env.step()
                    self.state = read_state(self.data, self.tracker)
            except Exception:
                # Pipe broke — restart MAME
                self._start_mame()

        self.prev_lives = self.state.lives
        self.prev_colored = self._count_colored()
        self.prev_coily_active = False
        self.steps = 0
        self.current_level = 1
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        action = int(action)

        # Execute action
        if action == NOOP:
            # Advance a few frames
            for _ in range(BUTTON_HOLD):
                self.data = self.env.step()
        else:
            row, col = self.state.qbert
            dr, dc = MOVE_DELTAS[action]
            nr, nc = row + dr, col + dc
            if not is_valid(nr, nc):
                # Invalid move — treat as NOOP with penalty
                for _ in range(BUTTON_HOLD):
                    self.data = self.env.step()
                self.state = read_state(self.data, self.tracker)
                return self._get_obs(), -0.5, False, False, {"invalid": True}

            # Valid move — hold button then wait for landing
            port, field = MOVE_BUTTONS[action]
            self.data = self.env.step_n(port, field, BUTTON_HOLD)
            for _ in range(25):
                self.data = self.env.step()
                if self.data.get("qb_anim", 0) >= 16:
                    break

        self.state = read_state(self.data, self.tracker)
        reward = self._compute_reward()
        terminated = self.state.lives == 0
        truncated = self.steps > 5000

        # Handle level transitions
        if self.state.remaining_cubes == 0 and not terminated:
            self.current_level += 1
            self.tracker.reset()
            # Wait for next level to start
            for _ in range(300):
                self.data = self.env.step()
                self.state = read_state(self.data, self.tracker)
                if (self.state.remaining_cubes > 0
                        and is_valid(self.state.qbert[0], self.state.qbert[1])
                        and self.data.get("qb_anim", 0) >= 16):
                    break

        # Handle death transitions
        if self.state.lives < self.prev_lives and self.state.lives > 0:
            self.tracker.reset()
            for _ in range(300):
                self.data = self.env.step()
                self.state = read_state(self.data, self.tracker)
                if (is_valid(self.state.qbert[0], self.state.qbert[1])
                        and self.data.get("qb_anim", 0) >= 16):
                    break

        # Update state tracking
        self.prev_lives = self.state.lives
        self.prev_colored = self._count_colored()
        coily_now = any(e.etype == "coily" and not e.harmless
                        for e in self.state.enemies)
        self.prev_coily_active = coily_now

        return self._get_obs(), reward, terminated, truncated, {}

    def _compute_reward(self):
        reward = 0.0

        # Cube coloring progress (dense signal)
        new_colored = self._count_colored()
        cubes_gained = new_colored - self.prev_colored
        if cubes_gained > 0:
            reward += cubes_gained * 10.0

        # Death penalty
        if self.state.lives < self.prev_lives:
            reward -= 50.0

        # Level complete bonus
        if self.state.remaining_cubes == 0:
            reward += 100.0

        # Disc kill bonus (Coily was active, now gone)
        coily_now = any(e.etype == "coily" and not e.harmless
                        for e in self.state.enemies)
        if self.prev_coily_active and not coily_now:
            reward += 30.0

        # Small step penalty (encourages efficiency)
        reward -= 0.1

        return reward

    def _count_colored(self):
        """Count cubes that have reached target_color."""
        count = 0
        for i in range(NUM_CUBES):
            if self.state.cube_states[i] == self.state.target_color:
                count += 1
        return count

    def _get_obs(self):
        """Build observation vector from game state."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0
        s = self.state

        # Q*bert position (normalized 0-1)
        if is_valid(s.qbert[0], s.qbert[1]):
            obs[idx] = s.qbert[0] / MAX_ROW
            obs[idx + 1] = s.qbert[1] / MAX_ROW
        idx += 2

        # Cube states (28 values: 1 if colored, 0 if not)
        for i in range(NUM_CUBES):
            obs[idx + i] = 1.0 if s.cube_states[i] == s.target_color else 0.0
        idx += NUM_CUBES

        # Enemy slots (10 × 4 = 40)
        enemies_by_slot = {e.slot: e for e in s.enemies}
        for slot in range(NUM_ENEMY_SLOTS):
            base = idx + slot * 4
            if slot in enemies_by_slot:
                e = enemies_by_slot[slot]
                obs[base] = 1.0  # active
                obs[base + 1] = e.pos[0] / MAX_ROW if is_valid(e.pos[0], e.pos[1]) else 0.0
                obs[base + 2] = e.pos[1] / MAX_ROW if is_valid(e.pos[0], e.pos[1]) else 0.0
                obs[base + 3] = 1.0 if e.etype == "coily" else 0.0
        idx += NUM_ENEMY_SLOTS * 4

        # Disc availability (4: left rows, right rows — simplified to 2+2)
        for d in s.discs:
            if d.side == "left":
                obs[idx] = 1.0
            else:
                obs[idx + 1] = 1.0
        obs[idx + 2] = min(len(s.discs), 4) / 4.0  # disc count normalized
        obs[idx + 3] = 0.0  # reserved
        idx += 4

        # Scalar features
        obs[idx] = s.lives / 5.0
        obs[idx + 1] = s.remaining_cubes / NUM_CUBES
        # Coily distance
        coily_d = 12.0
        for e in s.enemies:
            if e.etype == "coily" and not e.harmless and is_valid(e.pos[0], e.pos[1]):
                from test_survive import grid_dist
                d = grid_dist(s.qbert[0], s.qbert[1], e.pos[0], e.pos[1])
                coily_d = min(coily_d, d)
                break
        obs[idx + 2] = coily_d / 12.0
        obs[idx + 3] = self.current_level / 10.0

        return obs

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None


if __name__ == "__main__":
    # Quick test
    env = QbertEnv(render=False, throttle=False)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Obs sample: {obs[:10]}")

    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode done at step {i}, total reward: {total_reward:.1f}")
            break
    else:
        print(f"100 steps done, total reward: {total_reward:.1f}")

    env.close()
    print("OK")
