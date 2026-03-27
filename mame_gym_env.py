"""MAME Gymnasium environment registration.

Import this module to register MameArcade-v0 with gymnasium.
"""
import gymnasium as gym
from gymnasium.envs.registration import register

# Register on import
register(
    id="MameArcade-v0",
    entry_point="dreamer_train:MameGymWrapper",
)
