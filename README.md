# MLE — MAME Learning Environment

Train RL agents on any arcade game via MAME. No game-specific code needed for the general agent. Includes a ROM-accurate Q*bert specialist agent.

## Quick Start

```bash
pip3 install torch gymnasium stable-baselines3 pillow
```

### Train on any arcade game (pixel-based)

```bash
python3 mle_general.py pacman                    # Train Pac-Man
python3 mle_general.py dkong --timesteps 500000  # Train Donkey Kong
python3 mle_general.py galaga                    # Train Galaga

# Watch a trained model play
python3 mle_general.py qbert --eval qbert_ppo.zip
```

Auto-detects game controls from MAME metadata. Uses 84×84 grayscale frame stacks with CNN policy. Just drop a ROM in `~/mame/roms/` and go.

### Train Q*bert with structured RAM state (faster)

```bash
python3 train_ppo.py --timesteps 500000          # ~23 min, learns to color cubes
python3 train_ppo.py --timesteps 2000000 --resume qbert_ppo_500k --save qbert_ppo_2M

# Evaluate
python3 eval_ppo.py qbert_ppo_500k               # Watch it play
python3 eval_ppo.py qbert_ppo_500k --no-render    # Headless stats
```

Uses 78-dimensional structured state from RAM (Q*bert position, 28 cube states, 10 enemy slots, disc availability). MLP policy trains 10-100x faster than pixels.

### Run the handcrafted Q*bert bot

```bash
python3 qbert_bot.py
```

ROM-accurate frame simulation, depth-6 lookahead with unified scoring. Completes 3+ levels.

## Architecture

```
mle/
├── mle/              # MAME Learning Environment core
│   ├── env.py        # MameEnv — synchronous Python↔MAME bridge via named pipes
│   └── console.py    # MAME process management + Lua console
├── qbert/            # Q*bert game-specific modules (ROM-verified)
│   ├── state.py      # RAM reader, entity classification, disc detection
│   ├── frame_sim.py  # Frame-perfect tick-by-tick simulation
│   ├── predict.py    # Coily chase + ball path prediction
│   ├── collision.py  # ROM $BD1E collision detection
│   ├── spawn.py      # Spawn timer prediction
│   ├── sim.py        # Move deltas and constants
│   └── planner.py    # Neighbors, button mappings
├── mle_general.py    # General pixel-based RL agent (any game)
├── qbert_gym.py      # Q*bert Gymnasium wrapper (structured state)
├── train_ppo.py      # PPO training script for Q*bert
├── eval_ppo.py       # Evaluation script
└── qbert_bot.py      # Handcrafted Q*bert bot
```

## How MameEnv Works

MameEnv communicates with MAME through named pipes (FIFOs) and a Lua frame callback:

```
Python                          MAME (Lua)
  │                                │
  │  write action ──→ action pipe ──→ read buttons, press them
  │                                │  run 1 frame
  │  read RAM     ←── data pipe  ←── write RAM snapshot
  │                                │  (optional: write pixel data)
  │  repeat                        │  repeat
```

```python
from mle import MameEnv

env = MameEnv("/path/to/roms", "qbert", {"lives": 0x0D00, "score": 0x00BE})
data = env.step()                              # advance 1 frame
data = env.step(":IN4", "P1 Up (Up-Right)")    # press button for 1 frame
data = env.step_n(":IN4", "P1 Up", 18)         # hold for 18 frames
env.request_frame()                            # next step() includes pixels
data = env.step()                              # data["frame"] = RGB array
env.close()
```

Runs at ~2200 FPS headless (unthrottled, no rendering) on Apple M3.

## Requirements

- MAME (installed via Homebrew: `brew install mame`)
- Python 3.10+
- PyTorch, gymnasium, stable-baselines3 (for RL training)
- Game ROMs in `~/mame/roms/`
