# MLE — MAME Learning Environment

Train RL agents on any arcade game via MAME. No game-specific code needed.

## Quick Start

```bash
pip3 install torch gymnasium stable-baselines3 pillow pytesseract
brew install mame tesseract
```

### Train on any arcade game

```bash
python3 mle_general.py qbert                       # pixel-based, survival reward
python3 mle_general.py galaga --bootstrap           # auto-detect score via OCR+RAM
python3 mle_general.py dkong --model dqn            # use DQN instead of PPO
python3 mle_general.py frogger --model a2c          # use A2C

# Watch a trained model play
python3 mle_general.py qbert --eval qbert_ppo.zip
```

Auto-detects game controls via MAME's Lua API. Uses 84×84 grayscale frame stacks with CNN policy. Just drop a ROM in `~/mame/roms/` and go.

### Reward system (layered)

The agent uses the best available reward signal:

1. **RAM score delta** — if score address is known (via `game_configs.json`, `--score-addr`, or bootstrap)
2. **OCR score** — reads score digits from pixels when no RAM address
3. **Survival** — +0.1 per step alive (universal fallback)

### Bootstrap: auto-detect score RAM

```bash
python3 mle_general.py qbert --bootstrap
```

Starts training with survival reward. At step 2000, spawns an OCR+RAM scanner that:
1. Plays randomly, screenshots the game
2. OCRs the score digits from the screen
3. Scans all RAM for bytes matching the OCR'd value
4. Intersects candidates across multiple samples
5. Switches to RAM-based reward mid-training

Tested: correctly finds Q*bert's score at `$00BE` fully automatically.

### Known game configs

`game_configs.json` has verified RAM addresses for popular games:

```bash
python3 mle_general.py qbert       # auto-loads score=$00BE, lives=$0D00
```

Add new games by editing `game_configs.json` or use manual addresses:

```bash
python3 mle_general.py dkong --score-addr 0x6007,0x6008,0x6009 --lives-addr 0x6001
```

### Find score addresses for any game

```bash
python3 find_score_ram.py qbert             # OCR + RAM scan (opens game window)
python3 find_score_ram.py galaga --headless  # fast, tiny window
python3 validate_addrs.py qbert             # visually verify known addresses
```

### Train Q*bert with structured RAM state (10x faster)

```bash
python3 train_ppo.py --timesteps 500000
python3 train_ppo.py --timesteps 2000000 --resume qbert_ppo_500k --save qbert_ppo_2M
python3 eval_ppo.py qbert_ppo_500k          # watch it play
```

Uses 78-dimensional state from RAM (position, cubes, enemies, discs) with MLP policy. Trains 10x faster than pixels.

### Run the handcrafted Q*bert bot

```bash
python3 qbert_bot.py
```

ROM-accurate frame simulation, depth-6 lookahead with unified scoring. Completes 3+ levels.

## Architecture

```
mle/
├── mle/                # MAME Learning Environment core
│   ├── env.py          # MameEnv — Python↔MAME bridge via named pipes + Lua
│   └── console.py      # MAME process management
├── qbert/              # Q*bert game modules (ROM-verified)
│   ├── state.py        # RAM reader, entity classification
│   ├── frame_sim.py    # Frame-perfect simulation
│   ├── predict.py      # Enemy path prediction
│   ├── collision.py    # ROM collision detection
│   ├── spawn.py        # Spawn timer
│   ├── sim.py          # Constants
│   └── planner.py      # Move mappings
├── mle_general.py      # General RL agent (any game, PPO/DQN/A2C)
├── qbert_gym.py        # Q*bert Gymnasium wrapper (structured state)
├── train_ppo.py        # Q*bert PPO training script
├── eval_ppo.py         # Evaluation script
├── qbert_bot.py        # Handcrafted Q*bert bot
├── find_score_ram.py   # OCR-based score RAM address finder
├── validate_addrs.py   # Visual RAM address validator
└── game_configs.json   # Known score/lives addresses per game
```

## How MameEnv Works

```
Python                          MAME (Lua)
  │                                │
  │  write action ──→ action pipe ──→ press buttons
  │                                │  run 1 frame
  │  read RAM     ←── data pipe  ←── write RAM snapshot
  │                                │  (optional: pixel data)
  │  repeat                        │  repeat
```

```python
from mle import MameEnv

env = MameEnv("/path/to/roms", "qbert", {"lives": 0x0D00, "score": 0x00BE})
data = env.step()                              # advance 1 frame
data = env.step(":IN4", "P1 Up (Up-Right)")    # press button
data = env.step_n(":IN4", "P1 Up", 18)         # hold for 18 frames
env.request_frame()                            # next step includes pixels
data = env.step()                              # data["frame"] = RGB array
env.close()
```

~2200 FPS headless on Apple M3.

## Requirements

- MAME (`brew install mame`)
- Python 3.10+
- PyTorch, gymnasium, stable-baselines3 (for RL)
- pytesseract + tesseract (for score OCR)
- Game ROMs in `~/mame/roms/`
