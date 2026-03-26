#!/bin/bash
# Overnight training experiments
# Galaga PPO is already running separately

LOG="experiments.log"
echo "=== MLE Training Experiments ===" > $LOG
echo "Started: $(date)" >> $LOG
echo "" >> $LOG

# Kill any stale MAME
pkill -9 -f mame 2>/dev/null
sleep 2

# ── Experiment 1: Q*bert structured state PPO (2M steps, ~2 hrs) ──
echo "[$(date)] Exp 1: Q*bert structured PPO 2M" | tee -a $LOG
python3 train_ppo.py --timesteps 2000000 --save qbert_ppo_2M 2>&1 | \
    grep -E "ep_rew|ep_len|fps|total_timesteps|Model" | tail -5 >> $LOG
echo "" >> $LOG
pkill -9 -f mame 2>/dev/null; sleep 2

# ── Experiment 2: DK PPO 500K ──
echo "[$(date)] Exp 2: DK PPO 500K" | tee -a $LOG
python3 mle_general.py dkong --timesteps 500000 --save dkong_ppo_500k 2>&1 | \
    grep -E "ep_rew|ep_len|fps|total_timesteps|Saved|Loaded" | tail -5 >> $LOG
echo "" >> $LOG
pkill -9 -f mame 2>/dev/null; sleep 2

# ── Experiment 3: Frogger PPO 500K ──
echo "[$(date)] Exp 3: Frogger PPO 500K" | tee -a $LOG
python3 mle_general.py frogger --timesteps 500000 --save frogger_ppo_500k 2>&1 | \
    grep -E "ep_rew|ep_len|fps|total_timesteps|Saved|Loaded" | tail -5 >> $LOG
echo "" >> $LOG
pkill -9 -f mame 2>/dev/null; sleep 2

# ── Experiment 4: Dig Dug PPO 500K ──
echo "[$(date)] Exp 4: Dig Dug PPO 500K" | tee -a $LOG
python3 mle_general.py digdug --timesteps 500000 --save digdug_ppo_500k 2>&1 | \
    grep -E "ep_rew|ep_len|fps|total_timesteps|Saved|Loaded" | tail -5 >> $LOG
echo "" >> $LOG
pkill -9 -f mame 2>/dev/null; sleep 2

# ── Experiment 5: Galaga DQN 500K (compare vs PPO) ──
echo "[$(date)] Exp 5: Galaga DQN 500K" | tee -a $LOG
python3 mle_general.py galaga --model dqn --timesteps 500000 --save galaga_dqn_500k 2>&1 | \
    grep -E "ep_rew|ep_len|fps|total_timesteps|Saved|Loaded" | tail -5 >> $LOG
echo "" >> $LOG
pkill -9 -f mame 2>/dev/null; sleep 2

# ── Experiment 6: DK DQN 500K (compare vs PPO) ──
echo "[$(date)] Exp 6: DK DQN 500K" | tee -a $LOG
python3 mle_general.py dkong --model dqn --timesteps 500000 --save dkong_dqn_500k 2>&1 | \
    grep -E "ep_rew|ep_len|fps|total_timesteps|Saved|Loaded" | tail -5 >> $LOG
echo "" >> $LOG
pkill -9 -f mame 2>/dev/null; sleep 2

# ── Experiment 7: Galaga PPO extended to 2M ──
echo "[$(date)] Exp 7: Galaga PPO 2M (resume from 500K)" | tee -a $LOG
python3 mle_general.py galaga --timesteps 2000000 --save galaga_ppo_2M 2>&1 | \
    grep -E "ep_rew|ep_len|fps|total_timesteps|Saved|Loaded" | tail -5 >> $LOG
echo "" >> $LOG
pkill -9 -f mame 2>/dev/null; sleep 2

# ── Experiment 8: Joust PPO 500K (has lives addr, survival reward) ──
echo "[$(date)] Exp 8: Joust PPO 500K" | tee -a $LOG
python3 mle_general.py joust --timesteps 500000 --save joust_ppo_500k 2>&1 | \
    grep -E "ep_rew|ep_len|fps|total_timesteps|Saved|Loaded" | tail -5 >> $LOG
echo "" >> $LOG

echo "[$(date)] All experiments complete" | tee -a $LOG
echo "" >> $LOG

# Summary
echo "=== Final Results ===" >> $LOG
for model in qbert_ppo_2M galaga_ppo_500k galaga_ppo_2M galaga_dqn_500k \
             dkong_ppo_500k dkong_dqn_500k frogger_ppo_500k digdug_ppo_500k \
             joust_ppo_500k; do
    if [ -f "${model}.zip" ]; then
        size=$(du -h "${model}.zip" | cut -f1)
        echo "  ✓ ${model}.zip ($size)" >> $LOG
    else
        echo "  ✗ ${model} (not found)" >> $LOG
    fi
done
cat $LOG
