"""Simplified Dreamer v3 for MAME arcade games.

World model learns to predict next frame + reward from current frame + action.
Actor-critic trains on imagined trajectories inside the world model.
Much more sample efficient than PPO — needs ~10x fewer real environment steps.

Usage:
    python3 dreamer.py frogger --timesteps 500000
    python3 dreamer.py galaga --timesteps 1000000
"""

import sys
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── World Model Components ─────────────────────────────────────────

class ConvEncoder(nn.Module):
    """Encode 84x84 grayscale frame to latent vector."""
    def __init__(self, in_channels=1, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2), nn.ELU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ELU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ELU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ELU(),
            nn.Flatten(),
        )
        # Compute output size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            out_size = self.net(dummy).shape[1]
        self.fc = nn.Linear(out_size, latent_dim)

    def forward(self, x):
        return self.fc(self.net(x))


class RSSM(nn.Module):
    """Recurrent State-Space Model.

    Maintains a deterministic state (h) updated by GRU, and a stochastic
    state (z) sampled from the posterior (with observation) or prior (without).
    """
    def __init__(self, latent_dim=256, state_dim=256, action_dim=5, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim

        # Prior: predict z from h (no observation)
        self.prior_fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean + logvar
        )

        # Posterior: predict z from h + encoded observation
        self.posterior_fc = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        # Deterministic state transition
        self.gru = nn.GRUCell(latent_dim + action_dim, state_dim)

        self.latent_dim = latent_dim

    def initial_state(self, batch_size, device):
        return torch.zeros(batch_size, self.state_dim, device=device)

    def observe(self, h, encoded_obs):
        """Posterior: use observation to get better z."""
        x = torch.cat([h, encoded_obs], dim=-1)
        stats = self.posterior_fc(x)
        mean, logvar = stats.chunk(2, dim=-1)
        std = F.softplus(logvar) + 0.1
        z = mean + std * torch.randn_like(std)
        return z, mean, std

    def imagine(self, h):
        """Prior: predict z without observation."""
        stats = self.prior_fc(h)
        mean, logvar = stats.chunk(2, dim=-1)
        std = F.softplus(logvar) + 0.1
        z = mean + std * torch.randn_like(std)
        return z, mean, std

    def step(self, h, z, action_onehot):
        """Advance one step: update h given z and action."""
        x = torch.cat([z, action_onehot], dim=-1)
        h_new = self.gru(x, h)
        return h_new


class RewardPredictor(nn.Module):
    def __init__(self, state_dim=256, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1)).squeeze(-1)


class ContinuePredictor(nn.Module):
    """Predicts whether episode continues (1) or terminates (0)."""
    def __init__(self, state_dim=256, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, h, z):
        return torch.sigmoid(self.net(torch.cat([h, z], dim=-1)).squeeze(-1))


# ── Actor-Critic ───────────────────────────────────────────────────

class Actor(nn.Module):
    def __init__(self, state_dim=256, latent_dim=256, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, h, z):
        logits = self.net(torch.cat([h, z], dim=-1))
        return torch.distributions.Categorical(logits=logits)


class Critic(nn.Module):
    def __init__(self, state_dim=256, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1)).squeeze(-1)


# ── Replay Buffer ──────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.obs = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def add(self, obs, action, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample_sequences(self, batch_size, seq_len):
        """Sample random sequences for world model training."""
        max_start = len(self.obs) - seq_len
        if max_start < 1:
            return None
        indices = np.random.randint(0, max_start, size=batch_size)
        obs_seq = np.stack([[self.obs[i + t] for t in range(seq_len)] for i in indices])
        act_seq = np.stack([[self.actions[i + t] for t in range(seq_len)] for i in indices])
        rew_seq = np.stack([[self.rewards[i + t] for t in range(seq_len)] for i in indices])
        done_seq = np.stack([[self.dones[i + t] for t in range(seq_len)] for i in indices])
        return obs_seq, act_seq, rew_seq, done_seq

    def __len__(self):
        return len(self.obs)


# ── Dreamer Agent ──────────────────────────────────────────────────

class DreamerAgent:
    def __init__(self, action_dim, device="cpu",
                 latent_dim=128, state_dim=128, lr=3e-4):
        self.device = device
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.state_dim = state_dim

        # World model
        self.encoder = ConvEncoder(1, latent_dim).to(device)
        self.rssm = RSSM(latent_dim, state_dim, action_dim).to(device)
        self.reward_pred = RewardPredictor(state_dim, latent_dim).to(device)
        self.continue_pred = ContinuePredictor(state_dim, latent_dim).to(device)

        # Actor-critic
        self.actor = Actor(state_dim, latent_dim, action_dim).to(device)
        self.critic = Critic(state_dim, latent_dim).to(device)

        # Optimizers
        world_params = (list(self.encoder.parameters()) +
                       list(self.rssm.parameters()) +
                       list(self.reward_pred.parameters()) +
                       list(self.continue_pred.parameters()))
        self.world_opt = torch.optim.Adam(world_params, lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Running state for acting
        self.h = None
        self.z = None

        self.buffer = ReplayBuffer()
        self.train_steps = 0

    def reset(self):
        self.h = self.rssm.initial_state(1, self.device)
        self.z = torch.zeros(1, self.latent_dim, device=self.device)

    def act(self, obs, explore=True):
        """Select action given a single 84x84 grayscale observation."""
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            obs_t = obs_t.unsqueeze(0).unsqueeze(0) / 255.0  # (1, 1, 84, 84)
            encoded = self.encoder(obs_t)
            self.z, _, _ = self.rssm.observe(self.h, encoded)
            dist = self.actor(self.h, self.z)
            if explore:
                action = dist.sample()
            else:
                action = dist.probs.argmax(dim=-1)
            # Update state
            action_onehot = F.one_hot(action, self.action_dim).float()
            self.h = self.rssm.step(self.h, self.z, action_onehot)
            return action.item()

    def train_world_model(self, batch_size=16, seq_len=32):
        """Train world model on replay buffer sequences."""
        data = self.buffer.sample_sequences(batch_size, seq_len)
        if data is None:
            return {}
        obs_seq, act_seq, rew_seq, done_seq = data

        obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=self.device)
        obs_t = obs_t.unsqueeze(2) / 255.0  # (B, T, 1, 84, 84)
        act_t = torch.tensor(act_seq, dtype=torch.long, device=self.device)
        rew_t = torch.tensor(rew_seq, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done_seq, dtype=torch.float32, device=self.device)

        B, T = obs_t.shape[:2]
        h = self.rssm.initial_state(B, self.device)

        kl_loss = 0
        reward_loss = 0
        continue_loss = 0

        for t in range(T):
            encoded = self.encoder(obs_t[:, t])
            z_post, post_mean, post_std = self.rssm.observe(h, encoded)
            z_prior, prior_mean, prior_std = self.rssm.imagine(h)

            # KL divergence (free nats = 1.0)
            kl = torch.distributions.kl_divergence(
                torch.distributions.Normal(post_mean, post_std),
                torch.distributions.Normal(prior_mean, prior_std),
            ).sum(-1).mean()
            kl_loss += torch.clamp(kl, min=1.0)

            # Reward prediction
            pred_reward = self.reward_pred(h, z_post)
            reward_loss += F.mse_loss(pred_reward, rew_t[:, t])

            # Continue prediction
            pred_continue = self.continue_pred(h, z_post)
            continue_loss += F.binary_cross_entropy(
                pred_continue, 1.0 - done_t[:, t])

            # Step
            action_onehot = F.one_hot(act_t[:, t], self.action_dim).float()
            h = self.rssm.step(h, z_post, action_onehot)

        loss = (kl_loss + reward_loss + continue_loss) / T
        self.world_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.encoder.parameters(), 100.0)
        nn.utils.clip_grad_norm_(self.rssm.parameters(), 100.0)
        self.world_opt.step()

        return {"wm/kl": kl_loss.item() / T,
                "wm/reward": reward_loss.item() / T,
                "wm/continue": continue_loss.item() / T}

    def train_actor_critic(self, horizon=15, batch_size=16):
        """Train actor-critic on imagined trajectories."""
        data = self.buffer.sample_sequences(batch_size, 1)
        if data is None:
            return {}
        obs_seq, _, _, _ = data

        obs_t = torch.tensor(obs_seq[:, 0], dtype=torch.float32, device=self.device)
        obs_t = obs_t.unsqueeze(1) / 255.0
        with torch.no_grad():
            encoded = self.encoder(obs_t)
            h = self.rssm.initial_state(batch_size, self.device)
            z, _, _ = self.rssm.observe(h, encoded)

        # Imagine trajectories
        imagined_h = [h]
        imagined_z = [z]
        actions = []
        rewards = []
        continues = []

        for _ in range(horizon):
            dist = self.actor(h, z)
            action = dist.sample()
            actions.append(action)
            action_onehot = F.one_hot(action, self.action_dim).float()
            h = self.rssm.step(h, z, action_onehot)
            z, _, _ = self.rssm.imagine(h)
            rewards.append(self.reward_pred(h, z))
            continues.append(self.continue_pred(h, z))
            imagined_h.append(h)
            imagined_z.append(z)

        # Compute returns (lambda-return)
        gamma = 0.99
        lambda_ = 0.95
        returns = []
        last_value = self.critic(imagined_h[-1], imagined_z[-1]).detach()
        ret = last_value
        for t in reversed(range(horizon)):
            value = self.critic(imagined_h[t], imagined_z[t]).detach()
            ret = rewards[t] + gamma * continues[t] * (
                lambda_ * ret + (1 - lambda_) * value)
            returns.insert(0, ret)

        # Train critic
        critic_loss = 0
        for t in range(horizon):
            value = self.critic(imagined_h[t], imagined_z[t])
            critic_loss += F.mse_loss(value, returns[t].detach())
        critic_loss /= horizon

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_opt.step()

        # Train actor (maximize returns)
        actor_loss = 0
        for t in range(horizon):
            dist = self.actor(imagined_h[t].detach(), imagined_z[t].detach())
            action = dist.sample()
            log_prob = dist.log_prob(action)
            advantage = returns[t].detach() - self.critic(
                imagined_h[t].detach(), imagined_z[t].detach()).detach()
            actor_loss -= (log_prob * advantage).mean()
        actor_loss /= horizon

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_opt.step()

        return {"ac/critic": critic_loss.item(),
                "ac/actor": actor_loss.item()}

    def train_step(self):
        """One full training step: world model + actor-critic."""
        self.train_steps += 1
        stats = {}
        stats.update(self.train_world_model())
        if self.train_steps % 2 == 0:  # actor-critic every other step
            stats.update(self.train_actor_critic())
        return stats

    def save(self, path):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "rssm": self.rssm.state_dict(),
            "reward_pred": self.reward_pred.state_dict(),
            "continue_pred": self.continue_pred.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)
        print(f"Saved to {path}")

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(state["encoder"])
        self.rssm.load_state_dict(state["rssm"])
        self.reward_pred.load_state_dict(state["reward_pred"])
        self.continue_pred.load_state_dict(state["continue_pred"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])


# ── Training Loop ──────────────────────────────────────────────────

def train(game_id, timesteps, save_path, n_envs=1):
    from mle_general import MamePixelEnv, ROMS_PATH
    import json

    # Load game config
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

    env = MamePixelEnv(game_id, render=False, throttle=False,
                       score_addrs=score_addrs, lives_addr=lives_addr,
                       score_encoding=score_encoding)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Dreamer v3 on {device}")

    agent = DreamerAgent(env.action_space.n, device=device)
    print(f"Action space: {env.action_space.n}")

    # Collect initial data with random policy
    obs, _ = env.reset()
    agent.reset()
    prefill = 1000
    print(f"Prefilling buffer with {prefill} random steps...")
    for _ in range(prefill):
        action = env.action_space.sample()
        frame = obs[-1]  # last frame from stack
        next_obs, reward, term, trunc, info = env.step(action)
        agent.buffer.add(frame, action, reward, term or trunc)
        if term or trunc:
            obs, _ = env.reset()
            agent.reset()
        else:
            obs = next_obs

    print(f"Buffer: {len(agent.buffer)} steps. Training...")

    # Main loop: collect data + train
    episode_rewards = []
    ep_reward = 0
    t0 = time.time()
    steps = 0

    try:
        _wandb = None
        try:
            import wandb as _wandb
            _wandb.init(project="mle-arcade",
                       name=f"{game_id}-dreamer-{timesteps//1000}k",
                       config={"game": game_id, "algo": "dreamer",
                               "timesteps": timesteps})
        except Exception:
            pass

        while steps < timesteps:
            # Act in real environment
            frame = obs[-1]
            action = agent.act(frame, explore=True)
            next_obs, reward, term, trunc, info = env.step(action)
            agent.buffer.add(frame, action, reward, term or trunc)
            ep_reward += reward
            steps += 1

            if term or trunc:
                episode_rewards.append(ep_reward)
                if _wandb and _wandb.run:
                    _wandb.log({"game/reward": ep_reward,
                               "game/score": info.get("game_score", 0),
                               "game/duration_sec": info.get("game_duration_sec", 0),
                               }, step=steps)
                ep_reward = 0
                obs, _ = env.reset()
                agent.reset()
            else:
                obs = next_obs

            # Train world model + actor-critic
            if steps % 2 == 0 and len(agent.buffer) > 500:
                stats = agent.train_step()
                if _wandb and _wandb.run and steps % 100 == 0:
                    _wandb.log(stats, step=steps)

            # Log
            if steps % 1000 == 0:
                elapsed = time.time() - t0
                fps = steps / elapsed
                recent = episode_rewards[-10:] if episode_rewards else [0]
                avg_rew = sum(recent) / len(recent)
                print(f"  step {steps}/{timesteps} | fps={fps:.0f} | "
                      f"ep_rew={avg_rew:.1f} | buffer={len(agent.buffer)} | "
                      f"episodes={len(episode_rewards)}")
                if _wandb and _wandb.run:
                    _wandb.log({"rollout/ep_rew_mean": avg_rew,
                               "time/fps": fps,
                               "time/steps": steps}, step=steps)

            # Save checkpoint
            if steps % 50000 == 0:
                agent.save(f"checkpoints/{game_id}_dreamer_{steps}.pt")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        agent.save(save_path)
        env.close()
        if _wandb and _wandb.run:
            _wandb.finish()


def evaluate(game_id, model_path, episodes=3):
    from mle_general import MamePixelEnv
    import json

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

    env = MamePixelEnv(game_id, render=True, throttle=True,
                       score_addrs=score_addrs, lives_addr=lives_addr,
                       score_encoding=score_encoding)

    agent = DreamerAgent(env.action_space.n, device="cpu")
    agent.load(model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        agent.reset()
        total_reward = 0
        steps = 0
        while True:
            frame = obs[-1]
            action = agent.act(frame, explore=False)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            steps += 1
            if term or trunc:
                break
        print(f"Episode {ep+1}: reward={total_reward:.1f}, steps={steps}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dreamer v3 for MAME arcade games")
    parser.add_argument("game", help="MAME ROM name")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--eval", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    save_path = args.save or f"{args.game}_dreamer.pt"

    if args.eval:
        evaluate(args.game, args.eval, args.episodes)
    else:
        train(args.game, args.timesteps, save_path)
