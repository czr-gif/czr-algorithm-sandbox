"""Baseline PPO with MLP policy (no MPC)."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.critic import Critic
from utils.buffer import RolloutBuffer


class PPOMPMLP:
    """Standard PPO with Gaussian MLP actor."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: dict):
        self.gamma    = cfg.get("gamma", 0.99)
        self.lam      = cfg.get("lam", 0.95)
        self.clip_eps = cfg.get("clip_eps", 0.2)
        self.epochs   = cfg.get("epochs", 10)
        self.lr       = cfg.get("lr", 3e-4)
        self.device   = torch.device(cfg.get("device", "cpu"))

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),     nn.Tanh(),
            nn.Linear(256, action_dim),
        ).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=self.device))

        self.critic = Critic(obs_dim).to(self.device)
        self.opt_actor  = optim.Adam(list(self.actor.parameters()) + [self.log_std], lr=self.lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.lr)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        mean  = self.actor(obs_t)
        dist  = torch.distributions.Normal(mean, self.log_std.exp())
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum()
        value    = self.critic(obs_t)
        return action.cpu().numpy(), log_prob.item(), value.item()

    def update(self, buffer: RolloutBuffer):
        data = buffer.get(self.device)
        obs, actions, old_log_probs, returns, advantages = (
            data["obs"], data["actions"], data["log_probs"],
            data["returns"], data["advantages"],
        )
        for _ in range(self.epochs):
            dist     = torch.distributions.Normal(self.actor(obs), self.log_std.exp())
            log_probs = dist.log_prob(actions).sum(-1)
            ratio = (log_probs - old_log_probs).exp()
            surr  = torch.min(
                ratio * advantages,
                ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages,
            )
            actor_loss = -surr.mean()
            self.opt_actor.zero_grad(); actor_loss.backward(); self.opt_actor.step()

            critic_loss = (self.critic(obs) - returns).pow(2).mean()
            self.opt_critic.zero_grad(); critic_loss.backward(); self.opt_critic.step()
