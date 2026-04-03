"""Rollout buffer with GAE advantage estimation."""
import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, action_dim: int, gamma: float, lam: float):
        self.size = size
        self.gamma = gamma
        self.lam = lam
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.obs       = np.zeros((self.size, self.obs_dim),    dtype=np.float32)
        self.actions   = np.zeros((self.size, self.action_dim), dtype=np.float32)
        self.rewards   = np.zeros(self.size,                    dtype=np.float32)
        self.values    = np.zeros(self.size,                    dtype=np.float32)
        self.log_probs = np.zeros(self.size,                    dtype=np.float32)
        self.dones     = np.zeros(self.size,                    dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, value, log_prob, done):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.values[self.ptr]    = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr]     = done
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: float):
        """GAE-Lambda."""
        advantages = np.zeros(self.size, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.size)):
            next_value = last_value if t == self.size - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - (self.dones[t] if t == self.size - 1 else self.dones[t])
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae
            advantages[t] = last_gae
        self.returns = advantages + self.values

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.advantages = advantages

    def get(self, device: torch.device) -> dict:
        def t(x):
            return torch.tensor(x, dtype=torch.float32, device=device)
        return {
            "obs":        t(self.obs),
            "actions":    t(self.actions),
            "log_probs":  t(self.log_probs),
            "returns":    t(self.returns),
            "advantages": t(self.advantages),
        }

    @property
    def full(self):
        return self.ptr >= self.size
