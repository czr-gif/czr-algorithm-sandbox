"""AC-MPC with Model-Predictive Value Expansion (MPVE)."""
import torch
import torch.optim as optim
import numpy as np

from agents.ac_mpc import ACMPC
from models.dynamics import PendulumDynamics


class ACMPCWithMPVE(ACMPC):
    """
    Extends AC-MPC with MPVE: uses the learned dynamics to generate
    synthetic rollouts that augment the value-function update.
    """

    def __init__(self, obs_dim: int, action_dim: int, cfg: dict):
        super().__init__(obs_dim, action_dim, cfg)
        self.mpve_rollout_len = cfg.get("mpve_rollout_len", 3)

    def _mpve_targets(self, obs: torch.Tensor) -> torch.Tensor:
        """Expand value targets via short model rollouts."""
        x = obs.detach()
        value_targets = self.critic(x)
        discount = 1.0
        for _ in range(self.mpve_rollout_len):
            Q_diag, p = self.actor(x)
            u = self.mpc(x, Q_diag, p)
            x = self.mpc.solver.dynamics(x, u)
            discount *= self.gamma
            value_targets = value_targets + discount * self.critic(x)
        return value_targets

    def update(self, buffer):
        data = buffer.get(self.device)
        obs, actions, old_log_probs, returns, advantages = (
            data["obs"], data["actions"], data["log_probs"],
            data["returns"], data["advantages"],
        )
        mpve_returns = self._mpve_targets(obs).detach()
        blended = 0.5 * returns + 0.5 * mpve_returns

        for _ in range(self.epochs):
            Q_diag, p = self.actor(obs)
            mean_action = self.mpc(obs, Q_diag, p)
            dist = torch.distributions.Normal(mean_action, 0.1)
            log_probs = dist.log_prob(actions).sum(-1)
            ratio = (log_probs - old_log_probs).exp()
            surr = torch.min(
                ratio * advantages,
                ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages,
            )
            values = self.critic(obs)
            loss = -surr.mean() + 0.5 * (values - blended).pow(2).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
