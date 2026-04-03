"""AC-MPC agent: PPO with differentiable MPC as the policy layer."""
import torch
import torch.optim as optim
import numpy as np

from models.actor_costmap import ActorCostMap
from models.critic import Critic
from models.dynamics import PendulumDynamics
from mpc.mpc_layer import MPCLayer
from utils.buffer import RolloutBuffer


class ACMPC:
    """
    Actor-Critic MPC.

    Actor outputs cost-map parameters (Q, p) which are fed into a
    differentiable MPC layer to produce actions.
    """

    def __init__(self, obs_dim: int, action_dim: int, cfg: dict):
        self.gamma = cfg.get("gamma", 0.99)
        self.lam = cfg.get("lam", 0.95)
        self.clip_eps = cfg.get("clip_eps", 0.2)
        self.epochs = cfg.get("epochs", 10)
        self.lr = cfg.get("lr", 3e-4)
        self.horizon = cfg.get("horizon", 10)
        self.device = torch.device(cfg.get("device", "cpu"))

        self.actor = ActorCostMap(obs_dim, action_dim, self.horizon).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)
        dynamics = PendulumDynamics().to(self.device)
        self.mpc = MPCLayer(dynamics, self.horizon, action_dim).to(self.device)

        self.opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr
        )

    def select_action(self, obs: np.ndarray):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        Q_diag, p = self.actor(obs_t)
        action = self.mpc(obs_t, Q_diag, p)  # (1, action_dim)
        # Gaussian exploration noise
        noise = torch.randn_like(action) * 0.1
        action = (action + noise).clamp(-2.0, 2.0)
        log_prob = torch.distributions.Normal(action, 0.1).log_prob(action).sum()
        return action.squeeze(0).cpu().detach().numpy(), log_prob.item()

    def update(self, buffer: RolloutBuffer):
        data = buffer.get(self.device)
        obs, actions, old_log_probs, returns, advantages = (
            data["obs"], data["actions"], data["log_probs"],
            data["returns"], data["advantages"],
        )
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
            loss = -surr.mean() + 0.5 * (values - returns).pow(2).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
