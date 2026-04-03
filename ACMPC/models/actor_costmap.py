"""Actor network: observation -> (Q matrix, p vector) for MPC cost."""
import torch
import torch.nn as nn


class ActorCostMap(nn.Module):
    """Outputs quadratic cost parameters (Q, p) used by the MPC layer."""

    def __init__(self, obs_dim: int, action_dim: int, horizon: int, hidden: int = 256):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        # Q: diagonal of action-cost matrix (horizon * action_dim)
        self.Q_head = nn.Linear(hidden, horizon * action_dim)
        # p: linear cost term (horizon * action_dim)
        self.p_head = nn.Linear(hidden, horizon * action_dim)

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        Q_diag = torch.exp(self.Q_head(h))  # ensure positive
        p = self.p_head(h)
        return Q_diag, p
