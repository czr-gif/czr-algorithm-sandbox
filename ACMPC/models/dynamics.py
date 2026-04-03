"""Known dynamics model for Pendulum (PyTorch, differentiable)."""
import torch
import torch.nn as nn


class PendulumDynamics(nn.Module):
    """
    Differentiable Pendulum dynamics for use inside the MPC rollout.

    State:  x = [cos(theta), sin(theta), theta_dot]  (as returned by gym)
    Action: u = [torque] in [-2, 2]
    """

    def __init__(self, dt: float = 0.05, max_torque: float = 2.0):
        super().__init__()
        self.dt = dt
        self.max_torque = max_torque
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:    (..., 3)  [cos_th, sin_th, thdot]
            action: (..., 1)  torque
        Returns:
            next_obs: (..., 3)
        """
        cos_th = obs[..., 0]
        sin_th = obs[..., 1]
        thdot = obs[..., 2]
        u = action[..., 0].clamp(-self.max_torque, self.max_torque)

        theta = torch.atan2(sin_th, cos_th)
        newthdot = (
            thdot
            + (-3 * self.g / (2 * self.l) * torch.sin(theta + torch.pi)
               + 3.0 / (self.m * self.l ** 2) * u)
            * self.dt
        )
        newtheta = theta + newthdot * self.dt
        newthdot = newthdot.clamp(-8.0, 8.0)

        return torch.stack([torch.cos(newtheta), torch.sin(newtheta), newthdot], dim=-1)
