"""Wrapped Pendulum environment for AC-MPC."""
import gymnasium as gym
import numpy as np


class PendulumEnv:
    """Thin wrapper around gym Pendulum-v1."""

    def __init__(self, seed: int = 0):
        self.env = gym.make("Pendulum-v1")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._seed = seed

    def reset(self):
        obs, info = self.env.reset(seed=self._seed)
        return obs

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
