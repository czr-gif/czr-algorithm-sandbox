"""Train PPO MLP baseline on Pendulum-v1."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np

from envs.pendulum_env import PendulumEnv
from agents.ppo_mlp import PPOMPMLP
from utils.buffer import RolloutBuffer
from utils.logger import Logger


def train(cfg_path: str = "configs/pendulum_baseline.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    env    = PendulumEnv(seed=cfg.get("seed", 0))
    obs_dim    = env.observation_space.shape[0]   # 3
    action_dim = env.action_space.shape[0]         # 1

    agent  = PPOMPMLP(obs_dim, action_dim, cfg)
    buffer = RolloutBuffer(
        size=cfg["rollout_len"], obs_dim=obs_dim, action_dim=action_dim,
        gamma=cfg["gamma"], lam=cfg["lam"],
    )
    logger = Logger(log_dir=cfg.get("log_dir", "runs/baseline"))

    obs  = env.reset()
    ep_ret, ep_len = 0.0, 0
    total_steps = 0

    for update in range(cfg["n_updates"]):
        buffer.reset()

        # --- Collect rollout ---
        while not buffer.full:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            buffer.add(obs, action, reward, value, log_prob, float(done))
            obs = next_obs
            ep_ret += reward; ep_len += 1; total_steps += 1

            if done:
                logger.log("train/ep_return", ep_ret, total_steps)
                logger.log("train/ep_len",    ep_len, total_steps)
                ep_ret, ep_len = 0.0, 0
                obs = env.reset()

        # Bootstrap last value
        _, _, last_value = agent.select_action(obs)
        buffer.compute_returns_and_advantages(last_value)

        # --- Update ---
        agent.update(buffer)
        logger.step()

    env.close()
    logger.close()
    print("Training done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/pendulum_baseline.yaml")
    args = p.parse_args()
    train(args.cfg)
