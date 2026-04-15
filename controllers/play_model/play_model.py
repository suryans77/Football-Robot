"""
Play controller for discrete IQ-Learn + CQL striker brain.
Loads both Q-networks and takes min — consistent with how the
policy was computed during training.
"""
import torch
import torch.nn as nn
import numpy as np
from striker_env import StrikerRLEnv


def _mlp(in_dim: int, out_dim: int, hidden: list[int]) -> nn.Sequential:
    sizes  = [in_dim] + hidden + [out_dim]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DiscreteQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int]):
        super().__init__()
        self.net = _mlp(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def play():
    # Use the _best checkpoint, not the last one
    ckpt_path = "iq_striker_brain_best.pt"
    print(f"--- LOADING DISCRETE IQ-LEARN BRAIN ({ckpt_path}) ---")

    ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ACTION_SET = ckpt["action_set"]
    obs_dim    = ckpt["obs_dim"]
    n_actions  = ckpt["n_actions"]
    hidden     = ckpt["hidden"]

    # Rebuild both Q-networks
    q1 = DiscreteQNetwork(obs_dim, n_actions, hidden)
    q2 = DiscreteQNetwork(obs_dim, n_actions, hidden)
    q1.load_state_dict(ckpt["q1"])
    q2.load_state_dict(ckpt["q2"])
    q1.eval(); q2.eval()

    env = StrikerRLEnv()
    print(f"--- BRAIN LOADED. obs_dim={obs_dim}  n_actions={n_actions} ---")

    obs, _         = env.reset()
    episode_reward = 0.0
    episode_count  = 0

    while True:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

            # Take min of both Q-networks — same as during training
            q_vals         = torch.min(q1(obs_t), q2(obs_t))
            best_action_idx = q_vals.argmax(dim=-1).item()
            action          = ACTION_SET[best_action_idx].copy()

        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            episode_count += 1
            print(f"--> Episode {episode_count} finished. "
                  f"Total reward: {episode_reward:.3f}  "
                  f"last_action: {ACTION_SET[best_action_idx]}")
            obs, _         = env.reset()
            episode_reward = 0.0


if __name__ == "__main__":
    play()
