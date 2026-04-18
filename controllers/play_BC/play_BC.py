import torch
import torch.nn as nn
import numpy as np
from striker_env import StrikerRLEnv

# ──────────────────────────────────────────────────────────────────────────────
# 1. Universal Network Architecture
# ──────────────────────────────────────────────────────────────────────────────
def _mlp(in_dim: int, out_dim: int, hidden: list[int]) -> nn.Sequential:
    sizes  = [in_dim] + hidden + [out_dim]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class DiscreteNetwork(nn.Module):
    """Can act as either a Q-Network (RL) or a Classification Network (BC)"""
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int]):
        super().__init__()
        self.net = _mlp(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Main Play Loop
# ──────────────────────────────────────────────────────────────────────────────
def play():
    # Toggle this variable to test different brains!
    checkpoint_file = "bc_striker_brain.pt"  # or "iq_striker_brain_best.pt"
    
    print(f"--- LOADING BRAIN ({checkpoint_file}) ---")
    
    ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    ACTION_SET = ckpt["action_set"]
    env = StrikerRLEnv()
    
    # Instantiate the brain
    net = DiscreteNetwork(ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden"])
    
    # --- THE SMART LOADER ---
    if "q1" in ckpt:
        print("--> Detected Double-Q Architecture (IQ-Learn)")
        # During play, we don't need both Q-networks. Just load Q1 to make decisions.
        net.load_state_dict(ckpt["q1"])
    elif "q_net" in ckpt:
        print("--> Detected Single-Q Architecture (Behavioral Cloning / Old IQ)")
        net.load_state_dict(ckpt["q_net"])
    else:
        raise ValueError("Unknown checkpoint format! Cannot find 'q1' or 'q_net'.")
        
    net.eval() # Lock network into evaluation mode
    print("--- BRAIN READY. STARTING MATCH ---")
    
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_count  = 0

    while True:
        with torch.no_grad():
            obs_t  = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            # This outputs Q-values for RL, or Classification Logits for BC. 
            # In both cases, argmax() gives us the correct button!
            outputs = net(obs_t)
            best_action_idx = outputs.argmax(dim=-1).item()
            action = ACTION_SET[best_action_idx].copy()

        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            episode_count += 1
            print(f"--> Episode {episode_count} finished. Total score: {episode_reward:.2f}")
            obs, _ = env.reset()
            episode_reward = 0.0

if __name__ == "__main__":
    play()