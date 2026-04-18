import pickle
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
# 1. Shared Architecture (Exact same as IQ-Learn)
# ──────────────────────────────────────────────────────────────────────────────
ACTION_SET = np.array([
    [-1., -1.], [-1.,  0.], [-1., +1.],
    [ 0., -1.], [ 0.,  0.], [ 0., +1.],
    [+1., -1.], [+1.,  0.], [+1., +1.],
], dtype=np.float32)
N_ACTIONS = len(ACTION_SET)

def action_to_index(action: np.ndarray) -> int:
    dists = np.linalg.norm(ACTION_SET - np.array(action, dtype=np.float32), axis=1)
    return int(np.argmin(dists))

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
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int]):
        super().__init__()
        self.net = _mlp(obs_dim, n_actions, hidden)
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Simplified Data Loader
# ──────────────────────────────────────────────────────────────────────────────
def load_expert_bc(path: str, device: torch.device):
    with open(path, "rb") as f:
        raw = pickle.load(f)
    
    obs = np.array([d["obs"] for d in raw], dtype=np.float32)
    act_idxs = np.array([action_to_index(d["action"]) for d in raw], dtype=np.int64)
    
    print(f"[BC] Loaded {len(obs)} transitions for Behavioral Cloning.")
    return torch.as_tensor(obs, device=device), torch.as_tensor(act_idxs, device=device)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Supervised Training Loop
# ──────────────────────────────────────────────────────────────────────────────
def train_bc():
    print("=" * 60)
    print("Behavioral Cloning (Supervised)  –  Soccer Striker")
    print("=" * 60)
    
    device = torch.device("cpu")
    obs_dim = len(load_expert_bc("expert_data_mirrored.pkl", device)[0][0])
    hidden = [128, 128]
    batch_size = 256
    
    # Initialize Network and Supervised Optimizer
    net = DiscreteNetwork(obs_dim, N_ACTIONS, hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss() # The core of BC for discrete actions
    
    e_obs, e_acts = load_expert_bc("expert_data_mirrored.pkl", device)
    dataset_size = len(e_obs)
    
    # Train for 500 Epochs (Full passes over the data)
    epochs = 500
    t_start = time.time()
    
    for epoch in range(1, epochs + 1):
        # Shuffle data every epoch
        indices = torch.randperm(dataset_size)
        e_obs = e_obs[indices]
        e_acts = e_acts[indices]
        
        epoch_loss = 0.0
        correct_predictions = 0
        
        for i in range(0, dataset_size, batch_size):
            batch_obs = e_obs[i : i + batch_size]
            batch_acts = e_acts[i : i + batch_size]
            
            # 1. Predict
            logits = net(batch_obs)
            
            # 2. Calculate Cross-Entropy Loss
            loss = loss_fn(logits, batch_acts)
            
            # 3. Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Track Accuracy
            preds = logits.argmax(dim=-1)
            correct_predictions += (preds == batch_acts).sum().item()
            
        if epoch % 50 == 0:
            avg_loss = epoch_loss / (dataset_size / batch_size)
            accuracy = (correct_predictions / dataset_size) * 100
            print(f"[Epoch {epoch:>3}/{epochs}]  Loss: {avg_loss:.4f}  |  Accuracy: {accuracy:.1f}%")

    # Save exactly like IQ-Learn so play_model.py can load it
    save_path = "bc_striker_brain.pt"
    torch.save({
        "q_net":      net.state_dict(),  # Reusing 'q_net' key for compatibility
        "obs_dim":    obs_dim,
        "n_actions":  N_ACTIONS,
        "action_set": ACTION_SET,
        "hidden":     hidden,
        "cfg":        {},
    }, save_path)
    print(f"\n--- BC MODEL SAVED → {save_path} ---")

if __name__ == "__main__":
    train_bc()