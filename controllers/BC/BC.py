"""
Behavioural Cloning (BC) for the soccer striker robot
======================================================

How BC works
------------
BC is pure supervised learning. It treats imitation as a classification
problem:

    Given observation s, predict which of the 9 actions the expert took.

Loss: cross-entropy between predicted action logits and expert action index.
No environment interaction during training. No Q-functions. No rewards.

Why no imitation library
------------------------
The imitation library's BC implementation wraps SB3 policies and is built
around continuous action spaces. For our discrete 9-action setup, plain
PyTorch cross-entropy is cleaner, has no extra dependencies, and is
easier to audit. BC is literally a classifier — there is no algorithmic
complexity to hide behind a library.

Training procedure
------------------
  1. Load expert_data.pkl
  2. Split into train (90%) / val (10%) sets
  3. Train for N epochs, shuffling mini-batches each epoch
  4. Track validation accuracy and save best checkpoint
  5. Also run env evaluation every eval_interval epochs

Metrics tracked during training
---------------------------------
  train_loss    — cross-entropy on training set
  val_loss      — cross-entropy on held-out validation set
  val_acc       — % of val transitions where argmax matches expert action
                  This is the clearest BC training health signal.
                  If val_acc plateaus below ~60%, you need more data.
  eval_score    — mean episode reward from 10 live env episodes

BC limitations vs other methods
---------------------------------
  - Covariate shift: at test time the robot visits states never seen in
    expert data, and the policy has no way to recover. This is the
    fundamental weakness of BC.
  - No exploration: BC never improves beyond what the expert showed.
  - For comparison: IQ-Learn and SQIL both use online env interaction to
    partially correct for covariate shift. BC does not.

Action index mapping
---------------------
  0: [-1, -1]   1: [-1,  0]   2: [-1, +1]
  3: [ 0, -1]   4: [ 0,  0]   5: [ 0, +1]
  6: [+1, -1]   7: [+1,  0]   8: [+1, +1]

Dependencies
------------
    pip install torch numpy gymnasium
    (no imitation library needed)
"""

from __future__ import annotations

import pickle
import time
import warnings
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.set_num_threads(torch.get_num_threads())

# ──────────────────────────────────────────────────────────────────────────────
# Discrete action set
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


def index_to_action(idx: int) -> np.ndarray:
    return ACTION_SET[idx].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

CFG = dict(
    expert_path       = "type_2_data.pkl",
    save_path         = "bc_striker_brain.pt",
    save_path_best    = "type_2_bc.pt",

    # Training
    n_epochs          = 200,
    batch_size        = 64,
    lr                = 3e-4,
    weight_decay      = 1e-4,       # L2 reg — helps prevent overfitting on small datasets
    val_split         = 0.1,        # fraction of expert data held out for validation
    hidden            = [128, 128],

    # LR schedule — cosine annealing over all epochs
    lr_min            = 1e-5,

    # Eval
    eval_interval     = 20,         # run env eval every N epochs
    num_eval_episodes = 10,
    log_window        = 10,         # rolling average window for loss logging
)

# ──────────────────────────────────────────────────────────────────────────────
# Policy network  (classifier: obs → action logits)
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


class BCPolicy(nn.Module):
    """
    Maps obs → logits over N_ACTIONS.
    At inference: argmax(logits) = greedy action.
    During training: cross-entropy(logits, expert_action_idx).
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int]):
        super().__init__()
        self.net = _mlp(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns logits (NOT probabilities) — shape (batch, n_actions)."""
        return self.net(obs)

    def predict(self, obs: torch.Tensor) -> int:
        """Greedy action index for a single observation."""
        with torch.no_grad():
            logits = self.net(obs.unsqueeze(0))
            return logits.argmax(dim=-1).item()


# ──────────────────────────────────────────────────────────────────────────────
# Expert data loader
# ──────────────────────────────────────────────────────────────────────────────

def load_expert(path: str):
    pkl = Path(path)
    if not pkl.exists():
        raise FileNotFoundError(f"Expert data not found: '{path}'")

    with open(pkl, "rb") as f:
        raw = pickle.load(f)

    print(f"[Expert] Pickle type : {type(raw)}")
    print(f"[Expert] Length      : {len(raw) if hasattr(raw, '__len__') else 'N/A'}")
    if hasattr(raw, '__len__') and len(raw) > 0:
        print(f"[Expert] First item  : {raw[0]}")

    if not hasattr(raw, '__len__') or len(raw) == 0:
        raise ValueError("expert_data.pkl is empty. Re-run record_expert.py.")

    obs      = np.array([d["obs"]    for d in raw], dtype=np.float32)
    act_idxs = np.array([action_to_index(d["action"]) for d in raw], dtype=np.int64)

    # Log class balance — important for BC since imbalanced classes
    # (e.g. 80% "go forward") cause the policy to collapse to that action
    unique, counts = np.unique(act_idxs, return_counts=True)
    print("[Expert] Action class distribution (BC training targets):")
    for u, c in zip(unique, counts):
        pct = 100.0 * c / len(raw)
        bar = "█" * int(pct / 2)
        print(f"         idx {u} {ACTION_SET[u]} : {c:>5} ({pct:>5.1f}%)  {bar}")

    # Warn if any action is heavily dominant — BC will be biased toward it
    max_pct = 100.0 * counts.max() / len(raw)
    if max_pct > 50.0:
        warnings.warn(
            f"[Expert] Action idx {unique[counts.argmax()]} dominates at "
            f"{max_pct:.1f}% of transitions. BC may collapse to this action. "
            f"Consider collecting more varied demonstrations."
        )

    print(f"[Expert] {len(raw)} transitions loaded.")
    return torch.as_tensor(obs), torch.as_tensor(act_idxs)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helper
# ──────────────────────────────────────────────────────────────────────────────

def make_checkpoint(policy, obs_dim, hidden, cfg):
    return {
        "policy":     policy.state_dict(),
        "obs_dim":    obs_dim,
        "n_actions":  N_ACTIONS,
        "action_set": ACTION_SET,
        "hidden":     hidden,
        "cfg":        cfg,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Validation pass
# ──────────────────────────────────────────────────────────────────────────────

def validate(policy, val_loader, device):
    policy.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for obs_b, act_b in val_loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            logits = policy(obs_b)
            loss   = F.cross_entropy(logits, act_b)
            total_loss += loss.item() * len(act_b)
            correct    += (logits.argmax(dim=-1) == act_b).sum().item()
            total      += len(act_b)

    policy.train()
    return total_loss / total, 100.0 * correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_bc():
    print("=" * 60)
    print("Behavioural Cloning (BC)  –  Soccer Striker  (CPU)")
    print("Supervised classification: obs → expert action index")
    print("=" * 60)

    device = torch.device("cpu")

    # ── expert data ──────────────────────────────────────────────────
    obs_all, acts_all = load_expert(CFG["expert_path"])

    obs_dim  = obs_all.shape[1]
    n_data   = len(obs_all)

    # ── train / val split ─────────────────────────────────────────────
    n_val   = max(1, int(n_data * CFG["val_split"]))
    n_train = n_data - n_val

    dataset    = TensorDataset(obs_all, acts_all)
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size = CFG["batch_size"],
        shuffle    = True,
        drop_last  = False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size = CFG["batch_size"] * 4,
        shuffle    = False,
    )

    print(f"[Data]   train={n_train}  val={n_val}  obs_dim={obs_dim}")

    # ── policy ───────────────────────────────────────────────────────
    hidden = CFG["hidden"]
    policy = BCPolicy(obs_dim, N_ACTIONS, hidden).to(device)

    optimiser = torch.optim.AdamW(
        policy.parameters(),
        lr           = CFG["lr"],
        weight_decay = CFG["weight_decay"],
    )

    # Cosine annealing: smoothly reduces LR from lr to lr_min over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max  = CFG["n_epochs"],
        eta_min= CFG["lr_min"],
    )

    # ── environment (for periodic live eval) ─────────────────────────
    from my_envs.striker_env import StrikerRLEnv
    env = StrikerRLEnv(discrete=True)

    # ── tracking ─────────────────────────────────────────────────────
    best_val_acc     = 0.0
    best_eval_score  = -float("inf")
    train_losses     = deque(maxlen=CFG["log_window"])
    t_start          = time.time()

    print()
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  "
          f"{'Val Acc':>8}  {'LR':>8}  {'Eval Score':>11}")
    print("-" * 65)

    for epoch in range(1, CFG["n_epochs"] + 1):
        policy.train()
        epoch_loss = 0.0
        n_batches  = 0

        for obs_b, act_b in train_loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)

            logits = policy(obs_b)
            loss   = F.cross_entropy(logits, act_b)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimiser.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        mean_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(mean_train_loss)

        # Validation
        val_loss, val_acc = validate(policy, val_loader, device)

        # Save best by val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                make_checkpoint(policy, obs_dim, hidden, CFG),
                CFG["save_path_best"]
            )

        # Periodic live env evaluation
        eval_str = ""
        if epoch % CFG["eval_interval"] == 0:
            eval_score = 0.0
            for _ in range(CFG["num_eval_episodes"]):
                eval_obs, _ = env.reset()
                e_done = False
                while not e_done:
                    act_idx = policy.predict(
                        torch.as_tensor(eval_obs, dtype=torch.float32)
                    )
                    eval_obs, r, term, trunc, _ = env.step(act_idx)
                    eval_score += r
                    e_done = term or trunc

            avg_eval = eval_score / CFG["num_eval_episodes"]
            eval_str = f"  eval={avg_eval:>7.1f}"

            if avg_eval >= best_eval_score:
                best_eval_score = avg_eval
                eval_str += "  ★"

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"{epoch:>6}  "
            f"{mean_train_loss:>10.4f}  "
            f"{val_loss:>9.4f}  "
            f"{val_acc:>7.1f}%  "
            f"{current_lr:>8.6f}"
            f"{eval_str}"
        )

        # Early stopping hint (not enforced — you may want more epochs)
        if val_acc >= 99.0:
            print(f"[Early stop]  val_acc={val_acc:.1f}% — dataset fully memorised.")
            break

    # ── final save ────────────────────────────────────────────────────
    torch.save(make_checkpoint(policy, obs_dim, hidden, CFG), CFG["save_path"])

    elapsed = time.time() - t_start
    print()
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best val accuracy   : {best_val_acc:.1f}%")
    print(f"Best eval score     : {best_eval_score:.1f}")
    print(f"--- SAVED last  → {CFG['save_path']} ---")
    print(f"--- SAVED best  → {CFG['save_path_best']} (by val_acc) ---")
    print("Use type_2_bc.pt for play_bc.py")
    env.close()


if __name__ == "__main__":
    train_bc()