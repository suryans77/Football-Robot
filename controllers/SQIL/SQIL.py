"""
SQIL (Soft Q Imitation Learning) for the soccer striker robot
=============================================================

How SQIL works
--------------
SQIL is the simplest possible imitation learning algorithm that still
uses Q-learning. It makes one key observation:

  - Give expert transitions reward = +1
  - Give policy transitions reward =  0
  - Train a normal DQN on the mixed dataset

That's it. No adversarial training (GAIL), no inverse reward learning
(IQ-Learn), no behaviour cloning loss. The Q-function naturally learns
to assign high value to states/actions the expert visited (because those
transitions carry reward=1) and low value elsewhere (reward=0).

Why it works despite such a simple reward signal:
  Q*(s, a_expert) ≈ Σ γ^t * 1  = 1/(1-γ)   along expert trajectory
  Q*(s, a_other)  ≈ 0                        everywhere else
  → policy = argmax Q will imitate the expert

Architecture
------------
  - Double Q-networks Q1, Q2  (same as IQ-Learn scripts)
  - Expert buffer: fixed, loaded from expert_data.pkl, reward = 1
  - Policy buffer: rolling replay, reward = 0
  - Each update batch = bs//2 from expert + bs//2 from policy
  - Standard Double-DQN Bellman loss (no IQ or CQL terms)
  - Hard target network sync every `target_update_every` steps

Differences vs IQ-Learn
------------------------
  IQ-Learn:  learns the reward function implicitly via a soft Q loss
  SQIL:      assigns fixed binary rewards (+1 expert, 0 policy) and
             runs standard Q-learning — much simpler, often competitive

Action index mapping
---------------------
  0: [-1, -1]   1: [-1,  0]   2: [-1, +1]
  3: [ 0, -1]   4: [ 0,  0]   5: [ 0, +1]
  6: [+1, -1]   7: [+1,  0]   8: [+1, +1]

Dependencies
------------
    pip install torch numpy gymnasium
"""

from __future__ import annotations

import pickle
import time
import warnings
from copy import deepcopy
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    expert_path         = "type_2_data.pkl",
    save_path           = "sqil_striker_brain.pt",
    save_path_best      = "type_2_sqil.pt",
    total_timesteps     = 25_000,
    batch_size          = 64,            # bs//2 from expert, bs//2 from policy
    hidden              = [128, 128],
    lr_q                = 3e-4,
    gamma               = 0.99,

    # ε-greedy exploration
    eps_start           = 1.0,
    eps_end             = 0.05,
    eps_decay_steps     = 15_000,        # linear decay over this many steps

    # Target network — hard sync (not soft) — standard DQN style
    target_update_every = 200,

    warmup_steps        = 1_000,
    policy_buffer_size  = 50_000,
    eval_interval       = 5_000,
    num_eval_episodes   = 10,
    log_window          = 500,
)

# ──────────────────────────────────────────────────────────────────────────────
# Replay buffers
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Policy buffer — all stored rewards are 0."""

    def __init__(self, capacity: int, obs_dim: int):
        self.cap   = capacity
        self.ptr   = 0
        self.size  = 0
        self.obs   = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts  = np.zeros( capacity,           dtype=np.int64)
        self.nobs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1),       dtype=np.float32)

    def add(self, obs, act_idx: int, nobs, terminated: bool):
        self.obs  [self.ptr] = obs
        self.acts [self.ptr] = act_idx
        self.nobs [self.ptr] = nobs
        self.dones[self.ptr] = float(terminated)
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n: int, device: torch.device):
        idx  = np.random.randint(0, self.size, size=n)
        to_t = lambda x: torch.as_tensor(x[idx], device=device)
        return to_t(self.obs), to_t(self.acts), to_t(self.nobs), to_t(self.dones)


class ExpertBuffer:
    """
    Fixed expert buffer — reward is always +1.
    Loaded once from disk and never modified.
    """

    def __init__(self, obs, act_idxs, next_obs, dones, device):
        self.obs   = torch.as_tensor(obs,       device=device)
        self.acts  = torch.as_tensor(act_idxs,  device=device)
        self.nobs  = torch.as_tensor(next_obs,  device=device)
        self.dones = torch.as_tensor(dones,     device=device)   # (N,1)
        self.n     = len(obs)

    def sample(self, n: int):
        idx = torch.randint(self.n, (n,), device=self.obs.device)
        return (
            self.obs [idx],
            self.acts[idx],
            self.nobs[idx],
            self.dones[idx],
        )


# ──────────────────────────────────────────────────────────────────────────────
# Network
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


class DiscreteQNetwork(nn.Module):
    """Q(s) → R^N_ACTIONS"""
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int]):
        super().__init__()
        self.net = _mlp(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ──────────────────────────────────────────────────────────────────────────────
# Expert data loader
# ──────────────────────────────────────────────────────────────────────────────

def load_expert(path: str, device: torch.device) -> ExpertBuffer:
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

    if "done" in raw[0]:
        dones = np.array([float(d["done"]) for d in raw], dtype=np.float32)
        print("[Expert] Using stored 'done' key for episode boundaries.")
    elif "episode_id" in raw[0]:
        warnings.warn("[Expert] Inferring boundaries from 'episode_id'.")
        dones = np.zeros(len(raw), dtype=np.float32)
        for i in range(len(raw) - 1):
            if raw[i]["episode_id"] != raw[i + 1]["episode_id"]:
                dones[i] = 1.0
        dones[-1] = 1.0
    else:
        warnings.warn("[Expert] No boundary info — marking only final frame terminal.")
        dones = np.zeros(len(raw), dtype=np.float32)
        dones[-1] = 1.0

    next_obs = np.roll(obs, -1, axis=0)
    next_obs[dones.astype(bool)] = obs[dones.astype(bool)]

    unique, counts = np.unique(act_idxs, return_counts=True)
    print("[Expert] Action distribution:")
    for u, c in zip(unique, counts):
        print(f"         idx {u} {ACTION_SET[u]} : {c} ({100*c/len(raw):.1f}%)")
    print(f"[Expert] {len(raw)} transitions  terminal_frames={int(dones.sum())}")

    return ExpertBuffer(obs, act_idxs, next_obs, dones[:, None], device)


# ──────────────────────────────────────────────────────────────────────────────
# SQIL loss
# ──────────────────────────────────────────────────────────────────────────────

def compute_sqil_loss(q1, q2, q1_tgt, q2_tgt,
                      e_obs, e_acts, e_nobs, e_dones,
                      p_obs, p_acts, p_nobs, p_dones,
                      gamma: float):
    """
    Double-DQN Bellman loss on a mixed expert+policy batch.

    Expert reward  = 1.0  (hard-coded — no learning needed)
    Policy reward  = 0.0  (hard-coded)

    Target:
      y = r + γ · min(Q1_tgt, Q2_tgt)(s', a*)   where a* = argmax Q1(s', ·)
      y = r   (if terminal)

    Loss:
      L = MSE( Q1(s, a) - y ) + MSE( Q2(s, a) - y )
    """
    batch_e = len(e_acts)
    batch_p = len(p_acts)

    # ── concatenate expert (r=1) and policy (r=0) batches ─────────────
    obs   = torch.cat([e_obs,   p_obs],   dim=0)
    acts  = torch.cat([e_acts,  p_acts],  dim=0)
    nobs  = torch.cat([e_nobs,  p_nobs],  dim=0)
    dones = torch.cat([e_dones, p_dones], dim=0).squeeze(-1)

    # Reward vector: 1 for expert rows, 0 for policy rows
    rewards = torch.cat([
        torch.ones (batch_e, device=e_obs.device),
        torch.zeros(batch_p, device=p_obs.device),
    ], dim=0)

    total_batch = batch_e + batch_p

    # ── Double-DQN targets ────────────────────────────────────────────
    with torch.no_grad():
        # Action selection with Q1 (online), value from min(Q1_tgt, Q2_tgt)
        next_actions = q1(nobs).argmax(dim=-1)                     # (B,)
        q1_next      = q1_tgt(nobs)[torch.arange(total_batch), next_actions]
        q2_next      = q2_tgt(nobs)[torch.arange(total_batch), next_actions]
        q_next       = torch.min(q1_next, q2_next)
        targets      = rewards + gamma * q_next * (1.0 - dones)    # (B,)

    # ── compute loss for both Q-networks ─────────────────────────────
    q1_pred = q1(obs)[torch.arange(total_batch), acts]             # (B,)
    q2_pred = q2(obs)[torch.arange(total_batch), acts]

    loss_q1 = F.mse_loss(q1_pred, targets)
    loss_q2 = F.mse_loss(q2_pred, targets)
    loss    = loss_q1 + loss_q2

    # Mean Q-values on expert transitions only (for logging)
    expert_q = q1_pred[:batch_e].mean().item()
    policy_q = q1_pred[batch_e:].mean().item()

    return loss, {
        "sqil_loss": loss.item(),
        "expert_Q":  expert_q,
        "policy_Q":  policy_q,
        # Bellman error on each partition — should converge toward 0
        "td_expert": (q1_pred[:batch_e] - targets[:batch_e]).abs().mean().item(),
        "td_policy": (q1_pred[batch_e:] - targets[batch_e:]).abs().mean().item(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Hard target update  (standard DQN — not soft)
# ──────────────────────────────────────────────────────────────────────────────

def hard_update(src: nn.Module, tgt: nn.Module):
    tgt.load_state_dict(src.state_dict())


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helper
# ──────────────────────────────────────────────────────────────────────────────

def make_checkpoint(q1, q2, obs_dim, hidden, cfg):
    return {
        "q1":         q1.state_dict(),
        "q2":         q2.state_dict(),
        "obs_dim":    obs_dim,
        "n_actions":  N_ACTIONS,
        "action_set": ACTION_SET,
        "hidden":     hidden,
        "cfg":        cfg,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_sqil():
    print("=" * 60)
    print("SQIL  –  Soccer Striker  (CPU, discrete action space)")
    print(f"Expert reward = 1  |  Policy reward = 0")
    print(f"Action space  : {N_ACTIONS} discrete actions")
    print("=" * 60)

    device = torch.device("cpu")

    # ── environment ──────────────────────────────────────────────────
    from my_envs.striker_env import StrikerRLEnv
    env     = StrikerRLEnv(discrete=True)
    obs_dim = env.observation_space.shape[0]
    print(f"[Env]  obs_dim={obs_dim}  n_actions={N_ACTIONS}")

    # ── networks ─────────────────────────────────────────────────────
    hidden = CFG["hidden"]
    q1     = DiscreteQNetwork(obs_dim, N_ACTIONS, hidden)
    q2     = DiscreteQNetwork(obs_dim, N_ACTIONS, hidden)
    q1_tgt = deepcopy(q1); q1_tgt.eval()
    q2_tgt = deepcopy(q2); q2_tgt.eval()

    q_opt = torch.optim.Adam(
        list(q1.parameters()) + list(q2.parameters()),
        lr=CFG["lr_q"]
    )

    # ── expert buffer (fixed, reward=1) ───────────────────────────────
    expert_buf = load_expert(CFG["expert_path"], device)

    # ── policy buffer (rolling, reward=0) ────────────────────────────
    policy_buf = ReplayBuffer(CFG["policy_buffer_size"], obs_dim)

    # ── logging ──────────────────────────────────────────────────────
    log_keys = ("sqil_loss", "expert_Q", "policy_Q", "td_expert", "td_policy")
    logs     = {k: deque(maxlen=CFG["log_window"]) for k in log_keys}
    best_eval_score = -float("inf")

    total   = CFG["total_timesteps"]
    bs      = CFG["batch_size"]
    half    = bs // 2
    warmup  = CFG["warmup_steps"]
    gamma   = CFG["gamma"]

    # ε-greedy schedule (linear decay)
    def get_eps(t: int) -> float:
        frac = min(t / CFG["eps_decay_steps"], 1.0)
        return CFG["eps_start"] + frac * (CFG["eps_end"] - CFG["eps_start"])

    obs, _  = env.reset()
    t_start = time.time()

    for t in range(1, total + 1):

        # ── 1. collect one step (ε-greedy) ───────────────────────────
        eps = get_eps(t)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if t < warmup or np.random.rand() < eps:
                act_idx = np.random.randint(N_ACTIONS)
            else:
                act_idx = torch.min(q1(obs_t), q2(obs_t)).argmax(dim=-1).item()

        next_obs, _, terminated, truncated, _ = env.step(act_idx)
        done = terminated or truncated

        policy_buf.add(obs, act_idx, next_obs, terminated)
        obs = next_obs if not done else env.reset()[0]

        if t < warmup or policy_buf.size < half:
            continue

        # ── 2. SQIL update ────────────────────────────────────────────
        # Sample half batch from expert (r=1), half from policy (r=0)
        e_o, e_a, e_n, e_d = expert_buf.sample(half)
        p_o, p_a, p_n, p_d = policy_buf.sample(half, device)

        loss, info = compute_sqil_loss(
            q1, q2, q1_tgt, q2_tgt,
            e_o, e_a, e_n, e_d,
            p_o, p_a, p_n, p_d,
            gamma,
        )

        q_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(q1.parameters()) + list(q2.parameters()), 10.0
        )
        q_opt.step()

        for k, v in info.items():
            if k in logs: logs[k].append(v)

        # ── 3. hard update target networks ───────────────────────────
        if t % CFG["target_update_every"] == 0:
            hard_update(q1, q1_tgt)
            hard_update(q2, q2_tgt)

        # ── 4. eval & logging ─────────────────────────────────────────
        if t % CFG["eval_interval"] == 0:
            means      = {k: np.mean(v) for k, v in logs.items() if v}
            eval_score = 0.0

            for _ in range(CFG["num_eval_episodes"]):
                eval_obs, _ = env.reset()
                e_done = False
                while not e_done:
                    with torch.no_grad():
                        eval_t   = torch.as_tensor(eval_obs, dtype=torch.float32).unsqueeze(0)
                        eval_act = torch.min(q1(eval_t), q2(eval_t)).argmax(dim=-1).item()
                    eval_obs, r, term, trunc, _ = env.step(eval_act)
                    eval_score += r
                    e_done = term or trunc

            avg_eval = eval_score / CFG["num_eval_episodes"]

            print(
                f"[{t:>7,}/{total:,}]  "
                f"fps={t/(time.time()-t_start):.0f}  "
                f"ε={eps:.3f}  "
                f"loss={means.get('sqil_loss', 0):+.4f}  "
                f"Q_exp={means.get('expert_Q', 0):+.3f}  "
                f"Q_pol={means.get('policy_Q', 0):+.3f}  "
                f"td_exp={means.get('td_expert', 0):.4f}  "
                f"td_pol={means.get('td_policy', 0):.4f}  "
                f"| EVAL: {avg_eval:.1f}  "
                f"| best: {best_eval_score:.1f}"
            )

            if avg_eval >= best_eval_score:
                best_eval_score = avg_eval
                torch.save(
                    make_checkpoint(q1, q2, obs_dim, hidden, CFG),
                    CFG["save_path_best"]
                )
                print(f"   >>> NEW BEST SQIL MODEL SAVED! ({best_eval_score:.1f}) <<<")

    # ── final save ────────────────────────────────────────────────────
    torch.save(
        make_checkpoint(q1, q2, obs_dim, hidden, CFG),
        CFG["save_path"]
    )
    print(f"\n--- SAVED last  → {CFG['save_path']} ---")
    print(f"--- SAVED best  → {CFG['save_path_best']} ---")
    env.close()


if __name__ == "__main__":
    train_sqil()