"""
Discrete-Action IQ-Learn (PURE) for a soccer-dribbling robot  (CPU-friendly)
=============================================================================

This is the pure IQ-Learn baseline — identical to train_iq_cql.py in every
way EXCEPT the CQL regularisation term is removed.

Use this to compare against the CQL version:
  Pure IQ-Learn  →  iq_pure_brain_best.pt
  IQ-Learn + CQL →  iq_striker_brain_best.pt

What is removed vs train_iq_cql.py
-------------------------------------
- cql_weight removed from CFG
- CQL term removed from loss:
    REMOVED:  lse = logsumexp(Q(s,·))
    REMOVED:  cql_loss = (lse - Q(s, a_expert)).mean()
    REMOVED:  loss += cql_weight * cql_loss
- cql_loss and Q_spread removed from logs and print

Everything else is identical:
  double-Q networks, soft target updates, alpha auto-tune,
  eval loop, best-checkpoint saving, expert loader, replay buffer.

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
# Config  (cql_weight removed — only difference from train_iq_cql.py)
# ──────────────────────────────────────────────────────────────────────────────

CFG = dict(
    expert_path         = "full_data.pkl",
    save_path           = "full_pureIQ.pt",
    save_path_best      = "full_pureIQ_best.pt",
    total_timesteps     = 100_000,
    batch_size          = 256,
    hidden              = [256, 256],
    lr_q                = 3e-4,
    gamma               = 0.95,
    alpha               = 0.1,
    learn_alpha         = True,
    target_entropy      = -np.log(1.0 / N_ACTIONS) * 0.5,
    iq_reg              = 1e-3,
    q_updates_per_step  = 1,
    tau                 = 0.005,
    warmup_steps        = 1_000,
    buffer_size         = 50_000,
    eval_interval       = 5_000,
    log_window          = 500,
)

# ──────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
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
    """Q(s) → R^N_ACTIONS   (all action values in one forward pass)"""
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int]):
        super().__init__()
        self.net = _mlp(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ──────────────────────────────────────────────────────────────────────────────
# Soft value and policy  (exact — no sampling)
# ──────────────────────────────────────────────────────────────────────────────

def soft_value_exact(q_vals: torch.Tensor, alpha: float) -> torch.Tensor:
    """V(s) = α · logsumexp(Q(s,·) / α)  — shape (batch,)"""
    return alpha * torch.logsumexp(q_vals / alpha, dim=-1)


def policy_probs(q_vals: torch.Tensor, alpha: float) -> torch.Tensor:
    """π(a|s) = softmax(Q(s,·) / α)  — shape (batch, n_actions)"""
    return F.softmax(q_vals / alpha, dim=-1)


def min_q(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return torch.min(q1, q2)


# ──────────────────────────────────────────────────────────────────────────────
# Expert data loader
# ──────────────────────────────────────────────────────────────────────────────

def load_expert(path: str, device: torch.device):
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
        raise ValueError("full_data.pkl is empty. Re-run record_expert.py.")

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

    return (
        torch.as_tensor(obs,            device=device),
        torch.as_tensor(act_idxs,       device=device),
        torch.as_tensor(next_obs,       device=device),
        torch.as_tensor(dones[:, None], device=device),
    )


def sample_expert(e_obs, e_acts, e_nobs, e_dones, n: int):
    idx = torch.randint(len(e_obs), (n,), device=e_obs.device)
    return e_obs[idx], e_acts[idx], e_nobs[idx], e_dones[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Pure IQ-Learn loss  (CQL term removed)
# ──────────────────────────────────────────────────────────────────────────────

def compute_iq_loss(q1, q2, q1_tgt, q2_tgt,
                    e_obs, e_acts, e_nobs, e_dones,
                    p_obs, p_acts, p_nobs, p_dones,
                    alpha, gamma, iq_reg):
    """
    Pure IQ-Learn χ² loss — no CQL term.

    expert_loss : -E_expert[ Q(s,a) - γ·V(s') - V(s) ]
                  Pushes Q up for actions the expert took.

    chi2_loss   :  0.5 · E_policy[ (Q(s,a) - γ·V(s'))² ]
                  χ² penalty — the only thing keeping Q bounded.
                  (CQL version adds logsumexp on top of this.)

    l2_loss     :  iq_reg · E_expert[ Q(s,a)² ]
                  Light L2 to prevent unbounded growth.

    Bellman targets use min(Q1_tgt, Q2_tgt) — double-Q kept.
    """
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, dtype=torch.float32)
        return x

    e_obs  = to_tensor(e_obs)
    e_nobs = to_tensor(e_nobs)
    p_obs  = to_tensor(p_obs)
    p_nobs = to_tensor(p_nobs)
    batch  = len(e_acts)

    # ── Bellman targets — min of both target networks ─────────────────
    with torch.no_grad():
        v_e_next = soft_value_exact(min_q(q1_tgt(e_nobs), q2_tgt(e_nobs)), alpha)
        v_e_curr = soft_value_exact(min_q(q1_tgt(e_obs),  q2_tgt(e_obs)),  alpha)
        v_p_next = soft_value_exact(min_q(q1_tgt(p_nobs), q2_tgt(p_nobs)), alpha)

    total_loss = torch.tensor(0.0)
    info = {}

    for qi, q_net in enumerate((q1, q2)):
        q_e_all = q_net(e_obs)                               # (B, 9)
        q_p_all = q_net(p_obs)                               # (B, 9)

        q_e = q_e_all[torch.arange(batch), e_acts]           # expert Q
        q_p = q_p_all[torch.arange(batch), p_acts]           # policy Q

        # ── IQ Bellman residuals ──────────────────────────────────────
        br_e = (q_e
                - gamma * v_e_next * (1.0 - e_dones.squeeze(-1))
                - v_e_curr)

        br_p = (q_p
                - gamma * v_p_next * (1.0 - p_dones.squeeze(-1)))

        expert_loss = -br_e.mean()
        chi2_loss   =  0.5 * (br_p ** 2).mean()
        l2_loss     =  iq_reg * (q_e ** 2).mean()

        # ── NO CQL term ───────────────────────────────────────────────

        loss = expert_loss + chi2_loss + l2_loss
        total_loss = total_loss + loss

        if qi == 0:
            info = {
                "iq_loss":   loss.item(),
                "expert_Q":  q_e.mean().item(),
                "policy_Q":  q_p.mean().item(),
                "expert_br": br_e.mean().item(),
            }

    return total_loss, info


# ──────────────────────────────────────────────────────────────────────────────
# Alpha update
# ──────────────────────────────────────────────────────────────────────────────

def compute_alpha_loss(log_alpha, q1, q2, obs, alpha, target_entropy):
    with torch.no_grad():
        q_vals  = min_q(q1(obs), q2(obs))
        probs   = policy_probs(q_vals, alpha)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
    loss = -log_alpha * (entropy - target_entropy)
    return loss, entropy.item()


# ──────────────────────────────────────────────────────────────────────────────
# Soft target update
# ──────────────────────────────────────────────────────────────────────────────

def soft_update(src: nn.Module, tgt: nn.Module, tau: float):
    for sp, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


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

def train_iq():
    print("=" * 60)
    print("Discrete IQ-Learn (PURE)  –  Soccer Striker  (CPU)")
    print(f"Action space : {N_ACTIONS} discrete actions")
    print("No CQL regularisation — baseline for comparison")
    print("=" * 60)

    device = torch.device("cpu")

    # ── environment ──────────────────────────────────────────────────
    from my_envs.striker_env import StrikerRLEnv
    env     = StrikerRLEnv()
    obs_dim = env.observation_space.shape[0]
    print(f"[Env]  obs_dim={obs_dim}  n_actions={N_ACTIONS}")

    # ── double Q-networks ─────────────────────────────────────────────
    hidden = CFG["hidden"]
    q1     = DiscreteQNetwork(obs_dim, N_ACTIONS, hidden)
    q2     = DiscreteQNetwork(obs_dim, N_ACTIONS, hidden)
    q1_tgt = deepcopy(q1); q1_tgt.eval()
    q2_tgt = deepcopy(q2); q2_tgt.eval()

    q_opt = torch.optim.Adam(
        list(q1.parameters()) + list(q2.parameters()),
        lr=CFG["lr_q"]
    )

    # ── auto-tune α ──────────────────────────────────────────────────
    if CFG["learn_alpha"]:
        log_alpha = torch.tensor(np.log(CFG["alpha"]), requires_grad=True)
        alpha_opt = torch.optim.Adam([log_alpha], lr=1e-4)
        get_alpha = lambda: log_alpha.exp().item()
    else:
        log_alpha = None
        get_alpha = lambda: CFG["alpha"]

    # ── expert data ──────────────────────────────────────────────────
    e_obs, e_acts, e_nobs, e_dones = load_expert(CFG["expert_path"], device)

    # ── replay buffer ────────────────────────────────────────────────
    buf = ReplayBuffer(CFG["buffer_size"], obs_dim)

    # ── logging ──────────────────────────────────────────────────────
    log_keys = ("iq_loss", "expert_Q", "policy_Q", "expert_br", "entropy")
    logs     = {k: deque(maxlen=CFG["log_window"]) for k in log_keys}
    best_br          = -float("inf")
    best_eval_score  = -float("inf")

    total   = CFG["total_timesteps"]
    bs      = CFG["batch_size"]
    warmup  = CFG["warmup_steps"]
    gamma   = CFG["gamma"]

    obs, _  = env.reset()
    t_start = time.time()

    for t in range(1, total + 1):

        # ── 1. collect one step ───────────────────────────────────────
        alpha = get_alpha()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if t < warmup:
                act_idx = np.random.randint(N_ACTIONS)
            else:
                q_vals  = min_q(q1(obs_t), q2(obs_t))
                probs   = policy_probs(q_vals, alpha).squeeze(0).numpy()
                act_idx = int(np.random.choice(N_ACTIONS, p=probs))

        action   = index_to_action(act_idx)
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buf.add(obs, act_idx, next_obs, terminated)
        obs = next_obs if not done else env.reset()[0]

        if t < warmup or buf.size < bs:
            continue

        # ── 2. update Q (pure IQ loss) ────────────────────────────────
        for _ in range(CFG["q_updates_per_step"]):
            p_o, p_a, p_n, p_d = buf.sample(bs, device)
            e_o, e_a, e_n, e_d = sample_expert(e_obs, e_acts, e_nobs, e_dones, bs)

            q_l, q_info = compute_iq_loss(
                q1, q2, q1_tgt, q2_tgt,
                e_o, e_a, e_n, e_d,
                p_o, p_a, p_n, p_d,
                alpha, gamma, CFG["iq_reg"],
            )
            q_opt.zero_grad()
            q_l.backward()
            nn.utils.clip_grad_norm_(
                list(q1.parameters()) + list(q2.parameters()), 10.0
            )
            q_opt.step()
            for k, v in q_info.items():
                if k in logs: logs[k].append(v)

        # ── 3. update α ───────────────────────────────────────────────
        if CFG["learn_alpha"] and log_alpha is not None:
            p_o, *_ = buf.sample(bs, device)
            a_loss, ent = compute_alpha_loss(
                log_alpha, q1, q2, p_o, alpha, CFG["target_entropy"]
            )
            alpha_opt.zero_grad()
            a_loss.backward()
            alpha_opt.step()
            logs["entropy"].append(ent)

        # ── 4. soft-update target networks ────────────────────────────
        soft_update(q1, q1_tgt, CFG["tau"])
        soft_update(q2, q2_tgt, CFG["tau"])

        # ── 5. eval & logging ─────────────────────────────────────────
        if t % CFG["eval_interval"] == 0:

            means      = {k: np.mean(v) for k, v in logs.items() if v}
            current_br = means.get("expert_br", 0)
            if current_br > best_br:
                best_br = current_br

            eval_score       = 0.0
            NUM_EVAL_EPISODES = 10

            for _ in range(NUM_EVAL_EPISODES):
                eval_obs, _ = env.reset()
                e_done = False
                while not e_done:
                    with torch.no_grad():
                        eval_obs_t   = torch.as_tensor(eval_obs, dtype=torch.float32).unsqueeze(0)
                        eval_act_idx = min_q(q1(eval_obs_t), q2(eval_obs_t)).argmax(dim=-1).item()
                    e_action = index_to_action(eval_act_idx)
                    eval_obs, r, term, trunc, _ = env.step(e_action)
                    eval_score += r
                    e_done = term or trunc

            avg_eval_score = eval_score / NUM_EVAL_EPISODES

            print(
                f"[{t:>7,}/{total:,}]  "
                f"fps={t/(time.time()-t_start):.0f}  "
                f"IQ={means.get('iq_loss',  0):+.1f}  "
                f"Q_exp={means.get('expert_Q', 0):+.1f}  "
                f"br={means.get('expert_br', 0):+.3f}  "
                f"best_br={best_br:+.3f}  "
                f"| EVAL SCORE: {avg_eval_score:.1f}  "
                f"| best_eval={best_eval_score:.1f}"
            )

            if avg_eval_score >= best_eval_score:
                best_eval_score = avg_eval_score
                torch.save(
                    make_checkpoint(q1, q2, obs_dim, hidden, CFG),
                    CFG["save_path_best"]
                )
                print(f"   >>> NEW BEST MODEL SAVED! (High Score: {best_eval_score:.1f}) <<<")

    # ── final save ────────────────────────────────────────────────────
    torch.save(
        make_checkpoint(q1, q2, obs_dim, hidden, CFG),
        CFG["save_path"]
    )
    print(f"\n--- SAVED last  → {CFG['save_path']} ---")
    print(f"--- SAVED best  → {CFG['save_path_best']} (best_br={best_br:+.4f}) ---")
    print("Use the _best checkpoint for play_model.py")
    env.close()


if __name__ == "__main__":
    train_iq()