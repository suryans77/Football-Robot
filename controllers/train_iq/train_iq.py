"""
Pure IQ-Learn for a soccer-dribbling robot  (CPU-friendly, no SB3)
===================================================================

Fixes vs previous versions
----------------------------
1. Expert episode boundaries read from stored "done" key — no heuristics.
   If your pkl was recorded without "done", a fallback warning is printed
   and the final frame of each recorded run is marked terminal via a
   "episode_id" key (second fallback: mark only the very last frame).

2. Policy terminal states are now masked in compute_iq_loss:
       br_p = q_p - γ·V(s′)·(1 − done)
   Previously p_dones was sampled but then silently discarded (_).

3. Target network soft-updated every step (tau=0.005) — smoother than
   periodic hard syncs.

4. terminated vs truncated correctly separated in the replay buffer:
   only true termination zeros the Bellman bootstrap.

Dependencies
------------
    pip install torch numpy gymnasium
    (striker_env must be importable)
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
# Config
# ──────────────────────────────────────────────────────────────────────────────

CFG = dict(
    expert_path             = "expert_data_augmented.pkl",
    save_path               = "iq_striker_brain.pt",
    total_timesteps         = 100_000,
    batch_size              = 64,
    hidden                  = [128, 128],
    lr_q                    = 3e-4,
    lr_actor                = 3e-4,
    gamma                   = 0.99,
    alpha                   = 0.1,
    learn_alpha             = True,
    target_entropy_frac     = 0.5,       # target entropy = frac * -act_dim
    iq_reg                  = 1e-3,
    q_updates_per_step      = 1,
    actor_updates_per_step  = 1,
    tau                     = 0.005,     # soft target update — applied every step
    warmup_steps            = 1_000,
    buffer_size             = 50_000,
    eval_interval           = 5_000,
    log_window              = 500,
)

# ──────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.cap   = capacity
        self.ptr   = 0
        self.size  = 0
        self.obs   = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts  = np.zeros((capacity, act_dim), dtype=np.float32)
        self.nobs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1),       dtype=np.float32)

    def add(self, obs, act, nobs, terminated: bool):
        """
        Pass `terminated` (true MDP end), NOT `done = terminated or truncated`.
        Truncation is a time-limit bookkeeping detail — the MDP continues, so
        the Bellman bootstrap should NOT be zeroed for truncated transitions.
        """
        self.obs  [self.ptr] = obs
        self.acts [self.ptr] = act
        self.nobs [self.ptr] = nobs
        self.dones[self.ptr] = float(terminated)
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n: int, device: torch.device):
        idx  = np.random.randint(0, self.size, size=n)
        to_t = lambda x: torch.as_tensor(x[idx], device=device)
        return to_t(self.obs), to_t(self.acts), to_t(self.nobs), to_t(self.dones)


# ──────────────────────────────────────────────────────────────────────────────
# Networks
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


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int]):
        super().__init__()
        self.net = _mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int]):
        super().__init__()
        self.trunk   = _mlp(obs_dim, hidden[-1], hidden[:-1])
        self.mu_head = nn.Linear(hidden[-1], act_dim)
        self.ls_head = nn.Linear(hidden[-1], act_dim)

    def _dist_params(self, obs: torch.Tensor):
        h       = F.relu(self.trunk(obs))
        mu      = self.mu_head(h)
        log_std = self.ls_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor):
        """Returns (action, log_prob) with action in (-1, 1)^act_dim."""
        mu, log_std = self._dist_params(obs)
        std  = log_std.exp()
        u    = mu + std * torch.randn_like(mu)
        a    = torch.tanh(u)
        logp = (
            -0.5 * ((u - mu) / std).pow(2)
            - log_std
            - 0.5 * np.log(2 * np.pi)
            - torch.log(1 - a.pow(2) + 1e-6)
        ).sum(dim=-1)
        return a, logp

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self._dist_params(obs)
        return torch.tanh(mu)


# ──────────────────────────────────────────────────────────────────────────────
# Expert data loader  (fixed episode boundary handling)
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
        raise ValueError(
            "expert_data.pkl is empty or has unexpected format. "
            "Re-run your recording script and confirm transitions are appended."
        )

    obs  = np.array([d["obs"]    for d in raw], dtype=np.float32)
    acts = np.array([d["action"] for d in raw], dtype=np.float32)

    # ── episode boundary strategy (in priority order) ────────────────
    #
    # Strategy A: "done" key was stored at record time  ← best, use this
    # Strategy B: "episode_id" key groups frames into episodes
    # Strategy C: only the very last frame is terminal  ← last resort
    #
    # To use Strategy A, make sure your recording script does:
    #
    #   expert_data.append({
    #       "obs":    obs,
    #       "action": action,
    #       "done":   bool(terminated),   # NOT terminated or truncated
    #   })

    if "done" in raw[0]:
        dones = np.array([float(d["done"]) for d in raw], dtype=np.float32)
        print("[Expert] Using stored 'done' key for episode boundaries.")

    elif "episode_id" in raw[0]:
        warnings.warn(
            "[Expert] No 'done' key found — inferring boundaries from 'episode_id'. "
            "Add 'done' to your recording script for exact boundaries."
        )
        dones = np.zeros(len(raw), dtype=np.float32)
        for i in range(len(raw) - 1):
            if raw[i]["episode_id"] != raw[i + 1]["episode_id"]:
                dones[i] = 1.0
        dones[-1] = 1.0

    else:
        warnings.warn(
            "[Expert] No 'done' or 'episode_id' key found — marking only the "
            "final frame as terminal. Add 'done' to your recording script. "
            "Training may be slightly suboptimal until then."
        )
        dones = np.zeros(len(raw), dtype=np.float32)
        dones[-1] = 1.0

    # Build next_obs: shift by one; terminal frames point to themselves
    next_obs = np.roll(obs, -1, axis=0)
    terminal_mask = dones.astype(bool)
    next_obs[terminal_mask] = obs[terminal_mask]

    n_terminal = int(dones.sum())
    print(f"[Expert] {len(raw)} transitions  "
          f"obs={obs.shape}  acts={acts.shape}  "
          f"terminal_frames={n_terminal}")

    return (
        torch.as_tensor(obs,             device=device),
        torch.as_tensor(acts,            device=device),
        torch.as_tensor(next_obs,        device=device),
        torch.as_tensor(dones[:, None],  device=device),  # (N,1)
    )


def sample_expert(e_obs, e_acts, e_nobs, e_dones, n: int):
    idx = torch.randint(len(e_obs), (n,), device=e_obs.device)
    return e_obs[idx], e_acts[idx], e_nobs[idx], e_dones[idx]


# ──────────────────────────────────────────────────────────────────────────────
# IQ-Learn core
# ──────────────────────────────────────────────────────────────────────────────

def soft_value(q_net, actor, obs: torch.Tensor, alpha: float) -> torch.Tensor:
    """V(s) ≈ Q(s, ã) − α·log π(ã|s)   single-sample Monte Carlo."""
    a, logp = actor.sample(obs)
    return q_net(obs, a) - alpha * logp


def compute_iq_loss(q_net, q_tgt, actor,
                    e_obs, e_acts, e_nobs, e_dones,
                    p_obs, p_acts, p_nobs, p_dones,   # FIX: p_dones now used
                    alpha, gamma, reg):
    """
    χ²-regularised IQ-Learn loss.

    Expert terms push Q up on (s,a) pairs the expert visited.
    Policy χ² term prevents Q from collapsing off the expert support.
    Both sides now correctly zero the Bellman bootstrap at terminal states.
    """
    # ── expert side ───────────────────────────────────────────────────
    with torch.no_grad():
        v_e_next = soft_value(q_tgt, actor, e_nobs, alpha)
        v_e_curr = soft_value(q_tgt, actor, e_obs,  alpha)

    q_e  = q_net(e_obs, e_acts)
    br_e = (q_e
            - gamma * v_e_next * (1.0 - e_dones.squeeze(-1))  # zero at terminal
            - v_e_curr)

    # ── policy side ───────────────────────────────────────────────────
    with torch.no_grad():
        v_p_next = soft_value(q_tgt, actor, p_nobs, alpha)

    q_p  = q_net(p_obs, p_acts)
    br_p = (q_p
            - gamma * v_p_next * (1.0 - p_dones.squeeze(-1)))  # FIX: was missing

    # ── combine ───────────────────────────────────────────────────────
    expert_loss = -br_e.mean()
    chi2_loss   =  0.5 * (br_p ** 2).mean()
    l2_loss     =  reg  * (q_e ** 2).mean()

    loss = expert_loss + chi2_loss + l2_loss
    return loss, {
        "iq_loss":   loss.item(),
        "expert_Q":  q_e.mean().item(),
        "policy_Q":  q_p.mean().item(),
        "expert_br": br_e.mean().item(),
    }


def compute_actor_loss(q_net, actor, obs: torch.Tensor, alpha: float):
    """
    Soft policy improvement.
    Minimise: E[ α·log π(a|s) − Q(s,a) ]
    = maximise: E[ Q(s,a) ] + α·H(π)
    """
    a, logp = actor.sample(obs)
    loss    = (alpha * logp - q_net(obs, a)).mean()
    return loss, {"actor_loss": loss.item(), "entropy": -logp.mean().item()}


def soft_update(src: nn.Module, tgt: nn.Module, tau: float):
    for sp, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_iq():
    print("=" * 60)
    print("Pure IQ-Learn  –  Soccer Striker  (CPU)")
    print("=" * 60)

    device = torch.device("cpu")

    # ── environment ──────────────────────────────────────────────────
    from striker_env import StrikerRLEnv
    env     = StrikerRLEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"[Env]    obs_dim={obs_dim}  act_dim={act_dim}")

    # ── networks ─────────────────────────────────────────────────────
    hidden = CFG["hidden"]
    q_net  = SoftQNetwork(obs_dim, act_dim, hidden)
    q_tgt  = deepcopy(q_net); q_tgt.eval()
    actor  = SquashedGaussianActor(obs_dim, act_dim, hidden)

    q_opt  = torch.optim.Adam(q_net.parameters(), lr=CFG["lr_q"])
    a_opt  = torch.optim.Adam(actor.parameters(), lr=CFG["lr_actor"])

    # ── auto-tune α ──────────────────────────────────────────────────
    target_entropy = CFG["target_entropy_frac"] * (-act_dim)
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
    buf = ReplayBuffer(CFG["buffer_size"], obs_dim, act_dim)

    # ── logging ──────────────────────────────────────────────────────
    log_keys = ("iq_loss", "actor_loss", "entropy", "expert_Q", "policy_Q")
    logs     = {k: deque(maxlen=CFG["log_window"]) for k in log_keys}

    total   = CFG["total_timesteps"]
    bs      = CFG["batch_size"]
    warmup  = CFG["warmup_steps"]
    gamma   = CFG["gamma"]
    reg     = CFG["iq_reg"]

    obs, _  = env.reset()
    t_start = time.time()

    for t in range(1, total + 1):

        # ── 1. collect one step ───────────────────────────────────────
        alpha = get_alpha()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = (env.action_space.sample() if t < warmup
                      else actor.sample(obs_t)[0].squeeze(0).numpy())

        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buf.add(obs, action, next_obs, terminated)  # terminated only, not done
        obs = next_obs if not done else env.reset()[0]

        if t < warmup or buf.size < bs:
            continue

        # ── 2. update Q ───────────────────────────────────────────────
        for _ in range(CFG["q_updates_per_step"]):
            p_o, p_a, p_n, p_d = buf.sample(bs, device)          # p_d now used
            e_o, e_a, e_n, e_d = sample_expert(e_obs, e_acts, e_nobs, e_dones, bs)

            q_l, q_info = compute_iq_loss(
                q_net, q_tgt, actor,
                e_o, e_a, e_n, e_d,
                p_o, p_a, p_n, p_d,                               # pass p_d through
                alpha, gamma, reg,
            )
            q_opt.zero_grad()
            q_l.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            q_opt.step()
            for k, v in q_info.items():
                if k in logs: logs[k].append(v)

        # ── 3. update actor ───────────────────────────────────────────
        for _ in range(CFG["actor_updates_per_step"]):
            p_o, *_ = buf.sample(bs, device)
            a_l, a_info = compute_actor_loss(q_net, actor, p_o, alpha)
            a_opt.zero_grad()
            a_l.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            a_opt.step()
            for k, v in a_info.items():
                if k in logs: logs[k].append(v)

        # ── 4. update α ───────────────────────────────────────────────
        if CFG["learn_alpha"] and log_alpha is not None:
            with torch.no_grad():
                p_o, *_ = buf.sample(bs, device)
                _, logp = actor.sample(p_o)
            alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()

        # ── 5. soft-update target Q  (every step) ─────────────────────
        soft_update(q_net, q_tgt, CFG["tau"])

        # ── 6. log ────────────────────────────────────────────────────
        if t % CFG["eval_interval"] == 0:
            means = {k: np.mean(v) for k, v in logs.items() if v}
            print(
                f"[{t:>7,}/{total:,}]  "
                f"fps={t/(time.time()-t_start):.0f}  "
                f"IQ={means.get('iq_loss',  0):+.4f}  "
                f"actor={means.get('actor_loss', 0):+.4f}  "
                f"H={means.get('entropy',   0):.3f}  "
                f"α={alpha:.4f}  "
                f"Q_exp={means.get('expert_Q', 0):+.3f}  "
                f"Q_pol={means.get('policy_Q', 0):+.3f}"
            )

    # ── save ──────────────────────────────────────────────────────────
    torch.save({
        "actor":   actor.state_dict(),
        "q_net":   q_net.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden":  hidden,
        "cfg":     CFG,
    }, CFG["save_path"])
    print(f"\n--- SAVED → {CFG['save_path']} ---")
    env.close()


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def load_and_run(checkpoint_path: str):
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    actor = SquashedGaussianActor(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    from striker_env import StrikerRLEnv
    env  = StrikerRLEnv()
    obs, _ = env.reset()
    total_reward = 0.0

    for _ in range(1_000):
        with torch.no_grad():
            obs_t  = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = actor.deterministic(obs_t).squeeze(0).numpy()
        obs, r, terminated, truncated, _ = env.step(action)
        total_reward += r
        if terminated or truncated:
            print(f"Episode done  total_reward={total_reward:.2f}")
            obs, _ = env.reset()
            total_reward = 0.0

    env.close()


if __name__ == "__main__":
    train_iq()