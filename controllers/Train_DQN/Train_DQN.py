"""
IQ-Learn → DQN fine-tuning for the soccer striker robot
=========================================================

What this does
--------------
Takes the Q-network weights trained by IQ-Learn and continues
training them with standard Double-DQN using actual environment
rewards. This is "warm-start RL from IL".

Why not use SB3 DQN
--------------------
SB3's DQN network is:
    Linear → ReLU → Linear → ReLU → Linear

IQ-Learn's network is:
    Linear → LayerNorm → ReLU → Linear → LayerNorm → ReLU → Linear

Different architectures — weight shapes don't match at the LayerNorm
positions. Direct copy would silently corrupt all weights after layer 1.
This script keeps the exact same DiscreteQNetwork class so the copy
is guaranteed correct.

Why this should improve on pure IQ-Learn
-----------------------------------------
IQ-Learn learned from expert demonstrations — it copies what you did.
The expert (you) was probably not perfect, especially against 2 defenders
with random positions you never faced during recording. DQN fine-tuning
lets the policy discover better strategies through actual trial and error
in the environment, starting from an already-competent base rather than
from random noise.

Training flow
-------------
1. Load IQ-Learn checkpoint → copy into online Q1, Q2 and targets Q1_tgt, Q2_tgt
2. Warm the buffer: run 500 steps with low-epsilon policy to fill buffer
   with meaningful (not fully random) transitions
3. Standard DQN loop:
   a. ε-greedy action (epsilon starts low — policy already knows how to play)
   b. Store transition with REAL env reward
   c. Sample batch from replay buffer
   d. Double-DQN Bellman loss: MSE(Q(s,a), r + γ·min(Q1_tgt,Q2_tgt)(s', a*))
   e. Hard target sync every N steps

Key differences from pure IQ-Learn
------------------------------------
  IQ-Learn:  implicit reward from Q-function, expert data mixed in every step
  This DQN:  actual env reward (progress, beaten, goal, collision etc.),
             no expert data used after initialisation

Epsilon schedule
-----------------
  Starts at 0.20 — the policy is already competent, we just want some exploration
  Decays to 0.05 over the first 40% of training
  Stays at 0.05 for the remainder

Dependencies
------------
    pip install torch numpy gymnasium
"""

from __future__ import annotations

import time
from copy import deepcopy
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(torch.get_num_threads())

# ──────────────────────────────────────────────────────────────────────────────
# Action set — must be identical to the one used in IQ-Learn training
# ──────────────────────────────────────────────────────────────────────────────

ACTION_SET = np.array([
    [-1., -1.], [-1.,  0.], [-1., +1.],
    [ 0., -1.], [ 0.,  0.], [ 0., +1.],
    [+1., -1.], [+1.,  0.], [+1., +1.],
], dtype=np.float32)

N_ACTIONS = len(ACTION_SET)


def index_to_action(idx: int) -> np.ndarray:
    return ACTION_SET[idx].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

CFG = dict(
    # ── checkpoints ──────────────────────────────────────────────────
    iq_checkpoint       = "full_pureIQ_best.pt",   # IQ-Learn weights to load
    save_path           = "dqn_finetuned.pt",
    save_path_best      = "dqn_finetuned_best.pt",

    # ── training ─────────────────────────────────────────────────────
    total_timesteps     = 100_000,
    batch_size          = 256,
    gamma               = 0.95,            # match IQ-Learn gamma
    lr                  = 1e-4,            # lower than IQ-Learn (3e-4) — fine-tuning
    target_update_every = 500,             # hard sync every N steps
    grad_clip           = 10.0,

    # ── epsilon schedule ─────────────────────────────────────────────
    # Low start because the policy is already good.
    # 0.25 means 1-in-4 actions are random exploration.
    eps_start           = 0.25,
    eps_end             = 0.05,
    eps_decay_steps     = 40_000,          # decay over first 40% of training

    # ── replay buffer ─────────────────────────────────────────────────
    buffer_size         = 50_000,
    warmup_steps        = 500,             # fill buffer before first update

    # ── eval ─────────────────────────────────────────────────────────
    eval_interval       = 5_000,
    num_eval_episodes   = 10,

    # ── logging ──────────────────────────────────────────────────────
    log_window          = 500,
)

# ──────────────────────────────────────────────────────────────────────────────
# Network — identical to IQ-Learn (LayerNorm must be preserved)
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
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int]):
        super().__init__()
        self.net = _mlp(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ──────────────────────────────────────────────────────────────────────────────
# Replay buffer — stores actual env rewards (not implicit IQ rewards)
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.cap     = capacity
        self.ptr     = 0
        self.size    = 0
        self.obs     = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts    = np.zeros( capacity,           dtype=np.int64)
        self.nobs    = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1),       dtype=np.float32)
        self.dones   = np.zeros((capacity, 1),       dtype=np.float32)

    def add(self, obs, act_idx: int, nobs, reward: float, terminated: bool):
        self.obs    [self.ptr] = obs
        self.acts   [self.ptr] = act_idx
        self.nobs   [self.ptr] = nobs
        self.rewards[self.ptr] = reward
        self.dones  [self.ptr] = float(terminated)
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n: int, device: torch.device):
        idx  = np.random.randint(0, self.size, size=n)
        to_t = lambda x: torch.as_tensor(x[idx], device=device)
        return (to_t(self.obs), to_t(self.acts),
                to_t(self.nobs), to_t(self.rewards), to_t(self.dones))


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────────────────

def load_iq_checkpoint(path: str, device: torch.device):
    """
    Load IQ-Learn checkpoint and return two initialised Q-networks.

    Both online networks (q1, q2) AND their targets are initialised
    from the IQ-Learn weights. The target networks start identical to
    the online networks, exactly as in standard DQN initialisation.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    obs_dim    = ckpt["obs_dim"]
    n_actions  = ckpt["n_actions"]
    hidden     = ckpt["hidden"]

    print(f"[IQ-Load]  obs_dim={obs_dim}  n_actions={n_actions}  hidden={hidden}")
    print(f"[IQ-Load]  Loading weights from '{path}'")

    # Verify action set matches — mismatched action sets would silently
    # produce a policy that takes completely wrong physical actions
    saved_action_set = ckpt.get("action_set", None)
    if saved_action_set is not None:
        if not np.allclose(saved_action_set, ACTION_SET):
            raise ValueError(
                "ACTION_SET in checkpoint does not match this script. "
                "The IQ-Learn model was trained with different actions."
            )
        print("[IQ-Load]  Action set verified ✓")

    # Build online networks and load IQ-Learn weights
    q1 = DiscreteQNetwork(obs_dim, n_actions, hidden).to(device)
    q2 = DiscreteQNetwork(obs_dim, n_actions, hidden).to(device)
    q1.load_state_dict(ckpt["q1"])
    q2.load_state_dict(ckpt["q2"])

    # Target networks start as exact copies — standard DQN initialisation
    q1_tgt = deepcopy(q1); q1_tgt.eval()
    q2_tgt = deepcopy(q2); q2_tgt.eval()

    print("[IQ-Load]  Q1, Q2 and targets initialised from IQ-Learn weights ✓")

    return q1, q2, q1_tgt, q2_tgt, obs_dim, hidden


# ──────────────────────────────────────────────────────────────────────────────
# Double-DQN Bellman loss (standard RL — no IQ terms, no expert data)
# ──────────────────────────────────────────────────────────────────────────────

def compute_dqn_loss(q1, q2, q1_tgt, q2_tgt,
                     obs, acts, nobs, rewards, dones,
                     gamma: float):
    """
    Standard Double-DQN loss.

    Action selection : online Q1 picks best action in s'
    Value estimation : min(Q1_tgt, Q2_tgt) evaluates that action
    Target           : y = r + γ · min_Q(s', a*)  · (1 - done)
    Loss             : MSE(Q1(s,a) - y) + MSE(Q2(s,a) - y)

    Double-Q prevents overestimation:
      - Using Q1 to SELECT the action decouples selection from evaluation
      - Using the target network to EVALUATE prevents moving-target instability
      - Taking the MIN of both targets further suppresses overestimation
    """
    batch = len(acts)

    with torch.no_grad():
        # Action selection: which action does online Q1 think is best in s'?
        next_acts = q1(nobs).argmax(dim=-1)                        # (B,)

        # Value evaluation: min of both TARGET networks at that action
        q1_next   = q1_tgt(nobs)[torch.arange(batch), next_acts]  # (B,)
        q2_next   = q2_tgt(nobs)[torch.arange(batch), next_acts]
        q_next    = torch.min(q1_next, q2_next)

        # Bellman target — zero future value at terminal states
        targets   = rewards.squeeze(-1) + gamma * q_next * (1.0 - dones.squeeze(-1))

    # Online Q-values at the actions actually taken
    q1_pred = q1(obs)[torch.arange(batch), acts]
    q2_pred = q2(obs)[torch.arange(batch), acts]

    loss_q1 = F.mse_loss(q1_pred, targets)
    loss_q2 = F.mse_loss(q2_pred, targets)
    loss    = loss_q1 + loss_q2

    return loss, {
        "dqn_loss": loss.item(),
        "q1_mean":  q1_pred.mean().item(),
        "q2_mean":  q2_pred.mean().item(),
        "td_error": (q1_pred - targets).abs().mean().item(),
        "reward":   rewards.mean().item(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Hard target sync
# ──────────────────────────────────────────────────────────────────────────────

def hard_update(src: nn.Module, tgt: nn.Module):
    """Copy all weights from src to tgt — standard DQN target sync."""
    tgt.load_state_dict(src.state_dict())


# ──────────────────────────────────────────────────────────────────────────────
# Epsilon schedule
# ──────────────────────────────────────────────────────────────────────────────

def get_eps(t: int, cfg: dict) -> float:
    """Linear decay from eps_start to eps_end over eps_decay_steps."""
    frac = min(t / cfg["eps_decay_steps"], 1.0)
    return cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])


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

def train_dqn_finetuned():
    print("=" * 60)
    print("IQ-Learn → DQN Fine-tuning  –  Soccer Striker  (CPU)")
    print(f"IQ checkpoint : {CFG['iq_checkpoint']}")
    print(f"Epsilon       : {CFG['eps_start']} → {CFG['eps_end']} "
          f"over {CFG['eps_decay_steps']:,} steps")
    print("=" * 60)

    device = torch.device("cpu")

    # ── environment ──────────────────────────────────────────────────
    from my_envs.striker_env import StrikerRLEnv
    env     = StrikerRLEnv(discrete=True)
    obs_dim = env.observation_space.shape[0]

    # ── load IQ-Learn weights ─────────────────────────────────────────
    q1, q2, q1_tgt, q2_tgt, iq_obs_dim, hidden = load_iq_checkpoint(
        CFG["iq_checkpoint"], device
    )

    if iq_obs_dim != obs_dim:
        raise ValueError(
            f"Observation dimension mismatch: IQ-Learn was trained on "
            f"obs_dim={iq_obs_dim} but current env has obs_dim={obs_dim}. "
            f"Retrain IQ-Learn with the current striker_env.py first."
        )

    # Single optimiser for both networks (same as IQ-Learn)
    q_opt = torch.optim.Adam(
        list(q1.parameters()) + list(q2.parameters()),
        lr=CFG["lr"]
    )

    # ── replay buffer ─────────────────────────────────────────────────
    buf = ReplayBuffer(CFG["buffer_size"], obs_dim)

    # ── logging ──────────────────────────────────────────────────────
    log_keys = ("dqn_loss", "q1_mean", "td_error", "reward")
    logs     = {k: deque(maxlen=CFG["log_window"]) for k in log_keys}
    best_goals          = 0
    best_reward_at_best = -float("inf")

    total   = CFG["total_timesteps"]
    bs      = CFG["batch_size"]
    warmup  = CFG["warmup_steps"]
    gamma   = CFG["gamma"]

    obs, _  = env.reset()
    t_start = time.time()

    print(f"\n[Phase 1]  Warming buffer with {warmup} steps "
          f"(low-ε policy, not random)...")

    for t in range(1, total + 1):

        # ── 1. ε-greedy action ────────────────────────────────────────
        eps = get_eps(t, CFG)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if np.random.rand() < eps:
                act_idx = np.random.randint(N_ACTIONS)
            else:
                # Greedy: argmax over min(Q1, Q2)
                act_idx = torch.min(q1(obs_t), q2(obs_t)).argmax(dim=-1).item()

        # ── 2. step environment — collect REAL reward ─────────────────
        next_obs, reward, terminated, truncated, _ = env.step(act_idx)
        done = terminated or truncated

        buf.add(obs, act_idx, next_obs, reward, terminated)
        obs = next_obs if not done else env.reset()[0]

        if t == warmup:
            print(f"[Phase 2]  Buffer warmed ({buf.size} transitions). "
                  f"Starting DQN updates...")

        if t < warmup or buf.size < bs:
            continue

        # ── 3. Double-DQN update ──────────────────────────────────────
        obs_b, acts_b, nobs_b, rew_b, done_b = buf.sample(bs, device)

        loss, info = compute_dqn_loss(
            q1, q2, q1_tgt, q2_tgt,
            obs_b, acts_b, nobs_b, rew_b, done_b,
            gamma,
        )

        q_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(q1.parameters()) + list(q2.parameters()),
            CFG["grad_clip"]
        )
        q_opt.step()

        for k, v in info.items():
            if k in logs: logs[k].append(v)

        # ── 4. Hard target sync ────────────────────────────────────────
        if t % CFG["target_update_every"] == 0:
            hard_update(q1, q1_tgt)
            hard_update(q2, q2_tgt)

        # ── 5. Eval & best-model saving ───────────────────────────────
        if t % CFG["eval_interval"] == 0:
            means = {k: np.mean(v) for k, v in logs.items() if v}

            # Run eval episodes — greedy policy, no exploration
            eval_goals   = 0
            eval_rewards = []

            for _ in range(CFG["num_eval_episodes"]):
                eval_obs, _ = env.reset()
                ep_reward   = 0.0
                ep_done     = False

                while not ep_done:
                    with torch.no_grad():
                        eval_t   = torch.as_tensor(
                            eval_obs, dtype=torch.float32
                        ).unsqueeze(0)
                        eval_act = torch.min(
                            q1(eval_t), q2(eval_t)
                        ).argmax(dim=-1).item()

                    eval_obs, r, term, trunc, _ = env.step(eval_act)
                    ep_reward += r
                    ep_done    = term or trunc

                    if term:
                        ball_pos = env.ball_node.getPosition()
                        dist     = np.hypot(
                            env.MY_GOAL_CENTER[0] - ball_pos[0],
                            env.MY_GOAL_CENTER[1] - ball_pos[1],
                        )
                        if dist < 0.25:
                            eval_goals += 1

                eval_rewards.append(ep_reward)

            avg_reward = float(np.mean(eval_rewards))

            print(
                f"[{t:>7,}/{total:,}]  "
                f"fps={t/(time.time()-t_start):.0f}  "
                f"ε={eps:.3f}  "
                f"loss={means.get('dqn_loss', 0):.4f}  "
                f"Q={means.get('q1_mean', 0):+.2f}  "
                f"td={means.get('td_error', 0):.4f}  "
                f"r̄={means.get('reward', 0):+.3f}  "
                f"| EVAL goals={eval_goals}/{CFG['num_eval_episodes']}  "
                f"reward={avg_reward:.1f}  "
                f"| best_goals={best_goals}"
            )

            # Save on goal count first, reward as tiebreaker
            if (eval_goals > best_goals or
                    (eval_goals == best_goals and avg_reward > best_reward_at_best)):
                best_goals          = eval_goals
                best_reward_at_best = avg_reward
                torch.save(
                    make_checkpoint(q1, q2, obs_dim, hidden, CFG),
                    CFG["save_path_best"]
                )
                print(f"   >>> NEW BEST: {eval_goals} goals  "
                      f"reward={avg_reward:.1f} <<<")

    # ── final save ────────────────────────────────────────────────────
    torch.save(
        make_checkpoint(q1, q2, obs_dim, hidden, CFG),
        CFG["save_path"]
    )
    print(f"\n--- SAVED last  → {CFG['save_path']} ---")
    print(f"--- SAVED best  → {CFG['save_path_best']} ---")
    print(f"Final best: {best_goals} goals, reward={best_reward_at_best:.1f}")
    env.close()


if __name__ == "__main__":
    train_dqn_finetuned()