"""
Pure IQ-Learn evaluation — 100 episodes with final metrics
===========================================================

Loads iq_pure_brain_best.pt and runs the greedy policy
(argmax over min(Q1, Q2)) for exactly 100 episodes, then
prints a full metrics summary in the same format as
play_ppo.py and play_dqn.py for direct comparison.

Metrics reported
-----------------
  Success rate     — % of episodes where ball reached the goal
  Failure rate     — % terminated by collision / out of bounds
  Timeout rate     — % that hit the step limit
  Avg total reward — mean episode return ± std
  Avg episode steps
  Avg defenders beaten per episode
  Action distribution — which of the 9 actions was used most
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from my_envs.striker_env import StrikerRLEnv, ACTION_SET

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

CKPT_PATH   = "full_pureIQ_best.pt"
N_EPISODES  = 100
GOAL_THRESH = 0.25   # must match striker_env


# ──────────────────────────────────────────────────────────────────────────────
# Network — must match pure_iq.py exactly
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
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate():
    print("=" * 60)
    print(f"Pure IQ-Learn Evaluation  —  {N_EPISODES} episodes")
    print(f"Checkpoint: {CKPT_PATH}")
    print("=" * 60)

    # ── load checkpoint ───────────────────────────────────────────────
    ckpt      = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    obs_dim   = ckpt["obs_dim"]
    n_actions = ckpt["n_actions"]
    hidden    = ckpt["hidden"]

    q1 = DiscreteQNetwork(obs_dim, n_actions, hidden)
    q2 = DiscreteQNetwork(obs_dim, n_actions, hidden)
    q1.load_state_dict(ckpt["q1"])
    q2.load_state_dict(ckpt["q2"])
    q1.eval()
    q2.eval()

    print(f"[Model]  obs_dim={obs_dim}  n_actions={n_actions}  hidden={hidden}")

    # ── environment ───────────────────────────────────────────────────
    # discrete=True so the env accepts integer action indices directly,
    # consistent with how actions are resolved during training
    env = StrikerRLEnv(discrete=True)

    # ── per-episode trackers ──────────────────────────────────────────
    ep_rewards          = []
    ep_steps            = []
    ep_outcomes         = []   # "goal" | "fail" | "timeout"
    ep_defenders_beaten = []
    ep_action_counts    = []

    for ep in range(1, N_EPISODES + 1):
        np.random.seed(1000 + ep)                   # Force Numpy's global seed
        obs, _         = env.reset(seed=1000 + ep)
        ep_reward      = 0.0
        ep_step        = 0
        defenders_beat = 0
        action_counts  = [0] * n_actions
        prev_beaten    = len(env.beaten_defenders)

        while True:
            with torch.no_grad():
                obs_t   = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                # Greedy: argmax over min(Q1, Q2) — same as training inference
                q_vals  = torch.min(q1(obs_t), q2(obs_t))
                act_idx = q_vals.argmax(dim=-1).item()

            action_counts[act_idx] += 1

            obs, reward, terminated, truncated, _ = env.step(act_idx)
            ep_reward += reward
            ep_step   += 1

            new_beaten      = len(env.beaten_defenders) - prev_beaten
            defenders_beat += max(new_beaten, 0)
            prev_beaten     = len(env.beaten_defenders)

            if terminated or truncated:
                ball_pos     = env.ball_node.getPosition()
                dist_to_goal = np.hypot(
                    env.MY_GOAL_CENTER[0] - ball_pos[0],
                    env.MY_GOAL_CENTER[1] - ball_pos[1],
                )
                if terminated and dist_to_goal < GOAL_THRESH:
                    outcome = "goal"
                elif truncated:
                    outcome = "timeout"
                else:
                    outcome = "fail"
                break

        dominant_idx    = int(np.argmax(action_counts))
        dominant_action = ACTION_SET[dominant_idx]

        ep_rewards.append(ep_reward)
        ep_steps.append(ep_step)
        ep_outcomes.append(outcome)
        ep_defenders_beaten.append(defenders_beat)
        ep_action_counts.append(action_counts)

        print(
            f"  ep {ep:>3}/{N_EPISODES}  "
            f"reward={ep_reward:>7.2f}  "
            f"steps={ep_step:>3}  "
            f"beaten={defenders_beat}  "
            f"dominant_action={dominant_action}  "
            f"[{outcome.upper()}]"
        )

    env.close()

    # ── aggregate metrics ────────────────────────────────────────────
    n_goal    = ep_outcomes.count("goal")
    n_fail    = ep_outcomes.count("fail")
    n_timeout = ep_outcomes.count("timeout")

    success_rate = 100.0 * n_goal    / N_EPISODES
    fail_rate    = 100.0 * n_fail    / N_EPISODES
    timeout_rate = 100.0 * n_timeout / N_EPISODES
    avg_reward   = np.mean(ep_rewards)
    std_reward   = np.std(ep_rewards)
    avg_steps    = np.mean(ep_steps)
    avg_beaten   = np.mean(ep_defenders_beaten)

    # Reward breakdown by outcome
    goal_rewards    = [r for r, o in zip(ep_rewards, ep_outcomes) if o == "goal"]
    fail_rewards    = [r for r, o in zip(ep_rewards, ep_outcomes) if o == "fail"]
    timeout_rewards = [r for r, o in zip(ep_rewards, ep_outcomes) if o == "timeout"]

    # Action distribution
    total_counts    = np.sum(ep_action_counts, axis=0)
    total_steps_all = total_counts.sum()

    print()
    print("=" * 60)
    print("Pure IQ-Learn  —  FINAL EVALUATION METRICS  (100 episodes)")
    print("=" * 60)
    print(f"  Success rate       : {success_rate:>6.1f}%   ({n_goal}/{N_EPISODES} goals)")
    print(f"  Failure rate       : {fail_rate:>6.1f}%   ({n_fail}/{N_EPISODES})")
    print(f"  Timeout rate       : {timeout_rate:>6.1f}%   ({n_timeout}/{N_EPISODES})")
    print()
    print(f"  Avg episode reward : {avg_reward:>7.2f}  ± {std_reward:.2f}")
    if goal_rewards:
        print(f"    └ goals only     : {np.mean(goal_rewards):>7.2f}  ± {np.std(goal_rewards):.2f}")
    if fail_rewards:
        print(f"    └ failures only  : {np.mean(fail_rewards):>7.2f}  ± {np.std(fail_rewards):.2f}")
    if timeout_rewards:
        print(f"    └ timeouts only  : {np.mean(timeout_rewards):>7.2f}  ± {np.std(timeout_rewards):.2f}")
    print()
    print(f"  Avg episode steps  : {avg_steps:>7.1f}")
    print(f"  Avg defenders beat : {avg_beaten:>7.2f}")
    print()
    print("  Action distribution across all episodes:")
    for idx, (action_vec, count) in enumerate(zip(ACTION_SET, total_counts)):
        pct = 100.0 * count / total_steps_all if total_steps_all > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    idx {idx}  {action_vec}  {pct:>5.1f}%  {bar}")
    print("=" * 60)

    # ── CSV row — same format as play_ppo.py and play_dqn.py ─────────
    print()
    print("CSV row (algorithm, success%, avg_reward, avg_steps, avg_beaten):")
    print(
        f"IQ-Learn (Pure),"
        f"{success_rate:.1f},"
        f"{avg_reward:.2f},"
        f"{avg_steps:.1f},"
        f"{avg_beaten:.2f}"
    )


if __name__ == "__main__":
    evaluate()