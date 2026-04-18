"""
DQN evaluation — 100 episodes with final metrics
=================================================

Runs the trained DQN policy for exactly 100 episodes then prints
a summary table for comparison against other algorithms.

Metrics reported
-----------------
  Success rate     — % of episodes where ball reached the goal
  Failure rate     — % terminated by collision / out of bounds
  Timeout rate     — % that hit the 200-step limit without outcome
  Avg total reward — mean episode return across all 100 episodes
  Avg steps        — mean episode length (lower = more efficient)
  Defenders beaten — mean defenders passed per episode
"""

import numpy as np
from stable_baselines3 import DQN
from my_envs.striker_env import StrikerRLEnv, ACTION_SET

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

CKPT_PATH   = "dqn_striker_brain_best.zip"
N_EPISODES  = 100
GOAL_THRESH = 0.25   # must match striker_env — dist_ball_to_goal < this = goal


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate():
    print("=" * 60)
    print(f"DQN Evaluation  —  {N_EPISODES} episodes")
    print(f"Checkpoint: {CKPT_PATH}")
    print("=" * 60)

    # discrete=True so the env accepts integer action indices
    env   = StrikerRLEnv(discrete=True)
    model = DQN.load(CKPT_PATH, device="cpu")

    # Per-episode trackers
    ep_rewards          = []
    ep_steps            = []
    ep_outcomes         = []    # "goal" | "fail" | "timeout"
    ep_defenders_beaten = []
    ep_actions          = []    # track action distribution across all episodes

    for ep in range(1, N_EPISODES + 1):
        obs, _         = env.reset()
        ep_reward      = 0.0
        ep_step        = 0
        defenders_beat = 0
        action_counts  = [0] * len(ACTION_SET)

        prev_beaten = len(env.beaten_defenders)

        while True:
            # deterministic=True → pure greedy argmax Q, no ε-greedy
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action)
            action_counts[action_int] += 1

            obs, reward, terminated, truncated, info = env.step(action_int)

            ep_reward += reward
            ep_step   += 1

            new_beaten      = len(env.beaten_defenders) - prev_beaten
            defenders_beat += max(new_beaten, 0)
            prev_beaten     = len(env.beaten_defenders)

            if terminated or truncated:
                import numpy as _np
                ball_pos = env.ball_node.getPosition()
                dist_to_goal = _np.hypot(
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

        # Most-used action this episode
        dominant_idx    = int(np.argmax(action_counts))
        dominant_action = ACTION_SET[dominant_idx]

        ep_rewards.append(ep_reward)
        ep_steps.append(ep_step)
        ep_outcomes.append(outcome)
        ep_defenders_beaten.append(defenders_beat)
        ep_actions.append(action_counts)

        print(
            f"  ep {ep:>3}/{N_EPISODES}  "
            f"reward={ep_reward:>7.2f}  "
            f"steps={ep_step:>3}  "
            f"beaten={defenders_beat}  "
            f"dominant_action={dominant_action}  "
            f"[{outcome.upper()}]"
        )

    env.close()

    # ── final metrics ────────────────────────────────────────────────────────
    n_goal    = ep_outcomes.count("goal")
    n_fail    = ep_outcomes.count("fail")
    n_timeout = ep_outcomes.count("timeout")

    success_rate  = 100.0 * n_goal    / N_EPISODES
    fail_rate     = 100.0 * n_fail    / N_EPISODES
    timeout_rate  = 100.0 * n_timeout / N_EPISODES
    avg_reward    = np.mean(ep_rewards)
    std_reward    = np.std(ep_rewards)
    avg_steps     = np.mean(ep_steps)
    avg_beaten    = np.mean(ep_defenders_beaten)

    # Action distribution across all episodes
    total_action_counts = np.sum(ep_actions, axis=0)
    total_steps_all     = total_action_counts.sum()

    print()
    print("=" * 60)
    print("DQN  —  FINAL EVALUATION METRICS  (100 episodes)")
    print("=" * 60)
    print(f"  Success rate       : {success_rate:>6.1f}%   ({n_goal}/{N_EPISODES} goals)")
    print(f"  Failure rate       : {fail_rate:>6.1f}%   ({n_fail}/{N_EPISODES})")
    print(f"  Timeout rate       : {timeout_rate:>6.1f}%   ({n_timeout}/{N_EPISODES})")
    print(f"  Avg episode reward : {avg_reward:>7.2f}  ± {std_reward:.2f}")
    print(f"  Avg episode steps  : {avg_steps:>7.1f}")
    print(f"  Avg defenders beat : {avg_beaten:>7.2f}")
    print()
    print("  Action distribution across all episodes:")
    for idx, (action_vec, count) in enumerate(zip(ACTION_SET, total_action_counts)):
        pct = 100.0 * count / total_steps_all if total_steps_all > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    idx {idx}  {action_vec}  {pct:>5.1f}%  {bar}")
    print("=" * 60)

    # Machine-readable summary
    print()
    print("CSV row (algorithm, success%, avg_reward, avg_steps, avg_beaten):")
    print(
        f"DQN,"
        f"{success_rate:.1f},"
        f"{avg_reward:.2f},"
        f"{avg_steps:.1f},"
        f"{avg_beaten:.2f}"
    )


if __name__ == "__main__":
    evaluate()