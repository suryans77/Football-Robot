"""
PPO evaluation — 100 episodes with final metrics
=================================================

Runs the trained PPO policy for exactly 100 episodes then prints
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
from stable_baselines3 import PPO
from my_envs.striker_env import StrikerRLEnv

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

CKPT_PATH   = "ppo_striker_brain_best.zip"
N_EPISODES  = 100
GOAL_THRESH = 0.25   # must match striker_env — dist_ball_to_goal < this = goal


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate():
    print("=" * 60)
    print(f"PPO Evaluation  —  {N_EPISODES} episodes")
    print(f"Checkpoint: {CKPT_PATH}")
    print("=" * 60)

    env   = StrikerRLEnv(discrete=False)
    model = PPO.load(CKPT_PATH, device="cpu")

    # Per-episode trackers
    ep_rewards          = []
    ep_steps            = []
    ep_outcomes         = []    # "goal" | "fail" | "timeout"
    ep_defenders_beaten = []

    for ep in range(1, N_EPISODES + 1):
        obs, _         = env.reset()
        ep_reward      = 0.0
        ep_step        = 0
        defenders_beat = 0

        # Track how many defenders were beaten this episode by watching
        # the beaten_defenders set on the env directly
        prev_beaten = len(env.beaten_defenders)

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_step   += 1

            # Count new defenders beaten this step
            new_beaten      = len(env.beaten_defenders) - prev_beaten
            defenders_beat += max(new_beaten, 0)
            prev_beaten     = len(env.beaten_defenders)

            if terminated or truncated:
                # Determine outcome from final state
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

        ep_rewards.append(ep_reward)
        ep_steps.append(ep_step)
        ep_outcomes.append(outcome)
        ep_defenders_beaten.append(defenders_beat)

        # Per-episode line
        print(
            f"  ep {ep:>3}/{N_EPISODES}  "
            f"reward={ep_reward:>7.2f}  "
            f"steps={ep_step:>3}  "
            f"beaten={defenders_beat}  "
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

    print()
    print("=" * 60)
    print("PPO  —  FINAL EVALUATION METRICS  (100 episodes)")
    print("=" * 60)
    print(f"  Success rate       : {success_rate:>6.1f}%   ({n_goal}/{N_EPISODES} goals)")
    print(f"  Failure rate       : {fail_rate:>6.1f}%   ({n_fail}/{N_EPISODES})")
    print(f"  Timeout rate       : {timeout_rate:>6.1f}%   ({n_timeout}/{N_EPISODES})")
    print(f"  Avg episode reward : {avg_reward:>7.2f}  ± {std_reward:.2f}")
    print(f"  Avg episode steps  : {avg_steps:>7.1f}")
    print(f"  Avg defenders beat : {avg_beaten:>7.2f}")
    print("=" * 60)

    # Machine-readable summary for copy-pasting into a results table
    print()
    print("CSV row (algorithm, success%, avg_reward, avg_steps, avg_beaten):")
    print(
        f"PPO,"
        f"{success_rate:.1f},"
        f"{avg_reward:.2f},"
        f"{avg_steps:.1f},"
        f"{avg_beaten:.2f}"
    )


if __name__ == "__main__":
    evaluate()