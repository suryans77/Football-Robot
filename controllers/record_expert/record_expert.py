"""
Expert data recorder for IL methods (BC, GAIL, IQL, IQ-Learn, SQIL)
=====================================================================

Mirroring fix
--------------
The previous version interleaved original and mirrored transitions:
    [orig_0, mirror_0, orig_1, mirror_1, ...]

This is wrong. Every IL loader builds next_obs by shifting the array
by one index:
    next_obs[i] = obs[i+1]

With interleaving, transition orig_0 gets next_obs = obs[mirror_0],
which is the laterally flipped version of the same state — a physically
impossible teleport. The network learns a false transition model.

The fix is to keep all originals as one contiguous block followed by
all mirrors as a separate contiguous block:
    [orig_0, orig_1, ..., orig_N-1, mirror_0, mirror_1, ..., mirror_N-1]

Each block's episode boundaries (done=True flags) are preserved, so
the shift-by-one next_obs construction stays correct within each block.

Other features
---------------
  Resume        — loads existing expert_data.pkl on startup
  Auto-save     — writes to expert_data_autosave.pkl after every P press
  Emergency save — atexit fires even on crash, saves current data
  Stats         — prints action distribution and steer bias before/after

Controls
---------
  W / S  — forward / backward
  A / D  — turn left / right
  P      — accept episode
  R      — discard episode
  Q      — quit, mirror, save final expert_data.pkl

Observation layout (must match striker_env._get_obs)
------------------------------------------------------
  [0]  goal_dist
  [1]  sin(goal_angle)   ← negated on mirror
  [2]  cos(goal_angle)
  [3]  self_vx
  [4]  self_vy           ← negated on mirror
  [5]  d1_dist
  [6]  d1_sin            ← negated on mirror
  [7]  d1_cos
  [8]  d1_rel_vx
  [9]  d1_rel_vy         ← negated on mirror
  [10] d2_dist
  [11] d2_sin            ← negated on mirror
  [12] d2_cos
  [13] d2_rel_vx
  [14] d2_rel_vy         ← negated on mirror
"""

import atexit
import pickle
import sys
import numpy as np
from controller import Keyboard
from my_envs.striker_env import StrikerRLEnv

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

SAVE_PATH     = "full_data.pkl"
AUTOSAVE_PATH = "full_data_autosave.pkl"

# ──────────────────────────────────────────────────────────────────────────────
# Obs indices that get sign-flipped when mirroring across the x-axis
# ──────────────────────────────────────────────────────────────────────────────

MIRROR_OBS_INDICES = [1, 4, 6, 9, 11, 14]


# ──────────────────────────────────────────────────────────────────────────────
# Mirroring — FIX: append blocks, do NOT interleave
# ──────────────────────────────────────────────────────────────────────────────

def mirror_transition(t: dict) -> dict:
    """Return a new transition with the y-axis (left/right) flipped."""
    obs_m    = t["obs"].copy()
    action_m = t["action"].copy()
    for i in MIRROR_OBS_INDICES:
        obs_m[i] = -obs_m[i]
    action_m[1] = -action_m[1]   # negate steer, gas unchanged
    return {"obs": obs_m, "action": action_m, "done": t["done"]}


def mirror_dataset(data: list[dict]) -> list[dict]:
    """
    Return [all originals] + [all mirrors] as two contiguous blocks.

    WHY NOT INTERLEAVED:
    Every IL loader builds next_obs via:
        next_obs[i] = obs[i+1]

    Interleaving gives orig_0 a next_obs from mirror_0, which is a
    physically impossible transition (teleport to mirror universe).
    Two contiguous blocks keep each episode's next_obs correct because
    the shift stays within the same block, and episode terminal frames
    (done=True) already point to themselves.
    """
    mirrored = [mirror_transition(t) for t in data]
    return list(data) + mirrored   # originals first, mirrors second


# ──────────────────────────────────────────────────────────────────────────────
# Dataset stats
# ──────────────────────────────────────────────────────────────────────────────

ACTION_SET = np.array([
    [-1., -1.], [-1.,  0.], [-1., +1.],
    [ 0., -1.], [ 0.,  0.], [ 0., +1.],
    [+1., -1.], [+1.,  0.], [+1., +1.],
])


def print_stats(data: list[dict], label: str = "Dataset"):
    if not data:
        print(f"[{label}]  empty")
        return

    n          = len(data)
    n_episodes = sum(1 for t in data if t.get("done", False))
    actions    = np.array([t["action"] for t in data])
    steer_mean = actions[:, 1].mean()

    counts = np.zeros(len(ACTION_SET), dtype=int)
    for t in data:
        dists = np.linalg.norm(ACTION_SET - t["action"], axis=1)
        counts[np.argmin(dists)] += 1

    print(f"[{label}]  {n} transitions  {n_episodes} episodes  "
          f"avg_steer={steer_mean:+.3f} (0=no bias)")
    for idx, (av, c) in enumerate(zip(ACTION_SET, counts)):
        pct = 100.0 * c / n
        bar = "█" * int(pct / 2)
        print(f"  idx {idx}  {av}  {pct:>5.1f}%  {bar}")


# ──────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> list[dict]:
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"[Load]  {len(data)} transitions from '{path}'")
        return data
    except FileNotFoundError:
        return []


def save_data(data: list[dict], path: str, label: str = ""):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    tag = f" ({label})" if label else ""
    print(f"[Save{tag}]  {len(data)} transitions → '{path}'")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def record_data():
    env      = StrikerRLEnv()
    keyboard = Keyboard()
    keyboard.enable(env.timestep)

    # ── load existing data ────────────────────────────────────────────
    expert_data = load_data("full_data_raw.pkl")
    if not expert_data:
        autosave = load_data(AUTOSAVE_PATH)
        if autosave:
            print(f"[Resume]  Recovered {len(autosave)} transitions from autosave.")
            expert_data = autosave

    print_stats(expert_data, label="Existing data")

    # ── emergency save on unexpected exit ────────────────────────────
    def emergency_save():
        if expert_data:
            save_data(expert_data, AUTOSAVE_PATH, label="emergency")

    atexit.register(emergency_save)

    print()
    print("=" * 50)
    print("DATA COLLECTION")
    print("  W/S : forward / backward")
    print("  A/D : turn left / right")
    print("  P   : accept episode")
    print("  R   : discard episode")
    print("  Q   : quit, mirror, save")
    print("=" * 50)

    current_episode   = []
    session_accepted  = 0
    session_discarded = 0
    obs, _ = env.reset()

    while True:
        action         = np.zeros(2, dtype=np.float32)
        key            = keyboard.getKey()
        process_action = True

        while key != -1:

            if   key == ord('W'): action[0] =  1.0
            elif key == ord('S'): action[0] = -1.0
            if   key == ord('A'): action[1] =  1.0
            elif key == ord('D'): action[1] = -1.0

            # ── accept ────────────────────────────────────────────────
            if key == ord('P'):
                if len(current_episode) == 0:
                    print("[Warning]  No frames yet — drive first.")
                else:
                    current_episode[-1]["done"] = True
                    expert_data.extend(current_episode)
                    session_accepted += 1
                    save_data(expert_data, AUTOSAVE_PATH, label="autosave")
                    print(
                        f"--> Accepted  frames={len(current_episode)}  "
                        f"session={session_accepted}  "
                        f"total={len(expert_data)}"
                    )
                current_episode = []
                obs, _ = env.reset()
                process_action = False
                break

            # ── discard ───────────────────────────────────────────────
            if key == ord('R'):
                session_discarded += 1
                print(f"--> Discarded  frames={len(current_episode)}  "
                      f"discarded_this_session={session_discarded}")
                current_episode = []
                obs, _ = env.reset()
                process_action = False
                break

            # ── quit, mirror, save ────────────────────────────────────
            if key == ord('Q'):
                print()
                print("=" * 50)
                print(f"Session: accepted={session_accepted}  "
                      f"discarded={session_discarded}")
                print(f"Raw transitions: {len(expert_data)}")
                print()
                print_stats(expert_data, label="Raw (before mirror)")
                print()

                mirrored_data = mirror_dataset(expert_data)
                print_stats(mirrored_data, label="Final (after mirror)")
                print()

                # avg_steer should be exactly 0.0 after mirroring
                actions    = np.array([t["action"] for t in mirrored_data])
                steer_bias = actions[:, 1].mean()
                if abs(steer_bias) > 0.01:
                    print(f"[Warning]  Steer bias after mirror = {steer_bias:+.4f} "
                          f"(expected 0.0 — check MIRROR_OBS_INDICES)")

                save_data(mirrored_data, SAVE_PATH,          label="final+mirrored")
                save_data(expert_data,   "full_data_raw.pkl", label="raw backup")
                print("Use full_data.pkl for training.")
                sys.exit(0)

            key = keyboard.getKey()

        # ── record frame ──────────────────────────────────────────────
        if process_action:
            current_episode.append({
                "obs":    obs.copy(),
                "action": action.copy(),
                "done":   False,
            })

            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = next_obs

            # If we won (reward is high because of the +100 goal bonus)
            if terminated and reward > -200.0:
                current_episode[-1]["done"] = True
                expert_data.extend(current_episode)
                session_accepted += 1
                save_data(expert_data, AUTOSAVE_PATH, label="autosave")
                print(f"--> AUTO-ACCEPTED (Goal Reached!) frames={len(current_episode)} total={len(expert_data)}")
                current_episode = []
                obs, _ = env.reset()
            
            # If we failed (ran out of time or went out of bounds)
            elif terminated or truncated:
                print(f"--> Env ended in failure — discarding "
                      f"({len(current_episode)} frames).")
                current_episode = []
                obs, _ = env.reset()


if __name__ == "__main__":
    record_data()