"""
DQN training for the soccer striker robot
==========================================

Uses Stable-Baselines3 DQN with the discrete Discrete(9) action space.
Requires StrikerRLEnv(discrete=True) — the env resolves integer indices
to [gas, steer] vectors internally via ACTION_SET.

Key DQN differences vs PPO
----------------------------
- Off-policy: learns from a replay buffer, not freshly collected rollouts
- Exploration: ε-greedy schedule (starts at 1.0, decays to 0.05)
- Target network: separate frozen Q-net, synced every N steps
- No actor/critic split: one Q-network Q(s) → R^9

Dependencies
------------
    pip install stable-baselines3 torch numpy gymnasium
"""

import numpy as np
from collections import deque

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from my_envs.striker_env import StrikerRLEnv

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

CFG = dict(
    save_path       = "dqn_striker_brain.zip",
    save_path_best  = "dqn_striker_brain_best.zip",
    total_timesteps = 100_000,

    # DQN hyperparameters
    learning_rate           = 3e-4,
    buffer_size             = 50_000,
    learning_starts         = 1_000,   # steps of random play before training starts
    batch_size              = 64,
    gamma                   = 0.99,
    train_freq              = 1,       # update Q-net every step
    gradient_steps          = 1,
    target_update_interval  = 200,     # steps between target Q-net syncs

    # ε-greedy exploration schedule
    exploration_initial_eps = 1.0,
    exploration_final_eps   = 0.05,
    exploration_fraction    = 0.3,     # fraction of training spent decaying ε

    # Network
    hidden          = [128, 128],

    # Best-model tracking
    reward_window   = 20,
)

# ──────────────────────────────────────────────────────────────────────────────
# Callback
# ──────────────────────────────────────────────────────────────────────────────

class BestModelCallback(BaseCallback):
    """
    Same logic as PPO callback — saves on new best mean episode reward.
    Also prints ε (exploration rate) to track the decay schedule.
    """

    def __init__(self, save_path: str, reward_window: int = 20, verbose: int = 1):
        super().__init__(verbose)
        self.save_path     = save_path
        self.reward_window = reward_window
        self.ep_rewards    = deque(maxlen=reward_window)
        self.best_mean_rew = -float("inf")
        self.n_episodes    = 0
        self.log_interval  = 5_000     # print every N steps

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.n_episodes += 1

                if len(self.ep_rewards) >= self.reward_window:
                    mean_rew = np.mean(self.ep_rewards)
                    if mean_rew > self.best_mean_rew:
                        self.best_mean_rew = mean_rew
                        self.model.save(self.save_path)
                        if self.verbose:
                            print(
                                f"   >>> NEW BEST DQN MODEL  "
                                f"mean_rew={mean_rew:.2f}  "
                                f"(ep {self.n_episodes}) <<<"
                            )

        if self.verbose and self.num_timesteps % self.log_interval == 0:
            eps   = self.model.exploration_rate
            m_rew = np.mean(self.ep_rewards) if self.ep_rewards else float("nan")
            print(
                f"[DQN {self.num_timesteps:>7,}]  "
                f"ε={eps:.3f}  "
                f"mean_rew={m_rew:.2f}  "
                f"best={self.best_mean_rew:.2f}  "
                f"episodes={self.n_episodes}"
            )
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_dqn():
    print("=" * 60)
    print("DQN  –  Soccer Striker  (CPU, discrete action space)")
    print("=" * 60)

    # discrete=True — env maps integer actions through ACTION_SET internally
    env = DummyVecEnv([lambda: StrikerRLEnv(discrete=True)])

    policy_kwargs = dict(
        net_arch      = CFG["hidden"],
        activation_fn = torch.nn.ReLU,
    )

    model = DQN(
        policy                  = "MlpPolicy",
        env                     = env,
        learning_rate           = CFG["learning_rate"],
        buffer_size             = CFG["buffer_size"],
        learning_starts         = CFG["learning_starts"],
        batch_size              = CFG["batch_size"],
        gamma                   = CFG["gamma"],
        train_freq              = CFG["train_freq"],
        gradient_steps          = CFG["gradient_steps"],
        target_update_interval  = CFG["target_update_interval"],
        exploration_initial_eps = CFG["exploration_initial_eps"],
        exploration_final_eps   = CFG["exploration_final_eps"],
        exploration_fraction    = CFG["exploration_fraction"],
        policy_kwargs           = policy_kwargs,
        optimize_memory_usage   = False,
        verbose                 = 0,
        device                  = "cpu",
    )

    callback = BestModelCallback(
        save_path     = CFG["save_path_best"],
        reward_window = CFG["reward_window"],
        verbose       = 1,
    )

    print(f"Training for {CFG['total_timesteps']:,} timesteps...")
    print(f"Warmup: first {CFG['learning_starts']:,} steps are random exploration")
    model.learn(
        total_timesteps = CFG["total_timesteps"],
        callback        = callback,
        progress_bar    = False,
    )

    model.save(CFG["save_path"])
    print(f"\n--- SAVED last  → {CFG['save_path']} ---")
    print(f"--- SAVED best  → {CFG['save_path_best']} ---")
    print("Use dqn_striker_brain_best.zip for play_dqn.py")
    env.close()


if __name__ == "__main__":
    train_dqn()