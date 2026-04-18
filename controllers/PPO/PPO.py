"""
PPO training for the soccer striker robot
==========================================

Uses Stable-Baselines3 PPO with the continuous Box action space.
A custom callback tracks mean episode reward and saves the best model,
since Webots only runs one physics instance (no separate eval env).

Dependencies
------------
    pip install stable-baselines3 torch numpy gymnasium
"""

import numpy as np
from collections import deque

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from my_envs.striker_env import StrikerRLEnv

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

CFG = dict(
    save_path       = "ppo_striker_brain.zip",
    save_path_best  = "ppo_striker_brain_best.zip",
    total_timesteps = 100_000,

    # PPO hyperparameters
    # n_steps: steps collected per update — must be >= batch_size
    # Small value (512) keeps CPU memory usage low
    n_steps         = 512,
    batch_size      = 64,
    n_epochs        = 10,        # gradient passes over each collected batch
    gamma           = 0.99,
    gae_lambda      = 0.95,      # GAE smoothing — 0.95 is standard
    ent_coef        = 0.01,      # entropy bonus — keeps policy exploratory
    vf_coef         = 0.5,       # value function loss weight
    max_grad_norm   = 0.5,
    learning_rate   = 3e-4,

    # Network
    hidden          = [128, 128],

    # Best-model tracking
    reward_window   = 20,        # average over last N episodes for best-model check
)

# ──────────────────────────────────────────────────────────────────────────────
# Callback — saves best model based on mean episode reward
# (Can't use SB3's EvalCallback because Webots has only one physics instance)
# ──────────────────────────────────────────────────────────────────────────────

class BestModelCallback(BaseCallback):
    """
    Tracks mean episode reward over a rolling window.
    Saves the model whenever a new best mean reward is achieved.
    Also prints a clean training summary at each PPO update.
    """

    def __init__(self, save_path: str, reward_window: int = 20, verbose: int = 1):
        super().__init__(verbose)
        self.save_path      = save_path
        self.reward_window  = reward_window
        self.ep_rewards     = deque(maxlen=reward_window)
        self.best_mean_rew  = -float("inf")
        self.ep_reward_buf  = 0.0
        self.n_episodes     = 0

    def _on_step(self) -> bool:
        # SB3 VecEnv stores episode info in infos when an episode ends
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_rew = info["episode"]["r"]
                self.ep_rewards.append(ep_rew)
                self.n_episodes += 1

                if len(self.ep_rewards) >= self.reward_window:
                    mean_rew = np.mean(self.ep_rewards)
                    if mean_rew > self.best_mean_rew:
                        self.best_mean_rew = mean_rew
                        self.model.save(self.save_path)
                        if self.verbose:
                            print(
                                f"   >>> NEW BEST PPO MODEL  "
                                f"mean_rew={mean_rew:.2f}  "
                                f"(ep {self.n_episodes}) <<<"
                            )
        return True

    def _on_rollout_end(self) -> None:
        if self.verbose and len(self.ep_rewards) > 0:
            print(
                f"[PPO {self.num_timesteps:>7,}]  "
                f"mean_rew={np.mean(self.ep_rewards):.2f}  "
                f"best={self.best_mean_rew:.2f}  "
                f"episodes={self.n_episodes}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_ppo():
    print("=" * 60)
    print("PPO  –  Soccer Striker  (CPU, continuous action space)")
    print("=" * 60)

    # SB3 requires a VecEnv — DummyVecEnv wraps a single Webots env safely
    env = DummyVecEnv([lambda: StrikerRLEnv(discrete=False)])

    policy_kwargs = dict(
        net_arch = CFG["hidden"],
        activation_fn = torch.nn.ReLU,
    )

    model = PPO(
        policy         = "MlpPolicy",
        env            = env,
        n_steps        = CFG["n_steps"],
        batch_size     = CFG["batch_size"],
        n_epochs       = CFG["n_epochs"],
        gamma          = CFG["gamma"],
        gae_lambda     = CFG["gae_lambda"],
        ent_coef       = CFG["ent_coef"],
        vf_coef        = CFG["vf_coef"],
        max_grad_norm  = CFG["max_grad_norm"],
        learning_rate  = CFG["learning_rate"],
        policy_kwargs  = policy_kwargs,
        verbose        = 0,     # silenced — callback handles printing
        device         = "cpu",
    )

    callback = BestModelCallback(
        save_path     = CFG["save_path_best"],
        reward_window = CFG["reward_window"],
        verbose       = 1,
    )

    print(f"Training for {CFG['total_timesteps']:,} timesteps...")
    model.learn(
        total_timesteps = CFG["total_timesteps"],
        callback        = callback,
        progress_bar    = False,
    )

    model.save(CFG["save_path"])
    print(f"\n--- SAVED last  → {CFG['save_path']} ---")
    print(f"--- SAVED best  → {CFG['save_path_best']} ---")
    print("Use ppo_striker_brain_best.zip for play_ppo.py")
    env.close()


if __name__ == "__main__":
    train_ppo()