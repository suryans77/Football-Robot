import pickle
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import iq_learn
from imitation.data.types import Transitions
from striker_env import StrikerRLEnv

def train_iq():
    print("--- INITIALIZING IQ-LEARN ---")
    
    # 1. Setup the Environment
    # The imitation library expects a Vectorized Environment
    env = DummyVecEnv([lambda: StrikerRLEnv()])

    # 2. Load the Expert Data
    try:
        with open("expert_data.pkl", "rb") as f:
            expert_data = pickle.load(f)
        print(f"Loaded {len(expert_data)} expert transitions.")
    except FileNotFoundError:
        print("ERROR: expert_data.pkl not found! Run 1_record_expert.py first.")
        return

    # Extract arrays
    expert_obs = np.array([d["obs"] for d in expert_data])
    expert_acts = np.array([d["action"] for d in expert_data])
    
    # Construct Next Observations
    next_obs_list = []
    for i in range(len(expert_data) - 1):
        next_obs_list.append(expert_data[i+1]["obs"])
    # Pad the final observation
    next_obs_list.append(expert_data[-1]["obs"])
    expert_next_obs = np.array(next_obs_list)
    
    expert_dones = np.zeros(len(expert_data), dtype=bool)

    # 3. Format for the Imitation Library
    expert_transitions = Transitions(
        obs=expert_obs,
        acts=expert_acts,
        next_obs=expert_next_obs,
        dones=expert_dones,
        infos=np.array([{}] * len(expert_data))
    )

    # 4. Initialize Soft Actor-Critic Brain
    # Note: SAC needs a slightly different learning rate than PPO.
    sac_brain = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4)

    # 5. Wrap with IQ-Learn
    iq_trainer = iq_learn.IQLearn(
        demonstrations=expert_transitions,
        venv=env,
        rl_algo=sac_brain,
    )

    # 6. Train and Save
    print("--- STARTING TRAINING ---")
    iq_trainer.train(total_timesteps=20000)
    
    sac_brain.save("iq_striker_brain")
    print("--- TRAINING COMPLETE AND SAVED ---")

if __name__ == "__main__":
    train_iq()