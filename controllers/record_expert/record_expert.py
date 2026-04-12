import pickle
import sys
import numpy as np
from controller import Keyboard
from striker_env import StrikerRLEnv

def record_data():
    env = StrikerRLEnv()
    
    keyboard = Keyboard()
    keyboard.enable(env.timestep)
    
    expert_data = []
    current_episode = []
    
    # --- NEW: Load existing data so you don't overwrite previous sessions ---
    try:
        with open("expert_data.pkl", "rb") as f:
            expert_data = pickle.load(f)
        print(f"--- Resuming Session: Loaded {len(expert_data)} existing frames ---")
    except FileNotFoundError:
        print("--- Starting Fresh Session: No existing data found ---")
    # ------------------------------------------------------------------------

    print("--- IQ-LEARN DATA COLLECTION ---")
    print("Drive with W/A/S/D.")
    print("[P]     : Goal! Save episode.")
    print("[R]     : Messed up. Discard episode.")
    print("[Q]     : Quit and save data to disk.")
    
    obs, _ = env.reset()
    
    while True:
        action = np.zeros(2, dtype=np.float32)
        key = keyboard.getKey()
        
        process_action = True
        
        while key != -1:
            if key == ord('W'): action[0] = 1.0
            elif key == ord('S'): action[0] = -1.0
            
            if key == ord('A'): action[1] = 1.0
            elif key == ord('D'): action[1] = -1.0
            
            if key == ord('P'): 
                # --- NEW: Flag the final frame as the true end of the episode ---
                if len(current_episode) > 0:
                    current_episode[-1]["done"] = True 
                # ----------------------------------------------------------------
                
                print(f"--> Saved episode! (Clip length: {len(current_episode)} frames)")
                expert_data.extend(current_episode)
                current_episode = []
                obs, _ = env.reset()
                process_action = False
                break
                
            if key == ord('R'):
                print("--> Discarded episode.")
                current_episode = []
                obs, _ = env.reset()
                process_action = False
                break
                
            if key == ord('Q'):
                with open("expert_data.pkl", "wb") as f:
                    pickle.dump(expert_data, f)
                print(f"SUCCESS: Saved {len(expert_data)} total transitions to expert_data.pkl")
                sys.exit(0)
                
            key = keyboard.getKey()

        if process_action:
            # --- NEW: Every normal running frame defaults to done=False ---
            current_episode.append({"obs": obs, "action": action, "done": False})
            # --------------------------------------------------------------
            
            # Step the environment (This applies the action for 8 frames)
            next_obs, rew, done, truncated, info = env.step(action)
            obs = next_obs
            
            if done or truncated:
                print("--> Environment ended naturally. Discarding to ensure perfect data quality.")
                current_episode = []
                obs, _ = env.reset()

if __name__ == "__main__":
    record_data()