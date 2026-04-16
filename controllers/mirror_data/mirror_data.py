import pickle
import numpy as np

def mirror_dataset(input_file="expert_data.pkl", output_file="expert_data_mirrored.pkl"):
    print(f"--- Loading {input_file} ---")
    try:
        with open(input_file, "rb") as f:
            original_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}!")
        return

    print(f"Original dataset size: {len(original_data)} frames")

    mirrored_data = []

    for step in original_data:
        # 1. Copy the arrays so we don't accidentally modify the originals
        obs = step["obs"].copy()
        action = step["action"].copy()
        
        # 2. MIRROR THE OBSERVATION (Flip the Y-axis / Lateral variables)
        # Based on your _get_obs() layout:
        # [1] np.sin(goal_angle)
        # [4] self_vy
        # [6] d1["sin"]
        # [9] d1["rel_vy"]
        # [11] d2["sin"]
        # [14] d2["rel_vy"]
        y_axis_indices = [1, 4, 6, 9, 11, 14]
        obs[y_axis_indices] *= -1.0
        
        # 3. MIRROR THE ACTION (Flip the steering wheel)
        # action[0] is Gas Pedal (Leave alone)
        # action[1] is Steering  (Multiply by -1)
        action[1] *= -1.0
        
        # 4. Append to the new mirrored list
        mirrored_data.append({
            "obs": obs,
            "action": action,
            "done": step["done"]
        })

    # Combine the original data and the mirrored data
    augmented_dataset = original_data + mirrored_data

    # Save to a new file so you don't overwrite your raw original data
    with open(output_file, "wb") as f:
        pickle.dump(augmented_dataset, f)

    print(f"--- SUCCESS ---")
    print(f"Mirrored data created!")
    print(f"New augmented dataset size: {len(augmented_dataset)} frames")
    print(f"Saved as: {output_file}")

if __name__ == "__main__":
    mirror_dataset()