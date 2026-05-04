import json
import matplotlib.pyplot as plt

with open("teleop_dataset.json", "r") as f:
    data = json.load(f)

steps = [d["step"] for d in data]

# Create a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# --- Graph 1: Attacker v (Forward Speed) ---
axs[0, 0].plot(steps, [d["target_v"] for d in data], 'b--', label="Command (v_target)")
axs[0, 0].plot(steps, [d["att_actual_v"] for d in data], 'g-', label="Physics Response (v_actual)")
axs[0, 0].set_title("Attacker Forward Speed (v)")
axs[0, 0].set_ylabel("Wheel Speed (rad/s)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# --- Graph 2: Attacker w (Differential Speed) ---
axs[0, 1].plot(steps, [d["target_w"] for d in data], 'r--', label="Command (w_target)")
axs[0, 1].plot(steps, [d["att_actual_w"] for d in data], 'orange', label="Physics Response (w_actual)")
axs[0, 1].set_title("Attacker Differential Turn Speed (w)")
axs[0, 1].set_ylabel("Wheel Offset (rad/s)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# --- Graph 3: Defender v (Forward Speed) ---
axs[1, 0].plot(steps, [d["def_actual_v"] for d in data], 'purple', linewidth=2)
axs[1, 0].set_title("Defender 1 Forward Speed (v)")
axs[1, 0].set_xlabel("Simulation Steps")
axs[1, 0].set_ylabel("Wheel Speed (rad/s)")
axs[1, 0].grid(True)

# --- Graph 4: Defender w (Differential Speed) ---
axs[1, 1].plot(steps, [d["def_actual_w"] for d in data], 'brown', linewidth=2)
axs[1, 1].set_title("Defender 1 Differential Turn Speed (w)")
axs[1, 1].set_xlabel("Simulation Steps")
axs[1, 1].set_ylabel("Wheel Offset (rad/s)")
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()