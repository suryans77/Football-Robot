from controller import Supervisor
import torch
import torch.nn as nn
import numpy as np
import os

# ==========================================
# 1. DEFINE THE BRAIN (Must match training exactly)
# ==========================================
class BC_Policy(nn.Module):
    def __init__(self):
        super(BC_Policy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.network(x)

# ================= INITIALIZATION =================
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Setup Motors
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Setup Sensors 
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Supervisor Handles
defenders = []
for i in range(1, 4):
    node = robot.getFromDef(f"defender{i}")
    if node: defenders.append(node)

ball_node = robot.getFromDef("Ball")

# Parameters
MY_GOAL_CENTER = [2.0, 0.0]
SHOOT_POWER = 3.0  
DRIBBLER_OFFSET = 0.05  
DRIBBLER_PULL = 15.0    
v_curr = 0.0 # Used for velocity estimation

# ==========================================
# 2. LOAD THE TRAINED WEIGHTS
# ==========================================
model = BC_Policy()
weight_path = "C:\\Users\\dell\\Desktop\\robotics\\bc_policy.pth"

if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path))
    model.eval() # Set the model to evaluation mode (critical!)
    print("SUCCESS: AI Brain Loaded. The robot is now autonomous.")
else:
    print(f"ERROR: Could not find {weight_path}. Check your folder!")

print("--- AUTONOMOUS INFERENCE STARTED ---")

# ================= MAIN LOOP =================
while robot.step(timestep) != -1:
    
    # Get Self State
    pos = gps.getValues()
    comp = compass.getValues()
    heading = np.arctan2(comp[0], comp[1])
    
    # Estimate physical self-velocity for the ML inputs
    est_linear_v = v_curr * 0.02 
    self_vx = est_linear_v * np.cos(heading)
    self_vy = est_linear_v * np.sin(heading)

    # 1. Goal Calculations 
    goal_dx = MY_GOAL_CENTER[0] - pos[0]
    goal_dy = MY_GOAL_CENTER[1] - pos[1]
    goal_dist = np.hypot(goal_dx, goal_dy)
    goal_angle = np.arctan2(goal_dy, goal_dx) - heading
    
    # 2. Defender Calculations
    def_stats = []
    for d in defenders:
        d_pos = d.getPosition()
        d_vel = d.getVelocity() 
        
        d_dx = d_pos[0] - pos[0]
        d_dy = d_pos[1] - pos[1]
        d_dist = np.hypot(d_dx, d_dy)
        d_angle = np.arctan2(d_dy, d_dx) - heading
        
        rel_vx = d_vel[0] - self_vx
        rel_vy = d_vel[1] - self_vy
        
        def_stats.append({
            "dist": d_dist, "sin": np.sin(d_angle), "cos": np.cos(d_angle),
            "rel_vx": rel_vx, "rel_vy": rel_vy
        })
        
    def_stats.sort(key=lambda x: x["dist"])
    d1 = def_stats[0] if len(def_stats) > 0 else {"dist": 10.0, "sin": 0.0, "cos": 0.0, "rel_vx": 0.0, "rel_vy": 0.0}
    d2 = def_stats[1] if len(def_stats) > 1 else {"dist": 10.0, "sin": 0.0, "cos": 0.0, "rel_vx": 0.0, "rel_vy": 0.0}

    # ==========================================
    # 3. THE NEURAL NETWORK FORWARD PASS
    # ==========================================
    # Pack the 15 inputs exactly how they were trained
    state_array = [
        goal_dist, np.sin(goal_angle), np.cos(goal_angle),
        self_vx, self_vy,
        d1["dist"], d1["sin"], d1["cos"], d1["rel_vx"], d1["rel_vy"],
        d2["dist"], d2["sin"], d2["cos"], d2["rel_vx"], d2["rel_vy"]
    ]
    
    # Convert to PyTorch Tensor
    state_tensor = torch.tensor(state_array, dtype=torch.float32)
    
    # Ask the brain what to do (using torch.no_grad() to save CPU/Memory)
    with torch.no_grad():
        action = model(state_tensor).numpy()
        
    # Extract the AI's predicted actions
    ai_left_speed = action[0]
    ai_right_speed = action[1]
    ai_shoot_flag = action[2]

    # ==========================================
    # 4. APPLY ACTIONS TO PHYSICS
    # ==========================================
    # Clamp motor speeds just in case the AI predicts crazy numbers
    left_speed = max(min(ai_left_speed, 20.0), -20.0)
    right_speed = max(min(ai_right_speed, 20.0), -20.0)
    
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
    
    # Update v_curr for the next frame's velocity estimation
    v_curr = (left_speed + right_speed) / 2.0

    # --- CONDITIONAL DRIBBLER & SHOOTING ---
    is_dribbling = False
    if ball_node is not None:
        ball_pos = ball_node.getPosition()
        dx = ball_pos[0] - pos[0]
        dy = ball_pos[1] - pos[1]
        dist_to_ball = np.hypot(dx, dy)
        angle_to_ball = np.arctan2(dy, dx)
        
        angle_diff = angle_to_ball - heading
        while angle_diff > np.pi: angle_diff -= 2 * np.pi
        while angle_diff < -np.pi: angle_diff += 2 * np.pi
        
        if dist_to_ball < 0.1 and abs(angle_diff) < 0.5:
            is_dribbling = True

    # If the AI predicts a shoot flag > 0.5, execute the kick!
    if is_dribbling and ai_shoot_flag > 0.5:
        kick_vx = np.cos(heading) * SHOOT_POWER
        kick_vy = np.sin(heading) * SHOOT_POWER
        ball_node.setVelocity([kick_vx, kick_vy, 0.5, 0.0, 0.0, 0.0])
        is_dribbling = False
        
    elif is_dribbling:
        # Keep the ball attached if not shooting
        notch_x = pos[0] + np.cos(heading) * DRIBBLER_OFFSET
        notch_y = pos[1] + np.sin(heading) * DRIBBLER_OFFSET
        pull_vx = (notch_x - ball_pos[0]) * DRIBBLER_PULL
        pull_vy = (notch_y - ball_pos[1]) * DRIBBLER_PULL
        ball_node.setVelocity([pull_vx, pull_vy, 0.0, 0.0, 0.0, 0.0])