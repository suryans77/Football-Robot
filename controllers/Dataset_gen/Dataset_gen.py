from controller import Supervisor, Keyboard
import numpy as np
import csv
import os

# ================= INITIALIZATION =================
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# 1. Enable Keyboard
keyboard = Keyboard()
keyboard.enable(timestep)

# 2. Setup Motors
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# 3. Setup Sensors 
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# 4. Supervisor Handles
defenders = []
for i in range(1, 4):
    node = robot.getFromDef(f"defender{i}")
    if node: defenders.append(node)

ball_node = robot.getFromDef("Ball")

# ================= PARAMETERS =================
MY_GOAL_CENTER = [2.0, 0.0]  # Required for ML relative goal calculations

MAX_SPEED = 15.0
TURN_BIAS = 5.0    
ACCEL = 0.2       
SHOOT_POWER = 3.0  
LOG_FREQUENCY = 8  
DRIBBLER_OFFSET = 0.05  
DRIBBLER_PULL = 15.0    
step_counter = 0

# State variables for smoothing
v_curr = 0.0
w_curr = 0.0

# --- DATA LOGGING SETUP ---
CSV_FILENAME = "expert_demonstrations.csv"
episode_data = []  

# Create CSV and write header if it doesn't exist
if not os.path.exists(CSV_FILENAME):
    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "goal_dist", "goal_sin", "goal_cos", 
            "self_vx", "self_vy",
            "d1_dist", "d1_sin", "d1_cos", "d1_vx", "d1_vy",
            "d2_dist", "d2_sin", "d2_cos", "d2_vx", "d2_vy",
            "action_left", "action_right", "action_shoot"
        ])

print("--- TELEOP STARTED ---")
print("Move: W/A/S/D | Shoot: SPACE")
print("SAVE Run to CSV: Press 'R'")
print("DISCARD mistakes : Press 'X'")

while robot.step(timestep) != -1:
    step_counter += 1
    
    v_target = 0
    w_target = 0
    shoot_flag = 0
    
    # Get Self State
    pos = gps.getValues()
    comp = compass.getValues()
    heading = np.arctan2(comp[0], comp[1])
    
    # --- CONDITIONAL DRIBBLER CHECK ---
    is_dribbling = False
    if ball_node is not None:
        ball_pos = ball_node.getPosition()
        dx = ball_pos[0] - pos[0]
        dy = ball_pos[1] - pos[1]
        dist_to_ball = np.hypot(dx, dy)
        angle_to_ball = np.arctan2(dy, dx)
        
        # Normalize angle difference
        angle_diff = angle_to_ball - heading
        while angle_diff > np.pi: angle_diff -= 2 * np.pi
        while angle_diff < -np.pi: angle_diff += 2 * np.pi
        
        if dist_to_ball < 0.1 and abs(angle_diff) < 0.5:
            is_dribbling = True
    
    # 1. Keyboard Logic
    key = keyboard.getKey()
    while key != -1:
        if key == ord('W'): v_target = MAX_SPEED
        elif key == ord('S'): v_target = -MAX_SPEED
            
        if key == ord('A'): w_target = TURN_BIAS   
        elif key == ord('D'): w_target = -TURN_BIAS  
            
        # Shooting
        if key == ord(' '):
            if is_dribbling:
                shoot_flag = 1
                is_dribbling = False 
                
                kick_vx = np.cos(heading) * SHOOT_POWER
                kick_vy = np.sin(heading) * SHOOT_POWER
                ball_node.setVelocity([kick_vx, kick_vy, 0.5, 0.0, 0.0, 0.0])

        # --- ML SAVE/DISCARD LOGIC ---
        if key == ord('R'):
            if len(episode_data) > 0:
                with open(CSV_FILENAME, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(episode_data)
                print(f"SUCCESS: Saved {len(episode_data)} frames to CSV.")
                episode_data.clear()
        
        elif key == ord('X'):
            print("DISCARDED: Bad run trashed from RAM. Try again.")
            episode_data.clear()
                    
        key = keyboard.getKey()

    # --- APPLY PSEUDO-DRIBBLER PHYSICS ---
    if is_dribbling and shoot_flag == 0:
        notch_x = pos[0] + np.cos(heading) * DRIBBLER_OFFSET
        notch_y = pos[1] + np.sin(heading) * DRIBBLER_OFFSET
        
        pull_vx = (notch_x - ball_pos[0]) * DRIBBLER_PULL
        pull_vy = (notch_y - ball_pos[1]) * DRIBBLER_PULL
        
        ball_node.setVelocity([pull_vx, pull_vy, 0.0, 0.0, 0.0, 0.0])

    # 2. Apply Motor Smoothing 
    v_curr += (v_target - v_curr) * ACCEL
    w_curr += (w_target - w_curr) * ACCEL
    
    left_speed = max(min(v_curr - w_curr, 20.0), -20.0)
    right_speed = max(min(v_curr + w_curr, 20.0), -20.0)

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
    
    # ==========================================
    # --- ML DATA EXTRACTION & LOGGING ---
    # ==========================================
    if step_counter % LOG_FREQUENCY == 0:
        
        # Estimate physical self-velocity for the ML inputs
        est_linear_v = v_curr * 0.02 # Approx wheel radius
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
                "dist": d_dist, 
                "sin": np.sin(d_angle), 
                "cos": np.cos(d_angle),
                "rel_vx": rel_vx,
                "rel_vy": rel_vy
            })
            
        # Sort defenders by distance to guarantee D1 is the closest
        def_stats.sort(key=lambda x: x["dist"])
        
        d1 = def_stats[0] if len(def_stats) > 0 else {"dist": 10.0, "sin": 0.0, "cos": 0.0, "rel_vx": 0.0, "rel_vy": 0.0}
        d2 = def_stats[1] if len(def_stats) > 1 else {"dist": 10.0, "sin": 0.0, "cos": 0.0, "rel_vx": 0.0, "rel_vy": 0.0}

        # 3. Construct the 18-element row
        frame_data = [
            round(goal_dist, 3), round(np.sin(goal_angle), 3), round(np.cos(goal_angle), 3),
            round(self_vx, 3), round(self_vy, 3),
            round(d1["dist"], 3), round(d1["sin"], 3), round(d1["cos"], 3), round(d1["rel_vx"], 3), round(d1["rel_vy"], 3),
            round(d2["dist"], 3), round(d2["sin"], 3), round(d2["cos"], 3), round(d2["rel_vx"], 3), round(d2["rel_vy"], 3),
            round(left_speed, 2), round(right_speed, 2), shoot_flag
        ]
        
        episode_data.append(frame_data)