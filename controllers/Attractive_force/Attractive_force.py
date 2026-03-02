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

# 4. Supervisor Handles for Targets & Defenders
goal_node = robot.getFromDef("GOAL") # MAKE SURE YOUR GOAL HAS THIS DEF NAME
defenders = []
for i in range(1, 4):
    node = robot.getFromDef(f"defender{i}")
    if node: defenders.append(node)

# ================= CSV LOGGER SETUP =================
CSV_FILENAME = "residual_dataset.csv"
file_exists = os.path.isfile(CSV_FILENAME)
csv_file = open(CSV_FILENAME, mode='a', newline='')
writer = csv.writer(csv_file)
if not file_exists:
    # 6 State Values + 2 Residuals
    writer.writerow(['r_goal', 'theta_goal', 'r_def', 'theta_def', 'v_def_mag', 'v_def_head', 'res_left', 'res_right'])

# ================= PARAMETERS =================
MAX_SPEED = 10.0
TURN_BIAS = 2.0    # How sharply the robot turns while moving
ACCEL = 0.2        # Smoothing factor 
LOG_FREQUENCY = 8  
step_counter = 0

# State variables for smoothing
v_curr = 0.0
w_curr = 0.0

# APF Tuning Parameters
K_rho = 2.5   # Forward speed gain based on distance to goal
K_alpha = 4.0 # Turning gain based on angle to goal

print("--- HYBRID CONTROLLER STARTED ---")
print("APF driving automatically. Use W/A/S/D to dodge. Data logging active.")

# ================= HELPER FUNCTION =================
def get_ego_polar(r_pos, r_heading, t_pos):
    """Converts global X,Y to ego-centric distance (r) and angle (theta)"""
    dx = t_pos[0] - r_pos[0]
    dy = t_pos[1] - r_pos[1]
    r = np.hypot(dx, dy)
    global_theta = np.arctan2(dy, dx)
    relative_theta = global_theta - r_heading
    # Normalize to [-pi, pi]
    relative_theta = np.arctan2(np.sin(relative_theta), np.cos(relative_theta))
    return r, relative_theta

# ================= MAIN LOOP =================
while robot.step(timestep) != -1:
    step_counter += 1
    
    # --- 1. SENSOR READINGS & STATE EXTRACTION ---
    pos = gps.getValues()
    comp = compass.getValues()
    heading = np.arctan2(comp[0], comp[1])
    
    goal_pos = goal_node.getPosition()
    
    # Calculate goal polar coordinates
    r_goal, theta_goal = get_ego_polar(pos, heading, goal_pos)
    
    # Find the NEAREST defender for the State Vector
    closest_def = None
    min_dist = float('inf')
    r_def, theta_def = 0.0, 0.0
    
    for d in defenders:
        d_pos = d.getPosition()
        dist, angle = get_ego_polar(pos, heading, d_pos)
        if dist < min_dist:
            min_dist = dist
            closest_def = d
            r_def, theta_def = dist, angle
            
    # Extract nearest defender's velocity dynamics
    d_vel = closest_def.getVelocity() if closest_def else [0,0,0,0,0,0]
    v_def_mag = np.hypot(d_vel[0], d_vel[1])
    def_global_heading = np.arctan2(d_vel[1], d_vel[0])
    v_def_head = np.arctan2(np.sin(def_global_heading - heading), np.cos(def_global_heading - heading))

    # Compile the highly-efficient 6-value state
    state_vector = [r_goal, theta_goal, r_def, theta_def, v_def_mag, v_def_head]

    # --- 2. CALCULATE BASE APF (Attractive Force Only) ---
    # Proportional control for smooth goal tracking
    v_apf = min(K_rho * r_goal, MAX_SPEED) 
    w_apf = K_alpha * theta_goal
    
    # Convert v and w to left/right wheel speeds
    apf_left = max(min(v_apf - w_apf, 20.0), -20.0)
    apf_right = max(min(v_apf + w_apf, 20.0), -20.0)

    # --- 3. KEYBOARD LOGIC (Your Smooth Teleop) ---
    v_target, w_target = 0.0, 0.0
    key_pressed = False
    key = keyboard.getKey()
    
    while key != -1:
        key_pressed = True
        if key == ord('W'): v_target = MAX_SPEED
        elif key == ord('S'): v_target = -MAX_SPEED
        
        if key == ord('A'): w_target = TURN_BIAS   
        elif key == ord('D'): w_target = -TURN_BIAS  
        key = keyboard.getKey()

    # --- 4. CONTROL FUSION & LOGGING ---
    if key_pressed:
        # A. Apply your smoothing logic to human inputs
        v_curr += (v_target - v_curr) * ACCEL
        w_curr += (w_target - w_curr) * ACCEL
        
        human_left = max(min(v_curr - w_curr, 20.0), -20.0)
        human_right = max(min(v_curr + w_curr, 20.0), -20.0)
        
        # B. Calculate the Residual (What did the human do differently than APF?)
        res_left = human_left - apf_left
        res_right = human_right - apf_right
        
        # C. Save to Dataset
        writer.writerow(state_vector + [res_left, res_right])
        
        # D. Move robot using human commands
        left_motor.setVelocity(human_left)
        right_motor.setVelocity(human_right)
        
    else:
        # Autonomous mode: Drive to goal using APF, reset smoothing variables
        v_curr, w_curr = 0.0, 0.0
        left_motor.setVelocity(apf_left)
        right_motor.setVelocity(apf_right)
        # Note: We do NOT save data here. Only human corrections are saved.

    # --- 5. CONSOLE LOGGING ---
    if step_counter % LOG_FREQUENCY == 0:
        action_type = "HUMAN OVERRIDE" if key_pressed else "APF AUTO"
        print(f"[{action_type}] Goal Dist: {r_goal:.2f}m | Nearest Def Dist: {r_def:.2f}m")

# Close CSV cleanly when simulation stops
csv_file.close()