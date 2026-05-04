from controller import Supervisor, Keyboard
import numpy as np
import json

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

attacker_node = robot.getSelf()
teleop_data = [] # List to store all our logs
ball_node = robot.getFromDef("Ball")

# ================= PARAMETERS =================
BASE_SPEED = 15.0           # Common-mode linear speed command
DIFFERENTIAL_SPEED = 5.0    # Differential speed applied between wheels for rotational yaw
ACCEL = 0.2             # Smoothing factor for motor commands (0.0 = no movement, 1.0 = instant change) 
SHOOT_POWER = 3.0  
LOG_FREQUENCY = 8  
DRIBBLER_OFFSET = 0.05  # Distance from robot center to the front "notch"
DRIBBLER_PULL = 15.0    # Strength of the sticky effect

# Standard e-puck differential drive physical dimensions
WHEEL_RADIUS = 0.02    # 'R' in standard kinematic equations (meters)
AXLE_LENGTH = 0.052    # 'L' in standard kinematic equations (meters)
step_counter = 0

# State variables for smoothing
v_curr = 0.0  # Current base speed (rad/s)
w_curr = 0.0  # Current differential speed (rad/s)

print("--- TELEOP STARTED ---")
print("Use W/A/S/D to move. Press SPACE to shoot. Data logging active.")

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
        
        # Condition: Ball is close (< 0.08m) AND in front of the robot (within ~30 degrees)
        if dist_to_ball < 0.1 and abs(angle_diff) < 0.5:
            is_dribbling = True
    
    # 1. Keyboard Logic
    key = keyboard.getKey()
    while key != -1:
        if key == ord('W'): v_target = BASE_SPEED
        elif key == ord('S'): v_target = -BASE_SPEED
            
        if key == ord('A'): w_target = DIFFERENTIAL_SPEED
        elif key == ord('D'): w_target = -DIFFERENTIAL_SPEED  





######################################  not our focus currently (shooting logic) ######################################  
        # Shooting (Overrides Dribbling)
        if key == ord(' '):
            if is_dribbling:
                shoot_flag = 1
                is_dribbling = False # Break the sticky lock
                
                # Apply massive impulse
                kick_vx = np.cos(heading) * SHOOT_POWER
                kick_vy = np.sin(heading) * SHOOT_POWER
                ball_node.setVelocity([kick_vx, kick_vy, 0.5, 0.0, 0.0, 0.0])
#######################################################################################################################                    
        key = keyboard.getKey()
########################################## sticking ball to the front of the robot ####################################
    # --- APPLY PSEUDO-DRIBBLER PHYSICS ---
    if is_dribbling and shoot_flag == 0:
        # Calculate the ideal Cartesian coordinate of the dribbler notch
        notch_x = pos[0] + np.cos(heading) * DRIBBLER_OFFSET
        notch_y = pos[1] + np.sin(heading) * DRIBBLER_OFFSET
        
        # Spring logic: Velocity is proportional to the distance from the notch
        pull_vx = (notch_x - ball_pos[0]) * DRIBBLER_PULL
        pull_vy = (notch_y - ball_pos[1]) * DRIBBLER_PULL
        
        # Lock the ball's velocity to pull it into the notch, killing erratic spin
        ball_node.setVelocity([pull_vx, pull_vy, 0.0, 0.0, 0.0, 0.0])
#######################################################################################################################




    # 2. Apply Motor Smoothing
    v_curr += (v_target - v_curr) * ACCEL
    w_curr += (w_target - w_curr) * ACCEL
    
    # Calculate final wheel angular velocities (rad/s)
    # left_omega = v - w, right_omega = v + w
    # The 'w' component is the differential speed, increasing the speed of one wheel 
    # relative to the other to enable turning without lateral strafing.
    left_omega = max(min(v_curr - w_curr, 21.0), -21.0)
    right_omega = max(min(v_curr + w_curr, 21.0), -21.0)

    left_motor.setVelocity(left_omega)
    right_motor.setVelocity(right_omega)
    
    # 3. Data Logging
    if step_counter % LOG_FREQUENCY == 0:
        # --- ATTACKER MATH ---
        att_vel = attacker_node.getVelocity() # Returns Cartesian [vx, vy, vz, wx, wy, wz]
        
        # INVERSE KINEMATICS (Linear): 
        # Project global 2D Cartesian velocity onto the local forward axis (X)
        # Then divide by wheel radius (R) to find the equivalent common-mode wheel speed
        att_v_linear = att_vel[0] * np.cos(heading) + att_vel[1] * np.sin(heading)
        att_actual_v = att_v_linear / WHEEL_RADIUS
        
        # INVERSE KINEMATICS (Angular):
        # Extract chassis yaw rate (wz) and apply standard differential drive formula:
        # Differential wheel speed (w) = (Yaw_Rate * L) / (2 * R)
        att_yaw_rate = att_vel[5]
        att_actual_w = (att_yaw_rate * AXLE_LENGTH) / (2.0 * WHEEL_RADIUS)
        
        # --- DEFENDER MATH ---
        def_vel = defenders[0].getVelocity()
        
        #absolute 3x3 rotation matrix from the physics engine
        def_rot = defenders[0].getOrientation()
        
        # In Webots, the robot's local forward axis (X) is stored in matrix indices 0 and 3
        # We mathematically project the global X/Y velocity onto this forward vector
        def_v_linear = def_vel[0] * def_rot[0] + def_vel[1] * def_rot[3]
        def_actual_v = def_v_linear / WHEEL_RADIUS
        
        def_yaw_rate = def_vel[5]
        def_actual_w = (def_yaw_rate * AXLE_LENGTH) / (2.0 * WHEEL_RADIUS)
        
        log_entry = {
            "step": step_counter,
            "target_v": v_target,
            "target_w": w_target,
            "att_actual_v": att_actual_v,
            "att_actual_w": att_actual_w,
            "def_actual_v": def_actual_v,
            "def_actual_w": def_actual_w
        }
        teleop_data.append(log_entry)

with open("teleop_dataset.json", "w") as f:
    json.dump(teleop_data, f, indent=4)
print("Data saved to teleop_dataset.json!")