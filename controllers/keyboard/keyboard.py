from controller import Supervisor, Keyboard
import numpy as np

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

# 4. Supervisor Handles for Defenders & Ball
defenders = []
for i in range(1, 4):
    node = robot.getFromDef(f"defender{i}")
    if node: defenders.append(node)

ball_node = robot.getFromDef("Ball")

# ================= PARAMETERS =================
MAX_SPEED = 10.0
TURN_BIAS = 2.0    
ACCEL = 0.2       
SHOOT_POWER = 3.0  # Velocity applied to the ball when kicked
LOG_FREQUENCY = 8  
step_counter = 0

# State variables for smoothing
v_curr = 0.0
w_curr = 0.0

print("--- TELEOP STARTED ---")
print("Use W/A/S/D to move. Press SPACE to shoot. Data logging active.")

while robot.step(timestep) != -1:
    step_counter += 1
    
    # Target velocities and actions for this frame
    v_target = 0
    w_target = 0
    shoot_flag = 0
    
    # Get Self State (Moved up so the shoot logic can use the heading)
    pos = gps.getValues()
    comp = compass.getValues()
    heading = np.arctan2(comp[0], comp[1])
    
    # 1. Keyboard Logic (Multiple Keys Supported)
    key = keyboard.getKey()
    while key != -1:
        # Translation (Forward/Backward)
        if key == ord('W'):
            v_target = MAX_SPEED
        elif key == ord('S'):
            v_target = -MAX_SPEED
            
        # Rotation (Left/Right Arc)
        if key == ord('A'):
            w_target = TURN_BIAS   
        elif key == ord('D'):
            w_target = -TURN_BIAS  
            
        # Shooting (Spacebar)
        if key == ord(' '):
            if ball_node is not None:
                ball_pos = ball_node.getPosition()
                dist_to_ball = np.hypot(ball_pos[0] - pos[0], ball_pos[1] - pos[1])
                
                # Check if ball is physically touching the front dribbler
                if dist_to_ball < 0.08:
                    shoot_flag = 1
                    
                    # Calculate vector based on the robot's current heading
                    kick_vx = np.cos(heading) * SHOOT_POWER
                    kick_vy = np.sin(heading) * SHOOT_POWER
                    
                    # Apply velocity: [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
                    # Adding a slight Z velocity (0.5) lifts the ball off the ground slightly!
                    ball_node.setVelocity([kick_vx, kick_vy, 0.5, 0.0, 0.0, 0.0])
                    
        key = keyboard.getKey()

    # 2. Apply Smoothing 
    v_curr += (v_target - v_curr) * ACCEL
    w_curr += (w_target - w_curr) * ACCEL
    
    # 3. Combine Translation and Rotation
    left_speed = v_curr - w_curr
    right_speed = v_curr + w_curr
    
    # Safety clamp 
    left_speed = max(min(left_speed, 20.0), -20.0)
    right_speed = max(min(right_speed, 20.0), -20.0)

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
    
    # 4. Data Logging (Console Output)
    if step_counter % LOG_FREQUENCY == 0:
        # Get Defender States (Ground Truth)
        def_positions = []
        for d in defenders:
            def_positions.append(d.getPosition()[:2]) 
            
        # Format the log line (Now includes the shoot_flag!)
        log_entry = {
            "pos": [round(p, 2) for p in pos[:2]],
            "heading": round(heading, 2),
            "defenders": [[round(coord, 2) for coord in d_pos] for d_pos in def_positions],
            "action": [round(left_speed, 2), round(right_speed, 2), shoot_flag]
        }
        
        print(f"LOG | State: {log_entry['pos']}, H:{log_entry['heading']} | Action: {log_entry['action']}")