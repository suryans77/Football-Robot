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

# 4. Supervisor Handles
defenders = []
for i in range(1, 4):
    node = robot.getFromDef(f"defender{i}")
    if node: defenders.append(node)

ball_node = robot.getFromDef("Ball")

# ================= PARAMETERS =================
DRIBBLE_SPEED = 15.0
MAX_SPEED = 18.0
DRIBBLE_TURN = 5.0    
MAX_TURN = 6.0
DRIBBLE_ACCEL = 0.2   
MAX_ACCEL = 0.25
SHOOT_POWER = 3.0  
KICK_POWER = 0.8
LOG_FREQUENCY = 8  
DRIBBLER_OFFSET = 0.05  # Distance from robot center to the front "notch"
DRIBBLER_PULL = 15.0    # Strength of the sticky effect
step_counter = 0

# State variables for smoothing
v_curr = 0.0
w_curr = 0.0

print("--- TELEOP STARTED ---")
print("Use W/A/S/D to move. Press SPACE to shoot. Data logging active.")

while robot.step(timestep) != -1:
    step_counter += 1
    
    v_target = 0
    w_target = 0
    shoot_flag = 0
    kick_flag = 0
    
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
        if key == ord('W'): 
            if is_dribbling:
                v_target = DRIBBLE_SPEED
            else:
                v_target = MAX_SPEED
        elif key == ord('S'): 
            if is_dribbling:
                v_target = -DRIBBLE_SPEED
            else:
                v_target = -MAX_SPEED
            
        if key == ord('A'): 
            if is_dribbling:
                w_target = DRIBBLE_TURN
            else:                
                w_target = MAX_TURN   
        elif key == ord('D'): 
            if is_dribbling:
                w_target = -DRIBBLE_TURN
            else:
                w_target = -MAX_TURN


        # Shooting (Overrides Dribbling)
        if key == ord(' '):
            if is_dribbling:
                shoot_flag = 1
                is_dribbling = False # Break the sticky lock
                
                # Apply massive impulse
                kick_vx = np.cos(heading) * SHOOT_POWER
                kick_vy = np.sin(heading) * SHOOT_POWER
                ball_node.setVelocity([kick_vx, kick_vy, 0.5, 0.0, 0.0, 0.0])

        elif key == ord('K'):
            if is_dribbling:
                kick_flag = 1
                is_dribbling = False # Break the sticky lock
                
                # Apply massive impulse
                kick_vx = np.cos(heading) * KICK_POWER
                kick_vy = np.sin(heading) * KICK_POWER
                ball_node.setVelocity([kick_vx, kick_vy, 0.0, 0.0, 0.0, 0.0])
                    
        key = keyboard.getKey()

    # --- APPLY PSEUDO-DRIBBLER PHYSICS ---
    if is_dribbling and (shoot_flag == 0 and kick_flag== 0):
        # Calculate the ideal Cartesian coordinate of the dribbler notch
        notch_x = pos[0] + np.cos(heading) * DRIBBLER_OFFSET
        notch_y = pos[1] + np.sin(heading) * DRIBBLER_OFFSET
        
        # Spring logic: Velocity is proportional to the distance from the notch
        pull_vx = (notch_x - ball_pos[0]) * DRIBBLER_PULL
        pull_vy = (notch_y - ball_pos[1]) * DRIBBLER_PULL
        
        # Lock the ball's velocity to pull it into the notch, killing erratic spin
        ball_node.setVelocity([pull_vx, pull_vy, 0.0, 0.0, 0.0, 0.0])

    # 2. Apply Motor Smoothing 
    if is_dribbling:
        v_curr += (v_target - v_curr) * DRIBBLE_ACCEL
        w_curr += (w_target - w_curr) * DRIBBLE_ACCEL
    else:
        v_curr += (v_target - v_curr) * MAX_ACCEL
        w_curr += (w_target - w_curr) * MAX_ACCEL

    left_speed = max(min(v_curr - w_curr, 21.0), -21.0)
    right_speed = max(min(v_curr + w_curr, 21.0), -21.0)

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
    
    # 3. Data Logging
    if step_counter % LOG_FREQUENCY == 0:
        def_positions = [d.getPosition()[:2] for d in defenders]
        
        log_entry = {
            "pos": [round(p, 2) for p in pos[:2]],
            "heading": round(heading, 2),
            "defenders": [[round(coord, 2) for coord in d_pos] for d_pos in def_positions],
            "action": [round(left_speed, 2), round(right_speed, 2), shoot_flag, kick_flag]
        }
        
        print(f"LOG | Pos: {log_entry['pos']}, H:{log_entry['heading']} | Act: {log_entry['action']}")