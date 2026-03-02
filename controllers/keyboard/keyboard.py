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

# 3. Setup Sensors (for the AI "Features")
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# 4. Supervisor Handles for Defenders
defenders = []
for i in range(1, 4):
    node = robot.getFromDef(f"defender{i}")
    if node: defenders.append(node)

# ================= PARAMETERS =================
MAX_SPEED = 10.0
TURN_BIAS = 2.0    # How sharply the robot turns while moving
ACCEL = 0.2       # Smoothing factor (Lower = smoother, Higher = more responsive)
LOG_FREQUENCY = 8  
step_counter = 0

# State variables for smoothing
v_curr = 0.0
w_curr = 0.0

print("--- TELEOP STARTED ---")
print("Use W/A/S/D to move. Data logging active.")

while robot.step(timestep) != -1:
    step_counter += 1
    
    # Target velocities for this frame
    v_target = 0.0
    w_target = 0.0
    
    # 1. Keyboard Logic (Multiple Keys Supported)
    key = keyboard.getKey()
    while key != -1:
        # Translation (Forward/Backward)
        if key == ord('W'):
            v_target = MAX_SPEED
        elif key == ord('S'):
            v_target = -MAX_SPEED
            
        # Rotation (Left/Right Arc) applied on top of translation
        if key == ord('A'):
            w_target = TURN_BIAS   # Positive bias turns left
        elif key == ord('D'):
            w_target = -TURN_BIAS  # Negative bias turns right
            
        key = keyboard.getKey()

    # 2. Apply Smoothing (The "Interpolation" to prevent ball slip)
    v_curr += (v_target - v_curr) * ACCEL
    w_curr += (w_target - w_curr) * ACCEL
    
    # 3. Combine Translation and Rotation
    # If moving forward (v=10) and turning left (w=3.5):
    # Left = 6.5, Right = 13.5 -> Robot curves left without stopping!
    left_speed = v_curr - w_curr
    right_speed = v_curr + w_curr
    
    # Safety clamp to ensure we don't exceed motor max speed (usually ~20-21 in Webots)
    left_speed = max(min(left_speed, 20.0), -20.0)
    right_speed = max(min(right_speed, 20.0), -20.0)

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
    
    # 4. Data Logging (Console Output)
    if step_counter % LOG_FREQUENCY == 0:
        # Get Self State
        pos = gps.getValues()
        comp = compass.getValues()
        heading = np.arctan2(comp[0], comp[1])
        
        # Get Defender States (Ground Truth)
        def_positions = []
        for d in defenders:
            def_positions.append(d.getPosition()[:2]) 
            
        # Format the log line
        log_entry = {
            "pos": [round(p, 2) for p in pos[:2]],
            "heading": round(heading, 2),
            "defenders": [[round(coord, 2) for coord in d_pos] for d_pos in def_positions],
            "action": [round(left_speed, 2), round(right_speed, 2)]
        }
        
        print(f"LOG | State: {log_entry['pos']}, H:{log_entry['heading']} | Action: {log_entry['action']}")