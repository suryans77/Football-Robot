from controller import Robot
import cv2
import numpy as np
import math

# 1. INITIALIZATION
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Navigation Devices (Must be in extensionSlot and named correctly)
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Vision and Proximity
camera = robot.getDevice("camera")
camera.enable(timestep)
ps0, ps7 = robot.getDevice("ps0"), robot.getDevice("ps7")
ps0.enable(timestep); ps7.enable(timestep)

# Motors
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# ------------------ PARAMETERS ------------------
# Change these and watch the console 'Goal' print to verify they updated
GOAL_POS = [-1.20, 0.0]  

MAX_SPEED       = 21.0
DRIBBLE_SPEED   = 10.0
SEARCH_SPEED    = 4.0
IN_POCKET_LIMIT = 10
smoothed_heading = 0.0
CAPTURE_PIXEL_THRESHOLD = 350000

LOWER_ORANGE = np.array([0, 150, 50])
UPPER_ORANGE = np.array([20, 255, 255])

# ------------------ NAVIGATION HELPERS ------------------
def get_robot_heading(compass_values):
    # Webots Compass returns a vector. atan2(y, x) gives the angle.
    # If the robot circles the goal instead of hitting it, swap to (cv[0], cv[1])
    return math.atan2(compass_values[0], compass_values[1])

# ------------------ MAIN LOOP ------------------
state = "SEARCH"

print("--- Controller Started: Use Console to verify GPS/Compass ---")

while robot.step(timestep) != -1:
    # ─── 1. Position & Heading ───
    full_gps = gps.getValues() 
    # Check if GPS is returning valid numbers
    if math.isnan(full_gps[0]):
        print("Waiting for GPS signal...")
        continue
        
    curr_pos = [full_gps[0], full_gps[1]] # X and Y floor coordinates
    raw_heading = get_robot_heading(compass.getValues())
    # Filter: Take 90% of the old value and 10% of the new one to stop jitter
    smoothed_heading = (0.9 * smoothed_heading) + (0.1 * raw_heading)
    heading = smoothed_heading
    
    # ─── 2. Vision ───
    image_data = camera.getImage()
    width, height = camera.getWidth(), camera.getHeight()
    frame = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
    hsv = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    moments = cv2.moments(mask)
    pixels_found = moments['m00']

    # 2. Pixel-Based State Logic
    ball_visible = pixels_found > 100
    is_in_pocket = pixels_found > CAPTURE_PIXEL_THRESHOLD

    if is_in_pocket:
        state = "GO_TO_GOAL"
    elif ball_visible:
        state = "APPROACH"
        # Updated print statement with pixels
        print(f"APPROACH | Pixels: {pixels_found:.0f} | PS0: {ps0.getValue():.0f}")
    else:
        state = "SEARCH"

    # ─── 4. Motor Control ───
    if state == "GO_TO_GOAL":
        # Vector Math
        dx = GOAL_POS[0] - curr_pos[0]
        dy = GOAL_POS[1] - curr_pos[1]
        target_angle = math.atan2(dy, dx)
        
        # Calculate Shortest Rotation Error
        angle_error = target_angle - heading
        while angle_error > math.pi: angle_error -= 2*math.pi
        while angle_error < -math.pi: angle_error += 2*math.pi

        # Proportional Steering
        turn = angle_error * 12.0
        left_speed = DRIBBLE_SPEED - turn
        right_speed = DRIBBLE_SPEED + turn
        
        # --- CRITICAL DEBUG STATEMENTS ---
        # 1. Check if GOAL_POS matches what you typed
        # 2. Check if curr_pos is changing as the robot moves
        # 3. Check if My_Deg changes when the robot rotates
        dist = math.sqrt(dx**2 + dy**2)
        print(f"Pixels: {pixels_found:.0f} | PS0: {ps0.getValue():.0f}")
        print(f"STATE: {state} | Goal: {GOAL_POS} | Curr_Pos: [{curr_pos[0]:.2f}, {curr_pos[1]:.2f}]")
        print(f"Target_Deg: {math.degrees(target_angle):.1f} | My_Deg: {math.degrees(heading):.1f} | Dist: {dist:.2f}m")
        print("-" * 30)

    elif state == "APPROACH":
        ball_x = int(moments['m10'] / moments['m00']) if pixels_found > 0 else width//2
        turn = ((ball_x / width) - 0.5) * 15.0
        left_speed = 10.0 + turn
        right_speed = 10.0 - turn
        
    else: # SEARCH
        left_speed, right_speed = SEARCH_SPEED, -SEARCH_SPEED

    # 5. Actuate
    left_speed = max(min(left_speed, MAX_SPEED), -MAX_SPEED)
    right_speed = max(min(right_speed, MAX_SPEED), -MAX_SPEED)
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)