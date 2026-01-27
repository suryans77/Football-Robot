from controller import Robot
import cv2
import numpy as np

# 1. INITIALIZATION
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Devices
camera = robot.getDevice("camera")
camera.enable(timestep)

ps0 = robot.getDevice("ps0") # Right
ps7 = robot.getDevice("ps7") # Left
ps0.enable(timestep); ps7.enable(timestep)

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf')); right_motor.setPosition(float('inf'))

# ------------------ PARAMETERS ------------------
MAX_SPEED = 21.0
DRIBBLE_SPEED = 7.0
SEARCH_SPEED = 4.0
IN_POCKET_LIMIT = 3500

# HSV Ranges for Orange (Ignores Brown)
# Hue: 0-15 is the orange/red spectrum
# Saturation: 150-255 filters out gray/brown/muddy colors
# Value: 50-255 filters out pure black shadows
LOWER_ORANGE = np.array([0, 150, 50])
UPPER_ORANGE = np.array([20, 255, 255])

print("=== OpenCV High-Friction Dribbler ONLINE ===")

while robot.step(timestep) != -1:
    # --- 1. SENSOR READS ---
    dist_l = ps0.getValue()
    dist_r = ps7.getValue()
    
    # --- 2. OPENCV PROCESSING ---
    image_data = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()

    # Convert Webots BGRA string to Numpy Array
    frame = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
    
    # Convert to BGR then to HSV
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # Create the Mask
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    
    # Calculate Moments to find the center of the "blob"
    moments = cv2.moments(mask)
    pixels_found = moments['m00'] # This is the total area of the mask

    # --- 3. CONTROL LOGIC ---
    
    # If the ball is detected (Area > 0)
    if pixels_found > 100: 
        # Calculate horizontal center (X)
        ball_x = int(moments['m10'] / moments['m00'])
        deviation = (ball_x / width) - 0.5
        
        # Check if ball is in the funnel pocket
        is_in_pocket = dist_l > IN_POCKET_LIMIT or dist_r > IN_POCKET_LIMIT
        
        if is_in_pocket:
            # Maintain a steady nudge speed
            base_speed = DRIBBLE_SPEED
            gain = 4.0
        else:
            # Chase speed logic
            # Use area (pixels_found) as a distance estimate
            if pixels_found > 5000: # Ball is close
                base_speed = DRIBBLE_SPEED + 2
                gain = 8.0
            else: # Ball is far
                base_speed = MAX_SPEED
                gain = 15.0

        # Steering
        turn_effort = 0.0 if abs(deviation) < 0.03 else (deviation * gain)
        left_speed = base_speed + turn_effort
        right_speed = base_speed - turn_effort
        
    else:
        # Ball lost -> Search
        left_speed = SEARCH_SPEED
        right_speed = -SEARCH_SPEED

    # --- 4. ACTUATION ---
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)