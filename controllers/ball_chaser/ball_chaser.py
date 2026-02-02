from controller import Robot
import cv2
import numpy as np

# 1. INITIALIZATION
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Devices
camera = robot.getDevice("camera")
camera.enable(timestep)

ps0 = robot.getDevice("ps0")  # Right (front-right)
ps7 = robot.getDevice("ps7")  # Left  (front-left)
ps0.enable(timestep)
ps7.enable(timestep)

left_motor  = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# ------------------ PARAMETERS ------------------
MAX_SPEED     = 21.0
DRIBBLE_SPEED = 7.0
SEARCH_SPEED  = 4.0
SPIN_SPEED    = 5.0          # base spin velocity (rad/s)
IN_POCKET_LIMIT = 3500       # ≈ very close detection

# HSV Ranges for Orange ball
LOWER_ORANGE = np.array([0,   150,  50])
UPPER_ORANGE = np.array([20,  255, 255])

print("=== OpenCV High-Friction Dribbler with SPIN after capture ===")

state = "SEARCH"   # can be: SEARCH / APPROACH / DRIBBLE / SPIN

while robot.step(timestep) != -1:
    # ─── 1. Read sensors ────────────────────────────────────────
    dist_l = ps7.getValue()   # left front
    dist_r = ps0.getValue()   # right front
    
    # ─── 2. Camera + OpenCV ─────────────────────────────────────
    image_data = camera.getImage()
    width  = camera.getWidth()
    height = camera.getHeight()

    frame = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    moments = cv2.moments(mask)
    pixels_found = moments['m00']

    ball_visible = pixels_found > 100
    is_in_pocket = (dist_l > IN_POCKET_LIMIT) or (dist_r > IN_POCKET_LIMIT)

    # ─── 3. State machine + control logic ───────────────────────
    if ball_visible and is_in_pocket:
        # We have the ball → go to spinning
        state = "SPIN"
    elif ball_visible:
        state = "APPROACH"   # or DRIBBLE – here we merge them
    else:
        state = "SEARCH"

    # ─── 4. Set velocities according to state ───────────────────
    if state == "SPIN":
        # Spin in place – keep slight forward bias to maintain contact
        left_speed  =  SPIN_SPEED + 1.0
        right_speed = -SPIN_SPEED + 1.0
        
        # Optional: add very small random walk component every few seconds
        # (can help if stuck against wall – but usually not needed)

    elif state == "APPROACH":
        if not ball_visible:
            # safety fallback
            left_speed  = SEARCH_SPEED
            right_speed = -SEARCH_SPEED
        else:
            # Normal ball following
            ball_x = int(moments['m10'] / moments['m00']) if moments['m00'] > 0 else width//2
            deviation = (ball_x / width) - 0.5

            if pixels_found > 5000:     # close
                base_speed = DRIBBLE_SPEED
                gain = 8.0
            else:                       # far
                base_speed = MAX_SPEED
                gain = 15.0

            turn = deviation * gain
            left_speed  = base_speed + turn
            right_speed = base_speed - turn

    else:  # SEARCH
        left_speed  = SEARCH_SPEED
        right_speed = -SEARCH_SPEED

    # Clamp speeds
    left_speed  = max(min(left_speed,  MAX_SPEED), -MAX_SPEED)
    right_speed = max(min(right_speed,  MAX_SPEED), -MAX_SPEED)

    # ─── 5. Actuate ─────────────────────────────────────────────
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    # Optional debug print (uncomment when tuning)
    if state == "SPIN":
        print(f"SPINNING | psL={dist_l:.0f} psR={dist_r:.0f} area={pixels_found:.0f}")