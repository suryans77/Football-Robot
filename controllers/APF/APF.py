from controller import Supervisor
import cv2
import numpy as np
import math

# ================= INITIALIZATION =================
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Supervisor handles for defenders
def1 = robot.getFromDef("defender1")
def2 = robot.getFromDef("defender2")

# Standard devices
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)
camera = robot.getDevice("camera")
camera.enable(timestep)

# Proximity sensors for ball detection
ps0 = robot.getDevice("ps0")
ps7 = robot.getDevice("ps7")
ps0.enable(timestep)
ps7.enable(timestep)

# Motors
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# ================= DEBUG SWITCH =================
DEBUG = False

# ================= PARAMETERS =================
GOAL_POS = [1.2, 0.0]

# VRF (Virtual Range Finder) Parameters
NUM_BEAMS = 20
MAX_RANGE = 2.0
VRF_SENSORS = [MAX_RANGE] * NUM_BEAMS

# APF Gains
K_ATTRACT = 1.5
K_REPEL = 0.8
REPULSION_RADIUS = 0.3

# Speed Parameters
MAX_SPEED = 21.0
DRIBBLE_SPEED = 10.0  # Reduced from 10.0 for better ball control
APPROACH_SPEED = 10.0
SEARCH_SPEED = 4.0

# Ball Detection Parameters
CAPTURE_PIXEL_THRESHOLD = 350000
RELEASE_PIXEL_THRESHOLD = 300000
LOWER_ORANGE = np.array([0, 150, 50])
UPPER_ORANGE = np.array([20, 255, 255])

# Turning Parameters
MAX_TURN_RATE = 3.5  # Maximum turning differential
ANGLE_TOLERANCE = 0.2  # Radians - acceptable heading error

# Escape Mechanism
ESCAPE_GAIN = 0.6
MIN_FORCE_THRESHOLD = 0.08
escape_timer = 0

# Heading smoothing
smoothed_heading = 0.0

# ================= FUNCTIONS =================
def get_robot_heading(compass_values):
    """Calculate robot heading from compass values."""
    return math.atan2(compass_values[0], compass_values[1])

def update_vrf(curr_pos):
    """Dynamically fetch defender positions and update 20-beam virtual radar."""
    global VRF_SENSORS
    VRF_SENSORS = [MAX_RANGE] * NUM_BEAMS
    
    # Get defender positions from Supervisor
    dynamic_obstacles = []
    if def1 is not None:
        p1 = def1.getPosition()[:2]
        dynamic_obstacles.append(p1)
    if def2 is not None:
        p2 = def2.getPosition()[:2]
        dynamic_obstacles.append(p2)
    
    # Project obstacles onto VRF beams
    for obs in dynamic_obstacles:
        dx = obs[0] - curr_pos[0]
        dy = obs[1] - curr_pos[1]
        dist = math.hypot(dx, dy)
        
        if dist < MAX_RANGE:
            angle = math.atan2(dy, dx)
            idx = int((angle + math.pi) / (2 * math.pi) * NUM_BEAMS) % NUM_BEAMS
            VRF_SENSORS[idx] = min(VRF_SENSORS[idx], dist)

def calculate_apf_force(curr_pos):
    """Calculate APF force combining goal attraction and VRF-based repulsion."""
    global escape_timer
    
    # 1. Goal Attraction
    dx_g = GOAL_POS[0] - curr_pos[0]
    dy_g = GOAL_POS[1] - curr_pos[1]
    dist_g = math.hypot(dx_g, dy_g)
    
    fx = (dx_g / max(0.01, dist_g)) * K_ATTRACT
    fy = (dy_g / max(0.01, dist_g)) * K_ATTRACT
    
    # 2. VRF-based Repulsion
    for i in range(NUM_BEAMS):
        d = VRF_SENSORS[i]
        if d < REPULSION_RADIUS:
            angle = (i / NUM_BEAMS) * 2 * math.pi - math.pi
            repel_mag = K_REPEL * (1.0/max(0.1, d) - 1.0/REPULSION_RADIUS)
            
            fx -= math.cos(angle) * repel_mag
            fy -= math.sin(angle) * repel_mag
    
    # 3. Escape Mechanism
    force_mag = math.hypot(fx, fy)
    
    if force_mag < MIN_FORCE_THRESHOLD:
        escape_timer += 1
    else:
        escape_timer = 0
    
    if escape_timer > 10:
        random_angle = np.random.uniform(-math.pi, math.pi)
        fx += ESCAPE_GAIN * math.cos(random_angle)
        fy += ESCAPE_GAIN * math.sin(random_angle)
        escape_timer = 0
    
    return fx, fy

def detect_ball(camera):
    """Detect orange ball using computer vision."""
    image_data = camera.getImage()
    if image_data is None:
        return 0, 0, False
    
    width = camera.getWidth()
    height = camera.getHeight()
    
    # Convert to OpenCV format
    frame = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # Detect orange ball
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    moments = cv2.moments(mask)
    pixels_found = moments['m00']
    
    if pixels_found > 100:
        ball_x = int(moments['m10'] / moments['m00'])
        return pixels_found, ball_x, True
    else:
        return 0, 0, False

# ================= MAIN LOOP =================
state = "SEARCH"

while robot.step(timestep) != -1:
    
    # ---- Sensors ----
    gps_vals = gps.getValues()
    if math.isnan(gps_vals[0]):
        continue
    
    curr_pos = [gps_vals[0], gps_vals[1]]
    
    # Smooth heading to reduce jitter
    raw_heading = get_robot_heading(compass.getValues())
    smoothed_heading = (0.9 * smoothed_heading) + (0.1 * raw_heading)
    heading = smoothed_heading
    
    # Ball detection
    pixels_found, ball_x, ball_visible = detect_ball(camera)
    
    # Use hysteresis for state transitions
    if state == "NAVIGATE":
        is_in_pocket = pixels_found > RELEASE_PIXEL_THRESHOLD
    else:
        is_in_pocket = pixels_found > CAPTURE_PIXEL_THRESHOLD
    
    # Update VRF with defender positions
    update_vrf(curr_pos)
    
    # ---- State Machine ----
    if is_in_pocket:
        state = "NAVIGATE"
    elif ball_visible:
        state = "APPROACH"
    else:
        state = "SEARCH"
    
    # ---- Control Logic ----
    if state == "NAVIGATE":
        # Navigate to goal with ball, avoiding defenders using APF
        fx, fy = calculate_apf_force(curr_pos)
        target_angle = math.atan2(fy, fx)
        
        # Calculate angular error (shortest path)
        angle_error = target_angle - heading
        while angle_error > math.pi: 
            angle_error -= 2*math.pi
        while angle_error < -math.pi: 
            angle_error += 2*math.pi
        
        # GENTLE TURNING STRATEGY
        # Calculate turn command with limit
        turn_gain = 8.0
        turn = max(min(angle_error * turn_gain, MAX_TURN_RATE), -MAX_TURN_RATE)
        
        # Slow down based on how much we're turning
        # More turn = slower speed to keep ball
        turn_ratio = abs(turn) / MAX_TURN_RATE
        base_speed = DRIBBLE_SPEED * (1.0 - 0.65 * turn_ratio)
        
        # If turning sharply, reduce speed even more
        if abs(angle_error) > ANGLE_TOLERANCE:
            base_speed *= 0.7  # Additional 30% reduction for sharp turns
        
        left_speed = base_speed - turn
        right_speed = base_speed + turn
        
        if DEBUG:
            dist = math.hypot(GOAL_POS[0] - curr_pos[0], GOAL_POS[1] - curr_pos[1])
            print(f"STATE: {state} | Pixels: {pixels_found:.0f} | AngleErr: {math.degrees(angle_error):.1f}°")
            print(f"Turn: {turn:.2f} | BaseSpeed: {base_speed:.2f} | Dist: {dist:.2f}m")
    
    elif state == "APPROACH":
        # Approach the ball
        width = camera.getWidth()
        turn = ((ball_x / width) - 0.5) * 15.0
        
        left_speed = APPROACH_SPEED + turn
        right_speed = APPROACH_SPEED - turn
        
        if DEBUG:
            print(f"APPROACH | Pixels: {pixels_found:.0f} | Turn: {turn:.2f}")
    
    else:  # SEARCH
        # Rotate in place to find ball
        left_speed = SEARCH_SPEED
        right_speed = -SEARCH_SPEED
    
    # ---- Actuate Motors ----
    left_speed = max(min(left_speed, MAX_SPEED), -MAX_SPEED)
    right_speed = max(min(right_speed, MAX_SPEED), -MAX_SPEED)
    
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)