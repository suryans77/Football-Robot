from controller import Supervisor
import math

# 1. Initialize
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

ball_node = robot.getFromDef("Ball")

# --- GAME PARAMETERS ---
MY_GOAL_CENTER = [1.2, 0.0]   # Coordinate of YOUR goal
STOP_DEFENDING_DIST = 1.0   # Stop if ball is closer than 1.0m to goal

# --- MOVEMENT PARAMETERS ---
MAX_SPEED = 8.0
JOCKEY_SPEED = 4.0 
JOCKEY_DIS = 0.5    # Distance to switch from Sprint to Jockey

# PID Steering Constants
Kp = 8.0
Kd = 3.0
prev_error = 0

while robot.step(timestep) != -1:
    if ball_node is None: continue

    # Get Data
    curr_pos = gps.getValues()
    ball_pos = ball_node.getPosition()
    compass_val = compass.getValues()
    my_heading = math.atan2(compass_val[0], compass_val[1])

    # --- ZONE CHECK: IS THE BALL TOO CLOSE TO MY GOAL? ---
    # Calculate distance between Ball and My Goal
    dx_goal = ball_pos[0] - MY_GOAL_CENTER[0]
    dy_goal = ball_pos[1] - MY_GOAL_CENTER[1]
    dist_ball_to_goal = math.hypot(dx_goal, dy_goal)

    # DEFAULT: We assume we should defend
    should_defend = True

    # If ball is deep in our zone (closer than 1.0m), STOP defending
    if dist_ball_to_goal < STOP_DEFENDING_DIST:
        should_defend = False
    
    # --- BEHAVIOR LOGIC ---
    
    if should_defend:
        # === NORMAL HARASSER LOGIC ===
        
        # Calculate target (Ball)
        dx = ball_pos[0] - curr_pos[0]
        dy = ball_pos[1] - curr_pos[1]
        dist_to_ball = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        
        # Determine Speed (Sprint vs Jockey)
        if dist_to_ball > JOCKEY_DIS:
            base_speed = MAX_SPEED
        else:
            base_speed = JOCKEY_SPEED
            
    else:
        # === STOP / BACK OFF LOGIC ===
        # The ball is too close to goal. Let the goalie handle it.
        # We stop moving to avoid overcrowding.
        base_speed = 0.0
        target_angle = my_heading # Don't try to turn, just freeze
        
        # OPTIONAL: You could make it retreat to a holding spot here instead
        # e.g., target_x = -0.5, target_y = 0.5 (Midfield)

    # --- STEERING CONTROL (PID) ---
    angle_error = target_angle - my_heading
    while angle_error > math.pi: angle_error -= 2 * math.pi
    while angle_error < -math.pi: angle_error += 2 * math.pi

    error_rate = (angle_error - prev_error)
    turn_amount = (Kp * angle_error) + (Kd * error_rate)
    prev_error = angle_error

    # If we are stopped, turn_amount should also be zero
    if base_speed == 0:
        turn_amount = 0

    left_speed = base_speed - turn_amount
    right_speed = base_speed + turn_amount
    
    # Clamp
    left_speed = max(min(left_speed, MAX_SPEED), -MAX_SPEED)
    right_speed = max(min(right_speed, MAX_SPEED), -MAX_SPEED)

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
