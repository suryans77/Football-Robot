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
MY_GOAL_CENTER = [1.2, 0.0]  # The anchor point (center of your net)
MAX_RANGE_FROM_GOAL = 0.5    # The "Leash" length 

# --- MOVEMENT PARAMETERS ---
MAX_SPEED = 5.0
BLOCKING_SPEED = 3.0

# PID Steering Constants
Kp = 5.0
Kd = 2.0
prev_error = 0

while robot.step(timestep) != -1:
    if ball_node is None: continue

    # Get Sensor Data
    curr_pos = gps.getValues()
    ball_pos = ball_node.getPosition()
    compass_val = compass.getValues()
    my_heading = math.atan2(compass_val[0], compass_val[1])

    # --- STRATEGY: CALCULATE IDEAL BLOCKING SPOT ---
    
    # 1. Get Vector from Goal to Ball
    vec_x = ball_pos[0] - MY_GOAL_CENTER[0]
    vec_y = ball_pos[1] - MY_GOAL_CENTER[1]
    
    # 2. Normalize Vector (Direction only)
    dist_goal_to_ball = math.hypot(vec_x, vec_y)
    
    # Safety check to avoid divide by zero
    if dist_goal_to_ball < 0.001:
        unit_x, unit_y = 1.0, 0.0
    else:
        unit_x = vec_x / dist_goal_to_ball
        unit_y = vec_y / dist_goal_to_ball

    # 3. Determine Unclamped Target
    # Ideally, we want to be half the distance to the ball, 
    # OR at our max range, whichever is closer.
    
    # "Shadow" distance: Try to stay 0.5m in front of the goal to block
    desired_dist = 0.5 
    
    # 4. APPLY THE LEASH (The Constraint)
    # If the ball is further than our leash, we stay at MAX_RANGE.
    # If the ball comes inside our range, we meet it.
    
    final_dist = min(desired_dist, MAX_RANGE_FROM_GOAL)
    
    # Calculate the exact coordinate on the circle
    target_x = MY_GOAL_CENTER[0] + (unit_x * final_dist)
    target_y = MY_GOAL_CENTER[1] + (unit_y * final_dist)

    # --- NAVIGATION: DRIVE TO TARGET ---
    dx = target_x - curr_pos[0]
    dy = target_y - curr_pos[1]
    dist_to_target = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)

    # PID Steering
    angle_error = target_angle - my_heading
    while angle_error > math.pi: angle_error -= 2 * math.pi
    while angle_error < -math.pi: angle_error += 2 * math.pi

    error_rate = (angle_error - prev_error)
    turn_amount = (Kp * angle_error) + (Kd * error_rate)
    prev_error = angle_error
    
    # Motor Speed Mixing
    left_speed = BLOCKING_SPEED - turn_amount
    right_speed = BLOCKING_SPEED + turn_amount
    
    # Clamp
    left_speed = max(min(left_speed, MAX_SPEED), -MAX_SPEED)
    right_speed = max(min(right_speed, MAX_SPEED), -MAX_SPEED)

    # Jitter Prevention (Stop if we are at the spot)
    if dist_to_target < 0.05:
        left_speed = 0
        right_speed = 0

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
