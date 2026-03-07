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

# --- KEEPER PARAMETERS ---
GOAL_CENTER = [1.5, 0.0] 
KEEPER_RADIUS = 0.1     # The arc distance: 20cm out from the goal center
GOAL_Y_LIMIT = 0.25      # Goal width constraint

# --- MOVEMENT PARAMETERS ---
NORMAL_SPEED = 15.0      # Fast, controlled tracking
DASH_SPEED = 21.0        # Absolute maximum for diving saves
SHOT_THRESHOLD = 3.0     # Ball speed (m/s) that triggers a dive

# Steering Constants
Kp = 8.0
Kd = 1.5
prev_error = 0.0

while robot.step(timestep) != -1:
    if ball_node is None: continue

    curr_pos = gps.getValues()
    ball_pos = ball_node.getPosition()
    compass_val = compass.getValues()
    my_heading = math.atan2(compass_val[0], compass_val[1])

    # --- GET BALL SPEED (Detecting a Shot) ---
    ball_vel = ball_node.getVelocity()
    ball_speed = math.hypot(ball_vel[0], ball_vel[1])
    is_shot_incoming = ball_speed > SHOT_THRESHOLD

    # --- STRATEGY: THE DEFENSIVE ARC ---
    # 1. Calculate vector from Goal to Ball
    vec_x = ball_pos[0] - GOAL_CENTER[0]
    vec_y = ball_pos[1] - GOAL_CENTER[1]
    dist_goal_to_ball = math.hypot(vec_x, vec_y)
    
    if dist_goal_to_ball < 0.001:
        unit_x, unit_y = -1.0, 0.0
    else:
        unit_x = vec_x / dist_goal_to_ball
        unit_y = vec_y / dist_goal_to_ball
        
    # 2. Project the ideal spot on the arc
    target_x = GOAL_CENTER[0] + (unit_x * KEEPER_RADIUS)
    target_y = GOAL_CENTER[1] + (unit_y * KEEPER_RADIUS)
    
    # 3. Clamp Y so the keeper doesn't wander outside the goalposts
    target_y = max(min(target_y, GOAL_Y_LIMIT), -GOAL_Y_LIMIT)

    # --- BEHAVIOR STATE MACHINE ---
    dx = target_x - curr_pos[0]
    dy = target_y - curr_pos[1]
    dist_to_target = math.hypot(dx, dy)

    if dist_to_target < 0.04:
        # STATE 1: THE WALL (Active Waiting)
        # We are on the arc. Face the ball perfectly to present the thick dribbler shield.
        base_speed = 0.0
        target_angle = math.atan2(ball_pos[1] - curr_pos[1], ball_pos[0] - curr_pos[0])
    else:
        target_angle = math.atan2(dy, dx)
        
        if is_shot_incoming:
            # STATE 2: PANIC DIVE
            # A shot is fired. Ignore braking and throw the robot at the target.
            base_speed = DASH_SPEED
        else:
            # STATE 3: ARC TRACKING
            # Striker is dribbling. Smoothly slide along the arc.
            if dist_to_target > 0.1:
                base_speed = NORMAL_SPEED
            else:
                base_speed = NORMAL_SPEED * (dist_to_target / 0.1)
                base_speed = max(base_speed, 5.0) 

    # --- PD STEERING & KINEMATICS ---
    angle_error = target_angle - my_heading
    
    while angle_error > math.pi: angle_error -= 2 * math.pi
    while angle_error < -math.pi: angle_error += 2 * math.pi

    error_rate = angle_error - prev_error
    turn_amount = (Kp * angle_error) + (Kd * error_rate)
    prev_error = angle_error 

    # Forgiving pivot logic
    if abs(angle_error) < 0.5:
        speed_multiplier = 1.0
    else:
        speed_multiplier = max(0.0, 1.0 - (abs(angle_error) / (math.pi / 2)))
        
    current_base_speed = base_speed * speed_multiplier

    # Clamping and Motor Setup
    left_speed = max(min(current_base_speed - turn_amount, DASH_SPEED), -DASH_SPEED)
    right_speed = max(min(current_base_speed + turn_amount, DASH_SPEED), -DASH_SPEED)

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)