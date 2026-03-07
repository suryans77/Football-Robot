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
MY_GOAL_CENTER = [1.5, 0.0]  
MARKING_DIST = 0.3     # The distance to maintain from the attacker in the 1.0-1.5m zone

# --- MOVEMENT PARAMETERS ---
MAX_SPEED = 15.0       
MAX_TURN = 10.0        

# PD Steering & Smoothing Constants
Kp = 3.0
Kd = 1.0               
ACCEL = 0.2   
prev_error = 0.0
base_speed_curr = 0.0
turn_curr = 0.0

while robot.step(timestep) != -1:
    if ball_node is None: continue

    # Get Sensor Data
    curr_pos = gps.getValues()
    ball_pos = ball_node.getPosition()
    compass_val = compass.getValues()
    my_heading = math.atan2(compass_val[0], compass_val[1])

    # --- STRATEGY: CALCULATE DISTANCES AND VECTORS ---
    # Vector from Ball to Goal
    vec_bg_x = MY_GOAL_CENTER[0] - ball_pos[0]
    vec_bg_y = MY_GOAL_CENTER[1] - ball_pos[1]
    dist_goal_to_ball = math.hypot(vec_bg_x, vec_bg_y)
    
    # Normalize the vector (Direction from Ball pointing towards the Goal)
    if dist_goal_to_ball < 0.001:
        dir_bg_x, dir_bg_y = 1.0, 0.0
    else:
        dir_bg_x = vec_bg_x / dist_goal_to_ball
        dir_bg_y = vec_bg_y / dist_goal_to_ball

    # --- BEHAVIOR STATE MACHINE ---
    is_chasing = False

    if dist_goal_to_ball < 1.25:
        # STATE 1: PURE CHASING (Breached the 1.0m line)
        # Abandon spacing, target the ball directly for a tackle.
        target_x = ball_pos[0]
        target_y = ball_pos[1]
        is_chasing = True

    elif dist_goal_to_ball <= 2.5:
        # STATE 2: MAN-MARKING (Between 1.0m and 1.5m)
        # Stay exactly MARKING_DIST away from the ball, on the direct line to the goal.
        target_x = ball_pos[0] + (dir_bg_x * MARKING_DIST)
        target_y = ball_pos[1] + (dir_bg_y * MARKING_DIST)

    else:
        # STATE 3: HOLD THE LINE (Further than 1.5m)
        # Wait at the edge of the 1.5m defensive zone, staying between the ball and goal.
        target_x = MY_GOAL_CENTER[0] - (dir_bg_x * 1.5)
        target_y = MY_GOAL_CENTER[1] - (dir_bg_y * 1.5)

    # --- NAVIGATION & KINEMATICS ---
    dx = target_x - curr_pos[0]
    dy = target_y - curr_pos[1]
    dist_to_target = math.hypot(dx, dy)

    # How misaligned is our target vs where we're facing the ball?
    # When target is far off-axis, steer toward TARGET first, then ball when close
    ball_angle = math.atan2(ball_pos[1] - curr_pos[1], ball_pos[0] - curr_pos[0])
    target_drive_angle = math.atan2(dy, dx)

    # Blend: prioritize driving to target when far, face ball when arrived
    # blend=1.0 → steer to target, blend=0.0 → face ball
    if dist_to_target > 0.05:
        blend = min(1.0, dist_to_target / 0.3)  # tune 0.3 to taste
    else:
        blend = 0.0

    target_angle = ball_angle + blend * (target_drive_angle - ball_angle)

    # Normalize target_angle
    if target_angle > math.pi: target_angle -= 2 * math.pi
    if target_angle < -math.pi: target_angle += 2 * math.pi

    # Speed: no longer suppressed by projection, just use dist directly
    if not is_chasing and dist_to_target < 0.05:
        base_speed = 0.0
    elif dist_to_target > 0.15:
        base_speed = MAX_SPEED
    else:
        if is_chasing:
            base_speed = max(MAX_SPEED * 0.5, MAX_SPEED * (dist_to_target / 0.15))
        else:
            base_speed = max(5.0, MAX_SPEED * (dist_to_target / 0.15))

    # Remove the forward_component suppression entirely
    # The blend above handles steering, speed is purely distance-based

    # 2. PD Steering
    angle_error = target_angle - my_heading
    if angle_error > math.pi: angle_error -= 2 * math.pi
    if angle_error < -math.pi: angle_error += 2 * math.pi

    error_rate = angle_error - prev_error
    turn_amount = (Kp * angle_error) + (Kd * error_rate)
    turn_amount = max(min(turn_amount, MAX_TURN), -MAX_TURN)  
    prev_error = angle_error 
    
    # 3. Forgiving Pivot Logic
    if abs(angle_error) < 0.5:
        speed_multiplier = 1.0
    else:
        speed_multiplier = max(0.0, 1.0 - (abs(angle_error) / (math.pi / 2)))
        
    current_base_speed = base_speed * speed_multiplier
    
    # 4. Motor Mixing & Smoothing
    base_speed_curr += (current_base_speed - base_speed_curr) * ACCEL
    turn_curr += (turn_amount - turn_curr) * ACCEL

    left_speed = base_speed_curr - turn_curr
    right_speed = base_speed_curr + turn_curr
    
    # Motor Clamp
    left_speed = max(min(left_speed, MAX_SPEED), -MAX_SPEED)
    right_speed = max(min(right_speed, MAX_SPEED), -MAX_SPEED)

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)