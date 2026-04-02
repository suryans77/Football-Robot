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
MY_GOAL_CENTER = [2.0, 0.0]  
MARKING_DIST = 0.3     # The distance to maintain from the attacker in the 1.0-1.5m zone

# --- MOVEMENT PARAMETERS ---
MAX_SPEED = 18.0       
MAX_TURN = 6.0        

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

    if (dist_goal_to_ball < 1.5 and ball_pos[0] > curr_pos[0]) or dist_goal_to_ball < 1.0:
        target_x = ball_pos[0]
        target_y = ball_pos[1]
        is_chasing = True

    elif dist_goal_to_ball < 2.0:
        # STATE 2: LATERAL CONTAINMENT (The Wall)
        # Anchor the depth 2.0m away from the goal center.
        # Purely mirror the Y movement to slide laterally and block the lane.
        target_x = ball_pos[0] + 0.3
        target_y = max(min(ball_pos[1], 1.3), -1.3)

    elif dist_goal_to_ball <= 3.0:
        # STATE 3: MAN-MARKING 
        # Stay exactly MARKING_DIST away from the ball.
        target_x = ball_pos[0] + (dir_bg_x * MARKING_DIST)
        target_y = ball_pos[1] + (dir_bg_y * MARKING_DIST)

    else:
        # STATE 4: HOLD THE LINE 
        target_x = MY_GOAL_CENTER[0] - (dir_bg_x * 2.5)
        target_y = MY_GOAL_CENTER[1] - (dir_bg_y * 2.5)

    # --- NAVIGATION & KINEMATICS ---
    dx = target_x - curr_pos[0]
    dy = target_y - curr_pos[1]
    dist_to_target = math.hypot(dx, dy)

    angle_to_spot = math.atan2(dy, dx)
    angle_to_ball = math.atan2(ball_pos[1] - curr_pos[1], ball_pos[0] - curr_pos[0])

    # === HYBRID BEHAVIOR CORE LOGIC ===
    if dist_to_target > 0.15 or is_chasing:
        # 1. RECOVERY SPRINT: We are out of position or tackling!
        # Stop looking at the ball. Look exactly at the waypoint and drive there fast.
        target_angle = angle_to_spot
        base_speed = MAX_SPEED
            
    else:
        # 2. JOCKEY MODE: We are in the pocket!
        # Lock eyes on the ball and use forward/reverse to hold the gap.
        target_angle = angle_to_ball
        
        if dist_to_target < 0.05:
            base_speed = 0.0
        else:
            speed_mag = MAX_SPEED * (dist_to_target / 0.15)
            speed_mag = max(speed_mag, 5.0)
            
            # REVERSE GEAR CHECK
            angle_diff = angle_to_spot - my_heading
            while angle_diff > math.pi: angle_diff -= 2 * math.pi
            while angle_diff < -math.pi: angle_diff += 2 * math.pi
            
            if abs(angle_diff) > (math.pi / 2):
                base_speed = -speed_mag # Target is behind, backpedal!
            else:
                base_speed = speed_mag  # Target is in front, step up!

    # --- PD STEERING ---
    angle_error = target_angle - my_heading
    while angle_error > math.pi: angle_error -= 2 * math.pi
    while angle_error < -math.pi: angle_error += 2 * math.pi

    error_rate = angle_error - prev_error
    turn_amount = (Kp * angle_error) + (Kd * error_rate)
    turn_amount = max(min(turn_amount, MAX_TURN), -MAX_TURN)  
    prev_error = angle_error 
    
    # Forgiving Pivot Logic (Brakes if we need to make a sharp turn)
    if abs(angle_error) < 0.5:
        speed_multiplier = 1.0
    else:
        speed_multiplier = max(0.0, 1.0 - (abs(angle_error) / (math.pi / 2)))
        
    current_base_speed = base_speed * speed_multiplier
    
    # --- MOTOR MIXING & SMOOTHING ---
    base_speed_curr += (current_base_speed - base_speed_curr) * ACCEL
    turn_curr += (turn_amount - turn_curr) * ACCEL

    # Because base_speed correctly flips to negative in the logic above, 
    # standard mixing works perfectly without breaking the steering!
    left_speed = base_speed_curr - turn_curr
    right_speed = base_speed_curr + turn_curr
    
    # Motor Clamp
    left_speed = max(min(left_speed, MAX_SPEED), -MAX_SPEED)
    right_speed = max(min(right_speed, MAX_SPEED), -MAX_SPEED)

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)