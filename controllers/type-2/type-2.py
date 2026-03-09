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
DEFENSE_RADIUS = 0.6  # Distance from goal center to position the defender (tune as needed) 

# --- MOVEMENT PARAMETERS ---
MAX_SPEED = 18.0       # Upgraded to your custom e-puck max speed
MAX_TURN = 9.0       # Max turn speed (tune as needed)

# PD Steering Constants
Kp = 4.0
Kd = 1.2               # Added damping to prevent wobbling
prev_error = 0.0

ACCEL = 0.2   # Lower = slower reaction
base_speed_curr = 0.0
turn_curr = 0.0

while robot.step(timestep) != -1:
    if ball_node is None: continue

    # Get Sensor Data
    curr_pos = gps.getValues()
    ball_pos = ball_node.getPosition()
    compass_val = compass.getValues()
    my_heading = math.atan2(compass_val[0], compass_val[1])

    # --- STRATEGY: CALCULATE IDEAL BLOCKING SPOT ---
    vec_x = ball_pos[0] - MY_GOAL_CENTER[0]
    vec_y = ball_pos[1] - MY_GOAL_CENTER[1]
    dist_goal_to_ball = math.hypot(vec_x, vec_y)
    
    if dist_goal_to_ball < 0.001:
        unit_x, unit_y = 1.0, 0.0
    else:
        unit_x = vec_x / dist_goal_to_ball
        unit_y = vec_y / dist_goal_to_ball
    
    target_x = MY_GOAL_CENTER[0] + (unit_x * DEFENSE_RADIUS)
    target_y = MY_GOAL_CENTER[1] + (unit_y * DEFENSE_RADIUS)

    # --- NAVIGATION & KINEMATICS ---
    dx = target_x - curr_pos[0]
    dy = target_y - curr_pos[1]
    dist_to_target = math.hypot(dx, dy)
    
    # 1. Determine Target Angle and Base Speed
    if dist_to_target < 0.05:
        # ACTIVE WAITING: On the spot, face the ball.
        base_speed = 0.0
        target_angle = math.atan2(ball_pos[1] - curr_pos[1], ball_pos[0] - curr_pos[0])
    else:
        # DRIVING: Uncap the speed!
        target_angle = math.atan2(dy, dx)
        
        # LATE BRAKING: Go absolute MAX_SPEED until we are 15cm away, then brake hard.
        if dist_to_target > 0.15:
            base_speed = MAX_SPEED
        else:
            base_speed = MAX_SPEED * (dist_to_target / 0.15)
            # Ensure it doesn't drop so low that it stalls before reaching the point
            base_speed = max(base_speed, 10.0) 

    # 2. PD Steering
    angle_error = target_angle - my_heading
    while angle_error > math.pi: angle_error -= 2 * math.pi
    while angle_error < -math.pi: angle_error += 2 * math.pi

    error_rate = angle_error - prev_error
    turn_amount = (Kp * angle_error) + (Kd * error_rate)
    turn_amount = max(min(turn_amount, MAX_TURN), -MAX_TURN)  # Clamp turn amount to max turn speed
    prev_error = angle_error 
    
    # 3. FORGIVING PIVOT LOGIC
    # If the error is small (under ~28 degrees / 0.5 rads), don't slow down at all.
    # This allows it to smoothly ride the arc without stuttering.
    if abs(angle_error) < 0.5:
        speed_multiplier = 1.0
    else:
        speed_multiplier = max(0.0, 1.0 - (abs(angle_error) / (math.pi / 2)))
        
    current_base_speed = base_speed * speed_multiplier
    
    # 4. Motor Mixing & Clamping
    # --- REACTION SPEED NERF (SMOOTHING) ---
    base_speed_curr += (current_base_speed - base_speed_curr) * ACCEL
    turn_curr += (turn_amount - turn_curr) * ACCEL

    left_speed = base_speed_curr - turn_curr
    right_speed = base_speed_curr + turn_curr
    
    left_speed = max(min(left_speed, MAX_SPEED), -MAX_SPEED)
    right_speed = max(min(right_speed, MAX_SPEED), -MAX_SPEED)

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)