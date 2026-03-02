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

# --- MOVEMENT PARAMETERS ---
MAX_SPEED = 15.0
JOCKEY_DIS = 0.5              # Distance to switch from Sprint to Controlled Ram

# Steering Constants
# PD Steering Constants
Kp = 8.0  # Your original K value
Kd = 2.0  # The new damping value (start small and tune!)

prev_error = 0.0 # Initialize outside the loop

while robot.step(timestep) != -1:
    if ball_node is None: continue

    # Get Data
    curr_pos = gps.getValues()
    ball_pos = ball_node.getPosition()
    compass_val = compass.getValues()
    
    # Calculate heading
    my_heading = math.atan2(compass_val[0], compass_val[1])

    # --- BEHAVIOR LOGIC: RELENTLESS PURSUIT ---
    # Always calculate vector to the ball
    dx = ball_pos[0] - curr_pos[0]
    dy = ball_pos[1] - curr_pos[1]
    dist_to_ball = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)
    
    # TACTICAL RAMMING: Scale speed down slightly for steering control, but never stop.
    if dist_to_ball > JOCKEY_DIS:
        base_speed = MAX_SPEED
    else:
        # Scale down to a minimum of 50% max speed. 
        # Ensures a hard hit, but keeps enough traction to track evasions.
        speed_ratio = dist_to_ball / JOCKEY_DIS
        base_speed = max(MAX_SPEED * 0.5, MAX_SPEED * speed_ratio)

    # --- STEERING CONTROL (PID & KINEMATICS) ---
    angle_error = target_angle - my_heading
    
    # Normalize angle error between -pi and pi
    while angle_error > math.pi: angle_error -= 2 * math.pi
    while angle_error < -math.pi: angle_error += 2 * math.pi

    error_rate = angle_error - prev_error
    turn_amount = (Kp * angle_error) + (Kd * error_rate)
    
    # Save the current error for the next frame
    prev_error = angle_error

    # PIVOT LOGIC: Slow down forward speed if we need to turn sharply.
    # If the ball is behind the robot, speed_multiplier drops to 0, causing it to spin in place first.
    speed_multiplier = max(0.0, 1.0 - (abs(angle_error) / (math.pi / 2))) 
    current_base_speed = base_speed * speed_multiplier

    left_speed = current_base_speed - turn_amount
    right_speed = current_base_speed + turn_amount
    
    # Clamp velocities to the motor limits to prevent Webots warnings/errors
    left_speed = max(min(left_speed, MAX_SPEED), -MAX_SPEED)
    right_speed = max(min(right_speed, MAX_SPEED), -MAX_SPEED)

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)