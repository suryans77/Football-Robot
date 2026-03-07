from controller import Supervisor
import math

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
GOAL_CENTER    = [2.0, 0.0]
GOAL_Y_LIMIT   = 0.25
KEEPER_RADIUS  = 0.1   # Pushed out further — gives more reaction space
SHOT_THRESHOLD = 1.5    # Lower = reacts to slower shots too

# --- MOVEMENT PARAMETERS ---
SPEED = 20.0


# Steering
Kp = 8.0
Kd = 1.5
prev_error = 0.0

# Smoothing
ACCEL        = 0.35     # Higher than defender — keeper needs snappier response
base_speed_curr = 0.0
turn_curr       = 0.0

def predict_intercept(ball_pos, ball_vel, goal_x, steps=15, dt=0.032):
    """
    Walk the ball forward in time. Return the Y position where it
    crosses the keeper's X plane (goal_x). Falls back to current ball Y.
    """
    bx, by = ball_pos[0], ball_pos[1]
    vx, vy = ball_vel[0], ball_vel[1]
    for _ in range(steps):
        bx += vx * dt
        by += vy * dt
        if vx > 0 and bx >= goal_x:   # Ball crossed the goal line
            return by
        if vx <= 0:                    # Ball moving away — no threat
            return None
    return None

while robot.step(timestep) != -1:
    if ball_node is None:
        continue

    curr_pos    = gps.getValues()
    ball_pos    = ball_node.getPosition()
    compass_val = compass.getValues()
    my_heading  = math.atan2(compass_val[0], compass_val[1])

    ball_vel    = ball_node.getVelocity()
    ball_speed  = math.hypot(ball_vel[0], ball_vel[1])

    # Shot = fast AND moving toward our goal (positive X toward goal)
    is_shot_incoming = ball_speed > SHOT_THRESHOLD and ball_vel[0] > 0.2

    # --- STRATEGY ---
    if is_shot_incoming:
        # STATE 1: DIVE — intercept predicted ball position, not current position
        intercept_y = predict_intercept(ball_pos, ball_vel, GOAL_CENTER[0])

        if intercept_y is not None:
            # Dive to where the ball will cross the line
            target_x = GOAL_CENTER[0] - 0.05   # Just in front of goal line
            target_y = max(min(intercept_y, GOAL_Y_LIMIT), -GOAL_Y_LIMIT)
        else:
            # Ball not heading to goal — hold arc position
            is_shot_incoming = False

    if not is_shot_incoming:
        # STATE 2: ARC TRACKING — slide along arc facing the ball
        vec_x = ball_pos[0] - GOAL_CENTER[0]
        vec_y = ball_pos[1] - GOAL_CENTER[1]
        dist_goal_to_ball = math.hypot(vec_x, vec_y)

        if dist_goal_to_ball < 0.001:
            unit_x, unit_y = -1.0, 0.0
        else:
            unit_x = vec_x / dist_goal_to_ball
            unit_y = vec_y / dist_goal_to_ball

        target_x = GOAL_CENTER[0] + unit_x * KEEPER_RADIUS
        target_y = GOAL_CENTER[1] + unit_y * KEEPER_RADIUS
        target_y = max(min(target_y, GOAL_Y_LIMIT), -GOAL_Y_LIMIT)

    # --- NAVIGATION ---
    dx = target_x - curr_pos[0]
    dy = target_y - curr_pos[1]
    dist_to_target = math.hypot(dx, dy)

    ball_angle         = math.atan2(ball_pos[1] - curr_pos[1], ball_pos[0] - curr_pos[0])
    target_drive_angle = math.atan2(dy, dx)

    if not is_shot_incoming and dist_to_target < 0.04:
        # On the arc — face ball, stop
        base_speed   = 0.0
        target_angle = ball_angle
    else:
        if is_shot_incoming:
            base_speed   = SPEED
            target_angle = target_drive_angle  # Full speed, steer to intercept
        else:
            # Blend heading: face target while moving, rotate to ball on arrival
            blend        = min(1.0, dist_to_target / 0.2)
            target_angle = ball_angle + blend * (target_drive_angle - ball_angle)

            if dist_to_target > 0.1:
                base_speed = SPEED
            else:
                base_speed = max(4.0, SPEED * (dist_to_target / 0.1))

    # --- PD STEERING ---
    angle_error = target_angle - my_heading
    while angle_error >  math.pi: angle_error -= 2 * math.pi
    while angle_error < -math.pi: angle_error += 2 * math.pi

    error_rate   = angle_error - prev_error
    turn_amount  = (Kp * angle_error) + (Kd * error_rate)
    turn_amount  = max(min(turn_amount, 12.0), -12.0)
    prev_error   = angle_error

    if abs(angle_error) < 0.5:
        speed_multiplier = 1.0
    else:
        speed_multiplier = max(0.0, 1.0 - (abs(angle_error) / (math.pi / 2)))

    current_base_speed = base_speed * speed_multiplier

    # Smoothing — skip during dive for instant reaction
    if is_shot_incoming:
        base_speed_curr = current_base_speed
        turn_curr       = turn_amount
    else:
        base_speed_curr += (current_base_speed - base_speed_curr) * ACCEL
        turn_curr       += (turn_amount - turn_curr) * ACCEL

    left_speed  = max(min(base_speed_curr - turn_curr, SPEED), -SPEED)
    right_speed = max(min(base_speed_curr + turn_curr, SPEED), -SPEED)

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)