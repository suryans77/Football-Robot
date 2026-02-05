from controller import Robot
import cv2
import numpy as np
import math

# ================= INITIALIZATION =================
robot = Robot()
timestep = int(robot.getBasicTimeStep())

gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

camera = robot.getDevice("camera")
camera.enable(timestep)

ps0 = robot.getDevice("ps0")
ps7 = robot.getDevice("ps7")
ps0.enable(timestep)
ps7.enable(timestep)

left_motor  = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# ================= DEBUG SWITCH =================
DEBUG = True

# ================= PARAMETERS =================
GOAL_POS = [1.2, 0.0, 0.0]

OBSTACLES = [
    [0.3, 0.3, 0.0],
    [0.0, -0.1, 0.0],
    [0.4, 0.0, 0.0]
]

MAX_SPEED = 21.0
DRIBBLE_SPEED = 10.0
SEARCH_SPEED = 4.0
IN_POCKET_LIMIT = 3500
BALL_CLOSE_PIXELS = 150000

REPULSION_RADIUS = 0.5
K_ATTRACT = 1.5
K_REPEL = 0.5

ESCAPE_GAIN = 0.6
MIN_FORCE_THRESHOLD = 0.08
escape_timer = 0

LOWER_ORANGE = np.array([0, 150, 50])
UPPER_ORANGE = np.array([20, 255, 255])

# ================= FUNCTIONS =================
def get_robot_heading(compass_values):
    return math.atan2(compass_values[0], compass_values[1])

def calculate_apf_vector(curr_pos):
    global escape_timer

    dx_g = GOAL_POS[0] - curr_pos[0]
    dy_g = GOAL_POS[1] - curr_pos[1]
    dist_g = math.hypot(dx_g, dy_g)

    fx = (dx_g / max(0.01, dist_g)) * K_ATTRACT
    fy = (dy_g / max(0.01, dist_g)) * K_ATTRACT

    if DEBUG:
        print(f"\n[APF] Position: {curr_pos}")
        print(f"[APF] Goal Force: ({fx:.2f}, {fy:.2f}) DistGoal:{dist_g:.2f}")

    for i, obs in enumerate(OBSTACLES):
        dx_o = curr_pos[0] - obs[0]
        dy_o = curr_pos[1] - obs[1]
        dist_o = math.hypot(dx_o, dy_o)

        if dist_o < REPULSION_RADIUS:
            dir_x = dx_o / max(0.01, dist_o)
            dir_y = dy_o / max(0.01, dist_o)
            strength = K_REPEL * (REPULSION_RADIUS - dist_o) / REPULSION_RADIUS

            rep_x = dir_x * strength * 3.0
            rep_y = dir_y * strength * 3.0

            fx += rep_x
            fy += rep_y

            if DEBUG:
                print(f"[APF] Obstacle {i} Repel: ({rep_x:.2f}, {rep_y:.2f}) Dist:{dist_o:.2f}")

    force_mag = math.hypot(fx, fy)

    if force_mag < MIN_FORCE_THRESHOLD:
        escape_timer += 1
    else:
        escape_timer = 0

    if escape_timer > 10:
        random_angle = np.random.uniform(-math.pi, math.pi)
        fx += ESCAPE_GAIN * math.cos(random_angle)
        fy += ESCAPE_GAIN * math.sin(random_angle)
        if DEBUG:
            print("[APF] ESCAPE ACTIVATED")

    if DEBUG:
        print(f"[APF] Final Force: ({fx:.2f}, {fy:.2f}) | Mag:{force_mag:.2f}")

    return fx, fy

# ================= MAIN LOOP =================
state = "SEARCH"

while robot.step(timestep) != -1:

    full_gps = gps.getValues()
    curr_pos = [full_gps[0], full_gps[1]]
    heading = get_robot_heading(compass.getValues())

    image_data = camera.getImage()
    width, height = camera.getWidth(), camera.getHeight()
    frame = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
    hsv = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    moments = cv2.moments(mask)
    pixels_found = moments['m00']

    dist_l, dist_r = ps7.getValue(), ps0.getValue()
    ball_visible = pixels_found > 100
    is_in_pocket = (pixels_found > BALL_CLOSE_PIXELS)

    # ---- STATE DEBUG ----
    if DEBUG:
        print(f"\n===== STATE: {state} | BallPix:{pixels_found:.0f} | PS_L:{dist_l:.0f} PS_R:{dist_r:.0f} =====")

    if ball_visible and is_in_pocket:
        state = "NAVIGATE"
    elif ball_visible:
        state = "APPROACH"
    else:
        state = "SEARCH"

    # ---- CONTROL ----
    if state == "NAVIGATE":
        fx, fy = calculate_apf_vector(curr_pos)
        target_angle = math.atan2(fy, fx)

        angle_error = (target_angle - heading + math.pi) % (2*math.pi) - math.pi

        turn = max(min(angle_error * 8.0, 6), -6)
        forward = DRIBBLE_SPEED * (1 - min(abs(angle_error)/math.pi, 1))

        left_speed  = forward - turn
        right_speed = forward + turn

        if DEBUG:
            print(f"[NAV] Heading:{math.degrees(heading):.1f}° Target:{math.degrees(target_angle):.1f}° Error:{math.degrees(angle_error):.1f}°")
            print(f"[NAV] Speeds L:{left_speed:.2f} R:{right_speed:.2f}")

    elif state == "APPROACH":
        ball_x = int(moments['m10'] / moments['m00']) if pixels_found > 0 else width//2
        turn = ((ball_x - width/2) / (width/2)) * 8.0
        left_speed = 10.0 + turn
        right_speed = 10.0 - turn

    else:
        left_speed = SEARCH_SPEED
        right_speed = -SEARCH_SPEED

    left_speed = max(min(left_speed, MAX_SPEED), -MAX_SPEED)
    right_speed = max(min(right_speed, MAX_SPEED), -MAX_SPEED)
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
