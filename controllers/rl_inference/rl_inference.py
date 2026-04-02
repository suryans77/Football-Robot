from controller import Supervisor
from stable_baselines3 import PPO
import numpy as np
import sys

# ================= INITIALIZATION =================
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Setup Devices
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

ball_node = robot.getFromDef("Ball")
defenders = []
for i in range(1, 4):
    node = robot.getFromDef(f"defender{i}")
    if node: defenders.append(node)

# Physics Constants
MY_GOAL_CENTER = [2.0, 0.0]
MAX_SPEED = 20.0
DRIBBLER_OFFSET = 0.05  
DRIBBLER_PULL = 15.0  
v_curr = 0.0

# ================= LOAD THE BRAIN =================
print("Loading Neural Network...")
try:
    model = PPO.load("ppo_striker_brain")
    print("SUCCESS: Brain Loaded! The robot is autonomous.")
except Exception as e:
    print(f"ERROR: Could not find ppo_striker_brain.zip. Check your folder!")
    sys.exit(1)

# --- THE FIX: Let the sensors boot up for one frame ---
if robot.step(timestep) == -1:
    sys.exit(0)

# ================= MAIN MATCH LOOP =================
while True:
    
    # 1. READ THE SENSORS (Exactly how it was trained)
    pos = gps.getValues()
    comp = compass.getValues()
    heading = np.arctan2(comp[0], comp[1])
    
    est_linear_v = v_curr * 0.02 
    self_vx = est_linear_v * np.cos(heading)
    self_vy = est_linear_v * np.sin(heading)

    goal_dx = MY_GOAL_CENTER[0] - pos[0]
    goal_dy = MY_GOAL_CENTER[1] - pos[1]
    goal_dist = np.hypot(goal_dx, goal_dy)
    goal_angle = np.arctan2(goal_dy, goal_dx) - heading
    
    def_stats = []
    for d in defenders:
        d_pos = d.getPosition()
        d_vel = d.getVelocity() 
        d_dx = d_pos[0] - pos[0]
        d_dy = d_pos[1] - pos[1]
        d_dist = np.hypot(d_dx, d_dy)
        d_angle = np.arctan2(d_dy, d_dx) - heading
        rel_vx = d_vel[0] - self_vx
        rel_vy = d_vel[1] - self_vy
        def_stats.append({
            "dist": d_dist, "sin": np.sin(d_angle), "cos": np.cos(d_angle),
            "rel_vx": rel_vx, "rel_vy": rel_vy
        })
        
    def_stats.sort(key=lambda x: x["dist"])
    d1 = def_stats[0] if len(def_stats) > 0 else {"dist": 10.0, "sin": 0.0, "cos": 0.0, "rel_vx": 0.0, "rel_vy": 0.0}
    d2 = def_stats[1] if len(def_stats) > 1 else {"dist": 10.0, "sin": 0.0, "cos": 0.0, "rel_vx": 0.0, "rel_vy": 0.0}

    # Pack the 15 inputs
    obs = np.array([
        goal_dist, np.sin(goal_angle), np.cos(goal_angle),
        self_vx, self_vy,
        d1["dist"], d1["sin"], d1["cos"], d1["rel_vx"], d1["rel_vy"],
        d2["dist"], d2["sin"], d2["cos"], d2["rel_vx"], d2["rel_vy"]
    ], dtype=np.float32)

    # 2. ASK THE AI WHAT TO DO
    # deterministic=True tells the AI to stop exploring and only use its absolute best moves
    action, _states = model.predict(obs, deterministic=True)

    # 3. APPLY MOTOR MATH
    forward_speed = action[0] * MAX_SPEED        
    turn_speed = action[1] * (MAX_SPEED * 0.5)   
    
    left_target = max(min(forward_speed - turn_speed, 20.0), -20.0)
    right_target = max(min(forward_speed + turn_speed, 20.0), -20.0)
    
    left_motor.setVelocity(left_target)
    right_motor.setVelocity(right_target)
    v_curr = (left_target + right_target) / 2.0
    
    # 4. ACTION REPEAT & PHYSICS LOOP (Hold the move for 8 frames)
    for _ in range(8):
        if robot.step(timestep) == -1:
            sys.exit(0)
            
        # Maintain Dribbler Grip during the frames
        pos = gps.getValues()
        comp = compass.getValues()
        heading = np.arctan2(comp[0], comp[1])
        ball_pos = ball_node.getPosition()
        
        dx = ball_pos[0] - pos[0]
        dy = ball_pos[1] - pos[1]
        dist_to_ball = np.hypot(dx, dy)
        angle_to_ball = np.arctan2(dy, dx)
        
        angle_diff = angle_to_ball - heading
        while angle_diff > np.pi: angle_diff -= 2 * np.pi
        while angle_diff < -np.pi: angle_diff += 2 * np.pi
        
        if dist_to_ball < 0.15 and abs(angle_diff) < 0.8:
            notch_x = pos[0] + np.cos(heading) * DRIBBLER_OFFSET
            notch_y = pos[1] + np.sin(heading) * DRIBBLER_OFFSET
            pull_vx = (notch_x - ball_pos[0]) * DRIBBLER_PULL
            pull_vy = (notch_y - ball_pos[1]) * DRIBBLER_PULL
            ball_node.setVelocity([pull_vx, pull_vy, 0.0, 0.0, 0.0, 0.0])