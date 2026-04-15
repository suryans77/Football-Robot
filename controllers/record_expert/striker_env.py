import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from controller import Supervisor
from stable_baselines3 import PPO

class StrikerRLEnv(gym.Env):
    def __init__(self):
        super(StrikerRLEnv, self).__init__()
        
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # --- CHANGED: Action Space is now just [Gas Pedal, Steering Wheel] ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.timestep)
        
        self.striker_node = self.robot.getSelf()
        self.robot_trans_field = self.striker_node.getField("translation")
        self.robot_rot_field = self.striker_node.getField("rotation")
        
        self.ball_node = self.robot.getFromDef("Ball")
        self.ball_trans_field = self.ball_node.getField("translation")
        
        self.init_robot_pos = self.robot_trans_field.getSFVec3f()
        self.init_robot_rot = self.robot_rot_field.getSFRotation()
        self.init_ball_pos = self.ball_trans_field.getSFVec3f()
        
        self.defenders = []
        for i in range(1, 4):
            node = self.robot.getFromDef(f"defender{i}")
            if node: self.defenders.append(node)
            
        self.MY_GOAL_CENTER = [2.0, 0.0]
        
        self.MAX_SPEED = 15.0
        self.TURN_BIAS = 5.0    
        self.ACCEL = 0.2        
        self.DRIBBLER_OFFSET = 0.05  
        self.DRIBBLER_PULL = 15.0   
        
        self.v_curr = 0.0
        self.w_curr = 0.0 
        self.step_count = 0
        self.prev_ball_dist = 0.0 
        
        self.beaten_defenders = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.robot_trans_field.setSFVec3f(self.init_robot_pos)
        self.robot_rot_field.setSFRotation(self.init_robot_rot)
        
        curriculum_ball_pos = [self.init_robot_pos[0] + 0.05, self.init_robot_pos[1], self.init_ball_pos[2]]
        self.ball_trans_field.setSFVec3f(curriculum_ball_pos)
        
        self.robot.simulationResetPhysics()
        self.robot.step(self.timestep)
        
        self.v_curr = 0.0
        self.w_curr = 0.0
        self.step_count = 0
        self.prev_ball_dist = np.hypot(self.MY_GOAL_CENTER[0] - curriculum_ball_pos[0], self.MY_GOAL_CENTER[1] - curriculum_ball_pos[1])
        
        self.beaten_defenders.clear()
        
        return self._get_obs(), {}

    def _get_obs(self):
        pos = self.gps.getValues()
        comp = self.compass.getValues()
        heading = np.arctan2(comp[0], comp[1])
        
        est_linear_v = self.v_curr * 0.02 
        self_vx = est_linear_v * np.cos(heading)
        self_vy = est_linear_v * np.sin(heading)

        goal_dx = self.MY_GOAL_CENTER[0] - pos[0]
        goal_dy = self.MY_GOAL_CENTER[1] - pos[1]
        goal_dist = np.hypot(goal_dx, goal_dy)
        goal_angle = np.arctan2(goal_dy, goal_dx) - heading
        
        def_stats = []
        for d in self.defenders:
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

        return np.array([
            goal_dist, np.sin(goal_angle), np.cos(goal_angle),
            self_vx, self_vy,
            d1["dist"], d1["sin"], d1["cos"], d1["rel_vx"], d1["rel_vy"],
            d2["dist"], d2["sin"], d2["cos"], d2["rel_vx"], d2["rel_vy"]
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        
        v_target = action[0] * self.MAX_SPEED      
        w_target = action[1] * self.TURN_BIAS      
        
        for _ in range(8):
            self.v_curr += (v_target - self.v_curr) * self.ACCEL
            self.w_curr += (w_target - self.w_curr) * self.ACCEL
            
            left_speed = max(min(self.v_curr - self.w_curr, 20.0), -20.0)
            right_speed = max(min(self.v_curr + self.w_curr, 20.0), -20.0)
            
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)

            pos = self.gps.getValues()
            comp = self.compass.getValues()
            heading = np.arctan2(comp[0], comp[1])
            ball_pos = self.ball_node.getPosition()
            
            dx = ball_pos[0] - pos[0]
            dy = ball_pos[1] - pos[1]
            dist_to_ball = np.hypot(dx, dy)
            angle_to_ball = np.arctan2(dy, dx)
            
            angle_diff = angle_to_ball - heading
            while angle_diff > np.pi: angle_diff -= 2 * np.pi
            while angle_diff < -np.pi: angle_diff += 2 * np.pi
            
            # --- CHANGED: Dribbler purely locks the ball, no shooting check ---
            is_dribbling = False
            if dist_to_ball < 0.1 and abs(angle_diff) < 0.5:
                is_dribbling = True
                
            if is_dribbling:
                notch_x = pos[0] + np.cos(heading) * self.DRIBBLER_OFFSET
                notch_y = pos[1] + np.sin(heading) * self.DRIBBLER_OFFSET
                pull_vx = (notch_x - ball_pos[0]) * self.DRIBBLER_PULL
                pull_vy = (notch_y - ball_pos[1]) * self.DRIBBLER_PULL
                self.ball_node.setVelocity([pull_vx, pull_vy, 0.0, 0.0, 0.0, 0.0])

            if self.robot.step(self.timestep) == -1:
                sys.exit(0)

        pos = self.gps.getValues()
        ball_pos = self.ball_node.getPosition()
        dist_ball_to_goal = np.hypot(self.MY_GOAL_CENTER[0] - ball_pos[0], self.MY_GOAL_CENTER[1] - ball_pos[1])
        
        reward = 0.0
        done = False
        
        progress = self.prev_ball_dist - dist_ball_to_goal
        reward += progress * 100.0  
        self.prev_ball_dist = dist_ball_to_goal

        # Inside your step() function, check distance to all defenders
        for defender in self.defenders:
            def_pos = defender.getPosition()
            dist_to_def = np.hypot(def_pos[0] - pos[0], def_pos[1] - pos[1])
            
            # If the robot's physical body hits the defender's physical body
            if dist_to_def < 0.2: 
                reward -= 5.0 # Zap the agent every frame it stays in contact

        for idx, defender in enumerate(self.defenders):
            if idx not in self.beaten_defenders:
                def_pos = defender.getPosition()
                if ball_pos[0] > def_pos[0] + 0.15:
                    reward += 50.0  
                    self.beaten_defenders.add(idx) 

        if dist_ball_to_goal < 0.3:
            reward += 100.0 
            done = True
        elif pos[0] > 2.5 or pos[0] < -2.5 or pos[1] > 1.5 or pos[1] < -1.5:
            reward -= 10.0 
            done = True
        elif self.step_count > 250: 
            done = True

        return self._get_obs(), reward, done, False, {}