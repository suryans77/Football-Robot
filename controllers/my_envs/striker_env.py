"""
StrikerRLEnv — unified environment for benchmarking
=====================================================

Compatible with:
  BC        — continuous Box, no env interaction during training
  PPO       — continuous Box
  GAIL      — continuous Box
  IQL       — continuous Box (offline, no env interaction during training)
  IQ-Learn  — continuous Box (pass index_to_action result to step)
  DQN       — Discrete(9)  via  StrikerRLEnv(discrete=True)

Usage
------
    env = StrikerRLEnv()               # continuous — BC, PPO, GAIL, IQL, IQ-L
    env = StrikerRLEnv(discrete=True)  # discrete   — DQN

Action conventions
-------------------
  Continuous: action = [gas, steer]  both in [-1, 1]
  Discrete  : action = int 0-8, mapped via ACTION_SET below

Observation (15 dims)
----------------------
  [0]     goal_dist              — distance to goal centre
  [1]     sin(goal_angle)        — goal direction relative to heading
  [2]     cos(goal_angle)
  [3]     self_vx                — estimated robot velocity x
  [4]     self_vy                — estimated robot velocity y
  [5-9]   nearest defender       — dist, sin, cos, rel_vx, rel_vy
  [10-14] second nearest defender
  
Reward function (refined)
--------------------------
  Progress          :  clip(progress, -0.5, 0.5) * 20     — ball moving toward goal
  Defender beaten   :  +50 per defender passed             — key sub-goal
  Goal scored       :  +100,  terminated=True              — episode win
  Collision         :  -10 per step in contact              — soft continuous penalty
  Out of bounds     :  -15,   terminated=True              — hard boundary
  Time step tax     :  -0.2 per step                       — encourage efficiency
  Timeout           :  truncated=True  (NOT terminated)    — Bellman bootstrap kept
"""

import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from controller import Supervisor

# ──────────────────────────────────────────────────────────────────────────────
# Discrete action set  (shared between discrete and continuous modes)
# ──────────────────────────────────────────────────────────────────────────────

ACTION_SET = np.array([
    [-1., -1.], [-1.,  0.], [-1., +1.],
    [ 0., -1.], [ 0.,  0.], [ 0., +1.],
    [+1., -1.], [+1.,  0.], [+1., +1.],
], dtype=np.float32)


class StrikerRLEnv(gym.Env):

    def __init__(self, discrete: bool = False):
        super().__init__()

        self.discrete = discrete
        self.robot    = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        # ── action space ─────────────────────────────────────────────
        if discrete:
            # DQN — integer index 0-8
            self.action_space = spaces.Discrete(9)
        else:
            # PPO, BC, GAIL, IQL, IQ-Learn — continuous [gas, steer]
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        # ── motors ───────────────────────────────────────────────────
        self.left_motor  = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # ── sensors ───────────────────────────────────────────────────
        self.gps     = self.robot.getDevice("gps")
        self.compass = self.robot.getDevice("compass")
        self.gps.enable(self.timestep)
        self.compass.enable(self.timestep)

        # ── scene nodes ───────────────────────────────────────────────
        self.striker_node       = self.robot.getSelf()
        self.robot_trans_field  = self.striker_node.getField("translation")
        self.robot_rot_field    = self.striker_node.getField("rotation")

        self.ball_node        = self.robot.getFromDef("Ball")
        self.ball_trans_field = self.ball_node.getField("translation")

        # Save initial positions for reset
        self.init_robot_pos = self.robot_trans_field.getSFVec3f()
        self.init_robot_rot = self.robot_rot_field.getSFRotation()
        self.init_ball_pos  = self.ball_trans_field.getSFVec3f()

        # Defenders — tolerate missing nodes
        self.defenders = []
        for i in range(1, 3):
            node = self.robot.getFromDef(f"defender{i}")
            if node:
                self.defenders.append(node)

        # ── parameters ───────────────────────────────────────────────
        self.MY_GOAL_CENTER  = [2.0, 0.0]
        self.MAX_SPEED       = 15.0
        self.TURN_BIAS       = 5.0
        self.ACCEL           = 0.2
        self.DRIBBLER_OFFSET = 0.05
        self.DRIBBLER_PULL   = 15.0
        self.INNER_STEPS     = 8     # physics steps per RL step

        # ── runtime state ─────────────────────────────────────────────
        self.v_curr          = 0.0
        self.w_curr          = 0.0
        self.step_count      = 0
        self.prev_ball_dist  = 0.0
        self.beaten_defenders = set()

    # ──────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Teleport robot and ball back to start — no simulationReset()
        # (simulationReset kills the controller process mid-training)
        self.robot_trans_field.setSFVec3f(self.init_robot_pos)
        self.robot_rot_field.setSFRotation(self.init_robot_rot)

        # Place ball just ahead of robot so it starts in possession
        curriculum_ball_pos = [
            self.init_robot_pos[0] + 0.05,
            self.init_robot_pos[1],
            self.init_ball_pos[2],
        ]
        self.ball_trans_field.setSFVec3f(curriculum_ball_pos)
        self.ball_node.setVelocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.robot.simulationResetPhysics()  # safe — only resets velocities
        self.robot.step(self.timestep)

        self.v_curr          = 0.0
        self.w_curr          = 0.0
        self.step_count      = 0
        self.beaten_defenders = set()
        self.prev_ball_dist  = np.hypot(
            self.MY_GOAL_CENTER[0] - curriculum_ball_pos[0],
            self.MY_GOAL_CENTER[1] - curriculum_ball_pos[1],
        )

        return self._get_obs(), {}

    # ──────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        pos     = self.gps.getValues()
        comp    = self.compass.getValues()
        heading = np.arctan2(comp[0], comp[1])

        # Estimated world-frame velocity from internal state
        est_v  = self.v_curr * 0.02
        self_vx = est_v * np.cos(heading)
        self_vy = est_v * np.sin(heading)

        # Goal direction
        gdx        = self.MY_GOAL_CENTER[0] - pos[0]
        gdy        = self.MY_GOAL_CENTER[1] - pos[1]
        goal_dist  = np.hypot(gdx, gdy)
        goal_angle = np.arctan2(gdy, gdx) - heading

        # Defender features — sorted by proximity so nearest is always [5-9]
        def_feats = []
        for d in self.defenders:
            dp     = d.getPosition()
            dv     = d.getVelocity()
            ddx    = dp[0] - pos[0]
            ddy    = dp[1] - pos[1]
            ddist  = np.hypot(ddx, ddy)
            dangle = np.arctan2(ddy, ddx) - heading
            def_feats.append({
                "dist":   ddist,
                "sin":    np.sin(dangle),
                "cos":    np.cos(dangle),
                "rel_vx": dv[0] - self_vx,
                "rel_vy": dv[1] - self_vy,
            })

        def_feats.sort(key=lambda x: x["dist"])
        _pad = {"dist": 10.0, "sin": 0.0, "cos": 1.0, "rel_vx": 0.0, "rel_vy": 0.0}
        d1   = def_feats[0] if len(def_feats) > 0 else _pad
        d2   = def_feats[1] if len(def_feats) > 1 else _pad

        return np.array([
            goal_dist, np.sin(goal_angle), np.cos(goal_angle),
            self_vx, self_vy,
            d1["dist"], d1["sin"], d1["cos"], d1["rel_vx"], d1["rel_vy"],
            d2["dist"], d2["sin"], d2["cos"], d2["rel_vx"], d2["rel_vy"],
        ], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────

    def step(self, action):
        self.step_count += 1

        # ── resolve action → [gas, steer] in [-1, 1] ──────────────────
        if self.discrete:
            # DQN passes an integer index
            gas, steer = ACTION_SET[int(action)]
        else:
            # PPO / BC / GAIL / IQL / IQ-Learn pass a float array
            # IQ-Learn passes index_to_action result [±1,0] — also fine
            gas, steer = float(action[0]), float(action[1])

        v_target = gas   * self.MAX_SPEED
        w_target = steer * self.TURN_BIAS

        # ── inner physics loop (8 sub-steps) ──────────────────────────
        is_dribbling = False
        for _ in range(self.INNER_STEPS):
            self.v_curr += (v_target - self.v_curr) * self.ACCEL
            self.w_curr += (w_target - self.w_curr) * self.ACCEL

            left_speed  = np.clip(self.v_curr - self.w_curr, -20.0, 20.0)
            right_speed = np.clip(self.v_curr + self.w_curr, -20.0, 20.0)
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)

            # Dribbler physics
            pos      = self.gps.getValues()
            comp     = self.compass.getValues()
            heading  = np.arctan2(comp[0], comp[1])
            ball_pos = self.ball_node.getPosition()

            dx           = ball_pos[0] - pos[0]
            dy           = ball_pos[1] - pos[1]
            dist_to_ball = np.hypot(dx, dy)
            angle_to_ball = np.arctan2(dy, dx)
            angle_diff   = angle_to_ball - heading

            # Normalise angle to [-π, π]
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

            if dist_to_ball < 0.1 and abs(angle_diff) < 0.5:
                is_dribbling = True
                notch_x = pos[0] + np.cos(heading) * self.DRIBBLER_OFFSET
                notch_y = pos[1] + np.sin(heading) * self.DRIBBLER_OFFSET
                pull_vx = (notch_x - ball_pos[0]) * self.DRIBBLER_PULL
                pull_vy = (notch_y - ball_pos[1]) * self.DRIBBLER_PULL
                self.ball_node.setVelocity([pull_vx, pull_vy, 0.0, 0.0, 0.0, 0.0])

            if self.robot.step(self.timestep) == -1:
                sys.exit(0)

        # ── post-step state ───────────────────────────────────────────
        pos      = self.gps.getValues()
        ball_pos = self.ball_node.getPosition()

        dist_ball_to_goal = np.hypot(
            self.MY_GOAL_CENTER[0] - ball_pos[0],
            self.MY_GOAL_CENTER[1] - ball_pos[1],
        )

        # ── reward computation ────────────────────────────────────────
        reward     = -0.2   # time step tax — encourages efficiency
        
        terminated = False
        truncated  = False

        # Idle freeze penalty
        if abs(gas) < 0.1 and abs(steer) < 0.1:
            reward -= 0.5

        # 1. Ball progress toward goal
        #    Clipped to ±0.5 before scaling to prevent single-step spikes
        #    (e.g. ball teleporting after physics glitch giving ±50 reward)
        progress = self.prev_ball_dist - dist_ball_to_goal
        reward  += np.clip(progress, -0.5, 0.5) * 20.0
        self.prev_ball_dist = dist_ball_to_goal

        # 3. Collision penalty (soft, continuous)
        #    -10 per step in contact rather than terminal, so the agent
        #    has time to learn to go around defenders rather than
        #    ending every episode the moment it gets close
        for defender in self.defenders:
            def_pos      = defender.getPosition()
            dist_to_def  = np.hypot(def_pos[0] - pos[0], def_pos[1] - pos[1])
            if dist_to_def < 0.20:
                reward -= 10.0
                break   # one penalty per step, not per defender

        # 4. Defender beaten bonus
        for idx, defender in enumerate(self.defenders):
            if idx not in self.beaten_defenders:
                def_pos = defender.getPosition()
                if ball_pos[0] > def_pos[0] + 0.15:
                    reward += 50.0
                    self.beaten_defenders.add(idx)

        # 5. Goal scored — episode win
        #    FIX: previously missing `terminated = True`, episode never ended
        if dist_ball_to_goal < 0.25:
            reward    += 100.0
            terminated = True

        # 6. Out of bounds — episode loss
        if (
            pos[0] >  2.5 or pos[0] < -2.5 or
            pos[1] >  1.5 or pos[1] < -1.5
        ):
            reward    -= 15.0
            terminated = True

        # 7. Timeout — truncated, NOT terminated
        #    FIX: was returning (done, False) which zeroed Bellman bootstrap
        #    on timeouts. Truncation means episode continues conceptually,
        #    so γ·V(s') should NOT be zeroed.
        if not terminated and self.step_count > 200:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}