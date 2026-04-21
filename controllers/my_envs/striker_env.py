"""
StrikerRLEnv — unified environment for benchmarking
=====================================================

Changes in this version
------------------------
- Defenders reset to random positions within the attacking half (x > 0)
  every episode instead of fixed spawn points. This prevents all
  algorithms from overfitting to one specific defensive configuration
  and makes trained policies more robust.

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

Defender spawn zone (attacking half)
--------------------------------------
  x : [0.2,  2.2]   — in front of striker's starting position
  y : [-1.2, 1.2]   — inside field width
  z : preserved from initial Webots scene (ground height)

  Minimum separation of 0.4 m between any two defenders enforced via
  rejection sampling so they never spawn on top of each other.

Reward function
-----------------
  Progress          :  clip(progress, -0.5, 0.5) * 20
  Ball possession   :  +0.3 per step while dribbling
  Defender beaten   :  +50 per defender passed
  Goal scored       :  +100,  terminated=True
  Collision         :  -10 per step in contact
  Idle penalty      :  -0.5 per step (gas < 0.1 and steer < 0.1)
  Out of bounds     :  -15,   terminated=True
  Time step tax     :  -0.2 per step
  Timeout           :  truncated=True  (NOT terminated)
"""

import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from controller import Supervisor

# ──────────────────────────────────────────────────────────────────────────────
# Discrete action set
# ──────────────────────────────────────────────────────────────────────────────

ACTION_SET = np.array([
    [-1., -1.], [-1.,  0.], [-1., +1.],
    [ 0., -1.], [ 0.,  0.], [ 0., +1.],
    [+1., -1.], [+1.,  0.], [+1., +1.],
], dtype=np.float32)

# ── Defender random spawn bounds ──────────────────────────────────────────────
DEF_X_MIN   =  0.0
DEF_X_MAX   =  2.0
DEF_Y_MIN   = -1.3
DEF_Y_MAX   =  1.3
DEF_MIN_SEP =  0.4   # minimum distance between any two defenders at spawn


class StrikerRLEnv(gym.Env):

    def __init__(self, discrete: bool = False):
        super().__init__()

        self.discrete = discrete
        self.robot    = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        # ── action space ─────────────────────────────────────────────
        if discrete:
            self.action_space = spaces.Discrete(9)
        else:
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
        self.striker_node      = self.robot.getSelf()
        self.robot_trans_field = self.striker_node.getField("translation")
        self.robot_rot_field   = self.striker_node.getField("rotation")

        self.ball_node        = self.robot.getFromDef("Ball")
        self.ball_trans_field = self.ball_node.getField("translation")

        self.init_robot_pos = self.robot_trans_field.getSFVec3f()
        self.init_robot_rot = self.robot_rot_field.getSFRotation()
        self.init_ball_pos  = self.ball_trans_field.getSFVec3f()

        # ── defenders ─────────────────────────────────────────────────
        # Store both the node (velocity/position reads) and the
        # translation field (teleport on reset). The z-coordinate is
        # captured once from the scene so defenders stay on the ground.
        self.defenders        = []
        self.def_trans_fields = []
        self.def_init_z       = []

        for i in range(1, 3):
            node = self.robot.getFromDef(f"defender{i}")
            if node is not None:
                tf = node.getField("translation")
                self.defenders.append(node)
                self.def_trans_fields.append(tf)
                self.def_init_z.append(tf.getSFVec3f()[2])

        # ── parameters ───────────────────────────────────────────────
        self.MY_GOAL_CENTER  = [2.0, 0.0]
        self.MAX_SPEED       = 15.0
        self.TURN_BIAS       = 5.0
        self.ACCEL           = 0.2
        self.DRIBBLER_OFFSET = 0.05
        self.DRIBBLER_PULL   = 15.0
        self.INNER_STEPS     = 8

        # ── runtime state ─────────────────────────────────────────────
        self.v_curr           = 0.0
        self.w_curr           = 0.0
        self.step_count       = 0
        self.prev_ball_dist   = 0.0
        self.beaten_defenders = set()

    # ──────────────────────────────────────────────────────────────────
    # Defender randomisation
    # ──────────────────────────────────────────────────────────────────

    def _randomise_defenders(self):
        """
        Place each defender at a random position in the attacking half.

        Uses rejection sampling to enforce DEF_MIN_SEP between defenders.
        Caps at 100 attempts per defender to avoid infinite loops on
        very crowded configurations — in practice 3 defenders in the
        zone above always converge within a handful of tries.
        """
        placed = []   # accepted (x, y) positions so far

        for tf, z in zip(self.def_trans_fields, self.def_init_z):
            for _ in range(100):
                x = np.random.uniform(DEF_X_MIN, DEF_X_MAX)
                y = np.random.uniform(DEF_Y_MIN, DEF_Y_MAX)

                too_close = any(
                    np.hypot(x - px, y - py) < DEF_MIN_SEP
                    for px, py in placed
                )
                if not too_close:
                    break

            tf.setSFVec3f([x, y, z])
            placed.append((x, y))

        # Kill residual velocity so defenders don't slide after teleport
        for node in self.defenders:
            node.setVelocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # ──────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot_trans_field.setSFVec3f(self.init_robot_pos)
        self.robot_rot_field.setSFRotation(self.init_robot_rot)

        curriculum_ball_pos = [
            self.init_robot_pos[0] + 0.05,
            self.init_robot_pos[1],
            self.init_ball_pos[2],
        ]
        self.ball_trans_field.setSFVec3f(curriculum_ball_pos)
        self.ball_node.setVelocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Randomise defender positions every episode
        self._randomise_defenders()

        self.robot.simulationResetPhysics()
        self.robot.step(self.timestep)

        self.v_curr           = 0.0
        self.w_curr           = 0.0
        self.step_count       = 0
        self.beaten_defenders = set()
        self.prev_ball_dist   = np.hypot(
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

        est_v   = self.v_curr * 0.02
        self_vx = est_v * np.cos(heading)
        self_vy = est_v * np.sin(heading)

        gdx        = self.MY_GOAL_CENTER[0] - pos[0]
        gdy        = self.MY_GOAL_CENTER[1] - pos[1]
        goal_dist  = np.hypot(gdx, gdy)
        goal_angle = np.arctan2(gdy, gdx) - heading

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

        if self.discrete:
            gas, steer = ACTION_SET[int(action)]
        else:
            gas, steer = float(action[0]), float(action[1])

        v_target = gas   * self.MAX_SPEED
        w_target = steer * self.TURN_BIAS

        is_dribbling = False
        for _ in range(self.INNER_STEPS):
            self.v_curr += (v_target - self.v_curr) * self.ACCEL
            self.w_curr += (w_target - self.w_curr) * self.ACCEL

            left_speed  = np.clip(self.v_curr - self.w_curr, -20.0, 20.0)
            right_speed = np.clip(self.v_curr + self.w_curr, -20.0, 20.0)
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)

            pos           = self.gps.getValues()
            comp          = self.compass.getValues()
            heading       = np.arctan2(comp[0], comp[1])
            ball_pos      = self.ball_node.getPosition()

            dx            = ball_pos[0] - pos[0]
            dy            = ball_pos[1] - pos[1]
            dist_to_ball  = np.hypot(dx, dy)
            angle_to_ball = np.arctan2(dy, dx)
            angle_diff    = (angle_to_ball - heading + np.pi) % (2 * np.pi) - np.pi

            if dist_to_ball < 0.1 and abs(angle_diff) < 0.5:
                is_dribbling = True
                notch_x = pos[0] + np.cos(heading) * self.DRIBBLER_OFFSET
                notch_y = pos[1] + np.sin(heading) * self.DRIBBLER_OFFSET
                pull_vx = (notch_x - ball_pos[0]) * self.DRIBBLER_PULL
                pull_vy = (notch_y - ball_pos[1]) * self.DRIBBLER_PULL
                self.ball_node.setVelocity([pull_vx, pull_vy, 0.0, 0.0, 0.0, 0.0])

            if self.robot.step(self.timestep) == -1:
                sys.exit(0)

        pos      = self.gps.getValues()
        ball_pos = self.ball_node.getPosition()

        dist_ball_to_goal = np.hypot(
            self.MY_GOAL_CENTER[0] - ball_pos[0],
            self.MY_GOAL_CENTER[1] - ball_pos[1],
        )

        reward     = -0.2
        terminated = False
        truncated  = False

        if abs(gas) < 0.1 and abs(steer) < 0.1:
            reward -= 0.5

        progress = self.prev_ball_dist - dist_ball_to_goal
        reward  += np.clip(progress, -0.5, 0.5) * 20.0
        self.prev_ball_dist = dist_ball_to_goal

        if is_dribbling:
            reward += 0.3

        for defender in self.defenders:
            def_pos     = defender.getPosition()
            dist_to_def = np.hypot(def_pos[0] - pos[0], def_pos[1] - pos[1])
            if dist_to_def < 0.20:
                reward -= 10.0
                break

        for idx, defender in enumerate(self.defenders):
            if idx not in self.beaten_defenders:
                def_pos = defender.getPosition()
                if ball_pos[0] > def_pos[0] + 0.15:
                    reward += 50.0
                    self.beaten_defenders.add(idx)

        if dist_ball_to_goal < 0.25:
            reward    += 100.0
            terminated = True

        if (
            pos[0] >  2.5 or pos[0] < -2.5 or
            pos[1] >  1.5 or pos[1] < -1.5
        ):
            reward    -= 15.0
            terminated = True

        if not terminated and self.step_count > 250:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}