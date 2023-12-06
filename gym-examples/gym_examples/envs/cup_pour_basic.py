from typing import Dict

import numpy as np

import os

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class SimpleCupEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = os.path.dirname(os.path.abspath(__file__)) + "/assets/scene.xml",
        frame_skip: int = 2,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        reward_dist_weight: float = 10,
        reward_dist_weight2: float = 5,
        reward_dist_weight3: float = 5,
        reward_control_weight: float = 3,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_dist_weight,
            reward_dist_weight2,
            reward_dist_weight3,
            reward_control_weight,
            **kwargs,
        )

        self._reward_dist_weight = reward_dist_weight
        self._reward_dist_weight2 = reward_dist_weight2
        self._reward_dist_weight3 = reward_dist_weight3
        self._reward_control_weight = reward_control_weight
        self._past_act = [0]

        observation_space = Box(low=-np.inf, high=np.inf, shape=(33,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        waterStatic = []
        waterPour = []
        for i in range(0, 10):
            waterStatic.append(np.linalg.norm(self.get_body_com("particle_%i" % i) - self.get_body_com("mugStatic")))
            waterPour.append(np.linalg.norm(self.get_body_com("particle_%i" % i) - self.get_body_com("mug")))

        reward_dist = -np.average(waterStatic) * self._reward_dist_weight
        reward_dist2 = (self.data.qpos.flat[0] - 2) * self._reward_dist_weight2
        reward_dist3 = -np.max(waterStatic) * self._reward_dist_weight3
        reward_ctrl = -np.square(np.abs(action-self._past_act)).sum() * self._reward_control_weight

        self._past_act = action

        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()

        if np.average(waterStatic) < np.average(waterPour):
            reward = reward_dist + reward_dist2 + reward_dist3 + reward_ctrl
        else:
            reward = reward_dist2 + reward_ctrl

        info = {
            "rewardCloseness": reward_dist,
            "rewardDownwardCup": reward_dist2,
            "actionPenaltyMagnitude": reward_ctrl,
        }

        # print(info)

        if self.render_mode == "human":
            self.render()

        dones = True
        for loc in range(1, len(observation)):
            staticDist = np.linalg.norm(observation[loc] - self.get_body_com("mugStatic"))
            pourDist = np.linalg.norm(observation[loc] - self.get_body_com("mug"))

            if pourDist < 0.1 or observation[loc][-1] > 0.05:
                dones = False
                break
            if staticDist > 0.05 and staticDist < 0.1 and observation[loc][-1] < 0.1:
                dones = False
                break

        return observation, reward, dones, False, info

    def reset_model(self):
        qpos = self.init_qpos

        qvel = self.init_qvel

        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        obs = []

        obs.append([self.data.qpos.flat[0],0,0])

        for i in range(0, 10):
            obs.append(self.get_body_com("particle_%i" % i))

        return obs
