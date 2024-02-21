from typing import Any, Dict

import random
import math
import pybullet as p
import pybullet_data
import numpy as np
from panda_gym.envs.core import Task
from panda_gym.utils import distance
import os
MODULE_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# from pybullet_data.urdf


class Door(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",  # TODO: specify options (if there are any)
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.get_ee_position = get_ee_position
        # door
        # self.door_file_path = MODULE_PATH + "/assets/objects/door/door_2.urdf"
        # self.door_file_path = MODULE_PATH + "/assets/iai_shelves/urdf/shelves.urdf"

        pybullet_data.getDataPath()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.door_file_path = pybullet_data.getDataPath() + "/urdf/door.urdf"

        self.door_joint = 0
        self.door_body = "door"

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=2.5, width=1.2, height=0.4, x_offset=-0.3)
        self._create_door()
        # self.sim.create_sphere(
        #     body_name="target",
        #     radius=self.distance_threshold,
        #     mass=0.0,
        #     ghost=True,
        #     position=np.zeros(3),
        #     rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        # )

    def _create_door(self):
        door_ori = [0, 0, math.pi/2]
        self.sim.loadURDF(body_name=self.door_body,
                          fileName=self.door_file_path, basePosition=[0.5, -0.5, 0.0],
                          globalScaling=1, baseOrientation=self.sim.get_quat_euler(door_ori),
                          useFixedBase=True)

        self._reset_door()

    def _reset_door(self, random_pos=False):
        # TODO: add randomization to door init?
        init_door_joint_state = 0.15  # 0.7
        self.sim.set_joint_angle(body=self.door_body, joint=self.door_joint, angle=init_door_joint_state)

    def _get_door_joint_pos(self):
        j_pos = self.sim.get_joint_angle(self.door_body, self.door_joint)
        return j_pos

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no task-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        # ee_position = np.array(self.get_ee_position())
        # 1-dimensional
        door_joint_pos = self._get_door_joint_pos()
        return np.array([door_joint_pos])

    def get_goal(self):
        # fixed goal (close door_joint)
        return np.array([0.0001])

    def reset(self) -> None:
        self._reset_door()
        # self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        # d = distance(achieved_goal, desired_goal)
        return achieved_goal <= desired_goal

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            rew = 1 if self.is_success(achieved_goal, desired_goal) else 0
            return np.array([rew])
        else:
            # door opening distance measured in joint-space
            return -d.astype(np.float32)
