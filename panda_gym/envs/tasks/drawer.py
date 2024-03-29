from typing import Any, Dict

import random
import numpy as np
import pybullet as p
from panda_gym.envs.core import Task
from panda_gym.utils import distance
import os
MODULE_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class Drawer(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        # self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        # drawer
        self.drawer_file_path = MODULE_PATH + "/assets/objects/cabinet/drawer_1.urdf"
        self.drawer_joint = 1
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=2.5, width=1.2, height=0.4, x_offset=-0.3)
        self._create_drawer()
        # self.sim.create_sphere(
        #     body_name="target",
        #     radius=self.distance_threshold,
        #     mass=0.0,
        #     ghost=True,
        #     position=np.zeros(3),
        #     rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        # )

    def _create_drawer(self):
        self.sim.loadURDF(body_name="drawer",
                          fileName=self.drawer_file_path, basePosition=[0.3, 0.0, 0.18],
                          globalScaling=1.0, baseOrientation=[0, 180, 0, 1],
                          useFixedBase=True)
        self._reset_drawer()

    def _reset_drawer(self):
        init_drawer_joint_state = 0.15
        self.sim.set_joint_angle(body="drawer", joint=0, angle=init_drawer_joint_state)

    def _get_drawer_joint_pos(self):
        j_pos = self.sim.get_joint_angle("drawer", 0)
        return j_pos

    def _get_drawer_angle(self):
        return self._get_drawer_joint_pos()

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no task-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        # ee_position = np.array(self.get_ee_position())
        drawer_joint_pos = self._get_drawer_angle()
        return np.array([drawer_joint_pos])

    def get_goal(self):
        # fixed goal (close drawer_joint)
        return np.array([0.001])

    def reset(self) -> None:
        self._reset_drawer(random_pos=False)
        # self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        # d = distance(achieved_goal, desired_goal)
        return achieved_goal <= desired_goal

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > 0.0, dtype=np.float32)
        else:
            return -d.astype(np.float32)
