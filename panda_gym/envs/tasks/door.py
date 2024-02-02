from typing import Any, Dict

import random
import numpy as np
import pybullet as p
from panda_gym.envs.core import Task
from panda_gym.utils import distance
import os
MODULE_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class Door(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.1,
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        # door
        # self.door_file_path = MODULE_PATH + "/assets/objects/door/door.urdf"
        # self.door_file_path = MODULE_PATH + "/assets/objects/cabinet/cabinet_0004.urdf"
        self.door_file_path = MODULE_PATH + "/assets/objects/cabinet/drawer_1.urdf"
        # door
        # self.threshold_found = 0.7
        self.door_joint = 1
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
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
        # self.sim.loadURDF(body_name="door",
        #                   fileName=self.door_file_path, basePosition=[0.86, 0, 0.45],
        #                             globalScaling=1, baseOrientation=[0, 180, 0, 1])
        self.sim.loadURDF(body_name="door",
                          fileName=self.door_file_path, basePosition=[0.3, 0.0, 0.18],
                                    globalScaling=1.0, baseOrientation=[0, 180, 0, 1],
                          useFixedBase=True)

        random_pos = False
        self._reset_door(random_pos=random_pos)

        # self.sim.get_info("door")

        # self.sim.changeDynamics("door", 1, linearDamping=0, angularDamping=0, jointDamping=10)

            # p.resetJointState(self.door, i, init_door_joint_state[i])

        # self.sim.change_visual("door", 0, color=[0.8, 0.1, 0.1])
        # self.sim.change_visual("door", 1, color=[0.1, 0.1, 0.8])

        # p.changeVisualShape(self.door, 0,
        #                 rgbaColor=[random.uniform(0, 1), random.uniform(0, 1), random.randint(0, 1), 1])
        # p.changeVisualShape(self.door, 1,
        #                     rgbaColor=[random.uniform(0, 1), random.uniform(0, 1), random.randint(0, 1), 1])

    def _reset_door(self, random_pos=False):
        # self.sim.get_info("door")
        # if random_pos:
        #     init_door_joint_state = 0.6*random.uniform(0, 1)
        # else:
        init_door_joint_state = 0.15  # 0.7

        self.sim.set_joint_angle(body="door", joint=0, angle=init_door_joint_state)

    def _get_door_joint_pos(self):
        j_pos = self.sim.get_joint_angle("door", 0)
        return j_pos

    def _get_door_angle(self):
        return self._get_door_joint_pos()

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no task-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        # ee_position = np.array(self.get_ee_position())
        door_angle = self._get_door_angle()
        return np.array([door_angle])

    def get_goal(self):
        return np.array([0.001])

    def reset(self) -> None:
        self._reset_door(random_pos=False)
        # self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))


    # def _sample_goal(self) -> np.ndarray:
    #     """Randomize goal."""
    #     # goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
    #     # print(self.goal_range_low, self.goal_range_high, goal)
    #     goal = np.array([0.05, 0, 0.5])
    #     return goal

    def pre_sim_step(self):
        # self.sim.physics_client.setJointMotorControl2(self.sim._bodies_idx["door"], 1, controlMode=p.VELOCITY_CONTROL, targetVelocity=0,
        #                         force=10000)
        pass

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        # d = distance(achieved_goal, desired_goal)
        return achieved_goal <= 0.001

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -achieved_goal
            # return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)
