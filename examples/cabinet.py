import gymnasium as gym
import time

import numpy as np

import panda_gym
import panda_gym.envs


import pybullet as p
import time
import pybullet_data

# Start PyBullet in GUI mode
physicsClient = p.connect(p.GUI)

# Load the URDF
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Optionally set a search path
cabinetId = p.loadURDF("/home/supersglzc/code/Franka/panda_gym/assets/objects/cabinet/cabinet_0004.urdf", useFixedBase=True)

# Get the number of joints and print their names
num_joints = p.getNumJoints(cabinetId)
for i in range(num_joints):
    joint_info = p.getJointInfo(cabinetId, i)
    print(f"Joint index: {joint_info[0]}, Joint name: {joint_info[1].decode('utf-8')}")

# Identify the joint index for the cabinet door (replace 'door_joint' with the actual joint name)
door_joint_index = None
for i in range(num_joints):
    joint_info = p.getJointInfo(cabinetId, i)
    print(joint_info[1])
    if joint_info[1].decode('utf-8') == 'dof_rootd_Ba001_r':
        door_joint_index = i
        break

if door_joint_index is not None:
    # Apply a force to the door joint
    # Example: Apply a force to move the door
    p.setJointMotorControl2(bodyUniqueId=cabinetId,
                            jointIndex=door_joint_index,
                            controlMode=p.TORQUE_CONTROL,
                            force=10000)  # Adjust the force as needed

    # Simulate for a short period
    for i in range(1000):
        print(i)
        p.stepSimulation()
        time.sleep(0.1)

p.disconnect()

assert 0

# env_name = "PandaPegInsertionJoints-v3"
env_name = "PandaCabinetJoints-v3"
render_mode = "human"  # "human", rgb_array
env = gym.make(env_name, render_mode=render_mode)

observation, info = env.reset()

for _ in range(10000):
    # action = env.action_space.sample()
    action = np.array([0.0, 0, 0, 0, 0, 0, 0])
    # action = np.array([0.0, 0, 0.0])
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(1/30)
    print(f"reward: {reward}")
    print(observation["observation"])
    if terminated or truncated:
        observation, info = env.reset()

env.close()
