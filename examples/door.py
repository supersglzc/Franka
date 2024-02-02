
import gymnasium as gym
import time

import numpy as np

import panda_gym
import panda_gym.envs

# env_name = "PandaPegInsertionJoints-v3"
env_name = "PandaDoorJoints-v3"
render_mode = "human"  # "human", rgb_array
env = gym.make(env_name, render_mode=render_mode)

observation, info = env.reset()

for _ in range(10000):
    # action = env.action_space.sample()
    action = np.array([0]*7)
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(1/30)
    print(f"reward: {reward}")
    if terminated or truncated:
        observation, info = env.reset()

env.close()
