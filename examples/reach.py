import gymnasium as gym
import numpy as np
import panda_gym
import time 

env = gym.make("PandaReachJoints-v3", render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    action = np.zeros_like(action)
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)
    time.sleep(0.1)
    if terminated or truncated:
        print("reset")
        print(truncated)
        observation, info = env.reset()

env.close()
