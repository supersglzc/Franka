import gymnasium as gym
import numpy as np
import panda_gym
import time 

env = gym.make("PandaReach-v3", render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    action = np.zeros_like(action)
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(1/120)
    if terminated or truncated:
        print("reset")
        print(truncated)
        observation, info = env.reset()

env.close()
