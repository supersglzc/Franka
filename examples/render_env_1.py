import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

import panda_gym
import panda_gym.envs



# env_name = "PandaPegInsertionJoints-v3"
env_name = "PandaDrawerMultiJoints-v3"
render_mode = "rgb_array"  # "human", rgb_array
env = gym.make(env_name, render_mode=render_mode)

observation, info = env.reset()

for _ in range(10):
    # action = env.action_space.sample()
    action = np.array([0]*7)
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"reward: {reward}")
    if terminated or truncated:
        observation, info = env.reset()

frame = env.render()
fig = plt.imshow(frame)
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.show()
# plt.savefig(fig, "env_render.png")

env.close()
