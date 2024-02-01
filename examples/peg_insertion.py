import gymnasium as gym

import panda_gym
import panda_gym.envs

env = gym.make("PandaPegInsertionJoints-v3", render_mode="rgb_array")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"reward: {reward}")
    if terminated or truncated:
        observation, info = env.reset()

env.close()
