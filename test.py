import gymnasium as gym
import panda_gym
import time
import numpy as np
import torch


class PybulletEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = np.zeros(self.env.observation_space['observation'].shape[1])
        self.action_space = np.zeros(self.env.action_space.shape[1])
        self.max_episode_length = 100
        self.device = torch.device("cuda:0")

    def reset(self):
        ob, info = self.env.reset()
        return self.cast(ob['observation'])

    def step(self, actions):
        # actions = actions.cpu().numpy()
        next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
        dones = np.logical_or(terminated, truncated)
        timeout = torch.tensor(truncated).bool().to(self.device)
        success = torch.tensor(terminated).to(self.device)
        info_ret = {'time_outs': timeout, 'success': success}

        return self.cast(next_obs['observation']), self.cast((rewards + 1) * 10), self.cast(dones).long(), info_ret

    def cast(self, x):
        x = torch.tensor(x).to(self.device)
        return x
    
# env1 = gym.vector.make('PandaReachJoints-v3', control_type='joints', reward_type='sparse', num_envs=2)
env = gym.make('PandaReachJoints-v3', render_mode="human", control_type='joints', reward_type='sparse')
# env = PybulletEnvWrapper(env)
observation = env.reset()
for i in range(1000):
    action = env.action_space.sample() # random action
    observation, reward, done, _,  info = env.step(action)
    print(observation['achieved_goal'], observation['observation'])
    time.sleep(0.1)
    # if terminated or truncated:
    #     observation, info = env.reset()

env.close()