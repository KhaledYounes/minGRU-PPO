import time

import gymnasium as gym
import numpy as np


class CartPole(gym.Env):
    def __init__(self, max_steps=500, mask_velocity=False, realtime_mode=False):
        render_mode = "human" if realtime_mode else None
        self._env = gym.make("CartPole-v1", render_mode=render_mode)
        # Whether to make CartPole partial observable by masking out the velocity.
        if not mask_velocity:
            self._obs_mask = np.ones(4, dtype=np.float32)
        else:
            self._obs_mask = np.array([1, 0, 1, 0], dtype=np.float32)

        self.current_steps = 0
        self.max_steps = max_steps
        self.rewards = []

        self.observation_space = self._env.observation_space

        self.base_observation = self._env.observation_space
        self.additional_observation = None

        self.action_space = self._env.action_space
        self.metadata = self._env.metadata
        self.render_mode = render_mode

    @property
    def action_spaces(self):
        """Returns the action space of the agent as a list."""
        if isinstance(self._env.action_space, gym.spaces.Discrete):
            return [self._env.action_space.n]
        elif isinstance(self._env.action_space, gym.spaces.MultiDiscrete):
            return self._env.action_space.nvec.tolist()
        else:
            raise NotImplementedError("This action space type is not supported.")

    @property
    def action_space_dimensions(self):
        """Returns the dimensions of the action space of the agent."""
        return len(self.action_spaces)

    @property
    def has_additional_info(self):
        """Returns whether observation space is Dict."""
        return False

    def reset(self, seed=None, options=None):
        obs, _ = self._env.reset()
        return obs * self._obs_mask, {}

    def step(self, action):
        action = np.squeeze(action) if action.shape[0] == 1 else action
        obs, reward, done, truncated, info = self._env.step(action)
        self.current_steps += 1

        valid_step = self.current_steps < self.max_steps

        if valid_step:
            if done or truncated:
                done = truncated = False
                self.reset()
                reward = -10.0
            else:
                reward = 1.0
            info = {}
            self.rewards.append(reward)
        else:
            done = truncated = True
            self.current_steps = 0
            info = {'reward': sum(self.rewards)}
            self.rewards = []

        return obs * self._obs_mask, reward, done, truncated, info

    def render(self):
        self._env.render()
        time.sleep(0.033)

    def close(self):
        self._env.close()
