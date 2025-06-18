import os
import memory_gym
import gymnasium as gym
import numpy as np

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from random import randint
from gymnasium import spaces


def _process_vis(img):
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img


class MemoryGymWrapper(gym.Env):
    """
    This class wraps memory-gym environments.
    https://github.com/MarcoMeter/drl-memory-gym
    Available Environments:
        SearingSpotlights-v0
        MortarMayhem-v0
        MortarMayhem-Grid-v0
        MysteryPath-v0
        MysteryPath-Grid-v0
    """

    def __init__(self, env_name, reset_params=None, realtime_mode=False) -> None:
        """Instantiates the memory-gym environment.
        
        Arguments:
            env_name {string} -- Name of the memory-gym environment
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
            realtime_mode {bool} -- Whether to render the environment in realtime. (default: {False})
        """
        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, "num-seeds": 100}
        else:
            self._default_reset_params = reset_params

        render_mode = None if not realtime_mode else "human"
        self._env = gym.make(env_name, disable_env_checker=True, render_mode=render_mode)

        self._realtime_mode = realtime_mode

        base_space = self._env.observation_space
        if isinstance(base_space, spaces.Dict):
            self._dict_obs = True
            h, w, c = base_space["visual_observation"].shape
            self.observation_space = spaces.Dict({
                "base_observation": spaces.Box(low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32),
                "additional_observation": spaces.Box(low=0.0, high=1.0, shape=base_space["vector_observation"].shape,
                                                     dtype=np.float32)
            })
            self.base_observation = spaces.Box(low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32)
            self.additional_observation = spaces.Box(low=0.0, high=1.0, shape=base_space["vector_observation"].shape,
                                                     dtype=np.float32)
        else:
            self._dict_obs = False
            h, w, c = base_space.shape
            self.observation_space = spaces.Box(low=0, high=1.0, shape=(c, h, w), dtype=np.float32)
            self.base_observation = spaces.Box(low=0, high=1.0, shape=(c, h, w), dtype=np.float32)
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
        return self._dict_obs

    def reset(self, reset_params=None, seed=None, options=None):
        """Resets the environment.
        
        Keyword Arguments:
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
        
        Returns:
            {numpy.ndarray} -- Visual observation
        """
        # Process reset parameters
        if reset_params is None:
            reset_params = self._default_reset_params
        else:
            reset_params = reset_params

        # Sample seed
        self._seed = randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1)

        # Remove reset params that are not processed directly by the environment
        options = reset_params.copy()
        options.pop("start-seed", None)
        options.pop("num-seeds", None)
        options.pop("seed", None)

        # Reset the environment to retrieve the initial observation
        raw_obs, _ = self._env.reset(seed=self._seed, options=options)

        if self._dict_obs:
            obs = {
                "base_observation": _process_vis(raw_obs["visual_observation"]),
                "additional_observation": raw_obs["vector_observation"].astype(np.float32)
            }
        else:
            obs = _process_vis(raw_obs)
        return obs, {}

    def step(self, action):
        """Runs one timestep of the environment's dynamics.
        
        Arguments:
            action {list} -- The to be executed action
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {float} -- (Total) Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further episode information (e.g. cumulated reward) retrieved from the environment once an episode completed
        """
        action = np.squeeze(action, axis=0) if action.shape[0] == 1 else action
        raw_obs, reward, done, truncated, info = self._env.step(action)

        if self._dict_obs:
            obs = {
                "base_observation": _process_vis(raw_obs["visual_observation"]),
                "additional_observation": raw_obs["vector_observation"].astype(np.float32)
            }
        else:
            obs = _process_vis(raw_obs)

        if done or truncated:
            info = {'reward': info['reward']}
        else:
            info = {}

        return obs, reward, done, truncated, info

    def render(self):
        """Renders the environment."""
        self._env.render()

    def close(self):
        """Shuts down the environment."""
        self._env.close()
