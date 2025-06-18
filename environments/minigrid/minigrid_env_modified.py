from minigrid.wrappers import *


def _process_vis(img):
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img


class Minigrid(gym.Env):
    def __init__(self, env_name, realtime_mode=False):
        self._realtime_mode = realtime_mode
        render_mode = "human" if realtime_mode else "rgb_array"

        self._env = gym.make(env_name, agent_view_size=3, tile_size=28, render_mode=render_mode)
        self._env = RGBImgPartialObsWrapper(self._env, tile_size=28)
        self._env = ImgObsWrapper(self._env)

        self.observation_space = spaces.Box(low=0, high=1.0, shape=(3, 84, 84), dtype=np.float32)

        self.base_observation = spaces.Box(low=0, high=1.0, shape=(3, 84, 84), dtype=np.float32)
        self.additional_observation = None

        self.action_space = spaces.Discrete(3)
        self.metadata = self._env.metadata
        self.render_mode = render_mode

    @property
    def action_spaces(self):
        """Returns the action space of the agent as a list."""
        return [self.action_space.n]

    @property
    def action_space_dimensions(self):
        """Returns the dimensions of the action space of the agent."""
        return len(self.action_spaces)

    @property
    def has_additional_info(self):
        """Returns whether observation space is Dict."""
        return False

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed if seed else np.random.randint(0, 99))
        obs = _process_vis(obs)
        return obs, {}

    def step(self, action):
        obs, reward, done, truncated, info = self._env.step(action)
        obs = _process_vis(obs)

        if done or truncated:
            info = {"reward": reward}
        else:
            info = {}

        return obs, reward, done, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
