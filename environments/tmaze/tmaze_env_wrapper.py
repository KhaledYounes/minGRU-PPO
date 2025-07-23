import gymnasium as gym

from .tmaze import TMazeClassicActive


class TMaze(gym.Env):
    def __init__(self, corridor_length=10, realtime_mode=False):
        self._realtime_mode = realtime_mode
        render_mode = "human" if realtime_mode else "rgb_array"

        self._env = TMazeClassicActive(corridor_length=corridor_length)

        self.observation_space = self._env.observation_space

        self.base_observation = self._env.observation_space
        self.additional_observation = None

        self.action_space = self._env.action_space
        self.metadata = self._env.metadata
        self.render_mode = render_mode
        self.rewards = []

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
        obs, _ = self._env.reset()
        self.rewards = []
        return obs, {}

    def step(self, action):
        obs, reward, done, _, info = self._env.step(action[0])

        self.rewards.append(reward)

        if done:
            info = {"reward": sum(self.rewards)}
        else:
            info = {}

        return obs, reward, done, False, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
