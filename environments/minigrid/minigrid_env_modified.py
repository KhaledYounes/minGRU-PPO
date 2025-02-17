from minigrid.wrappers import *


class Minigrid(gym.Env):
    def __init__(self, env_name, realtime_mode=False):
        self._realtime_mode = realtime_mode
        render_mode = "human" if realtime_mode else "rgb_array"

        self._env = gym.make(env_name, agent_view_size=3, tile_size=28, render_mode=render_mode)
        self._env = RGBImgPartialObsWrapper(self._env, tile_size=28)
        self._env = ImgObsWrapper(self._env)

        self.observation_space = spaces.Box(
            low=0,
            high=1.0,
            shape=(3, 84, 84),  # CHW format
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self.metadata = self._env.metadata
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        self._rewards = []
        obs, info = self._env.reset(seed=seed if seed else np.random.randint(0, 99))
        obs = obs.astype(np.float32) / 255.
        obs = np.transpose(obs, (2, 0, 1))
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self._env.step(action)
        self._rewards.append(reward)
        obs = obs.astype(np.float32) / 255.
        obs = np.transpose(obs, (2, 0, 1))

        if done or truncated:
            info = {"reward": sum(self._rewards), "length": len(self._rewards)}
        else:
            info = {}

        return obs, reward, done, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
