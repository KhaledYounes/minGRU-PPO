from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from environments.cartpole.cartpole_env_modified import CartPole
from environments.memorygym.memory_gym_env_modified import MemoryGymWrapper
from environments.minigrid.minigrid_env_modified import Minigrid
from environments.tmaze.tmaze_env_wrapper import TMaze


def create_envs(configuration):
    if configuration.env_type == "Minigrid":
        return Minigrid(env_name=configuration.env_name), AsyncVectorEnv(
            env_fns=[lambda: Minigrid(env_name=configuration.env_name) for _ in range(configuration.num_envs)],
            autoreset_mode=AutoresetMode.SAME_STEP)
    elif configuration.env_type == "MemoryGym":
        return MemoryGymWrapper(env_name=configuration.env_name,
                                reset_params=configuration.reset_params), AsyncVectorEnv(
            env_fns=[lambda: MemoryGymWrapper(env_name=configuration.env_name, reset_params=configuration.reset_params)
                     for _ in range(configuration.num_envs)], autoreset_mode=AutoresetMode.SAME_STEP)
    elif configuration.env_type == "CartPoleMasked":
        return CartPole(mask_velocity=True), AsyncVectorEnv(
            env_fns=[lambda: CartPole(max_steps=configuration.T, mask_velocity=True) for _ in
                     range(configuration.num_envs)],
            autoreset_mode=AutoresetMode.SAME_STEP
        )
    elif configuration.env_type == "T-Maze":
        return TMaze(corridor_length=configuration.corridor_length), AsyncVectorEnv(
            env_fns=[lambda: TMaze(corridor_length=configuration.corridor_length) for _ in range(configuration.num_envs)],
            autoreset_mode=AutoresetMode.SAME_STEP)
    else:
        raise ValueError(f"Unsupported environment type: {configuration.env_type}")
