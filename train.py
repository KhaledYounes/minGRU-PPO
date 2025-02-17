import ale_py
import torch
from gymnasium.vector import SyncVectorEnv
from omegaconf import OmegaConf

from buffer import RolloutBuffer
from environments.minigrid.minigrid_env_modified import Minigrid
from model import ActorCritic
from trainer import Trainer

print("[ALE] Version in use:", ale_py.__version__)


def create_envs(env_name, num_environments: int = 1):
    return SyncVectorEnv([lambda: Minigrid(env_name) for _ in range(num_environments)])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = OmegaConf.load("configurations/minigrid.yaml")

    envs = create_envs(config.env_name, config.num_envs)

    buffer = RolloutBuffer(
        config=config,
        observation_shape=envs.single_observation_space.shape,
        device=device,
    )

    print("==== Running training with GRU ====")
    model = ActorCritic(
        observation_shape=envs.single_observation_space.shape,
        nb_actions=envs.single_action_space.n,
        config=config
    ).to(device)
    print(model)
    trainer = Trainer(config=config, model=model, envs=envs, buffer=buffer, device=device)
    trainer.train()
    print("\n==== Completed training with GRU ====\n")

    print("==== Running training with minGRU ====")
    config.recurrent_type = "minGRU"
    model = ActorCritic(
        config=config,
        observation_shape=envs.single_observation_space.shape,
        nb_actions=envs.single_action_space.n
    ).to(device)
    print(model)
    trainer = Trainer(config=config, model=model, envs=envs, buffer=buffer, device=device)
    trainer.train()
    print("\n==== Completed training with minGRU ====")
