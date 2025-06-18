import ale_py
import torch
from omegaconf import OmegaConf

from buffer import RolloutBuffer
from model import ActorCritic
from trainer import Trainer
from utils import create_envs

print("[ALE] Version in use:", ale_py.__version__)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = OmegaConf.load("configurations/tmaze.yaml")

    env, envs = create_envs(configuration=config)

    print("==== Running training with GRU ====")
    buffer = RolloutBuffer(
        config=config,
        base_observation=env.base_observation,
        additional_observation=env.additional_observation,
        action_space_dimensions=env.action_space_dimensions,
        device=device,
    )
    model = ActorCritic(
        config=config,
        base_observation=env.base_observation,
        additional_observation=env.additional_observation,
        action_spaces=env.action_spaces,
    ).to(device)
    print(OmegaConf.to_yaml(config))
    print(model)
    trainer = Trainer(config=config, model=model, envs=envs, has_additional_info=env.has_additional_info, buffer=buffer,
                      device=device)
    trainer.train()
    print("\n==== Completed training with GRU ====\n")

    print("==== Running training with minGRU ====")
    config.recurrent_type = "minGRU"
    config.use_fused_kernel = True
    buffer = RolloutBuffer(
        config=config,
        base_observation=env.base_observation,
        additional_observation=env.additional_observation,
        action_space_dimensions=env.action_space_dimensions,
        device=device,
    )
    model = ActorCritic(
        config=config,
        base_observation=env.base_observation,
        additional_observation=env.additional_observation,
        action_spaces=env.action_spaces,
    ).to(device)
    print(OmegaConf.to_yaml(config))
    print(model)
    trainer = Trainer(config=config, model=model, envs=envs, has_additional_info=env.has_additional_info, buffer=buffer,
                      device=device)
    trainer.train()
    print("\n==== Completed training with minGRU ====")
