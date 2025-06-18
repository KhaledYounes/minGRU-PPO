import csv
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.distributions import Categorical

from environments.cartpole.cartpole_env_modified import CartPole
from environments.memorygym.memory_gym_env_modified import MemoryGymWrapper
from environments.minigrid.minigrid_env_modified import Minigrid
from model import ActorCritic
from utils import create_envs


def _deconstruct_obs(has_info, obs, device):
    if has_info:
        base, add = obs["base_observation"], obs["additional_observation"]
        return (
            torch.from_numpy(base.copy()).unsqueeze(0).unsqueeze(1).float().to(device),
            torch.from_numpy(add.copy()).unsqueeze(0).float().to(device),
        )
    base = torch.from_numpy(obs.copy()).unsqueeze(0).unsqueeze(1).float().to(device)
    return base, None


def _shift_seed(cfg):
    cfg = deepcopy(cfg)
    if hasattr(cfg, "reset_params") and isinstance(cfg.reset_params, dict) and "start-seed" in cfg.reset_params:
        cfg.reset_params["start-seed"] += random.randint(1_000_000, 10_000_000)
    return cfg


def _append(path, row):
    path = Path(path)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["environment_name", "recurrent_type", "iteration", "Number of parameters", "average_reward"])
        w.writerow(row)


def _create_single_env(configuration):
    if configuration.env_type == "Minigrid":
        return Minigrid(env_name=configuration.env_name)
    if configuration.env_type == "MemoryGym":
        return MemoryGymWrapper(env_name=configuration.env_name, reset_params=configuration.reset_params)
    if configuration.env_type == "CartPoleMasked":
        return CartPole(max_steps=configuration.T, mask_velocity=True)
    raise ValueError


def evaluate(model, config, iteration, device="cpu", csv_path="evaluation_results.csv"):
    NUM_SEEDS = 10
    NUM_ENVS = 10
    NUM_EPISODES = 10
    model = model.to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    returns = []
    for _ in range(NUM_SEEDS):
        cfg = _shift_seed(config)
        returns_seed = []
        for _ in range(NUM_ENVS):
            env = _create_single_env(cfg)
            for _ in range(NUM_EPISODES):
                obs, _ = env.reset()
                has_info = model.additional_observation is not None
                hidden = model.init_hidden(1, device)
                done = False
                ep_return = 0.0
                while not done:
                    base, add = _deconstruct_obs(has_info, obs, device)
                    logits, _, hidden = model((base, add), hidden)
                    action = np.stack([Categorical(logits=logit).sample().item() for logit in logits], -1)
                    obs, reward, done, truncated, _ = env.step(action)
                    ep_return += reward
                    if truncated:
                        done = True
                returns_seed.append(ep_return)
            env.close()
            avg_seed = float(np.mean(returns_seed))
            returns.append(avg_seed)
    avg = float(np.mean(returns))
    _append(csv_path, [config.env_name, model.recurrent_type, iteration, total_params, f"{avg:.4f}"])
    return avg


def eval_models(
        models_dir: str | Path,
        config_path: str | Path,
        device: str = "cuda",
        pattern: str = "*.pth",
):
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        raise FileNotFoundError(f"{models_dir} is not a valid directory")

    config = OmegaConf.load(config_path)
    env, _ = create_envs(configuration=config)

    for model_path in sorted(models_dir.glob(pattern)):
        model = ActorCritic(
            config=config,
            base_observation=env.base_observation,
            additional_observation=env.additional_observation,
            action_spaces=env.action_spaces,
        ).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)

        evaluate(model, config, int(Path(model_path).stem.split("_")[-2]), device, models_dir.stem + "csv")


if __name__ == "__main__":
    eval_models(models_dir="models/mmag/gru", config_path="configurations/mortar_mayhem_grid.yaml")
