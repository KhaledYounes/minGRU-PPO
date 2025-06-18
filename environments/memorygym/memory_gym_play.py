import time

import torch
from omegaconf import OmegaConf

from memory_gym_env_modified import MemoryGymWrapper
from model import ActorCritic


def test_agent(configuration,
               model_path="best_model.pth",
               episodes=5,
               device="cpu"):
    env = MemoryGymWrapper(env_name=configuration.env_name, reset_params=configuration.reset_params, realtime_mode=True)

    model = ActorCritic(
        config=configuration,
        base_observation=env.base_observation,
        additional_observation=env.additional_observation,
        action_spaces=env.action_spaces,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    for ep in range(episodes):
        obs, _ = env.reset()
        hidden = model.init_hidden(batch_size=1, device=device)

        done = False
        total_reward = 0.0

        while not done:
            env.render()
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(1).to(device)
            hidden_state = hidden.clone()

            policy_logits, value, hidden = model(obs_tensor, hidden_state)
            actions = [torch.argmax(logits, dim=1) for logits in policy_logits]
            actions_tensor = torch.stack(actions, dim=1)
            obs, reward, done, truncated, info = env.step(actions_tensor.cpu().numpy())
            total_reward = total_reward + float(reward)
            time.sleep(0.2)
        env.render()

        print(f"Episode {ep + 1} finished with total reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    config = OmegaConf.load("../../configurations/endless_mayhem.yaml")
    # config.recurrent_type = "GRU"
    # config.num_layers = 1
    test_agent(
        configuration=config,
        model_path="../../models/Endless-MortarMayhem-v0_50000_GRU.pth",
        episodes=3,
        device="cpu"
    )
