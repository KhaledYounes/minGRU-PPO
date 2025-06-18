import time

import torch
from omegaconf import OmegaConf

from minigrid_env_modified import Minigrid
from model import ActorCritic


def test_agent(configuration,
               model_path="best_model.pth",
               episodes=5,
               device="cpu"):

    env = Minigrid(env_name=configuration.env_name, realtime_mode=True)

    model = ActorCritic(
        config=configuration,
        observation_shape=env.observation_space_shape,
        action_spaces=env.action_spaces
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    for ep in range(episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)

        hidden_state = model.init_hidden(batch_size=1, device=device)
        done = False
        total_reward = 0.0

        while not done:
            env.render()
            policy_logits, value, hidden_state = model(obs, hidden_state)
            actions = [torch.argmax(logits, dim=1) for logits in policy_logits]
            actions_tensor = torch.stack(actions, dim=1)
            obs_, reward, done, truncated_, info = env.step(actions_tensor.cpu().numpy())
            total_reward += reward
            obs = torch.tensor(obs_, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
            time.sleep(1)
        env.render()

        print(f"Episode {ep + 1} finished with total reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    config = OmegaConf.load("../../configurations/minigrid.yaml")
    # config.recurrent_type = "GRU"
    test_agent(
        configuration=config,
        model_path="../../models/MiniGrid-MemoryS9-v0_800_GRU.pth",
        episodes=5,
        device="cpu"
    )
