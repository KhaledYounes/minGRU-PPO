import time

import torch
from omegaconf import OmegaConf

from minigrid_env_modified import Minigrid
from model import ActorCritic


def test_agent(env_name="MiniGrid-MemoryS9-v0",
               model_path="best_model.pth",
               episodes=5,
               device="cpu"):
    env = Minigrid(env_name=env_name, realtime_mode=True)
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n
    config = OmegaConf.load("../../configurations/minigrid.yaml")
    config.recurrent_type = "minGRU"
    model = ActorCritic(
        config=config,
        observation_shape=obs_shape,
        nb_actions=action_dim,
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
            logits, value, new_hidden = model(obs, hidden_state)
            action = torch.argmax(logits, dim=1).cpu().numpy()
            obs_, reward, done_, truncated_, info = env.step(action)
            total_reward += reward

            if done_ or truncated_:
                done = True
            else:
                hidden_state = new_hidden

            obs = torch.tensor(obs_, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
            time.sleep(1)
        env.render()

        print(f"Episode {ep + 1} finished with total reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    test_agent(
        env_name="MiniGrid-MemoryS9-v0",
        model_path="../../models/model_800_minGRU.pth",
        episodes=5,
        device="cpu"
    )
