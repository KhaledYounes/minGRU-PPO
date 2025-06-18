import time

import torch
from omegaconf import OmegaConf

from cartpole_env_modified import CartPole
from model import ActorCritic


def test_agent(configuration,
               model_path="best_model.pth",
               episodes=5,
               device="cpu"):
    env = CartPole(max_steps=configuration.T, realtime_mode=True)

    model = ActorCritic(
        config=configuration,
        observation_shape=env.observation_space.shape,
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
    config = OmegaConf.load("../../configurations/cartpole.yaml")
    # config.recurrent_type = "GRU"
    # config.num_layers = 1
    test_agent(
        configuration=config,
        model_path="../../models/CartPoleMasked_300_GRU.pth",
        episodes=5,
        device="cpu"
    )
