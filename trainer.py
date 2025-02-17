import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange


class Trainer:
    def __init__(self, config, model, envs, buffer, device):
        self.model = model
        self.envs = envs
        self.buffer = buffer
        self.device = device
        self.num_envs = config.num_envs
        self.T = config.T
        self.epochs = config.epochs
        self.mini_batch_size = config.mini_batch_size
        self.gamma = config.gamma
        self.lamda = config.lamda
        self.clip_range = config.clip_range
        self.value_loss_coefficient = config.value_loss_coefficient
        self.entropy_loss_coefficient = config.entropy_loss_coefficient
        self.max_grad_norm = config.max_grad_norm
        self.updates = config.updates
        self.sequence_length = config.sequence_length
        self.learning_rate = config.learning_rate
        self.reset_hidden_state = config.reset_hidden_state
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=0.5,
                                                           total_iters=self.updates)
        self.total_rewards_for_iteration = [[] for _ in range(self.num_envs)]
        self.max_reward = -float("inf")

    def train(self):
        obs, _ = self.envs.reset()
        hidden = self.model.init_hidden(self.num_envs, self.device)
        pbar = trange(self.updates, desc="PPO Training", unit="update")
        for iteration in pbar:
            if (iteration + 1) % 100 == 0:
                torch.save(self.model.state_dict(), f"models/model_{iteration + 1}_{self.model.recurrent_type}.pth")
            self.buffer.reset()
            with torch.no_grad():
                for t in range(self.T):
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(1).to(self.device)
                    logits, values, new_hidden = self.model(obs_tensor, hidden)
                    dist = torch.distributions.Categorical(logits=logits)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                    actions_np = actions.cpu().numpy()
                    obs, rewards, dones, truncated, infos = self.envs.step(actions_np)
                    reward_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
                    done_tensor = torch.tensor(dones | truncated, device=self.device, dtype=torch.float32)
                    self.buffer.add(t, obs_tensor.squeeze(1), actions, log_probs, values, new_hidden, reward_tensor,
                                    done_tensor)
                    if self.reset_hidden_state:
                        done_tensor_for_hidden = done_tensor.view(1, self.num_envs, 1)
                        hidden = new_hidden * (1 - done_tensor_for_hidden)
                    else:
                        hidden = new_hidden
                    if '_reward' in infos and np.any(infos['_reward']):
                        for i, done in enumerate(infos['_reward']):
                            if done:
                                ep_reward = infos['reward'][i]
                                self.total_rewards_for_iteration[i].append(ep_reward)
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(1).to(self.device)
                _, next_values, _ = self.model(obs_tensor, hidden)
            self.buffer.state_values[:, self.T] = next_values
            self.buffer.compute_gae(self.gamma, self.lamda)
            data_loader = self.buffer.get_mini_batches(self.sequence_length, self.mini_batch_size)
            for _ in range(self.epochs):
                for batch in data_loader:
                    b_adv, b_states, b_actions, b_old_log_probs, b_old_state_values, b_init_hidden, masks = [
                        x.to(self.device)
                        for x in batch
                    ]
                    logits, values, _ = self.model(b_states, hidden=b_init_hidden.permute(1, 0, 2).contiguous())
                    dist = torch.distributions.Categorical(logits=logits)
                    log_probs = dist.log_prob(b_actions)
                    entropy = dist.entropy()

                    b_adv = b_adv[masks]
                    b_old_state_values = b_old_state_values[masks]
                    values = values[masks]
                    b_old_log_probs = b_old_log_probs[masks]
                    log_probs = log_probs[masks]
                    entropy = entropy[masks]

                    ratios = torch.exp(log_probs - b_old_log_probs)
                    alpha = 1.0 - (iteration / self.updates)
                    clip_range = self.clip_range * alpha
                    policy_loss1 = b_adv * ratios
                    policy_loss2 = b_adv * torch.clamp(ratios, 1 - clip_range, 1 + clip_range)
                    policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
                    returns = b_adv + b_old_state_values
                    value_loss1 = F.mse_loss(returns, values, reduction='none')
                    value_loss2 = F.mse_loss(returns, torch.clamp(values, b_old_state_values - clip_range,
                                                                  b_old_state_values + clip_range), reduction='none')
                    value_loss = torch.max(value_loss1, value_loss2).mean()
                    entropy_loss = -entropy.mean()
                    loss = policy_loss + self.value_loss_coefficient * value_loss + self.entropy_loss_coefficient * entropy_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
            self.scheduler.step()
            avg_reward = np.mean([np.mean(r) if r else 0 for r in self.total_rewards_for_iteration])
            pbar.set_postfix({"avg_reward": f"{avg_reward:.2f}"})
