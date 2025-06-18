import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import trange


def linear_decay(initial_value, final_value, current_step, total_steps):
    return initial_value + (final_value - initial_value) * (current_step / total_steps)


def deconstruct_obs(has_additional_info, obs, device):
    if has_additional_info:
        base, additional = obs["base_observation"], obs["additional_observation"]
        return (
            torch.from_numpy(base.copy()).unsqueeze(1).float().to(device),
            torch.from_numpy(additional.copy()).float().to(device),
        )
    base = torch.from_numpy(obs.copy()).unsqueeze(1).float().to(device)
    return base, None


def deconstruct_batch(has_additional_info, batch, device):
    if has_additional_info:
        b_adv, b_base_obs, b_additional_obs, b_actions, b_old_log_probs, b_old_state_values, b_init_hidden, masks = [
            x.to(device)
            for x in batch
        ]
        return b_adv, b_base_obs, b_additional_obs, b_actions, b_old_log_probs, b_old_state_values, b_init_hidden, masks
    else:
        b_adv, b_base_obs, b_actions, b_old_log_probs, b_old_state_values, b_init_hidden, masks = [
            x.to(device)
            for x in batch
        ]
        return b_adv, b_base_obs, None, b_actions, b_old_log_probs, b_old_state_values, b_init_hidden, masks


class Trainer:
    def __init__(self, config, model, envs, has_additional_info, buffer, device):
        self.model = model
        self.envs = envs
        self.has_additional_info = has_additional_info
        self.buffer = buffer
        self.device = device
        self.config = config
        self.env_name = config.env_name
        self.num_envs = config.num_envs
        self.T = config.T
        self.epochs = config.epochs
        self.gamma = config.gamma
        self.lamda = config.lamda
        self.learning_rate_initial = config.learning_rate_initial
        self.learning_rate_final = config.learning_rate_final
        self.clip_range_initial = config.clip_range_initial
        self.clip_range_final = config.clip_range_final
        self.entropy_loss_coefficient_initial = config.entropy_loss_coefficient_initial
        self.entropy_loss_coefficient_final = config.entropy_loss_coefficient_final
        self.value_loss_coefficient = config.value_loss_coefficient
        self.max_grad_norm = config.max_grad_norm
        self.updates = config.updates
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate_initial)
        self.total_rewards_for_iteration = [[] for _ in range(self.num_envs)]
        self.max_reward = -float("inf")

    def train(self):
        obs, _ = self.envs.reset()
        hidden = self.model.init_hidden(self.num_envs, self.device)
        pbar = trange(self.updates, desc="PPO Training", unit="update")
        for iteration in pbar:
            lr = linear_decay(initial_value=self.learning_rate_initial, final_value=self.learning_rate_final,
                              current_step=iteration, total_steps=self.updates)
            clip_range = linear_decay(initial_value=self.clip_range_initial, final_value=self.clip_range_initial,
                                      current_step=iteration, total_steps=self.updates)
            entropy_loss_coefficient = linear_decay(initial_value=self.entropy_loss_coefficient_initial,
                                                    final_value=self.entropy_loss_coefficient_final,
                                                    current_step=iteration, total_steps=self.updates)
            if (iteration + 1) % (self.updates / 10) == 0:
                torch.save(self.model.state_dict(),
                           f"models/{self.env_name}_{iteration + 1}_{self.model.recurrent_type}.pth")
            self.total_rewards_for_iteration = [[] for _ in range(self.num_envs)]
            with torch.no_grad():
                for t in range(self.T):
                    base_obs_tensor, additional_obs_tensor = deconstruct_obs(self.has_additional_info, obs, self.device)
                    hidden_states = hidden.clone()

                    policy_logits, values, hidden = self.model((base_obs_tensor, additional_obs_tensor), hidden_states)
                    dists = [Categorical(logits=logits) for logits in policy_logits]
                    actions = [dist.sample() for dist in dists]
                    log_probs = [dist.log_prob(action) for dist, action in zip(dists, actions)]
                    actions_tensor = torch.stack(actions, dim=1)
                    log_probs_tensor = torch.stack(log_probs, dim=1)
                    obs, rewards, dones, truncated, infos = self.envs.step(actions_tensor.cpu().numpy())
                    reward_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
                    done_tensor = torch.tensor(dones | truncated, device=self.device, dtype=torch.float32)
                    self.buffer.add(t, base_obs_tensor.squeeze(1), additional_obs_tensor, actions_tensor,
                                    log_probs_tensor, values, hidden_states,
                                    reward_tensor, done_tensor)
                    if infos:
                        for i, (done_info, reward_info) in enumerate(
                                zip(infos['final_info']['_reward'], infos['final_info']['reward'])):
                            if done_info:
                                hidden[:, i] = self.model.init_hidden(1, self.device).squeeze(1)
                                self.total_rewards_for_iteration[i].append(reward_info)
                                if reward_info > self.max_reward:
                                    self.max_reward = reward_info
                base_obs_tensor, additional_obs_tensor = deconstruct_obs(self.has_additional_info, obs, self.device)
                _, next_values, _ = self.model((base_obs_tensor, additional_obs_tensor), hidden)
            self.buffer.state_values[:, self.T] = next_values
            self.buffer.compute_gae(self.gamma, self.lamda)
            self.train_mini_batches(clip_range, lr, entropy_loss_coefficient)
            avg_reward = np.mean([np.mean(r) for r in self.total_rewards_for_iteration if r])
            pbar.set_postfix({"avg_reward": f"{avg_reward:.4f}", "max_reward": f"{self.max_reward:.1f}"})

    def train_mini_batches(self, clip_range, lr, entropy_loss_coefficient):
        data_loader = self.buffer.get_mini_batches()
        for _ in range(self.epochs):
            for batch in data_loader:
                b_adv, b_base_obs, b_additional_obs, b_actions, b_old_log_probs, b_old_state_values, b_init_hidden, masks = deconstruct_batch(
                    self.has_additional_info, batch, self.device)
                policy_logits, values, _ = self.model((b_base_obs, b_additional_obs),
                                                      hidden=b_init_hidden.permute(1, 0, 2).contiguous())
                dists = [Categorical(logits=logits) for logits in policy_logits]
                log_probs = [dist.log_prob(b_actions[:, :, i]) for i, dist in enumerate(dists)]
                entropies = [dist.entropy() for dist in dists]
                log_probs_tensor = torch.stack(log_probs, dim=2)
                entropies_tensor = torch.stack(entropies, dim=2)

                b_adv = b_adv[masks]
                b_old_state_values = b_old_state_values[masks]
                values = values[masks]
                b_old_log_probs = b_old_log_probs[masks]
                log_probs_tensor = log_probs_tensor[masks]
                entropies_tensor = entropies_tensor[masks]

                b_adv = b_adv.unsqueeze(1).repeat(1, self.buffer.action_space_dimensions)
                b_old_state_values = b_old_state_values.unsqueeze(1).repeat(1, self.buffer.action_space_dimensions)
                values = values.unsqueeze(1).repeat(1, self.buffer.action_space_dimensions)

                ratios = torch.exp(log_probs_tensor - b_old_log_probs)
                policy_loss1 = b_adv * ratios
                policy_loss2 = b_adv * torch.clamp(ratios, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
                returns = b_adv + b_old_state_values
                value_loss1 = F.mse_loss(returns, values, reduction='none')
                value_loss2 = F.mse_loss(returns, torch.clamp(values, b_old_state_values - clip_range,
                                                              b_old_state_values + clip_range),
                                         reduction='none')
                value_loss = torch.max(value_loss1, value_loss2).mean()
                entropy_loss = -entropies_tensor.mean()
                loss = policy_loss + self.value_loss_coefficient * value_loss + entropy_loss_coefficient * entropy_loss
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
