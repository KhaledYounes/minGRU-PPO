import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader


class RolloutBuffer:
    def __init__(self, config, base_observation, additional_observation, action_space_dimensions, device):
        self.base_observation = base_observation
        self.additional_observation = additional_observation
        self.action_space_dimensions = action_space_dimensions
        self.device = device
        self.sequence_length = config.sequence_length
        self.num_envs = config.num_envs
        self.T = config.T
        self.mini_batch_size = config.mini_batch_size
        self.hidden_state_size = config.hidden_state_size
        self.num_layers = config.num_layers

        self.base_observation = torch.zeros((self.num_envs, self.T, *base_observation.shape), dtype=torch.float32,
                                            device=device)
        self.additional_observation = torch.zeros((self.num_envs, self.T, *additional_observation.shape),
                                                  dtype=torch.float32,
                                                  device=device) if additional_observation is not None else None
        self.advantages = torch.zeros((self.num_envs, self.T), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.num_envs, self.T, action_space_dimensions),
                                   dtype=torch.long, device=device)
        self.log_probs = torch.zeros((self.num_envs, self.T, action_space_dimensions),
                                     dtype=torch.float32, device=device)
        self.state_values = torch.zeros((self.num_envs, self.T), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.num_envs, self.T), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.num_envs, self.T), dtype=torch.float32, device=device)
        self.hidden = torch.zeros((self.num_envs, self.T, self.num_layers, self.hidden_state_size),
                                  dtype=torch.float32, device=device)

    def add(self, t, base_observation, additional_observation, action, log_prob, state_value, new_hidden, reward, done):
        self.base_observation[:, t] = base_observation
        if self.additional_observation is not None:
            self.additional_observation[:, t] = additional_observation
        self.actions[:, t] = action
        self.log_probs[:, t] = log_prob
        self.state_values[:, t] = state_value
        self.hidden[:, t] = new_hidden.permute(1, 0, 2).contiguous()
        self.rewards[:, t] = reward
        self.dones[:, t] = done

    def compute_gae(self, last_value, gamma, gae_lambda):
        last_advantage = 0
        non_terminal = 1.0 - self.dones
        for t in reversed(range(self.T)):
            last_value = last_value * non_terminal[:, t]
            last_advantage = last_advantage * non_terminal[:, t]
            delta = self.rewards[:, t] + gamma * last_value - self.state_values[:, t]
            last_advantage = delta + gamma * gae_lambda * last_advantage
            self.advantages[:, t] = last_advantage
            last_value = self.state_values[:, t]

    def get_mini_batches(self):
        base_observation_list = []
        additional_observation_list = []
        actions_list = []
        log_probs_list = []
        state_values_list = []
        advantages_list = []
        hidden_list = []
        loss_mask_list = []

        for env in range(self.num_envs):
            done_idxs = self.dones[env].nonzero().squeeze().tolist()
            if not isinstance(done_idxs, list):
                done_idxs = [done_idxs]
            if len(done_idxs) == 0 or done_idxs[-1] != self.T - 1:
                done_idxs.append(self.T - 1)
            episode_start_idx = 0
            for done_idx in done_idxs:
                episode_end_idx = done_idx + 1

                episode_base_observation = self.base_observation[env, episode_start_idx:episode_end_idx]
                if self.additional_observation is not None:
                    episode_additional_observation = self.additional_observation[env, episode_start_idx:episode_end_idx]
                else:
                    episode_additional_observation = None
                episode_actions = self.actions[env, episode_start_idx:episode_end_idx]
                episode_log_probs = self.log_probs[env, episode_start_idx:episode_end_idx]
                episode_state_values = self.state_values[env, episode_start_idx:episode_end_idx]
                episode_advantages = self.advantages[env, episode_start_idx:episode_end_idx]

                episode_length = episode_base_observation.size(0)

                actual_sequence_length = self.sequence_length if self.sequence_length > 0 else episode_length

                for seq_start in range(0, episode_length, actual_sequence_length):
                    seq_end = min(seq_start + actual_sequence_length, episode_length)

                    seq_base_observation = episode_base_observation[seq_start:seq_end]
                    if self.additional_observation is not None:
                        seq_additional_observation = episode_additional_observation[seq_start:seq_end]
                    else:
                        seq_additional_observation = None
                    seq_actions = episode_actions[seq_start:seq_end]
                    seq_log_probs = episode_log_probs[seq_start:seq_end]
                    seq_state_values = episode_state_values[seq_start:seq_end]
                    seq_advantages = episode_advantages[seq_start:seq_end]
                    seq_hidden = self.hidden[env, episode_start_idx + seq_start]

                    base_observation_list.append(seq_base_observation)
                    if seq_additional_observation is not None:
                        additional_observation_list.append(seq_additional_observation)
                    actions_list.append(seq_actions)
                    log_probs_list.append(seq_log_probs)
                    state_values_list.append(seq_state_values)
                    advantages_list.append(seq_advantages)
                    hidden_list.append(seq_hidden)

                    loss_mask_list.append(
                        torch.ones(seq_base_observation.size(0), dtype=torch.bool, device=seq_base_observation.device))
                episode_start_idx = episode_end_idx

        padded_base_observation = pad_sequence(base_observation_list, batch_first=True, padding_value=0)
        if len(additional_observation_list) > 0:
            padded_additional_observation = pad_sequence(additional_observation_list, batch_first=True, padding_value=0)
        else:
            padded_additional_observation = None
        padded_actions = pad_sequence(actions_list, batch_first=True, padding_value=0)
        padded_log_probs = pad_sequence(log_probs_list, batch_first=True, padding_value=0)
        padded_state_values = pad_sequence(state_values_list, batch_first=True, padding_value=0)
        padded_advantages = pad_sequence(advantages_list, batch_first=True, padding_value=0)
        padded_loss_mask = pad_sequence(loss_mask_list, batch_first=True, padding_value=0)
        init_hidden = torch.stack(hidden_list, dim=0)

        if padded_additional_observation is not None:
            dataset = TensorDataset(
                padded_advantages,
                padded_base_observation,
                padded_additional_observation,
                padded_actions,
                padded_log_probs,
                padded_state_values,
                init_hidden,
                padded_loss_mask
            )
        else:
            dataset = TensorDataset(
                padded_advantages,
                padded_base_observation,
                padded_actions,
                padded_log_probs,
                padded_state_values,
                init_hidden,
                padded_loss_mask
            )

        return DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=True)
