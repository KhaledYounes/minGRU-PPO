import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader


class RolloutBuffer:
    def __init__(self, config, observation_shape, device):
        self.observation_shape = observation_shape
        self.device = device
        self.num_envs = config.num_envs
        self.T = config.T
        self.hidden_state_size = config.hidden_state_size
        self.num_layers = config.num_layers
        self.advantages = torch.zeros((self.num_envs, self.T), dtype=torch.float32, device=device)
        self.states = torch.zeros((self.num_envs, self.T, *observation_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.num_envs, self.T), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((self.num_envs, self.T), dtype=torch.float32, device=device)
        self.state_values = torch.zeros((self.num_envs, self.T + 1), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.num_envs, self.T), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.num_envs, self.T), dtype=torch.float32, device=device)
        self.hidden = torch.zeros((self.num_envs, self.T + 1, self.num_layers, self.hidden_state_size),
                                  dtype=torch.float32,
                                  device=device)

    def reset(self):
        self.advantages.zero_()
        self.states.zero_()
        self.actions.zero_()
        self.log_probs.zero_()
        self.state_values.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.hidden.zero_()

    def add(self, t, state, action, log_prob, state_value, new_hidden, reward, done):
        self.states[:, t] = state
        self.actions[:, t] = action
        self.log_probs[:, t] = log_prob
        self.state_values[:, t] = state_value
        self.hidden[:, t + 1] = new_hidden.permute(1, 0, 2).contiguous()
        self.rewards[:, t] = reward
        self.dones[:, t] = done

    def compute_gae(self, gamma, gae_lambda):
        for i in range(self.num_envs):
            adv = 0
            for t in reversed(range(self.T)):
                non_terminal = 1.0 - self.dones[i, t]
                delta = self.rewards[i, t] + gamma * self.state_values[i, t + 1] * non_terminal - self.state_values[
                    i, t]
                adv = delta + gamma * gae_lambda * non_terminal * adv
                self.advantages[i, t] = adv

    def get_mini_batches(self, sequence_length, mini_batch_size):
        states_list = []
        actions_list = []
        log_probs_list = []
        state_values_list = []
        advantages_list = []
        hidden_list = []
        loss_mask_list = []

        for env in range(self.num_envs):
            done_idxs = (self.dones[env]).nonzero(as_tuple=False).squeeze().tolist()
            if not isinstance(done_idxs, list):
                done_idxs = [done_idxs]
            if len(done_idxs) == 0 or done_idxs[-1] != self.T - 1:
                done_idxs.append(self.T - 1)

            episode_start_idx = 0
            for d in done_idxs:
                episode_end_idx = d + 1

                episode_states = self.states[env, episode_start_idx:episode_end_idx]
                episode_actions = self.actions[env, episode_start_idx:episode_end_idx]
                episode_log_probs = self.log_probs[env, episode_start_idx:episode_end_idx]
                episode_state_values = self.state_values[env, episode_start_idx:episode_end_idx]
                episode_advantages = self.advantages[env, episode_start_idx:episode_end_idx]

                episode_length = episode_states.size(0)

                for seq_start in range(0, episode_length, sequence_length):
                    seq_end = min(seq_start + sequence_length, episode_length)

                    seq_states = episode_states[seq_start:seq_end]
                    seq_actions = episode_actions[seq_start:seq_end]
                    seq_log_probs = episode_log_probs[seq_start:seq_end]
                    seq_state_values = episode_state_values[seq_start:seq_end]
                    seq_advantages = episode_advantages[seq_start:seq_end]
                    seq_hidden = self.hidden[env, episode_start_idx + seq_start]

                    states_list.append(seq_states)
                    actions_list.append(seq_actions)
                    log_probs_list.append(seq_log_probs)
                    state_values_list.append(seq_state_values)
                    advantages_list.append(seq_advantages)
                    hidden_list.append(seq_hidden)

                    loss_mask_list.append(torch.ones(seq_states.size(0), dtype=torch.bool, device=seq_states.device))
                episode_start_idx = episode_end_idx

        padded_states = pad_sequence(states_list, batch_first=True, padding_value=0)
        padded_actions = pad_sequence(actions_list, batch_first=True, padding_value=0)
        padded_log_probs = pad_sequence(log_probs_list, batch_first=True, padding_value=0)
        padded_state_values = pad_sequence(state_values_list, batch_first=True, padding_value=0)
        padded_advantages = pad_sequence(advantages_list, batch_first=True, padding_value=0)
        padded_loss_mask = pad_sequence(loss_mask_list, batch_first=True, padding_value=0)
        init_hidden = torch.stack(hidden_list, dim=0)

        dataset = TensorDataset(
            padded_advantages,
            padded_states,
            padded_actions,
            padded_log_probs,
            padded_state_values,
            init_hidden,
            padded_loss_mask
        )
        return DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)
