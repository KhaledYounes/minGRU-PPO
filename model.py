import torch
import torch.nn as nn
import torch.nn.functional as F

from mingru.stacked_min_gru import StackedMinGRU


class ActorCritic(nn.Module):
    def __init__(self, config, observation_shape, nb_actions):
        super().__init__()
        self.observation_shape = observation_shape
        self.num_layers = config.num_layers
        self.recurrent_type = config.recurrent_type
        self.use_norm = config.use_norm
        self.use_residual = config.use_residual
        self.hidden_state_size = config.hidden_state_size
        self.hidden_layer_size = config.hidden_layer_size

        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.cnn_out_dim = self._get_cnn_out_dim(observation_shape)
        if self.recurrent_type == "minGRU":
            self.gru = StackedMinGRU(input_size=self.cnn_out_dim, hidden_size=self.hidden_state_size,
                                     num_layers=self.num_layers,
                                     use_norm=self.use_norm, use_residual=self.use_residual)
        else:
            self.gru = nn.GRU(input_size=self.cnn_out_dim, hidden_size=self.hidden_state_size,
                              num_layers=self.num_layers,
                              batch_first=True)
        self.fc = nn.Linear(self.hidden_state_size, self.hidden_layer_size)
        self.actor = nn.Linear(self.hidden_layer_size, nb_actions)
        self.critic = nn.Linear(self.hidden_layer_size, 1)

    def _get_cnn_out_dim(self, observation_space) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space)
            dummy_output = self.cnn(dummy_input)
            return int(dummy_output.view(1, -1).size(1))

    def forward(self, x, hidden=None):
        batch_size, seq_len = x.size(0), x.size(1)
        if seq_len == 1:
            x = x.squeeze(1)
            x = self.cnn(x)
            x = x.unsqueeze(1)
            gru_out, new_hidden = self.gru(x, hidden)
            fc_out = F.relu(self.fc(gru_out))
            actor_logits = self.actor(fc_out).view(batch_size, -1)
            critic_values = self.critic(fc_out).view(batch_size)
        else:
            x = x.view(batch_size * seq_len, *self.observation_shape)
            x = self.cnn(x)
            x = x.view(batch_size, seq_len, self.cnn_out_dim)
            gru_out, new_hidden = self.gru(x, hidden)
            gru_out = gru_out.reshape(batch_size * seq_len, -1)
            fc_out = F.relu(self.fc(gru_out))
            actor_logits = self.actor(fc_out).view(batch_size, seq_len, -1)
            critic_values = self.critic(fc_out).view(batch_size, seq_len)
        return actor_logits, critic_values, new_hidden.squeeze(2)

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_state_size, device=device)
