import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gru.stacked_gru import StackedGRU
from mingru.stacked_min_gru import StackedMinGRU


class ActorCritic(nn.Module):
    def __init__(self, config, base_observation, additional_observation, action_spaces):
        super().__init__()
        self.base_observation = base_observation
        self.additional_observation = additional_observation
        self.base_shape = self.base_observation.shape if self.base_observation is not None else []
        self.additional_dim = int(
            np.prod(self.additional_observation.shape)) if self.additional_observation is not None else 0
        self.action_spaces = action_spaces
        self.num_layers = config.num_layers
        self.recurrent_type = config.recurrent_type
        self.use_norm = config.use_norm
        self.use_residual = config.use_residual
        self.hidden_state_size = config.hidden_state_size
        self.hidden_layer_size = config.hidden_layer_size
        self.use_fused_kernel = config.use_fused_kernel

        if self.additional_dim > 0:
            self.additional_encoder = nn.Sequential(
                nn.Linear(self.additional_dim, config.num_vec_encoder_units),
                nn.ReLU()
            )
            for m in self.additional_encoder:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, np.sqrt(2))
            self.additional_feat_dim = config.num_vec_encoder_units
        else:
            self.additional_encoder = None
            self.additional_feat_dim = 0

        self.base_obs_is_visual = len(self.base_shape) > 1

        if self.base_obs_is_visual:
            c, h, w = self.base_shape
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            self.cnn.apply(ActorCritic._init_cnn_weights)
            self.cnn_out_dim = self._get_cnn_out_dim(self.base_shape)
            in_features_next_layer = self.cnn_out_dim
        else:
            in_features_next_layer = int(np.prod(self.base_shape))

        merged_dim = in_features_next_layer + self.additional_feat_dim

        if self.recurrent_type == "minGRU":
            self.gru = StackedMinGRU(input_size=merged_dim, hidden_size=self.hidden_state_size,
                                     num_layers=self.num_layers,
                                     use_norm=self.use_norm, use_residual=self.use_residual,
                                     use_fused_kernel=self.use_fused_kernel)
        else:
            self.gru = StackedGRU(input_size=merged_dim, hidden_size=self.hidden_state_size,
                                  num_layers=self.num_layers,
                                  use_norm=self.use_norm, use_residual=self.use_residual)

        initialize_gru_weights(self.gru)

        self.fc = nn.Linear(self.hidden_state_size, self.hidden_layer_size)
        nn.init.orthogonal_(self.fc.weight, np.sqrt(2))

        self.lin_policy = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        self.lin_value = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        self.actors = nn.ModuleList(
            [nn.Linear(self.hidden_layer_size, nb_actions) for nb_actions in self.action_spaces])
        for actor in self.actors:
            nn.init.orthogonal_(actor.weight, np.sqrt(0.01))

        self.critic = nn.Linear(self.hidden_layer_size, 1)
        nn.init.orthogonal_(self.critic.weight, 1)

    def _get_cnn_out_dim(self, observation_space) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space)
            dummy_output = self.cnn(dummy_input)
            return int(dummy_output.view(1, -1).size(1))

    @staticmethod
    def _init_cnn_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, np.sqrt(2))

    def forward(self, obs, hidden=None):
        base, additional = obs
        batch_size, seq_len = base.size(0), base.size(1)
        if seq_len == 1:
            if self.base_obs_is_visual:
                base = base.squeeze(1)
                base = self.cnn(base)
            if self.additional_dim > 0:
                additional = self.additional_encoder(additional)
                full_obs = torch.cat([base, additional], dim=1)
            else:
                full_obs = base
            if self.base_obs_is_visual:
                full_obs = full_obs.unsqueeze(1)
            gru_out, new_hidden = self.gru(full_obs, hidden)
            fc_out = F.relu(self.fc(gru_out))
            lin_policy_out = F.relu(self.lin_policy(fc_out))
            lin_value_out = F.relu(self.lin_value(fc_out))
            actors_logits = [actor(lin_policy_out).view(batch_size, -1) for actor in self.actors]
            critic_values = self.critic(lin_value_out).view(batch_size)
        else:
            if self.base_obs_is_visual:
                base = base.view(batch_size * seq_len, *self.base_shape)
                base = self.cnn(base)
                base = base.view(batch_size, seq_len, self.cnn_out_dim)
            if self.additional_dim > 0:
                additional = self.additional_encoder(additional)
                full_obs = torch.cat([base, additional], dim=2)
            else:
                full_obs = base
            gru_out, new_hidden = self.gru(full_obs, hidden)
            gru_out = gru_out.reshape(batch_size * seq_len, -1)
            fc_out = F.relu(self.fc(gru_out))
            lin_policy_out = F.relu(self.lin_policy(fc_out))
            lin_value_out = F.relu(self.lin_value(fc_out))
            actors_logits = [actor(lin_policy_out).view(batch_size, seq_len, -1) for actor in self.actors]
            critic_values = self.critic(lin_value_out).view(batch_size, seq_len)
        return actors_logits, critic_values, new_hidden.squeeze(2)

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_state_size, device=device)


def initialize_gru_weights(model):
    if hasattr(model, 'blocks'):
        for block in model.blocks:
            if hasattr(block, 'gru'):
                if isinstance(block.gru, nn.GRU):
                    for name, param in block.gru.named_parameters():
                        if 'bias' in name:
                            nn.init.constant_(param, 0)
                        elif 'weight' in name:
                            nn.init.orthogonal_(param, np.sqrt(2))
                if hasattr(block, 'res_proj') and isinstance(block.res_proj, nn.Linear):
                    nn.init.orthogonal_(block.res_proj.weight, np.sqrt(2))
            elif hasattr(block, 'min_gru'):
                init_min_gru(block.min_gru)
                if hasattr(block, 'res_proj') and isinstance(block.res_proj, nn.Linear):
                    nn.init.kaiming_uniform_(block.res_proj.weight, np.sqrt(2))


def init_min_gru(m: nn.Module):
    nn.init.kaiming_uniform_(m.linear_z.weight, np.sqrt(2))
    nn.init.kaiming_uniform_(m.linear_h.weight, np.sqrt(2))
    nn.init.constant_(m.linear_z.bias, 0)
    nn.init.constant_(m.linear_h.bias, 0)
