from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit

# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features
def bottle(f, x_tuple):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))  # x[0]: x_tuple (tensor), x[1]: x_sizes (size)
    y_size = y.size()
    return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])


class TransitionModel(jit.ScriptModule):
    __constants__ = ['min_std_dev']

    def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation=F.relu, min_std_dev=0.1):
        super(TransitionModel, self).__init__()
        self.act_fn = activation
        self.min_std_dev = min_std_dev

        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)  # s_{t-1}, a_{t-1}
        self.rnn = nn.GRUCell(belief_size, belief_size)  # deterministic state model: h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)  # stochastic state model: s_t ~ p(s_t | h_t)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)  # \mu, \sigma ~ transition model(prior)

        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)  # \mu, \sigma ~ encoder(posterior): q(s_t | h_t, o_t): 이때 o_t는 encoder를 통과한 임베딩


    @jit.script_method
    def forward(self, prev_state:torch.Tensor, actions:torch.Tensor, prev_belief:torch.Tensor, observations:Optional[torch.Tensor]=None, nonterminals:Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        """
        If observations is None, we only compute prior states(imagination step)
        
        - prev_state: (batch_size, state_size)
        - actions: (seq_len, batch_size, action_size)
        - prev_belief: (batch_size, belief_size)
        """
        T = actions.size(0) + 1
        
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T

        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state

        for t in range(T - 1):
            """
            이 for loop에서 `t`는 t-1 시점을, `t + 1`, `t_`는 t 시점을 나타냄
            """
            # _state: s_{t-1} (If observation is None, use prior state (transition model); else use posterior state(encoder))
            _state = prior_states[t] if observations is None else posterior_states[t]
            _state = _state if nonterminals is None else _state * nonterminals[t]

            # Compute belief (deterministic state model): h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
            hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=-1)))  # s_{t-1}, a_{t-1}
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])  # h_t = f(h_{t-1}, s_{t-1}, a_{t-1})

            # Compute prior state from belief (stochastic state model): s_t ~ p(s_t | h_t)
            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))  # transition model: s_t ~ p(s_t | h_t)
            prior_means[t + 1], _prior_std_devs = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_devs) + self.min_std_dev  # std_dev는 양수여야 하므로 softplus 통과
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.rand_like(prior_means[t + 1])  # reparameterization trick

            if observations is not None:
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                hidden = self.act_fn(self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))  # encoder: s_t ~ q(s_t | h_t, o_t)
                posterior_means[t + 1], _posterior_std_devs = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_devs) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.rand_like(posterior_means[t + 1])  # reparameterization trick

        hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]

        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
        
        return hidden
    
class Encoder(jit.ScriptModule):
    def __init__(self, embedding_size, activation=F.relu):
        super(Encoder, self).__init__()
        self.activation = activation
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2)         # (batch, 3x64x64) -> (batch, 32x31x31)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)        # (batch, 32x31x31) -> (batch, 64x14x14)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)       # (batch, 64x14x14) -> (batch, 128x6x6)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)      # (batch, 128x6x6) -> (batch, 256x2x2)
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)  # (batch, 256x2x2) -> (batch, 1024) or (batch, embedding_size)
    
    @jit.script_method
    def forward(self, observations:torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.conv1(observations))
        hidden = self.activation(self.conv2(hidden))
        hidden = self.activation(self.conv3(hidden))
        hidden = self.activation(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)
        return self.fc(hidden)

'''
class nnEncoder(nn.Module):
    def __init__(self, embedding_size, activation=F.relu):
        super(nnEncoder, self).__init__()
        self.activation = activation
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)

    def forward(self, observations:torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.conv1(observations))
        hidden = self.activation(self.conv2(hidden))
        hidden = self.activation(self.conv3(hidden))
        hidden = self.activation(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)
        return self.fc(hidden)
'''

class ObservationModel(jit.ScriptModule):
    __constants__ = ['embedding_size']

    def __init__(self, belief_size, state_size, embedding_size, activation=F.relu):
        super(ObservationModel, self).__init__()
        self.act_fn = activation
        self.embedding_size = embedding_size

        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.deconv1 = nn.ConvTranspose2d(in_channels=embedding_size, out_channels=128, kernel_size=5, stride=2)    # (batch, embedding_sizex1x1) -> (batch, 128x5x5)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2)                # (batch, 128x5x5) -> (batch, 64x13x13)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2)                 # (batch, 64x13x13) -> (batch, 32x28x28)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2)                  # (batch, 32x28x28) -> (batch, 3x64x64)

    @jit.script_method
    def forward(self, belief, state):
        hidden = self.fc1(torch.cat([belief, state], dim=1))
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.deconv1(hidden))
        hidden = self.act_fn(self.deconv2(hidden))
        hidden = self.act_fn(self.deconv3(hidden))
        observation = self.deconv4(hidden)
        return observation
    

class RewardModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, activation=F.relu):
        super(RewardModel, self).__init__()
        self.act_fn = activation

        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    @jit.script_method
    def forward(self, belief:torch.Tensor, state:torch.Tensor) -> torch.Tensor:
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        reward = self.fc3(hidden).squeeze(dim=1)
        return reward