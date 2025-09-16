"""
Modified original implmentation of TD3.

Paper: https://arxiv.org/abs/1802.09477
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

HIDDEN_LAYERS = 256
N_FEATURES = 14
LR_A = 10e-5
LR_C = 10e-4
BATCH_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The state input to the Actor and Critic will actually be the enecoded state from the LSTM


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_dim)
        

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = F.relu(self.fc3(a))
        a = torch.tanh(self.fc4(a))
        return a * self.max_action
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount,
        tau,
        policy_noise,
        noise_clip,
        policy_freq
    ):
        self.lstm = nn.LSTM(input_size = N_FEATURES, 
                            hidden_size = HIDDEN_LAYERS,
                            num_layers = 1,
                            batch_first = True).to(device)
        self.lstm_target = copy.deepcopy(self.lstm)
        self.lstm_optimizer = torch.optim.Adam(self.lstm.parameters(), lr=LR_A)
        
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_A)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_C)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def pad_and_sort_batch(states_list):
        """
        As the number of vehicles is variant, we need padding to deal with it.
        For each state, we also want to feed to the LSTM first the ego state, and then
        by ascending distance to the intersection, the other vehicles. This requires sorting.

        states_list: [num_vehicules_i, n_features]. First row of every state is the ego state.

        Returns:
            padded_tensor: [batch, max_num_vehicles, n_features]
            seq_lengths: [batch] number of vehicules per state
        """

        seq_lengths = []
        sorted_states = []

        for state in states_list:
            # A state is a list of vehicule states
            ego_state = state[0:1] # 0:1 to preserve batch dimension
            others = state[1:]

            # Sort by distance to the intersection; first two vars are the position
            if len(others) > 0:
                distances = np.linalg.norm(others[:, :2], axis=1)
                sorted_idx = np.argsort(distances)
                sorted_others = others[sorted_idx]
            else:
                sorted_others = np.zeros((0, state.shape[1]))
            
            sorted_state = np.vstack([ego_state, sorted_others])
            sorted_states.append(sorted_state)
            seq_lengths.append(sorted_state.shape[0])
        
        # Obtain max length for the batch
        max_len = max(seq_lengths)
        batch_size = len(states_list)       # should be 128
        n_features = states_list[0].shape[1] # should be 14

        padded = np.zeros((batch_size, max_len, n_features), dtype=np.float32)
        for i, s in enumerate(sorted_states):
            padded[i, :s.shape[0], :] = s
        
        padded_tensor = torch.FloatTensor(padded).to(device)
        seq_lengths_tensor = torch.LongTensor(seq_lengths).to(device)
        return padded_tensor, seq_lengths_tensor
            

    def encode_state(self, padded_states, seq_lengths):
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
        padded_states, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed_input)
        lstm_embedding = h_n[-1]

        ego_states = padded_states[:, 0, :]
        return torch.cat([ego_states, lstm_embedding], dim=1)
    

    def select_action(self, raw_state):
        padded_state, seq_len = self.pad_and_sort_batch([raw_state])
        encoded_state = self.encode_state(padded_state, seq_len)
        return self.actor(encoded_state).cpu().data.numpy().flatten()
    

    def train(self, replay_buffer, batch_size=BATCH_SIZE):
        self.total_it += 1

        # Sample replay buffer
        raw_states, actions, raw_next_statws, rewards, not_done = replay_buffer.sample(batch_size)
        
        # Encode states using the LSTM
        padded_states, seq_lengths_state = self.pad_and_sort_batch(raw_states)
        state_enc = self.encode_state(padded_states, seq_lengths_state)

        padded_next_states, seq_lengths_next = self.pad_and_sort_batch(raw_next_states)
        next_state_enc = self.encode_state(padded_next_states, seq_lengths_next)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)


            next_action = (
                self.actor_target(next_state_enc) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state_enc, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state_enc, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state_enc, self.actor(state_enc)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.lstm.parameters(), self.lstm_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            