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
import utils

HIDDEN_LAYERS = 256
N_FEATURES = 14
LR_A = 1e-5
LR_C = 1e-4
BATCH_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, n_features, hidden_size, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action

        # Introduce an LSTM for encoding variable size inputs
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        encoded_dimension = n_features + hidden_size
        self.fc1 = nn.Linear(encoded_dimension, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_dim)

    
    def forward(self, padded_states, seq_lengths):
        """
        padded_states: [batch, max_len, n_features]
        seq_lengths: [batch] actual lengths
        """

        packed = nn.utils.rnn.pack_padded_sequence(
            padded_states, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (h_n, _) = self.lstm(packed)
        lstm_embedding = h_n[-1] 

        ego = padded_states[:, 0, :]

        enc = torch.cat([ego, lstm_embedding], dim=1)   
        a = F.relu(self.fc1(enc))
        a = F.relu(self.fc2(a)) 
        a = F.relu(self.fc3(a))
        a = self.fc4(a)
        return torch.tanh(a) * self.max_action


class Critic(nn.Module):
    def __init__(self, n_features, hidden_size, action_dim):
        super(Critic, self).__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        enc_dim = n_features + hidden_size

        # Q1
        self.l1 = nn.Linear(enc_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2
        self.l4 = nn.Linear(enc_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def encode(self, padded_states, seq_lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            padded_states, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        lstm_embedding = h_n[-1]

        ego = padded_states[:, 0, :]
        enc = torch.cat([ego, lstm_embedding], dim=1)
        return enc
    
    def forward(self, padded_states, seq_lengths, action):
        enc = self.encode(padded_states, seq_lengths)

        # Q1
        q1 = F.relu(self.l1(torch.cat([enc, action], dim=1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        # Q2
        q2 = F.relu(self.l4(torch.cat([enc, action], dim=1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2
    
    def Q1(self, padded_states, seq_lengths, action):
        enc = self.encode(padded_states, seq_lengths)
        q1 = F.relu(self.l1(torch.cat([enc, action], dim=1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
        self,
        n_features,
        hidden_size,
        action_dim,
        max_action,
        discount,
        tau,
        policy_noise,
        noise_clip,
        policy_freq,
    ):
        self.actor = Actor(n_features, hidden_size, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_A)

        self.critic = Critic(n_features, hidden_size, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_C)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, raw_states):
        padded_states, seq_lengths = utils.pad_and_sort_batch(raw_states)
        with torch.no_grad():
            actions = self.actor(padded_states, seq_lengths)
        return actions.cpu().numpy()
        
    def train(self, replay_buffer, batch_size=BATCH_SIZE):
        self.total_it += 1

        # Sample replay buffer
        raw_states, actions, raw_next_states, rewards, not_done = replay_buffer.sample(batch_size)

        # Encode states with padding/sorting
        padded_states, seq_lengths = utils.pad_and_sort_batch(raw_states)
        padded_next_states, seq_lengths_next = utils.pad_and_sort_batch(raw_next_states)

        with torch.no_grad():
            # Select target action with clipped noise
            noise = (
                torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(padded_next_states, seq_lengths_next) + noise
            ).clamp(-self.max_action, self.max_action)

            # Target Q
            target_Q1, target_Q2 = self.critic_target(
                padded_next_states, seq_lengths_next, next_action
            )
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + not_done * self.discount * target_Q

        # Current Q estimates
        current_Q1, current_Q2 = self.critic(padded_states, seq_lengths, actions)

        # Critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Actor loss (maximize Q1)
            actor_loss = -self.critic.Q1(
                padded_states, seq_lengths, self.actor(padded_states, seq_lengths)
            ).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update targets
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
		
