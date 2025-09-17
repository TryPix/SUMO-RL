"""
Modified original implmentation of TD3.

Paper: https://arxiv.org/abs/1802.09477
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_size, lstm_hidden_size, action_dimension):
        super(Actor, self).__init__()

        self.lstm = nn.LSTM(input_size=state_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=1,
                            batch_first=False)
        

        self.fc1 = nn.Linear(state_size + lstm_hidden_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_dimension)

    
    def forward(self, state):

        ego_state = torch.stack([seq[0] for seq in state], dim=0)

        # As the number of vehicles is variable, we need to pack the input
        packed_state = nn.utils.rnn.pack_sequence(state, enforce_sorted=False)
        
        _, (h_n, _) = self.lstm(packed_state)
        lstm_embedding = h_n[-1]

        actor_input = torch.cat([lstm_embedding, ego_state], dim=1)

        a = F.relu(self.fc1(actor_input))
        a = F.relu(self.fc2(a)) 
        a = F.relu(self.fc3(a))
        a = self.fc4(a)
        return torch.tanh(a)


class Critic(nn.Module):
    def __init__(self, state_size, lstm_hidden_size, action_dimension):
        super(Critic, self).__init__()
    
        self.lstm = nn.LSTM(input_size=state_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=1,
                            batch_first=False)

        # Q1 
        self.fc1 = nn.Linear(state_size + lstm_hidden_size + action_dimension, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

        # Q2
        self.fc5 = nn.Linear(state_size + lstm_hidden_size + action_dimension, 1024)
        self.fc6 = nn.Linear(1024, 512)
        self.fc7 = nn.Linear(512, 256)
        self.fc8 = nn.Linear(256, 1)

    
    def forward(self, state, action):

        ego_state = torch.stack([seq[0] for seq in state], dim=0)

        # As the number of vehicles is variable, we need to pack the input
        packed_state = nn.utils.rnn.pack_sequence(state, enforce_sorted=False)
        
        _, (h_n, _) = self.lstm(packed_state)
        lstm_embedding = h_n[-1]

        critic_input = torch.cat([lstm_embedding, ego_state], dim=1)
        critic_action_input = torch.cat([critic_input, action], dim=1)
        
        # Q1
        q1 = F.relu(self.fc1(critic_action_input))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.fc3(q1))
        q1 = self.fc4(q1)

        # Q2
        q2 = F.relu(self.fc5(critic_action_input))
        q2 = F.relu(self.fc6(q2))
        q2 = F.relu(self.fc7(q2))
        q2 = self.fc8(q2)

        return q1, q2


class TD3(object):

    def __init__(self,
                 state_size, 
                 lstm_hidden_size,
                 action_dimension,
                 learning_rate_actor,
                 learning_rate_critic):
        self.actor = Actor(state_size, lstm_hidden_size, action_dimension).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate_actor)

        self.critic = Critic(state_size, lstm_hidden_size, action_dimension).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate_critic)
    
    def select_action(self, state):
        return self.actor(state).detach().cpu().numpy()


    def train(self):
        ...