"""
Replay buffer implementation from the original paper.
https://github.com/sfujim/TD3/blob/master/utils.py
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, max_size=int(1e6)):
        self.max_size = max_size
        self.storage = []
        self.ptr = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        """
        state, action, next_state can be variable-length (lists, dicts, etc.)
        reward: float or list of floats (per-agent reward)
        done: bool
        """
        data = (state, action, next_state, reward, done)

        if len(self.storage) < self.max_size:
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        batch = np.random.sample(self.storage, batch_size)

        states, actions, next_states, rewards, dones = zip(*batch)

        # Return as raw Python objects (to be padded later before NN forward pass)
        return {
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=self.device),
            "dones": torch.tensor(dones, dtype=torch.float32, device=self.device)
        }


def pad_and_sort_batch(states_list):
    """
    states_list: list of np arrays [num_vehicles_i, n_features].
    First row is ego.
    Returns: padded_tensor [batch, max_len, n_features], seq_lengths [batch]
    """
    seq_lengths = []
    sorted_states = []

    for state in states_list:
        ego = state[0:1]
        others = state[1:]

        if len(others) > 0:
            dists = np.linalg.norm(others[:, :2], axis=1)
            idx = np.argsort(dists)
            sorted_others = others[idx]
        else:
            sorted_others = np.zeros((0, state.shape[1]))

        sorted_state = np.vstack([ego, sorted_others])
        sorted_states.append(sorted_state)
        seq_lengths.append(sorted_state.shape[0])

    max_len = max(seq_lengths)
    batch_size = len(states_list)
    n_features = states_list[0].shape[1]

    padded = np.zeros((batch_size, max_len, n_features), dtype=np.float32)
    for i, s in enumerate(sorted_states):
        padded[i, : s.shape[0], :] = s

    return torch.FloatTensor(padded).to(device), torch.LongTensor(seq_lengths).to(device)


def sort_state(state):
    """
    Sort by distances to the intersection point.
    """
    if len(state) == 0: return state
    distances = torch.norm(state[:, :2], dim=1)

    sorted_idx = torch.argsort(distances).to(device)
    rest_sorted = state[sorted_idx]

    return rest_sorted


def move_ego_to_front(state, ego_idx):
    """
    Move ego_state to the beginning of the tensor for the LSTM.
    """

    L = state.shape[0]

    ego = state[ego_idx:ego_idx+1]

    mask = torch.ones(L, dtype=torch.bool, device=device)
    mask[ego_idx] = False
    rest = state[mask]

    return torch.cat([ego, rest], dim=0)