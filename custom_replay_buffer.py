"""
Replay buffer that can handle variable state sizes.
"""

import numpy as np
import torch
from collections import deque
import random
import utils

class ReplayBuffer():
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
