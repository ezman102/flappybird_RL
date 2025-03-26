# replay_buffer.py
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, state_shape, device="cpu"):
        self.device = device
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        # Pre-allocate memory
        self.states = torch.empty((capacity, *state_shape), dtype=torch.float32, device=device)
        self.actions = torch.empty((capacity, 1), dtype=torch.long, device=device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.empty((capacity, *state_shape), dtype=torch.float32, device=device)
        self.dones = torch.empty((capacity, 1), dtype=torch.bool, device=device)

    def add(self, state, action, reward, next_state, done):
        idx = self.position % self.capacity
        
        self.states[idx] = torch.FloatTensor(state).to(self.device)
        self.actions[idx] = torch.tensor([action], device=self.device)
        self.rewards[idx] = torch.tensor([reward], device=self.device)
        self.next_states[idx] = torch.FloatTensor(next_state).to(self.device)
        self.dones[idx] = torch.tensor([done], device=self.device)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size