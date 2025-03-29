# replay_buffer.py
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, state_shape, device="cpu"):
        self.device = device
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        # Store transitions on CPU initially
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, reward, next_state, done):
        # Convert to tensors and store on CPU
        state_t = torch.FloatTensor(state).cpu()
        next_state_t = torch.FloatTensor(next_state).cpu()
        
        if self.size < self.capacity:
            self.states.append(state_t)
            self.actions.append(torch.LongTensor([action]))
            self.rewards.append(torch.FloatTensor([reward]))
            self.next_states.append(next_state_t)
            self.dones.append(torch.BoolTensor([done]))
            self.size += 1
        else:
            idx = self.position % self.capacity
            self.states[idx] = state_t
            self.actions[idx] = torch.LongTensor([action])
            self.rewards[idx] = torch.FloatTensor([reward])
            self.next_states[idx] = next_state_t
            self.dones[idx] = torch.BoolTensor([done])
        
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self), batch_size, replace=False)
        
        batch = (
            # States: [batch_size, state_dim]
            torch.stack([self.states[i] for i in indices]).to(self.device),
            
            # Actions: [batch_size, 1]
            torch.cat([self.actions[i] for i in indices]).unsqueeze(1).to(self.device),
            
            # Rewards: [batch_size, 1]
            torch.cat([self.rewards[i] for i in indices]).unsqueeze(1).to(self.device),
            
            # Next states: [batch_size, state_dim]
            torch.stack([self.next_states[i] for i in indices]).to(self.device),
            
            # Dones: [batch_size, 1]
            torch.cat([self.dones[i] for i in indices]).unsqueeze(1).float().to(self.device)
        )
        return batch

    def __len__(self):
        return self.size