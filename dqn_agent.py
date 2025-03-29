# dpn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, hidden_layers,
                 epsilon_start, epsilon_end, epsilon_decay):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Network setup
        self.q_net = self._build_network(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net = self._build_network(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Training setup
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def _build_network(self, state_dim, action_dim, hidden_layers):
        layers = []
        prev_size = state_dim
        for size in hidden_layers:
            layers += [nn.Linear(prev_size, size), nn.ReLU()]
            prev_size = size
        layers.append(nn.Linear(prev_size, action_dim))
        return nn.Sequential(*layers)

    def act(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            return torch.argmax(q_values).item()

    def train(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors (already properly dimensioned)
        states = states.to(self.device)
        actions = actions.to(self.device).long()
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values: [batch_size, 1]
        current_q = self.q_net(states).gather(1, actions)
        
        # Target Q values: [batch_size, 1]
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss between [batch_size, 1] and [batch_size, 1]
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())