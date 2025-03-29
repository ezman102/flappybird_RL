import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.shared(x)
        value = self.value(x)  # [batch_size, 1]
        advantage = self.advantage(x)  # [batch_size, action_dim]
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, 
                 epsilon_start, epsilon_end, epsilon_decay):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.action_dim = action_dim

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return torch.argmax(q_values).item()

    def train(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors directly (already from replay buffer)
        states = states.to(self.device)
        actions = actions.to(self.device).long()
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Reshape tensors
        actions = actions.view(-1, 1)  # [batch_size, 1]
        rewards = rewards.view(-1, 1)  # [batch_size, 1]
        dones = dones.view(-1, 1)      # [batch_size, 1]

        # Current Q values
        q_values = self.q_net(states)
        current_q = q_values.gather(1, actions)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())