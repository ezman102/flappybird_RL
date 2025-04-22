# dueling_dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[128, 128]):
        super().__init__()
        layers = []
        prev_size = input_dim
        for size in hidden_layers:
            layers += [nn.Linear(prev_size, size), nn.ReLU()]
            prev_size = size

        self.shared = nn.Sequential(*layers)
        self.value = nn.Linear(prev_size, 1)
        self.advantage = nn.Linear(prev_size, output_dim)

    def forward(self, x):
        x = self.shared(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, 
                 epsilon_start, epsilon_end, epsilon_decay,
                 hidden_layers=[128, 128]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DuelingQNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim, hidden_layers).to(self.device)
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

        states = states.to(self.device)
        actions = actions.to(self.device).long()
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        actions = actions.view(-1, 1)
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)

        current_q = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
