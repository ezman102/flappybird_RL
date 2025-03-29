import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPONetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.actor = nn.Linear(256, output_dim)
        self.critic = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.shared(x)
        return torch.distributions.Categorical(logits=self.actor(x)), self.critic(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, clip_epsilon=0.2, entropy_coeff=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            dist, value = self.policy(state)
        return dist.sample().item(), value.item(), dist.log_prob(dist.sample()).item()

    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        dist, values = self.policy(states)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
        loss = actor_loss + critic_loss - self.entropy_coeff * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))