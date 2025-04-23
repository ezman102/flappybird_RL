# ppo_agent.py
# ---------------------------------------------------------------
# Implements the Proximal Policy Optimization (PPO) agent for Flappy Bird.
# Includes actor-critic network, clipped surrogate objective,
# entropy regularization, and training with multiple epochs per batch.
# Supports model saving and loading for evaluation.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# PPO neural network with shared layers for actor and critic
class PPONetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # Shared base network
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
         # Actor outputs action logits
        self.actor = nn.Linear(256, output_dim)
        # Critic outputs state value
        self.critic = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.shared(x)
        # Return action distribution and value estimate
        return torch.distributions.Categorical(logits=self.actor(x)), self.critic(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, clip_epsilon=0.1, entropy_coeff=0.01, ppo_epochs=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network with actor-critic structure
        self.policy = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # PPO hyperparameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.ppo_epochs = ppo_epochs

    def act(self, state):
        
        # Choose action and record log probability and value estimate
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            dist, value = self.policy(state)
            action = dist.sample()
            return action.item(), value.item(), dist.log_prob(action).item()

    def update(self, states, actions, old_log_probs, returns, advantages):

        # Convert inputs to tensors and move to device
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            dist, values = self.policy(states)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Compute ratio for policy update
            ratio = (new_log_probs - old_log_probs).exp()

            # Compute clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss: MSE between returns and predicted values
            critic_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()

            # Total loss with entropy regularization
            loss = actor_loss + critic_loss - self.entropy_coeff * entropy

            # Backpropagation and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
