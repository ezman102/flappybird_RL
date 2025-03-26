import numpy as np
from train_flappy_dqn import train  # Reuse training logic
import flappy_bird_gymnasium
import torch
from dqn_agent import DQNAgent

def evaluate(model_path, n_episodes=100):
    env = flappy_bird_gymnasium.make("FlappyBird-v0")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    agent.q_net.load_state_dict(torch.load(model_path))
    
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.act(obs, epsilon=0.01)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    
    print(f"Average Reward: {np.mean(rewards):.2f}")