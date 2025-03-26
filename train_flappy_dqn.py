import gymnasium as gym
import torch
import numpy as np
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
import yaml
from pathlib import Path
import flappy_bird_gymnasium  # Environment registration

def load_config():
    config_path = Path(__file__).parent / "config" / "hyperparams.yml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def train():
    config = load_config()
    env_config = config['environment']
    train_config = config['training']
    algo_config = config['algorithm']
    net_config = config['network']

    # Initialize environment
    env = gym.make("FlappyBird-v0",
                  use_lidar=env_config['use_lidar'],
                  render_mode=env_config['render_mode'])
    
    # Initialize agent and buffer
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=algo_config['lr'],
        gamma=algo_config['gamma'],
        hidden_layers=net_config['hidden_layers'],
        epsilon_start=algo_config['epsilon_start'],
        epsilon_end=algo_config['epsilon_end'],
        epsilon_decay=algo_config['epsilon_decay']
    )
    
    buffer = ReplayBuffer(
        capacity=train_config['buffer_size'],
        state_shape=env.observation_space.shape,
        device=agent.device
    )

    # Training metrics
    total_steps = 0
    best_reward = -np.inf
    reward_history = []

    # Training loop
    for episode in range(train_config['episodes']):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0

        while not done:
            # Epsilon decay
            epsilon = max(
                algo_config['epsilon_end'],
                algo_config['epsilon_start'] * (algo_config['epsilon_decay'] ** total_steps)
            )

            # Agent action
            action = agent.act(state, epsilon)

            # Environment step
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1

            # Store transition
            buffer.add(state, action, reward, next_state, done)
            state = next_state

            # Train condition
            if len(buffer) > train_config['train_start']:
                # Training frequency
                if total_steps % algo_config['train_freq'] == 0:
                    batch = buffer.sample(train_config['batch_size'])
                    loss = agent.train(batch)

                # Target network update
                if total_steps % algo_config['target_update_steps'] == 0:
                    agent.update_target_network()

                total_steps += 1

        # Track performance
        reward_history.append(episode_reward)
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save("best_model.pth")

        # Logging
        print(f"Episode {episode+1}/{train_config['episodes']} | "
              f"Reward: {episode_reward:.1f} | "
              f"Steps: {episode_steps} | "
              f"Total Steps: {total_steps} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Buffer: {len(buffer)}")

    # Save final model
    agent.save("final_model.pth")
    np.save("reward_history.npy", np.array(reward_history))

if __name__ == "__main__":
    train()