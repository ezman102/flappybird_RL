# train_flappy.py
import argparse
import gymnasium as gym
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml
import shutil
from tqdm import tqdm
from dqn_agent import DQNAgent
from dueling_dqn_agent import DuelingDQNAgent
from ppo_agent import PPOAgent
from replay_buffer import ReplayBuffer
import flappy_bird_gymnasium
import sys
import os

# ---------------------------
# Logging Setup (Auto Timestamped)
# ---------------------------
def pre_parse_algo():
    for i, arg in enumerate(sys.argv):
        if arg == "--algo" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "default"

algo_for_log = pre_parse_algo()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"{algo_for_log}_{timestamp}.log"
sys.stdout = open(log_file, "w")
sys.stderr = sys.stdout


# ---------------------------
# Arg Parsing and Config
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Train Flappy Bird RL Agent')
    parser.add_argument('--algo', type=str, default='dqn', choices=['dqn', 'dueling', 'ppo'],
                        help='RL algorithm to use')
    parser.add_argument('--config', type=str, default="config/hyperparams.yml",
                        help='Path to configuration file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

# ---------------------------
# Main Training Function
# ---------------------------
def train():
    args = parse_args()
    config = load_config(args.config)

    # Configs
    env_config = config['environment']
    train_config = config['training']
    algo_config = config[args.algo]
    common_config = config['common']

    # Create result folders
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.algo.upper()}_lr{algo_config['lr']}_gam{common_config['gamma']}_{timestamp}"
    results_dir = Path("results") / experiment_name
    (results_dir / "models").mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, results_dir / "configs" / "training_config.yml")

    # Env
    env = gym.make("FlappyBird-v0", use_lidar=env_config['use_lidar'], render_mode=env_config['render_mode'])

    # Agent
    if args.algo == 'ppo':
        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            lr=algo_config['lr'],
            gamma=common_config['gamma'],
            clip_epsilon=algo_config['clip_epsilon'],
            entropy_coeff=algo_config['entropy_coeff']
        )
        buffer = []
    else:
        agent_cls = DuelingDQNAgent if args.algo == 'dueling' else DQNAgent
        agent = agent_cls(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            lr=algo_config['lr'],
            gamma=common_config['gamma'],
            epsilon_start=algo_config['epsilon_start'],
            epsilon_end=algo_config['epsilon_end'],
            epsilon_decay=algo_config['epsilon_decay'],
            hidden_layers=algo_config.get('hidden_layers', [128, 128])  # for DQN only
        )
        buffer = ReplayBuffer(
            capacity=train_config['buffer_size'],
            state_shape=env.observation_space.shape,
            device=agent.device
        )

    # Train loop
    total_steps = 0
    best_reward = -np.inf
    reward_history = []
    progress = tqdm(total=train_config['max_episodes'], desc="Training")

    for episode in range(train_config['max_episodes']):
        state, done = env.reset()[0], False
        episode_reward = 0
        episode_data = []

        while not done:
            if args.algo == 'ppo':
                action, value, log_prob = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)
                episode_data.append((state, action, log_prob, value, reward, done))
            else:
                epsilon = max(algo_config['epsilon_end'],
                              algo_config['epsilon_start'] * (algo_config['epsilon_decay'] ** total_steps))
                action = agent.act(state, epsilon)
                next_state, reward, done, _, _ = env.step(action)
                buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            # Train
            if args.algo != 'ppo' and len(buffer) > train_config['train_start']:
                if total_steps % algo_config['train_freq'] == 0:
                    batch = buffer.sample(train_config['batch_size'])
                    agent.train(batch)

                if total_steps % algo_config['target_update_steps'] == 0:
                    agent.update_target_network()

                total_steps += 1

        # PPO update
        if args.algo == 'ppo':
            states, actions, old_log_probs, values, rewards, dones = zip(*episode_data)
            returns, advantages = [], []
            R = 0
            for r in reversed(rewards):
                R = r + common_config['gamma'] * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            advantages = returns - torch.FloatTensor(values)
            agent.update(states, actions, old_log_probs, returns, advantages)

        # Save rewards + metrics
        reward_history.append(episode_reward)
        progress.update(1)
        progress.set_postfix({
            'avg': f"{np.mean(reward_history[-100:]):.2f}",
            'best': f"{best_reward:.0f}",
            'ep': episode + 1
        })

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(results_dir / "models" / f"best_model_ep{episode+1}.pth")

        # Early stop if target score is reached
        if train_config.get('target_reward') and len(reward_history) >= train_config['reward_window']:
            moving_avg = np.mean(reward_history[-train_config['reward_window']:])
            if moving_avg >= train_config['target_reward']:
                print(f"\n🎯 Target avg reward {train_config['target_reward']} reached at episode {episode + 1}.")
                agent.save(results_dir / "models" / f"target_reached_model_ep{episode+1}.pth")
                break


    # Save final artifacts
    progress.close()
    agent.save(results_dir / "models" / "final_model.pth")
    np.save(results_dir / "metrics" / "reward_history.npy", reward_history)
    print(f"\n✅ Training completed. Results saved to: {results_dir}")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    train()
