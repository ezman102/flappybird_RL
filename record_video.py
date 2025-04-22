import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from dqn_agent import DQNAgent
from dueling_dqn_agent import DuelingDQNAgent
from ppo_agent import PPOAgent 
import flappy_bird_gymnasium
import gymnasium
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Record Flappy Bird Agent Gameplay')
    parser.add_argument('--algo', type=str, required=True, choices=['dqn', 'dueling', 'ppo'],
                        help='RL algorithm used')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--num-episodes', type=int, default=5,
                        help='Number of episodes to record')
    parser.add_argument('--output-dir', type=str, default="recordings",
                        help='Directory to save recordings')
    return parser.parse_args()

def extract_timestamp_from_path(path: str) -> str:
    # Extract timestamp from path like .../DQN_lr0.0001_gam0.99_20250422_211052/...
    parts = Path(path).parts
    for part in parts:
        if "_" in part and part[-6:].isdigit():
            return part.split("_")[-1]
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_agent(algo, state_dim, action_dim, device, model_path):
    if algo == 'ppo':
        agent = PPOAgent(state_dim, action_dim, lr=0.0003, gamma=0.99)
    elif algo == 'dueling':
        agent = DuelingDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=0.0001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995
        )
    else:  # DQN
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=0.0001,
            gamma=0.99,
            hidden_layers=[128, 128],
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995
        )
    
    agent.load(model_path)
    agent.q_net.eval() if hasattr(agent, 'q_net') else agent.policy.eval()
    return agent

def record_videos():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract consistent timestamp from model path
    model_timestamp = extract_timestamp_from_path(args.model_path)

    # Environment and recording setup
    env = gym.make("FlappyBird-v0", use_lidar=False, render_mode="rgb_array")
    video_folder = Path(args.output_dir) / f"{args.algo}_{model_timestamp}"
    env = RecordVideo(env,
                      video_folder=video_folder,
                      episode_trigger=lambda x: True,
                      name_prefix=f"{args.algo}-episode")
    video_folder.mkdir(parents=True, exist_ok=True)

    # Load agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = load_agent(args.algo, state_dim, action_dim, device, args.model_path)

    # Logging
    rewards = []
    log_lines = []
    for ep in range(args.num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            if args.algo == 'ppo':
                action, _, _ = agent.act(state)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = agent.q_net(state_tensor)
                action = torch.argmax(q_values).item()

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = next_state

        rewards.append(total_reward)
        log_line = f"Episode {ep+1}: Score {total_reward:.2f}"
        print(log_line)
        log_lines.append(log_line)

    env.close()

    avg_score = np.mean(rewards)
    print(f"\nAverage score: {avg_score:.2f}")
    log_lines.append(f"\nAverage score: {avg_score:.2f}")

    # Save log
    log_path = video_folder / f"record_log.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    
    print(f"✅ Log saved to: {log_path}")
    print(f"🎥 Videos saved to: {video_folder}")

if __name__ == "__main__":
    record_videos()
