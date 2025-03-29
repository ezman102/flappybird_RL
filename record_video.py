# record_video.py
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

def parse_args():
    parser = argparse.ArgumentParser(description='Record Flappy Bird Agent Gameplay')
    parser.add_argument('--algo', type=str, required=True,
                      choices=['dqn', 'dueling', 'ppo'],
                      help='RL algorithm used')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--num-episodes', type=int, default=5,
                      help='Number of episodes to record')
    parser.add_argument('--output-dir', type=str, default="recordings",
                      help='Directory to save recordings')
    return parser.parse_args()

def load_agent(algo, state_dim, action_dim, device):
    args = parse_args()
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
    
    agent.load(args.model_path)
    agent.q_net.eval() if hasattr(agent, 'q_net') else agent.policy.eval()
    return agent

def record_videos():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment with video recording
    env = gym.make("FlappyBird-v0", 
                  use_lidar=False,
                  render_mode="rgb_array")
    
    # Wrap environment for recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_folder = Path(args.output_dir) / f"{args.algo}_{timestamp}"
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix=f"{args.algo}-episode"
    )
    
    # Initialize agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = load_agent(args.algo, state_dim, action_dim, device)

    # Recording loop
    rewards = []
    for ep in range(args.num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            if args.algo == 'ppo':
                action, _, _ = agent.act(state)
            else:  # DQN variants
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = agent.q_net(state_tensor)
                action = torch.argmax(q_values).item()
            
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = next_state
        
        rewards.append(total_reward)
        print(f"Episode {ep+1}: Score {total_reward}")

    env.close()
    print(f"\nAverage score: {np.mean(rewards):.1f}")
    print(f"Videos saved to: {video_folder}")

if __name__ == "__main__":
    record_videos()