import argparse
import gymnasium as gym
import torch
import numpy as np
from dqn_agent import DQNAgent
from dueling_dqn_agent import DuelingDQNAgent
from ppo_agent import PPOAgent
import flappy_bird_gymnasium
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Flappy Bird agent")
    parser.add_argument('--algo', type=str, required=True, choices=['dqn', 'dueling', 'ppo'], help='RL algorithm')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of test episodes')
    return parser.parse_args()

def load_agent(algo, model_path, state_dim, action_dim):
    if algo == 'ppo':
        agent = PPOAgent(state_dim, action_dim, lr=0.0003, gamma=0.99)
        agent.load(model_path)
        agent.policy.eval()
    elif algo == 'dueling':
        agent = DuelingDQNAgent(state_dim, action_dim, lr=0.0001, gamma=0.99,
                                epsilon_start=0, epsilon_end=0, epsilon_decay=1.0)
        agent.load(model_path)
        agent.q_net.eval()
    else:
        agent = DQNAgent(state_dim, action_dim, lr=0.0001, gamma=0.99, hidden_layers=[128, 128],
                         epsilon_start=0, epsilon_end=0, epsilon_decay=1.0)
        agent.load(model_path)
        agent.q_net.eval()
    return agent

def evaluate():
    args = parse_args()
    env = gym.make("FlappyBird-v0", use_lidar=False, render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = load_agent(args.algo, args.model_path, state_dim, action_dim)

    scores = []
    reached_1000 = -1
    start_time = datetime.now()

    for ep in tqdm(range(args.episodes), desc="Evaluating"):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            if args.algo == 'ppo':
                action, _, _ = agent.act(state)
            else:
                action = agent.act(state, epsilon=0.0)

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = next_state

        scores.append(total_reward)
        if reached_1000 == -1 and total_reward >= 1000:
            reached_1000 = ep + 1

    duration = (datetime.now() - start_time).total_seconds()
    scores_np = np.array(scores)

    # Build summary string
    summary = f"""
=== Evaluation Summary ===
Algorithm         : {args.algo.upper()}
Model Path        : {args.model_path}
Episodes Tested   : {args.episodes}
Average Score     : {np.mean(scores_np):.2f}
Max Score         : {np.max(scores_np):.0f}
Median Score      : {np.median(scores_np):.0f}
Score Std Dev     : {np.std(scores_np):.2f}
% Episodes > 100  : {(scores_np > 100).mean() * 100:.2f}%
% Episodes > 200  : {(scores_np > 200).mean() * 100:.2f}%
% Episodes > 300  : {(scores_np > 300).mean() * 100:.2f}%
Reached 300+      : {"Episode " + str(next((i+1 for i, s in enumerate(scores_np) if s >= 300), "Never"))}
Total Eval Time   : {duration:.2f} sec
"""


    print(summary)

    # Save to file
    output_dir = Path("eval")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"eval_{args.algo}_{Path(args.model_path).stem}.txt"
    with open(output_path, "w") as f:
        f.write(summary)

    print(f"✅ Summary saved to: {output_path}")

if __name__ == "__main__":
    evaluate()
