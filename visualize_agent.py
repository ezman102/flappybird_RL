import flappy_bird_gymnasium
from dqn_agent import DQNAgent
import check_cuda
import gymnasium as gym

def visualize(model_path="flappy_dqn.pth"):
    env = gym.make("FlappyBird-v0", render_mode="human")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    agent.q_net.load_state_dict(check_cuda.load(model_path))
    
    obs, _ = env.reset()
    while True:
        action = agent.act(obs, epsilon=0.01)
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()

if __name__ == "__main__":
    visualize()