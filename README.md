# 🐦 Flappy Bird Reinforcement Learning Agents
Train and evaluate Deep Reinforcement Learning agents to play Flappy Bird using **DQN**, **Dueling DQN**, and **PPO**. Includes training, video recording, and performance visualization utilities.
---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install flappy-bird-gymnasium
```

### 2. Train an Agent

```bash
python train_flappy_dqn.py --algo dqn
python train_flappy_dqn.py --algo dueling
python train_flappy_dqn.py --algo ppo
```

### 3. Record Agent Gameplay

```bash
python record_video.py --algo dqn --model-path path/to/model.pth --num-episodes 5
```

Videos will be saved under `recordings/`.

### 4. Visualize Performance
```bash
jupyter notebook visualization.ipynb
```

---

## 🧠 Algorithms Supported
| Algorithm      | Description                                   |
|----------------|-----------------------------------------------|
| **DQN**        | Classic Deep Q-Network with experience replay |
| **Dueling DQN**| Separates value & advantage streams           |
| **PPO**        | Policy-based method with clipped objective    |

---

## ⚙️ Configuration
All training-related settings are located in `config/hyperparams.yml`. You can adjust:
- Learning rate, gamma
- Epsilon decay (for DQNs)
- Network architecture
- PPO-specific params like `clip_epsilon`, `entropy_coeff`

---

## 📊 Outputs
After training, the following are saved to `results/<experiment_name>/`:
- `models/`: Best and final models
- `metrics/`: Numpy array of episode rewards
- `configs/`: A copy of the used config

---
## 📌 Requirements

- Python ≥ 3.8
- `torch`, `numpy`, `matplotlib`, `tqdm`, `gymnasium`, `flappy-bird-gymnasium`, `PyYAML`
```
