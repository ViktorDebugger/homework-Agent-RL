import os
import random
import sys
from datetime import date
from pathlib import Path

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
from typing import List, Tuple
import gymnasium as gym

from game.env import FlappyBirdEnv

GAMMA = 0.99
EPSILON_START = 1.0
DECAY = 0.995
MIN_EPSILON = 0.01
BATCH_SIZE = 128
LR = 1e-4
TARGET_UPDATE = 100
MAX_STEPS = 5000
MOVING_AVG_WINDOW = 50
PLOTS_DIR = "plots"


def save_learning_plot(
    episode_rewards: List[float],
    env_name: str = "FlappyBird",
    window: int = MOVING_AVG_WINDOW,
    base_dir: str | Path | None = None,
) -> Path:
    if base_dir is None:
        base_dir = Path(_script_dir)
    date_str = date.today().isoformat()
    save_dir = Path(base_dir) / PLOTS_DIR / date_str
    save_dir.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(len(episode_rewards))
    raw = np.array(episode_rewards, dtype=np.float64)
    ma = np.convolve(raw, np.ones(window) / window, mode="valid")
    ma_episodes = np.arange(window - 1, len(episode_rewards))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes, raw, color="lightcoral", alpha=0.6, linewidth=0.8, label="Raw Reward")
    ax.plot(ma_episodes, ma, color="darkred", linewidth=2, label=f"Moving Avg ({window} eps)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(f"Agent Learning Progress ({env_name})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = save_dir / "learning_progress.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# Q-Network class (MLP)
class DQN(nn.Module):
    def __init__(self, input_size: int, hidden_size: Tuple[int, int], output_size: int) -> None:
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ReplayBuffer class
class ReplayBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor]:
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.from_numpy(np.array(states, dtype=np.float32)),
            torch.from_numpy(np.array(actions, dtype=np.int64)),
            torch.from_numpy(np.array(rewards, dtype=np.float32)),
            torch.from_numpy(np.array(next_states, dtype=np.float32)),
            torch.from_numpy(np.array(dones, dtype=np.bool_))
        )
    
    def __len__(self) -> int:
        return len(self.buffer)

# DQN Agent class
class DQNAgent(object):
    def __init__(
        self,
        env: gym.Env,
        policy_net: nn.Module,
        target_net: nn.Module,
        optimizer: optim.Optimizer,
        replay_buffer: ReplayBuffer,
        epsilon_start: float,
        decay: float,
        min_epsilon: float,
        batch_size: int
    ) -> None:
        self.env = env
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.epsilon_start = epsilon_start
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.eps = epsilon_start

    def select_action(self, state: np.ndarray) -> int:
        sample = random.random()
        eps_threshold = max(self.eps, self.min_epsilon)
        if sample > eps_threshold:
            with torch.no_grad():
                state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)
                return int(self.policy_net(state_t).max(1)[1].item())
        return random.randrange(self.env.action_space.n)
    
    def update(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        # Compute the Q-values for the selected actions
        q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze()
        
        # Compute the target Q-values using the target network and max over next state
        with torch.no_grad():
            max_next_q = self.target_net(next_state).max(1)[0]
            done_f = done.float()
            target_q_values = reward + GAMMA * (1.0 - done_f) * max_next_q
        
        # Calculate the loss
        loss = nn.functional.mse_loss(q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def train(self, num_episodes: int, plot: bool = True, env_name: str = "FlappyBird") -> List[float]:
        episode_rewards: List[float] = []
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            done = False
            steps_done = 0
            episode_reward = 0.0
            while not done and steps_done < MAX_STEPS:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                self.replay_buffer.push(state, action, reward, next_state, terminated or truncated)
                state = next_state
                done = terminated or truncated
                self.update()
                self.eps = max(self.min_epsilon, self.eps * self.decay)
                steps_done += 1
            episode_rewards.append(episode_reward)
            if episode > 0 and episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        if plot and episode_rewards:
            path = save_learning_plot(episode_rewards, env_name=env_name)
            print(f"Plot saved: {path}")
        return episode_rewards

if __name__ == "__main__":
    env = FlappyBirdEnv(render_mode=None)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy_net = DQN(obs_size, (64, 64), n_actions)
    target_net = DQN(obs_size, (64, 64), n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(capacity=int(1e5))
    agent = DQNAgent(
        env,
        policy_net,
        target_net,
        optimizer,
        replay_buffer,
        epsilon_start=EPSILON_START,
        decay=DECAY,
        min_epsilon=MIN_EPSILON,
        batch_size=BATCH_SIZE,
    )
    agent.train(num_episodes=1000, plot=True, env_name="FlappyBird")
