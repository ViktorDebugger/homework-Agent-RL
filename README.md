# Flappy Bird + DQN Agent

A Flappy Bird clone with a Deep Q-Network (DQN) reinforcement learning agent. Play as a human or train and watch an AI agent play.

## Requirements

- Python 3.10+
- Pygame
- PyTorch
- Gymnasium
- NumPy
- Matplotlib

Install with:

```bash
pip install pygame torch gymnasium numpy matplotlib
```

## Project Structure

```
homework/
├── game/                 # Flappy Bird game and Gymnasium env
│   ├── game.py           # Human-playable game loop
│   ├── env.py            # FlappyBirdEnv (Gymnasium API)
│   ├── bird.py, pipe.py, ground.py, score.py, constants.py
│   └── __main__.py
├── rl-agent/
│   ├── dqn-agent.py      # DQN training and --play script
│   ├── plots/            # Learning curves (YYYY-MM-DD_HH-MM-SS.png)
│   └── models/           # Saved agents (YYYY-MM-DD_HH-MM-SS.pt)
├── COMMANDS.txt          # Short command reference
└── README.md
```

## Quick Start

Run all commands from the project root (`homework/`).

| Action | Command |
|--------|--------|
| **Play as human** | `python -m game` |
| **Watch untrained agent** | `python rl-agent/dqn-agent.py --play` |
| **Watch trained agent** | `python rl-agent/dqn-agent.py --play --model rl-agent/models/YYYY-MM-DD_HH-MM-SS.pt` |
| **Train the agent** | `python rl-agent/dqn-agent.py` |

Human controls: **SPACE** or **click** = flap, **ESC** = quit.

See `COMMANDS.txt` for more detail.

## Training

- Default: 1000 episodes. Progress is shown as `Episode X/Y (Z%) | Epsilon: ...`
- After training: a learning-curve plot is saved to `rl-agent/plots/` and the model to `rl-agent/models/` (filename = date and time).
- The agent uses DQN with experience replay, Double DQN, soft target updates, and reward shaping (e.g. penalties for flapping when high or when already moving up) to improve stability and behavior.

## Outputs

- **Plots:** `rl-agent/plots/YYYY-MM-DD_HH-MM-SS.png` — raw and moving-average reward per episode.
- **Models:** `rl-agent/models/YYYY-MM-DD_HH-MM-SS.pt` — PyTorch `state_dict`; use with `--play --model <path>`.
