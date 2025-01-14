# LLM-Powered Neural Fictitious Self-Play (NFSP) Agent

This project implements an AI agent demonstrating the Neural Fictitious Self-Play (NFSP) method using multiple large language models for imperfect-information games.

The development of this repository was inspired by the paper [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games](https://arxiv.org/pdf/1603.01121.pdf)

## Features

- Dual neural network architecture (RL and SL networks)
- Reservoir sampling for stable learning
- Anticipatory dynamics for opponent strategy prediction
- Support for Leduc poker and Limit Texas Hold'em
- Real-time visualization dashboard
- Educational LLM-powered explanation module
- Customizable parameters and game settings

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your environment variables in `.env`

## Project Structure

- `src/`: Core source code
  - `agents/`: NFSP agent implementation
  - `environments/`: Game environments
  - `models/`: Neural network architectures
  - `utils/`: Utility functions
- `dashboard/`: Visualization and monitoring
- `tests/`: Unit and integration tests
- `examples/`: Usage examples and tutorials

## Usage

1. Start the training dashboard:
```bash
python -m src.dashboard.app
```

2. Run training:
```bash
python -m src.train --game leduc --episodes 1000
```

## Configuration

Adjust parameters in `config.yaml`:
- Memory size
- Learning rates
- Anticipatory parameter
- Network architectures
- Game settings

## License

MIT License
