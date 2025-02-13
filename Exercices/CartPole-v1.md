# CartPole-v1

<p align="center">
  <img src="img/default.png" />
</p>

## Installation

```
pip install gymnasium numpy torch matplotlib
```

## Import

```
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
```

- **gymnasium**. Toolkit for developing and comparing reinforcement learning algorithms.
- **numpy**. It provides support for arrays, matrices, and a wide range of mathematical functions to operate on these data structures.
- **torch**. Open-source machine learning library.
  - **nn**. Contains classes and functions to build neural networks in PyTorch.
  - **optim**. provides various optimization algorithms (like SGD, Adam, etc.) that can be used to update the parameters of neural networks during training.
- **matplotlib.pyplot**. It is used for creating static, animated, and interactive visualizations in Python.

## Create environment

```
env = gym.make('CartPole-v1')
```
