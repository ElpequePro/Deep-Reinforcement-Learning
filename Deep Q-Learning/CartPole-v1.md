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
  - **optim**. Provides various optimization algorithms (like SGD, Adam, etc.) that can be used to update the parameters of neural networks during training.
- **matplotlib.pyplot**. It is used for creating static, animated, and interactive visualizations in Python.

## Create environment

```
env = gym.make('CartPole-v1')
```

## QNetwork

```
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

q_network = QNetwork(state_size, action_size)
```

Definimos la clase `class QNetwork(nn.Module)`.

### Constructor

Creamos el constructor `def __init__(self, state_size, action_size)` y añadimos las capas:
```
self.fc1 = nn.Linear(state_size, 64)
self.fc2 = nn.Linear(64, 64)
self.fc3 = nn.Linear(64, action_size)
```
El número que hay dentro de los paréntesis son las neuronas. En este ejercico con 64 o 32 está bien.

Seria como la siguiente imagen, pero en este caso tenemos solo tres capas:

- **fc1**. Capa del input, toma el `state_size` y lo mapea a 64 unidades. El `state_size` define:
  - **0**. Posición del Cart
  - **1**. Velocidad del Cart
  - **2**. Ángulo del Pole
  - **3**. Velocidad angular del Pole (velocidad con la que "cae" el Pole)
- **fc2**. Capa oculta, toma las 64 unidades anteriores y los mapea en otras 64 unidades.
- **fc3**. Capa del output, toma las 64 unidades anteriores y las mapea al tamaño del `action_size`. El `action_size` define:
  - **0**. Moverse a la izquierda
  - **1**. Moverse a la derecha

<p align='center'>
  <img src='https://chatgpt-farsi.com/wp-content/uploads/2023/08/5.png'>
</p>

[Más información...](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

### Forward