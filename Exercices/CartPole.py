import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Entorno de entrenamiento (sin renderizado)
env = gym.make('CartPole-v1')

# Entorno de evaluación (con renderizado)
eval_env = gym.make('CartPole-v1', render_mode='human')

# El resto del código sigue igual...
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

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        index = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in index])
        
        # Convertir los estados y next_states a arrays de NumPy con la misma forma
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards, dtype=np.float32)),
            torch.FloatTensor(next_states),
            torch.FloatTensor(np.array(dones, dtype=np.float32))
        )

buffer_size = 10000
replay_buffer = ReplayBuffer(buffer_size)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995 

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.FloatTensor(state)
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def train(self, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

agent = DQNAgent(state_size, action_size)

num_episodes = 1000
batch_size = 32
update_target_every = 10

rewards_history = []

# Entrenamiento (sin renderizado)
for episode in range(num_episodes):
    state, _ = env.reset()  # Ahora reset() devuelve dos valores: estado e info
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)  # Desempaqueta los 5 valores
        done = terminated or truncated  # Combina terminated y truncated
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(replay_buffer.buffer) > batch_size:
            agent.train(batch_size)

    rewards_history.append(total_reward)
    agent.decay_epsilon()

    if episode % update_target_every == 0:
        agent.update_target_network()

    print(f"Episodio: {episode + 1}, Recompensa: {total_reward}, Épsilon: {agent.epsilon:.2f}")

plt.plot(rewards_history)
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.title("Recompensa por episodio")
plt.show()

torch.save(agent, 'Exercices/DQNAgent_CartPole-v1_1k')

# Evaluación (con renderizado)
num_eval_episodes = 10
for episode in range(num_eval_episodes):
    state, _ = eval_env.reset()  # Usar el entorno de evaluación con renderizado
    done = False
    total_reward = 0

    while not done:
        state = torch.FloatTensor(state)
        action = torch.argmax(agent.q_network(state)).item()
        next_state, reward, terminated, truncated, _ = eval_env.step(action)  # Desempaqueta los 5 valores
        done = terminated or truncated  # Combina terminated y truncated
        state = next_state
        total_reward += reward

    print(f"Episodio de evaluación {episode + 1}, Recompensa: {total_reward}")

# Cerrar el entorno de evaluación
eval_env.close()