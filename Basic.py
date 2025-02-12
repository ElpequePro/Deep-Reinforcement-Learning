import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

train_env = gym.make("LunarLander-v3")
train_env = DummyVecEnv([lambda: train_env])

model = PPO('MlpPolicy', train_env, verbose=1)
model.learn(total_timesteps=10000)

eval_env = gym.make("LunarLander-v3", render_mode='human')
eval_env = DummyVecEnv([lambda: eval_env])

evaluate_policy(model, eval_env, n_eval_episodes=10, render=True)

model.save('PPO_Model_LunarLander')

train_env.close()
eval_env.close()

del model
model = PPO.load('PPO_Model_LunarLander', eval_env)