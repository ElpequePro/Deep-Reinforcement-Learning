import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('CartPole-v1', render_mode='human')
env = DummyVecEnv([lambda: env])

model = PPO.load('PPO_Model_CartPole-v1_250k', env)

evaluate_policy(model, env, n_eval_episodes=10, render=True)

env.close()