import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "CarRacing-v3"

training_env = gym.make(env_name)
training_env = DummyVecEnv([lambda: training_env])

model = PPO('MlpPolicy', training_env, verbose=1)
model.learn(total_timesteps=250000)

save_name = 'PPO_Model_CarRacing-v3_250k' 
model.save(save_name)

training_env.close()

eval_env = gym.make(env_name, render_mode='human')
eval_env = DummyVecEnv([lambda: eval_env])

model = PPO.load(save_name, eval_env)

evaluate_policy(model, eval_env, n_eval_episodes=10, render=True)

eval_env.close()