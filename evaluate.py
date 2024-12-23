import os
import torch as th

from stable_baselines3 import DQN


from snakeenv import SnakeEnv, SnakeEnvPartial
from stable_baselines3.common.evaluation import evaluate_policy


model = DQN.load('logs/best_model.zip', SnakeEnv(board_size=8, max_steps=1000))
#model.set_env(SnakeEnv(board_size=8, max_steps=1000))


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)

print(mean_reward, std_reward)


model = DQN.load('dqnevalcnnbig/best_model/42500000.zip', SnakeEnv(board_size=8, max_steps=1000))
#model.set_env(SnakeEnv(board_size=8, max_steps=1000))


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)

print(mean_reward, std_reward)
'dqnevalcnnbig/best_model/42500000.zip' # 87

model = DQN.load('dqnevalcnnbig/best_model/42500000.zip', SnakeEnv(board_size=8, max_steps=1000))
#model.set_env(SnakeEnv(board_size=8, max_steps=1000))


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)

print(mean_reward, std_reward)