import os
import torch as th

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor


from snakeenv import SnakeEnv, SnakeEnvPartial
from utils import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.callbacks import EvalCallback

from utils import CustomCNN


# first one
log_dir = "pposimple"
os.makedirs(log_dir, exist_ok=True)

board_size=8
n_envs=50
env = SnakeEnv(board_size=board_size)

#vec_env = make_vec_env(SnakeEnv, n_envs=n_envs, env_kwargs=dict(board_size=8))
# Logs will be saved in log_dir/monitor.csv
env = Monitor(env, log_dir)
#env = VecMonitor(vec_env, log_dir)

callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)

eval_env = SnakeEnv(board_size=board_size)
eval_env = Monitor(eval_env, log_dir)

#eval_vec_env = make_vec_env(SnakeEnv, n_envs=50, env_kwargs=dict(board_size=8))
#eval_env = VecMonitor(eval_vec_env, log_dir)
# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=10000,
                             deterministic=True, render=False,n_eval_episodes=10)


GAMMA = .9
#policy_kwargs = dict(net_arch=dict(pi=[128,128,128,128,128,128,128,128], vf=[128,128,128,128,128,128,128,128]),features_extractor_class=CustomCNN,features_extractor_kwargs=dict(input_dim=board_size,features_dim=128))

#policy_dqn = dict(net_arch=[256,256,256,256,128,128,128,128])

#policy_dqn = dict(net_arch=[128,128,128,128,128,128,128,128],features_extractor_class=CustomCNN,features_extractor_kwargs=dict(input_dim=board_size,features_dim=256))

#model = PPO("MlpPolicy", env, verbose=1, gamma=GAMMA, policy_kwargs=policy_kwargs, batch_size=256, tensorboard_log="./tensorboard/", device='cuda')
model = PPO("MlpPolicy", env, verbose=1, gamma=GAMMA, tensorboard_log="./tensorboard/", device='cuda')
#model = DQN("MlpPolicy", env, verbose=1, gamma=GAMMA, policy_kwargs=policy_dqn, batch_size=256, tensorboard_log="./tensorboard/", device='cuda') #batch_size=128
#model = DQN("MlpPolicy", env, verbose=1, gamma=GAMMA)

#model = A2C("MlpPolicy", env, verbose=1,gamma=GAMMA, tensorboard_log="./tensorboard/", device='cuda')

print(log_dir)
print(model.policy)


model.learn(total_timesteps=1000000, callback=[callback, eval_callback], reset_num_timesteps=False, tb_log_name=log_dir)


'''
# second
log_dir = "dqnLastPartial"
os.makedirs(log_dir, exist_ok=True)

board_size=8
n_envs=50
#env = SnakeEnv(board_size=board_size)

vec_env = make_vec_env(SnakeEnvPartial, n_envs=n_envs, env_kwargs=dict(board_size=8, mask_size=2))
# Logs will be saved in log_dir/monitor.csv
#env = Monitor(env, log_dir)
env = VecMonitor(vec_env, log_dir)

callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)

#eval_env = SnakeEnv(board_size=board_size)
#eval_env = Monitor(eval_env, log_dir)

eval_vec_env = make_vec_env(SnakeEnvPartial, n_envs=n_envs, env_kwargs=dict(board_size=8, mask_size=2))
eval_env = VecMonitor(eval_vec_env, log_dir)
# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=10000,
                             deterministic=True, render=False,n_eval_episodes=10)


GAMMA = .9
policy_kwargs = dict(net_arch=dict(pi=[128,128,128,128,128,128,128,128], vf=[128,128,128,128,128,128,128,128]))

#policy_dqn = dict(net_arch=[256,256,256,256,128,128,128,128])

policy_dqn = dict(net_arch=[256,256,128,128,128,128],features_extractor_class=CustomCNN,features_extractor_kwargs=dict(input_dim=5,features_dim=256))

#model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, gamma=GAMMA, tensorboard_log="./tensorboard/")
#model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0005, gamma=GAMMA)
model = DQN("MlpPolicy", env, verbose=1, gamma=GAMMA, policy_kwargs=policy_dqn, batch_size=256, tensorboard_log="./tensorboard/", device='cuda') #batch_size=128
#model = DQN("MlpPolicy", env, verbose=1, gamma=GAMMA)

#model = A2C("MlpPolicy", env, verbose=1,gamma=GAMMA, tensorboard_log="./tensorboard/", device='cuda')

print(log_dir)
print(model.policy)


model.learn(total_timesteps=1000000*n_envs, callback=[callback, eval_callback], reset_num_timesteps=False, tb_log_name=log_dir)


# third
log_dir = "PPOLastPartial"
os.makedirs(log_dir, exist_ok=True)

board_size=8
n_envs=50
#env = SnakeEnv(board_size=board_size)

vec_env = make_vec_env(SnakeEnvPartial, n_envs=n_envs, env_kwargs=dict(board_size=8, mask_size=2))
# Logs will be saved in log_dir/monitor.csv
#env = Monitor(env, log_dir)
env = VecMonitor(vec_env, log_dir)

callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)

#eval_env = SnakeEnv(board_size=board_size)
#eval_env = Monitor(eval_env, log_dir)

eval_vec_env = make_vec_env(SnakeEnvPartial, n_envs=n_envs, env_kwargs=dict(board_size=8, mask_size=2))
eval_env = VecMonitor(eval_vec_env, log_dir)
# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=10000,
                             deterministic=True, render=False,n_eval_episodes=10)


GAMMA = .9
policy_kwargs = dict(net_arch=dict(pi=[128,128,128,128,128,128,128,128], vf=[128,128,128,128,128,128,128,128]),features_extractor_class=CustomCNN,features_extractor_kwargs=dict(input_dim=5,features_dim=256))

#policy_dqn = dict(net_arch=[256,256,256,256,128,128,128,128])

#policy_dqn = dict(net_arch=[256,256,128,128,128,128],features_extractor_class=CustomCNN,features_extractor_kwargs=dict(input_dim=board_size,features_dim=256))

model = PPO("MlpPolicy", env, verbose=1, gamma=GAMMA, policy_kwargs=policy_kwargs, batch_size=256, tensorboard_log="./tensorboard/", device='cuda')
#model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0005, gamma=GAMMA)
#model = DQN("MlpPolicy", env, verbose=1, gamma=GAMMA, policy_kwargs=policy_dqn, batch_size=256, tensorboard_log="./tensorboard/", device='cuda') #batch_size=128
#model = DQN("MlpPolicy", env, verbose=1, gamma=GAMMA)

#model = A2C("MlpPolicy", env, verbose=1,gamma=GAMMA, tensorboard_log="./tensorboard/", device='cuda')

print(log_dir)
print(model.policy)


model.learn(total_timesteps=1000000*n_envs, callback=[callback, eval_callback], reset_num_timesteps=False, tb_log_name=log_dir)
'''