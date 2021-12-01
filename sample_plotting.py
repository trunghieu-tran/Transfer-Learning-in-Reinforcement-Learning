import gym

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from tl.utils.plot_utils import *
from tl.envs.new_pendulum_env import NewPendulumEnv



# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

policy_name = 'MlpPolicy'
step_number = 1000
env_name = 'Pendulum-v1'

source_env = gym.make(env_name)
target_env = NewPendulumEnv(source_env)

# Create and wrap the environment
# env = gym.make('LunarLanderContinuous-v2')
# Logs will be saved in log_dir/monitor.csv
env = Monitor(target_env, log_dir)

# Create the callback: check every 50 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=50, log_dir=log_dir)
# Create RL model
model = TD3('MlpPolicy', env, verbose=0)
# Train the agent
model.learn(total_timesteps=5000, callback=callback)

plot_results(log_dir, moving_window=10)
