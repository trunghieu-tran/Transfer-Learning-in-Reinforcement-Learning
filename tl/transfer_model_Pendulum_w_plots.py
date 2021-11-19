import gym
from tl.new_pendulum_env import NewPendulumEnv
from transfer_learning import *
import time

start_time = time.time()

env_name = 'Pendulum-v1'
algorithm = 'PPO'
moving_window = 50
log_dir_w_TL = "/tmp/gym/w_tl/"
log_dir_wo_TL = "/tmp/gym/wo_tl/"
extraInfo = "(" + env_name + '_' + algorithm + ")"

source_env = gym.make(env_name)
target_env = NewPendulumEnv(gym.make(env_name))

transfer_execute(source_env=source_env,
                 target_env=target_env,
                 algo=algorithm,
                 policy_name='MlpPolicy',
                 step_number=10000,
                 step_number_small=10000,
                 callback_check_freq=500,
                 moving_window=moving_window,
                 log_dir_w_TL=log_dir_w_TL,
                 log_dir_wo_TL=log_dir_wo_TL,
                 )

plot_results(log_dir_wo_TL, title=extraInfo+"Without-TL Learning Curve", moving_window=moving_window)
plot_results(log_dir_w_TL, title=extraInfo+"With-TL Learning Curve", moving_window=moving_window)
######
print("--- %s seconds ---" % (time.time() - start_time))
