from transfer_learning import *
from environment_generator import *
import time


start_time = time.time()

#### >>>>>> We only need to env_name and algorithm
env_name = 'Pendulum-v1'
algorithm = 'SAC'
#### <<<<<<<

moving_window = 50
log_dir_w_TL = "/tmp/gym/w_tl/"
log_dir_wo_TL = "/tmp/gym/wo_tl/"
extraInfo = "(" + env_name + '_' + algorithm + ")"

source_env = get_source_env(env_name)
target_env = get_target_env(env_name)

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
