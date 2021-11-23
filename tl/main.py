from tl.utils.transfer_learning import *
from tl.utils.environment_generator import *
import time


start_time = time.time()

#### >>>>>> We only need to env_name and algorithm
# env_name = 'Pendulum-v1'
env_name = 'CarRacing-v0'
algorithm = 'PPO'
#### <<<<<<<

moving_window = 5
log_dir_w_TL = "/tmp/gym/w_tl/"
log_dir_wo_TL = "/tmp/gym/wo_tl/"
extraInfo = "(" + env_name + '_' + algorithm + ")"

source_env = get_source_env(env_name)
target_env = get_target_env(env_name)

transfer_execute(source_env=source_env,
                 target_env=target_env,
                 algo=algorithm,
                 policy_name='MlpPolicy',
                 step_number=3000,
                 step_number_small=3000,
                 callback_check_freq=50,
                 evaluation_step=5,
                 log_dir_w_TL=log_dir_w_TL,
                 log_dir_wo_TL=log_dir_wo_TL,
                 )

plot_results(log_dir_wo_TL, title=extraInfo+"Without-TL Learning Curve", moving_window=moving_window)
plot_results(log_dir_w_TL, title=extraInfo+"With-TL Learning Curve", moving_window=moving_window)
######
print("--- %s seconds ---" % (time.time() - start_time))
