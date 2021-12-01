from tl.utils.transfer_learning import *
from tl.utils.environment_generator import *
import time


start_time = time.time()

#### >>>>>> We only need to env_name and algorithm
# env = ['Pendulum-v1', 'CarRacing-v0', 'Acrobot-v1']
env_name = 'Pendulum-v1'
# for each selected algorithm, please choose algorithm supporting its action space
# We currently focus on TD3,DDPG (for Pendulum) and DQN (for Acrobot)
algorithm = 'TD3'
#### <<<<<<<

moving_window = -1
log_dir_w_TL = "/tmp/gym/w_tl/"
log_dir_wo_TL = "/tmp/gym/wo_tl/"
log_dir_w_TL_rs = "/tmp/gym/w_tl_rs/"
extraInfo = "(" + env_name + '_' + algorithm + ")"

source_env = get_source_env(env_name)
target_env = get_target_env(env_name)

transfer_execute(source_env=source_env,
                 target_env=target_env,
                 algo=algorithm,
                 policy_name='MlpPolicy',
                 step_number=10000,
                 step_number_small=10000,
                 callback_check_freq=20,
                 evaluation_step=20,
                 log_dir_w_TL=log_dir_w_TL,
                 log_dir_wo_TL=log_dir_wo_TL,
                 run_evaluation=False
                 )

plot_results(log_dir_wo_TL, title=extraInfo+"Without-TL Learning Curve", moving_window=moving_window)
plot_results(log_dir_w_TL, title=extraInfo+"With-TL Learning Curve", moving_window=moving_window)
plot_results(log_dir_w_TL_rs, title=extraInfo+"With-TL-Reward-Shapings Learning Curve", moving_window=moving_window)
######
print("--- %s seconds ---" % (time.time() - start_time))
