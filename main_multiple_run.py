from tl.utils.transfer_learning import *
from tl.utils.environment_generator import *
import time


start_time = time.time()

#### >>>>>> We only need to env_name and algorithm
# env = ['Pendulum-v1', 'CarRacing-v0', 'Acrobot-v1']
env_name = 'Acrobot-v1'
# for each selected algorithm, please choose algorithm supporting its action space
# We currently focus on TD3,DDPG (for Pendulum) and DQN (for Acrobot)
algorithm = 'DQN'

#### <<<<<<<
moving_window = 3
running_time = 5
source_step_number = 10000
target_step_number = 10000
log_dir_w_TL = "/tmp/gym/w_tl/"
log_dir_wo_TL = "/tmp/gym/wo_tl/"
log_dir_w_TL_rs = "/tmp/gym/w_tl_rs/"

for i in range(running_time):

    i_log_dir_w_TL = log_dir_w_TL + str(i) + "/"
    i_log_dir_wo_TL = log_dir_wo_TL + str(i) + "/"
    i_log_dir_w_TL_rs = log_dir_w_TL_rs + str(i) + "/"

    extraInfo = "(" + env_name + '_' + algorithm + ")"

    source_env = get_source_env(env_name)
    target_env = get_target_env(env_name)

    transfer_execute(source_env=source_env,
                     target_env=target_env,
                     algo=algorithm,
                     policy_name='MlpPolicy',
                     step_number=source_step_number,
                     step_number_small=target_step_number,
                     callback_check_freq=20,
                     evaluation_step=20,
                     log_dir_w_TL=i_log_dir_w_TL,
                     log_dir_wo_TL=i_log_dir_wo_TL,
                     log_dir_w_TL_rs=i_log_dir_w_TL_rs,
                     run_evaluation=False
                     )

# This code is used for plotting log seperately
# plot_results(log_dir_wo_TL, title=extraInfo+"Without-TL Learning Curve", moving_window=moving_window)
# plot_results(log_dir_w_TL, title=extraInfo+"With-TL Learning Curve", moving_window=moving_window)
# plot_results(log_dir_w_TL_rs, title=extraInfo+"With-TL-Reward-Shapings Learning Curve", moving_window=moving_window)

# Use this code to plot all logs together
# plot_multiple_results(log_dir_w_TL=log_dir_w_TL,
#                       log_dir_wo_TL=log_dir_wo_TL,
#                       log_dir_w_TL_rs=log_dir_w_TL_rs,
#                       log_dir_w_full_TL_rs=log_dir_w_full_TL_rs,
#                       title=extraInfo,
#                       moving_window=moving_window)

plot_multiple_results_with_multiple_runing_time(
                      log_dir_w_TL=log_dir_w_TL,
                      log_dir_wo_TL=log_dir_wo_TL,
                      log_dir_w_TL_rs=log_dir_w_TL_rs,
                      title=extraInfo,
                      moving_window=moving_window,
                      running_time_num=running_time)
######
print("--- %s seconds ---" % (time.time() - start_time))
