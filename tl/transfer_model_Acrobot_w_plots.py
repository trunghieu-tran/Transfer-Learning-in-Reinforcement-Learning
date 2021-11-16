from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from tl.new_acrobot_env import NewAcrobotEnv
import gym
from model_evaluation import evaluate
from tl.plot_utils import *
import time
start_time = time.time()

def transfer_execute_with_A2C(source_env, target_env):
    source_model = A2C(policy_name, source_env, verbose=verbose)
    print(">>[Source] Evaluate un-trained agent:")
    evaluate(source_model, evaluate_episode_num)

    source_model.learn(total_timesteps=step_number)
    source_model.save("./source_model_trained")
    print(">>[Source] Evaluate trained agent:")
    evaluate(source_model, evaluate_episode_num)

    # sample an observation from the environment
    obs = source_model.env.observation_space.sample()

    # Check prediction before saving
    print("pre saved", source_model.predict(obs, deterministic=True))

    del source_model  # delete trained model to demonstrate loading

    ##### LOAD source model and train with target domain
    target_model = A2C.load("./source_model_trained")
    # Check that the prediction is the same after loading (for the same observation)
    print("loaded", target_model.predict(obs, deterministic=True))

    # as the environment is not serializable, we need to set a new instance of the environment
    target_env_monitor_with_TL = Monitor(target_env, log_dir_w_TL)
    target_model.set_env(target_env_monitor_with_TL)
    callback_w_TL = SaveOnBestTrainingRewardCallback(check_freq=callback_check_freq, log_dir=log_dir_w_TL)
    # and continue training
    target_model.learn(step_number_small, callback=callback_w_TL)
    print(">>[Target] Evaluate trained agent using source model:")
    evaluate(target_model, evaluate_episode_num)

    #### Train target model without transfer
    target_env_monitor = Monitor(target_env, log_dir_wo_TL)
    callback = SaveOnBestTrainingRewardCallback(check_freq=callback_check_freq, log_dir=log_dir_wo_TL)
    target_model_wo_TL = A2C(policy_name, target_env_monitor, verbose=verbose)
    target_model_wo_TL.learn(total_timesteps=step_number_small, callback=callback)
    print(">>[Target] Evaluate trained agent without TL:")
    evaluate(target_model_wo_TL, evaluate_episode_num)


######
policy_name = 'MlpPolicy'
step_number = 10000
step_number_small = 10000
evaluate_episode_num = 100
verbose = 2
env_name = 'Acrobot-v1'

callback_check_freq = 200
moving_window = 50
log_dir_w_TL = "/tmp/gym/w_tl/"
log_dir_wo_TL = "/tmp/gym/wo_tl/"
os.makedirs(log_dir_w_TL, exist_ok=True)
os.makedirs(log_dir_wo_TL, exist_ok=True)

source_env = gym.make(env_name)
target_env = NewAcrobotEnv(gym.make(env_name))

transfer_execute_with_A2C(source_env, target_env)
extraInfo = "(Acrobot_A2C)"

plot_results(log_dir_wo_TL, title="Without-TL Learning Curve"+extraInfo, moving_window=moving_window)
plot_results(log_dir_w_TL, title="With-TL Learning Curve"+extraInfo, moving_window=moving_window)
######
print("--- %s seconds ---" % (time.time() - start_time))
