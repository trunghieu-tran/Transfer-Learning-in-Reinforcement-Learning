from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
import gym
from tl.new_pendulum_env import NewPendulumEnv
from tl.model_evaluation import evaluate
from tl.plot_utils import *
import time
start_time = time.time()

def transfer_execute_with_DDPG(source_env, target_env):
    source_model = DDPG(policy_name, source_env, verbose=2)
    print(">>[Source] Evaluate un-trained agent:")
    evaluate(source_model, 100)

    source_model.learn(total_timesteps=step_number)
    source_model.save("./source_model_trained")
    print(">>[Source] Evaluate trained agent:")
    evaluate(source_model, 100)

    # sample an observation from the environment
    obs = source_model.env.observation_space.sample()

    # Check prediction before saving
    print("pre saved", source_model.predict(obs, deterministic=True))

    del source_model  # delete trained model to demonstrate loading

    ##### LOAD source model and train with target domain
    target_model = DDPG.load("./source_model_trained")
    # Check that the prediction is the same after loading (for the same observation)
    print("loaded", target_model.predict(obs, deterministic=True))

    # as the environment is not serializable, we need to set a new instance of the environment
    target_env_monitor_with_TL = Monitor(target_env, log_dir_w_TL)
    target_model.set_env(target_env_monitor_with_TL)
    callback_w_TL = SaveOnBestTrainingRewardCallback(check_freq=callback_check_freq, log_dir=log_dir_w_TL)
    print(">>[Target] Evaluate un-trained agent using source model:")
    evaluate(target_model, 100)
    # and continue training
    target_model.learn(step_number_small, callback=callback_w_TL)
    print(">>[Target] Evaluate trained agent using source model:")
    evaluate(target_model, 100)

    #### Train target model without transfer
    target_env_monitor = Monitor(target_env, log_dir_wo_TL)
    callback = SaveOnBestTrainingRewardCallback(check_freq=callback_check_freq, log_dir=log_dir_wo_TL)
    target_model_wo_TL = DDPG(policy_name, target_env_monitor, verbose=2)
    target_model_wo_TL.learn(total_timesteps=step_number_small, callback=callback)
    print(">>[Target] Evaluate trained agent without TL:")
    evaluate(target_model_wo_TL, 100)


######
policy_name = 'MlpPolicy'
step_number = 10000
step_number_small = 1000
env_name = 'Pendulum-v1'

callback_check_freq = 50
moving_window = 30
log_dir_w_TL = "/tmp/gym/w_tl/"
log_dir_wo_TL = "/tmp/gym/wo_tl/"
os.makedirs(log_dir_w_TL, exist_ok=True)
os.makedirs(log_dir_wo_TL, exist_ok=True)

source_env = gym.make(env_name)
target_env = NewPendulumEnv(gym.make(env_name))

transfer_execute_with_DDPG(source_env, target_env)
plot_results(log_dir_wo_TL, title="Without TL Learning Curve", moving_window=moving_window)
plot_results(log_dir_w_TL, title="With TL Learning Curve", moving_window=moving_window)
######
print("--- %s seconds ---" % (time.time() - start_time))
