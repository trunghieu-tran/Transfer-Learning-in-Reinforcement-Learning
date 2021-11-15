from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import HER

from tl.new_acrobot_env import NewAcrobotEnv
import gym
from model_evaluation import evaluate

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
    target_model.set_env(target_env)
    # print(">>[Target] Evaluate un-trained agent using source model:")
    # evaluate(target_model, evaluate_episode_num)
    # and continue training
    target_model.learn(step_number_small)
    print(">>[Target] Evaluate trained agent using source model:")
    evaluate(target_model, evaluate_episode_num)

    #### Train target model without transfer
    target_model_wo_TL = A2C(policy_name, source_env, verbose=verbose)
    target_model_wo_TL.learn(total_timesteps=step_number_small)
    print(">>[Target] Evaluate trained agent without TL:")
    evaluate(target_model_wo_TL, evaluate_episode_num)

def transfer_execute_with_PPO(source_env, target_env):
    source_model = PPO(policy_name, source_env, verbose=verbose)
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
    target_model = PPO.load("./source_model_trained")
    # Check that the prediction is the same after loading (for the same observation)
    print("loaded", target_model.predict(obs, deterministic=True))

    # as the environment is not serializable, we need to set a new instance of the environment
    target_model.set_env(target_env)
    # print(">>[Target] Evaluate un-trained agent using source model:")
    # evaluate(target_model, evaluate_episode_num)
    # and continue training
    target_model.learn(step_number_small)
    print(">>[Target] Evaluate trained agent using source model:")
    evaluate(target_model, evaluate_episode_num)

    #### Train target model without transfer
    target_model_wo_TL = PPO(policy_name, source_env, verbose=verbose)
    target_model_wo_TL.learn(total_timesteps=step_number_small)
    print(">>[Target] Evaluate trained agent without TL:")
    evaluate(target_model_wo_TL, evaluate_episode_num)

def transfer_execute_with_DQN(source_env, target_env):
    source_model = DQN(policy_name, source_env, verbose=verbose)
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
    target_model = DQN.load("./source_model_trained")
    # Check that the prediction is the same after loading (for the same observation)
    print("loaded", target_model.predict(obs, deterministic=True))

    # as the environment is not serializable, we need to set a new instance of the environment
    target_model.set_env(target_env)
    # print(">>[Target] Evaluate un-trained agent using source model:")
    # evaluate(target_model, evaluate_episode_num)
    # and continue training
    target_model.learn(step_number_small)
    print(">>[Target] Evaluate trained agent using source model:")
    evaluate(target_model, evaluate_episode_num)

    #### Train target model without transfer
    target_model_wo_TL = DQN(policy_name, target_env, verbose=verbose)
    target_model_wo_TL.learn(total_timesteps=step_number_small)
    print(">>[Target] Evaluate trained agent without TL:")
    evaluate(target_model_wo_TL, evaluate_episode_num)

######
policy_name = 'MlpPolicy'
step_number = 10000
step_number_small = 1000
evaluate_episode_num = 100
verbose = 2
env_name = 'Acrobot-v1'

source_env = gym.make(env_name)
target_env = NewAcrobotEnv(gym.make(env_name))

# transfer_execute_with_A2C(source_env, target_env)
transfer_execute_with_PPO(source_env, target_env)
# transfer_execute_with_DQN(source_env, target_env)
######
print("--- %s seconds ---" % (time.time() - start_time))
