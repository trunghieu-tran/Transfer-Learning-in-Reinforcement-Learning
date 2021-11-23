from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3


import gym
from tl.envs.new_pendulum_env import NewPendulumEnv
from tl.utils.model_evaluation import evaluate

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
    target_model.set_env(target_env)
    print(">>[Target] Evaluate un-trained agent using source model:")
    evaluate(target_model, 100)
    # and continue training
    target_model.learn(step_number_small)
    print(">>[Target] Evaluate trained agent using source model:")
    evaluate(target_model, 100)

    #### Train target model without transfer
    target_model_wo_TL = DDPG(policy_name, target_env, verbose=2)
    target_model_wo_TL.learn(total_timesteps=step_number_small)
    print(">>[Target] Evaluate trained agent without TL:")
    evaluate(target_model_wo_TL, 100)


def transfer_execute_with_A2C(source_env, target_env):
    source_model = A2C(policy_name, source_env, verbose=2)
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
    target_model = A2C.load("./source_model_trained")
    # Check that the prediction is the same after loading (for the same observation)
    print("loaded", target_model.predict(obs, deterministic=True))

    # as the environment is not serializable, we need to set a new instance of the environment
    target_model.set_env(target_env)
    print(">>[Target] Evaluate un-trained agent using source model:")
    evaluate(target_model, 100)
    # and continue training
    target_model.learn(step_number_small)
    print(">>[Target] Evaluate trained agent using source model:")
    evaluate(target_model, 100)

    #### Train target model without transfer
    target_model_wo_TL = A2C(policy_name, target_env, verbose=2)
    target_model_wo_TL.learn(total_timesteps=step_number_small)
    print(">>[Target] Evaluate trained agent without TL:")
    evaluate(target_model_wo_TL, 100)

def transfer_execute_with_PPO(source_env, target_env):
    source_model = PPO(policy_name, source_env, verbose=2)
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
    target_model = PPO.load("./source_model_trained")
    # Check that the prediction is the same after loading (for the same observation)
    print("loaded", target_model.predict(obs, deterministic=True))

    # as the environment is not serializable, we need to set a new instance of the environment
    target_model.set_env(target_env)
    print(">>[Target] Evaluate un-trained agent using source model:")
    evaluate(target_model, 100)
    # and continue training
    target_model.learn(step_number_small)
    print(">>[Target] Evaluate trained agent using source model:")
    evaluate(target_model, 100)

    #### Train target model without transfer
    target_model_wo_TL = PPO(policy_name, target_env, verbose=2)
    target_model_wo_TL.learn(total_timesteps=step_number_small)
    print(">>[Target] Evaluate trained agent without TL:")
    evaluate(target_model_wo_TL, 100)

def transfer_execute_with_SAC(source_env, target_env):
    source_model = SAC(policy_name, source_env, verbose=2)
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
    target_model = SAC.load("./source_model_trained")
    # Check that the prediction is the same after loading (for the same observation)
    print("loaded", target_model.predict(obs, deterministic=True))

    # as the environment is not serializable, we need to set a new instance of the environment
    target_model.set_env(target_env)
    print(">>[Target] Evaluate un-trained agent using source model:")
    evaluate(target_model, 100)
    # and continue training
    target_model.learn(step_number_small)
    print(">>[Target] Evaluate trained agent using source model:")
    evaluate(target_model, 100)

    #### Train target model without transfer
    target_model_wo_TL = SAC(policy_name, target_env, verbose=2)
    target_model_wo_TL.learn(total_timesteps=step_number_small)
    print(">>[Target] Evaluate trained agent without TL:")
    evaluate(target_model_wo_TL, 100)

def transfer_execute_with_TD3(source_env, target_env):
    source_model = TD3(policy_name, source_env, verbose=2)
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
    target_model = TD3.load("./source_model_trained")
    # Check that the prediction is the same after loading (for the same observation)
    print("loaded", target_model.predict(obs, deterministic=True))

    # as the environment is not serializable, we need to set a new instance of the environment
    target_model.set_env(target_env)
    print(">>[Target] Evaluate un-trained agent using source model:")
    evaluate(target_model, 100)
    # and continue training
    target_model.learn(step_number_small)
    print(">>[Target] Evaluate trained agent using source model:")
    evaluate(target_model, 100)

    #### Train target model without transfer
    target_model_wo_TL = TD3(policy_name, target_env, verbose=2)
    target_model_wo_TL.learn(total_timesteps=step_number_small)
    print(">>[Target] Evaluate trained agent without TL:")
    evaluate(target_model_wo_TL, 100)

######
policy_name = 'MlpPolicy'
step_number = 10000
step_number_small = 1000
env_name = 'Pendulum-v1'

source_env = gym.make(env_name)
target_env = NewPendulumEnv(gym.make(env_name))

transfer_execute_with_DDPG(source_env, target_env)
# transfer_execute_with_PPO(source_env, target_env)
# transfer_execute_with_A2C(source_env, target_env)
# transfer_execute_with_SAC(source_env, target_env)
# transfer_execute_with_TD3(source_env, target_env)
######
print("--- %s seconds ---" % (time.time() - start_time))

# ##### Render
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()
#
# env.close()
