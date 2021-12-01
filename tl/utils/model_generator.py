from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3 import DQN

algorithms = ['DDPG', 'A2C', 'PPO', 'SAC', 'TD3']


def get_model(policy_name, env, verbose=2, algo='DDPG'):
    if algo == 'DQN':
        return DQN(policy_name, env, verbose=verbose)

    if algo == 'A2C':
        return A2C(policy_name, env, verbose=verbose)

    if algo == 'PPO':
        return PPO(policy_name, env, verbose=verbose)

    if algo == 'TD3':
        return TD3(policy_name, env, verbose=verbose)

    if algo == 'SAC':
        return SAC(policy_name, env, verbose=verbose)

    return DDPG(policy_name, env, verbose=verbose)


def load_model(algo='DDPG', src="./source_model_trained"):
    if algo == 'DQN':
        return DQN.load(src)

    if algo == 'A2C':
        return A2C.load(src)

    if algo == 'PPO':
        return PPO.load(src)

    if algo == 'TD3':
        return TD3.load(src)

    if algo == 'SAC':
        return SAC.load(src)

    return DDPG.load(src)
