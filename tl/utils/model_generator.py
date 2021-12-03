from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3 import DQN

from stable_baselines3.ddpg.reward_shaped_ddpg import RewardShapedDDPG
from stable_baselines3.td3.reward_shaped_td3 import RewardShapedTD3
from stable_baselines3.dqn.reward_shaped_dqn import RewardShapedDQN
from tl.utils.reward_shapers import create_td3_reward_shaper, create_dqn_reward_shaper, create_ddpg_reward_shaper


algorithms = ['DDPG', 'A2C', 'PPO', 'SAC', 'TD3']


def get_reward_shaping_model(policy_name, env, src_model, num_sampling_episodes=10, verbose=2, algo='DDPG'):
    if algo == 'DQN':
        reward_shaper = create_dqn_reward_shaper(src_model, num_sampling_episodes)
        target_model_reshape = RewardShapedDQN(policy_name, env, verbose=verbose, reward_shaper=reward_shaper)
        return target_model_reshape

    if algo == 'TD3':
        reward_shaper = create_td3_reward_shaper(src_model, num_sampling_episodes)
        target_model_reshape = RewardShapedTD3(policy_name, env, verbose=verbose, reward_shaper=reward_shaper)
        return target_model_reshape

    reward_shaper = create_ddpg_reward_shaper(src_model, num_sampling_episodes)
    target_model_reshape = RewardShapedDDPG(policy_name, env, verbose=verbose, reward_shaper=reward_shaper)
    return target_model_reshape

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
