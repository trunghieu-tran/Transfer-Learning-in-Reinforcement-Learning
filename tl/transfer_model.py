import sys
sys.path.append("C:/Users/nbeck/OneDrive/Documents/Github/Transfer-Learning-in-Reinforcement-Learning")

from stable_baselines3.ddpg.ddpg import DDPG
from stable_baselines3.td3.td3 import TD3
from stable_baselines3.ddpg.reward_shaped_ddpg import RewardShapedDDPG
from stable_baselines3.td3.reward_shaped_td3 import RewardShapedTD3
from stable_baselines3.dqn.reward_shaped_dqn import RewardShapedDQN
import gym
from tl.new_pendulum_env import NewPendulumEnv
from tl.utils.reward_shapers import create_ddpg_reward_shaper, create_dqn_reward_shaper, create_td3_reward_shaper
from model_evaluation import evaluate

policy_name = 'MlpPolicy'
step_number = 100000
step_number_small = 5000
env_name = 'Pendulum-v1'

source_env = gym.make(env_name)
target_env = NewPendulumEnv(gym.make(env_name))

source_model = TD3(policy_name, source_env, verbose=2)
print(">>[Source] Evaluate un-trained agent:")
evaluate(source_model, 100)

source_model.learn(total_timesteps=step_number)
print(">>[Source] Evaluate trained agent:")
evaluate(source_model, 100)

##### LOAD source model and train with target domain
num_sampling_episodes = 10
reward_shaper = create_td3_reward_shaper(source_model, num_sampling_episodes)
target_model = RewardShapedTD3(policy_name, target_env, verbose=2, reward_shaper=reward_shaper)

# as the environment is not serializable, we need to set a new instance of the environment
target_model.set_env(target_env)
print(">>[Target] Evaluate un-trained agent using source model:")
evaluate(target_model, 100)
# and continue training
target_model.learn(step_number_small)
print(">>[Target] Evaluate trained agent using source model:")
evaluate(target_model, 100)

scratch_target_model = TD3(policy_name, target_env, verbose=2)
print(">>[Source] Evaluate un-trained agent:")
evaluate(scratch_target_model, 100)

scratch_target_model.learn(total_timesteps=step_number_small)
print(">>[Source] Evaluate trained agent:")
evaluate(scratch_target_model, 100)