import gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("Pendulum-v1")

model_ppo = PPO("MlpPolicy", env, verbose=2)
model_a2c = A2C("MlpPolicy", env, verbose=2)
model_ddpg = DDPG("MlpPolicy", env, verbose=2)


# Initial model evaluation (without training)
mean_reward, std_reward = evaluate_policy(model_ppo, env, n_eval_episodes=100)
print(f"PPO mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# mean_reward, std_reward = evaluate_policy(model_a2c, env, n_eval_episodes=100)
# print(f"A2C mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
#
# mean_reward, std_reward = evaluate_policy(model_ddpg, env, n_eval_episodes=100)
# print(f"DDPG mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


# Evaluate the trained agent
step_number = 100
#PPO
model_ppo.learn(total_timesteps=step_number)
mean_reward, std_reward = evaluate_policy(model_ppo, env, n_eval_episodes=100)
print(f"PPO mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")