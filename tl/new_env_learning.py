from stable_baselines3 import PPO
import gym
from tl.new_pendulum_env import NewPendulumEnv
from model_evaluation import evaluate


policy_name = 'MlpPolicy'
step_number = 1000

# env = gym.make("Pendulum-v1")
env = NewPendulumEnv(gym.make("Pendulum-v1"))
# show new action_space if needed
# print(env.action_space.low)
# for _ in range(10):
#   print(env.action_space.sample())

# Define and Train the agent
model_ppo = PPO(policy_name, env, verbose=2)
print(">>Evaluate un-trained agent:")
evaluate(model_ppo, 100)

# Evaluate the trained agent
model_ppo.learn(total_timesteps=step_number)
print(">>Evaluate trained agent:")
evaluate(model_ppo, 100)
