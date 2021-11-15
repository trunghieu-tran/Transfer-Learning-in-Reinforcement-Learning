from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
import gym
from tl.new_pendulum_env import NewPendulumEnv
from tl.new_acrobot_env import NewAcrobotEnv
from model_evaluation import evaluate

import time
start_time = time.time()


policy_name = 'MlpPolicy'
step_number = 10000

# env = gym.make("Pendulum-v1")
# env = NewPendulumEnv(gym.make("Pendulum-v1"))
# env = gym.make("BipedalWalker-v3")
# env = gym.make('Acrobot-v1')
env = NewAcrobotEnv(gym.make('Acrobot-v1'))
print(env.action_space)
print(env.AVAIL_TORQUE)
# show new action_space if needed
# print(env.action_space.low)
# for _ in range(10):
#   print(env.action_space.sample())

# Define and Train the agent
model = A2C(policy_name, env, verbose=0)
print(">>Evaluate un-trained agent:")
evaluate(model, 10)

# # Evaluate the trained agent
model.learn(total_timesteps=step_number)
print(">>Evaluate trained agent:")
evaluate(model, 100)

print("--- %s seconds ---" % (time.time() - start_time))