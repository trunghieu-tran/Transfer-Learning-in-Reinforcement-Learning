from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import PPO
import gym
from tl.new_pendulum_env import NewPendulumEnv
from model_evaluation import evaluate


policy_name = 'MlpPolicy'
step_number = 10000
env_name = 'Pendulum-v1'

source_env = gym.make(env_name)
target_env = NewPendulumEnv(gym.make(env_name))

source_model = DDPG(policy_name, source_env, verbose=2)
print(">>Evaluate un-trained agent:")
evaluate(source_model, 100)

source_model.learn(total_timesteps=step_number)
source_model.save("./source_model_trained")
print(">>Evaluate trained agent:")
evaluate(source_model, 100)

# sample an observation from the environment
obs = source_model.env.observation_space.sample()

# Check prediction before saving
print("pre saved", source_model.predict(obs, deterministic=True))

del source_model # delete trained model to demonstrate loading

##### LOAD source model and train with target domain
target_model = DDPG.load("./source_model_trained")
# Check that the prediction is the same after loading (for the same observation)
print("loaded", target_model.predict(obs, deterministic=True))

# # as the environment is not serializable, we need to set a new instance of the environment
# target_model.set_env(target_env)
# print(">>Evaluate un-trained agent using source model:")
# evaluate(target_model, 100)
# # and continue training
# target_model.learn(step_number)
# print(">>Evaluate trained agent using source model:")
# evaluate(target_model, 100)
