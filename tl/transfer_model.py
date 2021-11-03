from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import PPO
import gym
from tl.new_pendulum_env import NewPendulumEnv
from model_evaluation import evaluate

policy_name = 'MlpPolicy'
step_number = 10000
step_number_small = 1000
env_name = 'Pendulum-v1'

source_env = gym.make(env_name)
target_env = NewPendulumEnv(gym.make(env_name))

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
target_model_wo_TL = DDPG(policy_name, source_env, verbose=2)
target_model_wo_TL.learn(total_timesteps=step_number_small)
print(">>[Target] Evaluate trained agent without TL:")
evaluate(target_model_wo_TL, 100)

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
