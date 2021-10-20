import gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C

env = gym.make("CartPole-v1")

# The Proximal Policy Optimization algorithm combines ideas from A2C (having multiple workers)
# and TRPO (it uses a trust region to improve the actor).
# The main idea is that after an update, the new policy should be not too far form the old policy.
# For that, ppo uses clipping to avoid too large update.
# Details at: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000)

# Learn with A2C. Details at https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
# A synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C).
# It uses multiple workers to avoid the use of a replay buffer.
model = A2C('MlpPolicy', 'CartPole-v1').learn(10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()