import gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DDPG

import numpy as np


class NormalizeActionWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Retrieve the action space
        action_space = env.action_space
        assert isinstance(action_space,
                          gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
        # Retrieve the max/min values
        self.low, self.high = action_space.low, action_space.high

        # We modify the action space, so all actions will lie in [-1, 1]
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)

        # Call the parent constructor, so we can access self.env later
        super(NormalizeActionWrapper, self).__init__(env)

    def rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    def reset(self):
        """
        Reset the environment
        """
        # Reset the counter
        return self.env.reset()

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self.rescale_action(action)
        obs, reward, done, info = self.env.step(rescaled_action)
        return obs, reward, done, info



# env = gym.make("CartPole-v1")

# original_env= gym.make("Pendulum-v1")
# print(original_env.action_space.low)
# for _ in range(10):
#   print(original_env.action_space.sample())

env = NormalizeActionWrapper(gym.make("Pendulum-v1"))
print(env.action_space.low)
for _ in range(10):
  print(env.action_space.sample())

# The Proximal Policy Optimization algorithm combines ideas from A2C (having multiple workers)
# and TRPO (it uses a trust region to improve the actor).
# The main idea is that after an update, the new policy should be not too far form the old policy.
# For that, ppo uses clipping to avoid too large update.
# Details at: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Learn with A2C. Details at https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
# A synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C).
# It uses multiple workers to avoid the use of a replay buffer.
# model = A2C('MlpPolicy', 'CartPole-v1').learn(10000)

# Learn with DDPG. Details at https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html
# model = DDPG("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()

from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")