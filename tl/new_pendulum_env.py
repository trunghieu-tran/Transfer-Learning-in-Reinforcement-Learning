import gym
import numpy as np


class NewPendulumEnv(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Below is original env:
        # self.max_speed = 8
        # self.max_torque = 2.0
        # self.dt = 0.05
        # self.g = g (= 10.0)
        # self.m = 1.0
        # self.l = 1.0
        # self.viewer = None
        #
        # high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        # self.action_space = spaces.Box(
        #     low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        # )
        # shape=(HEIGHT, WIDTH, N_CHANNELS)
        # self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Modified env
        # Retrieve the action space
        action_space = env.action_space
        assert isinstance(action_space,
                          gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
        # Retrieve the max/min values
        self.low, self.high = action_space.low, action_space.high

        # We modify the action space, so all actions will lie in [-1, 1]
        # env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)

        # Call the parent constructor, so we can access self.env later
        super(NewPendulumEnv, self).__init__(env)

    def rescale_action(self, scaled_action):
        """
        Rescale the action from [low, high] to [0, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return np.clip(scaled_action, 0, self.high)
        # tmp = self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))
        # tmp = scaled_action
        # if tmp > 0:
        #     return tmp
        # else:
        #     return 0

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
        # Rescale the action from [low, high] to [0, high]
        rescaled_action = self.rescale_action(action)
        obs, reward, done, info = self.env.step(rescaled_action)
        # add reward-shaping here
        # TODO
        #
        return obs, reward, done, info



# What is Box in openAI gym?
# Box means that you are dealing with real valued quantities.
#
# The first array np.array([-1,0,0] are the lowest accepted values,
# and the second np.array([+1,+1,+1]) are the highest accepted values.
# In this case (using the comment) we see that we have 3 available actions:
#
# Steering: Real valued in [-1, 1]
# Gas: Real valued in [0, 1]
# Brake: Real valued in [0, 1]

# gym.spaces.Box: A (possibly unbounded) box in  𝑅^𝑛 .
# Specifically, a Box represents the Cartesian product of n closed intervals.
# Each interval has the form of one of [a, b], (-oo, b], [a, oo), or (-oo, oo).
# Example: A 1D-Vector or an image observation can be described with the Box space.
#
# # Example for using image as input:
# observation_space = spaces.Box(low=0, high=255,
#                                shape=(HEIGHT, WIDTH, N_CHANNELS),
#                                dtype=np.uint8)



