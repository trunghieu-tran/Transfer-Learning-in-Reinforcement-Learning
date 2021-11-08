import gym
import numpy as np

"""
Easiest continuous control task to learn from pixels, a top-down racing
environment.
Discrete control is reasonable in this environment as well, on/off
discretization is fine.

State consists of STATE_W x STATE_H pixels.

The reward is -0.1 every frame and +1000/N for every track tile visited, where
N is the total number of tiles visited in the track. For example, if you have
finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

The game is solved when the agent consistently gets 900+ points. The generated
track is random every episode.

The episode finishes when all the tiles are visited. The car also can go
outside of the PLAYFIELD -  that is far off the track, then it will get -100
and die.

Some indicators are shown at the bottom of the window along with the state RGB
buffer. From left to right: the true speed, four ABS sensors, the steering
wheel position and gyroscope.

To play yourself (it's rather fast for humans), type:

python gym/envs/box2d/car_racing.py

Remember it's a powerful rear-wheel drive car -  don't press the accelerator
and turn at the same time.

CarRacing is created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""
class NewCarRacingEnv(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):

        # self.action_space = spaces.Box(
        #     np.array([-1, 0, 0]).astype(np.float32),
        #     np.array([+1, +1, +1]).astype(np.float32),
        # )  # steer, gas, brake
        #
        action_space = env.action_space
        assert isinstance(action_space,
                          gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
        # Retrieve the max/min values
        self.low, self.high = action_space.low, action_space.high

        # Call the parent constructor, so we can access self.env later
        super(NewCarRacingEnv, self).__init__(env)

    def rescale_action(self, scaled_action):
        """
        Rescale the action from [low, high] to [0, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        # return np.clip(scaled_action, 0, self.high)
        return np.clip(scaled_action, self.low / 2, self.high / 2)
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



