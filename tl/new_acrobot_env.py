import gym
import numpy as np

"""
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    
    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    
    **ACTIONS:**
    The orginal action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.
    However, in the new env, we consider only +1 and 0 torque
    
    .. note::
        The dynamics equations were missing some terms in the NIPS paper which
        are present in the book. R. Sutton confirmed in personal correspondence
        that the experimental results shown in the paper and the book were
        generated with the equations shown in the book.
        However, there is the option to run the domain with the paper equations
        by setting book_or_nips = 'nips'
    **REFERENCE:**
    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
"""


class NewAcrobotEnv(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped

    """

    r"""A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.

    Example::

        >>> Discrete(2)

    """

    def __init__(self, env):


        action_space = env.action_space
        assert isinstance(action_space,
                          gym.spaces.Discrete), "This wrapper only works with discrete action space (spaces.Discrete)"
        # Retrieve the max/min values
        # self.low, self.high = action_space.low, action_space.high

        ### We modify the action Discrete(3) -> Discrete(2)
        self.action_space_transformation(env)

        # Call the parent constructor, so we can access self.env later
        super(NewAcrobotEnv, self).__init__(env)


    def action_space_transformation(self, env):
        # Original env
        # AVAIL_TORQUE = [-1.0, 0.0, +1]
        # torque = self.AVAIL_TORQUE[action]
        # self.action_space = spaces.Discrete(3)

        modified_AVAIL_TORQUE = env.AVAIL_TORQUE
        modified_AVAIL_TORQUE[1] = +1
        env.AVAIL_TORQUE = modified_AVAIL_TORQUE

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
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
