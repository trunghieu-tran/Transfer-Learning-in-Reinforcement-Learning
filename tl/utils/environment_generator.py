import gym
from tl.envs.new_pendulum_env import NewPendulumEnv
from tl.envs.new_acrobot_env import NewAcrobotEnv

supported_envs = ['Pendulum-v1', 'Acrobot-v1']

def get_source_env(env_name='Pendulum-v1'):
    return gym.make(env_name)


def get_target_env(env_name='Pendulum-v1'):
    if env_name == 'Acrobot-v1':
        return NewAcrobotEnv(gym.make(env_name))

    return NewPendulumEnv(gym.make(env_name))
