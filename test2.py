import gym
# import mujoco_py

env = gym.make('CartPole-v0')
# env = gym.make('HalfCheetah-v3')
# env = gym.make('MountainCar-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        print(">>> e={}, t={}\n".format(i_episode,t))
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()