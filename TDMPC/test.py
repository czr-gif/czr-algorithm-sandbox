import gym
import time

env = gym.make('Ant-v4', render_mode = 'human')

env.reset()


for _ in range(100):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    time.sleep(.1)
    if terminated:
        break

env.close()