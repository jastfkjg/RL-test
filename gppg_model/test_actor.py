import tensorflow as tf
import numpy as np
import gym
from actor import Actor
import matplotlib.pyplot as plt
from gym.spaces import Discrete, Box
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("env_name", help="gym env name: classic control (CartPole-v1, MountainCarContinuous-v0, Pendulum-v0 ... )")

args = parser.parse_args()

model_path = './checkpoints/' + args.env_name + '/'

env = gym.make(args.env_name)

# hyperparams
learning_rate = 0.02
reward_decay = 0.995
max_episode = 2


if isinstance(env.observation_space, Discrete):
    state_dim = env.observation_space.n
elif isinstance(env.observation_space, Box):
    state_dim = env.observation_space.shape[0]

if isinstance(env.action_space, Discrete):
    action_choice = env.action_space.n
    action_dim = 1
    discrete_ac = True
elif isinstance(env.action_space, Box):
    action_dim = env.action_space.shape[0]
    discrete_ac = False

controller = Actor(action_dim=action_dim, state_dim=state_dim, learning_rate=learning_rate, discrete_ac=discrete_ac)
controller.load_weights(model_path + 'actor.ckpt')

test_episode = 100
max_step = 200
render = False
test_reward_list = []

for i_episode in range(test_episode):
    observation = env.reset()
    step = 0
    total_reward = 0.0

    while True:
        step += 1
        if render:
            env.render()

        action = controller.take_quick_action(observation)
        observation_next, reward, done, info = env.step(action)
        total_reward += reward

        if done or step >= max_step:
            test_reward_list.append(total_reward)

            print("Episode: ", i_episode, " step in this episode:", step, " reward: ", total_reward)
            break

        # update observation
        observation = observation_next

random_reward_list = []
for i_episode in range(test_episode):
    observation = env.reset()
    step = 0
    total_reward = 0.0

    while True:
        step += 1
        if render:
            env.render()

        action = env.action_space.sample()
        observation_next, reward, done, info = env.step(action)
        total_reward += reward

        if done or step >= max_step:
            random_reward_list.append(total_reward)

            print("Episode: ", i_episode, " step in this episode:", step, " reward: ", total_reward)
            break

        # update observation
        observation = observation_next


plt.subplot(2, 1, 1)
plt.plot(np.arange(test_episode), test_reward_list)
plt.xlabel('Test Episode')
plt.ylabel('Test Episode Reward')
plt.subplot(2, 1, 2)
plt.plot(np.arange(test_episode), random_reward_list)
plt.xlabel('Episode')
plt.ylabel('Random Reward')
plt.show()

