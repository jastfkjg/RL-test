import numpy as np 
import gym
from pilco import PILCO
from utils import save_pilco, load_pilco
from actor import LinearActor, Actor 
from rewards import *
import pandas as pd
import matplotlib.pyplot as plt 
from gym.spaces import Discrete, Box
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("env_name", help="gym env name: classic control (CartPole-v1, MountainCarContinuous-v0, Pendulum-v0 ... )")

args = parser.parse_args()

# np.random.seed(0)

# classic control:
# env = gym.make('CartPole-v1')
# MountainCarContinuous-v0, Pendulum-v0

# mujoco:
# InvertedPendulum-v2, Swimmer-v2
# Reacher-v2, Walker2d-v2, Humanoid-v2
# Ant-v2, InvertedDoublePendulum-v2

env = gym.make(args.env_name)
model_path = './checkpoints/' + args.env_name + '/'

def rollout(policy, timesteps, discrete_ac=False):
	"""
	get training data for GP to model the transition function
	"""
	X, Y = [], []
	total_step, total_reward = 0, 0.
	env.reset()
	x, _, _, _ = env.step([0])

	for timestep in range(timesteps):
		total_step += 1
		# env.render()
		u = policy(x)
		x_new, r, done, _ = env.step(u)
		total_reward += r
		if done:
			break
		if discrete_ac:
			u += np.random.normal(0, 0.1)
		X.append(np.hstack((x, u)))
		Y.append(x_new - x)
		x = x_new

	return np.stack(X), np.stack(Y), total_step, total_reward


def random_policy(x):
	# policy in env
	return env.action_space.sample()


def pilco_policy(x):
	# policy in env
	# return pilco.take_action(x)
	return pilco.controller.take_quick_action(x)


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

# TODO
X, Y, total_step, reward = rollout(random_policy, 40, discrete_ac)

for i in range(1, 3):
	X_, Y_, step, reward = rollout(random_policy, 40, discrete_ac)
	total_step += step
	X = np.vstack((X, X_))
	Y = np.vstack((Y, Y_))

# we have here normally <40*3=120 dat.shape[0]a

# print("X: ", X)
# print("Y: ", Y)

# hyperparams
RENDER = False
learning_rate = 0.02
reward_decay = 0.995
max_episode = 20
# num_optim: how many real states to use as init state, default: None(use all real states)
num_optim = 40
# num_collect: how many fake data are we going to create in a batch (batch num) to optimize actor
num_collect = 10
# optim horizon: the horizon to calculate expected reward
optim_horizon = 20
# max_episode_step = 3000

controller = Actor(action_dim=action_dim, state_dim=state_dim, learning_rate=learning_rate, discrete_ac=discrete_ac)
# # load actor
try:
	controller.load_weights(model_path + 'actor.ckpt')
	print("load actor model successfully")
except:
	print("can not find checkpoint.")

# TODO
if args.env_name == 'CartPole-v1':
	env_reward = CartPoleReward()
elif args.env_name == 'Pendulum-v0':
	env_reward = PendulumReward()

# load gp
try:
	pilco = load_pilco(model_path, controller, env_reward)
	print("load gp model successfully")
except:
	print("can not find saved gp models")
	pilco = PILCO(X, Y, controller=controller, reward=env_reward)

# for numerical stability
for model in pilco.mgpr.models:
	model.likelihood.variance = 0.001
	model.likelihood.variance.trainable = False

print("num optim already: ", pilco.controller.get_num_optim())

reward_list = []
total_episode = 2
ep_step_list = []
X_init = X[:, 0: state_dim]

for rollouts in range(max_episode):
	print("***" * 30)
	print("the " + str(rollouts) + "th rollout begins.")
	print("num optim: ", pilco.controller.get_num_optim())
	print("***" * 30)
	# optimize GP
	pilco.optimize_gp()

	# the controller optimization
	# states = X[:, 0: state_dim]
	# print(states.shape[0])
	pilco.optimize_controller(X_init, optim_horizon, num_optim=num_optim, num_collect=num_collect, gamma=reward_decay)

	# save controller's weights
	print("saving the controller.")
	pilco.controller.save_weights(model_path + 'actor.ckpt')

	X_new, Y_new, step, reward = rollout(pilco_policy, 100, discrete_ac)
	# update dataset, why update instead of replace
	X = np.vstack((X, X_new))
	Y = np.vstack((Y, Y_new))
	pilco.mgpr.set_XY(X, Y)
	X_init = np.array(X_new)[:, 0: state_dim]

	total_step += step
	ep_step_list.append(total_step)
	reward_list.append(reward)
	total_episode += 1

# save controller's weights
pilco.controller.save_weights(model_path + 'actor.ckpt')
# save gp model and data
save_pilco(model_path, X, Y, pilco)

print("reward list: ", reward_list)

print("Finished !")
plt.subplot(2, 2, 1)
plt.plot(np.arange(1, len(reward_list)+1), reward_list)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
# plt.show()

plt.subplot(2, 2, 2)
plt.plot(ep_step_list, reward_list)
plt.xlabel('Total steps')
plt.ylabel('Episode Reward')
# plt.show()

test_episode = 10
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

		action = pilco_policy(observation)
		observation_next, reward, done, info = env.step(action)
		total_reward += reward

		if done or step >= max_step:
			test_reward_list.append(total_reward)

			print("Episode: ", i_episode, " step in this episode:", step, " reward: ", total_reward)
			break

		# update observation
		observation = observation_next

plt.subplot(2, 2, 3)
plt.plot(np.arange(test_episode), test_reward_list)
plt.xlabel('Test Episode')
plt.ylabel('Test Episode Reward')
plt.show()

# dataFrame = pd.DataFrame({'reward': reward_list})
# dataFrame.to_csv("./pilco_output/episode_reward.csv", index=True, sep=',')
#
# dataFrame2 = pd.DataFrame({'total_step': ep_step_list, 'reward': reward_list})
# dataFrame2.to_csv("./pilco_output/step_reward.csv", index=False, sep=',')


