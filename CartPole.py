import numpy as np 
import gym
from pilco import PILCO
from actor import LinearActor, Actor 
from rewards import CartPoleReward
import pandas as pd 
import matplotlib.pyplot as plt 
from gym.spaces import Discrete, Box

np.random.seed(0)

env = gym.make('CartPole-v1')

def rollout(policy, timesteps, discrete_ac):
	"""
	get training data for GP to model the transition function
	"""
	X, Y =  [], []
	env.reset()
	x, _, _, _ = env.step(0)

	for timestep in range(timesteps):
		# env.render()
		u = policy(x)
		x_new, _, done, _ = env.step(u)
		if done:
			break
		if discrete_ac:
			u += np.random.normal(0, 0.1)
		X.append(np.hstack((x, u)))
		Y.append(x_new - x)
		x = x_new

	return np.stack(X), np.stack(Y)

def random_policy(x):
	# policy in env
	return env.action_space.sample()

def pilco_policy(x):
	# policy in env
	return pilco.take_action(x)

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

X, Y = rollout(random_policy, 40, discrete_ac)

for i in range(1, 3):
	X_, Y_ = rollout(random_policy, 40, discrete_ac)
	X = np.vstack((X, X_))
	Y = np.vstack((Y, Y_))


# hyperparams
RENDER = False
learning_rate = 0.02
reward_decay = 0.995
max_episode = 20
# max_episode_step = 3000

linear_controller = LinearActor(action_dim=action_dim, action_choice=action_choice, state_dim=state_dim, learning_rate=learning_rate, discrete_ac=discrete_ac)
cartpole_reward = CartPoleReward()  # consider to reset Reward function
pilco = PILCO(X, Y, controller=linear_controller, reward=cartpole_reward, horizon=40)

for rollouts in range(max_episode):
	print("the " + str(rollouts) + "th rollout begins.")
	# optimize the GP
	pilco.optimize_gp()
	# the controller optimization
	states = X[:, 0: state_dim]
	pilco.optimize_controller(states, 40)

	X_new, Y_new = rollout(pilco_policy, 100, discrete_ac)
	# update dataset , why update instead of replace
	# X = np.vstack((X, X_new))
	# Y = np.vstack((Y, Y_new))
	X, Y = X_new, Y_new
	for i in range(1, 3):
		X_, Y_ = rollout(pilco_policy, 100, discrete_ac)
		X = np.vstack((X, X_))
		Y = np.vstack((Y, Y_))

	pilco.mgpr.set_XY(X, Y)
