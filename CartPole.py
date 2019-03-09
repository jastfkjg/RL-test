import numpy as np 
import gym
from pilco.models import PILCO 
from actor import LinearActor, Actor 
from rewards import CartPoleReward
import pandas as pd 
import matplotlib.pyplot as plt 
from gym.spaces import Discrete, Box

np.random.seed(0)

env = gym.make('CartPole-v1')

def rollout(policy, timesteps):
	X, Y =  [], []
	env.reset()
	x, _, _, _ = env.step(0)

	for timestep in range(timesteps):
		env.render()
		u = policy(x)
		x_new, _, done, _ = env.step(u)
		if done:
			break
		X.append(np.hstack((x, u)))
		Y.append(x_new - x)
		x = x_new

	return np.stack(X), np.stack(Y)

def random_policy(x):
	return env.action_space.sample()

def pilco_policy(x):
	return pilco.compute_action(x)

X, Y = rollout(policy=random_policy, timesteps=40)

for i in range(1, 3):
	X_, Y_ = rollout(policy=random_policy, timesteps=40)
	X = np.vstack((X, X_))
	Y = np.vstack((Y, Y_))

if isinstance(env.observation_space, Discrete):
	S_DIM = env.observation_space.n
elif isinstance(env.observation_space, Box):
	S_DIM = env.observation_space.shape[0]

if isinstance(env.action_space, Discrete):
	A_DIM = env.action_space.n
elif isinstance(env.action_space, Box):
	A_DIM = env.action_space.shape[0]

# hyperparams
RENDER = False
learning_rate = 0.02
reward_decay = 0.995
max_episode = 20
# max_episode_step = 3000

linear_controller = LinearActor(action_dim=A_DIM, state_dim=S_DIM, learning_rate=learning_rate)
cartpole_reward = CartPoleReward()  # state ? consider to reset Reward function
pilco = PILCO(X, Y, controller=linear_controller, reward=cartpole_reward, horizon=40)

for rollouts in range(max_episode):
	pilco.optimize()
	# the controller optimize ?
	states = X[:, 0:S_DIM]
	pilco.optimize_controller(states)

	X_new, Y_new = rollout(policy=pilco_policy, timesteps=100)
	# update dataset
	X = np.vstack((X, X_new))
	Y = np.vstack((Y, Y_new))
	pilco.mgpr.set_XY(X, Y)
