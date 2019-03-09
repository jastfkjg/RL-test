import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import pandas as pd
from actor import Actor, LinearActor
from PG_PILCO import PolicyGradient
import matplotlib.pyplot as plt
from gym.spaces import Discrete, Box


np.random.seed(0)
# more info about InvertedPendulum-v2: ex: spaces type
# how does PILCO handle discrete action space task, such as CartPole ?
env = gym.make('InvertedPendulum-v2')  #CartPole-v0  InvertedPendulum-v2

def rollout(policy, timesteps):  #timesteps: horizon
    X = []; Y = []
    env.reset()
    x, _, _, _ = env.step(0)
    for timestep in range(timesteps):
        env.render()
        u = policy(x)
        x_new, _, done, _ = env.step(u)
        if done: break
        X.append(np.hstack((x, u)))   # state X and action u
        Y.append(x_new - x)           # delta X
        x = x_new
    return np.stack(X), np.stack(Y)   # 默认 axis=0

def random_policy(x):
    return env.action_space.sample()

def pilco_policy(x):
    return pilco.compute_action(x[None, :])[0, :]

# Initial random rollouts to generate a dataset
X,Y = rollout(policy=random_policy, timesteps=40)
for i in range(1,3):
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

# hyper-params, to change
render = False
learning_rate = 0.02
reward_decay = 0.995
max_episode = 1000
max_episode_step = 3000

state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim      # X.shape[1]: dim of state + dim of action, control_dim: action dim
# use the PG controller
controller = Actor(action_dim=control_dim, state_dim=state_dim, learning_rate=learning_rate)
linear_controller = LinearActor(action_dim=control_dim, state_dim=state_dim, learning_rate=learning_rate)
#controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5)
#controller = LinearController(state_dim=state_dim, control_dim=control_dim)

pilco = PILCO(X, Y, controller=linear_controller, horizon=40)
# Example of user provided reward function, setting a custom target state
# R = ExponentialReward(state_dim=state_dim, t=np.array([0.1,0,0,0]))
# pilco = PILCO(X, Y, controller=controller, horizon=40, reward=R)

# Example of fixing a parameter, optional, for a linear controller only
#pilco.controller.b = np.array([[0.0]])
#pilco.controller.b.trainable = False

for rollouts in range(3):
    # add the policy optim
    pilco.optimize()
    # add the controller optim
    linear_controller.optimize()  ######## to do !
    #import pdb; pdb.set_trace()
    X_new, Y_new = rollout(policy=pilco_policy, timesteps=100)
    # Update dataset
    X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_XY(X, Y)
