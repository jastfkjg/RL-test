import numpy as np
from gpflow import autoflow
from gpflow import settings

from rewards import *
from pilco import PILCO

float_type = settings.dtypes.float_type

def save_pilco(path, X, Y, pilco, sparse=False):
    np.savetxt(path + 'X.csv', X, delimiter=',')
    np.savetxt(path + 'Y.csv', Y, delimiter=',')
    if sparse:
        with open(path+ 'n_ind.txt', 'w') as f:
            f.write('%d' % pilco.mgpr.num_induced_points)
            f.close()
    # np.save(path + 'pilco_values.npy', pilco.read_values())
    for i, m in enumerate(pilco.mgpr.models):
        np.save(path + "model_" + str(i) + ".npy", m.read_values())

def load_pilco(path, controller=None, reward=None, sparse=False):
    X = np.loadtxt(path + 'X.csv', delimiter=',')
    Y = np.loadtxt(path + 'Y.csv', delimiter=',')
    if not sparse:
        pilco = PILCO(X, Y, controller=controller, reward=reward)
    else:
        with open(path + 'n_ind.txt', 'r') as f:
            n_ind = int(f.readline())
            f.close()
        pilco = PILCO(X, Y, num_induced_points=n_ind)
    # params = np.load(path + "pilco_values.npy").item()
    # pilco.assign(params)
    for i, m in enumerate(pilco.mgpr.models):
        values = np.load(path + "model_" + str(i) + ".npy").item()
        m.assign(values)
    return pilco


def random_policy(env, pilco, x):
    return env.action_space.sample()

def pilco_policy(env, pilco, x):
    return pilco.controller.take_quick_action(x)

class Runner:
    def __init__(self, env, timesteps=40):
        self.env = env
        self.timesteps = timesteps
        

    def run(self, *,  timesteps=None, policy=random_policy, pilco=None, render=False, verbose=False):
        """
        get training data for GP to model the transition function
        """
        X, Y = [], []
        total_step, total_reward = 0, 0.
        x = self.env.reset()
        if not timesteps:
            timesteps = self.timesteps
        
        for timestep in range(timesteps):
            if render: self.env.render()
            u = policy(self.env, pilco, x)
            x_new, _, done, _ = self.env.step(u)
            
            if verbose:
                print("Action: ", u)
                print("State :", x_new)
            X.append(np.hstack((x, u)))
            Y.append(x_new - x)
            x = x_new
            if done:
                break

        return np.stack(X), np.stack(Y)


def get_env_reward(env_name):
    # TODO
    if env_name == 'CartPole-v1':    
        env_reward = CartPoleReward()
    elif env_name == 'Pendulum-v0':
        env_reward = PendulumReward()
    elif env_name == 'InvertedPendulum-v2':
        env_reward = InvertedPendulumReward() 
    else:
        raise NameError("dont have this env yet")
    return env_reward

