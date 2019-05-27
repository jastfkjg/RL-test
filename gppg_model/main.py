import numpy as np 
import os
import gym
import pandas as pd
import matplotlib.pyplot as plt 
from gym.spaces import Discrete, Box
import argparse
import ipdb
from tensorflow.python import debug as tf_debug

from utils import save_pilco, load_pilco, pilco_policy, Runner, get_env_reward 
from actor import Actor 
from pilco import PILCO
from test_actor import evaluate_policy, compare_policy

parser = argparse.ArgumentParser()
parser.add_argument("env_name", help="gym env name: classic control (CartPole-v1, MountainCarContinuous-v0, Pendulum-v0 ... )")
parser.add_argument("--load_actor_model", dest="load_actor_model", action='store_true', help="load actor model")
parser.add_argument("--load_gp_model", dest="load_gp_model", action='store_true', help="load gp model")
parser.add_argument("--debug", dest="debug", action='store_true', help="Use debugger to track down bad values")
parser.set_defaults(load_actor_model=False, load_gp_model=False, debug=False)
args = parser.parse_args()

# np.random.seed(0)
# classic control:
# env = gym.make('CartPole-v1')
# MountainCarContinuous-v0, Pendulum-v0

# mujoco:
# InvertedPendulum-v2, Swimmer-v2
# Reacher-v2, Walker2d-v2, Humanoid-v2
# Ant-v2, InvertedDoublePendulum-v2

# disable tensorflow info and warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

env = gym.make(args.env_name)
model_path = './checkpoints/' + args.env_name + '/'

runner = Runner(env=env, timesteps=50)

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

# hyperparams
RENDER = False
learning_rate = 0.01
reward_decay = 0.995
max_episode = 100
# num_optim: how many real states to use as init state, default: None(use all real states)
num_optim = None
# num_collect: how many fake data are we going to create in a batch (batch num) to optimize actor
num_collect = 5
# optim horizon: the horizon to calculate expected reward
optim_horizon = 20
# max_episode_step = 3000
# --- total_optim_times = max_episode * num_optim * num_collect

controller = Actor(action_dim=action_dim, state_dim=state_dim, learning_rate=learning_rate, debug=args.debug)
# # load actor
if args.load_actor_model:
    try:
        controller.load_weights(model_path + 'actor.ckpt')
        print("load actor model successfully")
    except:
        print("can not find checkpoint.")
    X, Y = runner.run(policy=pilco_policy, controller=controller, timesteps=100)
else:
    print("Get real data from random policy")
    X, Y = runner.run()
    for i in range(1, 3):
        X_, Y_ = runner.run()
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))

# get env reward
env_reward = get_env_reward(args.env_name)

# load gp
if args.load_gp_model:
    try:
        pilco = load_pilco(model_path, controller, env_reward, debug=args.debug)
        print("load gp model successfully")
    except:
        raise ValueError("can not find saved gp models")
        # pilco = PILCO(X, Y, controller=controller, reward=env_reward)
else:
    pilco = PILCO(X, Y, controller=controller, reward=env_reward, debug=args.debug)

# for numerical stability
for model in pilco.mgpr.models:
    model.likelihood.variance = 0.001
    model.likelihood.variance.trainable = False

print("num optim already: ", pilco.controller.get_num_optim())

# reward_list = []
# ep_step_list = []
X_init = X[:, 0: state_dim]

def random_policy(obs):
    return env.action_space.sample()
# random policy as comparing policy
policy2 = random_policy

ep_m_reward, ep_s_reward = [], []

for rollouts in range(max_episode):
    print("***" * 30)
    print("the " + str(rollouts) + "th rollout begins.")
    print("num optim: ", pilco.controller.get_num_optim())
    print("***" * 30)
    # optimize GP
    pilco.optimize_gp()

    # the controller optimization
    # ipdb.set_trace()
    pilco.optimize_controller(X_init, optim_horizon, num_optim=num_optim, num_collect=num_collect, gamma=reward_decay)

    # save controller's weights
    print("saving the controller.")
    pilco.controller.save_weights(model_path + 'actor.ckpt')

    X_new, Y_new = runner.run(policy=pilco_policy, controller=pilco.controller, timesteps=100)
    # update dataset, why update instead of replace
    # with more and more data, we are going to replace dataset, discard previous data
    X = np.vstack((X, X_new))
    Y = np.vstack((Y, Y_new))
    # X, Y = X_new, Y_new
    pilco.mgpr.set_XY(X, Y)
    X_init = np.array(X_new)[:, 0: state_dim]

    m, s = evaluate_policy(env, pilco.controller.take_quick_action)
    print("average reward for controller in " + str(rollouts) + "episodes: " + str(m))
    print("variance of reward for controller in " + str(rollouts) + "episodes: " + str(s))

    ep_m_reward.append(m)
    ep_s_reward.append(s)

# save controller's weights
pilco.controller.save_weights(model_path + 'actor.ckpt')
# save gp model and data
save_pilco(model_path, X, Y, pilco)

# print("reward list: ", reward_list)

print("Finished !")

policy1 = pilco.controller.take_quick_action

compare_policy(env, policy1, policy2)

fig = plt.figure()
plt.errorbar(np.arange(1, len(ep_m_reward)+1), ep_m_reward, yerr=ep_s_reward, fmt="o")
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
fig.savefig(model_path + 'actor_ep_reward.png')
print("figure store in: "+ model_path + 'actor_ep_reward.png')


# dataFrame = pd.DataFrame({'reward': reward_list})
# dataFrame.to_csv("./pilco_output/episode_reward.csv", index=True, sep=',')
#
# dataFrame2 = pd.DataFrame({'total_step': ep_step_list, 'reward': reward_list})
# dataFrame2.to_csv("./pilco_output/step_reward.csv", index=False, sep=',')


