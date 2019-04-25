import numpy as np 
import gym
import pandas as pd
import matplotlib.pyplot as plt 
from gym.spaces import Discrete, Box
import argparse
import ipdb

from actor import LinearActor, Actor 
from utils import *
from pilco import PILCO
from rewards import *
from test_actor import evaluate_policy

parser = argparse.ArgumentParser()
parser.add_argument("env_name", help="gym env name: classic control (CartPole-v1, MountainCarContinuous-v0, Pendulum-v0 ... )")
parser.add_argument("--load_actor_model", action='store_true', help="whether to load actor model")
parser.add_argument("--load_gp_model", action='store_true', help="whether to load gp model")
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

runner = Runner(env=env, timesteps=50)
X, Y = runner.run()
for i in range(1, 3):
    X_, Y_ = runner.run()
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))

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
max_episode = 20
# num_optim: how many real states to use as init state, default: None(use all real states)
num_optim = 30
# num_collect: how many fake data are we going to create in a batch (batch num) to optimize actor
num_collect = 10
# optim horizon: the horizon to calculate expected reward
optim_horizon = 20
# max_episode_step = 3000

controller = Actor(action_dim=action_dim, state_dim=state_dim, learning_rate=learning_rate, discrete_ac=discrete_ac)
# # load actor
if args.load_actor_model:
    try:
        controller.load_weights(model_path + 'actor.ckpt')
        print("load actor model successfully")
    except:
        print("can not find checkpoint.")

# get env reward
env_reward = get_env_reward(args.env_name)

# load gp
if args.load_gp_model:
    try:
        pilco = load_pilco(model_path, controller, env_reward)
        print("load gp model successfully")
    except:
        print("can not find saved gp models")
        pilco = PILCO(X, Y, controller=controller, reward=env_reward)
else:
    pilco = PILCO(X, Y, controller=controller, reward=env_reward)

# for numerical stability
for model in pilco.mgpr.models:
    model.likelihood.variance = 0.001
    model.likelihood.variance.trainable = False

print("num optim already: ", pilco.controller.get_num_optim())

# reward_list = []
# ep_step_list = []
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
    # ipdb.set_trace()
    pilco.optimize_controller(X_init, optim_horizon, num_optim=num_optim, num_collect=num_collect, gamma=reward_decay)

    # save controller's weights
    print("saving the controller.")
    pilco.controller.save_weights(model_path + 'actor.ckpt')

    X_new, Y_new = runner.run(policy=pilco_policy, pilco=pilco, timesteps=100)
    # update dataset, why update instead of replace
    # with more and more data, we are going to replace dataset, discard previous data
    X = np.vstack((X, X_new))
    Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_XY(X, Y)
    X_init = np.array(X_new)[:, 0: state_dim]


# save controller's weights
pilco.controller.save_weights(model_path + 'actor.ckpt')
# save gp model and data
save_pilco(model_path, X, Y, pilco)

# print("reward list: ", reward_list)

print("Finished !")

policy1 = pilco.controller.take_quick_action
def random_policy(obs):
    return env.action_space.sample()
policy2 = random_policy

evaluate_policy(env, policy1, policy2)

# plt.subplot(2, 2, 1)
# plt.plot(np.arange(1, len(reward_list)+1), reward_list)
# plt.xlabel('Episode')
# plt.ylabel('Episode Reward')
# plt.show()

# plt.subplot(2, 2, 2)
# plt.plot(ep_step_list, reward_list)
# plt.xlabel('Total steps')
# plt.ylabel('Episode Reward')
# plt.show()

# test_episode = 10
# max_step = 200
# render = False
# test_reward_list = []

# for i_episode in range(test_episode):
    # observation = env.reset()
    # step = 0
    # total_reward = 0.0

    # while True:
        # step += 1
        # if render:
            # env.render()

        # action = pilco_policy(observation)
        # observation_next, reward, done, info = env.step(action)
        # total_reward += reward

        # if done or step >= max_step:
            # test_reward_list.append(total_reward)

            # print("Episode: ", i_episode, " step in this episode:", step, " reward: ", total_reward)
            # break

        # update observation
        # observation = observation_next

# plt.subplot(2, 2, 3)
# plt.plot(np.arange(test_episode), test_reward_list)
# plt.xlabel('Test Episode')
# plt.ylabel('Test Episode Reward')
# plt.show()

# dataFrame = pd.DataFrame({'reward': reward_list})
# dataFrame.to_csv("./pilco_output/episode_reward.csv", index=True, sep=',')
#
# dataFrame2 = pd.DataFrame({'total_step': ep_step_list, 'reward': reward_list})
# dataFrame2.to_csv("./pilco_output/step_reward.csv", index=False, sep=',')


