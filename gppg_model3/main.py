import numpy as np 
import os
import random
import gym
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from gym.spaces import Discrete, Box
import argparse
# import ipdb
from tensorflow.python import debug as tf_debug

from utils import save_pilco, load_pilco, pilco_policy, Runner, get_env_reward, getLogger 
# from actor_s_obs import Actor 
# from pilco import PILCO
from actor import Actor
from pilco_m_obs import PILCO
from test_actor import evaluate_policy, compare_policy

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
# disable tensorflow info and warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("env_name", help="gym env name: classic control (CartPole-v1, MountainCarContinuous-v0, Pendulum-v0 ... )")
parser.add_argument("--load_actor_model", dest="load_actor_model", action='store_true', help="load actor model")
parser.add_argument("--load_gp_model", dest="load_gp_model", action='store_true', help="load gp model")
parser.add_argument("--debug", dest="debug", action='store_true', help="Use debugger to track down bad values")
parser.set_defaults(load_actor_model=False, load_gp_model=False, debug=False)
args = parser.parse_args()

# mujoco:
# InvertedPendulum-v2, Swimmer-v2
# Reacher-v2, Walker2d-v2, Humanoid-v2
# Ant-v2, InvertedDoublePendulum-v2

params = {"model_path": './checkpoints/' + args.env_name + '/',
        "env_name": args.env_name,
        "render": False,
        "learning_rate": 5e-4,
        "reward_decay": 0.99,
        "max_episode": 50,
        "num_optim": None,
        "num_collect": 10,
        "horizon": 30,
        "seed": 123,
        "pretrain": False,
        }

# set seed
random.seed(params["seed"])
np.random.seed(params["seed"])
tf.set_random_seed(params["seed"])

env = gym.make(args.env_name)
# model_path = './checkpoints/' + args.env_name + '/'

runner = Runner(env=env, timesteps=50)

if isinstance(env.observation_space, Discrete):
    state_dim = env.observation_space.n
elif isinstance(env.observation_space, Box):
    state_dim = env.observation_space.shape[0]

if isinstance(env.action_space, Discrete):
    action_choice = env.action_space.n
    action_dim = 1
elif isinstance(env.action_space, Box):
    action_dim = env.action_space.shape[0]

logger = getLogger('output.log')
logger.info(params)
controller = Actor(env=env, action_dim=action_dim, state_dim=state_dim, \
        learning_rate=params["learning_rate"], debug=args.debug)
def discount_reward(rewards, gamma):
    discounted = []
    r = 0.
    for reward in rewards[::-1]:
        r = reward + gamma * r
        discounted.append(r)
    return discounted[::-1]

def pretrain(controller, env, episodes=100, max_steps=200, s_obs=False):

    for episode in range(episodes):
        obs = env.reset()
        ep_obs, ep_ac, ep_r = [], [], []
        ep_s_obs = []
        for step in range(max_steps):
            ac = controller.take_quick_action(obs)
            n_obs, r, done, info = env.step(ac)
            ep_obs.append(obs)
            ep_ac.append(ac)
            ep_r.append(r)
            ep_s_obs.append(np.diag(np.ones(state_dim)) * 0.1)
            obs = n_obs
            if done or step == max_steps-1:
                ep_r = discount_reward(ep_r, 0.99)
                logger.info("--Pretrain:  episode %i with total reward %f" %(episode, ep_r[0]))
                break
        if s_obs:
            controller.store_transition(ep_obs, ep_s_obs, ep_r, ep_ac)
        else:
            controller.store_transition(ep_obs, ep_r, ep_ac)
        controller.optimize()

    return controller

if params["pretrain"]:
    controller = pretrain(controller, env, episodes=10, s_obs=False)

# # load actor
if args.load_actor_model:
    try:
        controller.load_weights(params["model_path"] + 'actor.ckpt')
        logger.info("load actor model successfully")
    except:
        logger.info("can not find checkpoint.")
    X, Y = runner.run(policy=pilco_policy, controller=controller, timesteps=100)
else:
    logger.info("Get real data from random policy")
    X, Y = runner.run()
    for i in range(1, 5):
        X_, Y_ = runner.run()
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))

# get env reward
env_reward = get_env_reward(args.env_name)

# load gp
if args.load_gp_model:
    try:
        pilco = load_pilco(model_path, controller, env_reward, debug=args.debug)
        logger.info("load gp model successfully")
    except:
        raise ValueError("can not find saved gp models")
        # pilco = PILCO(X, Y, controller=controller, reward=env_reward)
else:
    pilco = PILCO(X, Y, controller=controller, reward=env_reward, debug=args.debug)

# for numerical stability
# for model in pilco.mgpr.models:
    # model.likelihood.variance = 0.001
    # model.likelihood.variance.trainable = False

X_init = X[:, 0: state_dim]

# def random_policy(obs):
    # return env.action_space.sample()
# random policy as comparing policy
# policy2 = random_policy

m, s = evaluate_policy(env, pilco.controller.take_quick_action, test_episode=20)
logger.info("--Initial: \t average reward: %f \t variance of reward: %f" %(m, s))

ep_m_reward, ep_s_reward = [], []

for rollouts in range(params["max_episode"]):
    logger.info("***" * 30)
    logger.info("the %d th rollout begins." %rollouts)
    logger.info("***" * 30)
    # optimize GP
    pilco.optimize_gp()

    # the controller optimization
    # pilco.optimize_controller(X_init, params["horizon"], num_optim=params["num_optim"], \
            # num_collect=params["num_collect"], gamma=params["reward_decay"])
    pilco.optimize_controller_without_state(env, params["horizon"], total_epoch=10,
            num_collect=params["num_collect"], gamma=params["reward_decay"])

    # pilco.controller.save_weights(model_path + 'actor.ckpt')

    X_new, Y_new = runner.run(policy=pilco_policy, controller=pilco.controller, timesteps=100)
    # update dataset, why update instead of replace
    # with more and more data, we are going to replace dataset, discard previous data
    X = np.vstack((X, X_new))
    Y = np.vstack((Y, Y_new))
    # X, Y = X_new, Y_new
    pilco.mgpr.set_XY(X, Y)
    X_init = np.array(X_new)[:, 0: state_dim]

    m, s = evaluate_policy(env, pilco.controller.take_quick_action, test_episode=20)
    logger.info("%i episodes: \t average reward: %f \t variance of reward: %f" %(rollouts, m, s))

    ep_m_reward.append(m)
    ep_s_reward.append(s)

# save
pilco.controller.save_weights(params["model_path"] + 'actor.ckpt')
save_pilco(params["model_path"], X, Y, pilco)

logger.info("Finished !")

fig = plt.figure()
plt.errorbar(np.arange(1, len(ep_m_reward)+1), ep_m_reward, yerr=ep_s_reward, fmt="o")
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
fig.savefig(params["model_path"] + 'actor_ep_reward.png')
logger.info("figure store in: "+ params["model_path"] + 'actor_ep_reward.png')



