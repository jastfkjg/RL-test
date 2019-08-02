import numpy as np
import gym
import mujoco_py
import matplotlib.pyplot as plt
from gym.spaces import Discrete, Box
import argparse
# from actor import Actor

def evaluate_policy(env, policy, test_episode=50, max_step=500, render=False, plot=False):
    reward_list = []

    for i_episode in range(test_episode):
        observation = env.reset()
        step = 0
        total_reward = 0.0

        while True:
            step += 1
            if render:
                env.render()

            action = policy(observation)
            observation_next, reward, done, info = env.step(action)
            total_reward += reward

            if done or step >= max_step:
                reward_list.append(total_reward)
                break
            # update observation
            observation = observation_next

    if plot:
        plt.plot(np.arange(test_episode), reward_list)
        plt.xlabel('Episode')
        plt.ylabel('Episode reward')
        plt.show()

    m, s = np.mean(reward_list), np.var(reward_list)
    
    return m, s


def compare_policy(env, policy1, policy2, test_episode=100, max_step=200, render=False, plot=False):

    reward_list1 = []

    for i_episode in range(test_episode):
        observation = env.reset()
        step = 0
        total_reward = 0.0

        while True:
            step += 1
            if render:
                env.render()

            action = policy1(observation)
            observation_next, reward, done, info = env.step(action)
            total_reward += reward

            if done or step >= max_step:
                reward_list1.append(total_reward)

                print("Episode: ", i_episode, " step in this episode:", step, " reward: ", total_reward)
                break

            # update observation
            observation = observation_next

    reward_list2 = []
    for i_episode in range(test_episode):
        observation = env.reset()
        step = 0
        total_reward = 0.0

        while True:
            step += 1
            if render:
                env.render()

            action = policy2(observation)
            observation_next, reward, done, info = env.step(action)
            total_reward += reward

            if done or step >= max_step:
                reward_list2.append(total_reward)

                print("Episode: ", i_episode, " step in this episode:", step, " reward: ", total_reward)
                break

            # update observation
            observation = observation_next

    if plot:

        plt.subplot(2, 1, 1)
        plt.plot(np.arange(test_episode), reward_list1)
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward for policy 1')
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(test_episode), reward_list2)
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward for policy 2')
        plt.show()

    m1, s1, m2, s2 = np.mean(reward_list1), np.var(reward_list1), np.mean(reward_list2), np.var(reward_list2)
    print("average reward for first policy: ", m1)
    print("variance of reward for first policy: ", s1)
    print("average reward for second policy: ", m2)
    print("variance of reward for second policy: ", s2)

    return m1, s1, m2, s2


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", help="gym env name: classic control (CartPole-v1, MountainCarContinuous-v0, Pendulum-v0 ... )")
    parser.add_argument("--plot", dest="plot", action="store_true", help="plot two policy performance.")
    parser.add_argument("--test_episode", dest="test_episode", type=int, help="how many test episodes")
    parser.set_defaults(plot=False, test_episode=100)

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

    # controller = Actor(action_dim=action_dim, state_dim=state_dim, learning_rate=learning_rate, discrete_ac=discrete_ac)
    # controller.load_weights(model_path + 'actor.ckpt')

    # policy1 = controller.take_quick_action
    
    # def random_policy(obs):
        # return env.action_space.sample()
    # policy2 = random_policy
    

    m1, s1, m2, s2 = compare_policy(env, policy1, policy2, test_episode=args.test_episode, plot=args.plot)





