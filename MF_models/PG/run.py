import gym
import mujoco_py
import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from gym.spaces import Discrete, Box

from PG_ac_box import PolicyGradient, learn

params = {
        "nsteps": 500,
        "nepisodes": 200,
        "ent_coef": 0.01,
        "learning_rate": 0.001,
        "gamma": 0.99,
        }

def train():

    # env = gym.make("Pendulum-v0")
    env = gym.make("InvertedPendulum-v2")

    model = learn(env=env, load_path="../checkpoints/pg_actor.ckpt")
    return model, env

def main():
    model, env = train()
    model.save_model("../checkpoints/pg_actor.ckpt")

    obs = env.reset()

    max_episode, max_step = 50, 500
    total_reward = []
    for i in range(max_episode):
        episode_rewards = 0
        obs = env.reset()
        print("%d th test episode begin" %i)
        for j in range(max_step):

            actions = model.step(obs)

            obs, reward, done, _ = env.step(actions)
            episode_rewards += reward
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done
            if done or j == max_step - 1:
                print('episode_rew={}'.format(episode_rewards))
                total_reward.append(episode_rewards)
                break

    env.close()

    return model


if __name__ == '__main__':
    main()


