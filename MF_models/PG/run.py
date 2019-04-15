import gym
import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from gym.spaces import Discrete, Box

from PG_ac_box import PolicyGradient, learn

# hyperparams
timesteps = 50000

def train():

    env = gym.make("Pendulum-v0")
    # env.reset()

    model = learn(env=env, timesteps=timesteps, load_path="../checkpoints/pg_actor.ckpt")
    return model, env

def main():
    model, env = train()
    model.save_model("../checkpoints/pg_actor.ckpt")

    obs = env.reset()

    # state = model.initial_state if hasattr(model, 'initial_state') else None
    # dones = np.zeros((1,))

    episode_rewards = 0
    while True:

        actions = model.step(obs)

        obs, reward, done, _ = env.step(actions)
        episode_rewards += reward
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done
        if done:
            print('episode_rew={}'.format(episode_rewards))
            episode_rewards = 0
            obs = env.reset()
    env.close()

    return model


if __name__ == '__main__':
    main()


