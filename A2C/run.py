import numpy as np
import tensorflow as tf
import sys
import gym
from collections import defaultdict

def train(args):
    env_type, env_id = get_env_type(args)

    total_timesteps = int(args.num_timesteps)

    learn = get_learn_function(args.alg)
    
    env = build_env(args)
    
    model = learn(env=env, total_timesteps=total_timesteps)
    return model, env

def build_env(args):
    env = make_env(env_id, env_type, seed=seed)

def main(args):
    model, env = train(args)
    model.save(save_path)

    obs = env.reset()

    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    episode_rew = 0
    while True:
        is state is not None:
            actions, _, state, _ = model.step(obs, S=state, M=dones)
        else:
            actions, _, _, _ = model.step(obs)

        obs, rew, done, _ = env.step(actions)
        episode_rew += rew[0] if isinstance(env, VecEnv) else rew
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done
        if done:
            print('episode_rew={}'.format(episode_rew))
            episode_rew = 0
            obs = env.reset()
    env.close()

    return model

if __name__ == '__main__':
    main(sys.argv)
    


