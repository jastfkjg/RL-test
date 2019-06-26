import numpy as np
import ipdb
from abc import ABC, abstractmethod

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0.
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1.-done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

def discounted_cumulative_r(mb_rewards):
    discounted = []
    r = 0.
    for reward in mb_rewards[::-1]:
        r = reward + gamma * r 
        discounted.append(r)
    return discounted[::-1]

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        # ex: (2,) + (4,) = (2, 4)
        # self.batch_ob_shape = (nsteps,) + env.observation_space.shape
        # self.obs = np.zeros((1,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs = np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps

    @abstractmethod
    def run(self):
        raise NotImplementedError

class Runner(AbstractEnvRunner):
    """
    generate batches of experiences
    """

    def __init__(self, env, model, nsteps=500, gamma=0.99):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        # self.batch_action_shape = [x if x is not None else -1 for x in model.action.shape.as_list()]
        self.ob_dtype = model.tf_obs.dtype.as_numpy_dtype

    def run(self):
        # we initialize the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions = [], [], []
        cumulative_reward = 0.
        for n in range(self.nsteps):
            # we already have self.obs because Runner superclass: 
            # self.obs[:] = env.reset() 
            action = self.model.step(self.obs)

            # append the experiences
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(action)

            # take actions in env and look the results
            obs, reward, done, _ = self.env.step(action)

            self.obs = obs
            mb_rewards.append(reward)

            cumulative_reward += reward
            if done or n == self.nsteps - 1:
                print("reward in this episode: ", cumulative_reward)
                break
        
        mb_rewards = discounted_cumulative_r(mb_rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=self.model.action.dtype.name)
        # mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)

        print("mb_action: ", mb_actions)
        print("mb_obs: ", mb_obs)

        return mb_obs, mb_rewards, mb_actions



