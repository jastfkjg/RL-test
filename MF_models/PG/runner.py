import numpy as np
import ipdb
from abc import ABC, abstractmethod

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1.-done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        # self.nenv = env.num_envs if hasattr(env, 'num_envs') else 1  # num_envs ?
        # ex: (2,) + (4,) = (2, 4)
        self.batch_ob_shape = (nsteps,) + env.observation_space.shape
        # self.obs = np.zeros((1,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs = np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        # self.states = model.initial_state 
        self.dones = False

    @abstractmethod
    def run(self):
        raise NotImplementedError

class Runner(AbstractEnvRunner):
    """
    generate batches of experiences
    """

    def __init__(self, env, model, nsteps=5, gamma=0.99):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        self.batch_action_shape = [x if x is not None else -1 for x in model.action.shape.as_list()]
        self.ob_dtype = model.tf_obs.dtype.as_numpy_dtype

    def run(self):
        # we initialize the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_dones = [], [], [], []
        # mb_states = self.states
        for n in range(self.nsteps):
            # given obs, take action and value
            # we already have self.obs because Runner superclass: 
            # self.obs[:] = env.reset() 

            # choose action using actor model
            # our actor model step method return actions
            # actions, values, _ = self.model.step(self.obs)

            # ipdb.set_trace()

            # print("obs:", self.obs)
            action = self.model.step(self.obs)
            # print("action: ", action)

            # S=self.states, M=self.dones
            # append the experiences
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(action)
            # mb_values.append(values)
            mb_dones.append(self.dones)

            # take actions in env and look the results
            obs, rewards, dones, _ = self.env.step(action)
            # update obs, states, dones
            # self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=self.model.action.dtype.name)
        # mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        # mb_masks = mb_dones[:, :, -1]
        # mb_dones = mb_dones[:, 1:]

        if self.gamma > 0.0:
            # TODO: rewrite this for PG
            # action, last_values, _ = self.model.step(self.obs).tolist()
            mb_rewards = discount_with_dones(mb_rewards, mb_dones, self.gamma)
            print("mb_reward: ", mb_rewards)

            # reward = discount_with_dones(reward, done, self.gamma)

        mb_actions = mb_actions.reshape(self.batch_action_shape)
        print("mb_action: ", mb_actions)
        print("mb_obs: ", mb_obs)

        # mb_rewards = mb_rewards.flatten()
        # mb_values = mb_values.flatten()
        # mb_mask = mb_masks.flatten()
        return mb_obs, mb_rewards, mb_actions



