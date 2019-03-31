import numpy as np

from abc import ABC, abstractmethod

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        # ex: (2,) + (4,) = (2, 4)
        self.batch_ob_shape = (nenv * nsteps,) + env.bservation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape,
                dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

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
        self.batch_action_shape = [x if x is not None else -1 for x in
                model.train_model.action.shape.as_list()]
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype

    def run(self):
        # we initialize the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [],
        []
        mb_states = self.states
        for n in range(self.nsteps):
            # given obs, take action and value
            # we already have self.obs because Runner superclass: self.obs[:] =
            # env.reset() 
            actions, values, states, _ = self.model.step(self.obs,
                    S=self.states, M=self.dones)
            # append the experiences
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            # take actions in env and look the results
            obs, rewards, dones, _ = self.env.step(actions)
            # update obs, states, dones
            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs dtype=self.ob_dtype).swapaxes(1,
                0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions,
                dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :, -1]
        mb_dones = mb_dones[:, 1:]

        if self.gamma > 0.0:
            last_values = self.model.value(self.obs, S=self.states,
                    M=self.dones).tolist()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards,
                mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0],
                            self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[n] = rewards

        mb_actions = mb_actions.reshape(self.batch_action_shape)

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_mask = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values



