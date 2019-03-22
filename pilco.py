import numpy as np
import tensorflow as tf
import gpflow
import pandas as pd
import time

# from actor import Actor, LinearActor
from mgpr import MGPR
from smgpr import SMGPR
import rewards

# float64
float_type = gpflow.settings.dtypes.float_type


class PILCO:
    def __init__(self, X, Y, num_induced_points=None, controller=None,
                reward=None, m_init=None, S_init=None, name=None):
        # super(PILCO, self).__init__(name)
        if not num_induced_points:      # num_induced_points ?
            self.mgpr = MGPR(X, Y)
        else:
            self.mgpr = SMGPR(X, Y, num_induced_points)
        self.state_dim = Y.shape[1]
        self.control_dim = X.shape[1] - Y.shape[1]

        self.sess = gpflow.get_default_session()
        # self.sess.run(tf.global_variables_initializer())

        if controller is None:   # the policy  - to change
            print("controller cannot be None")
        else:
            self.controller = controller

        if reward is None:     # reward function
            self.reward = rewards.Reward()
        else:
            self.reward = reward
        
        if m_init is None or S_init is None:
            # If the user has not provided an initial state for the rollouts,
            # then define it as the first state in the dataset.
            self.m_init = X[0:1, 0:self.state_dim]
            self.S_init = np.diag(np.ones(self.state_dim) * 0.1)  # variance
        else:
            self.m_init = m_init
            self.S_init = S_init

    # @gpflow.name_scope('likelihood')   # to optimize the controller
    # def _build_likelihood(self):
    #     # This is for tuning controller's parameters
    #     reward = self.predict(self.m_init, self.S_init, self.horizon)[2]
    #     return reward      # the return represents the likelihood

    def compute_reward(self, m_x, s_x, horizon):
        # to compute cumulative reward
        reward = self.predict(m_x, s_x, horizon)[2]
        return reward

    def optimize_controller(self, states, horizon, num_optim=8, gamma=1.):
        """
        optimize controller's parameters
        :param: states: a array of init states
        """
        # how to choose init states:
        # 1. random sample from the state_space
        # 2. random sample from the dataset
        start = time.time()
        # s_x = np.random.rand(self.state_dim, self.state_dim) * 0.1
        s_x = np.diag(np.ones(self.state_dim) * 0.1)
        for i, x in enumerate(states[:num_optim]):
            print("Evaluate the " + str(i) + "th init state, total init states: " + str(len(states[:num_optim])))
            # self.predict(state, np.diag(np.ones(self.state_dim) * 0.1), horizon)
            m_x = np.expand_dims(x, 0)
            self.get_data(m_x, s_x, horizon, gamma=gamma)
            self.controller.optimize()
        end = time.time()
        print("Finished with " + str(len(states[:num_optim])) + " times policy/controller optimizations in %.1f seconds" % (end - start))

    def optimize_gp(self):
        """
        Optimizes  GP's hyperparamemeters.
        """
        start = time.time()
        self.mgpr.optimize()      # GP optim
        end = time.time()
        print("Finished with GPs' optimization in %.1f seconds" % (end - start))
        # we do not optimize the controller using pilco here
        # start = time.time()
        # optimizer = gpflow.train.ScipyOptimizer(options={'maxfun': 500})
        # optimizer.minimize(self, disp=True)     # optimize the controller   disp=True: to print convergence messages.
        # end = time.time()
        # print("Finished with Controller's optimization in5%.1f seconds" % (end - start))

        lengthscales = {}; variances = {}; noises = {};
        i = 0
        for model in self.mgpr.models:
            lengthscales['GP' + str(i)] = model.kern.lengthscales.value
            variances['GP' + str(i)] = np.array([model.kern.variance.value])
            noises['GP' + str(i)] = np.array([model.likelihood.variance.value])
            i += 1
        print('-----Learned models------')
        pd.set_option('precision', 3)
        print('---Lengthscales---')
        print(pd.DataFrame(data=lengthscales))
        print('---Variances---')
        print(pd.DataFrame(data=variances))
        print('---Noises---')
        print(pd.DataFrame(data=noises))

    # @gpflow.autoflow((float_type,[None, None]))
    # def compute_action(self, x_m):
    #     return self.controller.compute_action(x_m, tf.zeros([self.state_dim, self.state_dim], float_type))[0]
    def compute_action(self, x_m):
        """
        use for sampling, how to sample
        x_m: mean of state
        the first return from compute_action is the mean of action
        each time the variance of the state is zero, which is not the truth ?
        """
        return self.controller.compute_action(x_m, np.zeros([self.state_dim, self.state_dim], float_type))[0]

    def take_action(self, x_m):
        # to handle discrete action space
        return self.controller.take_action(x_m, np.zeros([self.state_dim, self.state_dim], float_type))

    def predict(self, m_x, s_x, horizon, gamma):  # n: horizon  what if the game is done before n horizon
        # loop_vars = [
        #     tf.constant(0, tf.int32),
        #     m_x,
        #     s_x,
        #     tf.constant([[0]], float_type)
        # ]                                     # initial condition
        #
        # _, m_x, s_x, reward = tf.while_loop(  # tf.while_loop
        #     # Termination condition
        #     lambda j, m_x, s_x, reward: j < n,
        #     # Body function
        #     lambda j, m_x, s_x, reward: (
        #         j + 1,
        #         *self.propagate(m_x, s_x),
        #         tf.add(reward, self.reward.compute_gaussian_reward(m_x, s_x))     # total reward  # reward need a compute_reward() func
        #     ), loop_vars
        # )           # we need to store transiton(m_x, s_x, reward) for every timestep here, perphas rewrite the loop ?
        ##################################################
        # consider to propagate more than n steps to gather more data
        reward = 0.0
        m_x_init, s_x_init = m_x, s_x
        for i in range(horizon):
            current_reward, done = self.reward.compute_gaussian_reward(m_x, s_x)
            # check if the game is done
            if done:
                reward = reward + pow(gamma, i) * current_reward
                break
            else:
                reward = reward + pow(gamma, i) * current_reward
                m_x, s_x = self.propagate(m_x, s_x)

        self.controller.store_transitions(m_x_init, s_x_init, reward)
        return m_x, s_x, reward  # mean, variance of state at timestep+n, cumulate reward in horizon

    def get_data(self, m_x, s_x, horizon, gamma=1.0):  # does not use gamma(discount factor) for now
        """
        Get training data for Controller
        :param m_x: mean of init/start observation [1, dim_state]
        :param s_x: variance of init observation   [dim_state, dim_state]
        :return: lists of mean, variance of different observations, cumulative reward and action_choosen
        """
        ep_m_x, ep_s_x, ep_reward, ep_ac = [], [], [], []

        # for i in range(horizon):
        #     print("Collecting " + str(i) + "th fake data for controller optimization.")
        #     # print(m_x, s_x, np.squeeze(m_x, 0), s_x.shape)
        #     current_reward, done = self.reward.compute_gaussian_reward(np.squeeze(m_x, 0), s_x)
        #     print("m_state: ", m_x, "s_state: ", s_x)
        #     print("reward for current state distribution: ", current_reward)
        #     if done:
        #         ep_reward = list(map(lambda x: x + current_reward, ep_reward))
        #         break
        #     else:
        #         ep_m_x.append(np.squeeze(m_x, 0))
        #         ep_s_x.append(s_x)
        #         m_u, s_u, _ = self.controller.compute_action(np.squeeze(m_x, 0), s_x)
        #         ac = self.controller.sample_action(m_u, s_u)
        #         ep_ac.append(ac)
        #         ep_reward = list(map(lambda x: x + current_reward, ep_reward))
        #         ep_reward.append(current_reward)
        #         m_x, s_x = self.propagate(m_x, s_x)
        #         # to solve the "nan in array" prob
        #         m_x[np.isnan(m_x)] = 0.
        #         s_x[np.isnan(s_x)] = 0.
        # self.controller.store_transition(ep_m_x, ep_s_x, ep_reward, ep_ac)

        for i in range(horizon):
            discount_factor = gamma
            print("Collecting " + str(i) + "th fake data for controller optimization.")
            ep_m_x.append(np.squeeze(m_x, 0))
            ep_s_x.append(s_x)
            m_u, s_u, c_xu = self.controller.compute_action(np.squeeze(m_x, 0), s_x)
            ac = self.controller.sample_action(m_u, s_u)
            ep_ac.append(ac)
            m_x, s_x = self.propagate(m_x, s_x, m_u, s_u, c_xu)
            # solve the "nan in array" prob
            # m_x[np.isnan(m_x)] = 0.01
            # s_x[np.isnan(s_x)] = ?
            # we need to change here if reward depends on action
            current_reward, done = self.reward.compute_gaussian_reward(np.squeeze(m_x, 0), s_x)
            print("m_state: ", m_x, "s_state: ", s_x)
            print("reward for next state distribution: ", current_reward)
            if done:
                ep_reward.append(0.0)
                break
            else:
                for n in range(len(ep_reward) - 1, -1, -1):
                    ep_reward[n] += discount_factor * current_reward
                    discount_factor *= gamma
                ep_reward.append(current_reward)

        self.controller.store_transition(ep_m_x, ep_s_x, ep_reward, ep_ac)

        return ep_m_x, ep_s_x, ep_reward

    def propagate(self, m_x, s_x, m_u, s_u, c_xu):
        """
        :param m_x: mean of current state
        :param s_x: variance of current state
        :param m_u: mean of action
        :param s_u: variance of action
        :param c_xu: input-output covariance for controller
        :return: mean and variance of next predict state
        """
        # return mean and variance of next state

        # m_u, s_u, c_xu = self.controller.compute_action(np.squeeze(m_x, 0), s_x)   # m_x: mean of state, s_x: variance of state
        m_u = np.expand_dims(m_u, 0)
        # m_u.astype(float)
        # s_u.astype(float)
        # c_xu.astype(float)
        # print(type(m_x[0]), type(m_u[0]))

        m = tf.concat([m_x, m_u], axis=1)
        s1 = tf.concat([s_x, s_x@c_xu], axis=1)
        s2 = tf.concat([tf.transpose(s_x@c_xu), s_u], axis=1)
        s = tf.concat([s1, s2], axis=0)

        M_dx, S_dx, C_dx = self.mgpr.predict_on_noisy_inputs(m, s)  # M_dx: mean of dx, S_dx: variance of dx, C_dx:
        M_x = M_dx + m_x
        #TODO: cleanup the following line
        S_x = S_dx + s_x + s1@C_dx + tf.matmul(C_dx, s1, transpose_a=True, transpose_b=True)   #(12) in PILCO paper

        # While-loop requires the shapes of the outputs to be fixed
        M_x.set_shape([1, self.state_dim]); S_x.set_shape([self.state_dim, self.state_dim])

        # M_x, S_x = self.sess.run([M_x, S_x])
        # M_x.eval()
        # S_x.eval()

        return M_x.eval(session=self.sess), S_x.eval(session=self.sess)  # m_u is always the same(the max proba),should we rather use the sampled action ?
