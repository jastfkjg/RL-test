import time
import random

import numpy as np
import gpflow
import pandas as pd
from tensorflow.python import debug as tf_debug

# from actor import Actor, LinearActor
from mgpr import MGPR
from smgpr import SMGPR
from rewards import *

# float64
float_type = gpflow.settings.dtypes.float_type

def discount_reward(rewards, gamma):
    # rewrite this
    discounted = []
    r = 0.
    for reward in rewards[::-1]:
        r = reward + gamma * r
        discounted.append(r)
    return discounted[::-1]

class PILCO:
    def __init__(self, X, Y, num_induced_points=None, controller=None,
                reward=None, m_init=None, S_init=None, name=None, debug=False):
        # super(PILCO, self).__init__(name)
        if not num_induced_points:      # num_induced_points ?
            self.mgpr = MGPR(X, Y)
        else:
            self.mgpr = SMGPR(X, Y, num_induced_points)
        self.state_dim = Y.shape[1]
        self.control_dim = X.shape[1] - Y.shape[1]

        self.sess = gpflow.get_default_session()
        if debug:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        # self.sess.run(tf.global_variables_initializer())

        if controller is None:   # the policy  - to change
            print("controller cannot be None")
        else:
            self.controller = controller

        if reward is None:     # reward function
            self.reward = Reward()
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

    def optimize_controller(self, states, horizon, num_optim=None, num_collect=10, gamma=1.):
        """
        optimize controller/actor's parameters
        :param: states: a array of states from env
        :param: horizon: horizon for culmulative reward
        :param: num_optim: 
        """
        # how to choose init states:
        # 1. random sample from the state_space
        # 2. random sample from the dataset
        start = time.time()

        # sample num_optim states from states
        if num_optim == None:
            num_optim = states.shape[0]
        row = np.random.choice(states.shape[0], num_optim)
        states = states[row, :]

        # variance of states
        s_x = np.diag(np.ones(self.state_dim) * 0.1)

        for i, x in enumerate(states):
            print("Evaluate the " + str(i) + "th init state, total init states: " + str(num_optim))
            m_x = np.expand_dims(x, 0)
            # get fake data with state distribution and gaussian process
            self.get_data(m_x, s_x, horizon, num_collect=num_collect, gamma=gamma)
            # use these data to optimize controller/actor
            self.controller.optimize()
        end = time.time()
        print("Finished with " + str(num_optim) + " times policy/controller optimizations in %.1f seconds" % (end - start))

    def optimize_gp(self):
        """
        Optimizes  GP's hyperparamemeters.
        """
        start = time.time()
        self.mgpr.optimize()    
        end = time.time()
        print("Finished with GPs' optimization in %.1f seconds" % (end - start))

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


    def get_data(self, m_x_init, s_x_init, horizon, num_collect=10, gamma=1.0, ac_sample_num=1):
        """
        Get training data for Controller/Actor
        :param m_x: mean of init/start observation [1, dim_state]
        :param s_x: variance of init observation   [dim_state, dim_state]
        :param num_collect: how many fake data(ep_m_x, ep_s_x, ep_reward, ep_ac) are going to create
        :return: lists of mean, variance of different observations, cumulative reward and action_choosen
        """
        ep_m_x, ep_reward, ep_ac = [], [], []
        m_x = m_x_init
        s_x = s_x_init
        
        for i in range(horizon):
            
            # we fix s_u to 0.1
            m_u, s_u = self.controller.compute_action(m_x)
            m_u = self.controller.sample_action(m_u, s_u)
            s_u = np.ones(self.control_dim) * 0.1

            ep_m_x.append(np.squeeze(m_x, 0))
            # s_u is very small
            ep_ac.append(m_u)
            c_xu = self.controller.compute_covariance(m_x, s_x, m_u, s_u)
            # print("m_u: ", m_u, "s_u: ", s_u, "c_xu: ", c_xu)
            m_x, s_x = self.propagate(m_x, s_x, m_u, np.diag(s_u), c_xu)
            
            if self.reward.with_action:
                current_reward, done = self.reward.compute_gaussian_reward(np.squeeze(m_x, 0), s_x, m_u)
            else:
                current_reward, done = self.reward.compute_gaussian_reward(np.squeeze(m_x, 0), s_x,\
                        threshold=0.4)
            # print("m_x: ", m_x, "s_x: ", s_x)
            # print("current reward %f" %current_reward)
            ep_reward.append(current_reward)
            
            if done:
                # ep_reward.append(total_reward)
                break

        ep_reward = discount_reward(ep_reward, gamma=gamma)
        print(np.mean(np.array(ep_reward)), len(ep_reward))
        self.controller.store_transition(ep_m_x, ep_reward, ep_ac)

        return ep_m_x, ep_reward, ep_ac

    def propagate(self, m_x, s_x, m_u, s_u, c_xu):
        """
        :param m_x: mean of current state
        :param s_x: variance of current state
        :param m_u: mean of action
        :param s_u: variance of action
        :param c_xu: input-output covariance for controller
        :return: mean and variance of next predict state
        """

        # m_u, s_u, c_xu = self.controller.compute_action(np.squeeze(m_x, 0), s_x)   # m_x: mean of state, s_x: variance of state
        m_u = np.expand_dims(m_u, 0)
        # with tf.Graph().as_default() as graph:
            # with tf.Session(graph=graph).as_default():
        
        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                m = tf.concat([m_x, m_u], axis=1)
                s1 = tf.concat([s_x, s_x@c_xu], axis=1)
                s2 = tf.concat([tf.transpose(s_x@c_xu), s_u], axis=1)
                s = tf.concat([s1, s2], axis=0)
                m, s1, s = sess.run([m, s1, s])
        # print("m, s in propagate: ", m.shape, s.shape)
        
        M_dx, S_dx, C_dx = self.mgpr.predict_on_noisy_inputs(m, s)  # M_dx: mean of dx, S_dx: variance of dx
        M_x = M_dx + m_x
        # TODO: cleanup the following line

        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                S_x = S_dx + s_x + s1@C_dx + tf.matmul(C_dx, s1, transpose_a=True, transpose_b=True)   #(12) in PILCO paper

                # While-loop requires the shapes of the outputs to be fixed
                # M_x.set_shape([1, self.state_dim])
                M_x.reshape((1, self.state_dim))
                S_x.set_shape([self.state_dim, self.state_dim])
                S_x = sess.run(S_x)

        # return M_x.eval(session=self.sess), S_x.eval(session=self.sess)
        # return M_x.eval(session=gpflow.get_default_session()),S_x.eval(session=gpflow.get_default_session())
        return M_x, S_x


