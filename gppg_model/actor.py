import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp 

# np.random.seed(1)
# tf.set_random_seed(1)

class Actor():

    def __init__(self, action_dim, state_dim, learning_rate, action_choice=0, hidden_size=10, discrete_ac=False):
        self.action_dim = action_dim
        self.action_choice = action_choice
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.discrete_ac = discrete_ac
        self.hidden_size = hidden_size

        self.ep_m_obs, self.ep_s_obs, self.ep_ac_choosen, self.ep_pilco_r = [], [], [], []

        self.tfd = tfp.distributions

        self.weight1 = tf.Variable(tf.random_normal([self.state_dim * (self.state_dim + 1), self.hidden_size], dtype=tf.float64), name="fc1_weight")
        self.bias1 = tf.Variable(tf.random_normal([self.hidden_size], dtype=tf.float64), name="fc1_bias")
        self.weight2 = tf.Variable(tf.random_normal([self.hidden_size, self.action_dim], dtype=tf.float64), name="fc2_weight")
        self.bias2 = tf.Variable(tf.random_normal([self.action_dim], dtype=tf.float64), name="fc2_bias")
        self.weight3 = tf.Variable(tf.random_normal([self.hidden_size, self.action_dim], dtype=tf.float64), name="fc3_weight")
        self.bias3 = tf.Variable(tf.random_normal([self.action_dim], dtype=tf.float64), name="fc3_bias")

        # num of optimizations already done
        self.num_optim = tf.Variable(0)

        self._build_net()

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(config=config)

        # self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):

        with tf.variable_scope("Inputs"):
            self.m_obs = tf.placeholder(tf.float64, [None, self.state_dim], name="mean_observations")
            self.s_obs = tf.placeholder(tf.float64, [None, self.state_dim * self.state_dim], name="variance_observation")

        with tf.variable_scope("Optim_inputs"):
            # the type of action in gym env
            self.ac = tf.placeholder(tf.float64, [None, self.action_dim], name="action_choosen")
            self.r = tf.placeholder(tf.float64, [None, ], name="return_from_pilco")

        # s_obs = tf.reshape(self.s_obs, [None, self.state_dim * self.state_dim])

        # m_obs = tf.expand_dims(self.m_obs, axis=2)    # [None, state_dim, 1]

        # consider to concatenate m_obs and s_obs
        self.obs = tf.concat([self.m_obs, self.s_obs], 1)   # [None, state_dim * (state_dim + 1)]

        # How to choose the network architecture ? normally we have hidden size of 5, 10, 15 ?

        # fc1
        layer = tf.add(tf.matmul(self.obs, self.weight1), self.bias1)
        layer = tf.nn.tanh(layer)
        # fc2
        self.m_ac = tf.add(tf.matmul(layer, self.weight2), self.bias2)
        # fc3
        self.s_ac = tf.add(tf.matmul(layer, self.weight3), self.bias3)


        # layer = tf.layers.dense(inputs=self.obs, units=10, activation=tf.nn.tanh,
        # 						kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        # 						bias_initializer=tf.constant_initializer(0.1), name='fc1')
        #
        # # fc2
        # self.m_ac = tf.layers.dense(inputs=layer, units=self.action_dim, activation=None,
        # 						kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        # 						bias_initializer=tf.constant_initializer(0.1), name='fc2')

        # self.all_act_prob = tf.nn.softmax(all_act, name="act_prob")

        # self.m_ac = tf.add(tf.matmul(self.m_obs, self.weight), self.bias)  # tf.add
        # Use re-parameterization method to get action variance
        # Note: we can not apply the same weights on action variance ?   How to restrict the output, when action dim is small
        # whether it's possilbe to use deterministic policy -- do not have action variance here, maybe add a small variance for GP
        # we can not apply deterministic policy since the Q function is not a function of action here, we can not do BP.
        ##weight = tf.reshape(self.weight, [1, self.state_dim, self.action_dim])
        # weight = tf.tile(weight, [self.s_obs.shape[0], 1, 1]) XXX
        ##self.s_ac = tf.transpose(weight, perm=[0, 2, 1]) @ self.s_obs @ weight
        # self.s_ac = tf.matmul(self.s_obs, self.weight)

        # How to get variance of action ?
        # 1. give a constant variance
        # self.s_ac = 0.1 * tf.diag(s_ac)
        # 2. use a neural network with m_obs, s_obs as input, just like m_ac
        # here we only output the diag values in s_ac

        # self.s_ac = tf.layers.dense(inputs=layer, units=self.action_dim, activation=None,
        # 						kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        # 						bias_initializer=tf.constant_initializer(0.1), name='fc3')
        # self.s_ac = tf.diag(self.s_ac) # what if s_ac: [None, action_dim]

        # V: How to calculate input-output covariance
        # V: input-output covariance TODO:  calculate V
        # 1. for NN, it's too complicate to calculate
        # 2. gibbs sampling to calculate V = E(state*action) - m_state*m_action
        # To get E[state * action]:
        # - sample states from self.m_obs, self.s_obs, actions from self.m_ac, self.s_ac
        # - then compute mean of sum(state * action) = E(state * action]
        # - V = E(state * action) - self.m_ac * self.m_obs
        # # shape: [state_dim, action_dim]
        # maybe move it to compute_action() method, because we don't need to calculate a batch of V
        # self.V =

        # distribution of action, we need to change to multi-variate gaussian dist
        # or we may use re-parameterization trick
        self.dist = self.tfd.MultivariateNormalDiag(loc=self.m_ac, scale_diag=self.s_ac)
        # self.dist = self.tfd.Normal(loc=self.m_ac, scale=self.s_ac)

        with tf.variable_scope("loss"):
            # policy gradient: minimize -(log(pi)*r)   #TODO: discrete case
            neg_log_prob = - tf.log(self.dist.prob(self.ac))
            loss = tf.reduce_mean(neg_log_prob * self.r)

        with tf.variable_scope("train"):
            # choose Adam optimizer
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    #  output the mean and variance of the action
    #  the aciton space is continuous ? what if the action space is discret

    def compute_action(self, m, s, sample_num=10):
        """
        This function only calculate a single action, not a batch of actions
        :param m: mean of observation
        :param s: variance of observation
        :param sample_num: sample num to calculate input-output covariance
        :return: the mean, variance of action, input-output covariance
        """
        e, v = tf.linalg.eigh(s)
        eps = 1e-5
        e = tf.maximum(e, eps)
        s_state_pos_def = tf.matmul(tf.matmul(v, tf.diag(e)), tf.transpose(v))
        # print("m, s_state_pos_def: ", m, s_state_pos_def)
        # add noise to solve Cholesky decomposition prob
        # batched_eye = np.eye(s.shape[0])
        # s_with_noise = s + 0.1 * batched_eye
        try:
            dist_obs = self.tfd.MultivariateNormalFullCovariance(loc=m, covariance_matrix=s_state_pos_def)
        except:
            print("Cholesky decomposition failed. In this case, we only take diag element of obs variance")
            dist_obs = self.tfd.MultivariateNormalDiag(loc=m, scale_diag=np.diag(s))
        states = dist_obs.sample([sample_num])   # [10, state_dim]

        m = np.reshape(m, (1, self.state_dim))
        m_obs = tf.transpose(m)
        s = np.reshape(s, (1, self.state_dim * self.state_dim))
        # obs = np.concatenate((m, s), 1)
        m_ac = self.sess.run(self.m_ac, feed_dict={self.m_obs: m, self.s_obs: s})  # [1, action_dim] ? [1, 4, 1]
        # print("m action", m_ac)
        m_ac = np.squeeze(m_ac, 0)     # [action_dim]

        s_ac = self.sess.run(self.s_ac, feed_dict={self.m_obs: m, self.s_obs: s})  # [1, action_dim]
        s_ac = np.squeeze(s_ac, 0)	  # [action_dim]

        # abs, s_ac should not be negative
        s_ac = abs(s_ac)
        # print("s action", s_ac)

        # add noise to solve Cholesky decomposition prob
        # batched_eye = np.eye(s_ac.shape[0])
        # s_ac_with_noise = s_ac + 0.001 * batched_eye
        dist_ac = self.tfd.MultivariateNormalDiag(loc=m_ac, scale_diag=s_ac)
        actions = dist_ac.sample([sample_num])   # [10, action_dim]

        s_ac = np.diag(s_ac)

        # s_ac = s_ac * np.eye(self.action_dim)  # [action_dim, action_dim]

        # dist_ac = self.tfd.MultivariateNormalFullCovariance(loc=m_ac, covariance_matrix=s_ac_with_noise)
        # actions = dist_ac.sample([sample_num])

        # E[state * action]
        states = tf.transpose(states)
        V = tf.matmul(states, actions) / sample_num - tf.matmul(m_obs, tf.expand_dims(m_ac, 0))

        V = self.sess.run(V)
        # print(V)

        # return the mean, variance of action; input-output covariance, the sample action ?
        return m_ac, s_ac, V

    # def compute_det_action(self, m):
    # 	"""
    # 	:param m: the mean of observation
    # 	:return: the mean of action
    # 	"""
    # 	m_ac = self.compute_action(m, tf.zeros([self.state_dim, self.state_dim]))[0]
    # 	return m_ac

    def sample_action(self, m, s):
        """
        :param m: mean of action
        :param s: variance of action
        :return: a random sample from the action distribution
        """
        # whether to use multinominal guassian distribution
        # multivariate normal
        ac = np.random.multivariate_normal(m, s, 1)
        ac = np.squeeze(ac, 0)
        # dist = self.tfd.Normal(loc=m, scale=s)
        # # sample one action ?
        # ac = dist.sample([1])
        # ac = self.sess.run(ac)

        return ac

    def take_action(self, m_x, s_x):
        """
        :param m_x: mean of observation
        :param s_x: variance of observation
        :return: the choosen action in gym env -- to handle discrete case ?
        """
        m_ac, s_ac, _ = self.compute_action(m_x, s_x)
        ac = self.sample_action(m_ac, s_ac)  # a list of n num, n is the action dim, for discrete action space, we only need one

        if self.discrete_ac:
            # only for CartPole, TODO
            if ac < 0:
                ac = 0
            else:
                ac = 1
        # ac_prob = tf.nn.softmax(ac, name="discrete_act_prob")
        # ac = self.sess.run(ac_prob)
        # ac = np.random.choice(ac.shape[1], p=ac.ravel())  # ac: a number between 0 and ac.shape[1] which is the action dim
        else:
            # cliprangge already exist in gym env
            pass
        return ac

    def take_quick_action(self, state):
        """
        compute_action is too slow because we need to calculate the input-output covariance
        but in real env, it's unnecessary, this method is for interact with real env
        """
        s_obs = np.diag(np.ones(self.state_dim) * 0.1)  # [state_dim, state_dim]
        m_obs = np.reshape(state, (1, self.state_dim))  # [1, state_dim]
        s_obs = np.reshape(s_obs, (1, self.state_dim * self.state_dim))  # [1, state_dim, state_dim]

        m_ac, s_ac = self.sess.run([self.m_ac, self.s_ac], feed_dict={self.m_obs: m_obs, self.s_obs: s_obs})  # [1, action_dim]
        m_ac = np.squeeze(m_ac, 0)
        s_ac = np.squeeze(s_ac, 0)
        s_ac = abs(s_ac)
        s_ac = np.diag(s_ac)
        # import ipdb
        # ipdb.set_trace()
        action = self.sample_action(m_ac, s_ac)

        # action = np.squeeze(action, 0)  # [action_dim]
        # if self.discrete_ac:
            # only for CartPole TODO
            # we should find a better way to find action for discrete case
            # if m_ac < 0:
                # return 0
            # else:
                # return 1
        # m_ac = np.reshape(m_ac, (1, ))
        return action

    def optimize(self, m_obs=None, s_obs=None, pilco_return=None, action_choosen=None):
        """
        optimize the policy, we take action_choosen as the mean action output from policy
        """
        # m_obs: a list of mean state in an episode
        # pilco_return: a list of cumulative reward from a list of state

        if not (action_choosen and m_obs and s_obs and pilco_return):
            # m_ac = action_choosen  # random sample a serie of action from action distribution ?
            m_obs = self.ep_m_obs
            s_obs = self.ep_s_obs
            action_choosen = self.ep_ac_choosen
            pilco_return = self.ep_pilco_r

        print("Now we begin the optimization for controller.")
        batch_size = min(len(m_obs), len(s_obs), len(action_choosen), len(pilco_return))
        m_obs = m_obs[:batch_size]
        s_obs = s_obs[:batch_size]
        action_choosen = action_choosen[:batch_size]
        pilco_return = pilco_return[:batch_size]
        # assert len(s_obs) == batch_size
        # assert len(action_choosen) == batch_size
        # assert len(pilco_return) == batch_size

        # or we can concatenate m_obs, s_obs together for input
        s_obs = np.reshape(s_obs, (batch_size, self.state_dim * self.state_dim))

        self.sess.run(self.train_op, feed_dict={
            self.m_obs: m_obs,
            self.s_obs: s_obs,
            self.ac: action_choosen,
            self.r: pilco_return,
        })

        print("Controller optimization finished.")
        self.sess.run(self.num_optim.assign(self.num_optim + 1))
        # reset the episode record
        self.ep_m_obs, self.ep_s_obs, self.ep_pilco_r, self.ep_ac_choosen = [], [], [], []

    def get_num_optim(self):
        return self.sess.run(self.num_optim)

    def store_transition(self, m_obs, s_obs, pilco_return, action_choosen=None):
        """
        pilco_return: cumulative reward for a state
        """
        self.ep_m_obs = m_obs
        self.ep_s_obs = s_obs
        if action_choosen:
            self.ep_ac_choosen = action_choosen
        self.ep_pilco_r = pilco_return

    def save_weights(self, path):
        saver = tf.train.Saver([self.weight1, self.weight2, self.weight3, self.bias1, self.bias2, self.bias3, self.num_optim])
        save_path = saver.save(self.sess, path)
        print("model saved in file: %s" % save_path)

    def load_weights(self, path):
        saver = tf.train.Saver([self.weight1, self.weight2, self.weight3, self.bias1, self.bias2, self.bias3, self.num_optim])
        saver.restore(self.sess, path)


class LinearActor():
    """
    To handle the input, output uncertainty,
    we first test a simple linear policy.
    """
    def __init__(self, action_dim, action_choice, state_dim, learning_rate, discrete_ac=False):
        self.action_dim = action_dim
        self.action_choice = action_choice
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.discrete_ac = discrete_ac

        self.ep_m_obs, self.ep_s_obs, self.ep_ac_choosen, self.ep_pilco_r = [], [], [], []

        self.tfd = tfp.distributions

        self.weight = tf.Variable(tf.random_normal([self.state_dim, self.action_dim], dtype=tf.float64))
        self.bias = tf.Variable(tf.random_normal([self.action_dim], dtype=tf.float64))

        self._build_net()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):

        with tf.variable_scope("linear_inputs"):
            self.m_obs = tf.placeholder(tf.float64, [None, self.state_dim], name="mean_observations")
            self.s_obs = tf.placeholder(tf.float64, [None, self.state_dim, self.state_dim], name="variance_observation")

        with tf.variable_scope("linear_optim_inputs"):
            # the type of action in gym env
            self.ac = tf.placeholder(tf.float64, [None, self.action_dim], name="action_choosen")
            self.r = tf.placeholder(tf.float64, [None, ], name="return_from_pilco")

        self.m_ac = tf.add(tf.matmul(self.m_obs, self.weight), self.bias)   # tf.add
        # Use re-parameterization method to get action variance
        # Note: we can not apply the same weights on action variance ?   How to restrict the output, when action dim is small
        # whether it's possilbe to use deterministic policy -- do not have action variance here, maybe add a small variance for GP
        # we can not apply deterministic policy since the Q function is not a function of action here, we can not do BP.
        weight = tf.reshape(self.weight, [1, self.state_dim, self.action_dim])
        # weight = tf.tile(weight, [self.s_obs.shape[0], 1, 1]) XXX
        self.s_ac = tf.transpose(weight, perm=[0, 2, 1]) @ self.s_obs @ weight
        # self.s_ac = tf.matmul(self.s_obs, self.weight)

        # V: How to calculate input-output covariance
        # V: input-output covariance TODO
        # 1. for NN, it's too complicate to calculate
        # 2. gibbs sampling to calculate E(state*action) - m_state*m_action
        self.V = self.weight  # shape: [state_dim, action_dim]

        # distribution of action
        self.dist = self.tfd.Normal(loc=self.m_ac, scale=self.s_ac)

        with tf.variable_scope("loss"):
            # policy gradient: minimize -(log(pi)*r)   #TODO: discrete case
            # if we use deterministic policy, see DDPG or DPG for gradient calculating
            neg_log_prob = - tf.log(self.dist.prob(self.ac))
            loss = tf.reduce_mean(neg_log_prob * self.r)

        with tf.variable_scope("train"):
            # choose Adam optimizer
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        #  output the mean and variance of the action
        #  the aciton space is continuous ? what if the action space is discret

    def compute_action(self, m, s):
        """
        :param m: mean of observation
        :param s: variance of observation
        :return: the mean, variance of action, input-output covariance?
        """
        # m: mean of observation, s: variance of observation
        # m = np.expand_dims(m, 0)
        m = np.reshape(m, (1, self.state_dim))
        s = np.reshape(s, (1, self.state_dim, self.state_dim))
        m_ac = self.sess.run(self.m_ac, feed_dict={self.m_obs: m})
        s_ac = self.sess.run(self.s_ac, feed_dict={self.s_obs: s})
        m_ac = np.squeeze(m_ac, 0)
        s_ac = np.squeeze(s_ac, 0)
        V = self.sess.run(self.V)

        # dist = self.tfd.Normal(loc=m_ac, scale=s_ac)
        # a = dist.sample([1])  # sample one action ?

        # return the mean, variance of action; input-output covariance, the sample action ?
        return m_ac, s_ac, V

    def compute_det_action(self, m):
        """
        calculate action with only mean of state
        :param m: the mean of observation
        :return: the mean of action
        """
        m_ac = self.compute_action(m, tf.zeros([self.state_dim, self.state_dim]))[0]
        return m_ac

    def sample_action(self, m, s):
        """
        :param m: mean of action
        :param s: variance of action
        :return: a random sample from the action distribution
        """
        # whether to use multinominal guassian distribution
        # multivariate normal
        ac = np.random.multivariate_normal(m, s, 1)
        ac = np.squeeze(ac, 0)
        # dist = self.tfd.Normal(loc=m, scale=s)
        # # sample one action ?
        # ac = dist.sample([1])
        # ac = self.sess.run(ac)

        return ac

    def take_action(self, m_x, s_x):
        """
        :param m_x: mean of observation
        :param s_x: variance of observation
        :return: the choosen action in gym env -- to handle discrete case ?
        """
        m_ac, s_ac, _ = self.compute_action(m_x, s_x)
        ac = self.sample_action(m_ac, s_ac)   # a list of n num, n is the action dim, for discrete action space, we only need one

        if self.discrete_ac:
            # only for CartPole, TODO
            if ac < 0:
                ac = 0
            else:
                ac = 1
            # ac_prob = tf.nn.softmax(ac, name="discrete_act_prob")
            # ac = self.sess.run(ac_prob)
            # ac = np.random.choice(ac.shape[1], p=ac.ravel())  # ac: a number between 0 and ac.shape[1] which is the action dim
        else:
            # cliprangge already exist in gym env
            pass
        return ac


    def optimize(self, m_obs=None, s_obs=None, pilco_return=None, action_choosen=None):
        """
        optimize the policy, we take action_choosen as the mean action output from policy
        """
        # m_obs: a list of mean state in an episode
        # pilco_return: a list of cumulative reward from a list of state

        if not (action_choosen and m_obs and s_obs and pilco_return):
            # m_ac = action_choosen  # random sample a serie of action from action distribution ?
            m_obs = self.ep_m_obs
            s_obs = self.ep_s_obs
            action_choosen = self.ep_ac_choosen
            pilco_return = self.ep_pilco_r

        print("Now we begin the optimization for controller.")

        self.sess.run(self.train_op, feed_dict={
            self.m_obs: m_obs,
            self.s_obs: s_obs,
            self.ac: action_choosen,
            self.r: pilco_return,
            })

        print("Controller optimization finished.")

        # reset the episode record
        self.ep_m_obs, self.ep_s_obs, self.ep_pilco_r, self.ep_ac_choosen = [], [], [], []

    def store_transition(self, m_obs, s_obs, pilco_return, action_choosen=None):
        """
        pilco_return: cumulative reward for a state
        """
        self.ep_m_obs = m_obs
        self.ep_s_obs = s_obs
        if action_choosen:
            self.ep_ac_choosen = action_choosen
        self.ep_pilco_r = pilco_return

    def save_weights(self, path):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path)
        print("model saved in file: %s" % save_path)



