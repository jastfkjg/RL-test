import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp
from tensorflow.python import debug as tf_debug

# np.random.seed(1)
# tf.set_random_seed(1)

class Actor():

    def __init__(self, env, action_dim, state_dim, learning_rate, hidden_size=10, discrete_ac=False, debug=False):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.discrete_ac = discrete_ac
        self.hidden_size = hidden_size
        self.env = env

        self.graph = tf.Graph()
        # assert self.graph is tf.get_default_graph()

        self.ep_m_obs, self.ep_s_obs, self.ep_ac_choosen, self.ep_pilco_r = [], [], [], []

        with self.graph.as_default():
            self.tfd = tfp.distributions

            self.weight1 = tf.Variable(tf.random_normal([self.state_dim * (self.state_dim + 1), self.hidden_size], dtype=tf.float64), name="fc1_weight")
            self.bias1 = tf.Variable(tf.random_normal([self.hidden_size], dtype=tf.float64), name="fc1_bias")
            self.weight2 = tf.Variable(tf.random_normal([self.hidden_size, self.action_dim], dtype=tf.float64), name="fc2_weight")
            self.bias2 = tf.Variable(tf.random_normal([self.action_dim], dtype=tf.float64), name="fc2_bias")
            # self.weight3 = tf.Variable(tf.random_normal([self.hidden_size, self.action_dim], dtype=tf.float64), name="fc3_weight")
            # self.bias3 = tf.Variable(tf.random_normal([self.action_dim], dtype=tf.float64), name="fc3_bias")

            self._build_net()
            self.init = tf.global_variables_initializer()

        assert self.weight1.graph is self.graph

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(graph=self.graph, config=config)

        if debug:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

        assert self.sess.graph is self.graph

        self.sess.run(self.init)

    def _build_net(self):

        with tf.variable_scope("Inputs"):
            self.m_obs = tf.placeholder(tf.float64, [None, self.state_dim], name="mean_observations")
            self.s_obs = tf.placeholder(tf.float64, [None, self.state_dim * self.state_dim], name="variance_observation")

        with tf.variable_scope("Optim_inputs"):
            self.ac = tf.placeholder(tf.float64, [None, self.action_dim], name="action_choosen")
            self.r = tf.placeholder(tf.float64, [None, ], name="return_from_pilco")

        # m_obs = tf.expand_dims(self.m_obs, axis=2)    # [None, state_dim, 1]

        # consider to concatenate m_obs and s_obs
        self.obs = tf.concat([self.m_obs, self.s_obs], 1)   # [None, state_dim * (state_dim + 1)]

        # fc1
        layer = tf.add(tf.matmul(self.obs, self.weight1), self.bias1)
        layer = tf.nn.tanh(layer)
        # fc2
        self.m_ac = tf.add(tf.matmul(layer, self.weight2), self.bias2)
        s_ac_init = tf.constant(np.random.rand(self.action_dim))
        self.s_ac = tf.get_variable("s_action", initializer=s_ac_init, dtype=tf.float64)
        # fc3
        # self.s_ac = tf.add(tf.matmul(layer, self.weight3), self.bias3)

        # layer = tf.layers.dense(inputs=self.obs, units=10, activation=tf.nn.tanh,
        # 						kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        # 						bias_initializer=tf.constant_initializer(0.1), name='fc1')
        #
        # # fc2
        # self.m_ac = tf.layers.dense(inputs=layer, units=self.action_dim, activation=None,
        # 						kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        # 						bias_initializer=tf.constant_initializer(0.1), name='fc2')

        # or we may use re-parameterization trick
        self.dist = self.tfd.MultivariateNormalDiag(loc=self.m_ac, scale_diag=self.s_ac)

        with tf.variable_scope("loss"):
            # policy gradient: minimize -(log(pi)*r)   
            neg_log_prob = - tf.log(self.dist.prob(self.ac))
            loss = tf.reduce_mean(neg_log_prob * self.r)
            # loss = tf.reduce_mean(neg_log_prob * self.r) + 0.01 * tf.nn.l2_loss(self.s_ac)

        with tf.variable_scope("train"):
            # Adam optimizer
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            
            gvs = self.optimizer.compute_gradients(loss)
            print(gvs)
            try:
                capped_gvs = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in gvs]
                self.train_op = self.optimizer.apply_gradients(capped_gvs)
            except: 
                pass


    def compute_action(self, m, s, sample_num=10):
        """
        This function only calculate a single action, not a batch of actions
        :param m: mean of observation
        :param s: variance of observation
        :param sample_num: sample num to calculate input-output covariance
        :return: the mean, variance of action, input-output covariance
        """
        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                e, v = tf.linalg.eigh(s)
                eps = 1e-5
                e = tf.maximum(e, eps)
                s_state_pos_def = tf.matmul(tf.matmul(v, tf.diag(e)), tf.transpose(v))
                # add noise to solve Cholesky decomposition prob
                # batched_eye = np.eye(s.shape[0])
                # s_with_noise = s + 0.1 * batched_eye
                try:
                    dist_obs = self.tfd.MultivariateNormalFullCovariance(loc=m, covariance_matrix=s_state_pos_def)
                    states = dist_obs.sample([sample_num])
                    states = sess.run(states)
                except tf.errors.InvalidArgumentError:
                    print("Cholesky decomposition failed. In this case, we only take diag element of obs variance")
                    dist_obs = self.tfd.MultivariateNormalDiag(loc=m, scale_diag=abs(np.diag(s)))
                    states = dist_obs.sample([sample_num])   # [10, state_dim]
                    states = sess.run(states)

                m = np.reshape(m, (1, self.state_dim))
                m_obs = tf.transpose(m)
                s = np.reshape(s, (1, self.state_dim * self.state_dim))
                # obs = np.concatenate((m, s), 1)
                m_ac = self.sess.run(self.m_ac, feed_dict={self.m_obs: m, self.s_obs: s})  # [1, action_dim] ? [1, 4, 1]
                # print("m action", m_ac)
                m_ac = np.squeeze(m_ac, 0)     # [action_dim]

                s_ac = self.sess.run(self.s_ac) 
                print("s_ac: ", s_ac)
                # s_ac = np.squeeze(s_ac, 0)	  # [action_dim]

                # s_ac should not be negative
                s_ac = abs(s_ac)

                dist_ac = self.tfd.MultivariateNormalDiag(loc=m_ac, scale_diag=s_ac)
                actions = dist_ac.sample([sample_num])   # [10, action_dim]

                s_ac = np.diag(s_ac)

                # E[state * action]
                states = tf.transpose(states)

                assert sample_num > 0
                V = tf.matmul(states, actions) / sample_num - tf.matmul(m_obs, tf.expand_dims(m_ac, 0))

                V = sess.run(V)
        # print(V)

        # return the mean, variance of action; input-output covariance
        return m_ac, s_ac, V

    def random_action(self, m, s, sample_num=10):
        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                e, v = tf.linalg.eigh(s)
                eps = 1e-5
                e = tf.maximum(e, eps)
                s_state_pos_def = tf.matmul(tf.matmul(v, tf.diag(e)), tf.transpose(v))
                # add noise to solve Cholesky decomposition prob
                # batched_eye = np.eye(s.shape[0])
                # s_with_noise = s + 0.1 * batched_eye
                try:
                    dist_obs = self.tfd.MultivariateNormalFullCovariance(loc=m, covariance_matrix=s_state_pos_def)
                    states = dist_obs.sample([sample_num])
                    states = sess.run(states)
                except tf.errors.InvalidArgumentError:
                    print("Cholesky decomposition failed. In this case, we only take diag element of obs variance")
                    dist_obs = self.tfd.MultivariateNormalDiag(loc=m, scale_diag=abs(np.diag(s)))
                    states = dist_obs.sample([sample_num])   # [10, state_dim]
                    states = sess.run(states)

                m_ac = self.env.action_space.sample()
                m_ac = m_ac.astype('float64')
                s_ac = np.ones(self.action_dim) * 0.1
                dist_ac = self.tfd.MultivariateNormalDiag(loc=m_ac, scale_diag=s_ac)
                actions = dist_ac.sample([sample_num])   # [10, action_dim]

                s_ac = abs(np.diag(s_ac))
                states = tf.transpose(states)
                m = np.reshape(m, (1, self.state_dim))
                m = tf.transpose(m)
                assert sample_num > 0

                V = tf.matmul(states, actions) / sample_num - tf.matmul(m, tf.expand_dims(m_ac, 0))

                V = sess.run(V)

        return m_ac, s_ac, V

    def sample_action(self, m, s, sample_num=1):
        """
        :param m: mean of action
        :param s: variance of action
        :return: a random sample from the action distribution
        """
        # whether to use multinominal guassian distribution
        # multivariate normal
        ac = np.random.multivariate_normal(m, s, sample_num)
        if sample_num == 1:
            ac = np.squeeze(ac, 0)

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
        s_ac = abs(s_ac)
        s_ac = np.diag(s_ac)
        # import ipdb
        # ipdb.set_trace()
        action = self.sample_action(m_ac, s_ac)

        return action

    def optimize(self, m_obs=None, s_obs=None, pilco_return=None, action_choosen=None):
        """
        optimize the policy, we take action_choosen as the mean action output from policy
        """

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

        # or we can concatenate m_obs, s_obs together for input
        s_obs = np.reshape(s_obs, (batch_size, self.state_dim * self.state_dim))

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

    def store_fake_transition(self, m_obs, s_obs, pilco_r, actions):
        self.ep_m_obs = m_obs
        self.ep_s_obs = s_obs
        self.ep_pilco_r = pilco_r
        self.ep_acs = actions

    def save_weights(self, path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            save_path = saver.save(self.sess, path)
        print("model saved in file: %s" % save_path)

    def load_weights(self, path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, path)



