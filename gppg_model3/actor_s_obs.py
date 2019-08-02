import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp
from tensorflow.python import debug as tf_debug

# np.random.seed(1)
# tf.set_random_seed(1)

class Actor():

    def __init__(self, env, action_dim, state_dim, learning_rate, hidden_units=[64, 64], debug=False):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.env = env

        self.graph = tf.Graph()

        self.ep_m_obs, self.ep_s_obs, self.ep_ac_choosen, self.ep_pilco_r = [], [], [], []

        with self.graph.as_default():
            self.tfd = tfp.distributions

            self._build_net()
            self.init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(graph=self.graph, config=config)

        if debug:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

        assert self.sess.graph is self.graph

        self.sess.run(self.init)

    def _build_net(self):

        with tf.variable_scope("Inputs"):
            self.m_obs = tf.placeholder(tf.float64, [None, self.state_dim], name="mean_observations")
            self.s_obs = tf.placeholder(tf.float64, [None, self.state_dim, self.state_dim], name="variance_observation")

        with tf.variable_scope("Optim_inputs"):
            self.ac = tf.placeholder(tf.float64, [None, self.action_dim], name="action_choosen")
            self.r = tf.placeholder(tf.float64, [None, ], name="return_from_pilco")

        # consider to concatenate m_obs and s_obs
        m_obs = tf.expand_dims(self.m_obs, axis=-1)
        obs = tf.concat([m_obs, self.s_obs], -1)   # [None, state_dim, (state_dim + 1)]
        obs = tf.layers.flatten(obs)

        fc1 = tf.layers.dense(obs, self.hidden_units[0], activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, self.hidden_units[1], activation=tf.nn.relu)
        self.m_ac = tf.layers.dense(fc2, self.action_dim, activation=None)
        self.log_s_ac = tf.get_variable(name="s_action", shape=[1, self.action_dim],
                initializer=tf.zeros_initializer(), dtype=tf.float64)
        self.s_ac = tf.exp(self.m_ac * 0.0 + self.log_s_ac)     # [batch_size, action_dim]
        self.dist = self.tfd.MultivariateNormalDiag(loc=self.m_ac, scale_diag=self.s_ac)
        self.sample_ac = self.dist.sample([1])
        # s_ac_init = tf.constant(np.random.rand(self.action_dim))
        # self.s_ac = tf.get_variable("s_action", initializer=s_ac_init, dtype=tf.float64)
        # layer = tf.layers.dense(inputs=self.obs, units=10, activation=tf.nn.tanh,
        # 						kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        # 						bias_initializer=tf.constant_initializer(0.1), name='fc1')
        #
        # # fc2
        # self.m_ac = tf.layers.dense(inputs=layer, units=self.action_dim, activation=None,
        # 						kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        # 						bias_initializer=tf.constant_initializer(0.1), name='fc2')

        with tf.variable_scope("loss"):
            # policy gradient: minimize -(log(pi)*r)   
            neg_log_prob = - tf.log(self.dist.prob(self.ac))
            self.loss = tf.reduce_mean(neg_log_prob * self.r)
            # loss = tf.reduce_mean(neg_log_prob * self.r) + 0.01 * tf.nn.l2_loss(self.s_ac)

        with tf.variable_scope("train"):
            # Adam optimizer
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            
            gvs = self.optimizer.compute_gradients(self.loss)
            print(gvs)
            capped_gvs = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in gvs]
            self.train_op = self.optimizer.apply_gradients(capped_gvs)

    def compute_action(self, m_obs, s_obs):
        """Outputs: mean of action: [action_dim], variance of action: [action_dim]"""
        m_obs = np.reshape(m_obs, (1, self.state_dim))
        s_obs = np.reshape(s_obs, (1, self.state_dim, self.state_dim))
        m_ac, s_ac = self.sess.run([self.m_ac, self.s_ac], feed_dict={self.m_obs: m_obs, self.s_obs:s_obs})
        return np.squeeze(m_ac, 0), np.squeeze(s_ac, 0)

    def compute_covariance(self, m_obs, s_obs, m_ac, s_ac, sample_num=20):
        """
        :param sample_num: sample num to calculate input-output covariance
        :return: input-output covariance
        """
        m_obs = np.reshape(m_obs, (self.state_dim, ))
        s_obs = np.reshape(s_obs, (self.state_dim, self.state_dim))
        m_ac = np.reshape(m_ac, (self.action_dim, ))
        s_ac = np.reshape(s_ac, (self.action_dim, ))
        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                e, v = tf.linalg.eigh(s_obs)
                eps = 1e-5
                e = tf.maximum(e, eps)
                s_state_pos_def = tf.matmul(tf.matmul(v, tf.diag(e)), tf.transpose(v))
                # add noise to solve Cholesky decomposition prob
                # batched_eye = np.eye(s.shape[0])
                # s_with_noise = s + 0.1 * batched_eye
                try:
                    dist_obs = self.tfd.MultivariateNormalFullCovariance(loc=m_obs, covariance_matrix=s_state_pos_def)
                    states = dist_obs.sample([sample_num])
                    states = sess.run(states)
                except tf.errors.InvalidArgumentError:
                    print("Cholesky decomposition failed. In this case, we only take diag element of obs variance")
                    dist_obs = self.tfd.MultivariateNormalDiag(loc=m_obs, scale_diag=abs(np.diag(s_obs)))
                    states = dist_obs.sample([sample_num])
                    states = sess.run(states)
                
                m_obs = np.expand_dims(m_obs, 0)
                m_obs = tf.transpose(m_obs)

                dist_ac = self.tfd.MultivariateNormalDiag(loc=m_ac, scale_diag=s_ac)
                actions = dist_ac.sample([sample_num])   # [sample_num, action_dim]
                # E[state * action]
                states = tf.transpose(states)

                assert sample_num > 0
                V = tf.matmul(states, actions) / sample_num - tf.matmul(m_obs,np.expand_dims(m_ac, 0))

                V = sess.run(V)
        # return the mean, variance of action; input-output covariance
        return V

    def random_action(self):

        m_ac = self.env.action_space.sample()
        m_ac = m_ac.astype('float64')
        s_ac = np.ones(self.action_dim) * 0.01

        return m_ac, s_ac

    def sample_action(self, m, s, sample_num=1):
        """
        :param m: mean of action
        :param s: variance of action, (size: [action_dim])
        :return: a random sample from the action distribution
        """
        # whether to use multinominal guassian distribution
        # multivariate normal
        # ac = np.random.multivariate_normal(m, s, sample_num)
        ac = m + s * np.random.normal(size=m.shape)
        # if sample_num == 1:
            # ac = np.squeeze(ac, 0)

        return ac

    def take_action(self, m_x, s_x):
        """
        :param m_x: mean of observation
        :param s_x: variance of observation
        :return: the choosen action in gym env -- to handle discrete case ?
        """
        m_ac, s_ac, _ = self.compute_action(m_x, s_x)
        ac = self.sample_action(m_ac, s_ac)  # a list of n num, n is the action dim, for discrete action space, we only need one

        return ac

    def take_quick_action(self, state):
        
        s_obs = np.diag(np.ones(self.state_dim) * 0.1)  # [state_dim, state_dim]
        m_obs = np.reshape(state, (1, self.state_dim))  # [1, state_dim]
        s_obs = np.reshape(s_obs, (1, self.state_dim, self.state_dim))  # [1, state_dim, state_dim]

        m_ac, s_ac = self.sess.run([self.m_ac, self.s_ac], feed_dict={self.m_obs: m_obs, self.s_obs: s_obs})  # [1, action_dim]
        m_ac = np.squeeze(m_ac, 0)
        s_ac = np.squeeze(s_ac, 0)

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

        batch_size = min(len(m_obs), len(s_obs), len(action_choosen), len(pilco_return))
        m_obs = m_obs[:batch_size]
        s_obs = s_obs[:batch_size]
        action_choosen = action_choosen[:batch_size]
        pilco_return = pilco_return[:batch_size]
        # print("optimization for controller with %i batch size" % batch_size)

        s_obs = np.reshape(s_obs, (batch_size, self.state_dim, self.state_dim))

        self.sess.run(self.train_op, feed_dict={
            self.m_obs: m_obs,
            self.s_obs: s_obs,
            self.ac: action_choosen,
            self.r: pilco_return,
        })

        # reset the episode record
        self.ep_m_obs, self.ep_s_obs, self.ep_pilco_r, self.ep_ac_choosen = [], [], [], []


    def store_transition(self, m_obs, s_obs, pilco_return, action_choosen):
        """
        pilco_return: cumulative reward for a state
        """
        self.ep_m_obs = m_obs
        self.ep_s_obs = s_obs
        self.ep_ac_choosen = action_choosen
        self.ep_pilco_r = pilco_return

    def save_weights(self, path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            save_path = saver.save(self.sess, path)
        print("model saved in file: %s" % save_path)

    def load_weights(self, path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, path)



