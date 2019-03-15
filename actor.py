import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp 

np.random.seed(1)
tf.set_random_seed(1)

class Actor():

	def __init__(self, action_dim, state_dim, learning_rate):
		self.action_dim = action_dim
		self.state_dim = state_dim
		self.learning_rate = learning_rate

		self._build_net()

		self.sess = tf.Session()

		self.sess.run(tf.global_variables_initializer())


	def _build_net(self):
		with tf.variable_scope("inputs"):
			self.obs = tf.placeholder(tf.float32, [None, self.state_dim], name="observations")
			# for cartpole-v0, dim of action is 1.
			self.acts = tf.placeholder(tf.int32, [None, self.action_dim], name="actions_num")
			# we use the pilco to estimate the future reward
			self.vt = tf.placeholder(tf.float32, [None, ], name="actions_value") 

		# fc1
		layer = tf.layers.dense(inputs=self.obs, units=10, activation=tf.nn.tanh,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3), 
			bias_initializer=tf.constant_initializer(0.1), name='fc1')

		# fc2
		all_act = tf.layers.dense(inputs=layer, units=self.action_dim, activation=None,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.1), name='fc2')

		self.all_act_prob = tf.nn.softmax(all_act, name="act_prob")

		with tf.name_scope('loss'):
			# minimize -(log(pi)*R)
			neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.acts)

			loss = tf.reduce_mean(neg_log_prob * self.vt)

		with tf.variable_scope('train'):
			self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

	def compute_action(self, m_ob, s_ob):
		"""
		# in: mean, variance of the state
		# out: mean, variance of the action; inv(s)*cov(state, action)
		# apply the re-param ????
		"""
		
		prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.obs: m_ob[np.newaxis, :]})
		action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
		return action

	def learn(self, pilco_reward):

		self.sess.run(self.train_op, feed_dict={
			self.obs: np.vstack(self.ep_obs),
			self.acts: np.array(self.ep_acts),
			self.vt: pilco_reward,
			})

	def store_transition(self, s, a, r):
		self.ep_obs.append(s)
		self.ep_acts.append(a)
		self.ep_rs.append(r)

	def save_model(self, path):
		# save model weights
		saver = tf.train.Saver()
		save_path = saver.save(self.sess, path)
		print("model saved in file: %s" % save_path)


class Actor():

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

		with tf.variable_scope("Inputs"):
			self.m_obs = tf.placeholder(tf.float64, [None, self.state_dim], name="mean_observations")
			self.s_obs = tf.placeholder(tf.float64, [None, self.state_dim, self.state_dim], name="variance_observation")

		with tf.variable_scope("Optim_inputs"):
			# the type of action in gym env
			self.ac = tf.placeholder(tf.float64, [None, self.action_dim], name="action_choosen")
			self.r = tf.placeholder(tf.float64, [None, ], name="return_from_pilco")

		self.m_obs = tf.expand_dims(self.m_obs, axis=2)    # [None, state_dim, 1]

		# consider to concatenate m_obs and s_obs
		self.obs = tf.concat([self.m_obs, self.s_obs], 2)   # [None, state_dim, state_dim + 1]

		# How to choose the network architecture ?

		# fc1
		layer = tf.layers.dense(inputs=self.obs, units=10, activation=tf.nn.tanh,
								kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
								bias_initializer=tf.constant_initializer(0.1), name='fc1')

		# fc2
		self.m_ac = tf.layers.dense(inputs=layer, units=self.action_dim, activation=None,
								kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
								bias_initializer=tf.constant_initializer(0.1), name='fc2')

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

		self.s_ac = tf.layers.dense(inputs=layer, units=self.action_dim, activation=None,
								kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
								bias_initializer=tf.constant_initializer(0.1), name='fc3')
		self.s_ac = tf.diag(self.s_ac) # what if s_ac: [None, action_dim] 

		# V: How to calculate input-output covariance
		# V: input-output covariance TODO
		# 1. for NN, it's too complicate to calculate
		# 2. gibbs sampling to calculate E(state*action) - m_state*m_action
		self.V = self.weight  # shape: [state_dim, action_dim]

		# distribution of action, we need to change to multi-variate gaussian dist
		self.dist = self.tfd.Normal(loc=self.m_ac, scale=self.s_ac)

		with tf.variable_scope("loss"):
			# policy gradient: minimize -(log(pi)*r)   #TODO: discrete case
			# whether to use cholesky decomposition ?
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
		ac = self.sample_action(m_ac,
								s_ac)  # a list of n num, n is the action dim, for discrete action space, we only need one

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



