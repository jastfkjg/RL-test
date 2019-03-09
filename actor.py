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

class LinearActor():
	"""
	To handle the input, output uncertainty, 
	we first test a simple linear policy.
	"""
	def __init__(self, action_dim, state_dim, learning_rate):
		self.action_dim = action_dim
		self.state_dim = state_dim
		self.learning_rate = learning_rate

		self.ep_m_obs, self.ep_s_obs, self.ep_ac_choosen, self.ep_pilco_r = [], [], [], []

		self.tfd = tfp.distributions

		self.weight = tf.Variable(tf.random_normal([self.state_dim, self.action_dim]))
		self.bias = tf.Variable(tf.random_normal([self.action_dim]))

		self._build_net()

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def _build_net(self, dertermine_state=False):   # to do !

		# We may get a state as input instead of mean
		# and variance of state as input
		#
		with tf.variable_scope("linear_inputs"):
			self.m_obs = tf.placeholder(tf.float32, [None, self.state_dim], name="mean_observations")
			self.s_obs = tf.placeholder(tf.float32, [None, self.state_dim, self.state_dim], name="variance_observation")

		with tf.variable_scope("linear_optim_inputs"):
			# the type of action in gym env
			self.ac = tf.placeholder(tf.float32, [None, self.action_dim], name="action_choosen")
			self.r = tf.placeholder(tf.float32, [None, ], name="return_from_pilco")

		self.m_ac = tf.add(tf.matmul(self.m_obs, self.weight), self.bias)   # tf.add
		weight = tf.reshape(self.weight, [1, self.state_dim, self.action_dim])
		# weight = tf.tile(weight, [self.s_obs.shape[0], 1, 1])

		self.s_ac = tf.transpose(weight, perm=[0, 2, 1]) @ self.s_obs @ weight
		# V ?
		self.V = tf.transpose(self.weight)

		# distribution of action
		self.dist = self.tfd.Normal(loc=self.m_ac, scale=self.s_ac)



		with tf.variable_scope("loss"):
			# policy gradient: minimize -(log(pi)*r)
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
		m_ac = self.sess.run(self.m_ac, feed_dict={self.m_obs: m})
		s_ac = self.sess.run(self.s_ac, feed_dict={self.s_obs: s})
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
		m_ac, s_ac, V = self.compute_action(m, tf.zeros([self.state_dim, self.state_dim]))
		return m_ac

	def sample_action(self, m, s):
		"""
		:param m: mean of action
		:param s: variance of action
		:return: a random sample from the action distribution
		"""
		dist = self.tfd.Normal(loc=m, scale=s)
		# sample one action ?
		a = dist.sample([1])

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

		self.sess.run(self.train_op, feed_dict={
			self.m_obs: m_obs,
			self.s_obs: s_obs,
			self.ac: action_choosen,
			self.r: pilco_return,
			})

		# reset the episode record 
		self.ep_m_obs, self.ep_s_obs, self.ep_pilco_r, self.ep_ac_choosen = [], [], [], []

	def store_transition(self, m_obs, s_obs, pilco_return, action_choosen=None):
		"""
		pilco_return: cumulative reward for a state
		"""
		self.ep_m_obs.append(m_obs)
		self.ep_s_obs.append(s_obs)
		if action_choosen:
			self.ep_ac_choosen.append(action_choosen)
		self.ep_pilco_r.append(pilco_return)

	def save_weights(self, path):
		saver = tf.train.Saver()
		save_path = saver.save(self.sess, path)
		print("model saved in file: %s" % save_path)


