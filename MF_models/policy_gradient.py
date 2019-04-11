import numpy as np 
import tensorflow as tf
from tensorflow import losses

from distributions import DiagGaussianPd

np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient(object):
    """PolicyGradient"""
    def __init__(self, action_dim, state_dim, learning_rate=0.01, reward_decay=0.95, output_graph=False, discrete_ac=False):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.lr = learning_rate
        self.gamma = reward_decay
        self.discrete_ac = discrete_ac

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        tf.reset_default_graph()

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope("PG_inputs"):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.state_dim], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # fc1
        layer = tf.layers.dense(inputs=self.tf_obs, units=10, activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1), name='fc1')

        # fc2
        self.all_act = tf.layers.dense(inputs=layer, units=self.action_dim, activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1), name='fc2')

        self.all_act_prob = tf.nn.softmax(self.all_act, name="act_prob")

        with tf.name_scope('PG_loss'):
            # to maximize total reward (log(pi)*R) is to minimize -(log(pi)*R)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.all_act, labels=self.tf_acts)
            # or
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

        with tf.name_scope('PG_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):

        if self.discrete_ac:
            prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
            # print("proba:", prob_weights)
            action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        else:

            action = self.sess.run(self.all_act)
            # action = np.reshape(action, ())

        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: self.ep_obs,  # shape=[None, n_obs]
            self.tf_acts: self.ep_as,  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,    # shape=[None, ]
            })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data

        # return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)  # return an array of zero with the same shape and type as ep_rs
        running_add = 0

        for t in reversed(range(len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        if np.std(discounted_ep_rs) != 0:
            discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save_model(self, path):
        # save model weights
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path)
        print("model saved in file: %s" % save_path)

    def load_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        print("model load successfully.")

class ActorCritic:
    def __init__(self, action_dim, state_dim,lr=0.001, ent_coef=0.01, value_coef=0.5, reward_decay=0.95, output_graph=False):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.lr = lr
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        # self.actor_lr = actor_lr
        # self.critic_lr = critic_lr
        self.gamma = reward_decay
        self.output_graph = output_graph

        tf.reset_default_graph()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

    
        with tf.name_scope("inputs"):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.state_dim], name="observations")
            self.tf_ac = tf.placeholder(tf.float32, [None, self.action_dim], name="actions")
            self.advantage = tf.placeholder(tf.float32, [None, ], name="advantage")
            self.R = tf.placeholder(tf.float32, [None, ], name="return")
        # fc1
        layer = tf.layers.dense(inputs=self.tf_obs, units=10, activation=tf.nn.tanh,
                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                bias_initializer=tf.constant_initializer(0.1), name='actor_fc1')

        # fc2
        mean_all_act = tf.layers.dense(inputs=layer, units=self.action_dim, activation=None,
                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                bias_initializer=tf.constant_initializer(0.1), name='actor_fc2')
        
        logstd_act = tf.get_variable(name="logstd", shape=[1, self.action_dim],
                initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean_all_act, mean_all_act * 0.0 + logstd_act], axis=1)
        self.pd = DiagGaussianPd(pdparam)

        self.action = self.pd.sample()
        self.neglogp = self.pd.neglogp(self.action)
        
        # for critic network, we share the first layer with actor
        value = tf.layers.dense(input=layer, units=1, activation=None,
                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                bias_initializer=tf.constant_initializer(0.1), name="critic_fc2")

        with tf.name_scope("Loss"):
            # Total loss = Policy loss - entropy * ent_coef + value loss * value_coef
            pg_loss = tf.reduce_mean(self.advantage * self.neglogp)
            value_loss = losses.mean_squared_error(tf.squeeze(value, self.R))
            entropy = tf.reduce_mean(self.pd.entropy())
            
            # total loss
            loss = pg_loss - entropy * self.ent_coef + value_loss * self.value_coef
            
        with tf.name_scope("Train"):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)


        def step(self, observation):
            action = self.sess.run(self.action, feed_dict={self.tf_obs: observation[np.newaxis, :]})
            return action


        def learn(self, obs, actions, rewards, values):
            # calculate adv = reward - V(s)
            # reward = r + yV(s')
            advs = rewards - values
            # value = self.sess.run(self.value, feed_dict={self.obs: state})

            td_map = {self.tf_obs: obs, self.tf_ac: actions, self.advantage: advs, self.R: rewards}

            policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                    [pg_loss, value_loss, entropy, train_op],
                    feed_dict=td_map
                    )
            return policy_loss, value_loss, policy_entropy

        self.step = step
        self.learn = learn


    def save_model(self, path):
        # save model weights
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path)
        print("model saved in file: %s" % save_path)

    def load_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        print("model load successfully.")





