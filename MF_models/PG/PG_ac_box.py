"""
PolicyGradient for Continuous action space
"""
import numpy as np
import tensorflow as tf

# from .. import distributions
from distributions import DiagGaussianPd

from runner import Runner

class PolicyGradient:
    """
    Policy gradient for continuous action space
    """
    def __init__(self, action_dim, state_dim, *, lr=0.001, ent_coef=0.01, reward_decay=0.95, output_graph=False):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.lr = lr
        self.ent_coef = ent_coef
        # self.actor_lr = actor_lr
        # self.critic_lr = critic_lr
        self.gamma = reward_decay
        self.output_graph = output_graph

        tf.reset_default_graph()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        with tf.name_scope("inputs"):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.state_dim], name="observations")
            self.tf_ac = tf.placeholder(tf.float32, [None, self.action_dim], name="actions")
            self.advantage = tf.placeholder(tf.float32, [None, ], name="advantage")
            # self.R = tf.placeholder(tf.float32, [None, ], name="return")
        # fc1
        layer = tf.layers.dense(inputs=self.tf_obs, units=10, activation=tf.nn.tanh,
                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                bias_initializer=tf.constant_initializer(0.1), name='actor_fc1')

        # fc2
        mean_all_act = tf.layers.dense(inputs=layer, units=self.action_dim, activation=None,
                                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                       bias_initializer=tf.constant_initializer(0.1), name='actor_fc2')

        logstd_act = tf.get_variable(name="logstd", shape=[1, self.action_dim], \
                                     initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean_all_act, mean_all_act * 0.0 + logstd_act], axis=1)
        self.pd = DiagGaussianPd(pdparam)

        self.action = self.pd.sample()
        self.neglogp = self.pd.neglogp(self.action)

        with tf.name_scope("Loss"):
            # Total loss = Policy loss - entropy * ent_coef
            pg_loss = tf.reduce_mean(self.advantage * self.neglogp)

            entropy = tf.reduce_mean(self.pd.entropy())

            # total loss
            loss = pg_loss - entropy * ent_coef

        with tf.name_scope("Train"):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        def step(observation):
            action = self.sess.run(self.action, feed_dict={self.tf_obs: observation[np.newaxis, :]})   #
            return np.squeeze(action, 0)

        def train(obs, actions, rewards):
            # calculate adv = reward - V(s)
            # reward = r + yV(s')
            advs = rewards

            td_map = {self.tf_obs: obs, self.tf_ac: actions, self.advantage: advs}

            policy_loss, policy_entropy, _ = self.sess.run(
                [pg_loss, entropy, train_op],
                feed_dict=td_map
            )
            return policy_loss, policy_entropy

        self.step = step
        self.train = train

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def save_model(self, path):
        # save model weights
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path)
        print("model saved in file: %s" % save_path)

    def load_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        print("model load successfully.")


def learn(env, nsteps=500, nepisodes=200, ent_coef=0.01, lr=0.001, gamma=0.95, log_interval=10,\
          load_path=None):
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    model = PolicyGradient(action_dim, state_dim, lr=lr, ent_coef=ent_coef, reward_decay=gamma)

    if load_path is not None:
        model.load_model(load_path)

    # instantiate the runner object
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    for update in range(nepisodes):
        # get mini batch of experiences
        obs, rewards, actions = runner.run()

        policy_loss, policy_entropy = model.train(obs, actions, rewards)

        if update % log_interval == 0 or update == 1:
            # calculate if value function is a good predictor of return
            # ev = explained_variance(values, rewards)
            # TODO
            print("the " + str(update) + "batch)
    return model


