import tensorflow as tf
import gym

class PolicyWithValue:
    def __init__(self, env, observations, latent, sess=None):
        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None

        latent = tf.layers.flatten(latent)

        # based on the action space, select what prob distribution type
        self.pdtype = make_pdtype(env.action_space)

        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.1)

        # take an action
        self.action = self.pd.sample()

        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session() 

    def evaluate(self, variables, observation):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        return sess.run(variables, feed_dict)

    def step(self, observation):
        a, v, state, neglogp = self.evaluate([self.action, self.vf, self.state,
            self.neglogp], observation)
        if state.size == 0:
            state = None
        return a, v, state, neglogp 

    def value(self, ob):
        return self.evaluate(self.vf, ob)

    def save(self, save_path):
        pass
