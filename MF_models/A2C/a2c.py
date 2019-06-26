import time
import tensorflow as tf
import numpy as np
from tensorflow import losses

from policies import build_policy

class Model:
    def __init__(self, policy, env, nsteps, ent_coef=0.01, vf_coef=0.5,
            lr=1e-3, alpha=0.99, epsilon=1e-5):
        sess = tf.Session()
        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model for sampling
            step_model = policy(nenvs, 1, sess)
            # train_model to train our network
            train_model = policy(nbatch, nsteps, sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        # total_loss = Policy gradient loss - entropy * entropy coeff + value coeff * value loss
        
        # policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = -log(pi(a|s)) * Adv(s, a)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        #entropy is used to improve exploration by limiting the premature
        # convergence to suboptimal policy
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # update params using loss
        # 1. get model params
        params = find_trainable_variables("a2c_model")

        # 2. calculate the gradients
        grads = tf.gradients(loss, params)
        grads = list(zip(grads, params))

        # 3. make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha,
                epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        def train(obs, states, rewards, mask, actions, values):
            # we calculate advantage A(s, a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values
            #for step in range(len(obs)):
            td_map = {train_model.X:obs, A:actions, ADV: advs, R: rewards, LR:lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                    [pg_loss, vf_loss, entropy, _train], td_map)
            return policy_loss, value_loss, policy_entropy

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state

    def save(self, path):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path)
        print("model saved in file: %s" % save_path)

    def load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        print("model load successfully.")
                

def learn(network, env, nsteps=5, total_timesteps=1e6, vf_coef=0.5,
        ent_coef=0.01, lr=1e-3, epsilon=1e-5, alpha=0.99, gamma=0.99):
    """
    main entrypoint for A2C. train a policy with given network using A2C.
    """
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef,
            vf_coef=vf_coef, lr=lr, alpha=alpha, epsilon=epsilon,
            total_timesteps = total_timesteps)
    if load_path is not None:
        model.load(load_path)

    # instantiate the runner object
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    nbatch = nenvs * nsteps

    for update in range(1, total_timesteps//nbatch + 1):

        # get mini batch of experiences
        obs, states, rewards, masks, actions, values = runner.run()

        policy_loss, value_loss, policy_entropy = model.train(obs, states,
                rewards, masks, actions, values)


    return model

    


