import os
from os import path
import gym
# from gpflow import Parameterized, Param, params_as_tensors, settings
import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm
import math
import mujoco_py
import tensorflow_probability as tfp
import tensorflow as tf

# try:
# import mujoco_py
# except:
    # print("you need to install mujoco_py")

# float_type = settings.dtypes.float_type
# class Reward(Parameterized):
#     def __init__(self):
#         Parameterized.__init__(self)
#
#     @abc.abstractmethod
#     def compute_reward(self, m, s):
#         raise NotImplementedError
class Reward:
    def __init__(self):
        self.tfd = tfp.distributions

    def compute_reward(self):
        raise NotImplementedError

class CartPoleReward(Reward):
    """
    To calculate reward for gym CartPole env
    the state of CartPole: Box(4)
    """

    def __init__(self):
        super().__init__()
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.with_action = False

    def compute_reward(self, state):
        """
        :param state: type: np.ndarray
        :return: reward
        """
        s = state
        x, x_dot, theta, theta_dot = s
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        if done:
            # if done, the reward is 0.0
            return 0.0
        else:
            # if not, the reward is 1.0
            return 1.0

    def compute_gaussian_reward(self, m_state, s_state):
        """
        :param m_state: mean of state        np.array
        :param s_state: variance of state    np.array
        :return: the mean of the reward, whether it's done
        """
        # m_x, m_x_dot, m_theta, m_theta_dot = m_state
        # s_x, s_x_dot, s_theta, s_theta_dot = s_state

        # the function to be integrated
        def f(u, m_state, s_state):
            # print(m_state, s_state)
            # proba = norm.pdf(u1, loc=0, scale=1) * norm.pdf(u2, 0, 1) * norm.pdf(u3, 0, 1) * norm.pdf(u4, 0, 1)
            # sqrt_s = np.array(list(map(math.sqrt, s_state)))
            proba = norm.pdf(u, loc=0, scale=1)
            batched_eye = np.eye(s_state.shape[0])
            try:
                L = np.linalg.cholesky(s_state + 0.01 * batched_eye)
            except np.linalg.linalg.LinAlgError:
                print('matrix is singular.')
                return 0.
            u = np.array([u] * s_state.shape[0])
            reward_density = proba * self.compute_reward(m_state + np.dot(L, u))
            return reward_density

        # def f(x, x_dot, theta, theta_dot):
        #     state = (x, x_dot, theta, theta_dot)
        #     prob_density = 1.0
        #     for s, m_x, s_x in zip(state, m_state, s_state):
        #         prob_density = prob_density * norm.pdf(s, loc=m_x, scale=s_x)
        #     reward_density = prob_density * self.compute_reward(state)
        #     return reward_density

        reward = integrate.quad(f, -20., 20., args=(m_state, s_state))[0]
        # too complicated to calculate
        # reward = integrate.nquad(f, [[-5., 5.], [-5., 5.], [-5., 5.], [-5., 5.]])[0]

        # normally when we calculate gaussian reward, the expectation of result is always > 0.,
        # if reward < 0.2, we consider it's done.
        if reward < 0.2:
            done = True
        else:
            done = False
        return reward, done


class ContinuousMountainCarReward(Reward):
    def __init__(self):
        super().__init__()
        self.goal_position = 0.45
        self.with_action = True

    def compute_reward(self, state, action):
        position = state[0]
        done = bool(position >= self.goal_position)
        reward = 0
        if done:
            reward = 100.0
        reward -= math.pow(action[0], 2) * 0.1
        return reward

    def compute_gaussian_reward(self, m_state, s_state, m_action):
        def f(u, m_state, s_state):
            # proba = norm.pdf(u1, loc=0, scale=1) * norm.pdf(u2, 0, 1) * norm.pdf(u3, 0, 1) * norm.pdf(u4, 0, 1)
            # sqrt_s = np.array(list(map(math.sqrt, s_state)))
            proba = norm.pdf(u, loc=0, scale=1)
            batched_eye = np.eye(s_state.shape[0])
            try:
                L = np.linalg.cholesky(s_state + 0.01 * batched_eye)
            except np.linalg.linalg.LinAlgError:
                print('matrix is singular.')
                return 0.
            u = np.array([u] * s_state.shape[0])
            reward_density = proba * self.compute_reward(m_state + np.dot(L, u), m_action)
            return reward_density

        reward = integrate.quad(f, -20., 20., args=(m_state, s_state))[0]
        if reward > 60.0:
            done = True
        else:
            done = False
        return reward, done

class PendulumReward(Reward):
    """
    Calculate reward for gym Pendulum env
    the state of Pendulum is Box(2)
    """
    def __init__(self):
        super().__init__()
        self.with_action = True
        self.max_torque = 2.

    def clip(self, th):
        if th > 1:
            th = 1
        elif th < -1:
            th = -1
        return th

    def compute_reward(self, state, action):
        cos_th, sin_th, thdot = state
        cos_th, sin_th = self.clip(cos_th), self.clip(sin_th)
        # calculate th by cos_th and sin_th
        # TODO
        # print("cos:", cos_th)
        th = np.arcsin(sin_th)
        if np.cos(th) - cos_th > 0.01:
            th = math.pi - th if sin_th >= 0 else -math.pi - th
        action = np.clip(action, -self.max_torque, self.max_torque)[0]
        costs = self.angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (action ** 2)
        return -costs

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def compute_gaussian_reward(self, m_state, s_state, m_action, sample_num=20):
        # TODO: find a better way to calculate expected reward, deal with cholesky decomp prob
        # maybe do sample ourselves
        # add noise to solve Cholesky decomposition prob
        # print("type of s_x for r: ----- ", type(s_state[0][0]), type(m_state[0]), type(m_action[0]))
        e, v = tf.linalg.eigh(s_state)
        eps = 1e-5
        e = tf.maximum(e, eps)
        s_state_pos_def = tf.matmul(tf.matmul(v, tf.diag(e)), tf.transpose(v))
        # batched_eye = np.eye(s_state.shape[0])
        # batched_eye = np.random.rand(s.shape[0], s.shape[0])
        # s_with_noise = s_state + 0.01 * batched_eye
        try:
            dist_obs = self.tfd.MultivariateNormalFullCovariance(loc=m_state, covariance_matrix=s_state_pos_def)
            states = dist_obs.sample([sample_num])   # [sample_num, state_dim]
            with tf.Session() as sess:
                states = sess.run(states)
        except tf.errors.InvalidArgumentError:
            print("Cholesky decomposition failed. In this case, we only take diag element of obs variance")
            dist_obs = self.tfd.MultivariateNormalDiag(loc=m_state, scale_diag=np.diag(s_state))
            states = dist_obs.sample([sample_num])   # [sample_num, state_dim]
            with tf.Session() as sess:
                states = sess.run(states)

        total_reward = 0.
        for state in states:
            total_reward += self.compute_reward(state, m_action)
        reward = total_reward / sample_num

        done = False
        return reward, done

        # # the function to be integrated
        # batched_eye = np.eye(s_state.shape[0])
        # try:
        #     L = np.linalg.cholesky(s_state + 0.1 * batched_eye)
        # except np.linalg.linalg.LinAlgError:
        #     print("cholesky demcomposition failed. matrix is singular.")
        #     return 0., True
        #
        # def f(u):
        #     # proba = norm.pdf(u1, loc=0, scale=1) * norm.pdf(u2, 0, 1) * norm.pdf(u3, 0, 1) * norm.pdf(u4, 0, 1)
        #     # sqrt_s = np.array(list(map(math.sqrt, s_state)))
        #     proba = norm.pdf(u, loc=0, scale=1)
        #     # batched_eye = np.eye(s_state.shape[0])
        #     # try:
        #     #     L = np.linalg.cholesky(s_state + 0.01 * batched_eye)
        #     # except np.linalg.linalg.LinAlgError:
        #     #     print('matrix is singular.')
        #     #     return 0.
        #     u = np.array([u] * s_state.shape[0])
        #     reward_density = proba * self.compute_reward(m_state + np.dot(L, u), m_action)
        #     return reward_density
        #
        # reward = integrate.quad(f, -10., 10.)[0]
        # # too complicated to calculate  , args=(m_state, s_state, m_action)
        # # reward = integrate.nquad(f, [[-5., 5.], [-5., 5.], [-5., 5.], [-5., 5.]])[0]
        #
        # # never done
        # done = False
        #
        # return reward, done

class InvertedPendulumReward(Reward):
    """
    Reward function of inverted_pendulum in mujoco
    """
    def __init__(self):
        super().__init__()
        self.with_action = False

    def compute_reward(self, state):
        s = state
        notdone = np.isfinite(s).all() and (np.abs(s[1]) <= .2)
        done = not notdone
        if done:
            return 0.0
        else:
            return 1.0

    def compute_gaussian_reward(self, m_state, s_state, sample_num=20):
        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                e, v = tf.linalg.eigh(s_state)
                eps = 1e-5
                e = tf.maximum(e, eps)
                s_state_pos_def = tf.matmul(tf.matmul(v, tf.diag(e)), tf.transpose(v))
                # batched_eye = np.eye(s_state.shape[0])
                # batched_eye = np.random.rand(s.shape[0], s.shape[0])
                # s_with_noise = s_state + 0.01 * batched_eye
                try:
                    dist_obs = self.tfd.MultivariateNormalFullCovariance(loc=m_state, covariance_matrix=s_state_pos_def)
                    states = dist_obs.sample([sample_num])   # [sample_num, state_dim]
                    # with tf.Session() as sess:
                    states = sess.run(states)
                except tf.errors.InvalidArgumentError:
                    print("Cholesky decomposition failed in reward calculating. In this case, we only take diag element of obs variance")
                    dist_obs = self.tfd.MultivariateNormalDiag(loc=m_state, scale_diag=abs(np.diag(s_state)))
                    states = dist_obs.sample([sample_num])   # [sample_num, state_dim]
                    # with tf.Session() as sess:
                    states = sess.run(states)

        total_reward = 0.
        for state in states:
            total_reward += self.compute_reward(state)
        reward = total_reward / sample_num

        if reward < 0.4:
            done = True
        else:
            done = False

        return reward, done

class HumanoidReward(Reward):
    # see https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoid.py
    def __init__(self, model_path):
        super().__init__()
        self.with_action = True
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("file %s does not exist" % fullpath)

        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)

    def mass_center(self, model, sim):
        mass = np.expand_dims(model.body_mass, 1)
        xpos = sim.data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

    def compute_reward(self, state, action):
        pos_before = self.mass_center(self.model, self.sim)
        pass

    def mass_center(self, ):
        pass

# np.random.RandomState
# - what is the reward function for gym games?


# # we don't need to use the special Reward function in PILCO
# class ExponentialReward(Reward):
#     def __init__(self, state_dim, W=None, t=None):
#         Reward.__init__(self)
#         self.state_dim = state_dim
#         if W is not None:                                         # W ?
#             self.W = Param(np.reshape(W, (state_dim, state_dim)), trainable=False)
#         else:
#             self.W = Param(np.ones((state_dim, state_dim)), trainable=False)
#         if t is not None:
#             self.t = Param(np.reshape(t, (1, state_dim)), trainable=False)     # target
#         else:
#             self.t = Param(np.zeros((1, state_dim)), trainable=False)

    # @params_as_tensors
    # def compute_reward(self, m, s):
    #     '''
    #     Reward function, calculating mean and variance of rewards, given
    #     mean and variance of state distribution, along with the target State
    #     and a weight matrix.
    #     Input m : [1, k]
    #     Input s : [k, k]
    #
    #     Output M : [1, 1]
    #     Output S  : [1, 1]
    #     '''
    #     # TODO: Clean up this
    #
    #     SW = s @ self.W
    #
    #     iSpW = tf.transpose(
    #             tf.matrix_solve( (tf.eye(self.state_dim, dtype=float_type) + SW),
    #             tf.transpose(self.W), adjoint=True))
    #
    #     muR = tf.exp(-(m-self.t) @  iSpW @ tf.transpose(m-self.t)/2) / \
    #             tf.sqrt( tf.linalg.det(tf.eye(self.state_dim, dtype=float_type) + SW) )
    #
    #     i2SpW = tf.transpose(
    #             tf.matrix_solve( (tf.eye(self.state_dim, dtype=float_type) + 2*SW),
    #             tf.transpose(self.W), adjoint=True))
    #
    #     r2 =  tf.exp(-(m-self.t) @ i2SpW @ tf.transpose(m-self.t)) / \
    #             tf.sqrt( tf.linalg.det(tf.eye(self.state_dim, dtype=float_type) + 2*SW) )
    #
    #     sR = r2 - muR @ muR
    #     muR.set_shape([1, 1])
    #     sR.set_shape([1, 1])
    #     return muR, sR
