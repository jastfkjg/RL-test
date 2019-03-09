
import tensorflow as tf
# from gpflow import Parameterized, Param, params_as_tensors, settings
import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm
import math

# float_type = settings.dtypes.float_type


# class Reward(Parameterized):
#     def __init__(self):
#         Parameterized.__init__(self)
#
#     @abc.abstractmethod
#     def compute_reward(self, m, s):
#         raise NotImplementedError
class Reward:
    def __init__(self, state):
        self.state = state

    def compute_reward(self):
        raise NotImplementedError

class CartPoleReward(Reward):
    """
    To calculate reward for gym CartPole env
    the state of CartPole: Box(4)
    """

    def __init__(self, state):
        super().__init__(state)
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

    def compute_reward(self, state=None):
        """
        :param state: type: np.ndarray
        :return: reward
        """
        if state:
            s = state
        else:
            s = self.state

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
        :return: the mean of the reward
        """
        # m_x, m_x_dot, m_theta, m_theta_dot = m_state
        # s_x, s_x_dot, s_theta, s_theta_dot = s_state

        # the function to be integrated
        def f(x, x_dot, theta, theta_dot):
            state = (x, x_dot, theta, theta_dot)
            prob_density = 1.0
            for s, m_x, s_x in zip(state, m_state, s_state):
                prob_density = prob_density * norm.pdf(s, loc=m_x, scale=s_x)
            reward_density = prob_density * self.compute_reward(state)
            return reward_density

        # reward = integrate.quad(, -np.inf, np.inf)    # quatruple
        reward = integrate.nquad(f, [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])

        return reward


class PendulumReward(Reward):
    """
    Calculate reward for gym Pendulum env
    the state of Pendulum is Box(2)
    """
    def __init__(self, state):
        super().__init__(state)
        self.max_torque = 2.

    def compute_reward(self, u, state=None):
        if state:
            s = state
        else:
            s = self.state

        th, thdot = s

        u = np.clip(u, -self.max_torque, self.max_torque)[0]

        costs = self.angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        return -costs

    def angle_normalize(self, x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)

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
