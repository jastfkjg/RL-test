import numpy as np
import tensorflow as tf


class DiagGaussian():

    def __init__(self, mean, std=None, logstd=None):
        self.mean = mean
        if logstd is not None and std is not None:
            self.std = std
            self.logstd = logstd
            
        elif std is not None:
            self.std = std
            self.logstd = tf.log(std)
        elif logstd is not None:
            self.logstd = logstd
            self.std = tf.exp(mean * 0.0 + logstd)


    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
                + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], dtype=tf.float64) \
                + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussian)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean\
                - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=-1)
    def entropy(self):
        return tf.reduce_sum(self.logstd + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean), dtype=tf.float64)
