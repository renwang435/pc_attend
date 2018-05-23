from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

distributions = tf.contrib.distributions

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)

  return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)

    return tf.Variable(initial)

#We map past locations (mean_arr) to some distribution over potential locations at the current time step
#and return the log likelihood of these possible samples
def loglikelihood(mean_arr, sampled_arr, sigma):
    mu = tf.stack(mean_arr)  # mu = [timesteps, batch_size, loc_dim]
    sampled = tf.stack(sampled_arr)  # same shape as mu
    gaussian = distributions.Normal(mu, sigma)
    logll = gaussian.log_prob(sampled)  # [timesteps, batch_size, loc_dim]
    logll = tf.reduce_sum(logll, 2)
    logll = tf.transpose(logll)  # [batch_size, timesteps]

    return logll
