from Utils import *

class LocNet(object):
    #See Params.py for an explanation on parameters
    def __init__(self, params):
        self.loc_dim = params.loc_dim
        self.input_dim = params.cell_output_size
        self.loc_std = params.loc_std
        self._sampling = True

        self.init_weights()

    def init_weights(self):
        self.w = weight_variable((self.input_dim, self.loc_dim))
        self.b = bias_variable((self.loc_dim,))

    def __call__(self, input):
        #Take some hidden representation from the Core network and return the next location
        mean = tf.clip_by_value(tf.nn.xw_plus_b(input, self.w, self.b), -1., 1.)
        mean = tf.stop_gradient(mean)   #Need to stop gradients as we train the Location network exclusively with REINFORCE

        if self._sampling:
          loc = mean + tf.random_normal(
              (tf.shape(input)[0], self.loc_dim), stddev=self.loc_std)
          loc = tf.clip_by_value(loc, -1., 1.)
        else:
          loc = mean

        loc = tf.stop_gradient(loc)

        return loc, mean

    #Define a sampling property from when we want to train w/ REINFORCE or evaluate from a set of samples
    @property
    def sampling(self):
      return self._sampling

    @sampling.setter
    def sampling(self, sampling):
      self._sampling = sampling
