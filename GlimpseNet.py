from Utils import *

class GlimpseNet(object):
    def __init__(self, params, images_ph):
      #See Params.py for explanation on parameters
      self.original_size = params.original_size
      self.num_channels = params.num_channels
      self.sensor_size = params.sensor_size
      self.win_size = params.win_size
      self.minRadius = params.minRadius
      self.depth = params.depth

      self.hg_size = params.hg_size
      self.hl_size = params.hl_size
      self.g_size = params.g_size
      self.loc_dim = params.loc_dim

      self.images_ph = images_ph

      self.init_weights()

    def init_weights(self):

      #Initialize networks mapping retina representation and location to some hidden state
      self.w_g0 = weight_variable((self.sensor_size, self.hg_size))
      self.b_g0 = bias_variable((self.hg_size,))
      self.w_l0 = weight_variable((self.loc_dim, self.hl_size))
      self.b_l0 = bias_variable((self.hl_size,))

      self.w_g1 = weight_variable((self.hg_size, self.g_size))
      self.b_g1 = bias_variable((self.g_size,))
      self.w_l1 = weight_variable((self.hl_size, self.g_size))
      self.b_l1 = weight_variable((self.g_size,))

    def get_glimpse(self, loc):

      imgs = tf.reshape(self.images_ph, [
          tf.shape(self.images_ph)[0], self.original_size, self.original_size,
          self.num_channels
      ])

      #Extract a set of glimpses (both normalized and centered, areas of non-overlap between window and image are
      #filled with RANDOM noise
      glimpse_imgs = tf.image.extract_glimpse(imgs,
                                              [self.win_size, self.win_size], loc)

      #Return a vector with glimpses
      glimpse_imgs = tf.reshape(glimpse_imgs, [
          tf.shape(loc)[0], self.win_size * self.win_size * self.num_channels
      ])

      return glimpse_imgs

    def __call__(self, loc):
      glimpse_input = self.get_glimpse(loc)
      glimpse_input = tf.reshape(glimpse_input,
                                 (tf.shape(loc)[0], self.sensor_size))

      #See Section 4
      #Mapping glimpse to some hidden representation
      g = tf.nn.relu(tf.nn.xw_plus_b(glimpse_input, self.w_g0, self.b_g0))
      g = tf.nn.xw_plus_b(g, self.w_g1, self.b_g1)

      #Mapping location to some hidden representation
      l = tf.nn.relu(tf.nn.xw_plus_b(loc, self.w_l0, self.b_l0))
      l = tf.nn.xw_plus_b(l, self.w_l1, self.b_l1)

      #Glimpse network combines glimpse sensor and location network encodings
      g = tf.nn.relu(g + l)

      return g