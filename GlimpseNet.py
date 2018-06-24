from Utils import *

class GlimpseNet(object):
    def __init__(self, params, images_ph):
      #See Params.py for explanation on parameters
      self.original_size = params.original_size
      self.num_channels = params.num_channels
      self.sensor_size = params.sensor_size
      self.bandwidth = params.bandwidth
      self.minRadius = params.minRadius
      self.depth = params.depth

      self.hg_size = params.hg_size
      self.hl_size = params.hl_size
      self.g_size = params.g_size
      self.loc_dim = params.loc_dim

      self.images_ph = images_ph
      self.batch_size = params.batch_size

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

        loc = tf.round(((loc + 1) / 2.0) * self.original_size)  # normLoc coordinates are between -1 and 1
        loc = tf.cast(loc, tf.int32)

        img = tf.reshape(self.images_ph,
                         (self.batch_size, self.original_size, self.original_size, self.num_channels))

        # process each image individually
        zooms = []
        for k in range(self.batch_size):
            imgZooms = []
            one_img = img[k, :, :, :]
            max_radius = self.minRadius * (2 ** (self.depth - 1))
            offset = 2 * max_radius

            # pad image with zeros
            one_img = tf.image.pad_to_bounding_box(one_img, offset, offset, \
                                                   max_radius * 4 + self.original_size,
                                                   max_radius * 4 + self.original_size)

            for i in range(self.depth):
                r = int(self.minRadius * (2 ** (i)))

                d_raw = 2 * r
                d = tf.constant(d_raw, shape=[1])
                d = tf.tile(d, [2])
                loc_k = loc[k, :]
                adjusted_loc = offset + loc_k - r
                one_img2 = tf.reshape(one_img, (one_img.get_shape()[0].value, one_img.get_shape()[1].value))

                # crop image to (d x d)
                zoom = tf.slice(one_img2, adjusted_loc, d)

                # resize cropped image to (sensorBandwidth x sensorBandwidth)
                zoom = tf.image.resize_bilinear(tf.reshape(zoom, (1, d_raw, d_raw, 1)),
                                                (self.bandwidth, self.bandwidth))
                zoom = tf.reshape(zoom, (self.bandwidth, self.bandwidth))
                imgZooms.append(zoom)

            zooms.append(tf.stack(imgZooms))

        zooms = tf.stack(zooms)

        return zooms

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