from Utils import *
import sys

class GlimpseNet(object):
    def __init__(self, params, images_ph):
        #See Params.py for explanation on parameters
        self.num_points_per_pc = params.num_points_per_pc
        self.num_points_per_glimpse = params.num_points_per_glimpse
        self.half_extent = params.cube_half_extent

        self.sensor_size = params.sensor_size
        self.hg_size = params.hg_size
        self.hl_size = params.hl_size
        self.g_size = params.g_size
        self.loc_dim = params.loc_dim

        self.images_ph = images_ph
        self.batch_size = params.batch_size

        self.init_weights()

    def init_weights(self):
        #Initialize networks mapping retina representation and location to some hidden state
        self.w_g0 = weight_variable((self.num_points_per_glimpse * self.loc_dim, self.hg_size))
        self.b_g0 = bias_variable((self.hg_size,))
        self.w_l0 = weight_variable((self.loc_dim, self.hl_size))
        self.b_l0 = bias_variable((self.hl_size,))

        self.w_g1 = weight_variable((self.hg_size, self.g_size))
        self.b_g1 = bias_variable((self.g_size,))
        self.w_l1 = weight_variable((self.hl_size, self.g_size))
        self.b_l1 = weight_variable((self.g_size,))

    def true_fn(self, reduced_pc_z):
        num_rand_samples = self.num_points_per_glimpse - tf.cast(tf.size(reduced_pc_z) / self.loc_dim, tf.int64)
        rand_samples = tf.random_uniform((num_rand_samples, 3), minval=-1.001, maxval=1.001)
        reduced_pc_z = tf.concat([reduced_pc_z, rand_samples], 0)

        return reduced_pc_z

    def false_fn(self, reduced_pc_z):
        len_tensor = tf.cast(tf.size(reduced_pc_z) / self.loc_dim, tf.int64)
        idx = tf.random_shuffle(tf.range(len_tensor))[:100]
        idx = tf.reshape(idx, (100, 1))
        reduced_pc_z = tf.gather_nd(reduced_pc_z, idx)

        return reduced_pc_z

    def get_glimpse(self, loc):
        imgs = tf.reshape(self.images_ph, 
                            [self.batch_size, self.num_points_per_pc, self.loc_dim])

        #Extract a set of glimpses (both normalized and centered, areas of non-overlap between window and image are
        #filled with RANDOM noise
        zooms= []
        for k in range(self.batch_size):
            x = loc[k][0]
            y = loc[k][1]
            z = loc[k][2]

            x_min = x - self.half_extent
            y_min = y - self.half_extent
            z_min = z - self.half_extent
            x_max = x + self.half_extent
            y_max = y + self.half_extent
            z_max = z + self.half_extent

            one_pc = imgs[k, :, :]

            greater_x = tf.greater_equal(one_pc[:, 0], x_min)
            reduced_pc_x = tf.boolean_mask(one_pc, greater_x)
            less_x = tf.less_equal(reduced_pc_x[:, 0], x_max)
            reduced_pc_x = tf.boolean_mask(reduced_pc_x, less_x)

            greater_y = tf.greater_equal(reduced_pc_x[:, 1], y_min)
            reduced_pc_y = tf.boolean_mask(reduced_pc_x, greater_y)
            less_y = tf.less_equal(reduced_pc_y[:, 1], y_max)
            reduced_pc_y = tf.boolean_mask(reduced_pc_y, less_y)

            greater_z = tf.greater_equal(reduced_pc_y[:, 2], z_min)
            reduced_pc_z = tf.boolean_mask(reduced_pc_y, greater_z)
            less_z = tf.less_equal(reduced_pc_z[:, 2], z_max)
            reduced_pc_z = tf.boolean_mask(reduced_pc_z, less_z)

            reduced_pc_z = tf.cond(tf.less(tf.size(reduced_pc_z), self.num_points_per_glimpse * self.loc_dim), 
                        lambda: self.true_fn(reduced_pc_z), lambda: self.false_fn(reduced_pc_z))

            zooms.append(tf.reshape(reduced_pc_z, (self.num_points_per_glimpse, self.loc_dim, 1)))
        
        glimpse_imgs = tf.stack(zooms)

        return glimpse_imgs

    def __call__(self, loc):
        glimpse_input = self.get_glimpse(loc)
        glimpse_input = tf.reshape(glimpse_input,
                                (tf.shape(loc)[0], self.num_points_per_glimpse * self.loc_dim))

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