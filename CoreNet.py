from Utils import *

rnn_cell = tf.contrib.rnn
seq2seq = tf.contrib.legacy_seq2seq

class CoreNet(object):
    def __init__(self, params, glimpse_net, loc_net):
        self.batch_size = params.batch_size
        self.cell_core_size = params.cell_core_size
        self.num_glimpses = params.num_glimpses
        self.glimpse_net = glimpse_net
        self.loc_net = loc_net

        self.glimpse_images = []
        self.loc_mean_arr = []
        self.sampled_loc_arr = []

    #Function to get the next glimpse and update our location parameters
    def get_next_input(self, output, i):
      loc, loc_mean = self.loc_net(output)
      gl_next = self.glimpse_net(loc)
      self.glimpse_images.append(self.glimpse_net.get_glimpse(loc))
      self.loc_mean_arr.append(loc_mean)
      self.sampled_loc_arr.append(loc)

      return gl_next

    def init_core(self, num_examples):
        # Take an initial glimpse based on some random initial location
        init_loc = tf.random_uniform((num_examples, 2), minval=-1, maxval=1)
        init_glimpse = self.glimpse_net(init_loc)

        return init_glimpse

    def __call__(self, num_examples):
        # Core network.
        lstm_cell = rnn_cell.LSTMCell(self.cell_core_size, state_is_tuple=True)
        init_state = lstm_cell.zero_state(num_examples, tf.float32)
        inputs = [self.init_core(num_examples)]
        inputs.extend([0] * (self.num_glimpses))
        outputs, _ = seq2seq.rnn_decoder(
            inputs, init_state, lstm_cell, loop_function=self.get_next_input)


        return outputs, self.sampled_loc_arr, self.glimpse_images



