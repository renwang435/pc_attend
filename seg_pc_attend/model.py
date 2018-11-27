import math

import torch
import torch.nn as nn

from torch.distributions import Normal

from attention_modules import baseline_network
from attention_modules import glimpse_network, location_network, box_network
from attention_modules import core_network, action_network


class RecurrentAttention(nn.Module):
    """
    A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References
    ----------
    - Mnih et. al., https://arxiv.org/abs/1406.6247
    """
    def __init__(self,
                 num_points_per_pc,
                 num_points_per_sample,
                 num_samples,
                 box_size,
                 s,
                 c,
                 h_g,
                 h_l,
                 std,
                 hidden_size,
                 num_classes,
                 use_gpu):
        """
        Initialize the recurrent attention model and its
        different components.

        Args
        ----
        - g: size of the square patches in the glimpses extracted
          by the retina.
        - k: number of patches to extract per glimpse.
        - s: scaling factor that controls the size of successive patches.
        - c: number of channels in each image.
        - h_g: hidden layer size of the fc layer for `phi`.
        - h_l: hidden layer size of the fc layer for `l`.
        - std: standard deviation of the Gaussian policy.
        - hidden_size: hidden size of the rnn.
        - num_classes: number of classes in the dataset.
        - num_glimpses: number of glimpses to take per image,
          i.e. number of BPTT steps.
        """
        super(RecurrentAttention, self).__init__()
        self.std = std

        self.sensor = glimpse_network(h_g, h_l, num_points_per_pc, num_points_per_sample,
                                      box_size, num_samples, s, c, std, use_gpu)
        self.rnn = core_network(hidden_size, hidden_size)
        self.locator = location_network(hidden_size, 3, std)
        self.box_net = box_network(hidden_size, 1, std)
        self.classifier = action_network(hidden_size, num_classes)
        self.baseliner = baseline_network(hidden_size, 1)

    def forward(self, x, l_t_prev, d_t_prev, h_t_prev, last=False):
        """
        Run the recurrent attention model for 1 timestep
        on the minibatch of images `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
        - l_t_prev: a 2D tensor of shape (B, 3). The location vector
          containing the glimpse coordinates [x, y, z] for the previous
          timestep `t-1`.
        - h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the previous timestep `t-1`.
        - last: a bool indicating whether this is the last timestep.
          If True, the action network returns an output probability
          vector over the classes and the baseline `b_t` for the
          current timestep `t`. Else, the core network returns the
          hidden state vector for the next timestep `t+1` and the
          location vector for the next timestep `t+1`.

        Returns
        -------
        - h_t: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the current timestep `t`.
        - mu: a 2D tensor of shape (B, 3). The mean that parametrizes
          the Gaussian policy.
        - l_t: a 2D tensor of shape (B, 3). The location vector
          containing the glimpse coordinates [x, y, z] for the
          current timestep `t`.
        - b_t: a vector of length (B,). The baseline for the
          current time step `t`.
        - log_probas: a 2D tensor of shape (B, num_classes). The
          output log probability vector over the classes.
        - log_pi: a vector of length (B,).
        """
        g_t = self.sensor(x, l_t_prev, d_t_prev)
        h_t = self.rnn(g_t, h_t_prev)
        mu_l, l_t = self.locator(h_t)
        mu_d, d_t = self.box_net(h_t)
        b_t = self.baseliner(h_t).squeeze()

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi_l = Normal(mu_l, self.std).log_prob(l_t)
        log_pi_l = torch.sum(log_pi_l, dim=1)

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi_d = Normal(mu_l, self.std).log_prob(l_t)
        log_pi_d = torch.sum(log_pi_d, dim=1)


        if last:
            log_probas = self.classifier(h_t)
            return h_t, l_t, d_t, b_t, log_probas, log_pi_l, log_pi_d

        return h_t, l_t, d_t, b_t, log_pi_l, log_pi_d
