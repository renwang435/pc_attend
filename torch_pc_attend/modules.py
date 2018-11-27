import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_sets import DTanh

from torch.autograd import Variable

import numpy as np
import sys

class retina(object):
    """
    A retina that extracts a foveated glimpse `phi`
    around location `l` from an image `x`. It encodes
    the region around `l` at a high-resolution but uses
    a progressively lower resolution for pixels further
    from `l`, resulting in a compressed representation
    of the original image `x`.

    Args
    ----
    - x: a 4D Tensor of shape (B, H, W, D). The minibatch
      of images.
    - l: a 2D Tensor of shape (B, 3). Contains normalized
      coordinates in the range [-1, 1].
    - g: size of the first square patch.
    - k: number of patches to extract in the glimpse.
    - s: scaling factor that controls the size of
      successive patches.

    Returns
    -------
    - phi: a 5D tensor of shape (B, num_samples, num_points, D). The
      foveated glimpse of the image.
    """
    def __init__(self, num_points, box_size, num_samples, s, std, use_gpu):
        self.num_points = num_points
        self.box_size = box_size
        self.num_samples = num_samples
        self.s = s
        self.std = std
        self.use_gpu = use_gpu

    def foveate(self, x, l):
        """
        Extract `k` square patches of size `g`, centered
        at location `l`. The initial patch is a square of
        size `g`, and each subsequent patch is a square
        whose side is `s` times the size of the previous
        patch.

        The `k` patches are finally resized to (g, g) and
        concatenated into a tensor of shape (B, k, g, g, C).
        """
        phi = []
        size = self.box_size

        # extract points from k boxes of increasing size
        for i in range(self.num_samples):
            phi.append(self.extract_box(x, l, size))
            size = int(self.s * size)

        # concatenate into a single tensor and flatten
        phi = torch.cat(phi, 1)
        phi = phi.squeeze(-1)

        return phi

    def extract_box(self, imgs, l, size):
        """
        Extract a single patch for each image in the
        minibatch `imgs`.

        Args
        ----
        - x: a 4D Tensor of shape (B, P, D). The minibatch
          of images.
        - l: a 2D Tensor of shape (B, 3).
        - size: a scalar defining the size of the extracted patch.

        Returns
        -------
        - patch: a 4D Tensor of shape (B, D, num_points, C)
        """
        B, C, P, D = imgs.shape
        imgs = imgs.view(-1, 3, 10000)

        # loop through mini-batch and extract num_points from the box defined by size, centered at l
        zooms = []
        for i in range(B):
            im = imgs[i].float()

            x = l[i][0]
            y = l[i][1]
            z = l[i][2]

            half_extent = size / 2
            x_min = x - half_extent
            y_min = y - half_extent
            z_min = z - half_extent
            x_max = x + half_extent
            y_max = y + half_extent
            z_max = z + half_extent

            bool_mask_x = (im[0, :] >= x_min) & (im[0, :] <= x_max)
            bool_mask_y = (im[1, :] >= y_min) & (im[1, :] <= y_max)
            bool_mask_z = (im[2, :] >= z_min) & (im[2, :] <= z_max)
            bool_mask = bool_mask_x & bool_mask_y & bool_mask_z
            im = im[:, bool_mask]

            if (im.numel() < self.num_points * 3):
                num_missing = int(self.num_points - im.numel() // 3)

                # If there are no points...fill with zeros
                # If there are some points, randomly sample from those points to get self.num_points
                if (num_missing == self.num_points):
                    fill_box = torch.zeros([3, num_missing])
                else:
                    indices = np.random.choice(im.numel() // 3, (num_missing,), replace=True)
                    fill_box = im[:, indices]

                if self.use_gpu:
                    fill_box = fill_box.cuda()
                im = torch.cat((im, fill_box), -1)
            else:
                perm = torch.randperm(im.numel() // 3)
                idx = perm[:self.num_points]
                im = im[:, idx]

            zooms.append(im.view((1, 3, self.num_points)))

        # concatenate into a single tensor
        glimpse_imgs = torch.stack(zooms)

        return glimpse_imgs


class glimpse_network(nn.Module):
    """
    A network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.

    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.

    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.

    In other words:

        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`.
    - h_l: hidden layer size of the fc layer for `l`.
    - g: size of the square patches in the glimpses extracted
      by the retina.
    - k: number of patches to extract per glimpse.
    - s: scaling factor that controls the size of successive patches.
    - c: number of channels in each image.
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l_t_prev: a 2D tensor of shape (B, 3). Contains the glimpse
      coordinates [x, y, z] for the previous timestep `t-1`.

    Returns
    -------
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    """
    def __init__(self, h_g, h_l, num_points, box_size, num_samples, s, c, std, use_gpu):
        super(glimpse_network, self).__init__()
        self.retina = retina(num_points, box_size, num_samples, s, std, use_gpu)

        # glimpse layer
        # D_in = 3
        # self.DTanh1 = DTanh(D_in, h_g)
        D_in = num_points * 3
        self.fc1 = nn.Linear(D_in, h_g)

        # location layer
        D_in = 3
        self.fc2 = nn.Linear(D_in, h_l)

        self.fc3 = nn.Linear(h_g, h_g + h_l)
        self.fc4 = nn.Linear(h_l, h_g + h_l)

    def forward(self, x, l_t_prev):
        # generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev)
        phi = phi.view(-1, 300)
        # phi = phi.view(-1, 100, 3)

        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        # feed phi and l to respective fc layers
        # phi_out = F.relu(self.DTanh1(phi))
        phi_out = F.relu(self.fc1(phi))
        l_out = F.relu(self.fc2(l_t_prev))

        what = self.fc3(phi_out)
        where = self.fc4(l_out)

        # feed to fc layer
        g_t = F.relu(what + where)

        return g_t


class core_network(nn.Module):
    """
    An RNN that maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args
    ----
    - input_size: input size of the rnn.
    - hidden_size: hidden size of the rnn.
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    - h_t_prev: a 2D tensor of shape (B, hidden_size). The
      hidden state vector for the previous timestep `t-1`.

    Returns
    -------
    - h_t: a 2D tensor of shape (B, hidden_size). The hidden
      state vector for the current timestep `t`.
    """
    def __init__(self, input_size, hidden_size):
        super(core_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t, h_t_prev):
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = F.relu(h1 + h2)
        return h_t


class action_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - a_t: output probability vector over the classes.
    """
    def __init__(self, input_size, output_size):
        super(action_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t


class location_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - mu: a 2D vector of shape (B, 3).
    - l_t: a 2D vector of shape (B, 3).
    """
    def __init__(self, input_size, output_size, std):
        super(location_network, self).__init__()
        self.std = std
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        # compute mean
        mu = torch.tanh(self.fc(h_t.detach()))

        # reparametrization trick
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std)
        l_t = mu + noise

        # bound between [-1, 1]
        # l_t = torch.tanh(l_t)
        l_t = torch.clamp(l_t, -1.001, 1.001)

        return mu, l_t


class baseline_network(nn.Module):
    """
    Regresses the baseline in the reward function
    to reduce the variance of the gradient update.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network
      for the current time step `t`.

    Returns
    -------
    - b_t: a 2D vector of shape (B, 1). The baseline
      for the current time step `t`.
    """
    def __init__(self, input_size, output_size):
        super(baseline_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = F.relu(self.fc(h_t.detach()))
        return b_t
