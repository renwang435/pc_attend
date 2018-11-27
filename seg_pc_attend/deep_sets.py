from torch import nn
import torch
import torch.nn.functional as F
import sys

class PermEqui1_max(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui1_max, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm, _ = x.max(1, keepdim=True)
        x = self.Gamma(x - xm)

        return x

class PermEqui1_mean(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui1_mean, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm = x.mean(1, keepdim=True)
        x = self.Gamma(x - xm)

        return x

class PermEqui2_max(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui2_max, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        xm, _ = x.max(1, keepdim=True)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm

        return x

class PermEqui2_mean(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui2_mean, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        xm = x.mean(1, keepdim=True)

        xm = self.Lambda(xm)
        x = self.Gamma(x)

        x = x - xm

        return x

class DTanh(nn.Module):
    def __init__(self, x_dim, out_dim, pool='mean'):
        super(DTanh, self).__init__()
        self.out_dim = out_dim
        self.x_dim = x_dim

        if pool == 'max':
            self.phi = nn.Sequential(
                PermEqui2_max(self.x_dim, 64),
                nn.Tanh(),
                PermEqui2_max(64, 128),
                nn.Tanh(),
                PermEqui2_max(128, 256),
                nn.Tanh(),
                PermEqui2_max(256, 512),
                nn.Tanh(),
                PermEqui2_max(512, 1024),
                nn.Tanh(),
                PermEqui2_max(1024, 2048),
                nn.Tanh(),
                PermEqui2_max(2048, 1024),
                nn.Tanh(),
                PermEqui2_max(1024, 512),
                nn.Tanh(),
                PermEqui2_max(512, 256),
                nn.Tanh(),
                PermEqui2_max(256, 128),
                nn.Tanh(),
                PermEqui2_max(128, 64),
                nn.Tanh(),
                PermEqui2_max(64, self.out_dim),
                nn.Tanh(),
            )
        # elif pool == 'max1':
        #     self.phi = nn.Sequential(
        #       PermEqui1_max(self.x_dim, self.d_dim),
        #       nn.Tanh(),
        #       PermEqui1_max(self.d_dim, self.d_dim),
        #       nn.Tanh(),
        #       PermEqui1_max(self.d_dim, self.d_dim),
        #       nn.Tanh(),
        #     )
        elif pool == 'mean':
            self.phi = nn.Sequential(
                PermEqui2_mean(self.x_dim, 1024),
                nn.Tanh(),
                PermEqui2_mean(1024, 512),
                nn.Tanh(),
                PermEqui2_mean(512, 256),
                nn.Tanh(),
                PermEqui2_mean(256, 128),
                nn.Tanh(),
                PermEqui2_mean(128, 64),
                nn.Tanh(),
                PermEqui2_mean(64, self.out_dim),
                # nn.Tanh(),
                # nn.Sigmoid(),
            )
        # elif pool == 'mean1':
        #     self.phi = nn.Sequential(
        #       PermEqui1_mean(self.x_dim, self.d_dim),
        #       nn.Tanh(),
        #       PermEqui1_mean(self.d_dim, self.d_dim),
        #       nn.Tanh(),
        #       PermEqui1_mean(self.d_dim, self.d_dim),
        #       nn.Tanh(),
        #     )

    def forward(self, x):
        phi_output = self.phi(x)
        # sum_output, _ = phi_output.max(1)

        return phi_output
