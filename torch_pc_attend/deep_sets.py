from torch import nn
import torch
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
    def __init__(self, x_dim, d_dim, pool='mean'):
        super(DTanh, self).__init__()
        self.d_dim = d_dim
        self.x_dim = x_dim

        if pool == 'max':
            self.phi = nn.Sequential(
              PermEqui2_max(self.x_dim, self.d_dim),
              nn.Tanh(),
              PermEqui2_max(self.d_dim, self.d_dim),
              nn.Tanh(),
              PermEqui2_max(self.d_dim, self.d_dim),
              nn.Tanh(),
            )
        elif pool == 'max1':
            self.phi = nn.Sequential(
              PermEqui1_max(self.x_dim, self.d_dim),
              nn.Tanh(),
              PermEqui1_max(self.d_dim, self.d_dim),
              nn.Tanh(),
              PermEqui1_max(self.d_dim, self.d_dim),
              nn.Tanh(),
            )
        elif pool == 'mean':
            self.phi = nn.Sequential(
              PermEqui2_mean(self.x_dim, self.d_dim),
              nn.Tanh(),
              PermEqui2_mean(self.d_dim, self.d_dim),
              nn.Tanh(),
              PermEqui2_mean(self.d_dim, self.d_dim),
              nn.Tanh(),
            )
        elif pool == 'mean1':
            self.phi = nn.Sequential(
              PermEqui1_mean(self.x_dim, self.d_dim),
              nn.Tanh(),
              PermEqui1_mean(self.d_dim, self.d_dim),
              nn.Tanh(),
              PermEqui1_mean(self.d_dim, self.d_dim),
              nn.Tanh(),
            )

        self.ro = nn.Sequential(
           nn.Dropout(p=0.5),
           nn.Linear(self.d_dim, self.d_dim),
           nn.Tanh(),
           nn.Dropout(p=0.5),
           nn.Linear(self.d_dim, self.d_dim),
        )

    def forward(self, x):
        phi_output = self.phi(x)
        sum_output, _ = phi_output.max(1)
        ro_output = self.ro(sum_output)

        return ro_output
