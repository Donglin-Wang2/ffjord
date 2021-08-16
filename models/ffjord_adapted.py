import numpy as np
import torch.nn as nn
from ..lib import odenvp as odenvp


class ODENVPAdapted(odenvp.ODENVP):
    def __init__(
        self,
        input_size,
        n_scale=float('inf'),
        n_blocks=2,
        intermediate_dims=(32,),
        nonlinearity="softplus",
        squash_input=True,
        alpha=0.05,
        cnf_kwargs=None,
    ):
        print(n_scale)
        super(odenvp.ODENVP, self).__init__(
            input_size,
            n_scale=n_scale,
            n_blocks=n_blocks,
            intermediate_dims=intermediate_dims,
            nonlinearity=nonlinearity,
            squash_input=squash_input,
            alpha=alpha,
            cnf_kwargs=cnf_kwargs,
        )
        self.encoder = nn.Sequential(
            nn.Linear(np.prod(input_size[1:]), 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, np.prod(input_size[1:])),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = super().forward(x)
        z = self.encoder(z)
        return z


class ODENVPAdapted(odenvp.ODENVP):
    def __init__(self, input_size, n_scale, n_blocks, intermediate_dims, nonlinearity, squash_input, alpha, cnf_kwargs):
        super().__init__(input_size, n_scale=n_scale, n_blocks=n_blocks, intermediate_dims=intermediate_dims,
                         nonlinearity=nonlinearity, squash_input=squash_input, alpha=alpha, cnf_kwargs=cnf_kwargs)
