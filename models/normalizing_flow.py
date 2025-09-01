import torch
import torch.nn as nn


class AffineCouplingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = config['normflow_params']['conv']
        self.im_latent_size = config['dataset_params']['im_size'] // 2 ** sum(
            config['autoencoder_params']['down_sample'])
        self.z_channels = config['autoencoder_params']['z_channels']
        hidden = config['normflow_params']['hidden_dim']
        assert self.z_channels % 2 == 0, 'Latent channels must be divisible by 2'
        if not self.conv:
            self.D = self.z_channels * self.im_latent_size * self.im_latent_size
            assert self.D % 2 == 0, "Dimension of image must be divisible by 2"
            self.d = self.D // 2

            self.sig_net = nn.Sequential(
                nn.Linear(self.d, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.D - self.d)
            )

            self.mu_net = nn.Sequential(
                nn.Linear(self.d, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.D - self.d))

            self.log_scale_factor = nn.Parameter(torch.zeros(self.D - self.d))
        else:
            in_channels = self.z_channels // 2
            self.mu_net = nn.Sequential(
                nn.Conv2d(in_channels, hidden, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden * 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden * 2, in_channels, 3, padding=1),
            )
            self.sig_net = nn.Sequential(
                nn.Conv2d(in_channels, hidden, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden * 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden * 2, in_channels, 3, padding=1),
            )
            self.log_scale_factor = nn.Parameter(torch.zeros(1,
                                                             in_channels,
                                                             self.im_latent_size,
                                                             self.im_latent_size)
                                                 )

    def forward(self, x, flip):
        batch_size = x.shape[0]
        if not self.conv:
            x1, x2 = x[:, :self.d], x[:, self.d:]
        else:
            x1, x2 = x.chunk(2, 1)

        if flip:
            x1, x2 = x2, x1

        sig = (torch.nn.Tanh()(self.sig_net(x1)) * self.log_scale_factor.exp()).exp()
        x2 = x2 * sig + self.mu_net(x1)

        if flip:
            x1, x2 = x2, x1

        x = torch.cat([x1, x2], dim=1)
        log_jacobian_det = torch.sum((sig.log()).view(batch_size, -1), 1)
        return x, log_jacobian_det

    def inverse(self, x, flip):
        if not self.conv:
            x1, x2 = x[:, :self.d], x[:, self.d:]
        else:
            x1, x2 = x.chunk(2, 1)

        if flip:
            x1, x2 = x2, x1

        sig = (torch.nn.Tanh()(self.sig_net(x1)) * self.log_scale_factor.exp()).exp()
        x2 = (x2 - self.mu_net(x1)) / sig

        if flip:
            x1, x2 = x2, x1

        x = torch.cat([x1, x2], dim=1)
        return x


class SimpleRealNVP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n = config['normflow_params']['num_layers']
        self.flips = [True if i % 2 else False for i in range(self.n)]
        self.coupling_layers = nn.ModuleList([
            AffineCouplingLayer(config) for _ in range(self.n)
        ])
        self.prior = torch.distributions.Normal(loc=0.0, scale=1.0)

    def forward(self, x):
        log_jacobian_determinants = []
        coupling_layers_count = len(self.coupling_layers)
        for idx in range(coupling_layers_count):
            x, log_jacobian_determinant = self.coupling_layers[idx](x, flip=self.flips[idx])
            log_jacobian_determinants.append(log_jacobian_determinant)

        log_prob = self.prior.log_prob(x).view(x.shape[0], -1).sum(dim=1)
        return x, log_prob, sum(log_jacobian_determinants)

    def inverse(self, x):
        coupling_layers_count = len(self.coupling_layers)
        for idx in reversed(range(coupling_layers_count)):
            x = self.coupling_layers[idx].inverse(x, flip=self.flips[idx])
        return x
