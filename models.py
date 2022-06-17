from torch import nn
from functools import partial


class AffineProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1000, 1000)

    def forward(self, x):
        return self.net(x)


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = [nn.Linear(1000, 500),
                    nn.ReLU(),
                    nn.Linear(500, 1000)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


net_definitions = {'affine_projection': partial(AffineProjection),
                   'simple_net': partial(SimpleNet)}
