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


class SimpleNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = [nn.Linear(1000, 500),
                    nn.ReLU(),
                    nn.Linear(500, 500),
                    nn.ReLU(),
                    nn.Linear(500, 500),
                    nn.ReLU(),
                    nn.Linear(500, 1000)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class SimpleNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = [nn.Linear(1000, 1000),
                    nn.ReLU(),
                    nn.Linear(1000, 1000),
                    nn.ReLU(),
                    nn.Linear(1000, 1000),
                    nn.ReLU(),
                    nn.Linear(1000, 1000)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_part = [nn.Conv1d(1, 32, 5, padding=2),
                          nn.BatchNorm1d(32),
                          nn.ReLU(),
                          nn.Conv1d(32, 64, 5, padding=2),
                          nn.BatchNorm1d(64),
                          nn.ReLU(),
                          nn.Conv1d(64, 128, 5, padding=2),
                          nn.BatchNorm1d(128),
                          nn.ReLU(),
                          nn.Conv1d(128, 256, 5, padding=2),
                          nn.BatchNorm1d(256),
                          nn.ReLU(),
                          nn.Conv1d(256, 1, 1),
                          nn.BatchNorm1d(1),
                          nn.ReLU()]
        self.conv_part = nn.Sequential(*self.conv_part)
        self.fully_part = [nn.Linear(1000, 1000),
                           nn.BatchNorm1d(1000),
                           nn.ReLU(),
                           nn.Linear(1000, 1000, bias=False)]
        self.fully_part = nn.Sequential(*self.fully_part)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_part(x)
        x = x[:, 0, :]
        return self.fully_part(x)


net_definitions = {'affine_projection': partial(AffineProjection),
                   'simple_net': partial(SimpleNet),
                   'simple_net_2': partial(SimpleNet2),
                   'simple_net_3': partial(SimpleNet3),
                   'conv_net': partial(ConvNet)}
