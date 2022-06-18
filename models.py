import torch
import math
from torch import nn
from functools import partial


class LinearProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1000, 1000, bias=False)

    def forward(self, x):
        return self.net(x)


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
                    nn.BatchNorm1d(500),
                    nn.ReLU(),
                    nn.Linear(500, 1000, bias=False)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class SimpleNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = [nn.Linear(1000, 500),
                    nn.BatchNorm1d(500),
                    nn.ReLU(),
                    nn.Linear(500, 500),
                    nn.BatchNorm1d(500),
                    nn.ReLU(),
                    nn.Linear(500, 500),
                    nn.BatchNorm1d(500),
                    nn.ReLU(),
                    nn.Linear(500, 1000, bias=False)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class SimpleNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = [nn.Linear(1000, 1000),
                    nn.BatchNorm1d(1000),
                    nn.ReLU(),
                    nn.Linear(1000, 1000),
                    nn.BatchNorm1d(1000),
                    nn.ReLU(),
                    nn.Linear(1000, 1000),
                    nn.BatchNorm1d(1000),
                    nn.ReLU(),
                    nn.Linear(1000, 1000, bias=False)]
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
                          nn.Conv1d(256, 5, 5, padding=2),
                          nn.BatchNorm1d(5),
                          nn.ReLU()]
        self.conv_part = nn.Sequential(*self.conv_part)
        self.fully_part = [nn.Linear(5*1000, 1000),
                           nn.BatchNorm1d(1000),
                           nn.ReLU(),
                           nn.Linear(1000, 1000, bias=False)]
        self.fully_part = nn.Sequential(*self.fully_part)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_part(x)
        x = torch.flatten(x, start_dim=1)
        return self.fully_part(x)


class ConvLSTMNet(nn.Module):
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
                          nn.ReLU()]
        self.conv_part = nn.Sequential(*self.conv_part)
        self.recurrent_part = nn.LSTM(256, 128, 2, bidirectional=True)
        self.aggregation_part = [nn.Conv1d(256, 5, 5, padding=2),
                                 nn.BatchNorm1d(5),
                                 nn.ReLU()]
        self.aggregation_part = nn.Sequential(*self.aggregation_part)
        self.fully_part = [nn.Linear(5*1000, 1000),
                           nn.BatchNorm1d(1000),
                           nn.ReLU(),
                           nn.Linear(1000, 1000, bias=False)]
        self.fully_part = nn.Sequential(*self.fully_part)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_part(x)
        x = x.permute(2, 0, 1)
        x, _ = self.recurrent_part(x)
        x = x.permute(1, 2, 0)
        x = self.aggregation_part(x)
        x = torch.flatten(x, start_dim=1)
        return self.fully_part(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ConvAttentionNet(nn.Module):
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
                          nn.ReLU()]
        self.conv_part = nn.Sequential(*self.conv_part)
        self.pos_encoder = PositionalEncoding(d_model=256, dropout=0, max_len=1000)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, dropout=0)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.aggregation_part = [nn.Conv1d(256, 5, 5, padding=2),
                                 nn.BatchNorm1d(5),
                                 nn.ReLU()]
        self.aggregation_part = nn.Sequential(*self.aggregation_part)
        self.fully_part = [nn.Linear(5*1000, 1000),
                           nn.BatchNorm1d(1000),
                           nn.ReLU(),
                           nn.Linear(1000, 1000, bias=False)]
        self.fully_part = nn.Sequential(*self.fully_part)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_part(x)
        x = x.permute(2, 0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)
        x = self.aggregation_part(x)
        x = torch.flatten(x, start_dim=1)
        return self.fully_part(x)


net_definitions = {'linear_projection': partial(LinearProjection),
                   'affine_projection': partial(AffineProjection),
                   'simple_net': partial(SimpleNet),
                   'simple_net_2': partial(SimpleNet2),
                   'simple_net_3': partial(SimpleNet3),
                   'conv_net': partial(ConvNet),
                   'conv_lstm_net': partial(ConvLSTMNet),
                   'conv_attention_net': partial(ConvAttentionNet)}
