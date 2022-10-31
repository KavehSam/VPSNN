import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_layers.snn_layers import (LIFSpike, MembraneOutputLayer, tdBatchNorm,
                                  tdConv, tdLinear)

""" Implementation of candidate convulitional spiking and non-spiking NNs
for single channel ECG analysis and beat classification task.
"""

class CnnFcnnEcg(nn.Module):
    """ Cnn coupled with conventional FCNN
    """
    def __init__(self, in_channels, output_classes) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.output_classes = output_classes

        modules = []
        hidden_dims = [32, 64, 128, 128]
        output_paddings = [0, 0, 0, 1]
        self.hidden_dims = hidden_dims.copy()
        self.output_paddings = output_paddings

        # Build cnn
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(self.in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            self.in_channels = h_dim

        self.cnn = nn.Sequential(*modules)
        self.fc1 = nn.Linear(self.hidden_dims[-1] * 7, self.hidden_dims[-1])
        self.fc2 = nn.Linear(self.hidden_dims[-1], self.hidden_dims[-2])
        self.fc3 = nn.Linear(self.hidden_dims[-2], self.output_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_x):
        out_x = self.cnn(input_x)
        out_x = out_x.reshape(-1, self.hidden_dims[-1] * 7)
        out_x = F.leaky_relu(self.fc1(out_x))
        out_x = torch.tanh(self.fc2(out_x))
        out_x = self.fc3(out_x)
        out_x = self.softmax(out_x)
        out_x = torch.unsqueeze(out_x, 1)

        return out_x


class CnnFcSnnEcg(nn.Module):
    """ Cnn coupled with Spiking FCNN
    """
    def __init__(self, in_channels, output_classes, n_steps=16) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.output_classes = output_classes
        self.n_setps = n_steps

        modules = []
        hidden_dims = [32, 64, 128, 256]
        output_paddings = [[0, 0], [0, 0], [0, 0], [0, 1]]
        self.hidden_dims = hidden_dims.copy()
        self.output_paddings = output_paddings

        # Build cnn
        modules = []
        is_first_conv = True
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                        out_channels=h_dim,
                        kernel_size=[1, 3],
                        stride=[1, 2],
                        padding=[0, 1],
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike(),
                        is_first_conv=is_first_conv)
            )
            in_channels = h_dim
            is_first_conv = False

        self.cnn = nn.Sequential(*modules)
        self.fc1 = tdLinear(self.hidden_dims[-1]*7,
                            self.hidden_dims[-2],
                            bias=True,
                            spike=LIFSpike())
        self.fc2 = tdLinear(self.hidden_dims[-2],
                            self.hidden_dims[-2],
                            bias=False,
                            spike=LIFSpike())
        self.fc3 = tdLinear(self.hidden_dims[-2],
                            self.output_classes,
                            bias=False,
                            spike=LIFSpike())
        self.membrane_output_layer = MembraneOutputLayer()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_x):
        input_x = input_x.unsqueeze(dim=2)  # (N,C,H,W)
        input_x = input_x.unsqueeze(-1).repeat(1, 1, 1, 1, self.n_setps)  # (N,C,H,W,T)
        out_x = self.cnn(input_x)
        out_x = torch.flatten(out_x, start_dim=1, end_dim=3)  # (N,C*H*W,T)
        out_x = self.fc1(out_x)
        out_x = self.fc2(out_x)
        out_x = self.fc3(out_x)
        out_x = self.membrane_output_layer(out_x)
        out_x = torch.flatten(out_x, start_dim=0, end_dim=2)  # (N, num_classes)
        out_x = self.softmax(out_x)
        out_x = torch.unsqueeze(out_x, 1)

        return out_x
