import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_layers.snn_layers import LIFSpike, MembraneOutputLayer, tdLinear

""" Implementation of candidate fully connected spiking and non-spiking NNs
for single channel ECG analysis and beat classification task.
"""


class FcnnEcg(nn.Module):
    """ Fully connected NN for single channel ECG analysis
    """
    def __init__(self, input_shape, output_classes) -> None:
        super().__init__()
        self.output_classes = output_classes
        self.hidden_dims = [128, 256]

        self.fc1 = nn.Linear(input_shape, self.hidden_dims[-1])
        self.fc2 = nn.Linear(self.hidden_dims[-1], self.hidden_dims[-2])
        self.fc3 = nn.Linear(self.hidden_dims[-2], self.output_classes)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, input_x):
        out_x = F.leaky_relu(self.fc1(input_x))
        out_x = F.tanh(self.fc2(out_x))
        out_x = self.fc3(out_x)

        return self.softmax(out_x)


class FcSnnEcg(nn.Module):
    """ Fully connected spiking NN for single channel ECG analysis
    """
    def __init__(self, input_shape, output_classes, n_steps=16) -> None:
        super().__init__()
        self.output_classes = output_classes
        self.n_setps = n_steps
        self.hidden_dims = [128, 256]

        self.fc1 = tdLinear(input_shape,
                            self.hidden_dims[-1],
                            bias=True,
                            spike=LIFSpike())
        self.fc2 = tdLinear(self.hidden_dims[-1],
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
        out_x = torch.flatten(input_x, start_dim=1, end_dim=3)  # (N,C*H*W,T)
        out_x = self.fc1(out_x)
        out_x = self.fc2(out_x)
        out_x = self.fc3(out_x)
        out_x = self.membrane_output_layer(out_x)
        out_x = torch.flatten(out_x, start_dim=0, end_dim=2)  # (N, num_classes)
        out_x = self.softmax(out_x)
        out_x = torch.unsqueeze(out_x, 1)

        return out_x
