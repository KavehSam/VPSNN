import math

import torch
import torch.nn as nn
from nn_layers.snn_layers import (LIFSpike, MembraneOutputLayer, tdConv2d,
                                  tdConvTranspose2d, tdLinear)
from nn_layers.vp_layers import (ada_hermite, tdmvp_layer, tdvp_layer,
                                 temporal_tdmvp_layer, temporal_tdvp_layer,
                                 vp_layer)

""" Implementation of spiking and non-spiking variable projection NN.
"""


class FCVPNN(nn.Module):
    """This network performs a single VP operation in the first layer coupled with a fully connected NN
    """
    def __init__(self, vp_latent_dim, num_classes, device, penalty=None, init_vp=None):
        super().__init__()

        input_length = 100

        if init_vp is not None:
            self.vp = vp_layer(ada_hermite, n_in=input_length,
                               n_out=vp_latent_dim, nparams=2,
                               device=device, penalty=penalty, init=init_vp)
        else:
            self.vp = vp_layer(ada_hermite, n_in=input_length,
                               n_out=vp_latent_dim, nparams=2,
                               device=device, penalty=penalty)

        if int(torch.__version__[2]) > 7:
            self.linear0 = nn.Linear(input_length, vp_latent_dim)#, device=device)
            self.linear1 = nn.Linear(vp_latent_dim, 5)#, device=device)
            self.linear2 = nn.Linear(5, num_classes)#, device=device)
            self.softmax = nn.Softmax(dim=-1)
            self.ReLU = nn.ReLU()
            self.BNorm = torch.nn.BatchNorm1d(1)#, device=device)
        else:
            self.linear0 = nn.Linear(input_length, vp_latent_dim, device=device)
            self.linear1 = nn.Linear(vp_latent_dim, 5, device=device)
            self.linear2 = nn.Linear(5, num_classes, device=device)
            self.softmax = nn.Softmax(dim=-1)
            self.ReLU = nn.ReLU()
            self.BNorm = torch.nn.BatchNorm1d(1, device=device)

    def forward(self, input_x):
        latent = self.vp(input_x)
        # latent = self.linear0(input_x)
        # latent = self.BNorm(latent)
        out_x = self.linear1(latent)
        out_x = self.ReLU(out_x)
        out_x = self.linear2(out_x)
        out_x = self.softmax(out_x)

        return out_x


class FCVPSNN(nn.Module):
    """This network performs a single VP operation in the first layer coupled with fully connected spiking NN
    """
    def __init__(self, vp_latent_dim, num_classes, device, penalty=None, init_vp=None, n_steps=0):
        super().__init__()

        input_length = 100

        if init_vp is not None:
            self.tdvp = tdvp_layer(ada_hermite, n_in=input_length,
                               n_out=vp_latent_dim, nparams=2, device=device,
                               penalty=penalty, init=init_vp, td=n_steps)
        else:
            self.tdvp = tdvp_layer(ada_hermite, n_in=input_length,
                               n_out=vp_latent_dim, nparams=2,
                               device=device, penalty=penalty, td=n_steps)

        self.tdlinear1 = tdLinear(vp_latent_dim,
                                            vp_latent_dim*2,
                                            bias=True,
                                            bn=None,
                                            spike=LIFSpike())

        self.tdlinear2 = tdLinear(vp_latent_dim*2,
                                            vp_latent_dim*5,
                                            bias=False,
                                            bn=None,
                                            spike=LIFSpike())

        self.tdlinear3 = tdLinear(vp_latent_dim*5,
                                            num_classes,
                                            bias=False,
                                            bn=None,
                                            spike=LIFSpike())

        self.softmax = nn.Softmax(dim=-1)
        self.ReLU = nn.ReLU()
        self.membrane_output_layer = MembraneOutputLayer()

    def forward(self, input_x):
        latent = self.tdvp(input_x)
        out_x = self.tdlinear1(latent.squeeze(1)) #<- hidden LIF activation was used in tdLinear
        out_x = self.tdlinear2(out_x)    #<- hidden LIF activation was used in tdLinear
        out_x = self.tdlinear3(out_x)
        out_x = self.membrane_output_layer(out_x.unsqueeze(1))
        out_x = self.softmax(out_x.squeeze(0))

        return out_x


class FCMVPSNN(nn.Module):
    """This network performs multiple VP operations in parallel in the first layer followed by Spiking NN
    """
    def __init__(self, vp_latent_dim, num_classes, device, penalty=None, init_vp=None, n_steps=0, m_vp=1):
        super().__init__()

        input_length = 100

        if init_vp is not None:
            self.tdmvp = tdmvp_layer(ada_hermite, n_in=input_length,
                                   n_out=vp_latent_dim, nparams=2,
                                   device=device, penalty=penalty, init=init_vp,
                                   td=n_steps, m_vp=m_vp)
        else:
            self.tdmvp = tdmvp_layer(ada_hermite, n_in=input_length,
                                   n_out=vp_latent_dim, nparams=2,
                                   device=device, penalty=penalty,
                                   td=n_steps, m_vp=m_vp)

        self.tdlinear1 = tdLinear(vp_latent_dim,
                                            vp_latent_dim*2,
                                            bias=True,
                                            bn=None,
                                            spike=LIFSpike())
        self.tdlinear2 = tdLinear(vp_latent_dim*2,
                                            vp_latent_dim*5,
                                            bias=False,
                                            bn=None,
                                            spike=LIFSpike())

        self.tdlinear3 = tdLinear(vp_latent_dim*5,
                                            num_classes,
                                            bias=False,
                                            bn=None,
                                            spike=LIFSpike())

        self.softmax = nn.Softmax(dim=-1)
        self.ReLU = nn.ReLU()
        self.membrane_output_layer = MembraneOutputLayer()

    def forward(self, input_x):
        latent = self.tdmvp(input_x)
        out_x = self.tdlinear1(latent.squeeze(1)) #<- hidden LIF activation was used in tdLinear
        out_x = self.tdlinear2(out_x)    #<- hidden LIF activation was used in tdLinear
        out_x = self.tdlinear3(out_x)
        out_x = self.membrane_output_layer(out_x.unsqueeze(1))
        out_x = self.softmax(out_x.squeeze(0))

        return out_x


class FCMVPCSNN(nn.Module):
    """This network performs multiple VP operations in parallel in the first layer and applies a td convolution in the second layer.
    """
    def __init__(self, vp_latent_dim, num_classes, device, penalty=None, init_vp=None, n_steps=0, m_vp=1, out_channels=1):
        super().__init__()
        input_length = 100

        if init_vp is not None:
            self.tdmvp = tdmvp_layer(ada_hermite, n_in=input_length,
                                     n_out=vp_latent_dim, nparams=2,
                                     device=device, penalty=penalty, init=init_vp,
                                     td=n_steps, m_vp=m_vp)
        else:
            self.tdmvp = tdmvp_layer(ada_hermite, n_in=input_length,
                                     n_out=vp_latent_dim, nparams=2,
                                     device=device, penalty=penalty, td=n_steps,
                                     m_vp=m_vp)

        self.tdconv = tdConv2d(in_channels=1,
               out_channels=out_channels,
               kernel_size=3,
               stride=2,
               padding=1,
               bias=True,
               bn=None,
               spike=LIFSpike())

        self.tdconvtranspose = tdConvTranspose2d(in_channels=1,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            bn=None,
                            spike=None)
        # hidden output dimension after the convolutional layer
        hidden_dim = math.floor((vp_latent_dim + 2*self.tdconv.padding[0] - self.tdconv.dilation[0]*(
            self.tdconv.kernel_size[0]-1) - 1) / self.tdconv.stride[0] + 1)
        self.tdlinear2 = tdLinear(hidden_dim,
                                  num_classes,
                                  bias=True,
                                  bn=None,
                                  spike=None)

        self.softmax = nn.Softmax(dim=-1)
        self.ReLU = nn.ReLU()
        self.membrane_output_layer = MembraneOutputLayer()

    def forward(self, input_x):
        latent = self.tdmvp(input_x)
        out_x = self.tdconv(latent)  # <- a hidden LIF activation was used in tdconv
        out_x = self.tdlinear2(out_x.squeeze(1))  # <- no hidden LIF activation was used in tdlinear2
        # out_x = self.tdconvtranspose(out_x)
        out_x = self.membrane_output_layer(out_x.unsqueeze(1))
        out_x = self.softmax(out_x.squeeze(0))

        return out_x


class FCVPTSNN(nn.Module):
    """This network performs a single VP operation in the first layer
    and then constructs spike tarils using temporal deattenuation before a spiking NN
    """
    def __init__(self, vp_latent_dim, num_classes, device, penalty=None, init_vp=None, n_steps=0):
        super().__init__()

        input_length = 100

        if init_vp is not None:
            self.tdvp = temporal_tdvp_layer(ada_hermite, n_in=input_length,
                               n_out=vp_latent_dim, nparams=2, device=device,
                               penalty=penalty, init=init_vp, td=n_steps)
        else:
            self.tdvp = temporal_tdvp_layer(ada_hermite, n_in=input_length,
                               n_out=vp_latent_dim, nparams=2,
                               device=device, penalty=penalty, td=n_steps)

        self.tdlinear1 = tdLinear(vp_latent_dim,
                                            vp_latent_dim*2,
                                            bias=True,
                                            bn=None,
                                            spike=LIFSpike())

        self.tdlinear2 = tdLinear(vp_latent_dim*2,
                                            vp_latent_dim*5,
                                            bias=False,
                                            bn=None,
                                            spike=LIFSpike())

        self.tdlinear3 = tdLinear(vp_latent_dim*5,
                                            num_classes,
                                            bias=False,
                                            bn=None,
                                            spike=LIFSpike())

        self.softmax = nn.Softmax(dim=-1)
        self.ReLU = nn.ReLU()
        self.membrane_output_layer = MembraneOutputLayer()

    def forward(self, input_x):
        latent = self.tdvp(input_x)
        out_x = self.tdlinear1(latent.squeeze(1)) #<- hidden LIF activation was used in tdLinear
        out_x = self.tdlinear2(out_x)    #<- hidden LIF activation was used in tdLinear
        out_x = self.tdlinear3(out_x)
        out_x = self.membrane_output_layer(out_x.unsqueeze(1))
        out_x = self.softmax(out_x.squeeze(0))

        return out_x


class FCTMVPSNN(nn.Module):
    """This network first constructs spike tarils using temporal deattenuation,
    and then performs multiple VP operations in parallel in the first layer.
    """
    def __init__(self, vp_latent_dim, num_classes, device, penalty=None, init_vp=None, n_steps=0, m_vp=1):
        super().__init__()

        input_length = 100

        if init_vp is not None:
            self.tdmvp = temporal_tdmvp_layer(ada_hermite, n_in=input_length,
                                   n_out=vp_latent_dim, nparams=2,
                                   device=device, penalty=penalty, init=init_vp,
                                   td=n_steps, m_vp=m_vp)
        else:
            self.tdmvp = temporal_tdmvp_layer(ada_hermite, n_in=input_length,
                                   n_out=vp_latent_dim, nparams=2,
                                   device=device, penalty=penalty, td=n_steps, m_vp=m_vp)

        self.tdlinear1 = tdLinear(vp_latent_dim,
                                            vp_latent_dim*2,
                                            bias=True,
                                            bn=None,
                                            spike=LIFSpike())
        self.tdlinear2 = tdLinear(vp_latent_dim*2,
                                            vp_latent_dim*5,
                                            bias=False,
                                            bn=None,
                                            spike=LIFSpike())

        self.tdlinear3 = tdLinear(vp_latent_dim*2,
                                            num_classes,
                                            bias=False,
                                            bn=None,
                                            spike=LIFSpike())

        self.softmax = nn.Softmax(dim=-1)
        self.ReLU = nn.ReLU()
        self.membrane_output_layer = MembraneOutputLayer()

    def forward(self, input_x):
        latent = self.tdmvp(input_x)
        out_x = self.tdlinear1(latent.squeeze(1)) #<- hidden LIF activation was used in tdLinear
        # out_x = self.tdlinear2(out_x)    #<- hidden LIF activation was used in tdLinear
        out_x = self.tdlinear3(out_x)
        out_x = self.membrane_output_layer(out_x.unsqueeze(1))
        out_x = self.softmax(out_x.squeeze(0))

        return out_x
