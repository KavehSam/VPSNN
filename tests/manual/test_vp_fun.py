import nn_layers.vp_layers as vp
import matplotlib.pyplot as plt
import torch

def realline(m, params):
    dilation, translation = params
    t = torch.arange(-(m // 2), m // 2 + 1) if m % 2 else torch.arange(-(m / 2), m / 2)
    x = dilation * (t - translation * m / 2)
    return x

def test_vp_fun(m, n, params, coeffs):
    x = realline(m, params)
    Phi, dPhi, ind = vp.ada_hermite(m, n, params)
    signal = Phi @ torch.transpose(coeffs, -1, -2)
    c = torch.zeros_like(coeffs)
    for i in torch.arange(n):
        c[:, :i+1] = coeffs[:, :i+1]
        aprx = Phi @ torch.transpose(c, -1, -2)
        ax = plt.subplot(n, 1, int(i+1))
        ax.plot(x, torch.squeeze(signal), 'b', label='signal')
        ax.plot(x, torch.squeeze(aprx), 'r--', label='approx')
    plt.show()

    return signal

'''Checking the projection.'''
params = torch.tensor([0.1, 0.0])
coeffs = torch.tensor([3., 3., 2., 1., -1.], dtype=torch.double).unsqueeze(0)
m = 101
n = coeffs.size()[1]
signal = test_vp_fun(m, n, params, coeffs)
