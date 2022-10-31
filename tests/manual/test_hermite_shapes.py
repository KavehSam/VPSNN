import nn_layers.vp_layers as vp
import matplotlib.pyplot as plt
import torch

def realline(m, params):
    dilation, translation = params
    t = torch.arange(-(m // 2), m // 2 + 1) if m % 2 else torch.arange(-(m / 2), m / 2)
    x = dilation * (t - translation * m / 2)
    return x

def test_hermite_shapes(m, n, params):
    x = realline(m, params)
    Phi, dPhi, ind = vp.ada_hermite(m, n, params)

    for i in torch.arange(n):
        ax = plt.subplot(n, 2, int(2*i+1))
        ax.plot(x, torch.squeeze(Phi[:, i]), 'b')
        ax = plt.subplot(n, 2, int(2*i+2))
        ax.plot(x, torch.squeeze(dPhi[:, i]), 'b')
    plt.show()

'''Check the shapes of the Hermite functions visually'''
m = 101
n = 7
params = [0.1, 0.0]
test_hermite_shapes(m, n, params)