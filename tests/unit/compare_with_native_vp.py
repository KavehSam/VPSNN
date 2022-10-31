import nn_layers.vp_layers as vp
import torch
import torch.testing
import unittest
import torch.nn as nn
import NativeVPNet.src.vpnet as oldvp
from torch.autograd import gradcheck
import numpy as np

class TestVP(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    '''It generates an input signal with predefined coefficients.
       The script tests whether we get back the same coefficients by the projection.'''

    def test_compare_forward(self):
        # Generating the input signal
        m = 101
        n = 5
        batch = 3
        penalty = 0.0
        init = [0.1, 0.0]
        signal = torch.randn(batch, 1, m).reshape(batch, 1, m)
        new_vp = vp.vp_layer(vp.ada_hermite, m, n, 2, device='cpu', penalty=penalty, init=init)
        old_vp = oldvp.VPLayer(m, n, oldvp.hermite_ada, init, penalty)
        with torch.no_grad():
            c1 = new_vp(signal)
            c2 = torch.tensor(old_vp.forward(signal.squeeze(1).numpy()), dtype=c1.dtype).unsqueeze(1)
        self.assertIsNone(torch.testing.assert_close(c1, c2))

    def test_compare_backward(self):
        # Generating the input signal
        m = 101
        n = 5
        batch = 30
        penalty = 0.5
        init = [0.1, 0.0]
        signal = torch.randn(batch, 1, m).reshape(batch, 1, m)
        new_vp = vp.vp_layer(vp.ada_hermite, m, n, 2, device='cpu', penalty=penalty, init=init)
        vp_model = nn.Sequential(new_vp)
        old_vp = oldvp.VPLayer(m, n, oldvp.hermite_ada, init, penalty)
        c1 = vp_model(signal)
        c2 = old_vp.forward(signal.squeeze(1).numpy())
        loss = c1.sum()
        loss.backward()
        grad1 = vp_model[0].weight.grad
        _, grad2 = old_vp.backward(np.ones((1, 5)))
        self.assertIsNone(torch.testing.assert_close(grad1, torch.tensor(grad2)))

if __name__ == '__main__':
    unittest.main()