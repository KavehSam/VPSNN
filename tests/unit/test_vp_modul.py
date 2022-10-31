import nn_layers.vp_layers as vp
import torch
import torch.testing
import unittest
import torch.nn as nn
from torch.autograd import gradcheck

class TestVPModul(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    '''It generates an input signal with predefined coefficients.
       The script tests whether we get back the same coefficients by the projection.'''

    def test_coeffs(self):
        # Generating the input signal
        m = 101
        n = 5
        batch = 1
        c1 = torch.randn(batch, 1, n)
        test_vp_fun = vp.vp_layer(vp.ada_hermite, m, n, 2, device='cpu')
        Phi, _, _ = vp.ada_hermite(m, n, test_vp_fun.weight)
        signal = (Phi @ torch.transpose(c1, -1, -2)).reshape(batch, 1, m)
        # Recovering the coefficients from the input signal
        with torch.no_grad():
            c2 = test_vp_fun(signal)
        self.assertIsNone(torch.testing.assert_close(c1, c2))

    def test_coeffs_on_spike_input(self):
        # Generating the input signal
        m = 101
        n = 5
        batch = 3
        n_steps = 2
        c1 = torch.randn(batch, 1, n)
        spike_c1 = c1.unsqueeze(-1).repeat(1, 1, 1, n_steps)
        test_vp_fun = vp.tdvp_layer(vp.ada_hermite, m, n, 2, device='cpu', td=n_steps)
        Phi, _, _ = vp.ada_hermite(m, n, test_vp_fun.weight)
        signal = (Phi @ torch.transpose(c1, -1, -2)).reshape(batch, 1, m)
        spike_input = signal.unsqueeze(-1).repeat(1, 1, 1, n_steps)  # (N,C,L,T)

        # Recovering the coefficients from the input signal
        with torch.no_grad():
            spike_c2 = test_vp_fun(signal)
        self.assertIsNone(torch.testing.assert_close(spike_c1, spike_c2))
  
    def test_vpfun_grad(self):
        # Generating the input signal
        m = 101
        n = 8
        batch = 1 # Note that the VP_modul computes averaged gradients over the batches.
                  # For checking other cases (batch>1), change 'return dp' to 'return dp*batch' in vp_layers.vpfun.
        penalty = 0.0
        c1 = torch.randn(batch, 1, n, dtype=torch.double)
        test_vp_fun = vp.vp_layer(vp.ada_hermite, m, n, 2, device='cpu')
        Phi, dPhi, _ = vp.ada_hermite(m, n, test_vp_fun.weight, dtype=c1.dtype)
        signal = (Phi @ torch.transpose(c1, -2, -1)).requires_grad_()
        # Checking the gradient of vpfun
        ada = lambda params: vp.ada_hermite(m, n, params, dtype=c1.dtype)
        weight = torch.randn(2, dtype=c1.dtype, requires_grad=True)
        vpfun = vp.vpfun.apply
        gradcheck(vpfun, (torch.transpose(signal, -1, -2), weight, ada, 'cpu', penalty), eps=1e-6, atol=1e-4)

    def test_vpfun_grad_on_spike_input(self):
        # Generating the input signal
        m = 101
        n = 8
        batch = 1 # Note that the VP_modul computes averaged gradients over the batches.
                  # For checking other cases (batch>1), change 'return dp' to 'return dp*batch' in vp_layers.vpfun.
        n_steps = 5
        penalty = 0.0
        c1 = torch.randn(batch, 1, n, dtype=torch.double)
        test_vp_fun = vp.tdvp_layer(vp.ada_hermite, m, n, 2, device='cpu')
        Phi, dPhi, _ = vp.ada_hermite(m, n, test_vp_fun.weight, dtype=c1.dtype)
        signal = (Phi @ torch.transpose(c1, -2, -1)).reshape(batch, 1, m).requires_grad_()
        spike_input = signal.unsqueeze(-1).repeat(1, 1, 1, n_steps)  # (N,C,L,T)
        # Checking the gradient of tdvpfun
        ada = lambda params: vp.ada_hermite(m, n, params, dtype=c1.dtype)
        weight = torch.randn(2, dtype=c1.dtype, requires_grad=True)
        vpfun = vp.vpfun.apply
        gradcheck(vpfun, (signal, weight, ada, 'cpu', penalty), eps=1e-6, atol=1e-4)


    def test_vpmodul(self):
        # Generating the input signal
        m = 101
        n = 8
        batch = 2
        init = [0.1, 0.0]
        c1 = torch.randn(batch, 1, n, dtype=torch.float)
        VP = vp.vp_layer(vp.ada_hermite, m, n, 2, device='cpu')
        Phi, dPhi, _ = vp.ada_hermite(m, n, VP.weight, dtype=c1.dtype)
        signal = (Phi @ c1.mT).requires_grad_()

        # Constructing a general NN model
        model = nn.Sequential(
            VP,
            nn.Linear(n, n),
            nn.ReLU(),
            nn.ConvTranspose1d(1, 3, 10)
        )
        # Checking the gradient of NN model
        gradcheck(model, signal.mT, eps=1e-3, atol=1e-4)    #The precision may seem to be low at this point, but even built-in torch layers could not pass the gradient check with higher precision.

if __name__ == '__main__':
    unittest.main()