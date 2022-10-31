import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function


def ada_hermite(m, n, params, dtype=torch.float, device=None):
    """ada is a user-supplied function which computes the values and the derivatives of
    the function system matrix 'Phi'.
    ada_hermite computes the values and the derivatives of the classical Hermite functions
    parametrized by dilation and translation.

    Input
    ----------
    m: int
        Number of samples, i.e., row dimension of 'Phi'.
    n: int
        Number of basis functions, i.e., column dimension of 'Phi'.
    device: torch.device, optional
        the desired device of returned tensor. Default: None
    params: torch Tensor of floats
        nonlinear parameters of the basic functions, e.g., params = torch.tensor([dilation, translation])

    Output
    -------
    Phi: 2D torch Tensor of floats, whose [i,j] entry is equal to the jth basic function evaluated
        at the ith time instance t[i], e.g., each column of the matrix 'Phi' contains a sampling of the
        parametrized Hermite functions for a given 'params'.

    dPhi: 2D torch Tensor of floats, whose kth column contains the partial derivative of the jth basic function
        with respect to the ith nonlinear parameter, where j = ind[0,k] and i = ind[1,k],
        e.g., each column of the matrix 'dPhi' contains a sampling of the partial derivatives of the
         Hermite functions with respect to the dilation or to the translation parameter.

    ind: 2D torch Tensor of floats, auxiliary matrix for dPhi, i.e., column dPhi[:,k] contains
        the partial derivative of Phi[:,j]
        with respect to params[i], where j=ind[0,k] and i=ind[1,k],
        e.g., for the first three parametrized Hermite functions:
        ind = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
    """

    dilation, translation = params[:2]
    t = torch.arange(-(m // 2), m // 2 + 1, dtype=dtype) if m % 2 else torch.arange(-(m / 2), m / 2, dtype=dtype,
                                                                                    device=device)
    x = dilation * (t - translation * m / 2)
    w = torch.exp(-0.5 * x ** 2)
    dw = -x * w
    pi_sqrt = torch.sqrt(torch.sqrt(torch.tensor(math.pi, device=device)))

    # Phi, dPhi
    Phi = torch.zeros((m, n), dtype=dtype, device=device)
    Phi[:, 0] = 1
    Phi[:, 1] = 2 * x
    for j in range(1, n - 1):
        Phi[:, j + 1] = 2 * (x * Phi[:, j] - j * Phi[:, j - 1])

    Phi[:, 0] = w * Phi[:, 0] / pi_sqrt
    dPhi = torch.zeros(m, 2 * n, dtype=dtype, device=device)
    dPhi[:, 0] = dw / pi_sqrt
    dPhi[:, 1] = dPhi[:, 0]

    f = 1
    for j in range(1, n):
        f *= j
        Phi[:, j] = w * Phi[:, j] / \
            torch.sqrt(torch.tensor(2 ** j * f, dtype=dtype, device=device)) / pi_sqrt
        dPhi[:, 2 * j] = torch.sqrt(torch.tensor(2 * j, dtype=dtype, device=device)) * Phi[:, j - 1] - x * Phi[:, j]
        dPhi[:, 2 * j + 1] = dPhi[:, 2 * j]

    t = t[:, None]
    dPhi[:, 0::2] = dPhi[:, 0::2] * (t - translation * m / 2)
    dPhi[:, 1::2] = -dPhi[:, 1::2] * dilation * m / 2

    # ind
    ind = torch.zeros((2, 2 * n), dtype=torch.int64, device=device)
    ind[0, 0::2] = torch.arange(n, dtype=torch.int64, device=device)
    ind[0, 1::2] = torch.arange(n, dtype=torch.int64, device=device)
    ind[1, 0::2] = torch.zeros((1, n), dtype=torch.int64, device=device)
    ind[1, 1::2] = torch.ones((1, n), dtype=torch.int64, device=device)

    return Phi, dPhi, ind


class vpfun(Function):
    """Performs orthogonal projection, i.e. projects the input 'x' to the
    space spanned by the columns of 'Phi', where the matrix 'Phi' is provided by the 'ada' function.

    Input
    ----------
    x: torch Tensor of size (N,C,L) where
        N is the batch_size,
        C is the number of channels, and
        L is the number of signal samples
    params: torch Tensor of floats
          Contains the nonlinear parameters of the function system stored in Phi.
          For instance, if Phi(params) is provided by 'ada_hermite',
          then 'params' is a tensor of size (2,) that contains the dilation and the translation
          parameters of the Hermite functions.
    ada: callable
        Builder for the function system. For a given set of parameters 'params',
        it computes the matrix Phi(params) and its derivatives dPhi(params).
        For instance, in this package 'ada = ada_hermite' could be used.
    device: torch.device
             The desired device of the returned tensor(s).
    penalty: L2 regularization penalty that is added to the training loss.
              For instance, in the case of classification, the training loss is calculated as

                  loss = cross-entropy loss + penalty * ||x - projected_input||_2 / ||x||_2,

              where the projected_input is equal to the orthogonal projection of
              the 'x' to the columnspace of 'Phi(params)',
              i.e., projected_input =  Phi.mm( torch.linalg.pinv(Phi(params).mm(x) )

    Output
    -------
    coeffs: torch Tensor
             Coefficients of the projected input signal:

                 projected_input =  Phi.mm( torch.linalg.pinv(Phi(params).mm(x) ),

             where coeffs = torch.linalg.pinv(Phi(params).mm(x)
    """

    @staticmethod
    def forward(ctx, x, params, ada, device, penalty):
        ctx.device = device
        ctx.penalty = penalty
        phi, dphi, ind = ada(params)
        phip = torch.linalg.pinv(phi)
        coeffs = phip @ torch.transpose(x, 1, 2)
        y_est = torch.transpose(phi @ coeffs, 1, 2)
        nparams = torch.tensor(max(params.shape))
        ctx.save_for_backward(x, phi, phip, dphi, ind, coeffs, y_est, nparams)

        return torch.transpose(coeffs, 1, 2)

    @staticmethod
    def backward(ctx, dy):
        x, phi, phip, dphi, ind, coeffs, y_est, nparams = ctx.saved_tensors
        dx = dy @ phip
        dp = None
        wdphi_r = (x - y_est) @ dphi
        phipc = torch.transpose(phip, -1, -2) @ coeffs  # (N,L,C)

        batch = x.shape[0]
        t2 = torch.zeros(
            batch, 1, phi.shape[1], nparams, dtype=x.dtype, device=ctx.device)
        jac1 = torch.zeros(
            batch, 1, phi.shape[0], nparams, dtype=x.dtype, device=ctx.device)
        jac3 = torch.zeros(
            batch, 1, phi.shape[1], nparams, dtype=x.dtype, device=ctx.device)
        for j in range(nparams):
            rng = ind[1, :] == j
            indrows = ind[0, rng]
            jac1[:, :, :, j] = torch.transpose(dphi[:, rng] @ coeffs[:, indrows, :], 1, 2)  # (N,C,L)
            t2[:, :, indrows, j] = wdphi_r[:, :, rng]
            jac3[:, :, indrows, j] = torch.transpose(phipc, 1, 2) @ dphi[:, rng]

        # Jacobian matrix of the forward pass with respect to the nonlinear parameters 'params'
        jac = -phip @ jac1 + phip @ (torch.transpose(phip, -1, -2) @ t2) + jac3 - phip @ (phi @ jac3)
        dy = dy.unsqueeze(-1)
        res = (x - y_est) / (x ** 2).sum(dim=2, keepdim=True)
        res = res.unsqueeze(-1)
        dp = (jac * dy).mean(dim=0).sum(dim=1) - 2 * \
            ctx.penalty * (jac1 * res).mean(dim=0).sum(dim=1)

        return dx, dp, None, None, None


class vp_layer(nn.Module):
    """Basic Variable Projection (VP) layer class.
    The output of a single VP operator is forwarded to the subsequent layers.

        Input
        ----------
        ada: callable
            Builder for the function system and its derivatives (see e.g., 'ada_hermite').
        n_in: int
            Input dimension of the VP layer.
        n_out: int
            Output dimension of the VP layer.
        nparams: int
            Number of trainable weights,
            e.g., nparams=2 in the case of 'ada_hermite' function.
        penalty: L2 regularization penalty that is added directily to the training loss (see e.g., 'vpfun').
            This can be intepreted as a skip connection from this layer to the cost function. Default: 0.0.
        device: torch.device. Default: None.
            The desired device of the returned tensor(s).
        init: a list of values to initialize the VP layer.
            Default for Hermite functions: init=[0.1, 0.0].
        """

    def __init__(self, ada, n_in, n_out, nparams, penalty=0.0, dtype=torch.float, device=None, init=None):
        if init is None:
            init = [0.1, 0.0]
        super().__init__()
        self.device = device
        self.n_in = n_in
        self.n_out = n_out
        self.nparams = nparams
        self.penalty = penalty
        self.ada = lambda params: ada(n_in, n_out, params, dtype=dtype, device=self.device)
        self.weight = nn.Parameter(torch.tensor(init))

    def forward(self, input):
        return vpfun.apply(input, self.weight, self.ada, self.device, self.penalty)


class tdvp_layer(nn.Module):
    """Time Domain Variable Projection (VP) layer class. Same as 'vp_layer', but the output of
    a single VP operator is repeated 'td' number of times and then forwarded to the subsequent layers.

        Input
        ----------
        ada: callable
            Builder for the function system and its derivatives (see e.g., 'ada_hermite').
        n_in: int
            Input dimension of the VP layer.
        n_out: int
            Output dimension of the VP layer.
        nparams: int
            Number of trainable weights,
            e.g., nparams=2 in the case of 'ada_hermite' function.
        penalty: L2 regularization penalty that is added directily to the training loss (see e.g., 'vpfun').
            This can be intepreted as a skip connection from this layer to the cost function. Default: 0.0.
        device: torch.device. Default: None.
            The desired device of the returned tensor(s).
        init: a list of values to initialize the VP layer.
            Default for Hermite functions: init=[0.1, 0.0].
        td: int
            Time dimension to support spiking neural activations. Default: 0.
        """

    def __init__(self, ada, n_in, n_out, nparams, penalty=0.0, dtype=torch.float, device=None, init=None, td=0):
        if init is None:
            init = [0.1, 0.0]
        super().__init__()
        self.device = device
        self.n_in = n_in
        self.n_out = n_out
        self.nparams = nparams
        self.penalty = penalty
        self.ada = lambda params: ada(n_in, n_out, params, dtype=dtype, device=self.device)
        self.weight = nn.Parameter(torch.tensor(init))
        self.td = td

    def forward(self, input):
        vp_out = vpfun.apply(input, self.weight, self.ada, self.device, self.penalty)

        return vp_out.unsqueeze(-1).repeat(1, 1, 1, self.td)


class tdmvp_layer(nn.Module):
    """Multiple Time Domain Variable Projection (VP) layer class. Same as 'vp_layer', but the output of
    multiple VP operators are connected in parallel, whose outputs are forwarded to the subsequent layers.
    In other words, the output channel of this layer is formed by concatenating 'td' number of different VP layers'.

        Input
        ----------
        ada: callable
            Builder for the function system and its derivatives (see e.g., 'ada_hermite').
        n_in: int
            Input dimension of the VP layer.
        n_out: int
            Output dimension of the VP layer.
        nparams: int
            Number of trainable weights,
            e.g., nparams=2 in the case of 'ada_hermite' function.
        penalty: L2 regularization penalty that is added directily to the training loss (see e.g., 'vpfun').
            This can be intepreted as a skip connection from this layer to the cost function. Default: 0.0.
        device: torch.device. Default: None.
            The desired device of the returned tensor(s).
        init: a list of values to initialize the VP layer.
            Default for Hermite functions: init=[0.1, 0.0].
        td: int
            Time dimension to support spiking neural activations. Default: 0.
        m_vp: int
            Number of basic VP layers to be concatenated. Default: 1.
        """

    def __init__(self, ada, n_in, n_out, nparams,
                 penalty=0.0, dtype=torch.float,
                 device=None, init=None, td=0,
                 m_vp=1
                 ):
        if init is None:
            init = [0.1, 0.0]
        super().__init__()
        self.device = device
        self.n_in = n_in
        self.n_out = n_out
        self.nparams = nparams
        self.penalty = penalty
        self.m_vp = m_vp
        self.ada = [lambda params: ada(n_in, n_out, params, dtype=dtype, device=self.device) for _ in range(m_vp)]
        if self.m_vp * 2 != len(init):
            self.weight = nn.Parameter(torch.tensor(init).unsqueeze(0).repeat(self.m_vp, 1))
        else:
            self.weight = nn.Parameter(torch.tensor(init).reshape(m_vp, 2))
        self.td = td

    def forward(self, input):
        batch_size = input.size(0)
        mvp_out = torch.zeros((batch_size, 1, self.n_out, self.m_vp))
        for i in range(self.m_vp):
            mvp_out[:, :, :, i] = vpfun.apply(input, self.weight[i, :], self.ada[i], self.device, self.penalty)

        return mvp_out


class temporal_tdvp_layer(nn.Module):
    """Temporally scaled Time Domain Variable Projection (VP) layer class. Same as 'tdvp_layer', but instead of simply
       repeating the output of a single VP operator, the temporal repetitions are scaled by factors ranging from 0.25 to 1.

        Input
        ----------
        ada: callable
            Builder for the function system and its derivatives (see e.g., 'ada_hermite').
        n_in: int
            Input dimension of the VP layer.
        n_out: int
            Output dimension of the VP layer.
        nparams: int
            Number of trainable weights,
            e.g., nparams=2 in the case of 'ada_hermite' function.
        penalty: L2 regularization penalty that is added directily to the training loss (see e.g., 'vpfun').
            This can be intepreted as a skip connection from this layer to the cost function. Default: 0.0.
        device: torch.device. Default: None.
            The desired device of the returned tensor(s).
        init: a list of values to initialize the VP layer.
            Default for Hermite functions: init=[0.1, 0.0].
        td: int
            Time dimension to support spiking neural activations. Default: 0.
        """

    def __init__(self, ada, n_in, n_out, nparams, penalty=0.0, dtype=torch.float, device=None, init=None, td=0):
        if init is None:
           init = [0.1, 0.0]
        super().__init__()
        self.device = device
        self.n_in = n_in
        self.n_out = n_out
        self.nparams = nparams
        self.penalty = penalty
        self.ada = lambda params: ada(n_in, n_out, params, dtype=dtype, device=self.device)
        self.weight = nn.Parameter(torch.tensor(init))
        self.td = td
        self.attenuation = np.linspace(0.25, 1, self.td, endpoint=True)

    def forward(self, input):
        vp_out = vpfun.apply(input, self.weight, self.ada, self.device, self.penalty)
        tdvp_out = vp_out.unsqueeze(-1).repeat(1, 1, 1, self.td)
        for i in range(self.td):
            tdvp_out[:, :, :, i] = tdvp_out[:, :, :, i] * self.attenuation[i]

        return tdvp_out


class temporal_tdmvp_layer(nn.Module):
    """Temporally scaled Multiple Time Domain Variable Projection (VP) layer class. Same as 'tdmvp_layer', but instead of
       simply concatenating the outputs of multiple VP operators, the VP outputs are scaled by factors ranging from 0.25 to 1.

        Input
        ----------
        ada: callable
            Builder for the function system and its derivatives (see e.g., 'ada_hermite').
        n_in: int
            Input dimension of the VP layer.
        n_out: int
            Output dimension of the VP layer.
        nparams: int
            Number of trainable weights,
            e.g., nparams=2 in the case of 'ada_hermite' function.
        penalty: L2 regularization penalty that is added directily to the training loss (see e.g., 'vpfun').
            This can be intepreted as a skip connection from this layer to the cost function. Default: 0.0.
        device: torch.device. Default: None.
            The desired device of the returned tensor(s).
        init: a list of values to initialize the VP layer.
            Default for Hermite functions: init=[0.1, 0.0].
        td: int
            Time dimension to support spiking neural activations. Default: 0.
        """

    def __init__(self, ada, n_in, n_out, nparams,
                 penalty=0.0, dtype=torch.float, device=None,
                 init=None, td=0, m_vp=1
                 ):

        if init is None:
            init = [0.1, 0.0]
        super().__init__()
        self.device = device
        self.n_in = n_in
        self.n_out = n_out
        self.nparams = nparams
        self.penalty = penalty
        self.m_vp = m_vp
        self.ada = []
        self.attenuation = np.linspace(0.25, 1, self.m_vp, endpoint=True)
        self.ada.extend(lambda params: ada(n_in, n_out, params, dtype=dtype, device=self.device) for _ in range(m_vp))
        if self.m_vp * 2 != len(init):
            self.weight = nn.Parameter(torch.tensor(
                init).unsqueeze(0).repeat(self.m_vp, 1))
        else:
            self.weight = nn.Parameter(torch.tensor(init).reshape(m_vp, 2))
        self.td = td

    def forward(self, input):
        batch_size = input.size(0)
        mvp_out = torch.zeros((batch_size, 1, self.n_out, self.m_vp))
        for i in range(self.m_vp):
            mvp_out[:, :, :, i] = vpfun.apply(input * self.attenuation[i], self.weight[i, :], self.ada[i], self.device,
                                              self.penalty)

        return mvp_out
