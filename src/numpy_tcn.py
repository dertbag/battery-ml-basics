"""
A minimal Temporal Convolutional Network (TCN), implemented from scratch in
NumPy -- same philosophy as numpy_lstm.py: every operation visible, no
framework black box.

Why the paper reaches for a TCN at all
---------------------------------------
An LSTM processes a sequence one timestep at a time, carrying information
forward through a hidden state -- that's inherently SEQUENTIAL (step 30
depends on step 29 depends on step 28...). Two consequences:
  1. It's slow to train (can't parallelize across time).
  2. Gradients have to flow backward through many multiplications
     (backprop-through-time), which is what causes the vanishing/exploding
     gradient problems LSTMs were specifically designed to fight.

A TCN instead uses 1D CONVOLUTIONS with DILATION. Instead of "remembering"
the past through a running state, it looks directly at specific past
timesteps using a sliding filter -- like an LSTM's context window, but
computed all at once (parallelizable) via convolution.

Causal convolution: output at time t may only depend on inputs at time
<= t (never the future) -- we enforce this with left-padding.

Dilation: layer 1 looks at every timestep (dilation=1), layer 2 looks at
every 2nd timestep back (dilation=2), layer 3 every 4th (dilation=4), etc.
Stacking dilated layers lets a small kernel (size 3 here) cover a huge
receptive field with few layers -- with dilations [1,2,4,8] and kernel 3,
receptive field = 1 + 2*(1+2+4+8) = 31, which comfortably covers our
30-cycle input window in 4 layers.

Residual connection: each block adds its own input back to its output
(y = x + F(x)), which is what makes it feasible to stack several of these
layers without gradients vanishing -- same idea as ResNets.
"""
import numpy as np


def relu(x):
    return np.maximum(0, x)


class CausalConv1D:
    """
    y[:, o, t] = sum_i sum_k W[o, i, k] * x_padded[:, i, t + k*dilation] + b[o]

    x: (batch, in_channels, seq_len)  ->  y: (batch, out_channels, seq_len)
    Left-pads with (kernel_size-1)*dilation zeros so the output is causal
    (only sees present and past) and same length as the input.
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation, seed=0):
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6 / (in_ch * kernel_size + out_ch))
        self.W = rng.uniform(-limit, limit, size=(out_ch, in_ch, kernel_size))
        self.b = np.zeros(out_ch)
        self.k = kernel_size
        self.d = dilation
        self.pad = (kernel_size - 1) * dilation

        self._mW = np.zeros_like(self.W)
        self._vW = np.zeros_like(self.W)
        self._mb = np.zeros_like(self.b)
        self._vb = np.zeros_like(self.b)
        self._t = 0

    def forward(self, x):
        B, Cin, T = x.shape
        xp = np.pad(x, ((0, 0), (0, 0), (self.pad, 0)))  # left pad only (causal)
        y = np.zeros((B, self.W.shape[0], T))
        for kk in range(self.k):
            offset = kk * self.d
            # xp[:, :, offset : offset+T] is the slice of the padded input
            # that tap kk of the kernel sees, for every output timestep at once
            x_slice = xp[:, :, offset:offset + T]           # (B, Cin, T)
            y += np.einsum("oi,bit->bot", self.W[:, :, kk], x_slice)
        y += self.b[None, :, None]
        self._cache = (x, xp)
        return y

    def backward(self, dY):
        x, xp = self._cache
        B, Cin, T = x.shape
        dW = np.zeros_like(self.W)
        db = dY.sum(axis=(0, 2))
        dxp = np.zeros_like(xp)
        for kk in range(self.k):
            offset = kk * self.d
            x_slice = xp[:, :, offset:offset + T]
            dW[:, :, kk] = np.einsum("bot,bit->oi", dY, x_slice)
            dxp[:, :, offset:offset + T] += np.einsum("oi,bot->bit", self.W[:, :, kk], dY)
        dx = dxp[:, :, self.pad:]  # undo the left padding
        self.grads = (dW, db)
        return dx

    def adam_step(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        dW, db = self.grads
        np.clip(dW, -5, 5, out=dW)
        np.clip(db, -5, 5, out=db)
        for name, param, grad, m_attr, v_attr in [
            ("W", self.W, dW, "_mW", "_vW"), ("b", self.b, db, "_mb", "_vb")
        ]:
            m = getattr(self, m_attr)
            v = getattr(self, v_attr)
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** self._t)
            v_hat = v / (1 - beta2 ** self._t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)


class ResidualBlock:
    """Two causal conv layers + ReLU, with a residual (skip) connection."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, seed=0):
        self.conv1 = CausalConv1D(in_ch, out_ch, kernel_size, dilation, seed=seed)
        self.conv2 = CausalConv1D(out_ch, out_ch, kernel_size, dilation, seed=seed + 1)
        # 1x1 conv to match channel dims for the residual add, only if needed
        self.need_proj = in_ch != out_ch
        if self.need_proj:
            self.proj = CausalConv1D(in_ch, out_ch, kernel_size=1, dilation=1, seed=seed + 2)

    def forward(self, x):
        h = relu(self.conv1.forward(x))
        h2 = self.conv2.forward(h)
        self._relu1_mask = (h > 0)
        residual = self.proj.forward(x) if self.need_proj else x
        out = relu(h2 + residual)
        self._relu_out_mask = (h2 + residual > 0)
        self._residual_is_x = not self.need_proj
        return out

    def backward(self, dOut):
        dPreRelu = dOut * self._relu_out_mask
        dH2 = dPreRelu
        dResidual = dPreRelu
        dH = self.conv2.backward(dH2)
        dH_pre = dH * self._relu1_mask
        dX_from_conv1 = self.conv1.backward(dH_pre)
        if self.need_proj:
            dX_from_proj = self.proj.backward(dResidual)
            dX = dX_from_conv1 + dX_from_proj
        else:
            dX = dX_from_conv1 + dResidual
        return dX

    def adam_step(self, lr=0.01):
        self.conv1.adam_step(lr)
        self.conv2.adam_step(lr)
        if self.need_proj:
            self.proj.adam_step(lr)


class NumpyTCNRegressor:
    """Stack of dilated residual blocks -> take last timestep -> linear -> scalar."""

    def __init__(self, input_dim, hidden_channels=8, kernel_size=3,
                 dilations=(1, 2, 4, 8), seed=0):
        self.blocks = []
        in_ch = input_dim
        for i, d in enumerate(dilations):
            block = ResidualBlock(in_ch, hidden_channels, kernel_size, d, seed=seed + i * 10)
            self.blocks.append(block)
            in_ch = hidden_channels

        rng = np.random.default_rng(seed + 999)
        limit = np.sqrt(6 / (hidden_channels + 1))
        self.Wy = rng.uniform(-limit, limit, size=(hidden_channels, 1))
        self.by = np.zeros(1)
        self._mWy = np.zeros_like(self.Wy)
        self._vWy = np.zeros_like(self.Wy)
        self._mby = np.zeros_like(self.by)
        self._vby = np.zeros_like(self.by)
        self._t = 0

    def forward(self, X):
        """X: (batch, seq_len, input_dim) -> y_pred: (batch, 1)"""
        x = np.transpose(X, (0, 2, 1))  # -> (batch, channels, seq_len)
        for block in self.blocks:
            x = block.forward(x)
        self._last_feat_map = x           # (batch, hidden_channels, seq_len)
        last_step = x[:, :, -1]           # causal: last timestep summarizes the whole window
        self._last_step = last_step
        y_pred = last_step @ self.Wy + self.by
        return y_pred

    def backward(self, y_true):
        y_pred = self._last_step @ self.Wy + self.by
        B = y_true.shape[0]
        diff = y_pred - y_true.reshape(-1, 1)
        loss = float(np.mean(diff ** 2))
        dY = (2.0 / B) * diff

        dWy = self._last_step.T @ dY
        dby = dY.sum(axis=0)
        d_last_step = dY @ self.Wy.T

        dx = np.zeros_like(self._last_feat_map)
        dx[:, :, -1] = d_last_step

        for block in reversed(self.blocks):
            dx = block.backward(dx)

        self._grads_y = (dWy, dby)
        return loss

    def adam_step(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        dWy, dby = self._grads_y
        np.clip(dWy, -5, 5, out=dWy)
        np.clip(dby, -5, 5, out=dby)
        for param, grad, m_attr, v_attr in [
            (self.Wy, dWy, "_mWy", "_vWy"), (self.by, dby, "_mby", "_vby")
        ]:
            m = getattr(self, m_attr)
            v = getattr(self, v_attr)
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** self._t)
            v_hat = v / (1 - beta2 ** self._t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)
        for block in self.blocks:
            block.adam_step(lr)

    def predict(self, X):
        return self.forward(X).ravel()
