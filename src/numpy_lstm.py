"""
A minimal LSTM implemented from scratch in NumPy.

Why build this instead of using a library?
--------------------------------------------
The whole point of today's session is to understand how the network in the
paper actually works, not just call nn.LSTM(). Every equation below is the
*standard* LSTM formulation (Hochreiter & Schmidhuber 1997 / Gers et al.),
which is exactly what Keras/PyTorch's nn.LSTM computes under the hood.

At each timestep t, given input x_t and the previous hidden/cell state
(h_{t-1}, c_{t-1}), an LSTM computes four "gates":

    f_t = sigmoid(W_f x_t + U_f h_{t-1} + b_f)      # forget gate
    i_t = sigmoid(W_i x_t + U_i h_{t-1} + b_i)      # input gate
    g_t = tanh   (W_g x_t + U_g h_{t-1} + b_g)      # candidate cell content
    o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)      # output gate

    c_t = f_t * c_{t-1} + i_t * g_t                 # new cell state
    h_t = o_t * tanh(c_t)                           # new hidden state

Intuition:
  - forget gate f_t:  how much of the OLD memory (c_{t-1}) to keep
  - input gate i_t:   how much of the NEW candidate info (g_t) to write in
  - cell state c_t:   the network's long-term memory conveyor belt
  - output gate o_t:  how much of the memory to expose as the hidden state

We run this for every cycle in the input window (e.g. 30 capacity values),
and feed the FINAL hidden state h_T into a linear layer to produce one
number: the predicted RUL (or SOH).

Training: plain backprop-through-time (BPTT), the manual gradient equations
for each gate, updated with Adam. Everything is batched (first axis = batch)
for speed, using only numpy.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class NumpyLSTMRegressor:
    """Many-to-one LSTM: sequence in, single scalar out."""

    def __init__(self, input_dim, hidden_dim, seed=0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)

        def glorot(shape):
            limit = np.sqrt(6 / sum(shape))
            return rng.uniform(-limit, limit, size=shape)

        H, D = hidden_dim, input_dim
        # Stack the 4 gates (f, i, g, o) into single matrices for speed.
        self.Wx = glorot((D, 4 * H))       # input -> gates
        self.Wh = glorot((H, 4 * H))       # hidden -> gates
        self.b = np.zeros(4 * H)
        self.b[H:2 * H] = 1.0              # forget-gate bias trick: start
                                            # "remembering by default", a
                                            # well-known LSTM init trick that
                                            # makes early training much more
                                            # stable.
        self.Wy = glorot((H, 1))           # hidden -> output (RUL / SOH)
        self.by = np.zeros(1)

        self.params = ["Wx", "Wh", "b", "Wy", "by"]
        self._m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self._t = 0

    def forward(self, X):
        """X: (batch, seq_len, input_dim) -> caches everything needed for backward."""
        B, T, D = X.shape
        H = self.hidden_dim
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        cache = {"X": X, "h": [h], "c": [c], "gates": []}

        for t in range(T):
            x_t = X[:, t, :]
            z = x_t @ self.Wx + h @ self.Wh + self.b  # (B, 4H)
            f = sigmoid(z[:, 0 * H:1 * H])
            i = sigmoid(z[:, 1 * H:2 * H])
            g = np.tanh(z[:, 2 * H:3 * H])
            o = sigmoid(z[:, 3 * H:4 * H])

            c = f * c + i * g
            h = o * np.tanh(c)

            cache["gates"].append((f, i, g, o))
            cache["h"].append(h)
            cache["c"].append(c)

        y_pred = h @ self.Wy + self.by  # use final hidden state
        cache["y_pred"] = y_pred
        return y_pred, cache

    def backward(self, cache, y_true):
        """MSE loss. Returns grads dict + loss."""
        X = cache["X"]
        B, T, D = X.shape
        H = self.hidden_dim

        y_pred = cache["y_pred"]
        diff = (y_pred - y_true.reshape(-1, 1))  # (B,1)
        loss = float(np.mean(diff ** 2))

        grads = {p: np.zeros_like(getattr(self, p)) for p in self.params}

        dY = (2.0 / B) * diff  # dL/dy_pred
        h_last = cache["h"][-1]
        grads["Wy"] = h_last.T @ dY
        grads["by"] = dY.sum(axis=0)

        dh_next = dY @ self.Wy.T  # gradient flowing into hidden state
        dc_next = np.zeros((B, H))

        for t in reversed(range(T)):
            f, i, g, o = cache["gates"][t]
            c_t = cache["c"][t + 1]
            c_prev = cache["c"][t]
            x_t = X[:, t, :]
            h_prev = cache["h"][t]

            tanh_c = np.tanh(c_t)
            do = dh_next * tanh_c
            dc = dh_next * o * (1 - tanh_c ** 2) + dc_next

            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_prev = dc * f

            # gate raw (pre-activation) gradients
            df_raw = df * f * (1 - f)
            di_raw = di * i * (1 - i)
            dg_raw = dg * (1 - g ** 2)
            do_raw = do * o * (1 - o)

            dz = np.concatenate([df_raw, di_raw, dg_raw, do_raw], axis=1)  # (B,4H)

            grads["Wx"] += x_t.T @ dz
            grads["Wh"] += h_prev.T @ dz
            grads["b"] += dz.sum(axis=0)

            dh_next = dz @ self.Wh.T
            dc_next = dc_prev

        for p in self.params:
            np.clip(grads[p], -5, 5, out=grads[p])  # gradient clipping: LSTMs
                                                      # trained with BPTT can
                                                      # get exploding gradients
        return grads, loss

    def adam_step(self, grads, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for p in self.params:
            g = grads[p]
            self._m[p] = beta1 * self._m[p] + (1 - beta1) * g
            self._v[p] = beta2 * self._v[p] + (1 - beta2) * (g ** 2)
            m_hat = self._m[p] / (1 - beta1 ** self._t)
            v_hat = self._v[p] / (1 - beta2 ** self._t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, p, getattr(self, p) - update)

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return y_pred.ravel()