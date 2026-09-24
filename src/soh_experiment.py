"""
SOH experiment, following the paper's approach (simplified for time):

For each discharge cycle, take the raw (Voltage_measured, Current_measured,
Temperature_measured) curves recorded during that discharge, and resample
them (via linear interpolation over normalized time 0->1) to a FIXED number
of points -- the paper uses 660 points; we use 100 here to keep training
fast in an hour, same idea, coarser resolution. This turns every cycle,
regardless of how many raw samples it happened to have, into a fixed-size
(100, 3) sequence: 100 timesteps x [voltage, current, temperature].

SOH = discharge capacity / rated capacity (2.0 Ah).

The LSTM (same implementation as numpy_lstm.py, just with input_dim=3
instead of 1) reads the whole 100-step curve and outputs a single SOH
number.

Same battery-level split as the RUL experiments: train on B0005 + B0006,
test entirely held out on B0018.

Caveat worth remembering (same theme as the RUL experiments): B0005, B0006,
and B0018 have different discharge voltage cutoffs (2.7V, 2.5V, 2.5V
respectively). B0018 happens to share B0006's cutoff, so this split is
slightly "easier" than it would be against B0005's cutoff. B0007 (2.2V
cutoff) is left out entirely here too, to avoid training on a mix of
cutoffs that could teach the model to key off the cutoff artifact rather
than real degradation.
"""
import numpy as np
import scipy.io as sio
from numpy_lstm import NumpyLSTMRegressor

RATED_CAPACITY = 2.0
N_POINTS = 100  # paper uses 660; smaller here to keep training time short


def load_discharge_cycles(mat_path, battery_key):
    m = sio.loadmat(mat_path, simplify_cells=True)
    cycles = m[battery_key]["cycle"]
    return [c["data"] for c in cycles if c["type"] == "discharge"]


def resample_curve(values, n_points=N_POINTS):
    t_orig = np.linspace(0, 1, len(values))
    t_new = np.linspace(0, 1, n_points)
    return np.interp(t_new, t_orig, values)


def build_soh_dataset(mat_path, battery_key, n_points=N_POINTS):
    discharges = load_discharge_cycles(mat_path, battery_key)
    X, y = [], []
    for d in discharges:
        v = resample_curve(d["Voltage_measured"], n_points)
        i = resample_curve(d["Current_measured"], n_points)
        t = resample_curve(d["Temperature_measured"], n_points)
        X.append(np.stack([v, i, t], axis=-1))  # (n_points, 3)
        y.append(d["Capacity"] / RATED_CAPACITY)  # SOH
    return np.array(X), np.array(y)


def pct_within(y_true, y_pred, frac):
    err = np.abs(y_pred - y_true) / np.maximum(y_true, 1e-6)
    return float(np.mean(err <= frac)) * 100


def report(name, yt, yp):
    rmse = np.sqrt(np.mean((yp - yt) ** 2))
    mae = np.mean(np.abs(yp - yt))
    p5 = pct_within(yt, yp, 0.05)
    p10 = pct_within(yt, yp, 0.10)
    print(f"[{name:12s}] RMSE={rmse:.4f} (SOH units) | MAE={mae:.4f} | "
          f"within 5%={p5:5.1f}% | within 10%={p10:5.1f}%")
    return dict(rmse=rmse, mae=mae, p5=p5, p10=p10)


def main():
    X_b5, y_b5 = build_soh_dataset("data/B0005.mat", "B0005")
    X_b6, y_b6 = build_soh_dataset("data/B0006.mat", "B0006")
    X_b18, y_b18 = build_soh_dataset("data/B0018.mat", "B0018")

    print(f"B0005: {len(X_b5)} cycles | B0006: {len(X_b6)} cycles | "
          f"B0018: {len(X_b18)} cycles")

    X_train = np.concatenate([X_b5, X_b6], axis=0)
    y_train = np.concatenate([y_b5, y_b6], axis=0)
    X_test, y_test = X_b18, y_b18

    print(f"Train pool (B0005+B0006): {len(X_train)} cycles")
    print(f"Held-out test (B0018):    {len(X_test)} cycles\n")

    # normalize each channel independently, using TRAIN stats only
    ch_mean = X_train.mean(axis=(0, 1), keepdims=True)  # (1,1,3)
    ch_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    def norm_x(a):
        return (a - ch_mean) / ch_std

    Xtr, Xte = norm_x(X_train), norm_x(X_test)
    ytr = y_train  # SOH is already O(1), no need to rescale

    model = NumpyLSTMRegressor(input_dim=3, hidden_dim=16, seed=42)

    epochs = 300
    batch_size = 16
    n = len(Xtr)
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            y_pred, cache = model.forward(xb)
            grads, loss = model.backward(cache, yb)
            model.adam_step(grads, lr=0.005)
            epoch_loss += loss * len(idx)
        epoch_loss /= n
        if epoch % 50 == 0 or epoch == epochs - 1:
            test_pred = model.predict(Xte)
            test_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))
            print(f"epoch {epoch:4d}  train_loss={epoch_loss:.6f}  "
                  f"held_out_RMSE(SOH)={test_rmse:.4f}")

    print()
    train_pred = model.predict(Xtr)
    test_pred = model.predict(Xte)
    report("TRAIN (B5+B6)", y_train, train_pred)
    test_metrics = report("TEST (B0018)", y_test, test_pred)

    np.savez("soh_results.npz",
             y_train=y_train, train_pred=train_pred,
             y_test=y_test, test_pred=test_pred,
             cycles_test=np.arange(len(y_test)))
    return test_metrics


if __name__ == "__main__":
    main()
