"""
RUL experiment, proper version: train on B0005 + B0006, evaluate on a
COMPLETELY HELD-OUT battery, B0018. This is the rigorous test that the
single-battery version from before couldn't do.

B0007 is excluded: it was discharged to a much lower voltage cutoff (2.2V vs
2.5-2.7V for the others), which pulls more Ah out of every cycle before
stopping. Its capacity never numerically crosses the 1.4Ah EOL threshold in
the recorded 168 cycles, so there's no valid RUL=0 point to count down to.
This is a known quirk of this NASA subset, not a bug in our code.
"""
import numpy as np
import scipy.io as sio
from numpy_lstm import NumpyLSTMRegressor

RATED_CAPACITY = 2.0
EOL_CAPACITY = 1.4          # 30% fade from rated 2.0 Ah
WINDOW = 30


def load_discharge_capacities(mat_path, battery_key):
    m = sio.loadmat(mat_path, simplify_cells=True)
    cycles = m[battery_key]["cycle"]
    caps = np.array([c["data"]["Capacity"] for c in cycles if c["type"] == "discharge"])
    return caps


def build_rul_dataset(caps, eol_capacity=EOL_CAPACITY, window=WINDOW):
    eol_idx = int(np.argmax(caps <= eol_capacity))
    X, y, t_idx = [], [], []
    for t in range(window, eol_idx + 1):
        X.append(caps[t - window:t])
        y.append(eol_idx - t)
        t_idx.append(t)
    X = np.array(X)[:, :, None]
    y = np.array(y, dtype=float)
    return X, y, np.array(t_idx), eol_idx


def pct_within(y_true, y_pred, frac):
    denom = np.maximum(y_true, 1.0)
    err = np.abs(y_pred - y_true) / denom
    return float(np.mean(err <= frac)) * 100


def report(name, yt, yp):
    rmse = np.sqrt(np.mean((yp - yt) ** 2))
    mae = np.mean(np.abs(yp - yt))
    p5 = pct_within(yt, yp, 0.05)
    p10 = pct_within(yt, yp, 0.10)
    print(f"[{name:12s}] RMSE={rmse:6.2f} cycles | MAE={mae:6.2f} cycles | "
          f"within 5%={p5:5.1f}% | within 10%={p10:5.1f}%")
    return dict(rmse=rmse, mae=mae, p5=p5, p10=p10)


def main():
    caps = {}
    data = {}
    for bid in ["B0005", "B0006", "B0018"]:
        c = load_discharge_capacities(f"data/{bid}.mat", bid)
        caps[bid] = c
        X, y, t_idx, eol_idx = build_rul_dataset(c)
        data[bid] = dict(X=X, y=y, t_idx=t_idx, eol_idx=eol_idx)
        print(f"{bid}: {len(c)} discharge cycles, EOL at cycle {eol_idx} "
              f"(cap={c[eol_idx]:.3f} Ah), {len(X)} windowed samples")

    # --- Train pool: B0005 + B0006. Test: B0018 entirely held out. ---
    X_train = np.concatenate([data["B0005"]["X"], data["B0006"]["X"]], axis=0)
    y_train = np.concatenate([data["B0005"]["y"], data["B0006"]["y"]], axis=0)
    X_test, y_test = data["B0018"]["X"], data["B0018"]["y"]
    t_test = data["B0018"]["t_idx"]

    print(f"\nTrain pool (B0005+B0006): {len(X_train)} samples")
    print(f"Held-out test (B0018):    {len(X_test)} samples\n")

    # normalize using TRAIN stats only
    x_mean, x_std = X_train.mean(), X_train.std()
    y_max = y_train.max()

    def norm_x(a):
        return (a - x_mean) / x_std

    def norm_y(a):
        return a / y_max

    def denorm_y(a):
        return a * y_max

    Xtr, Xte = norm_x(X_train), norm_x(X_test)
    ytr = norm_y(y_train)

    model = NumpyLSTMRegressor(input_dim=1, hidden_dim=16, seed=42)

    epochs = 400
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
            model.adam_step(grads, lr=0.01)
            epoch_loss += loss * len(idx)
        epoch_loss /= n
        if epoch % 50 == 0 or epoch == epochs - 1:
            test_pred = denorm_y(model.predict(Xte))
            test_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))
            print(f"epoch {epoch:4d}  train_loss(norm)={epoch_loss:.5f}  "
                  f"held_out_RMSE(cycles)={test_rmse:.2f}")

    print()
    train_pred = denorm_y(model.predict(Xtr))
    test_pred = denorm_y(model.predict(Xte))
    report("TRAIN (B5+B6)", y_train, train_pred)
    test_metrics = report("TEST (B0018)", y_test, test_pred)

    np.savez("rul_results_multibattery.npz",
             t_test=t_test, y_test=y_test, test_pred=test_pred,
             y_train=y_train, train_pred=train_pred,
             caps_b5=caps["B0005"], eol_idx_b5=data["B0005"]["eol_idx"],
             caps_b6=caps["B0006"], eol_idx_b6=data["B0006"]["eol_idx"],
             caps_b18=caps["B0018"], eol_idx_b18=data["B0018"]["eol_idx"])
    return test_metrics


if __name__ == "__main__":
    main()