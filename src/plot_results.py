"""
Generates 4 diagnostic plots from the results saved by
rul_experiment_multibattery.py. Run that script first, then run this one
in the same directory (it reads rul_results_multibattery.npz).

Saves PNG files instead of calling plt.show(), since interactive matplotlib
windows can hang/crash in some environments (as you saw). Open the PNGs
from your file browser / editor instead.

Produces:
  1. capacity_fade.png   - raw degradation curves for all 3 batteries,
                            with the EOL threshold marked. This is the
                            rawest possible view of "what the model is
                            trying to learn from."
  2. rul_prediction.png  - true vs. predicted RUL over B0018's life
                            (the held-out test battery). This is the
                            main "did it work" plot.
  3. parity_plot.png     - predicted vs. true RUL as a scatter, train
                            and test overlaid, with a y=x reference line.
                            Points on the line = perfect predictions.
                            Shows train (tight, low error) vs test
                            (looser, real generalization error) side by
                            side in one picture.
  4. residuals.png       - prediction error (pred - true) vs. cycle
                            number for the held-out battery. Flat near
                            zero = good. Look for where errors cluster;
                            that tells you WHERE in the battery's life
                            the model struggles, not just how much.
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

EOL_CAPACITY = 1.4


def load_discharge_capacities(mat_path, battery_key):
    m = sio.loadmat(mat_path, simplify_cells=True)
    cycles = m[battery_key]["cycle"]
    return np.array([c["data"]["Capacity"] for c in cycles if c["type"] == "discharge"])


def plot_capacity_fade(d):
    fig, ax = plt.subplots(figsize=(8, 5))
    for bid, caps, eol_idx, color in [
        ("B0005 (train)", d["caps_b5"], int(d["eol_idx_b5"]), "tab:blue"),
        ("B0006 (train)", d["caps_b6"], int(d["eol_idx_b6"]), "tab:orange"),
        ("B0018 (held-out test)", d["caps_b18"], int(d["eol_idx_b18"]), "tab:green"),
    ]:
        cycles = np.arange(len(caps))
        ax.plot(cycles, caps, label=bid, color=color)
        ax.scatter([eol_idx], [caps[eol_idx]], color=color, marker="x", s=80, zorder=5)
    ax.axhline(EOL_CAPACITY, color="gray", linestyle="--", linewidth=1,
               label="EOL threshold (1.4 Ah)")
    ax.set_xlabel("Discharge cycle")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title("Capacity fade: what the model learns from (x = end-of-life point)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("capacity_fade.png", dpi=150)
    plt.close(fig)
    print("saved capacity_fade.png")


def plot_rul_prediction(d):
    t_test, y_test, test_pred = d["t_test"], d["y_test"], d["test_pred"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_test, y_test, label="True RUL", color="black", linewidth=2)
    ax.plot(t_test, test_pred, label="Predicted RUL", color="tab:red",
             linestyle="--", marker="o", markersize=3)
    ax.set_xlabel("Discharge cycle (B0018)")
    ax.set_ylabel("RUL (cycles remaining)")
    ax.set_title("Held-out test: B0018 (never seen during training)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("rul_prediction.png", dpi=150)
    plt.close(fig)
    print("saved rul_prediction.png")


def plot_parity(d):
    y_train, train_pred = d["y_train"], d["train_pred"]
    y_test, test_pred = d["y_test"], d["test_pred"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_train, train_pred, alpha=0.5, label="Train (B0005+B0006)",
               color="tab:blue", s=25)
    ax.scatter(y_test, test_pred, alpha=0.7, label="Test (B0018, held-out)",
               color="tab:red", s=25)
    lims = [0, max(y_train.max(), y_test.max()) * 1.05]
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title("Parity plot: closer to the diagonal = better")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig("parity_plot.png", dpi=150)
    plt.close(fig)
    print("saved parity_plot.png")


def plot_residuals(d):
    t_test, y_test, test_pred = d["t_test"], d["y_test"], d["test_pred"]
    residuals = test_pred - y_test
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, color="gray", linewidth=1)
    ax.bar(t_test, residuals, color=np.where(residuals >= 0, "tab:red", "tab:blue"), width=0.8)
    ax.set_xlabel("Discharge cycle (B0018)")
    ax.set_ylabel("Prediction error (predicted - true), cycles")
    ax.set_title("Residuals over time: red = overestimate, blue = underestimate")
    fig.tight_layout()
    fig.savefig("residuals.png", dpi=150)
    plt.close(fig)
    print("saved residuals.png")


def main():
    d = np.load("rul_results_multibattery.npz")
    plot_capacity_fade(d)
    plot_rul_prediction(d)
    plot_parity(d)
    plot_residuals(d)
    print("\nAll 4 plots saved to the current directory.")


if __name__ == "__main__":
    main()