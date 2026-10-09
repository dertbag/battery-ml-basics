"""Make figures from results/dataset.csv and (if present) the experiment outputs.

Usage:  python make_plots.py --results results
Each figure is made independently; a failure in one prints a message and the rest continue.

  fig_relax_curves.png      relaxation curves of one cell at early / middle / late cycles
  fig_stats_vs_soh.png      the six statistics vs SOH, coloured by temperature (cf. Zhu Fig. 2)
  fig_ecm_vs_soh.png        the six ECM parameters vs SOH (cf. Feng Fig. 4)
  fig_rmse_by_family.png    RMSE of selected models in each split family
  fig_parity.png            predicted vs true SOH on one random cell split
  fig_trajectories.png      predicted vs true SOH over cycles for four held-out cells
  fig_importance.png        which time point of the rest curve each raw-voltage model relies on
"""
import argparse
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from soh_common import ECM_NAMES, STAT_NAMES, VCOLS


def safe(name):
    def deco(fn):
        def run(*a, **k):
            try:
                fn(*a, **k)
                print(f"saved {name}")
            except Exception as e:
                print(f"could not make {name}: {e!r}")
                traceback.print_exc(limit=1)
        return run
    return deco


@safe("fig_relax_curves.png")
def fig_relax(ds, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, T in zip(axes, sorted(ds["T_C"].unique())[:: max(1, len(ds["T_C"].unique()) - 1)][:2]):
        g = ds[ds["T_C"] == T]
        cell = g.groupby("cell_id").size().idxmax()
        c = g[g["cell_id"] == cell].sort_values("cycle")
        for frac, col in zip((0.02, 0.5, 0.98), ("tab:blue", "tab:green", "tab:red")):
            r = c.iloc[int(frac * (len(c) - 1))]
            ax.plot(120 * (np.arange(14) + 1) / 60, r[VCOLS].to_numpy(float), "o-", color=col,
                    label=f"cycle {int(r['cycle'])}, SOH {100 * r['soh']:.0f}%")
        ax.set_title(f"{cell} ({int(T)} C)")
        ax.set_xlabel("minutes after the CV charge ends")
        ax.set_ylabel("cell voltage / V")
        ax.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_relax_curves.png", dpi=130)
    plt.close()


def feature_grid(ds, names, out, fname, logy=()):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, n in zip(axes.ravel(), names):
        sc = ax.scatter(100 * ds["soh"], ds[n], c=ds["T_C"], s=3, alpha=0.4, cmap="coolwarm")
        ax.set_xlabel("SOH / % of nominal")
        ax.set_ylabel(n)
        if n in logy:
            ax.set_yscale("log")
    fig.colorbar(sc, ax=axes, label="temperature / C", shrink=0.8)
    plt.savefig(out / fname, dpi=130, bbox_inches="tight")
    plt.close()


@safe("fig_stats_vs_soh.png")
def fig_stats(ds, out):
    feature_grid(ds, STAT_NAMES, out, "fig_stats_vs_soh.png")


@safe("fig_ecm_vs_soh.png")
def fig_ecm(ds, out):
    g = ds[ds["ecm_ok"].astype(bool)]
    feature_grid(g, ECM_NAMES, out, "fig_ecm_vs_soh.png", logy=("C1", "C2"))


@safe("fig_rmse_by_family.png")
def fig_rmse(m, out):
    m = m[m["error"].fillna("") == ""]
    models = [x for x in ["urelax_linear", "cycle_cond_xgb", "xgb_stats3", "svr_stats3", "gpr_ecm6",
                          "xgb_raw14", "mlp_raw14"] if x in set(m["model"])]
    fams = [f for f in ["cell", "cell50", "temp", "condition", "time"] if f in set(m["family"])]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.8 / len(models)
    for i, name in enumerate(models):
        means, stds = [], []
        for f in fams:
            v = m[(m["family"] == f) & (m["model"] == name)]
            # for temp / condition families, plot the mean over held-out groups
            means.append(v["rmse_pct"].mean())
            stds.append(v["rmse_pct"].std() if len(v) > 1 else 0.0)
        ax.bar(np.arange(len(fams)) + i * w, means, w, yerr=np.nan_to_num(stds), label=name, capsize=2)
    ax.set_xticks(np.arange(len(fams)) + 0.4 - w / 2)
    ax.set_xticklabels(fams)
    ax.set_ylabel("RMSE / % of nominal capacity")
    ax.set_title("Error by evaluation family (bars: mean over splits; whiskers: std over splits)")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "fig_rmse_by_family.png", dpi=130)
    plt.close()


@safe("fig_parity.png")
def fig_parity(p, out):
    p = p[(p["family"] == "cell")]
    models = [x for x in ["xgb_stats3", "gpr_ecm6", "mlp_raw14", "urelax_linear"] if x in set(p["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, models):
        g = p[p["model"] == name]
        ax.scatter(100 * g["y_true"], 100 * g["y_pred"], s=3, alpha=0.3)
        lim = [65, 100]
        ax.plot(lim, lim, "r-", lw=1)
        rm = 100 * np.sqrt(np.mean((g["y_true"] - g["y_pred"]) ** 2))
        ax.set_title(f"{name}\nRMSE {rm:.2f}%")
        ax.set_xlabel("true SOH / %")
    axes[0].set_ylabel("predicted SOH / %")
    plt.tight_layout()
    plt.savefig(out / "fig_parity.png", dpi=130)
    plt.close()


@safe("fig_trajectories.png")
def fig_traj(p, out):
    p = p[(p["family"] == "cell")]
    cells = list(p["cell_id"].drop_duplicates())
    cells = sorted(cells, key=lambda c: -(p["cell_id"] == c).sum())[:4]
    models = [x for x in ["xgb_stats3", "gpr_ecm6", "mlp_raw14"] if x in set(p["model"])]
    fig, axes = plt.subplots(1, len(cells), figsize=(4.2 * len(cells), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cells):
        g0 = p[(p["cell_id"] == c) & (p["model"] == models[0])].sort_values("cycle")
        ax.plot(g0["cycle"], 100 * g0["y_true"], "k-", lw=2, label="true")
        for name in models:
            g = p[(p["cell_id"] == c) & (p["model"] == name)].sort_values("cycle")
            ax.plot(g["cycle"], 100 * g["y_pred"], ".", ms=3, alpha=0.6, label=name)
        ax.set_title(c.split("/")[-1], fontsize=9)
        ax.set_xlabel("cycle")
    axes[0].set_ylabel("SOH / %")
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "fig_trajectories.png", dpi=130)
    plt.close()


@safe("fig_importance.png")
def fig_imp(imp, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, g in imp.groupby("model"):
        ax.plot(g["t_s"] / 60, g["d_rmse_pct"], "o-", label=name)
    ax.set_xlabel("minutes after the CV charge ends")
    ax.set_ylabel("RMSE increase when this point is shuffled (SOH % points)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_importance.png", dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    res = Path(args.results)
    ds = pd.read_csv(res / "dataset.csv")
    fig_relax(ds, res)
    fig_stats(ds, res)
    fig_ecm(ds, res)
    if (res / "metrics_long.csv").exists():
        fig_rmse(pd.read_csv(res / "metrics_long.csv"), res)
    if (res / "predictions_subset.csv.gz").exists():
        p = pd.read_csv(res / "predictions_subset.csv.gz")
        fig_parity(p, res)
        fig_traj(p, res)
    if (res / "importance_raw14.csv").exists():
        fig_imp(pd.read_csv(res / "importance_raw14.csv"), res)


if __name__ == "__main__":
    main()
