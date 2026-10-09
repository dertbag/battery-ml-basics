"""Shared helpers for the relaxation-voltage SOH experiments (no sklearn/xgboost here).

Features
  stats*  : statistics of the 14-point relaxation curve, as in Zhu et al. 2022,
            Supplementary Table 5 (variance uses 1/(n-1); skewness and kurtosis are
            normalised by that variance's square root; kurtosis is "excess").
  ecm6    : second-order RC fit to the relaxation curve, as in Feng et al. 2023:
                V(t) = OCV + a1*exp(-t/tau1) + a2*exp(-t/tau2)
            with R_k = a_k / I_cut, C_k = tau_k / R_k and
                R0 = (V_last_charge - OCV) / I_cut - R1 - R2.
            The paper does not say which voltage is "U_t(0)"; I use the voltage of the last
            charge row (the CV hold, ~4.2 V). Because the cell is not equilibrated during CV,
            this "R0" also absorbs slow diffusion overpotential, so treat it as a feature,
            not as a physical ohmic resistance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

N_PTS = 14
DT_S = 120.0
VCOLS = [f"v{k:02d}" for k in range(N_PTS)]
STAT_NAMES = ["Var", "Ske", "Max", "Min", "Mean", "Kur"]
ECM_NAMES = ["OCV", "R0", "R1", "R2", "C1", "C2"]

FEATURE_SETS = {
    "stats3": ["Var", "Ske", "Max"],          # Zhu et al.'s best subset
    "stats6": STAT_NAMES,
    "ecm6": ECM_NAMES,
    "raw14": VCOLS,
    "cyc": ["cycle", "T_C", "chg_rate"],      # baseline that knows the cell's age
    "urelax": ["v13"],                         # Baghdadi et al.: voltage after ~28 min rest
}


# --------------------------------------------------------------------------- nominal capacity
def nominal_mAh(dataset_name: str) -> float:
    """NCA and NCM cells are 3.5 Ah; the NCM+NCA cells are 2.5 Ah (Zhu et al., Table 1)."""
    n = str(dataset_name).upper().replace(" ", "")
    if "NCM+NCA" in n or "NCMNCA" in n:
        return 2500.0
    return 3500.0


def capacity_window(nominal: float) -> tuple[float, float]:
    """Authors' extraction script keeps cycles with 2500 <= capacity <= 3500 mAh (NCA)."""
    if nominal == 3500.0:
        return 2500.0, 3500.0
    return 0.714 * nominal, nominal


# --------------------------------------------------------------------------- features
def stat_features(V: np.ndarray) -> pd.DataFrame:
    """V has shape (n_cycles, n_points)."""
    V = np.asarray(V, float)
    n = V.shape[1]
    mean = V.mean(axis=1)
    dev = V - mean[:, None]
    var = (dev ** 2).sum(axis=1) / (n - 1)
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = dev / sd[:, None]
    return pd.DataFrame({
        "Var": var,
        "Ske": (z ** 3).mean(axis=1),
        "Max": V.max(axis=1),
        "Min": V.min(axis=1),
        "Mean": mean,
        "Kur": (z ** 4).mean(axis=1) - 3.0,
    })


def _ecm_curve(p, t):
    ocv, a1, tau1, a2, tau2 = p
    return ocv + a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)


def fit_ecm(t, v, i_cut_mA, v_last) -> dict:
    """Fit the two-exponential relaxation model. Never raises; ecm_ok marks a usable fit."""
    out = {"OCV": np.nan, "R0": np.nan, "R1": np.nan, "R2": np.nan,
           "C1": np.nan, "C2": np.nan, "ecm_ok": False, "ecm_rmse_mV": np.nan}
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    I = abs(float(i_cut_mA)) / 1000.0 if np.isfinite(i_cut_mA) else np.nan
    if not (np.isfinite(I) and I > 0.01 and np.isfinite(v_last) and np.all(np.isfinite(v))):
        return out
    v_end = v[-1]
    drop = max(v[0] - v_end, 1e-3)
    lo = np.array([v_end - 0.15, 1e-4, 20.0, 1e-4, 400.0])
    hi = np.array([v_end, 0.30, 400.0, 0.30, 40000.0])
    best = None
    for tau1, tau2 in ((60.0, 1000.0), (150.0, 3000.0), (100.0, 800.0)):
        p0 = np.clip([v_end - 0.01, 0.5 * drop, tau1, 0.5 * drop, tau2], lo + 1e-9, hi - 1e-9)
        try:
            res = least_squares(lambda p: _ecm_curve(p, t) - v, p0, bounds=(lo, hi),
                                x_scale=[0.01, 0.01, 100.0, 0.01, 1000.0], max_nfev=300)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        return out
    ocv, a1, tau1, a2, tau2 = best.x
    rmse_mV = 1000.0 * np.sqrt(2.0 * best.cost / len(v))
    R1, R2 = a1 / I, a2 / I
    out.update(OCV=ocv, R1=R1, R2=R2, C1=tau1 / R1, C2=tau2 / R2,
               R0=(v_last - ocv) / I - R1 - R2,
               ecm_rmse_mV=rmse_mV, ecm_ok=bool(rmse_mV < 3.0))
    return out


# --------------------------------------------------------------------------- metrics
def metrics(y, p, cond=None) -> dict:
    """rmse_bal_pct = mean over operating conditions of each condition's RMSE, so the many
    45 C cycles do not dominate (cf. the 'weighted average' variant in Zhu's Supplementary Table 8)."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    err = p - y
    eol = y < 0.80
    bal = np.nan
    if cond is not None:
        cond = np.asarray(cond)
        bal = 100.0 * float(np.mean([np.sqrt(np.mean(err[cond == c] ** 2)) for c in np.unique(cond)]))
    return {
        "rmse_pct": 100.0 * float(np.sqrt(np.mean(err ** 2))),
        "rmse_bal_pct": bal,
        "mae_pct": 100.0 * float(np.mean(np.abs(err))),
        "bias_pct": 100.0 * float(np.mean(err)),
        "rmse_eol_pct": (100.0 * float(np.sqrt(np.mean(err[eol] ** 2))) if eol.sum() >= 20 else np.nan),
    }


# --------------------------------------------------------------------------- splits
# df must be sorted by (cell_id, cycle) and have a clean 0..n-1 index.
def split_cell_stratified(df: pd.DataFrame, seed: int, test_frac: float = 0.2):
    """Zhu et al. Strategy D / Feng et al. strategy 1: whole cells held out, stratified by
    condition. n_test = max(1, round(0.2 * n_cells)) reproduces Zhu's Supplementary Table 9
    cell counts (1/7, 4/19, 2/9, 1/3, 6/28 test cells per condition)."""
    rng = np.random.default_rng(seed)
    test_cells = []
    for _, g in df.groupby("condition"):
        cells = np.array(sorted(g["cell_id"].unique()))
        if len(cells) < 2:
            continue
        n_test = min(len(cells) - 1, max(1, int(round(test_frac * len(cells)))))
        test_cells += list(rng.choice(cells, size=n_test, replace=False))
    is_test = df["cell_id"].isin(test_cells).to_numpy()
    return np.flatnonzero(~is_test), np.flatnonzero(is_test), sorted(test_cells)


def make_splits(df: pd.DataFrame, family: str, repeats: int):
    """Return a list of dicts: split_id, label, seed, train, test."""
    out = []
    if family == "cell":
        for r in range(repeats):
            tr, te, cells = split_cell_stratified(df, seed=r)
            out.append(dict(split_id=r, label=f"repeat{r}", seed=r, train=tr, test=te))
    elif family == "cell50":     # Feng et al.: cells of each condition split evenly into train / test
        for r in range(repeats):
            tr, te, cells = split_cell_stratified(df, seed=1000 + r, test_frac=0.5)
            out.append(dict(split_id=r, label=f"repeat{r}", seed=1000 + r, train=tr, test=te))
    elif family == "temp":      # Zhu Supplementary Table 6: hold out one temperature
        for T in sorted(df["T_C"].unique()):
            te = np.flatnonzero((df["T_C"] == T).to_numpy())
            tr = np.flatnonzero((df["T_C"] != T).to_numpy())
            if len(tr) and len(te):
                out.append(dict(split_id=len(out), label=f"hold_out_{int(T)}C", seed=0, train=tr, test=te))
    elif family == "condition":  # new: leave one whole operating condition out
        for c in sorted(df["condition"].unique()):
            te = np.flatnonzero((df["condition"] == c).to_numpy())
            tr = np.flatnonzero((df["condition"] != c).to_numpy())
            if len(tr) and len(te):
                out.append(dict(split_id=len(out), label=f"hold_out_{c}", seed=0, train=tr, test=te))
    elif family == "time":       # Zhu Supplementary Table 7 / Feng strategy 4: first 80% vs last 20%
        rank = df.groupby("cell_id").cumcount().to_numpy()
        size = df.groupby("cell_id")["cycle"].transform("size").to_numpy()
        is_train = rank < np.floor(0.8 * size)
        out.append(dict(split_id=0, label="first80_vs_last20", seed=0,
                        train=np.flatnonzero(is_train), test=np.flatnonzero(~is_train)))
    else:
        raise ValueError(f"unknown split family {family!r}")
    return out
