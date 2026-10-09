"""Model registry for the SOH experiments.

Paper models
  enet / xgb / svr : Zhu et al. 2022. Hyperparameters follow their Supplementary Table 11:
                     XGBoost learning_rate 0.8, 6 trees, depth 6, lambda 1, alpha 0, subsample 1;
                     ElasticNet alpha and l1_ratio on 20-point grids from 0.001 to 1;
                     SVR with RBF kernel, gamma='auto', C and epsilon from a grid (the paper does
                     not give the grid ranges, so the ones below are my choice).
  gpr              : Feng et al. 2023. Gaussian process with an ARD exponential kernel (Matern 1/2).
Deviations forced by compute (printed in run_config.json):
  - SVR is tuned and fitted on every `svr_stride`-th training cycle (neighbouring cycles of a cell
    are near-duplicates).
  - GPR is fitted on at most `gpr_max_train` randomly chosen training rows (exact GPR needs O(n^3)),
    and is only run on the first `gpr_max_splits` splits of each family (it dominates the run time).
  - Hyperparameter cross-validation groups folds by cell, so no cell is in both a tuning-train and
    tuning-validation fold.
Our variation
  mlp              : small neural net (scikit-learn MLPRegressor, 64-32 ReLU, early stopping).
Baselines
  mean, urelax_linear (Baghdadi: line through the voltage after ~28 min rest), cycle_cond_xgb
  (knows cycle number, temperature and charge rate but sees no voltage at all).
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

# name -> (feature set, model kind)
MODEL_SPECS = {
    "mean": ("raw14", "mean"),
    "urelax_linear": ("urelax", "linear"),
    "cycle_cond_xgb": ("cyc", "xgb_big"),
    "enet_stats3": ("stats3", "enet"),
    "xgb_stats3": ("stats3", "xgb"),
    "svr_stats3": ("stats3", "svr"),
    "xgb_stats6": ("stats6", "xgb"),
    "svr_stats6": ("stats6", "svr"),
    "gpr_stats6": ("stats6", "gpr"),
    "xgb_ecm6": ("ecm6", "xgb"),
    "svr_ecm6": ("ecm6", "svr"),
    "gpr_ecm6": ("ecm6", "gpr"),
    "xgb_raw14": ("raw14", "xgb"),
    "svr_raw14": ("raw14", "svr"),
    "gpr_raw14": ("raw14", "gpr"),
    "mlp_stats3": ("stats3", "mlp"),
    "mlp_ecm6": ("ecm6", "mlp"),
    "mlp_raw14": ("raw14", "mlp"),
}

PRESETS = {
    "quick": dict(repeats=3, svr_stride=6, gpr_max_train=600, gpr_max_splits=2, mlp_ensemble=1),
    "standard": dict(repeats=20, svr_stride=3, gpr_max_train=1000, gpr_max_splits=5, mlp_ensemble=3),
}


class MeanEnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


class Fitted:
    """A fitted pipeline: optional X scaler, optional y scaler, estimator."""

    def __init__(self, est, xs=None, ys=None):
        self.est, self.xs, self.ys = est, xs, ys

    def predict(self, X):
        Xs = X if self.xs is None else self.xs.transform(X)
        p = np.asarray(self.est.predict(Xs), float).reshape(-1)
        if self.ys is not None:
            p = self.ys.inverse_transform(p.reshape(-1, 1)).ravel()
        return p


def _scale(Xtr, ytr, scale_y=True):
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(ytr.reshape(-1, 1)) if scale_y else None
    Xs = xs.transform(Xtr)
    yt = ys.transform(ytr.reshape(-1, 1)).ravel() if scale_y else ytr
    return xs, ys, Xs, yt


def _xgb(seed, cfg, big=False):
    kw = dict(n_estimators=300, learning_rate=0.05, max_depth=4) if big else \
        dict(n_estimators=6, learning_rate=0.8, max_depth=6)
    return XGBRegressor(objective="reg:squarederror", reg_lambda=1.0, reg_alpha=0.0, subsample=1.0,
                        tree_method="hist", random_state=seed, n_jobs=cfg.get("xgb_jobs", 1), **kw)


def fit_model(kind, Xtr, ytr, groups, cfg, seed):
    Xtr = np.asarray(Xtr, float)
    ytr = np.asarray(ytr, float)
    n_groups = len(np.unique(groups))

    if kind == "mean":
        return Fitted(DummyRegressor(strategy="mean").fit(Xtr, ytr))
    if kind == "linear":
        return Fitted(LinearRegression().fit(Xtr, ytr))
    if kind in ("xgb", "xgb_big"):
        return Fitted(_xgb(seed, cfg, big=(kind == "xgb_big")).fit(Xtr, ytr))

    if kind == "enet":
        xs, _, Xs, _ = _scale(Xtr, ytr, scale_y=False)
        grid = np.linspace(0.001, 1.0, 20)
        folds = list(GroupKFold(n_splits=min(5, n_groups)).split(Xs, ytr, groups))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            est = ElasticNetCV(alphas=grid, l1_ratio=list(grid), cv=folds, max_iter=5000).fit(Xs, ytr)
        return Fitted(est, xs=xs)

    if kind == "svr":
        xs, ys, Xs, yt = _scale(Xtr, ytr)
        stride = cfg["svr_stride"]
        tune = np.arange(0, len(Xs), 2 * stride)
        gs = GridSearchCV(SVR(kernel="rbf", gamma="auto"),
                          {"C": [3.0, 30.0, 300.0], "epsilon": [0.02, 0.1]},
                          cv=GroupKFold(n_splits=min(3, n_groups)),
                          scoring="neg_mean_squared_error", n_jobs=1)
        gs.fit(Xs[tune], yt[tune], groups=groups[tune])
        est = SVR(kernel="rbf", gamma="auto", **gs.best_params_)
        fit_idx = np.arange(0, len(Xs), stride)
        est.fit(Xs[fit_idx], yt[fit_idx])
        return Fitted(est, xs=xs, ys=ys)

    if kind == "gpr":
        xs, ys, Xs, yt = _scale(Xtr, ytr)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(Xs), size=min(len(Xs), cfg["gpr_max_train"]), replace=False))
        d = Xs.shape[1]
        kernel = (C(1.0, (1e-2, 1e2)) * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e3), nu=0.5)
                  + WhiteKernel(1e-2, (1e-6, 1.0)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=seed)
            est.fit(Xs[idx], yt[idx])
        return Fitted(est, xs=xs, ys=ys)

    if kind == "mlp":
        xs, ys, Xs, yt = _scale(Xtr, ytr)
        members = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for j in range(cfg["mlp_ensemble"]):
                m = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                                 alpha=1e-4, learning_rate_init=1e-3, batch_size=256, max_iter=300,
                                 early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
                                 random_state=seed * 100 + j)
                members.append(m.fit(Xs, yt))
        return Fitted(MeanEnsemble(members), xs=xs, ys=ys)

    raise ValueError(f"unknown model kind {kind!r}")
