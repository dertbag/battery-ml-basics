"""Stage 2b-4: run every model on every split family and save the metrics.

Usage (from the folder that contains results/dataset.csv):
    python run_experiments.py --preset quick --n-jobs 4         # try this first (tens of minutes on a laptop)
    python run_experiments.py --preset standard --n-jobs 4      # 20 random cell splits; can take an hour or more
  GPR is by far the slowest model; use --models to leave it out for fast checks, e.g.
    --models xgb_stats3,svr_stats3,enet_stats3,mlp_raw14,xgb_raw14

Split families
  cell       whole cells held out (about 20% per condition; Zhu Strategy D), repeated
  cell50     cells of each condition split 50/50 into train and test (Feng et al. strategy 1), repeated
  temp       train on two temperatures, test on the third (Zhu Supplementary Table 6)
  condition  leave one whole operating condition out (new; harder than temp)
  time       first 80% of each cell's cycles vs the last 20% (Zhu Supplementary Table 7)

Writes into --out: metrics_long.csv, predictions_subset.csv.gz, importance_raw14.csv, run_config.json.
A model that fails on a split is recorded with an error message instead of stopping the run.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from soh_common import FEATURE_SETS, VCOLS, make_splits, metrics
from soh_models import MODEL_SPECS, PRESETS, fit_model

SAVE_PRED_MODELS = ["urelax_linear", "cycle_cond_xgb", "xgb_stats3", "svr_stats3", "gpr_ecm6", "mlp_raw14"]
IMPORTANCE_MODELS = ["xgb_raw14", "mlp_raw14"]


def perm_importance(model, X, y, rng, n_repeats=5):
    """Increase in RMSE (in SOH percentage points) when one input column is shuffled."""
    base = np.sqrt(np.mean((model.predict(X) - y) ** 2))
    out = []
    for j in range(X.shape[1]):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            deltas.append(np.sqrt(np.mean((model.predict(Xp) - y) ** 2)) - base)
        out.append(100.0 * float(np.mean(deltas)))
    return out


def run_split(df, family, split, models, cfg, n_jobs):
    ctx = threadpool_limits(limits=1) if n_jobs != 1 else None
    train, test = df.iloc[split["train"]], df.iloc[split["test"]]
    groups = train["cell_id"].to_numpy()
    ytr, yte = train["soh"].to_numpy(float), test["soh"].to_numpy(float)
    rows, preds, imps = [], [], []
    for name in models:
        fs, kind = MODEL_SPECS[name]
        if kind == "gpr" and split["split_id"] >= cfg["gpr_max_splits"]:
            continue  # GPR only on the first few splits of each family (see soh_models.py)
        cols = FEATURE_SETS[fs]
        Xtr, Xte = train[cols].to_numpy(float), test[cols].to_numpy(float)
        t0 = time.time()
        base = dict(family=family, split_id=split["split_id"], test_label=split["label"], model=name,
                    features=fs, kind=kind, n_train=len(train), n_test=len(test), seed=split["seed"])
        try:
            m = fit_model(kind, Xtr, ytr, groups, cfg, split["seed"])
            p = m.predict(Xte)
            rows.append({**base, **metrics(yte, p, test["condition"].to_numpy()), "fit_seconds": time.time() - t0, "error": ""})
            if name in SAVE_PRED_MODELS and (family not in ("cell", "cell50") or split["split_id"] == 0):
                preds.append(pd.DataFrame({
                    "family": family, "split_id": split["split_id"], "test_label": split["label"],
                    "model": name, "cell_id": test["cell_id"].to_numpy(), "condition": test["condition"].to_numpy(),
                    "cycle": test["cycle"].to_numpy(), "y_true": yte, "y_pred": p}))
            if family == "cell" and split["split_id"] == 0 and name in IMPORTANCE_MODELS:
                imp = perm_importance(m, Xte, yte, np.random.default_rng(0))
                imps.append(pd.DataFrame({"model": name, "point": range(len(imp)),
                                          "t_s": 120 * (np.arange(len(imp)) + 1), "d_rmse_pct": imp}))
        except Exception as e:  # keep going
            rows.append({**base, "rmse_pct": np.nan, "rmse_bal_pct": np.nan, "mae_pct": np.nan, "bias_pct": np.nan,
                         "rmse_eol_pct": np.nan, "fit_seconds": time.time() - t0, "error": repr(e)[:300]})
    if ctx is not None:
        ctx.unregister()
    print(f"  done {family}/{split['label']}: {len(models)} models", flush=True)
    return rows, preds, imps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="results/dataset.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--preset", choices=list(PRESETS), default="quick")
    ap.add_argument("--families", default="cell,temp,condition,time")
    ap.add_argument("--models", default="all", help="comma-separated names, or 'all'")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=None, help="override the preset's cell-split repeats")
    ap.add_argument("--gpr-splits", type=int, default=None,
                    help="run GPR models on the first N splits of each family (default from the preset)")
    ap.add_argument("--no-common-rows", action="store_true",
                    help="do not restrict to cycles whose ECM fit succeeded")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.dataset)
    n0 = len(df)
    if not args.no_common_rows:
        df = df[df["ecm_ok"].astype(bool)]
    df = df.sort_values(["cell_id", "cycle"]).reset_index(drop=True)
    print(f"{len(df)} cycles from {df['cell_id'].nunique()} cells "
          f"({n0 - len(df)} dropped because the ECM fit failed)" if not args.no_common_rows else
          f"{len(df)} cycles from {df['cell_id'].nunique()} cells")

    cfg = dict(PRESETS[args.preset])
    if args.repeats:
        cfg["repeats"] = args.repeats
    if args.gpr_splits is not None:
        cfg["gpr_max_splits"] = args.gpr_splits
    cfg["xgb_jobs"] = 1 if args.n_jobs != 1 else 4
    models = list(MODEL_SPECS) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    bad = [m for m in models if m not in MODEL_SPECS]
    if bad:
        sys.exit(f"unknown models: {bad}\nchoices: {list(MODEL_SPECS)}")
    families = [f.strip() for f in args.families.split(",")]

    jobs = []
    for fam in families:
        for sp in make_splits(df, fam, cfg["repeats"]):
            jobs.append((fam, sp))
    print(f"{len(jobs)} splits x {len(models)} models, preset={args.preset}, n_jobs={args.n_jobs}", flush=True)

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_split)(df, fam, sp, models, cfg, args.n_jobs) for fam, sp in jobs)
    rows = [r for res in results for r in res[0]]
    preds = [p for res in results for p in res[1]]
    imps = [i for res in results for i in res[2]]

    pd.DataFrame(rows).to_csv(out / "metrics_long.csv", index=False)
    if preds:
        pd.concat(preds).to_csv(out / "predictions_subset.csv.gz", index=False)
    if imps:
        pd.concat(imps).to_csv(out / "importance_raw14.csv", index=False)
    import sklearn, xgboost
    json.dump(dict(args=vars(args), cfg=cfg, n_rows=len(df), n_cells=int(df["cell_id"].nunique()),
                   python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                   sklearn=sklearn.__version__, xgboost=xgboost.__version__, seconds=round(time.time() - t0, 1)),
              open(out / "run_config.json", "w"), indent=2)
    m = pd.DataFrame(rows)
    n_err = int((m["error"] != "").sum())
    print(f"\nfinished in {time.time() - t0:.0f} s; {len(m)} model fits, {n_err} failed")
    if n_err:
        print(m[m["error"] != ""][["family", "test_label", "model", "error"]].drop_duplicates(["model", "error"]).to_string(index=False))


if __name__ == "__main__":
    main()
