"""Stage 2a: turn Stage 1 outputs into one analysis table, with every drop accounted for.

Usage (from the folder that contains stage1_out):
    python build_dataset.py --stage1 stage1_out --out results

Reads  stage1_out/cycles.csv and stage1_out/relax_long.csv
Writes results/dataset.csv        one row per kept cycle: 14 voltages, stats, ECM params, SOH
       results/dataset_report.txt which cycles were dropped and why, ECM fit quality, jump check
       results/jump_check.csv     capacity jumps and whether the relaxation features moved too

Filters, applied in this order (each count is reported):
  1. filename not parsed          -> cannot assign a condition
  2. no capacity                  -> cycle has no discharge
  3. capacity outside the window  -> the authors' script keeps 2500..3500 mAh for NCA
  4. rest rows != 14              -> EIS/check-up cycles (26 rows) and cut-short rests
  5. rest timing off the 120 s grid, or voltage outside 3.9..4.25 V / last point <= 4.0 V
SOH = capacity / nominal capacity (3500 mAh), the normalisation used by Zhu et al.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from soh_common import (DT_S, N_PTS, VCOLS, capacity_window, fit_ecm, nominal_mAh, stat_features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="stage1_out")
    ap.add_argument("--out", default="results")
    ap.add_argument("--allow-long-rest", action="store_true",
                    help="keep cycles with more than 14 rest rows (uses the first 14)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    s1 = Path(args.stage1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("cycles.csv", "relax_long.csv"):
        if not (s1 / f).exists():
            sys.exit(f"Missing {s1 / f}. Run stage1_build_table.py first.")

    cyc = pd.read_csv(s1 / "cycles.csv")
    rl = pd.read_csv(s1 / "relax_long.csv")
    rep = []
    rep.append(f"cycles.csv: {len(cyc)} cycles from {cyc['file'].nunique()} files; "
               f"relax_long.csv: {len(rl)} rows")

    # relax_long from an older Stage 1 run has no dataset column; that is only safe for one dataset
    if "dataset" not in rl.columns:
        if cyc["dataset"].nunique() > 1:
            sys.exit("relax_long.csv has no 'dataset' column but cycles.csv has several datasets. "
                     "Re-run the updated stage1_build_table.py.")
        rl["dataset"] = cyc["dataset"].iloc[0]
    key = ["dataset", "file", "cycle"]
    if cyc.duplicated(key).any():
        sys.exit("Duplicate (dataset, file, cycle) rows in cycles.csv.")

    # ---------------- filters with counts
    cyc["nominal"] = cyc["dataset"].map(nominal_mAh)
    lo_hi = cyc["nominal"].map(capacity_window)
    cyc["cap_lo"] = [a for a, _ in lo_hi]
    cyc["cap_hi"] = [b for _, b in lo_hi]
    steps = []
    keep = pd.Series(True, index=cyc.index)

    def apply(name, bad):
        nonlocal keep
        newly = keep & bad
        steps.append((name, int(newly.sum())))
        keep = keep & ~bad

    apply("filename not parsed", ~cyc["name_parsed"].astype(bool))
    apply("no capacity (no discharge)", cyc["capacity_mAh"].isna())
    apply("capacity outside window", (cyc["capacity_mAh"] < cyc["cap_lo"]) | (cyc["capacity_mAh"] > cyc["cap_hi"]))
    if args.allow_long_rest:
        apply("rest rows < 14", cyc["n_rest"] < N_PTS)
    else:
        apply("rest rows != 14", cyc["n_rest"] != N_PTS)
    c1 = cyc[keep].copy()

    # ---------------- wide voltage/time table
    rl = rl[rl["k"] < N_PTS]
    try:
        wv = rl.pivot(index=key, columns="k", values="Ecell_V")
        wt = rl.pivot(index=key, columns="k", values="t_rel_s")
    except ValueError as e:
        sys.exit(f"Could not pivot relax_long.csv ({e}).")
    wv.columns = VCOLS[: wv.shape[1]]
    wt.columns = [f"t{k:02d}" for k in range(wt.shape[1])]
    c1 = c1.merge(wv.reset_index(), on=key, how="left").merge(wt.reset_index(), on=key, how="left")
    tcols = [f"t{k:02d}" for k in range(N_PTS)]
    missing_cols = [c for c in VCOLS + tcols if c not in c1.columns]
    if missing_cols:
        sys.exit(f"relax_long.csv does not have {N_PTS} points per cycle (missing {missing_cols[:3]}...).")

    V = c1[VCOLS].to_numpy(float)
    T = c1[tcols].to_numpy(float)
    grid = DT_S * (np.arange(N_PTS) + 1)
    bad_time = ~np.isfinite(V).all(axis=1) | ~(np.abs(T - grid) <= 10.0).all(axis=1)
    bad_volt = ~((V > 3.9) & (V < 4.25)).all(axis=1) | ~(V[:, -1] > 4.0)
    before = len(c1)
    ok = ~(bad_time | bad_volt)
    steps.append(("rest not on 120 s grid / voltage out of range", int((~ok).sum())))
    c1 = c1[ok].copy()
    V = V[ok]

    rep.append("\n=== filters (cycles dropped at each step) ===")
    rep.append(f"start: {len(cyc)} cycles")
    for name, n in steps:
        rep.append(f"  - {name}: {n}")
    rep.append(f"kept: {len(c1)} cycles")
    if len(c1) == 0:
        open(out / "dataset_report.txt", "w").write("\n".join(rep))
        sys.exit("\n".join(rep) + "\nNo cycles left after filtering.")

    # ---------------- features
    print(f"fitting the ECM to {len(c1)} relaxation curves ...")
    Tk = DT_S * (np.arange(N_PTS) + 1)
    ecm = Parallel(n_jobs=args.n_jobs, batch_size=64)(
        delayed(fit_ecm)(Tk, V[i], c1["I_cut_mA"].iloc[i], c1["V_last_charge"].iloc[i])
        for i in range(len(c1)))
    ecm = pd.DataFrame(ecm, index=c1.index)
    stats = stat_features(V)
    stats.index = c1.index

    ds = pd.DataFrame({
        "dataset": c1["dataset"], "file": c1["file"],
        "cell_id": c1["dataset"].astype(str) + "/" + c1["file"].astype(str).str.replace(".csv", "", regex=False),
        "T_C": c1["T_C"], "chg_rate": c1["chg_rate"], "dis_rate": c1["dis_rate"],
        "cycle": c1["cycle"], "n_rest": c1["n_rest"],
        "capacity_mAh": c1["capacity_mAh"], "soh": c1["capacity_mAh"] / c1["nominal"],
        "I_cut_mA": c1["I_cut_mA"], "V_last_charge": c1["V_last_charge"],
    })
    ds["condition"] = ("T" + ds["T_C"].astype(int).astype(str) + "_chg" + ds["chg_rate"].map(lambda x: f"{x:g}")
                       + "_dis" + ds["dis_rate"].map(lambda x: f"{x:g}"))
    ds = pd.concat([ds, c1[VCOLS], stats, ecm], axis=1)
    ds = ds.sort_values(["cell_id", "cycle"]).reset_index(drop=True)
    ds.to_csv(out / "dataset.csv", index=False)

    # ---------------- report
    rep.append("\n=== kept cycles per condition (cells / cycles) ===")
    g = ds.groupby("condition").agg(cells=("cell_id", "nunique"), cycles=("cycle", "size"),
                                   soh_min=("soh", "min"), soh_max=("soh", "max"))
    rep.append(g.round(3).to_string())
    per_cell = ds.groupby("cell_id").size()
    rep.append(f"\ncells with data: {len(per_cell)}; cycles per cell: min {per_cell.min()}, "
               f"median {int(per_cell.median())}, max {per_cell.max()}")
    rep.append("\n=== ECM fit quality ===")
    rep.append(f"fit ok (RMSE < 3 mV): {int(ds['ecm_ok'].sum())} of {len(ds)} "
               f"({100 * ds['ecm_ok'].mean():.1f}%)")
    rep.append("fit RMSE quantiles (mV): " + ", ".join(
        f"{q:.0%}={ds['ecm_rmse_mV'].quantile(q):.3f}" for q in (0.5, 0.9, 0.99)))
    for c in ("OCV", "R0", "R1", "R2", "C1", "C2"):
        rep.append(f"  {c}: median {ds[c].median():.4g}, 5%..95% = "
                   f"{ds[c].quantile(0.05):.4g} .. {ds[c].quantile(0.95):.4g}")

    # capacity jumps and whether the relaxation features moved with them
    d = ds.groupby("cell_id")[["capacity_mAh", "v13", "Var", "cycle"]].diff()
    jump = ds.loc[d["capacity_mAh"] > 30, ["cell_id", "cycle", "capacity_mAh", "v13", "Var"]].copy()
    jump["d_cap_mAh"] = d.loc[jump.index, "capacity_mAh"]
    jump["d_v13_mV"] = 1000 * d.loc[jump.index, "v13"]
    jump["d_Var_pct"] = 100 * d.loc[jump.index, "Var"] / (ds.loc[jump.index, "Var"] - d.loc[jump.index, "Var"])
    jump["cycles_since_prev_kept"] = d.loc[jump.index, "cycle"]
    jump = jump.rename(columns={"capacity_mAh": "cap_after_mAh"})
    jump.to_csv(out / "jump_check.csv", index=False)
    rep.append(f"\n=== capacity jumps > 30 mAh between consecutive kept cycles: {len(jump)} "
               f"(details in jump_check.csv) ===")
    if len(jump):
        rep.append(jump.sort_values("d_cap_mAh", ascending=False).head(12).round(3).to_string(index=False))
    text = "\n".join(rep)
    (out / "dataset_report.txt").write_text(text)
    print(text)
    print(f"\nwrote {out / 'dataset.csv'} ({len(ds)} rows), dataset_report.txt, jump_check.csv")


if __name__ == "__main__":
    main()
