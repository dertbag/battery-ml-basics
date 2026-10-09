"""Summarise results/metrics_long.csv into results/summary.md (and print it).

Usage:  python summarize.py --results results
All errors are RMSE in percentage points of nominal capacity (SOH = capacity / 3500 mAh).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_REFS = """
Reference numbers from the papers (their data splits differ from ours, so compare loosely):
- Zhu et al. 2022, cell-stratified split (Strategy D), XGBoost / SVR on [Var, Ske, Max]: 1.1% / 1.1%
- Zhu et al. Supplementary Table 6, temperature held out (XGBoost / SVR):
    train 25+35 C, test 45 C: 2.5% / 3.1%;  train 25+45 C, test 35 C: 1.5% / 1.5%;  train 35+45 C, test 25 C: 4.4% / 3.8%
  (the main text quotes only the best of these three, 1.5%)
- Zhu et al. Supplementary Table 7, first 80% vs last 20% of the data: XGBoost 2.3%, SVR 3.1%
- Zhu et al. benchmark, linear model on the 30-min rest voltage: 2.5%
- Feng et al. 2023, GPR on the six ECM features, Dataset 1 (their own split): 0.90%
"""


def fmt(x):
    return "" if pd.isna(x) else f"{x:.2f}"


REPRO = [  # label, family, model, held-out label (or None), paper RMSE %
    ("Zhu cell split, XGBoost [Var,Ske,Max]", "cell", "xgb_stats3", None, 1.1),
    ("Zhu cell split, SVR [Var,Ske,Max]", "cell", "svr_stats3", None, 1.1),
    ("Zhu temp: hold out 45 C, XGBoost", "temp", "xgb_stats3", "hold_out_45C", 2.5),
    ("Zhu temp: hold out 45 C, SVR", "temp", "svr_stats3", "hold_out_45C", 3.1),
    ("Zhu temp: hold out 35 C, XGBoost", "temp", "xgb_stats3", "hold_out_35C", 1.5),
    ("Zhu temp: hold out 35 C, SVR", "temp", "svr_stats3", "hold_out_35C", 1.5),
    ("Zhu temp: hold out 25 C, XGBoost", "temp", "xgb_stats3", "hold_out_25C", 4.4),
    ("Zhu temp: hold out 25 C, SVR", "temp", "svr_stats3", "hold_out_25C", 3.8),
    ("Zhu time split, XGBoost", "time", "xgb_stats3", None, 2.3),
    ("Zhu time split, SVR", "time", "svr_stats3", None, 3.1),
    ("Zhu benchmark: linear on 30-min rest voltage (random 80/20 of cycles)", "cell", "urelax_linear", None, 2.5),
    ("Feng GPR on 6 ECM features, 50/50 cell split", "cell50", "gpr_ecm6", None, 0.90),
]


def reproduction_check(m):
    rows = ["## Reproduction check (ours vs the papers' reported RMSE, in % of nominal capacity)", "",
            "| paper setting | paper | ours | difference | our splits |", "|---|---|---|---|---|"]
    for label, fam, model, lab, ref in REPRO:
        g = m[(m["family"] == fam) & (m["model"] == model)]
        if lab is not None:
            g = g[g["test_label"] == lab]
        if g.empty:
            rows.append(f"| {label} | {ref:.2f} | not run | | |")
        else:
            ours = g["rmse_pct"].mean()
            rows.append(f"| {label} | {ref:.2f} | {ours:.2f} | {ours - ref:+.2f} | {len(g)} |")
    rows += ["", "Splits differ from the papers' (theirs are not published as cell lists), so differences of a few "
             "tenths of a point are expected.", ""]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    res = Path(args.results)
    m = pd.read_csv(res / "metrics_long.csv")
    err = m[m["error"].fillna("") != ""]
    m = m[m["error"].fillna("") == ""]
    lines = ["# Results summary", "", "RMSE in % of nominal capacity. Lower is better. 'RMSE balanced' averages the per-condition RMSEs, so the "
             "many 45 C cycles (71% of the data) do not dominate.", ""]

    for fam in ["cell", "cell50", "time"]:
        g = m[m["family"] == fam]
        if g.empty:
            continue
        agg = g.groupby(["model", "features"]).agg(
            splits=("rmse_pct", "size"), rmse_mean=("rmse_pct", "mean"), rmse_std=("rmse_pct", "std"),
            rmse_bal=("rmse_bal_pct", "mean"),
            mae=("mae_pct", "mean"), rmse_eol=("rmse_eol_pct", "mean"), bias=("bias_pct", "mean"),
            fit_s=("fit_seconds", "mean")).reset_index().sort_values("rmse_mean")
        lines += [f"## Family: {fam}" + {"cell": " (about 20% of cells held out; mean ± std over repeats)",
                                          "cell50": " (50/50 cell split like Feng; mean ± std over repeats)",
                                          "time": " (first 80% of each cell vs last 20%)"}[fam], "",
                  "| model | features | splits | RMSE | ± std | RMSE balanced | MAE | RMSE (SOH<80%) | bias | fit s |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in agg.iterrows():
            lines.append(f"| {r.model} | {r.features} | {int(r.splits)} | {fmt(r.rmse_mean)} | {fmt(r.rmse_std)} | {fmt(r.rmse_bal)} | "
                         f"{fmt(r.mae)} | {fmt(r.rmse_eol)} | {fmt(r.bias)} | {r.fit_s:.1f} |")
        lines.append("")

    for fam, title in [("temp", "temperature held out"), ("condition", "operating condition held out")]:
        g = m[m["family"] == fam]
        if g.empty:
            continue
        pv = g.pivot_table(index="model", columns="test_label", values="rmse_pct")
        pv["mean"] = pv.mean(axis=1)
        pv["worst"] = pv.drop(columns="mean").max(axis=1)
        pv = pv.sort_values("mean")
        lines += [f"## Family: {fam} ({title}; RMSE per held-out group)", "",
                  "| model | " + " | ".join(pv.columns) + " |", "|---|" + "---|" * len(pv.columns)]
        for name, r in pv.iterrows():
            lines.append(f"| {name} | " + " | ".join(fmt(v) for v in r.values) + " |")
        lines.append("")

    lines += reproduction_check(m)
    lines += ["## Paper reference numbers", PAPER_REFS]
    if len(err):
        lines += ["## Failed fits", "", err[["family", "test_label", "model", "error"]]
                  .drop_duplicates(["model", "error"]).to_string(index=False), ""]
    text = "\n".join(lines)
    (res / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
