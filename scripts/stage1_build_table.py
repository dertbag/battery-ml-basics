"""Stage 1: build a compact cycle table + relaxation curves from all cell CSVs.

Read-only on the CSVs. Writes three files into --out:
  cycles.csv      one row per (file, cycle): capacity, rest-segment QC, flags
  relax_long.csv  one row per rest sample: dataset, file, cycle, k, t_rel_s, Ecell_V
  qc_summary.txt  one row per file: cycle count, rest-length distribution, flags

The "rest after charge" is found with the control columns, not with thresholds
alone: after the last row with I > 5 mA (end of CV charge), take the contiguous
rows where |I| < 5 mA and control/V == 0 and control/mA == 0. The discharge-start
row has control/mA = -3500, so it is excluded.

Usage:
    python stage1_build_table.py <root_folder> [--out stage1_out] [--max-files N]
<root_folder> is searched recursively for *.csv; the first folder below the root
is recorded as the dataset name.
"""
import argparse
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

I_TOL = 5.0        # mA: |I| above this counts as charging / discharging
CTRL_TOL = 1e-6    # control columns are exactly 0 during rest
NEEDED = ["time/s", "Ecell/V", "<I>/mA", "Q discharge/mA.h",
          "control/V", "control/mA", "cycle number"]
NAME_RE = re.compile(r"CY(?P<T>\d+)-(?P<chg>[0-9.]+)_(?P<dis>[0-9.]+)-#(?P<n>\d+)")


def parse_rate(s):
    """'1' -> 1.0, '025' -> 0.25, '05' -> 0.5, '0.5' -> 0.5."""
    if s is None:
        return np.nan
    if "." in s:
        return float(s)
    if len(s) > 1 and s.startswith("0"):
        return float("0." + s[1:])
    return float(s)


def parse_name(stem):
    m = NAME_RE.search(stem)
    if not m:
        return dict(T_C=np.nan, chg_rate=np.nan, dis_rate=np.nan, cell_tag=np.nan,
                    chg_raw="", name_parsed=False)
    return dict(T_C=float(m["T"]), chg_rate=parse_rate(m["chg"]),
                dis_rate=parse_rate(m["dis"]), cell_tag=int(m["n"]),
                chg_raw=m["chg"], name_parsed=True)


def process_cycle(c):
    """Return (record, rest_rows, flags) for one cycle's rows."""
    t = c["time/s"].to_numpy()
    v = c["Ecell/V"].to_numpy()
    I = c["<I>/mA"].to_numpy()
    cv = c["control/V"].to_numpy()
    cm = c["control/mA"].to_numpy()
    qd = c["Q discharge/mA.h"].to_numpy()
    n = len(c)

    pos = np.flatnonzero(I > I_TOL)
    neg = np.flatnonzero(I < -I_TOL)
    flags = []
    rec = {
        "n_rows": n,
        "capacity_mAh": float(np.nanmax(qd)) if len(neg) else np.nan,
        "rows_after_discharge": int(n - 1 - neg[-1]) if len(neg) else np.nan,
        "n_rest": 0, "rest_first_s": np.nan, "rest_last_min": np.nan,
        "step_min": np.nan, "I_cut_mA": np.nan,
        "V_last_charge": np.nan, "V_rest_end": np.nan,
    }
    if len(pos) == 0:
        flags.append("no_charge")
        return rec, [], flags
    if len(neg) == 0:
        flags.append("no_discharge")

    a = pos[-1]
    j = a + 1
    while (j < n and abs(I[j]) < I_TOL
           and abs(cv[j]) < CTRL_TOL and abs(cm[j]) < CTRL_TOL):
        j += 1
    idx = np.arange(a + 1, j)
    rec["n_rest"] = len(idx)
    rec["I_cut_mA"] = float(I[a])
    rec["V_last_charge"] = float(v[a])
    if len(idx) == 0:
        flags.append("no_rest_rows")
        return rec, [], flags

    rec["rest_first_s"] = float(t[idx[0]] - t[a])
    rec["rest_last_min"] = float((t[idx[-1]] - t[a]) / 60.0)
    rec["V_rest_end"] = float(v[idx[-1]])
    if j < n:
        rec["step_min"] = float((t[j] - t[a]) / 60.0)
        if not cm[j] < -I_TOL:
            flags.append("rest_not_followed_by_discharge")
    else:
        flags.append("rest_cut_by_end_of_data")
    rest_rows = [(k, float(t[i] - t[a]), float(v[i])) for k, i in enumerate(idx)]
    return rec, rest_rows, flags


def process_file(path, root):
    rel = path.relative_to(root)
    dataset = rel.parts[0] if len(rel.parts) > 1 else root.name
    df = pd.read_csv(path, usecols=lambda col: col.strip() in NEEDED)
    df.columns = [col.strip() for col in df.columns]
    missing = [col for col in NEEDED if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    meta = parse_name(path.stem)
    meta.update(dataset=dataset, file=path.name)

    cyc_rows, relax_rows = [], []
    for cyc, c in df.groupby("cycle number", sort=True):
        rec, rest_rows, flags = process_cycle(c.reset_index(drop=True))
        rec.update(meta)
        rec["cycle"] = int(cyc)
        rec["flags"] = ";".join(flags)
        cyc_rows.append(rec)
        for k, trel, volt in rest_rows:
            relax_rows.append((dataset, path.name, int(cyc), k, trel, volt))
    cycles = pd.DataFrame(cyc_rows)

    # flag outliers within this file (after all cycles are known)
    typical_rows = cycles["n_rows"].median()
    typical_n_rest = cycles["n_rest"].mode().iloc[0]
    extra = []
    for i, r in cycles.iterrows():
        f = [r["flags"]] if r["flags"] else []
        if r["n_rows"] < 0.85 * typical_rows:
            f.append("short_cycle")
        if r["n_rest"] != typical_n_rest:
            f.append(f"n_rest_differs_from_{int(typical_n_rest)}")
        extra.append(";".join(f))
    cycles["flags"] = extra
    d = cycles["capacity_mAh"].diff()
    cycles["cap_rise_mAh"] = d.where(d > 0)
    return cycles, relax_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="stage1_out")
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = sorted(root.rglob("*.csv"))
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        sys.exit(f"No csv files found under {root}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} csv files under {root}\n")

    all_cycles, all_relax, summary, errors = [], [], [], []
    for i, p in enumerate(files, 1):
        try:
            cycles, relax = process_file(p, root)
        except Exception as e:  # keep going; report at the end
            errors.append((p.name, repr(e)))
            print(f"[{i}/{len(files)}] {p.name}: ERROR {e!r}")
            traceback.print_exc(limit=1)
            continue
        all_cycles.append(cycles)
        all_relax.extend(relax)
        counts = cycles["n_rest"].value_counts().to_dict()
        nflag = int((cycles["flags"] != "").sum())
        big_rise = cycles["cap_rise_mAh"].max()
        summary.append({
            "dataset": cycles["dataset"].iloc[0], "file": p.name,
            "cycles": len(cycles), "n_rest_counts": str(counts),
            "flagged_cycles": nflag,
            "cap_first": round(cycles["capacity_mAh"].iloc[0], 1),
            "cap_last": round(cycles["capacity_mAh"].iloc[-1], 1),
            "max_cap_rise": round(float(big_rise), 1) if pd.notna(big_rise) else 0.0,
            "name_parsed": bool(cycles["name_parsed"].iloc[0]),
        })
        print(f"[{i}/{len(files)}] {p.name}: {len(cycles)} cycles, "
              f"n_rest {counts}, flagged {nflag}, max cap rise {summary[-1]['max_cap_rise']}")

    if not all_cycles:
        sys.exit("No files processed successfully.")
    cycles_df = pd.concat(all_cycles, ignore_index=True)
    relax_df = pd.DataFrame(all_relax, columns=["dataset", "file", "cycle", "k", "t_rel_s", "Ecell_V"])
    cycles_df.to_csv(out / "cycles.csv", index=False)
    relax_df.to_csv(out / "relax_long.csv", index=False)
    summ = pd.DataFrame(summary)
    with open(out / "qc_summary.txt", "w") as f:
        f.write(summ.to_string(index=False))
        f.write(f"\n\nfiles with errors: {errors}\n")
        f.write("\nflag counts across all cycles:\n")
        flat = cycles_df["flags"].str.split(";").explode()
        f.write(flat[flat != ""].value_counts().to_string() if (flat != "").any() else "none")
        f.write("\n")
    print(f"\nwrote {out}/cycles.csv ({len(cycles_df)} rows), "
          f"relax_long.csv ({len(relax_df)} rows), qc_summary.txt")
    if errors:
        print(f"{len(errors)} file(s) failed; see qc_summary.txt")


if __name__ == "__main__":
    main()
