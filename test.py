"""Stage 0: inspect one cell file. Read-only.

Prints (1) a per-cycle summary of the rest segments and capacity,
(2) the raw rows around each step transition for an early and a late cycle,
and saves (3) a plot of the rest after charge for three cycles.

Usage, from the folder holding the csv:
    python stage0.py "CY25-1_1-#1.csv"
"""
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILE = sys.argv[1] if len(sys.argv) > 1 else "CY25-1_1-#1.csv"
I_TOL = 5.0  # mA: |I| above this counts as charging / discharging

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)

df = pd.read_csv(FILE)
df.columns = [c.strip() for c in df.columns]
need = ["time/s", "Ecell/V", "<I>/mA", "Q discharge/mA.h", "cycle number"]
missing = [c for c in need if c not in df.columns]
if missing:
    sys.exit(f"Missing columns: {missing}\nFound: {list(df.columns)}")

SHOW = [c for c in ["time/s", "Ecell/V", "<I>/mA", "control/V/mA", "control/V",
                    "control/mA", "Q discharge/mA.h"] if c in df.columns]

cycles = sorted(df["cycle number"].dropna().unique())
print(f"file: {FILE}\nrows: {len(df)}   cycles: {len(cycles)}   "
      f"first/last: {cycles[0]:.0f} / {cycles[-1]:.0f}\n")


def split(c):
    """Return (a, b, e): a = last charge row, b = first discharge row,
    e = last discharge row (positions within the cycle). None if incomplete."""
    I = c["<I>/mA"].to_numpy()
    pos = np.flatnonzero(I > I_TOL)
    neg = np.flatnonzero(I < -I_TOL)
    if len(pos) == 0 or len(neg) == 0:
        return None
    return pos[-1], neg[0], neg[-1]


# ---- 1. per-cycle summary -------------------------------------------------
records = []
cycle_frames = {}
for cyc in cycles:
    c = df[df["cycle number"] == cyc].reset_index(drop=True)
    cycle_frames[cyc] = c
    s = split(c)
    if s is None:
        records.append({"cycle": int(cyc), "n_rows": len(c), "note": "no charge or no discharge"})
        continue
    a, b, e = s
    t = c["time/s"].to_numpy()
    records.append({
        "cycle": int(cyc),
        "n_rows": len(c),
        "rows_between_charge_and_discharge": int(b - a - 1),
        "span_min": round((t[b] - t[a]) / 60, 1),
        "rows_after_discharge": int(len(c) - 1 - e),
        "rest_after_dis_min": round((t[-1] - t[e]) / 60, 1),
        "I_last_charge_row_mA": round(float(c["<I>/mA"].iloc[a]), 1),
        "Q_dis_max_mAh": round(float(c["Q discharge/mA.h"].max()), 1),
    })
summary = pd.DataFrame(records)
print("=== per-cycle summary ===")
print(summary.to_string(index=False))

valid = [cyc for cyc in cycles if split(cycle_frames[cyc]) is not None]
if len(valid) < 2:
    sys.exit("Fewer than 2 complete cycles found; cannot show transitions.")

# ---- 2. raw rows around the transitions -----------------------------------
for cyc in (valid[0], valid[-1]):
    c = cycle_frames[cyc]
    a, b, e = split(c)
    print(f"\n=== cycle {int(cyc)}: last charge row -> rest -> first discharge row "
          f"(positions {max(a - 2, 0)}..{b + 2}) ===")
    print(c.loc[max(a - 2, 0): b + 2, SHOW].to_string())
    print(f"\n=== cycle {int(cyc)}: end of discharge -> final rest "
          f"(positions {max(e - 2, 0)}..end) ===")
    print(c.loc[max(e - 2, 0):, SHOW].to_string())

# ---- 3. plot the rest after charge ----------------------------------------
picks = [valid[0], valid[len(valid) // 2], valid[-1]]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for cyc in picks:
    c = cycle_frames[cyc]
    a, b, e = split(c)
    t = c["time/s"].to_numpy()
    seg = c.loc[a: b - 1]
    axes[0].plot((seg["time/s"] - t[a]) / 60, seg["Ecell/V"], "o-", label=f"cycle {int(cyc)}")
    seg2 = c.loc[e:]
    axes[1].plot((seg2["time/s"] - t[e]) / 60, seg2["Ecell/V"], "o-", label=f"cycle {int(cyc)}")
axes[0].set_title("From last charge row to just before discharge (includes boundary rows)")
axes[1].set_title("From last discharge row to end of cycle")
for ax in axes:
    ax.set_xlabel("minutes since segment start")
    ax.set_ylabel("Ecell / V")
    ax.legend()
plt.tight_layout()
plt.savefig("stage0_rest_plot.png", dpi=130)
print("\nsaved stage0_rest_plot.png")