# Estimating battery state of health from a 30-minute rest after charging

Can a short rest after charging predict how worn a lithium-ion cell is, without a full discharge?
This project reproduces two published methods on a public dataset and adds a neural-net variant.

- **State of health (SOH)** = discharge capacity / 3500 mAh nominal, in percent.
- **Input:** 14 voltage readings, one every 120 s, during the 30-minute rest after the charge.
- **Data:** Zhu et al. 2022, Nature Communications 13, 2261. Zenodo record 6405084, Dataset 1 (66 NCA 18650 cells, 5 operating conditions). Download it yourself from Zenodo and check its terms; the data is not stored in this repository.

Important limit: the data is the relaxation after a constant-voltage charge, not a true rest-then-pulse test.

## What is here

| Method | Features | Model |
|---|---|---|
| Zhu et al. (reproduced) | variance, skewness, max of the 14 voltages | XGBoost, SVR |
| Feng et al. (partly reproduced) | 6 equivalent-circuit parameters (OCV, R0, R1, R2, C1, C2) | Gaussian process |
| This project | raw 14 voltages | MLP (64-32, ReLU), compared with SVR, XGBoost, GPR, ElasticNet and an age-only baseline |

## Headline results

All errors are RMSE in SOH percentage points (1 point = 35 mAh).

- Zhu et al. reproduced within 0.62 points on 11 comparisons (cell split: XGBoost 1.10 vs 1.10 reported, SVR 0.98 vs 1.10).
- Feng et al. not matched: 1.25 here against 0.90 reported on a 50/50 cell split.
- On held-out cells the MLP on raw voltages scores 0.90, SVR 0.95, XGBoost 1.10. The MLP's lead over SVR is within noise (about 0.15 across random splits).
- On a new temperature or operating condition, errors rise to about 2 to 3 points on average. An MLP on engineered features failed badly in two tests (68 and 26 points); SVR on raw voltages was steadier.

See `results/summary.md` for every table.

## How to run

Set up the environment, put the dataset in `data/raw/Dataset_1_NCA_battery`, then run from the repository root:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/stage1_build_table.py data/raw/Dataset_1_NCA_battery --out stage1_out
python scripts/build_dataset.py --stage1 stage1_out --out results
python scripts/run_experiments.py --preset quick --repeats 10 --gpr-splits 5 --n-jobs 4 --families cell50,cell,temp,condition,time
python scripts/summarize.py --results results
python scripts/make_plots.py --results results
```

Each step reads the files the step before wrote:

1. `stage1_build_table.py` finds the rest after charge in every cell file.
2. `build_dataset.py` filters cycles (every drop is counted) and builds the features.
3. `run_experiments.py` trains every model on every split family.
4. `summarize.py` and `make_plots.py` write the tables and figures.

Shared code is in `scripts/soh_common.py` (features, splits, metrics) and `scripts/soh_models.py` (models).

## Split families

| Name | What is held out |
|---|---|
| cell | about 20% of cells per condition |
| cell50 | half the cells per condition (Feng et al. style) |
| temp | one whole temperature |
| condition | one whole operating condition |
| time | the last 20% of each cell's cycles |

## Known gaps

- GPR was trained on at most 600 to 1000 rows and run on only a few splits because it is slow.
- Feng et al.'s choice of voltage for R0 is not stated; ours is the last charge-row voltage.
- Five capacity jumps above 30 mAh and the 26-row cycles were left unexplained.
- Shifted-condition tests have few held-out groups, so their numbers are rough.
