# Results summary

RMSE in % of nominal capacity. Lower is better. 'RMSE balanced' averages the per-condition RMSEs, so the many 45 C cycles (71% of the data) do not dominate.

## Family: cell (about 20% of cells held out; mean ± std over repeats)

| model | features | splits | RMSE | ± std | RMSE balanced | MAE | RMSE (SOH<80%) | bias | fit s |
|---|---|---|---|---|---|---|---|---|---|
| mlp_raw14 | raw14 | 10 | 0.90 | 0.18 | 1.57 | 0.62 | 0.86 | -0.05 | 3.4 |
| mlp_ecm6 | ecm6 | 10 | 0.91 | 0.11 | 1.69 | 0.60 | 0.88 | -0.03 | 3.9 |
| mlp_stats3 | stats3 | 10 | 0.93 | 0.19 | 1.55 | 0.64 | 0.89 | -0.04 | 2.6 |
| svr_stats6 | stats6 | 10 | 0.95 | 0.15 | 1.76 | 0.65 | 0.97 | -0.05 | 4.3 |
| svr_raw14 | raw14 | 10 | 0.95 | 0.18 | 1.70 | 0.65 | 0.91 | -0.02 | 2.9 |
| svr_stats3 | stats3 | 10 | 0.98 | 0.15 | 1.83 | 0.65 | 1.01 | -0.05 | 3.9 |
| gpr_stats6 | stats6 | 5 | 1.09 | 0.14 | 1.98 | 0.67 | 0.87 | -0.08 | 3.9 |
| xgb_stats6 | stats6 | 10 | 1.10 | 0.16 | 1.85 | 0.76 | 1.02 | -0.05 | 0.0 |
| xgb_stats3 | stats3 | 10 | 1.10 | 0.15 | 1.83 | 0.76 | 1.02 | -0.06 | 0.0 |
| gpr_raw14 | raw14 | 5 | 1.14 | 0.17 | 2.04 | 0.73 | 0.96 | -0.08 | 17.7 |
| gpr_ecm6 | ecm6 | 5 | 1.15 | 0.16 | 2.35 | 0.68 | 1.02 | -0.08 | 3.5 |
| xgb_ecm6 | ecm6 | 10 | 1.16 | 0.11 | 2.03 | 0.81 | 1.07 | -0.04 | 0.0 |
| svr_ecm6 | ecm6 | 10 | 1.16 | 0.11 | 2.27 | 0.71 | 1.04 | -0.09 | 3.8 |
| xgb_raw14 | raw14 | 10 | 1.25 | 0.16 | 2.06 | 0.85 | 1.12 | -0.02 | 0.1 |
| enet_stats3 | stats3 | 10 | 1.58 | 0.15 | 2.57 | 1.08 | 1.85 | -0.10 | 0.2 |
| cycle_cond_xgb | cyc | 10 | 1.81 | 0.22 | 1.87 | 1.46 | 2.06 | -0.03 | 0.2 |
| urelax_linear | urelax | 10 | 2.51 | 0.16 | 4.14 | 1.59 | 1.39 | 0.07 | 0.0 |
| mean | raw14 | 10 | 5.95 | 0.19 | 6.13 | 4.95 | 6.69 | -0.06 | 0.0 |

## Family: cell50 (50/50 cell split like Feng; mean ± std over repeats)

| model | features | splits | RMSE | ± std | RMSE balanced | MAE | RMSE (SOH<80%) | bias | fit s |
|---|---|---|---|---|---|---|---|---|---|
| svr_raw14 | raw14 | 10 | 1.08 | 0.06 | 1.66 | 0.76 | 1.12 | -0.03 | 1.7 |
| mlp_raw14 | raw14 | 10 | 1.12 | 0.08 | 1.67 | 0.77 | 1.20 | -0.02 | 2.1 |
| svr_stats6 | stats6 | 10 | 1.12 | 0.07 | 1.84 | 0.79 | 1.22 | 0.01 | 2.0 |
| svr_stats3 | stats3 | 10 | 1.13 | 0.08 | 1.87 | 0.79 | 1.25 | 0.01 | 1.8 |
| mlp_ecm6 | ecm6 | 10 | 1.14 | 0.09 | 1.89 | 0.76 | 1.17 | -0.03 | 2.1 |
| mlp_stats3 | stats3 | 10 | 1.16 | 0.10 | 1.70 | 0.81 | 1.25 | -0.03 | 2.0 |
| gpr_raw14 | raw14 | 5 | 1.24 | 0.06 | 2.09 | 0.85 | 1.23 | 0.02 | 15.7 |
| gpr_stats6 | stats6 | 5 | 1.24 | 0.06 | 2.16 | 0.84 | 1.23 | 0.05 | 3.7 |
| gpr_ecm6 | ecm6 | 5 | 1.25 | 0.03 | 2.25 | 0.81 | 1.33 | 0.04 | 3.5 |
| xgb_stats6 | stats6 | 10 | 1.28 | 0.08 | 1.97 | 0.89 | 1.25 | -0.03 | 0.0 |
| xgb_stats3 | stats3 | 10 | 1.29 | 0.05 | 1.88 | 0.92 | 1.27 | -0.03 | 0.0 |
| svr_ecm6 | ecm6 | 10 | 1.29 | 0.05 | 2.21 | 0.83 | 1.25 | -0.03 | 1.9 |
| xgb_ecm6 | ecm6 | 10 | 1.33 | 0.06 | 2.09 | 0.93 | 1.31 | -0.05 | 0.0 |
| xgb_raw14 | raw14 | 10 | 1.42 | 0.04 | 2.16 | 0.99 | 1.29 | -0.06 | 0.0 |
| enet_stats3 | stats3 | 10 | 1.64 | 0.08 | 2.25 | 1.19 | 1.98 | -0.04 | 0.2 |
| cycle_cond_xgb | cyc | 10 | 1.95 | 0.12 | 1.97 | 1.52 | 2.32 | -0.06 | 0.1 |
| urelax_linear | urelax | 10 | 2.58 | 0.09 | 4.12 | 1.73 | 1.53 | -0.09 | 0.0 |
| mean | raw14 | 10 | 5.98 | 0.07 | 5.77 | 5.01 | 6.73 | 0.01 | 0.0 |

## Family: time (first 80% of each cell vs last 20%)

| model | features | splits | RMSE | ± std | RMSE balanced | MAE | RMSE (SOH<80%) | bias | fit s |
|---|---|---|---|---|---|---|---|---|---|
| mlp_ecm6 | ecm6 | 1 | 1.54 |  | 2.76 | 1.01 | 1.61 | 0.81 | 2.3 |
| mlp_raw14 | raw14 | 1 | 1.66 |  | 2.34 | 1.30 | 1.74 | 1.12 | 1.6 |
| mlp_stats3 | stats3 | 1 | 1.84 |  | 2.61 | 1.44 | 1.93 | 1.29 | 2.0 |
| svr_ecm6 | ecm6 | 1 | 2.07 |  | 3.22 | 1.56 | 2.18 | 1.43 | 3.8 |
| gpr_ecm6 | ecm6 | 1 | 2.19 |  | 3.37 | 1.66 | 2.30 | 1.54 | 2.2 |
| urelax_linear | urelax | 1 | 2.29 |  | 2.25 | 2.06 | 2.36 | 1.84 | 0.0 |
| gpr_stats6 | stats6 | 1 | 2.30 |  | 3.77 | 1.73 | 2.42 | 1.61 | 2.1 |
| gpr_raw14 | raw14 | 1 | 2.38 |  | 3.41 | 1.88 | 2.50 | 1.74 | 8.5 |
| xgb_raw14 | raw14 | 1 | 2.48 |  | 4.04 | 1.89 | 2.60 | 1.69 | 0.0 |
| xgb_ecm6 | ecm6 | 1 | 2.49 |  | 3.66 | 1.91 | 2.61 | 1.73 | 0.0 |
| xgb_stats6 | stats6 | 1 | 2.52 |  | 3.99 | 1.99 | 2.65 | 1.79 | 0.0 |
| xgb_stats3 | stats3 | 1 | 2.52 |  | 3.68 | 2.01 | 2.62 | 1.84 | 0.0 |
| enet_stats3 | stats3 | 1 | 2.60 |  | 3.26 | 1.96 | 2.73 | 0.49 | 0.3 |
| svr_raw14 | raw14 | 1 | 2.94 |  | 3.48 | 2.25 | 3.11 | 2.13 | 2.4 |
| svr_stats6 | stats6 | 1 | 3.23 |  | 4.46 | 2.45 | 3.41 | 2.33 | 3.6 |
| svr_stats3 | stats3 | 1 | 3.44 |  | 4.62 | 2.58 | 3.63 | 2.45 | 3.5 |
| cycle_cond_xgb | cyc | 1 | 3.78 |  | 3.92 | 3.36 | 3.94 | 3.04 | 0.2 |
| mean | raw14 | 1 | 9.46 |  | 9.05 | 8.98 | 9.99 | 8.85 | 0.0 |

## Family: temp (temperature held out; RMSE per held-out group)

| model | hold_out_25C | hold_out_35C | hold_out_45C | mean | worst |
|---|---|---|---|---|---|
| gpr_ecm6 | 3.28 | 1.14 | 1.63 | 2.02 | 3.28 |
| svr_ecm6 | 3.44 | 1.19 | 1.87 | 2.17 | 3.44 |
| gpr_stats6 | 3.78 | 1.34 | 1.65 | 2.26 | 3.78 |
| gpr_raw14 | 4.07 | 1.37 | 1.62 | 2.35 | 4.07 |
| svr_raw14 | 3.12 | 1.36 | 2.95 | 2.47 | 3.12 |
| xgb_ecm6 | 3.82 | 1.39 | 2.42 | 2.54 | 3.82 |
| svr_stats3 | 3.72 | 1.47 | 2.48 | 2.56 | 3.72 |
| svr_stats6 | 3.48 | 1.46 | 2.76 | 2.57 | 3.48 |
| mlp_raw14 | 4.28 | 1.38 | 2.05 | 2.57 | 4.28 |
| xgb_stats3 | 4.05 | 1.54 | 2.62 | 2.74 | 4.05 |
| mlp_stats3 | 4.43 | 1.38 | 2.77 | 2.86 | 4.43 |
| xgb_stats6 | 4.30 | 1.70 | 2.75 | 2.92 | 4.30 |
| xgb_raw14 | 4.93 | 1.72 | 2.42 | 3.02 | 4.93 |
| enet_stats3 | 5.65 | 0.73 | 2.80 | 3.06 | 5.65 |
| urelax_linear | 5.36 | 1.06 | 3.89 | 3.44 | 5.36 |
| cycle_cond_xgb | 4.87 | 7.15 | 2.32 | 4.78 | 7.15 |
| mean | 6.10 | 5.44 | 6.45 | 6.00 | 6.45 |
| mlp_ecm6 | 68.24 | 1.20 | 1.73 | 23.72 | 68.24 |

## Family: condition (operating condition held out; RMSE per held-out group)

| model | hold_out_T25_chg0.25_dis1 | hold_out_T25_chg0.5_dis1 | hold_out_T25_chg1_dis1 | hold_out_T35_chg0.5_dis1 | hold_out_T45_chg0.5_dis1 | mean | worst |
|---|---|---|---|---|---|---|---|
| mlp_raw14 | 2.17 | 4.11 | 4.68 | 1.38 | 2.05 | 2.88 | 4.68 |
| gpr_ecm6 | 2.75 | 3.52 | 5.85 | 1.14 | 1.63 | 2.98 | 5.85 |
| svr_raw14 | 2.05 | 3.34 | 5.73 | 1.36 | 2.95 | 3.09 | 5.73 |
| svr_ecm6 | 2.45 | 3.54 | 7.05 | 1.19 | 1.87 | 3.22 | 7.05 |
| gpr_stats6 | 2.06 | 5.34 | 6.02 | 1.34 | 1.65 | 3.28 | 6.02 |
| gpr_raw14 | 2.10 | 4.63 | 6.90 | 1.37 | 1.62 | 3.32 | 6.90 |
| xgb_ecm6 | 2.54 | 3.78 | 7.24 | 1.39 | 2.42 | 3.47 | 7.24 |
| svr_stats6 | 2.26 | 5.72 | 5.65 | 1.46 | 2.76 | 3.57 | 5.72 |
| svr_stats3 | 2.11 | 6.56 | 5.73 | 1.47 | 2.48 | 3.67 | 6.56 |
| xgb_raw14 | 2.52 | 5.31 | 6.82 | 1.72 | 2.42 | 3.76 | 6.82 |
| xgb_stats6 | 2.25 | 5.13 | 10.01 | 1.70 | 2.75 | 4.37 | 10.01 |
| xgb_stats3 | 2.29 | 5.37 | 10.11 | 1.54 | 2.62 | 4.38 | 10.11 |
| enet_stats3 | 2.97 | 2.83 | 13.06 | 0.73 | 2.80 | 4.48 | 13.06 |
| urelax_linear | 2.45 | 5.36 | 10.84 | 1.06 | 3.89 | 4.72 | 10.84 |
| mlp_ecm6 | 1.95 | 3.49 | 16.37 | 1.20 | 1.73 | 4.95 | 16.37 |
| cycle_cond_xgb | 4.51 | 3.76 | 9.75 | 7.15 | 2.32 | 5.50 | 9.75 |
| mean | 5.81 | 6.17 | 5.42 | 5.44 | 6.45 | 5.86 | 6.45 |
| mlp_stats3 | 2.35 | 5.28 | 26.09 | 1.38 | 2.77 | 7.58 | 26.09 |

## Reproduction check (ours vs the papers' reported RMSE, in % of nominal capacity)

| paper setting | paper | ours | difference | our splits |
|---|---|---|---|---|
| Zhu cell split, XGBoost [Var,Ske,Max] | 1.10 | 1.10 | +0.00 | 10 |
| Zhu cell split, SVR [Var,Ske,Max] | 1.10 | 0.98 | -0.12 | 10 |
| Zhu temp: hold out 45 C, XGBoost | 2.50 | 2.62 | +0.12 | 1 |
| Zhu temp: hold out 45 C, SVR | 3.10 | 2.48 | -0.62 | 1 |
| Zhu temp: hold out 35 C, XGBoost | 1.50 | 1.54 | +0.04 | 1 |
| Zhu temp: hold out 35 C, SVR | 1.50 | 1.47 | -0.03 | 1 |
| Zhu temp: hold out 25 C, XGBoost | 4.40 | 4.05 | -0.35 | 1 |
| Zhu temp: hold out 25 C, SVR | 3.80 | 3.72 | -0.08 | 1 |
| Zhu time split, XGBoost | 2.30 | 2.52 | +0.22 | 1 |
| Zhu time split, SVR | 3.10 | 3.44 | +0.34 | 1 |
| Zhu benchmark: linear on 30-min rest voltage (random 80/20 of cycles) | 2.50 | 2.51 | +0.01 | 10 |
| Feng GPR on 6 ECM features, 50/50 cell split | 0.90 | 1.25 | +0.35 | 5 |

Splits differ from the papers' (theirs are not published as cell lists), so differences of a few tenths of a point are expected.

## Paper reference numbers

Reference numbers from the papers (their data splits differ from ours, so compare loosely):
- Zhu et al. 2022, cell-stratified split (Strategy D), XGBoost / SVR on [Var, Ske, Max]: 1.1% / 1.1%
- Zhu et al. Supplementary Table 6, temperature held out (XGBoost / SVR):
    train 25+35 C, test 45 C: 2.5% / 3.1%;  train 25+45 C, test 35 C: 1.5% / 1.5%;  train 35+45 C, test 25 C: 4.4% / 3.8%
  (the main text quotes only the best of these three, 1.5%)
- Zhu et al. Supplementary Table 7, first 80% vs last 20% of the data: XGBoost 2.3%, SVR 3.1%
- Zhu et al. benchmark, linear model on the 30-min rest voltage: 2.5%
- Feng et al. 2023, GPR on the six ECM features, Dataset 1 (their own split): 0.90%
