# VisDA-C Group-LBI Stage-1 Anchor Summary

Generated from the completed VisDA-C Group-LBI Stage-1 search through 2026-08-26.

## Aggregation and fixed settings

- Formal seed: `2026`
- Transfer: `train -> validation`
- Budgets: `0.0005 / 0.001 / 0.002`, corresponding to `K = 3 / 6 / 13`
- `PU` and `FO`: fixed-12-class macro accuracy, in percent
- All utilization, step, rollback, realized-K, violation, and failure diagnostics: aggregated over all online batches in the VisDA-C target stream
- Fixed across A1--A8: `omega=0.20`, `stage2_lr=1e-5`, `stage2_steps=1`, `prox_lambda=1.0`, `tau_g=1e-4`, `stage1_max_steps=3000`
- Rate and utilization fields are reported as fractions in `[0, 1]`

Source root:

```text
/home/nas3/biod/wangkangyi/results/transformer_otta_group_lbi_tuning/stage1/visda-c/
```

## Budget 0.0005 (K=3)

| ID | alpha | kappa | nu | PU | FO | mean_utilization | min_utilization | util90_rate | util95_rate | 3000_hit_rate | stage1_mean_steps | stage1_max_steps | rollback_rate | mean_realized_K | budget_violation_rate | failure_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 0.1 | 1 | 0.5 | 49.22 | 50.99 | 0.9939 | 0.6667 | 0.9816 | 0.9816 | 0.0000 | 153.5 | 213 | 0.0184 | 2.98 | 0.0000 | 0.0000 |
| A2 | 0.15 | 1 | 0.5 | 48.58 | 51.42 | 0.9908 | 0.6667 | 0.9724 | 0.9724 | 0.0000 | 100.3 | 142 | 0.0276 | 2.97 | 0.0000 | 0.0000 |
| A3 | 0.2 | 1 | 0.5 | 50.47 | 52.80 | 0.9816 | 0.3333 | 0.9493 | 0.9493 | 0.0000 | 76.4 | 104 | 0.0507 | 2.94 | 0.0000 | 0.0000 |
| A4 | 0.1 | 1.5 | 0.5 | 48.17 | 50.83 | 0.9969 | 0.6667 | 0.9908 | 0.9908 | 0.0000 | 157.5 | 223 | 0.0092 | 2.99 | 0.0000 | 0.0000 |
| A5 | 0.1 | 2 | 0.5 | 53.15 | 56.07 | 0.9923 | 0.6667 | 0.9770 | 0.9770 | 0.0000 | 137.1 | 191 | 0.0230 | 2.98 | 0.0000 | 0.0000 |
| A6 | 0.1 | 1 | 0.25 | 50.65 | 53.33 | 0.9892 | 0.6667 | 0.9677 | 0.9677 | 0.0000 | 118.4 | 169 | 0.0323 | 2.97 | 0.0000 | 0.0000 |
| A7 | 0.1 | 1 | 1 | 46.20 | 46.41 | 0.9954 | 0.6667 | 0.9862 | 0.9862 | 0.0000 | 233.5 | 331 | 0.0138 | 2.99 | 0.0000 | 0.0000 |
| A8 | 0.15 | 1 | 1 | 47.98 | 48.63 | 0.9846 | 0.6667 | 0.9539 | 0.9539 | 0.0000 | 142.4 | 198 | 0.0461 | 2.95 | 0.0000 | 0.0000 |

## Budget 0.001 (K=6)

| ID | alpha | kappa | nu | PU | FO | mean_utilization | min_utilization | util90_rate | util95_rate | 3000_hit_rate | stage1_mean_steps | stage1_max_steps | rollback_rate | mean_realized_K | budget_violation_rate | failure_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 0.1 | 1 | 0.5 | 49.33 | 51.92 | 0.9946 | 0.8333 | 0.9677 | 0.9677 | 0.0000 | 218.0 | 323 | 0.0323 | 5.97 | 0.0000 | 0.0000 |
| A2 | 0.15 | 1 | 0.5 | 49.91 | 52.37 | 0.9908 | 0.8333 | 0.9447 | 0.9447 | 0.0000 | 140.2 | 197 | 0.0553 | 5.94 | 0.0000 | 0.0000 |
| A3 | 0.2 | 1 | 0.5 | 52.25 | 57.35 | 0.9816 | 0.6667 | 0.8986 | 0.8986 | 0.0000 | 97.7 | 129 | 0.1014 | 5.89 | 0.0000 | 0.0000 |
| A4 | 0.1 | 1.5 | 0.5 | 51.16 | 53.13 | 0.9962 | 0.8333 | 0.9770 | 0.9770 | 0.0000 | 315.8 | 564 | 0.0230 | 5.98 | 0.0000 | 0.0000 |
| A5 | 0.1 | 2 | 0.5 | 55.91 | 61.06 | 0.9946 | 0.8333 | 0.9677 | 0.9677 | 0.0000 | 200.5 | 277 | 0.0323 | 5.97 | 0.0000 | 0.0000 |
| A6 | 0.1 | 1 | 0.25 | 49.73 | 52.13 | 0.9992 | 0.8333 | 0.9954 | 0.9954 | 0.0000 | 194.5 | 305 | 0.0046 | 6.00 | 0.0000 | 0.0000 |
| A7 | 0.1 | 1 | 1 | 47.99 | 49.90 | 0.9954 | 0.8333 | 0.9724 | 0.9724 | 0.0000 | 404.1 | 678 | 0.0276 | 5.97 | 0.0000 | 0.0000 |
| A8 | 0.15 | 1 | 1 | 50.06 | 52.73 | 0.9892 | 0.8333 | 0.9355 | 0.9355 | 0.0000 | 192.9 | 292 | 0.0645 | 5.94 | 0.0000 | 0.0000 |

## Budget 0.002 (K=13)

| ID | alpha | kappa | nu | PU | FO | mean_utilization | min_utilization | util90_rate | util95_rate | 3000_hit_rate | stage1_mean_steps | stage1_max_steps | rollback_rate | mean_realized_K | budget_violation_rate | failure_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 0.1 | 1 | 0.5 | 55.13 | 61.30 | 0.9979 | 0.9231 | 1.0000 | 0.9724 | 0.0000 | 575.6 | 1136 | 0.0276 | 12.97 | 0.0000 | 0.0000 |
| A2 | 0.15 | 1 | 0.5 | 55.91 | 60.23 | 0.9965 | 0.9231 | 1.0000 | 0.9539 | 0.0000 | 389.7 | 934 | 0.0461 | 12.95 | 0.0000 | 0.0000 |
| A3 | 0.2 | 1 | 0.5 | 57.25 | 61.53 | 0.9968 | 0.9231 | 1.0000 | 0.9585 | 0.0000 | 327.4 | 721 | 0.0415 | 12.96 | 0.0000 | 0.0000 |
| A4 | 0.1 | 1.5 | 0.5 | 57.41 | 61.90 | 0.9979 | 0.9231 | 1.0000 | 0.9724 | 0.0000 | 789.3 | 1942 | 0.0276 | 12.97 | 0.0000 | 0.0000 |
| A5 | 0.1 | 2 | 0.5 | 59.29 | 64.35 | 0.9996 | 0.9231 | 1.0000 | 0.9954 | 0.0000 | 988.6 | 2579 | 0.0046 | 13.00 | 0.0000 | 0.0000 |
| A6 | 0.1 | 1 | 0.25 | 55.19 | 61.35 | 0.9986 | 0.9231 | 1.0000 | 0.9816 | 0.0000 | 510.7 | 1074 | 0.0184 | 12.98 | 0.0000 | 0.0000 |
| A7 | 0.1 | 1 | 1 | 50.51 | 54.55 | 0.9993 | 0.9231 | 1.0000 | 0.9908 | 0.0000 | 850.0 | 1553 | 0.0092 | 12.99 | 0.0000 | 0.0000 |
| A8 | 0.15 | 1 | 1 | 56.83 | 62.26 | 0.9982 | 0.9231 | 1.0000 | 0.9770 | 0.0000 | 473.8 | 923 | 0.0230 | 12.98 | 0.0000 | 0.0000 |

## Integrity checks

- Completed anchor/budget combinations: `24 / 24`
- Each combination contains the complete VisDA-C `train -> validation` online stream and an independent FO pass.
- All matrix records report `status=completed` with no launcher failures.
- All `stage1_3000_step_hit_rate` values are `0.0000`.
- All `budget_violation_rate` values are `0.0000`.
- All `failure_rate` values are `0.0000`.
