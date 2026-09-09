# VisDA-C PU and FO class-wise comparison

## Scope and provenance

- All Formal LBI rows were read exclusively from `visda_fc_lbi_formal_seed2026_20260825`.
- All non-LBI rows were read exclusively from `visda_fc_lbi_seed2026_search_20260819`.
- Each of the 15 selected `summary.json` files was read directly. No tuning LBI artifact was used.
- `class-names` was identical in all 15 summaries, in this order: aeroplane, bicycle, bus, car, horse, knife, motorcycle, person, plant, skateboard, train, truck.
- For every row and phase, the mean recomputed from 12 class accuracies matched the corresponding summary mean-class accuracy within 1e-9.

| Method | Budget | Run ID | summary.json absolute path | PU mAcc | PU overall | FO mAcc | FO overall |
|---|---|---|---|---|---|---|---|
| Source-only | - | 20260819_113043_150389_TV | /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/none/none/source_only/seed_2026/20260819_113043_150389_TV/summary.json | 47.878 | 52.990 | 48.212 | 53.279 |
| Module-Dense | full | 20260819_113043_109480_TV | /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/full/dense/seed_2026/20260819_113043_109480_TV/summary.json | 47.113 | 51.179 | 49.019 | 52.649 |
| Full-Dense | full | 20260819_113043_130632_TV | /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netF_netB/full/dense/seed_2026/20260819_113043_130632_TV/summary.json | 70.435 | 69.031 | 73.389 | 71.674 |
| Module-Random | 0.0005 | 20260819_113039_410471_TV | /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.0005/random/seed_2026/20260819_113039_410471_TV/summary.json | 47.866 | 52.987 | 48.201 | 53.273 |
| Module-Random | 0.001 | 20260819_113039_409926_TV | /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.001/random/seed_2026/20260819_113039_409926_TV/summary.json | 47.861 | 52.986 | 48.190 | 53.269 |
| Module-Random | 0.002 | 20260819_113039_410110_TV | /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.002/random/seed_2026/20260819_113039_410110_TV/summary.json | 47.858 | 52.993 | 48.171 | 53.270 |
| Module-Magnitude | 0.0005 | 20260819_113043_415625_TV | /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.0005/magnitude/seed_2026/20260819_113043_415625_TV/summary.json | 47.878 | 52.997 | 48.200 | 53.270 |
| Module-Magnitude | 0.001 | 20260819_113043_294425_TV | /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.001/magnitude/seed_2026/20260819_113043_294425_TV/summary.json | 47.878 | 52.997 | 48.217 | 53.291 |
| Module-Magnitude | 0.002 | 20260819_113610_190923_TV | /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.002/magnitude/seed_2026/20260819_113610_190923_TV/summary.json | 47.875 | 53.001 | 48.180 | 53.279 |
| Module-Saliency | 0.0005 | 20260819_113612_041746_TV | /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.0005/saliency/seed_2026/20260819_113612_041746_TV/summary.json | 47.652 | 52.950 | 47.772 | 53.104 |
| Module-Saliency | 0.001 | 20260819_113612_204319_TV | /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.001/saliency/seed_2026/20260819_113612_204319_TV/summary.json | 47.551 | 52.919 | 47.525 | 52.992 |
| Module-Saliency | 0.002 | 20260819_113612_266151_TV | /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.002/saliency/seed_2026/20260819_113612_266151_TV/summary.json | 47.437 | 52.910 | 47.266 | 52.874 |
| Formal LBI | 0.0005 | 20260825_040836_611977_TV | /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_formal_seed2026_20260825/runs/VISDA-C/TV/netB_bottleneck/0.0005/LBI/seed_2026/20260825_040836_611977_TV/summary.json | 51.573 | 55.037 | 53.393 | 55.019 |
| Formal LBI | 0.001 | 20260825_040836_612277_TV | /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_formal_seed2026_20260825/runs/VISDA-C/TV/netB_bottleneck/0.001/LBI/seed_2026/20260825_040836_612277_TV/summary.json | 50.487 | 54.967 | 51.966 | 55.716 |
| Formal LBI | 0.002 | 20260825_040836_611580_TV | /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_formal_seed2026_20260825/runs/VISDA-C/TV/netB_bottleneck/0.002/LBI/seed_2026/20260825_040836_611580_TV/summary.json | 50.958 | 55.183 | 52.985 | 56.292 |

## 12-class PU accuracy

All entries are PU accuracy (%), rounded to three decimals.

| Method | Budget | aeroplane | bicycle | bus | car | horse | knife | motorcycle | person | plant | skateboard | train | truck |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Source-only | - | 54.224 | 29.180 | 49.403 | 74.762 | 65.700 | 4.434 | 76.380 | 21.575 | 67.773 | 47.041 | 77.597 | 6.471 |
| Module-Dense | full | 57.323 | 11.194 | 79.211 | 65.551 | 71.413 | 0.771 | 86.991 | 15.300 | 39.261 | 68.523 | 69.145 | 0.667 |
| Full-Dense | full | 94.021 | 60.719 | 83.859 | 58.187 | 88.446 | 68.241 | 84.023 | 62.275 | 75.555 | 61.201 | 82.082 | 26.604 |
| Module-Random | 0.0005 | 54.196 | 29.103 | 49.517 | 74.762 | 65.686 | 4.434 | 76.513 | 21.517 | 67.663 | 46.953 | 77.573 | 6.471 |
| Module-Random | 0.001 | 54.178 | 29.017 | 49.680 | 74.756 | 65.651 | 4.434 | 76.570 | 21.500 | 67.597 | 46.924 | 77.557 | 6.465 |
| Module-Random | 0.002 | 54.178 | 28.844 | 49.986 | 74.740 | 65.608 | 4.434 | 76.754 | 21.450 | 67.443 | 46.865 | 77.542 | 6.453 |
| Module-Magnitude | 0.0005 | 54.224 | 29.122 | 49.552 | 74.762 | 65.764 | 4.434 | 76.501 | 21.500 | 67.641 | 46.997 | 77.573 | 6.471 |
| Module-Magnitude | 0.001 | 54.251 | 29.007 | 49.680 | 74.733 | 65.700 | 4.434 | 76.639 | 21.475 | 67.531 | 47.041 | 77.573 | 6.471 |
| Module-Magnitude | 0.002 | 54.196 | 28.863 | 49.957 | 74.714 | 65.615 | 4.434 | 76.812 | 21.450 | 67.421 | 47.041 | 77.550 | 6.453 |
| Module-Saliency | 0.0005 | 53.812 | 28.403 | 49.680 | 75.820 | 65.167 | 4.434 | 77.053 | 21.100 | 66.850 | 46.164 | 77.266 | 6.074 |
| Module-Saliency | 0.001 | 53.620 | 27.885 | 50.299 | 76.050 | 65.103 | 4.434 | 77.588 | 20.775 | 66.344 | 45.726 | 77.007 | 5.786 |
| Module-Saliency | 0.002 | 53.346 | 27.338 | 51.407 | 76.348 | 65.018 | 4.482 | 78.416 | 20.225 | 65.465 | 44.849 | 76.818 | 5.534 |
| Formal LBI | 0.0005 | 54.690 | 44.288 | 48.273 | 70.118 | 75.698 | 18.410 | 74.310 | 22.950 | 73.269 | 51.469 | 76.889 | 8.508 |
| Formal LBI | 0.001 | 54.553 | 38.446 | 50.533 | 73.464 | 72.415 | 14.940 | 78.106 | 20.575 | 74.764 | 43.753 | 78.045 | 6.255 |
| Formal LBI | 0.002 | 56.171 | 40.604 | 49.275 | 72.733 | 73.865 | 10.217 | 76.018 | 24.725 | 73.730 | 49.759 | 77.266 | 7.138 |

## 12-class FO accuracy

All entries are FO accuracy (%), rounded to three decimals.

| Method | Budget | aeroplane | bicycle | bus | car | horse | knife | motorcycle | person | plant | skateboard | train | truck |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Source-only | - | 53.730 | 28.835 | 50.426 | 75.002 | 66.404 | 4.675 | 76.311 | 21.875 | 68.015 | 48.575 | 78.329 | 6.363 |
| Module-Dense | full | 60.889 | 12.777 | 81.194 | 65.100 | 74.142 | 0.096 | 86.301 | 26.000 | 37.942 | 72.907 | 70.562 | 0.324 |
| Full-Dense | full | 94.844 | 64.230 | 84.797 | 61.513 | 90.386 | 77.060 | 86.404 | 67.900 | 79.534 | 64.621 | 83.026 | 26.352 |
| Module-Random | 0.0005 | 53.721 | 28.710 | 50.682 | 74.986 | 66.432 | 4.675 | 76.375 | 21.908 | 67.854 | 48.502 | 78.266 | 6.297 |
| Module-Random | 0.001 | 53.684 | 28.604 | 50.931 | 74.990 | 66.446 | 4.675 | 76.415 | 21.825 | 67.744 | 48.444 | 78.250 | 6.267 |
| Module-Random | 0.002 | 53.694 | 28.451 | 51.308 | 74.996 | 66.425 | 4.675 | 76.714 | 21.633 | 67.495 | 48.254 | 78.234 | 6.176 |
| Module-Magnitude | 0.0005 | 53.730 | 28.719 | 50.725 | 74.954 | 66.510 | 4.675 | 76.380 | 21.800 | 67.817 | 48.531 | 78.234 | 6.327 |
| Module-Magnitude | 0.001 | 53.703 | 28.604 | 50.959 | 74.954 | 66.596 | 4.675 | 76.570 | 21.700 | 67.663 | 48.619 | 78.234 | 6.327 |
| Module-Magnitude | 0.002 | 53.703 | 28.432 | 51.237 | 74.974 | 66.446 | 4.675 | 76.829 | 21.500 | 67.553 | 48.312 | 78.258 | 6.236 |
| Module-Saliency | 0.0005 | 53.044 | 27.712 | 50.661 | 76.310 | 65.764 | 4.675 | 77.277 | 21.075 | 66.256 | 46.909 | 77.880 | 5.696 |
| Module-Saliency | 0.001 | 52.688 | 26.964 | 51.514 | 76.771 | 65.402 | 4.627 | 77.985 | 20.450 | 65.179 | 45.901 | 77.526 | 5.299 |
| Module-Saliency | 0.002 | 52.194 | 25.122 | 53.753 | 76.820 | 65.189 | 4.578 | 79.348 | 19.525 | 63.970 | 44.717 | 77.125 | 4.849 |
| Formal LBI | 0.0005 | 55.019 | 57.640 | 47.974 | 64.369 | 79.663 | 41.398 | 70.618 | 22.125 | 66.872 | 48.356 | 76.818 | 9.859 |
| Formal LBI | 0.001 | 54.800 | 46.331 | 50.490 | 72.926 | 75.272 | 32.771 | 77.295 | 16.775 | 76.302 | 36.563 | 78.683 | 5.389 |
| Formal LBI | 0.002 | 58.557 | 50.906 | 49.339 | 70.224 | 78.427 | 18.265 | 74.500 | 27.200 | 74.126 | 49.101 | 78.069 | 7.102 |

## PU hard classes

| Method | Budget | knife | person | truck | PU worst-class name | PU worst-class acc |
|---|---|---|---|---|---|---|
| Source-only | - | 4.434 | 21.575 | 6.471 | knife | 4.434 |
| Module-Dense | full | 0.771 | 15.300 | 0.667 | truck | 0.667 |
| Full-Dense | full | 68.241 | 62.275 | 26.604 | truck | 26.604 |
| Module-Random | 0.0005 | 4.434 | 21.517 | 6.471 | knife | 4.434 |
| Module-Random | 0.001 | 4.434 | 21.500 | 6.465 | knife | 4.434 |
| Module-Random | 0.002 | 4.434 | 21.450 | 6.453 | knife | 4.434 |
| Module-Magnitude | 0.0005 | 4.434 | 21.500 | 6.471 | knife | 4.434 |
| Module-Magnitude | 0.001 | 4.434 | 21.475 | 6.471 | knife | 4.434 |
| Module-Magnitude | 0.002 | 4.434 | 21.450 | 6.453 | knife | 4.434 |
| Module-Saliency | 0.0005 | 4.434 | 21.100 | 6.074 | knife | 4.434 |
| Module-Saliency | 0.001 | 4.434 | 20.775 | 5.786 | knife | 4.434 |
| Module-Saliency | 0.002 | 4.482 | 20.225 | 5.534 | knife | 4.482 |
| Formal LBI | 0.0005 | 18.410 | 22.950 | 8.508 | truck | 8.508 |
| Formal LBI | 0.001 | 14.940 | 20.575 | 6.255 | truck | 6.255 |
| Formal LBI | 0.002 | 10.217 | 24.725 | 7.138 | truck | 7.138 |

## FO hard classes

| Method | Budget | knife | person | truck | FO worst-class name | FO worst-class acc |
|---|---|---|---|---|---|---|
| Source-only | - | 4.675 | 21.875 | 6.363 | knife | 4.675 |
| Module-Dense | full | 0.096 | 26.000 | 0.324 | knife | 0.096 |
| Full-Dense | full | 77.060 | 67.900 | 26.352 | truck | 26.352 |
| Module-Random | 0.0005 | 4.675 | 21.908 | 6.297 | knife | 4.675 |
| Module-Random | 0.001 | 4.675 | 21.825 | 6.267 | knife | 4.675 |
| Module-Random | 0.002 | 4.675 | 21.633 | 6.176 | knife | 4.675 |
| Module-Magnitude | 0.0005 | 4.675 | 21.800 | 6.327 | knife | 4.675 |
| Module-Magnitude | 0.001 | 4.675 | 21.700 | 6.327 | knife | 4.675 |
| Module-Magnitude | 0.002 | 4.675 | 21.500 | 6.236 | knife | 4.675 |
| Module-Saliency | 0.0005 | 4.675 | 21.075 | 5.696 | knife | 4.675 |
| Module-Saliency | 0.001 | 4.627 | 20.450 | 5.299 | knife | 4.627 |
| Module-Saliency | 0.002 | 4.578 | 19.525 | 4.849 | knife | 4.578 |
| Formal LBI | 0.0005 | 41.398 | 22.125 | 9.859 | truck | 9.859 |
| Formal LBI | 0.001 | 32.771 | 16.775 | 5.389 | truck | 5.389 |
| Formal LBI | 0.002 | 18.265 | 27.200 | 7.102 | truck | 7.102 |

## Formal LBI deltas: PU

Every delta is Formal LBI minus the named comparator, in percentage points (pp). The same-budget best sparse baseline is selected from Random, Magnitude, and Saliency by PU mAcc.

| LBI budget | Best sparse baseline (PU mAcc) | truck vs Source | knife vs Source | truck vs best sparse | knife vs best sparse | mAcc vs best sparse | truck vs Module-Dense | knife vs Module-Dense |
|---|---|---|---|---|---|---|---|---|
| 0.0005 | Module-Magnitude (mAcc 47.878) | +2.037 | +13.976 | +2.037 | +13.976 | +3.694 | +7.841 | +17.639 |
| 0.001 | Module-Magnitude (mAcc 47.878) | -0.216 | +10.506 | -0.216 | +10.506 | +2.609 | +5.588 | +14.169 |
| 0.002 | Module-Magnitude (mAcc 47.875) | +0.667 | +5.783 | +0.685 | +5.783 | +3.083 | +6.471 | +9.446 |

## Formal LBI deltas: FO

Every delta is Formal LBI minus the named comparator, in percentage points (pp). The same-budget best sparse baseline is selected from Random, Magnitude, and Saliency by FO mAcc.

| LBI budget | Best sparse baseline (FO mAcc) | truck vs Source | knife vs Source | truck vs best sparse | knife vs best sparse | mAcc vs best sparse | truck vs Module-Dense | knife vs Module-Dense |
|---|---|---|---|---|---|---|---|---|
| 0.0005 | Module-Random (mAcc 48.201) | +3.497 | +36.723 | +3.563 | +36.723 | +5.192 | +9.535 | +41.301 |
| 0.001 | Module-Magnitude (mAcc 48.217) | -0.973 | +28.096 | -0.937 | +28.096 | +3.749 | +5.065 | +32.675 |
| 0.002 | Module-Magnitude (mAcc 48.180) | +0.739 | +13.590 | +0.865 | +13.590 | +4.805 | +6.777 | +18.169 |

## PU-to-FO hard-class changes

Each delta is FO minus PU, in pp; positive values are higher under FO.

| Method | Budget | PU knife | FO knife | knife delta | PU person | FO person | person delta | PU truck | FO truck | truck delta |
|---|---|---|---|---|---|---|---|---|---|---|
| Source-only | - | 4.434 | 4.675 | +0.241 | 21.575 | 21.875 | +0.300 | 6.471 | 6.363 | -0.108 |
| Module-Dense | full | 0.771 | 0.096 | -0.675 | 15.300 | 26.000 | +10.700 | 0.667 | 0.324 | -0.342 |
| Full-Dense | full | 68.241 | 77.060 | +8.819 | 62.275 | 67.900 | +5.625 | 26.604 | 26.352 | -0.252 |
| Module-Random | 0.0005 | 4.434 | 4.675 | +0.241 | 21.517 | 21.908 | +0.392 | 6.471 | 6.297 | -0.174 |
| Module-Random | 0.001 | 4.434 | 4.675 | +0.241 | 21.500 | 21.825 | +0.325 | 6.465 | 6.267 | -0.198 |
| Module-Random | 0.002 | 4.434 | 4.675 | +0.241 | 21.450 | 21.633 | +0.183 | 6.453 | 6.176 | -0.276 |
| Module-Magnitude | 0.0005 | 4.434 | 4.675 | +0.241 | 21.500 | 21.800 | +0.300 | 6.471 | 6.327 | -0.144 |
| Module-Magnitude | 0.001 | 4.434 | 4.675 | +0.241 | 21.475 | 21.700 | +0.225 | 6.471 | 6.327 | -0.144 |
| Module-Magnitude | 0.002 | 4.434 | 4.675 | +0.241 | 21.450 | 21.500 | +0.050 | 6.453 | 6.236 | -0.216 |
| Module-Saliency | 0.0005 | 4.434 | 4.675 | +0.241 | 21.100 | 21.075 | -0.025 | 6.074 | 5.696 | -0.379 |
| Module-Saliency | 0.001 | 4.434 | 4.627 | +0.193 | 20.775 | 20.450 | -0.325 | 5.786 | 5.299 | -0.487 |
| Module-Saliency | 0.002 | 4.482 | 4.578 | +0.096 | 20.225 | 19.525 | -0.700 | 5.534 | 4.849 | -0.685 |
| Formal LBI | 0.0005 | 18.410 | 41.398 | +22.988 | 22.950 | 22.125 | -0.825 | 8.508 | 9.859 | +1.352 |
| Formal LBI | 0.001 | 14.940 | 32.771 | +17.831 | 20.575 | 16.775 | -3.800 | 6.255 | 5.389 | -0.865 |
| Formal LBI | 0.002 | 10.217 | 18.265 | +8.048 | 24.725 | 27.200 | +2.475 | 7.138 | 7.102 | -0.036 |

## Direct answers supported by the artifacts

- The artifacts do not define a numeric threshold for 'low'. Under PU, Source-only is 4.434% knife and 6.471% truck; sparse baselines span 4.434% to 4.482% knife and 5.534% to 6.471% truck; Module-Dense is 0.771%/0.667%; Full-Dense is 68.241%/26.604%.
- Under FO, Source-only is 4.675% knife and 6.363% truck; sparse baselines span 4.578% to 4.675% knife and 4.849% to 6.327% truck; Module-Dense is 0.096%/0.324%; Full-Dense is 77.060%/26.352%.
- Formal LBI PU worst classes: rho=0.0005: truck (8.508%); rho=0.001: truck (6.255%); rho=0.002: truck (7.138%).
- Formal LBI FO worst classes: rho=0.0005: truck (9.859%); rho=0.001: truck (5.389%); rho=0.002: truck (7.102%). Thus, all three FO worst classes are truck.
- From PU to FO: Source-only and all nine sparse baselines have higher knife accuracy and lower truck accuracy; person is higher for Source-only, Random, and Magnitude, and lower for Saliency. Module-Dense has lower knife and truck and higher person; Full-Dense has higher knife and person and lower truck.
- For Formal LBI from PU to FO: knife is higher at all three budgets (+22.988, +17.831, +8.048 pp for 0.0005/0.001/0.002); person is lower at 0.0005 and 0.001 (-0.825, -3.800 pp) and higher at 0.002 (+2.475 pp); truck is higher at 0.0005 (+1.352 pp) and lower at 0.001 and 0.002 (-0.865, -0.036 pp).
- Versus the phase-specific same-budget best sparse baseline, LBI improves both knife and truck at 0.0005 and 0.002 in PU and FO. At 0.001, it improves knife but worsens truck in PU (-0.216 pp truck) and FO (-0.937 pp truck).
