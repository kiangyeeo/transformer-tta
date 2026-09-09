# VisDA-C FO class-wise comparison

## Scope and provenance

- All LBI rows below were read exclusively from `visda_fc_lbi_formal_seed2026_20260825`.
- All non-LBI rows below were read exclusively from `visda_fc_lbi_seed2026_search_20260819`.
- The class order was read from each run's `class-names` field and was identical for all 15 summaries: aeroplane, bicycle, bus, car, horse, knife, motorcycle, person, plant, skateboard, train, truck.

| Method | Budget | Run ID | `summary.json` path | FO mAcc | FO overall |
|---|---:|---|---|---:|---:|
| Source-only | — | 20260819_113043_150389_TV | `/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/none/none/source_only/seed_2026/20260819_113043_150389_TV/summary.json` | 48.212 | 53.279 |
| Module-Dense | full | 20260819_113043_109480_TV | `/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/full/dense/seed_2026/20260819_113043_109480_TV/summary.json` | 49.019 | 52.649 |
| Full-Dense | full | 20260819_113043_130632_TV | `/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netF_netB/full/dense/seed_2026/20260819_113043_130632_TV/summary.json` | 73.389 | 71.674 |
| Module-Random | 0.0005 | 20260819_113039_410471_TV | `/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.0005/random/seed_2026/20260819_113039_410471_TV/summary.json` | 48.201 | 53.273 |
| Module-Random | 0.001 | 20260819_113039_409926_TV | `/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.001/random/seed_2026/20260819_113039_409926_TV/summary.json` | 48.190 | 53.269 |
| Module-Random | 0.002 | 20260819_113039_410110_TV | `/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.002/random/seed_2026/20260819_113039_410110_TV/summary.json` | 48.171 | 53.270 |
| Module-Magnitude | 0.0005 | 20260819_113043_415625_TV | `/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.0005/magnitude/seed_2026/20260819_113043_415625_TV/summary.json` | 48.200 | 53.270 |
| Module-Magnitude | 0.001 | 20260819_113043_294425_TV | `/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.001/magnitude/seed_2026/20260819_113043_294425_TV/summary.json` | 48.217 | 53.291 |
| Module-Magnitude | 0.002 | 20260819_113610_190923_TV | `/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.002/magnitude/seed_2026/20260819_113610_190923_TV/summary.json` | 48.180 | 53.279 |
| Module-Saliency | 0.0005 | 20260819_113612_041746_TV | `/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.0005/saliency/seed_2026/20260819_113612_041746_TV/summary.json` | 47.772 | 53.104 |
| Module-Saliency | 0.001 | 20260819_113612_204319_TV | `/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.001/saliency/seed_2026/20260819_113612_204319_TV/summary.json` | 47.525 | 52.992 |
| Module-Saliency | 0.002 | 20260819_113612_266151_TV | `/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819/runs/VISDA-C/TV/netB_bottleneck/0.002/saliency/seed_2026/20260819_113612_266151_TV/summary.json` | 47.266 | 52.874 |
| Formal LBI | 0.0005 | 20260825_040836_611977_TV | `/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_formal_seed2026_20260825/runs/VISDA-C/TV/netB_bottleneck/0.0005/LBI/seed_2026/20260825_040836_611977_TV/summary.json` | 53.393 | 55.019 |
| Formal LBI | 0.001 | 20260825_040836_612277_TV | `/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_formal_seed2026_20260825/runs/VISDA-C/TV/netB_bottleneck/0.001/LBI/seed_2026/20260825_040836_612277_TV/summary.json` | 51.966 | 55.716 |
| Formal LBI | 0.002 | 20260825_040836_611580_TV | `/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_formal_seed2026_20260825/runs/VISDA-C/TV/netB_bottleneck/0.002/LBI/seed_2026/20260825_040836_611580_TV/summary.json` | 52.985 | 56.292 |

## 12-class FO accuracy

All entries are FO accuracy (%), rounded to three decimals.

| Method | Budget | aeroplane | bicycle | bus | car | horse | knife | motorcycle | person | plant | skateboard | train | truck |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source-only | — | 53.730 | 28.835 | 50.426 | 75.002 | 66.404 | 4.675 | 76.311 | 21.875 | 68.015 | 48.575 | 78.329 | 6.363 |
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

## Truck comparison

| Method | Budget | Truck accuracy | FO mAcc | Worst-class name | Worst-class accuracy |
|---|---:|---:|---:|---|---:|
| Source-only | — | 6.363 | 48.212 | knife | 4.675 |
| Module-Dense | full | 0.324 | 49.019 | knife | 0.096 |
| Full-Dense | full | 26.352 | 73.389 | truck | 26.352 |
| Module-Random | 0.0005 | 6.297 | 48.201 | knife | 4.675 |
| Module-Random | 0.001 | 6.267 | 48.190 | knife | 4.675 |
| Module-Random | 0.002 | 6.176 | 48.171 | knife | 4.675 |
| Module-Magnitude | 0.0005 | 6.327 | 48.200 | knife | 4.675 |
| Module-Magnitude | 0.001 | 6.327 | 48.217 | knife | 4.675 |
| Module-Magnitude | 0.002 | 6.236 | 48.180 | knife | 4.675 |
| Module-Saliency | 0.0005 | 5.696 | 47.772 | knife | 4.675 |
| Module-Saliency | 0.001 | 5.299 | 47.525 | knife | 4.627 |
| Module-Saliency | 0.002 | 4.849 | 47.266 | knife | 4.578 |
| Formal LBI | 0.0005 | 9.859 | 53.393 | truck | 9.859 |
| Formal LBI | 0.001 | 5.389 | 51.966 | truck | 5.389 |
| Formal LBI | 0.002 | 7.102 | 52.985 | truck | 7.102 |

## Formal-LBI deltas

Each delta is `Formal LBI − comparator` in percentage points (pp). The best sparse baseline is selected independently at each budget by the largest FO mAcc among Module-Random, Module-Magnitude, and Module-Saliency.

| Formal LBI budget | Best sparse baseline (by FO mAcc) | Truck vs Source (pp) | FO mAcc vs Source (pp) | Truck vs best sparse (pp) | FO mAcc vs best sparse (pp) | Truck vs Module-Dense (pp) | FO mAcc vs Module-Dense (pp) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.0005 | Module-Random (mAcc 48.201) | +3.497 | +5.181 | +3.563 | +5.192 | +9.535 | +4.373 |
| 0.001 | Module-Magnitude (mAcc 48.217) | -0.973 | +3.755 | -0.937 | +3.749 | +5.065 | +2.947 |
| 0.002 | Module-Magnitude (mAcc 48.180) | +0.739 | +4.773 | +0.865 | +4.805 | +6.777 | +3.965 |

## Direct answers supported by the artifacts

- Source-only truck FO accuracy is **6.363%**.
- Yes. Module-Random, Module-Magnitude, and Module-Saliency also have low truck FO accuracy: across their nine listed runs it ranges from **4.849%** to **6.327%**.
- Module-Dense truck FO accuracy is **0.324%**.
- Full-Dense truck FO accuracy is **26.352%**.
- Yes. Full-Dense improves truck accuracy by **19.989 pp** relative to Source-only (26.352% versus 6.363%) and by **26.027 pp** relative to Module-Dense (26.352% versus 0.324%).
- Yes. The worst class in every formal-LBI summary is truck: 9.859% (rho=0.0005), 5.389% (rho=0.001), and 7.102% (rho=0.002).
- Relative to the same-budget best sparse baseline, formal LBI improves truck at rho=0.0005 (+3.563 pp) and rho=0.002 (+0.865 pp), and worsens it at rho=0.001 (-0.937 pp).
