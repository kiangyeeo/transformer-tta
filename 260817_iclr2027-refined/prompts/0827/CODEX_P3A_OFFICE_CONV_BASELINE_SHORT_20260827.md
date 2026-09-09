# Codex Prompt — Office D→A Conv Baseline Budget Pilot（4 GPU）

项目：
`/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined`

请以当前 frozen Conv 实现为唯一规范，先阅读并复用已有代码/工具，不要重复造轮子：

- Conv protocol：`protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_v1.md`
- FC protocol（共享 OTTA / evaluation 语义参考）：`protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md`
- Conv base config：`configs/otta_conv_lbi_protocol_20260826_v1.yaml`
- 现有 planner / launcher / summary 工具：直接复用 `tools/` 下当前 refined 版本
- 可参考之前 FC non-interactive 实验脚本/phase-record 的组织方式，但不要修改 FC

## 任务

只做 **Office-31 DSLR → Amazon（D→A）Conv baseline budget pilot**。

不要跑：
- Conv-LBI
- VisDA
- Office 其他 transfer
- 搜参
- 正式全量实验

候选 budget：

```text
0.0005
0.001
0.002
0.005
```

理论 group budget 必须核对：

```text
out_channel:        4 / 9 / 18 / 46
filter_connection:  3276 / 6553 / 13107 / 32768
```

运行条件共 25 个 scientific conditions：

```text
conv_module_dense                         1

conv_out_random                           4 budgets
conv_out_magnitude                        4 budgets
conv_out_saliency                         4 budgets

conv_filter_random                        4 budgets
conv_filter_magnitude                     4 budgets
conv_filter_saliency                      4 budgets
```

Random 严格复用当前 frozen 语义：
`seed=2026 + 3 deterministic child masks`。

## 调度

只用 GPU `0,1,2,3`，每卡最多 1 个任务，严格按 wave 顺序：

```text
Wave 0  conv_module_dense
Wave 1  conv_out_random       × 4 budgets
Wave 2  conv_out_magnitude    × 4 budgets
Wave 3  conv_out_saliency     × 4 budgets
Wave 4  conv_filter_random    × 4 budgets
Wave 5  conv_filter_magnitude × 4 budgets
Wave 6  conv_filter_saliency  × 4 budgets
```

一个 wave 全部结束后才能进入下一个 wave。

## PREPARE

只准备，不启动训练。

请：
1. 确认 repository 中 D→A 的真实 transfer identifier；
2. 创建 pilot matrix / plan；
3. 检查只有上述 25 个 conditions，且没有 `conv_*_lbi`；
4. 检查四个 budget 对应 K_G 正确；
5. 创建：
   `tools/run_office_conv_baseline_budget_pilot_noninteractive.sh`
6. launcher 使用：
   - GPU 0-3
   - max-workers=4
   - workers-per-gpu=1
   - `--resume --resume-partial-runs`
7. `.sh` 内所有 Python/runner 命令统一通过：
   `conda run --no-capture-output -n SHOT_TTA ...`
   不要要求用户手动 `conda activate`。
8. 做静态检查：plan uniqueness、identity uniqueness、`bash -n`、`git diff --check`。
9. 写 PREPARE record 到独立 pilot log root，例如：
   `experiment_logs/office_conv_baseline_budget_pilot_seed2026_20260827/phase_records/p3a_office_da/PREPARE.md`

不要修改算法、protocol、FC 或 frozen revision。

## FINALIZE

只汇总，不 launch / retry / rerun。

至少报告：

```text
variant
group_mode
rho
K_G
selected_group_count
selected_scalar_count
realized_scalar_ratio
PU
FO
runtime
peak GPU memory
```

同时给：
- `FO - conv_module_dense`；
- Random 3-mask aggregation；
- out/filter 两种 grouping 的 scalar footprint 对比；
- 是否有 collapse / runtime anomaly。

这一步只做 budget sanity，不要根据最高 accuracy 自动冻结 rho，也不要自动进入下一阶段。

## PREPARE 最后输出

Codex 最后只输出下面两行，不要再输出解释，也不要输出 `conda activate`：

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
bash tools/run_office_conv_baseline_budget_pilot_noninteractive.sh
```
