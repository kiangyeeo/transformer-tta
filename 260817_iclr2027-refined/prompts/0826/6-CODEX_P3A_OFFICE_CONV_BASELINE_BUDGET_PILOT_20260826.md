# P3a — Office-31 Conv Baseline Budget Pilot（PREPARE / FINALIZE）

Project:
`/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined`

Conda:
`SHOT_TTA`

Conv frozen protocol:
`protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_v1.md`

Conv frozen scientific implementation revision:
`iclr2027_refined_conv_20260826_v1`

FC frozen protocol（仅作为共享 OTTA / evaluation 语义参考，不修改）:
`protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md`

Use current refined code and the frozen Conv protocol as the only algorithm source of truth.

---

## 0. 本任务目标

这是 P3a：

> 只在 Office-31 的 DSLR → Amazon（D→A）上做 Conv sparse baseline budget pilot。

目标是检查 4 个候选 group-ratio budget 的：

- integer group budget；
- selected scalar footprint；
- accuracy sanity；
- runtime / GPU memory；
- 两种 grouping 下的实际 budget geometry。

本轮不跑 Conv-LBI。

本轮不根据 accuracy 自动冻结 rho。

---

## 1. Hard boundaries

- 不修改 algorithm code。
- 不修改 frozen Conv protocol。
- 不修改 frozen FC protocol。
- 不修改已有 correctness tests。
- 不修改 frozen scientific implementation revision。
- 不运行 Conv-LBI。
- 不搜索 alpha / kappa / nu / omega / stage2_lr。
- 不运行 VisDA-C。
- 不运行 Office 其他 5 个 transfers。
- 不做正式 baseline 全量实验。
- 不在 FINALIZE 后自动扩展 budget grid。
- 不自行冻结 formal rho。

允许创建/修改的只有：

- experiment matrix / plan；
- launcher；
- pilot report；
- phase record；
- 必要的 pilot-specific selected/config metadata。

---

## 2. Scientific condition

Dataset:
Office-31

Transfer:
DSLR → Amazon（D→A）

请先检查 repository 当前 Office planner / config 的真实 transfer identifier，
使用项目已有命名，不要自己猜 `D2A` / `d2a` 的字符串格式。

Seed:
2026

其他：

- batch size；
- workers；
- SHOT objective；
- PU / FO；
- BN freeze；
- augmentation；
- target stream；
- optimizer；
- evaluation metric

全部继承 frozen Conv protocol 和当前 refined Office pipeline。

不要为了 pilot 自行修改。

---

## 3. Candidate budget grid

只允许以下 4 个 requested group ratios：

```text
0.0005
0.001
0.002
0.005
```

对应理论 integer group budget 应为：

### out_channel

total_group_count = 9216

```text
rho=.0005 -> K_G=floor(.0005*9216)=4
rho=.001  -> K_G=9
rho=.002  -> K_G=18
rho=.005  -> K_G=46
```

### filter_connection

total_group_count = 6553600

```text
rho=.0005 -> K_G=3276
rho=.001  -> K_G=6553
rho=.002  -> K_G=13107
rho=.005  -> K_G=32768
```

PREPARE 阶段必须检查 planner/config 计算出的 K_G 与上述值完全一致。

不一致则停止，不生成 launcher。

---

## 4. 本轮精确运行条件

先跑一个 dense Conv anchor：

```text
conv_module_dense
```

只跑一次，不带 sparse budget。

然后运行：

### out_channel

```text
conv_out_random
conv_out_magnitude
conv_out_saliency
```

每个 variant 都跑：

```text
rho=.0005
rho=.001
rho=.002
rho=.005
```

共：

3 × 4 = 12 个 scientific conditions。

### filter_connection

```text
conv_filter_random
conv_filter_magnitude
conv_filter_saliency
```

同样每个跑：

```text
rho=.0005
rho=.001
rho=.002
rho=.005
```

共：

3 × 4 = 12 个 scientific conditions。

因此本轮 matrix 总计：

```text
1 conv_module_dense
+ 12 out_channel sparse conditions
+ 12 filter_connection sparse conditions
= 25 scientific conditions
```

---

## 5. Random semantics

不要手工创建新的 random seeds。

严格复用当前 frozen random semantics：

```text
formal seed = 2026
3 deterministic child masks
```

Random parent condition 的 accuracy 使用当前 framework 已冻结的 3-mask aggregation。

保留每个 child mask 的：

- selected_group_count；
- selected_scalar_count；
- realized_group_ratio；
- realized_scalar_ratio。

Parent summary 必须保留：

- selected_scalar_count mean/std/min/max；
- realized_scalar_ratio mean/std/min/max。

不要把最后一个 child mask 当 parent result。

---

## 6. Sparse baseline selection semantics

必须保持当前 frozen 实现：

Random:

```text
global group pool uniform exact-K_G
```

Magnitude:

```text
score_g = ||W_g||_2
global top-K_G
```

Saliency:

```text
score_g = ||W_g * grad_g||_2
每个 online batch dynamic global top-K_G
```

两种 grouping 都禁止：

- per-layer budget；
- per-layer top-K；
- minimum-one-per-layer；
- ceil；
- tolerance slack。

对于 baseline：

```text
selected_group_count == K_G
```

应当严格成立。

如果不成立，视为 invalid，不要静默继续解释结果。

---

## 7. Pilot 主要记录字段

每个 sparse condition 至少汇总：

```text
variant
group_mode
requested_budget
total_group_count
max_group_count / K_G

selected_group_count
realized_group_ratio

selected_scalar_count
realized_scalar_ratio

PU accuracy
FO accuracy

adapt runtime
PU runtime
online runtime

peak allocated GPU memory
peak reserved GPU memory
```

Random parent 额外记录 3-mask mean/std。

Saliency 如果当前 artifacts 已提供 per-batch scalar footprint，
同时报告其 mean/min/max；
不要为了增加该字段修改算法代码。

---

## 8. P3a 的解释规则

本轮的核心不是“哪个 rho accuracy 最高”。

FINALIZE 报告必须把四个 budget 按下面逻辑展示。

### A. Budget geometry

对于每个 grouping / rho：

```text
K_G
group ratio
selected scalar count
realized scalar ratio
```

重点展示：

同一个 nominal rho_G 在 out_channel 与 filter_connection 下，
实际 selected scalar footprint 是否明显不同。

### B. Accuracy sanity

报告：

```text
PU
FO
```

以及相对 `conv_module_dense` 的差值。

Accuracy 只用于发现：

- 明显 collapse；
- 明显异常；
- 极端 budget 是否完全不可用。

不要按最高 accuracy 自动挑 rho。

### C. Efficiency sanity

报告：

```text
runtime
peak GPU memory
```

用于发现某种 grouping / baseline 是否存在明显工程异常。

---

## 9. 不要自动做的事情

FINALIZE 后不要：

- 不自动删除某个 rho；
- 不自动冻结 3 个 formal budgets；
- 不启动 VisDA；
- 不启动 Office 6-transfer full baseline；
- 不启动 LBI；
- 不扩展到 .01 / .0002 等新 budget；
- 不根据 accuracy 做额外搜索。

只给出完整 pilot report，让我们人工决定下一阶段 formal budget。

---

## 10. Search root / output

使用独立 pilot root，例如：

`experiment_logs/office_conv_baseline_budget_pilot_seed2026_20260826`

如果 repository 已有统一的 Conv experiment log naming convention，
可按现有 convention 调整目录名，但必须：

- 与 FC logs 分开；
- 与未来 formal Conv baseline logs 分开；
- 明确标记 pilot；
- 明确 seed=2026。

Phase records：

`experiment_logs/office_conv_baseline_budget_pilot_seed2026_20260826/phase_records/p3a_office_da/`

---

## 11. MODE=PREPARE

PREPARE 只准备，不启动训练。

1. 阅读 frozen Conv protocol 与当前 Office experiment planning / launcher 工具。
2. 验证 frozen Conv protocol / implementation revision。
3. 验证 D→A transfer identifier。
4. 创建 pilot matrix。
5. 创建 plan。
6. 验证 matrix 只有上述 25 个 scientific conditions。
7. 验证四个 budget 对应的 K_G：
   - out: 4 / 9 / 18 / 46
   - filter: 3276 / 6553 / 13107 / 32768
8. 验证没有任何 `conv_*_lbi`。
9. 验证没有 VisDA。
10. 验证没有 Office 其他 transfer。
11. 创建 non-interactive launcher：

`tools/run_office_conv_baseline_budget_pilot_noninteractive.sh`

Launcher 要求：

```text
foreground
non-interactive

GPUs: 0-7
max-workers: 8
workers-per-gpu: 1

--resume
--resume-partial-runs
```

每张 GPU 同时最多运行 1 个 task。

不要 GPU probing。
不要自行启动 launcher。

12. 做静态检查：
   - plan uniqueness；
   - scientific identity uniqueness；
   - expected condition count；
   - bash -n；
   - git diff --check。
13. 写：

`${SEARCH_ROOT}/phase_records/p3a_office_da/PREPARE.md`

记录：

- exact matrix；
- exact condition count；
- random child-mask semantics；
- K_G；
- output paths；
- launcher path；
- static checks。

14. 最后只打印给用户真正需要在非交互式窗口执行的命令。

建议最终输出形式：

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
conda activate SHOT_TTA
bash tools/run_office_conv_baseline_budget_pilot_noninteractive.sh
```

然后停止。

不要自行运行。

---

## 12. MODE=FINALIZE

FINALIZE 只汇总，不 launch / retry / rerun。

首先验证：

- 所有 expected conditions 是否完成；
- no error / NaN；
- sparse baseline `selected_group_count == K_G`；
- requested budget / grouping / candidate scope 与 plan 一致；
- Random 3 child masks 完整。

若缺 run：
明确列出缺失项。
不要自动 retry。

然后生成：

### 1. 完整 run table

包含：

- variant；
- grouping；
- rho；
- K_G；
- selected groups；
- selected scalars；
- realized scalar ratio；
- PU；
- FO；
- runtime；
- GPU memory。

### 2. Budget geometry table

按：

```text
group_mode × rho
```

汇总 scalar footprint。

Random 报 mean/std；
Magnitude 报 deterministic value；
Saliency 报当前 artifact 能支持的 summary。

### 3. Accuracy sanity table

按 grouping / baseline / rho 报：

```text
FO
delta vs conv_module_dense
```

不要只给 winner。

### 4. Efficiency table

报告：

- online/adapt runtime；
- peak allocated；
- peak reserved。

### 5. Recommendation section

只讨论：

- 哪些 rho 在 integer granularity 上明显过于极端；
- 哪些 rho 的 realized scalar footprint 能形成合理低/中/高层级；
- 是否存在明显 collapse / engineering anomaly；
- out_channel 与 filter_connection 是否适合共享相同 nominal rho grid。

不要自动 freeze。

最后明确写：

> P3a does not determine the formal Conv budget by accuracy alone. Formal rho selection remains pending manual review.

写：

`${SEARCH_ROOT}/phase_records/p3a_office_da/FINALIZE.md`

然后停止。

---

## 13. 最终限制

无论 PREPARE 还是 FINALIZE：

- 不要修改算法。
- 不要修改 frozen protocol。
- 不要修改 FC。
- 不要启动 Conv-LBI。
- 不要启动 VisDA。
- 不要扩展 budget。
- 不要自行进入下一阶段。
