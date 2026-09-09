# P3a — Office-31 Conv Baseline Budget Pilot（4-GPU Sequential Waves）

Project:
`/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined`

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
- PREPARE 阶段不要启动真实训练。
- 不做 GPU probing。

允许创建/修改的只有：
- experiment matrix / plan；
- launcher；
- pilot report；
- phase record；
- 必要的 pilot-specific metadata。

---

## 2. Scientific condition

Dataset:
Office-31

Transfer:
DSLR → Amazon（D→A）

先检查 repository 当前 Office planner / config 的真实 transfer identifier，
使用项目已有命名，不要自己猜 `D2A` / `d2a` 的字符串格式。

Formal seed:
`2026`

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

理论 integer group budget：

### out_channel

total_group_count = 9216

```text
rho=.0005 -> K_G=4
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

PREPARE 必须检查 planner/config 实际计算出的 K_G 与上述值完全一致。

不一致则停止，不生成 launcher。

---

## 4. 本轮精确运行条件

总 scientific conditions：

```text
1 conv_module_dense
+ 12 out_channel sparse conditions
+ 12 filter_connection sparse conditions
= 25 scientific conditions
```

具体：

### Dense anchor

```text
conv_module_dense
```

只跑一次，不带 sparse budget。

### out_channel

```text
conv_out_random
conv_out_magnitude
conv_out_saliency
```

每个 variant 跑：

```text
rho=.0005
rho=.001
rho=.002
rho=.005
```

### filter_connection

```text
conv_filter_random
conv_filter_magnitude
conv_filter_saliency
```

每个 variant 同样跑：

```text
rho=.0005
rho=.001
rho=.002
rho=.005
```

禁止出现任何：

```text
conv_out_lbi
conv_filter_lbi
```

---

## 5. 4-GPU sequential-wave 调度

本轮只使用：

```text
GPU 0
GPU 1
GPU 2
GPU 3
```

并发限制：

```text
max-workers=4
workers-per-gpu=1
```

禁止一张 GPU 同时运行两个 task。

必须严格按以下 wave 顺序调度，后续 wave 不允许抢跑：

### Wave 0

```text
conv_module_dense
```

只有 1 个任务。

Wave 0 完成后才进入 Wave 1。

### Wave 1

```text
conv_out_random
```

4 个 budget：

```text
.0005
.001
.002
.005
```

四个 budget 并行分配到 GPU 0-3。

如果 Random parent condition 在当前 framework 内部展开为 3 个 deterministic child masks，
仍由 launcher 在当前 wave 内按：

```text
max-workers=4
workers-per-gpu=1
```

排队完成。

必须等 Wave 1 的所有 Random child runs / parent aggregation 完整结束，
才能进入 Wave 2。

### Wave 2

```text
conv_out_magnitude
```

4 个 budget 并行占 GPU 0-3。

全部完成后进入 Wave 3。

### Wave 3

```text
conv_out_saliency
```

4 个 budget 并行占 GPU 0-3。

全部完成后进入 Wave 4。

### Wave 4

```text
conv_filter_random
```

4 个 budget并行。

Random child-mask 语义同上。

全部完成后进入 Wave 5。

### Wave 5

```text
conv_filter_magnitude
```

4 个 budget 并行。

全部完成后进入 Wave 6。

### Wave 6

```text
conv_filter_saliency
```

4 个 budget 并行。

这是最后一个 wave。

要求：

> 每个 wave 完整结束后才进入下一 wave。不得把不同 baseline family 混在同一个并行池里。

---

## 6. Random semantics

不要手工创建新的 random seeds。

严格复用 frozen semantics：

```text
formal seed = 2026
3 deterministic child masks
```

Random parent accuracy 使用当前 framework 已冻结的 3-mask aggregation。

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

## 7. Sparse baseline semantics

Random：

```text
global group pool uniform exact-K_G
```

Magnitude：

```text
score_g = ||W_g||_2
global top-K_G
```

Saliency：

```text
score_g = ||W_g * grad_g||_2
current online batch dynamic global top-K_G
```

两种 grouping 都禁止：
- per-layer budget；
- per-layer top-K；
- minimum-one-per-layer；
- ceil；
- tolerance slack。

对于所有 sparse baselines：

```text
selected_group_count == K_G
```

必须严格成立。

若不成立，视为 invalid，不要静默接受。

---

## 8. Pilot 主要记录字段

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

Saliency 如果当前 artifact 已经提供 per-batch scalar footprint，
同时报告 mean/min/max。

不要为了增加统计字段修改算法代码。

---

## 9. P3a 解释规则

这一步不是“哪个 rho accuracy 最高就选哪个”。

FINALIZE 必须分别看：

### A. Budget geometry

对每个：

```text
group_mode × rho
```

报告：

```text
K_G
requested group ratio
selected group count
selected scalar count
realized scalar ratio
```

重点比较：

> 同一个 nominal rho_G 在 out_channel 与 filter_connection 下，对应的实际 scalar adaptation footprint 是否明显不同。

### B. Accuracy sanity

报告：

```text
PU
FO
delta vs conv_module_dense
```

Accuracy 只用来识别：
- 明显 collapse；
- 明显异常；
- 极端 budget 是否完全不可用。

禁止按最高 accuracy 自动冻结 rho。

### C. Efficiency sanity

报告：

```text
adapt runtime
online runtime
peak allocated memory
peak reserved memory
```

检查 grouping / baseline 是否存在明显工程异常。

---

## 10. Search root / output

使用独立 pilot root：

```text
experiment_logs/office_conv_baseline_budget_pilot_seed2026_20260827
```

Phase records：

```text
experiment_logs/office_conv_baseline_budget_pilot_seed2026_20260827/phase_records/p3a_office_da/
```

必须与：
- FC logs；
- future formal Conv baseline logs；
- future Conv-LBI logs

分离。

---

# MODE=PREPARE

PREPARE 只准备，不启动真实训练。

1. 阅读 frozen Conv protocol。
2. 阅读当前 Office experiment planner / launcher。
3. 验证：
   ```text
   protocol_revision = OTTA_CONV_LBI_PROTOCOL_20260826_v1
   implementation_revision = iclr2027_refined_conv_20260826_v1
   ```
4. 验证 Office D→A 的 repository 内正式 transfer identifier。
5. 创建 pilot matrix。
6. 创建 plan。
7. 验证 scientific conditions 精确为 25 个。
8. 验证 budget 只有：
   ```text
   .0005
   .001
   .002
   .005
   ```
9. 验证 K_G：
   ```text
   out_channel:
   4 / 9 / 18 / 46

   filter_connection:
   3276 / 6553 / 13107 / 32768
   ```
10. 验证没有任何 Conv-LBI。
11. 验证没有 VisDA。
12. 验证没有 Office 其他 transfer。
13. 验证 Random 继续使用 formal seed=2026 + 3 deterministic child masks。
14. 创建 launcher：

```text
tools/run_office_conv_baseline_budget_pilot_noninteractive.sh
```

Launcher 必须：

```text
foreground
non-interactive

GPUs=0,1,2,3
max-workers=4
workers-per-gpu=1

--resume
--resume-partial-runs
```

并严格实现 Wave 0 → Wave 6 顺序。

---

## Launcher 的 Conda 要求

**不要要求用户手动 `conda activate SHOT_TTA`。**

`.sh` 内部所有 Python / training / planning 命令统一通过：

```bash
conda run --no-capture-output -n SHOT_TTA ...
```

执行。

例如：

```bash
conda run --no-capture-output -n SHOT_TTA python train.py ...
```

或者调用 repository runner 时，也必须由 `.sh` 自己负责进入 `SHOT_TTA` 环境。

目标：

> 用户在非交互式窗口只需要 `cd` + `bash` 两步，不需要提前 activate conda environment。

不要在 launcher 中写：

```bash
source activate SHOT_TTA
conda activate SHOT_TTA
```

只使用：

```bash
conda run --no-capture-output -n SHOT_TTA
```

---

## PREPARE 静态检查

只做：
- matrix exact-count check；
- plan uniqueness；
- scientific identity uniqueness；
- wave membership check；
- wave order check；
- GPU assignment / concurrency check；
- `bash -n`；
- `git diff --check`。

不要启动训练。
不要 GPU probing。

写：

```text
${SEARCH_ROOT}/phase_records/p3a_office_da/PREPARE.md
```

记录：
- exact matrix；
- scientific condition count；
- actual expanded run count（如果 Random 展开 child masks）；
- wave 0-6 明细；
- K_G；
- Random child-mask semantics；
- output root；
- launcher path；
- static checks。

---

## PREPARE 最终输出格式

完成后，Codex **最后最好只输出下面两行命令，不要再输出解释文字**：

```bash
cd /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined
bash tools/run_office_conv_baseline_budget_pilot_noninteractive.sh
```

特别注意：

**不要输出：**

```bash
conda activate SHOT_TTA
```

因为 launcher 自己已经使用：

```bash
conda run --no-capture-output -n SHOT_TTA
```

---

# MODE=FINALIZE

FINALIZE 只汇总。

禁止：
- launch；
- retry；
- rerun；
- 补跑；
- 自动扩展 rho；
- 自动启动下一阶段。

首先验证：
- expected scientific conditions 是否完整；
- expanded Random child runs 是否完整；
- no error / NaN；
- `selected_group_count == K_G`；
- grouping / budget / candidate scope 与 plan 完全一致。

若缺 run：

> 列出来，然后停止；不要自动 retry。

---

## FINALIZE 输出

### 1. Complete run table

包含：

```text
variant
group_mode
rho
K_G
selected groups
selected scalars
realized scalar ratio
PU
FO
runtime
GPU memory
```

### 2. Budget geometry

按：

```text
group_mode × rho
```

汇总。

Random：
mean/std/min/max。

Magnitude：
deterministic value。

Saliency：
按现有 artifacts 汇总可获得统计。

### 3. Accuracy sanity

每个 grouping / baseline / rho：

```text
FO
delta vs conv_module_dense
```

不要只展示 winner。

### 4. Efficiency

报告：

```text
adapt runtime
online runtime
peak allocated
peak reserved
```

### 5. Recommendation

只分析：
- `.0005` 对 out-channel 的 K=4 是否过于离散；
- 四个 rho 是否能形成合理的低/中/高 sparse regimes；
- realized scalar ratio 是否有清楚层级；
- out_channel 与 filter_connection 是否适合共享 nominal rho grid；
- 是否存在明显 collapse；
- 是否存在明显 efficiency anomaly。

不要自动 freeze formal rho。

最后明确写：

> P3a does not select formal Conv budgets by accuracy alone. Formal rho selection remains pending manual review.

写：

```text
${SEARCH_ROOT}/phase_records/p3a_office_da/FINALIZE.md
```

然后停止。

---

## 最终限制

无论 PREPARE / FINALIZE：

- 不修改算法；
- 不修改 frozen protocol；
- 不修改 FC；
- 不启动 Conv-LBI；
- 不运行 VisDA；
- 不运行 Office 其他 transfers；
- 不扩展 rho grid；
- 不搜参；
- 不自行进入下一阶段。
