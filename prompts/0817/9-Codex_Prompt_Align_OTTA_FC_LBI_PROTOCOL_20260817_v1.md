# Codex Prompt：严格对齐 `OTTA_FC_LBI_PROTOCOL_20260817_v1`

只允许修改目录：

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/
```

其他目录全部只读，不得修改。

本次任务不是重新设计实验协议，而是把现有实现、config、launcher、identity、artifact 和 tests 严格对齐以下唯一 normative protocol：

```text
260817_iclr2027-refined/protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md
```

请先完整阅读该文档，再开始修改。

如果代码、旧 YAML、README 或历史默认值与该 protocol 冲突：

> 以 protocol MD 为唯一准则。

不要自行补充新的 scientific setting，不要重新解释 protocol，不要修改 protocol MD 本身。

当前 scientific implementation revision 必须保持：

```text
IMPLEMENTATION_REVISION =
"iclr2027_refined_20260817_v1"
```

当前 efficiency protocol revision 必须保持：

```text
EFFICIENCY_PROTOCOL_REVISION =
"otta_fc_batch_efficiency_20260817_v1"
```

不要运行任何真实 Office / VisDA 实验。

---

## 1. 建立唯一 formal config / experiment entry

根据 protocol 新建清晰的正式配置体系，例如：

```text
configs/otta_fc_lbi_protocol_20260817_v1.yaml
experiments/shot_otta_fc_lbi_formal_20260817_v1.yaml
```

命名可以根据当前仓库结构微调，但必须做到：

- 后续 formal SHOT-OTTA + FC-LBI 实验只有一个明确入口；
- 不再依赖旧 SHOT configs；
- 所有 frozen setting 来自 protocol MD；
- 不复制多套彼此可能漂移的默认值。

正式 protocol 必须落实：

### Office

```text
ResNet-50
BS=64
workers=4
seed=2026
6 transfers
```

### VisDA

```text
ResNet-101
BS=256
workers=4
seed=2026
S->R
```

两者统一：

```text
one-pass target stream
drop_last=false
current frozen transforms
formal seed ONLY 2026
```

---

## 2. SHOT-OTTA setting 严格对齐 protocol

统一：

```text
cls_par = 0.3
ent_par = 1.0
threshold = 0.0
```

components：

```text
ent
div
pseudo
```

保持当前 causal current-batch pseudo-label 实现。

Office：

```text
lr = 0.01
```

VisDA：

```text
lr = 0.001
```

其余冻结：

```text
lr_decay1 = 0.1
lr_decay2 = 1.0
momentum = 0.9
weight_decay = 0.001
nesterov = true
lr_gamma = 10.0
lr_power = 0.75
```

不要修改 SHOT loss、pseudo-label、optimizer 或 scheduler implementation。

---

## 3. Source checkpoint protocol

所有 formal variant 必须使用：

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/nips2026/SHOT-OTTA/ckpt/source/uda
```

中的 source checkpoint。

注意当前 resolver 如果采用：

```text
root / da / dataset / source
```

则 config root 应为：

```text
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/nips2026/SHOT-OTTA/ckpt/source
```

并设置：

```text
da = uda
```

不要产生：

```text
.../source/uda/uda/...
```

新增唯一 source checkpoint revision：

```text
SOURCE_CHECKPOINT_REVISION =
"nips2026_shot_otta_uda_source_v1"
```

或等价唯一常量。

要求：

```text
source_checkpoint_revision
```

必须进入 `scientific_config` 并参与 `experiment_config_sha256`。

修改 source checkpoint revision 必须改变 scientific hash。

同时 manifest / result metadata 中记录实际解析后的：

```text
source_F
source_B
source_C
```

路径。

如果容易实现可附加 SHA256，但不要为了 checkpoint hashing 大规模重构。

---

## 4. Formal seed 与 Random

所有 top-level formal run：

```text
seed = 2026
```

删除 formal seed=2020/2021/2022 的旧设置。

Random 仍然：

```text
3 independent masks
```

它们不是 3 个 formal seeds。

保持 deterministic mask-seed derivation，只要求：

- 基于 formal seed=2026；
- 三个 seed 不同；
- 可复现；
- artifact 中可追踪；
- 不修改现有 Random scientific behavior。

---

## 5. FC controlled family

必须严格保持 protocol：

candidate only：

```text
netB.bottleneck.weight
netB.bottleneck.bias
```

```text
N_FC = 524544
```

controlled variants：

```text
module_dense
module_random
module_magnitude
module_saliency
module_lbi
```

全部：

- backbone frozen；
- classifier frozen；
- BN affine frozen；
- BN running stats frozen。

不要修改现有算法代码。

Sparse budgets：

```text
rho =
0.0005
0.001
0.002
```

$$
K = \lfloor \rho \times 524544 \rfloor
$$

即：

```text
0.0005 -> 262
0.001  -> 524
0.002  -> 1049
```

禁止重新引入：

- ceil；
- per-tensor budget；
- weight/bias 独立 budget；
- minimum-one-per-tensor；
- hidden slack。

---

## 6. LBI protocol

保持当前 refined scientific behavior，不修改算法。

固定：

```text
support_threshold = 1e-4
stage1_max_steps = 3000
stage2_steps = 1
```

保留：

- corrected old-state Eq.(5)；
- strict integer global budget；
- strict rollback；
- masked-delta initialization；
- Stage-2 mask-out value freezing；
- Stage-2 fixed LR；
- momentum reset semantics；
- per-batch local LBI restart；
- BN frozen；
- omega persistent accumulation。

只允许未来 tuning：

```text
alpha
kappa
nu
omega
stage2_lr
```

tuning granularity：

```text
(dataset, OTTA method, budget)
```

当前 SHOT 应有 6 个最终 tuple：

```text
Office × 0.0005
Office × 0.001
Office × 0.002

VisDA × 0.0005
VisDA × 0.001
VisDA × 0.002
```

这些数值目前未确定。

不要使用旧默认值偷偷执行 formal LBI。

要求：

- planner/dry-run 可以生成或展示对应 formal identity；
- 但真正 launch `module_lbi` 时，如果该 dataset×budget 的 frozen tuned tuple 尚未提供，必须 fail closed / 明确报 unresolved；
- 不允许 fallback 到旧 config；
- Office 同一 budget 的一组 LBI 参数必须用于全部 6 transfers。

---

## 7. Native SHOT 与 FC-controlled 不能混淆

```text
full_dense = native SHOT reference
```

保持当前：

- netF/netB adaptation；
- netC frozen；
- native BN behavior。

`module_dense/random/magnitude/saliency/lbi`：

保持 controlled FC-only + BN frozen。

不要为了“统一”而把 native SHOT BN 也 freeze。

---

## 8. PU / FO protocol

严格保持 protocol：

PU：

```text
current batch
-> adaptation
-> same-batch post-update PU
-> evaluation must not mutate persistent state
```

Native SHOT：

保留现有 BN snapshot/restore PU semantics。

Controlled FC：

BN 已 frozen。

FO：

```text
target stream 完成后
-> stop adaptation
-> frozen final state
-> eval mode
-> full target evaluation
-> no additional adaptation
```

Office：

```text
overall sample accuracy
```

VisDA：

```text
fixed-12-class mean per-class accuracy
```

不要修改 metric semantics。

---

## 9. Checkpoint / resume

正式 baseline：

```text
source_only
full_dense
module_dense
module_random
module_magnitude
module_saliency
```

全部：

```text
save_model = false
stream_checkpoint = false
partial_resume = false
```

不要保存 adapted target model。

只有：

```text
module_lbi
```

允许 completed-online-batch-boundary stream checkpoint/resume。

LBI：

```text
save_model = false
```

stream checkpoint 只用于 exact engineering resume。

要求 Office 和 VisDA 的 `module_lbi` 都支持该 batch-boundary resume。

检查 launcher，特别是 mixed plan。

即使用户运行：

```text
--resume-partial-runs
```

也只能给：

```text
variant == module_lbi
```

添加 stream checkpoint / resume flags。

不能给 baseline 传：

```text
--enable-stream-checkpoint
```

示意逻辑：

```text
checkpoint_eligible = (variant == "module_lbi")
```

同一 mixed plan 中：

```text
full_dense
module_random
module_lbi
```

必须能够同时工作：

- full_dense 正常从头跑；
- random 正常从头跑；
- lbi 支持 resume。

---

## 10. Efficiency protocol

现有 per-batch runtime/GPU implementation 已经完成。

不要重新设计，只检查其与 protocol MD 完全一致。

每 batch 保留：

```text
batch_index
batch_size
adapt_runtime_sec
pu_runtime_sec
online_runtime_sec
peak_gpu_memory_allocated_*
peak_gpu_memory_reserved_*
```

LBI 额外：

```text
lbi_stage1_runtime_sec
lbi_stage2_runtime_sec
```

run aggregate：

```text
mean
std
median
P95
total
```

GPU：

```text
batch peak mean
batch peak max
```

主 GPU metric：

```text
gpu_peak_allocated_max_mb
```

正式 efficiency 条件：

```text
1 experiment / GPU
same GPU model
same precision
same PyTorch/CUDA
same BS
same workers
```

不要加入 NVML / GPU utilization polling。

---

## 11. Random efficiency aggregation

检查并确保：

Random accuracy：

```text
3 masks mean
```

Random online efficiency：

```text
单 mask efficiency 的 mean
```

不能使用三个 masks runtime 的 sum 作为方法单实例 runtime。

FO 同理：

```text
fo_eval_runtime_sec
    = mean(child fo runtime)

fo_eval_runtime_mask_std_sec
    = std(child fo runtime)

random_total_fo_eval_runtime_sec
    = sum(child fo runtime)
```

online operational total 单独保留：

```text
random_total_online_compute_runtime_sec
```

Random GPU peak：

```text
所有 child masks / batches 的 maximum
```

---

## 12. Efficiency revision 唯一 source of truth

```text
EFFICIENCY_PROTOCOL_REVISION =
"otta_fc_batch_efficiency_20260817_v1"
```

只能有一个定义。

其他模块 import 它。

它写入：

- manifest；
- summary/results；
- aggregate/finalize。

但：

> 不得进入 scientific `experiment_config_sha256`。

---

## 13. 删除旧 configs

用户明确要求删除旧 configs，不保留 fallback。

请先搜索所有引用，再删除所有已经被本 protocol 替代的旧 SHOT formal/exploratory configs。

重点检查：

```text
configs/shot_otta.yaml
configs/shot_otta_visda.yaml

experiments/shot_otta_office_fixed_eval.yaml
experiments/shot_otta_baselines.yaml
```

以及其他包含：

- seed 2020/2021/2022；
- 旧 `save_model=true` policy；
- 旧 checkpoint policy；
- 旧 protocol formal matrix；
- 旧 source checkpoint setting；

的 SHOT formal YAML。

要求：

- 删除前先 `rg` 全部引用；
- tools/tests/README 全部迁移到新 formal config；
- 不留 dangling path；
- 不保留旧 config fallback；
- 不删除任何历史结果、checkpoint、source model 或用户数据；
- `nips2026/` 完全只读。

---

## 14. Formal experiment matrix

Formal top-level variants：

```text
source_only
full_dense
module_dense
module_random
module_magnitude
module_saliency
module_lbi
```

非 budget variants：

```text
source_only
full_dense
module_dense
```

每 transfer 各跑一次。

Sparse variants：

```text
module_random
module_magnitude
module_saliency
module_lbi
```

分别跑：

```text
rho = 0.0005 / 0.001 / 0.002
```

Random 每个 top-level identity 内部仍然 3 masks。

Office：

每 transfer：

```text
3 non-budgeted
+
4 × 3 sparse
=
15 identities
```

6 transfers：

```text
90 top-level identities
```

VisDA：

```text
15 top-level identities
```

总计：

```text
105 top-level formal experiment identities
```

dry-run 必须验证这个数量。

建议同时验证全局 variant 数：

```text
source_only:       7
full_dense:        7
module_dense:      7
module_random:     21
module_magnitude:  21
module_saliency:   21
module_lbi:        21
```

总计：

```text
105
```

注意：

LBI tuned tuple 未解决时可以出现在 plan/dry-run 中，但正式 execute 必须阻止。

---

## 15. Protocol provenance

formal plan / manifest / result 中应明确记录：

```text
protocol_revision:
OTTA_FC_LBI_PROTOCOL_20260817_v1

implementation_revision:
iclr2027_refined_20260817_v1

efficiency_protocol_revision:
otta_fc_batch_efficiency_20260817_v1

source_checkpoint_revision:
nips2026_shot_otta_uda_source_v1
```

其中：

```text
implementation_revision
source_checkpoint_revision
以及实际 scientific configuration
```

进入 scientific identity。

```text
efficiency_protocol_revision
```

不进入 scientific hash。

---

## 16. README

更新 README。

README 不再重复维护大段可能漂移的 protocol 数值。

README 顶部明确指向：

```text
protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md
```

并声明：

> 该 MD 是当前 SHOT-OTTA + FC-layer LBI formal experiments 的唯一 normative protocol。

README 只保留必要使用说明与入口。

---

## 17. Tests

增加/修改轻量 tests，至少验证：

### A. protocol config

- formal seed=2026；
- Office BS64；
- VisDA BS256；
- workers=4；
- cls_par=.3；
- ent_par=1；
- threshold=0；
- Office LR=.01；
- VisDA LR=.001；
- tau=1e-4；
- stage1_max_steps=3000；
- stage2_steps=1；
- budgets 正确。

### B. source checkpoint

- resolver 正确得到 `/source/uda/...`；
- 不出现 `/uda/uda/`；
- `source_checkpoint_revision` 进入 scientific config；
- 修改 revision -> scientific SHA 改变。

### C. efficiency identity

修改 efficiency protocol revision：

```text
不改变 scientific SHA
```

### D. checkpoint policy

mixed plan：

```text
full_dense
module_random
module_lbi
```

`resume_partial_runs=true`

验证：

- baseline 不收到 checkpoint flag；
- LBI 收到；
- Office LBI resume 工作；
- VisDA LBI resume 工作；
- 所有 variant `save_model=false`。

### E. Random FO runtime

child：

```text
[1,2,3]
```

验证：

```text
mean=2
total=6
std 正确
```

### F. formal matrix

planner dry-run：

```text
Office=90
VisDA=15
total=105
```

variant totals：

```text
7 / 7 / 7 / 21 / 21 / 21 / 21
```

### G. unresolved LBI

formal execute 若缺 tuned tuple：

- 必须明确失败；
- 不能使用默认值。

### H. old config cleanup

不能再有代码/test/README 引用已删除旧 YAML。

### I. scientific regression

现有所有：

- LBI；
- SHOT；
- budget；
- Random；
- runtime；
- stream checkpoint；
- launcher；
- multi-GPU；
- experiment engineering；
- Office；
- VisDA；
- summary/finalize；

轻量 tests 继续通过。

运行：

```text
compileall
git diff --check
planner dry-run
```

不要运行任何真实 Office / VisDA 实验。

---

## 18. 最终汇报

完成后只需清晰汇报：

1. 删除了哪些旧 configs；
2. 新 formal config/spec 路径；
3. protocol MD 如何成为唯一 normative source；
4. formal seed=2026 如何落实；
5. source checkpoint revision 如何进入 scientific identity；
6. baseline/LBI checkpoint policy；
7. launcher mixed-plan resume 如何处理；
8. Random runtime/FO aggregation语义；
9. efficiency revision唯一 source；
10. formal planner dry-run 的 105 identities 分布；
11. unresolved LBI 如何阻止正式 launch；
12. 跑了哪些 tests 及结果；
13. 明确确认：
   - 只修改了 `260817_iclr2027-refined/`；
   - 未修改 protocol MD；
   - 未修改任何 scientific algorithm behavior；
   - 未修改任何 protocol 已冻结的数值 setting；
   - 未运行真实 Office / VisDA 实验。

不要顺手做任何与 protocol 对齐无关的重构。
