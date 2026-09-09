请只修改目录：

`260817_iclr2027-refined/`

本次仅 refinement **runtime / GPU-memory instrumentation 与 aggregation**。

**禁止修改任何 scientific behavior**，包括：
- SHOT-OTTA loss / pseudo-label；
- model update；
- optimizer semantics；
- BN policy；
- Office / VisDA metric；
- FC candidate；
- sparse budget；
- Random/Magnitude/Saliency selection；
- LBI Eq.(5)、rollback、support threshold、masked initialization、Stage-2、omega；
- data stream / augmentation；
- seed；
- batch size；
- source checkpoint；
- 任何实验数值超参数。

不要运行任何真实 Office / VisDA 实验。

当前 scientific implementation revision：

```text
iclr2027_refined_20260817_v1
```

**不要 bump scientific implementation revision**，因为本次只改 efficiency measurement，不改变模型或结果语义。

---

# 1. Efficiency measurement unit 改为 online batch

OTTA 的 efficiency 最小测量单元统一为：

$$
B_1,B_2,\ldots,B_T.
$$

对于每个真正被处理的 online batch，记录一条完整 efficiency record。

至少包含：

```text
batch_index
batch_size

adapt_runtime_sec
pu_runtime_sec
online_runtime_sec

peak_gpu_memory_allocated_bytes
peak_gpu_memory_reserved_bytes
peak_gpu_memory_allocated_mb
peak_gpu_memory_reserved_mb
```

其中：

$$
\boxed{
T_t^{online}
=
T_t^{adapt}
+
T_t^{PU}
}
$$

建议 `online_runtime_sec` 直接由前两项相加得到，避免再引入第三套略有差异的 timer boundary。

这些字段应进入现有 per-batch `metrics.jsonl` 或等价 batch-level artifact，而不是只在 run 结束时保存 aggregate。

---

# 2. 每个 batch 的 timer boundary

统一定义：

## 2.1 Timer 开始之前

以下不计入正式 batch compute runtime：

- DataLoader 等待；
- 从 DataLoader 取 batch；
- CPU preprocessing；
- input/label 搬到 GPU；
- checkpoint load/save；
- JSONL / summary serialization；
- 其他 artifact I/O。

也就是说：

> input 已经在目标 device 上、当前 batch 正式准备开始 adaptation 后，再开始计时。

---

## 2.2 Adaptation timer

GPU 环境：

```python
torch.cuda.synchronize(device)
adapt_start = time.perf_counter()

# adaptation

torch.cuda.synchronize(device)
adapt_end = time.perf_counter()
```

记录：

```text
adapt_runtime_sec
```

这段只包括当前 method 真正的 online adaptation compute。

---

## 2.3 PU timer

adaptation 完成后：

```python
torch.cuda.synchronize(device)
pu_start = time.perf_counter()

# post-update PU inference

torch.cuda.synchronize(device)
pu_end = time.perf_counter()
```

记录：

```text
pu_runtime_sec
```

PU 原有 read-only scientific semantics保持不变。

然后：

```text
online_runtime_sec
    = adapt_runtime_sec + pu_runtime_sec
```

CPU fallback 也必须正常工作，不调用 CUDA-only API。

---

# 3. 每个 batch 单独记录 GPU peak memory

对于 CUDA run，在：

> inputs 已在 GPU，当前 batch adaptation 即将开始

时执行：

```python
torch.cuda.reset_peak_memory_stats(device)
```

然后完成：

```text
adaptation + PU
```

之后读取：

```python
torch.cuda.max_memory_allocated(device)
torch.cuda.max_memory_reserved(device)
```

记录为该 batch：

```text
peak_gpu_memory_allocated_bytes
peak_gpu_memory_reserved_bytes
peak_gpu_memory_allocated_mb
peak_gpu_memory_reserved_mb
```

因此 runtime 与 GPU-memory 测量区间一致：

$$
\boxed{\text{adaptation + PU}}
$$

注意：

- model 本身已经加载在 GPU 上；
- current allocated model/input memory 会自然计入该 batch peak；
- 不要把 checkpoint I/O 纳入；
- 不要高频调用 NVML / `nvidia-smi`；
- 暂时不记录 GPU utilization percentage 或 power。

---

# 4. LBI 增加 Stage-1 / Stage-2 batch-level runtime diagnostics

只做计时，不改 LBI。

对于 `module_lbi`，每 batch额外记录：

```text
lbi_stage1_runtime_sec
lbi_stage2_runtime_sec
```

要求使用与上面相同的 CUDA synchronize 原则。

继续保留已有：

```text
stage1_steps
stage1_stop_reason
stage1_rollback_used
selected_param_count
selected_support_ratio
```

注意：

$$
lbi\_stage1\_runtime
+
lbi\_stage2\_runtime
$$

不强制必须与完整 `adapt_runtime_sec` bitwise 相等，因为 adaptation 可能还有少量 mask construction、omega accumulation 等外围开销。

但应满足：

```text
lbi_stage1_runtime_sec >= 0
lbi_stage2_runtime_sec >= 0
adapt_runtime_sec >= 0
```

不要为了计时重构 LBI 算法。

---

# 5. Run-level aggregate 必须由 batch records 计算

正式 efficiency summary 不再只依赖一个整 stream timer。

对于：

$$
t=1,\ldots,T
$$

的 `online_runtime_sec`，计算：

```text
online_batch_runtime_mean_sec
online_batch_runtime_std_sec
online_batch_runtime_median_sec
online_batch_runtime_p95_sec
online_compute_runtime_sec
```

其中：

$$
\boxed{
online\_compute\_runtime\_sec
=
\sum_{t=1}^{T}T_t^{online}
}
$$

保持 `online_compute_runtime_sec` 表示**完整 target stream 的总 online compute time**，不要把这个已有字段重新解释成 mean。

同理新增：

```text
adapt_batch_runtime_mean_sec
adapt_batch_runtime_std_sec
adapt_runtime_total_sec

pu_batch_runtime_mean_sec
pu_batch_runtime_std_sec
pu_runtime_total_sec
```

要求：

$$
adapt\_runtime\_total
=
\sum_t T_t^{adapt}
$$

$$
pu\_runtime\_total
=
\sum_t T_t^{PU}
$$

并近似满足：

$$
online\_compute\_runtime
=
adapt\_runtime\_total
+
pu\_runtime\_total.
$$

---

# 6. GPU memory run-level aggregation

对每 batch 的 peak memory：

$$
M_1,\ldots,M_T
$$

至少记录：

```text
gpu_peak_allocated_mean_mb
gpu_peak_allocated_max_mb

gpu_peak_reserved_mean_mb
gpu_peak_reserved_max_mb
```

正式论文/比较中主显存指标定义为：

$$
\boxed{
\max_t M_t^{allocated}
}
$$

即：

```text
gpu_peak_allocated_max_mb
```

因为它回答实际运行时的 maximum memory requirement。

mean 作为辅助 diagnostic。

如已有兼容字段：

```text
peak_gpu_memory_allocated_mb
peak_gpu_memory_reserved_mb
```

可以继续保留，并明确映射为整个 run 的 batch-wise maximum，不要产生两套互相矛盾的定义。

---

# 7. FO runtime 保持独立

继续保留：

```text
fo_eval_runtime_sec
```

定义仍然只是 final frozen-model FO evaluation。

它不进入：

```text
online_batch_runtime_*
online_compute_runtime_sec
```

---

# 8. wall runtime 保留 operational 语义

继续记录：

```text
wall_runtime_sec
```

它允许包含：

- setup；
- dataloading；
- artifact I/O；
- LBI checkpoint I/O；
- FO；
- final serialization。

不要拿 `wall_runtime_sec` 作为算法 compute efficiency 主指标。

正式 efficiency 主指标是：

```text
online_batch_runtime_mean_sec
online_compute_runtime_sec
gpu_peak_allocated_max_mb
```

---

# 9. Random 三 mask 的 efficiency aggregation

Random formal run 保持：

```text
experiment seed = 2026
3 independent random masks
```

不要修改 mask seed 或 accuracy aggregation。

每个 random child mask 必须独立保存完整 per-batch efficiency records，并先独立计算自己的：

```text
online_batch_runtime_mean_sec
online_compute_runtime_sec
gpu_peak_allocated_max_mb
...
```

Random 顶层正式 efficiency：

## Runtime

定义：

$$
\boxed{
T_{\mathrm{Random,batch}}
=
\frac{1}{3}
\sum_{m=1}^{3}
\overline T_m
}
$$

即：

```text
online_batch_runtime_mean_sec
    = mean(child online_batch_runtime_mean_sec)

online_batch_runtime_mask_std_sec
    = std(child online_batch_runtime_mean_sec)
```

完整 stream compute cost同样：

```text
online_compute_runtime_sec
    = mean(child online_compute_runtime_sec)

online_compute_runtime_mask_std_sec
    = std(child online_compute_runtime_sec)
```

用于和 Magnitude / Saliency / LBI 的单实例成本公平比较。

同时另外记录实际跑完三个 masks 的 operational compute：

```text
random_total_online_compute_runtime_sec
    = sum(child online_compute_runtime_sec)
```

不要把 3-mask sum 当成 Random 方法的单实例 efficiency。

## GPU

Random 顶层：

```text
gpu_peak_allocated_max_mb
    = max(all child mask batch peaks)

gpu_peak_reserved_max_mb
    = max(all child mask batch peaks)
```

各 child 的完整 metadata 继续保留。

---

# 10. LBI resume 必须正确恢复 per-batch efficiency history

LBI 保持现有：

> completed-online-batch-boundary resume

不修改科学语义。

由于现在 efficiency 是 per-batch measurement，resume 后应能够得到完整：

$$
B_1,\ldots,B_T
$$

的 efficiency records。

要求：

- 已完成 batch 的 efficiency records/aggregate state 随现有 stream checkpoint 恢复；
- resume 从 next batch 继续 append；
- 不重复计算之前 batch；
- 不丢失之前 batch；
- 最终 run-level mean/std/median/p95/total 基于**全部已完成 batch**；
- checkpoint save/load I/O 仍在 batch compute timer 之外。

如果 checkpoint 中保存完整 batch efficiency records最简单，可以直接保存；如果已有 accumulator 架构更合适，也可以同时保存必要 accumulator，但最终 artifact 必须仍能获得完整 per-batch raw records。

记录：

```text
runtime_resume_used: true/false
runtime_segment_count
```

### 重要

不要因为发生 resume 自动把 per-batch compute metrics 判无效。

只要每个 segment 都使用相同正式 runtime protocol：

> per-batch adaptation/PU runtime 与 peak GPU memory仍可用于 aggregation。

`wall_runtime_sec` 则继续只视为 operational diagnostic。

---

# 11. runtime_comparable

继续保留：

```text
runtime_comparable
```

但它的语义只表示：

> 本 run 被显式安排在 formal efficiency execution condition 下。

formal 条件至少是：

```text
1 experiment / GPU
```

launcher/config 显式确定。

不要尝试通过 `nvidia-smi` 自动猜服务器是否有竞争 workload。

即使：

```text
runtime_comparable = false
```

仍然照常记录所有 per-batch efficiency metrics，只是不进入正式 efficiency table。

---

# 12. GPU / environment metadata

继续/确保保存：

```text
gpu_name
gpu_device_index
gpu_total_memory_bytes

torch_version
cuda_version
runtime_comparable
runtime_resume_used
runtime_segment_count
```

正式跨 baseline runtime comparison 必须在相同：

- GPU model；
- precision；
- PyTorch/CUDA environment；
- batch size；
- worker setting；

下进行。

不要增加 GPU utilization polling。

---

# 13. Metadata propagation

新的 batch-level efficiency 字段应进入：

```text
metrics.jsonl
```

run-level aggregate 至少传播至：

```text
summary.json
results.json
status CSV / JSON
aggregate outputs
finalize outputs
```

不要把每个 batch 的几十/几百个 raw records硬塞进 status CSV；CSV 只保存 aggregate。

raw per-batch data留在 `metrics.jsonl` 或已有适合的 structured artifact 中。

---

# 14. Efficiency protocol revision

建议新增一个仅用于 metadata 的：

```python
EFFICIENCY_PROTOCOL_REVISION = "otta_fc_batch_efficiency_20260817_v1"
```

它用于标记 runtime/GPU measurement definition。

要求：

- 写入 summary / results / manifest / aggregate；
- **不参与 scientific experiment_config_sha256**；
- 不修改当前：

```text
IMPLEMENTATION_REVISION =
"iclr2027_refined_20260817_v1"
```

因为本次没有改变 scientific result semantics。

---

# 15. Tests

新增/更新轻量测试，至少覆盖：

### A. Per-batch record

3 个 fake online batches 后必须得到 3 条 efficiency records。

每条包含：

```text
adapt_runtime_sec
pu_runtime_sec
online_runtime_sec
batch_size
```

并满足：

```text
online_runtime_sec
≈ adapt_runtime_sec + pu_runtime_sec
```

---

### B. Aggregation

用人工 records：

```text
online = [1.0, 2.0, 3.0, 4.0]
```

验证：

```text
mean = 2.5
median = 2.5
total = 10.0
p95 正确
std 正确
```

不要依赖真实 sleep 来验证精确统计公式。

---

### C. GPU memory aggregation

人工 records：

```text
allocated = [100, 120, 110]
reserved  = [150, 170, 160]
```

验证：

```text
allocated mean = 110
allocated max  = 120

reserved mean = 160
reserved max  = 170
```

GPU unavailable 时测试必须正常 skip/fallback。

---

### D. Random

构造三个 child masks，其：

```text
batch means = [1.0, 1.2, 0.8]
stream totals = [10, 12, 8]
```

验证：

```text
Random formal batch mean = 1.0
Random formal stream total = 10
Random operational 3-mask total = 30
```

以及 mask std 正确。

GPU peak取 child/batch 全局 maximum。

---

### E. Resume

构造：

```text
segment 1: batch 0,1
checkpoint
resume
segment 2: batch 2,3
```

最终必须：

- raw efficiency records 正好 4 条；
- batch index 无重复；
- 无遗漏；
- aggregate 与一次 uninterrupted 0–3 batch 的人工 records 一致；
- `runtime_resume_used=true`；
- `runtime_segment_count=2`。

checkpoint mock I/O 不得进入任何 batch `adapt/PU/online` timer。

---

### F. LBI sub-timers

检查 LBI batch record包含：

```text
lbi_stage1_runtime_sec
lbi_stage2_runtime_sec
```

且非负。

不要验证具体耗时数值。

---

### G. Scientific regression

instrumentation 前后使用相同 fake seed/input，验证：

- final model parameters 相同；
- PU predictions 相同；
- FO result 相同；
- RNG progression 不因 instrumentation 改变；
- LBI resume scientific result 相同；
- Random masks 相同。

现有以下测试继续全部通过：

- LBI smoke；
- SHOT-OTTA smoke；
- runtime efficiency；
- stream checkpoint；
- budget diagnostics；
- random multimask；
- experiment engineering；
- launcher / multi-GPU；
- VisDA adapter / planner-summary；
- compileall；
- git diff --check。

不要运行真实 Office / VisDA 实验。

---

# 16. README

同步更新 README，明确 frozen efficiency definition：

### Per online batch

$$
T_t^{online}
=
T_t^{adapt}
+
T_t^{PU}
$$

记录：

- adaptation runtime；
- PU runtime；
- online runtime；
- peak allocated GPU memory；
- peak reserved GPU memory。

### Run aggregation

Runtime：

- mean；
- std；
- median；
- P95；
- total。

GPU：

- batch-peak mean；
- batch-peak max。

正式主效率指标：

```text
online_batch_runtime_mean_sec
online_compute_runtime_sec
gpu_peak_allocated_max_mb
```

Random 使用 3 masks 的 per-mask efficiency mean；3-mask总机器成本单独记录。

LBI resume继续有效，最终效率 aggregate 基于完整 per-batch records；checkpoint I/O 不进入 batch compute timer。

---

完成后汇报：

1. 修改了哪些文件；
2. per-batch timer 的准确开始/结束边界；
3. per-batch GPU peak memory 如何 reset/read；
4. LBI Stage-1/Stage-2 runtime 如何记录；
5. run-level mean/std/median/P95/total 如何从 raw batches 聚合；
6. Random 三 mask如何聚合；
7. resume 如何避免 efficiency records 重复/丢失；
8. metadata 如何传播；
9. 跑了哪些轻量测试及结果；
10. 明确确认没有修改任何 scientific behavior 或实验数值超参数。

不要顺手重构无关代码，不要运行真实实验。