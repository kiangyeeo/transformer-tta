只修改：

`260817_iclr2027-refined/`

本次只增加 efficiency instrumentation，不得修改任何 scientific behavior、模型更新、loss、metric、BN policy、LBI、budget、data stream、seed、batch size 或 optimizer semantics。

不要运行真实 Office / VisDA 实验。

目标是为后续统一 OTTA baseline comparison 增加公平、可复现的 runtime 与 GPU-memory 指标。

## 1. Online compute runtime

新增主效率指标：

```text
online_compute_runtime_sec
```

定义为整个 target stream 中所有 online step 的：

```text
adaptation compute
+
post-update PU inference
```

累计时间。

要求：

- GPU timing 前后正确调用 `torch.cuda.synchronize()`，避免 CUDA asynchronous execution 导致低估；
- 不包括：
  - source/model loading；
  - experiment initialization；
  - final FO evaluation；
  - artifact/summary serialization；
  - LBI stream checkpoint save/load I/O；
- 对 CPU fallback 保持可运行。

LBI checkpoint 写盘必须位于该 compute timer 之外。

不要因为 timing 改变任何 adaptation/evaluation state。

## 2. FO runtime

另外记录：

```text
fo_eval_runtime_sec
```

只统计最终冻结模型的 FO evaluation。

同样保证 CUDA synchronization 正确。

## 3. End-to-end wall runtime

保留/明确当前整体：

```text
wall_runtime_sec
```

该值允许包含：

- setup；
- dataloading；
- checkpoint I/O；
- summary/writeout；

作为 operational runtime。

不要用它替代 `online_compute_runtime_sec`。

## 4. Peak GPU memory

在 source model 已加载、run 即将正式开始 online stream 时：

```python
torch.cuda.reset_peak_memory_stats(device)
```

运行结束后记录：

```text
peak_gpu_memory_allocated_bytes
peak_gpu_memory_reserved_bytes
peak_gpu_memory_allocated_mb
peak_gpu_memory_reserved_mb
```

同时记录：

```text
gpu_name
gpu_device_index
gpu_total_memory_bytes
torch_version
cuda_version
```

不要加入会持续高频 polling、显著影响实验性能的 profiler。

暂时不把平均 GPU utilization / power 作为正式 scientific metric。

## 5. Checkpoint policy

保持：

```text
source_only/native/FC-dense/random/magnitude/saliency
    -> 不保存 stream/model checkpoint

module_lbi
    -> 保留现有 completed-online-batch partial resume
```

不要给 baseline 新增 checkpoint/resume。

不要修改现有 LBI resume scientific semantics。

## 6. Metadata / summary

将以下字段传播到：

```text
summary.json
results.json
status/summary CSV
aggregate/finalize output
```

至少包括：

```text
online_compute_runtime_sec
fo_eval_runtime_sec
wall_runtime_sec
peak_gpu_memory_allocated_mb
peak_gpu_memory_reserved_mb
gpu_name
runtime_comparable
```

`runtime_comparable` 默认只有在当前 run 明确满足“一 GPU 单实验”的 formal efficiency protocol 时才为 true。

不要根据 GPU utilization 猜测该字段。

如果 launcher 已知道 workers-per-GPU / GPU assignment，请由 launcher/config 显式传播。

## 7. Tests

使用 tiny CPU/GPU-safe fake model / stream，至少验证：

- online timer 能累计多 step；
- FO timer 独立；
- checkpoint callback / mock sleep 不进入 online compute runtime；
- wall runtime 可以包含该开销；
- GPU 可用时 peak memory 字段为非负有效值；
- CPU 环境不报错；
- instrumentation 不改变 final model state、PU/FO 数值或 RNG state；
- baseline 不产生 stream checkpoint；
- LBI 原有 resume tests 全部继续通过。

运行现有相关 smoke/unit tests、compileall、git diff --check。

不要运行任何真实 Office / VisDA 实验。

同步 README，明确 runtime measurement boundary 和 GPU-memory definition。

完成后汇报：
- 修改文件；
- timer boundary；
- checkpoint I/O 如何排除；
- GPU memory 如何测；
- metadata propagation；
- 测试结果；
- 明确确认没有修改 scientific behavior。