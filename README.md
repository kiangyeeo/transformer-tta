# SHOT-OTTA + FC-LBI formal experiments

Transformer DeiT-S source-model training is documented separately in
[`source_training/README.md`](source_training/README.md). It provides a
single-GPU launcher for the three Office-31 source models and the VisDA-C
synthetic-train source model.

当前 formal study 的唯一 normative source 是
[`protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md`](protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md)。
本 README 只保留入口和使用说明；协议数值不在此重复维护。

唯一 formal 配置入口：

- [`configs/otta_fc_lbi_protocol_20260817_v1.yaml`](configs/otta_fc_lbi_protocol_20260817_v1.yaml)
- [`experiments/shot_otta_fc_lbi_formal_20260817_v1.yaml`](experiments/shot_otta_fc_lbi_formal_20260817_v1.yaml)

生成 formal plan（只生成计划，不执行实验）：

```bash
python tools/plan_experiments.py \
  experiments/shot_otta_fc_lbi_formal_20260817_v1.yaml \
  configs/otta_fc_lbi_protocol_20260817_v1.yaml \
  --dry-run
```

预期为 105 个 top-level identities：Office 90、VisDA-C 15。`module_lbi`
的六个 dataset×budget tuned tuples 仍为 TBD；因此 formal dry-run 可以展示
身份，但正式 launcher 会 fail closed，绝不回退到历史/default LBI 参数。

执行已解析的计划时，只有 `module_lbi` 支持 completed-online-batch-boundary
stream checkpoint/resume；其它 variant 始终 `save_model=false` 且从头运行。
Random 的三个 mask 是同一 formal seed 下的 child executions，不是三个 formal
seeds；accuracy 取 mask mean，单 mask efficiency 取 mean，实际三 mask 操作成本
单独记录为 total。

仓库不运行真实 Office / VisDA 实验。轻量验证：

```bash
python -m compileall -q .
git diff --check
python tests/protocol_alignment_smoke_test.py
```
