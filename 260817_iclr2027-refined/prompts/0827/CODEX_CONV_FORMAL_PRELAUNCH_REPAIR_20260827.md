# PRE-LAUNCH REPAIR — Conv Formal Baselines

Project:
`/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined`

Formal protocol:
`protocol/shot-otta_conv/OTTA_CONV_BASELINE_FORMAL_20260827_v1.md`

本轮只修 planning / provenance metadata，不启动实验，不修改算法、protocol、FC、scientific config 或任何实验条件。

## Hard boundary

当前 8 个 node 的 scientific plans 已确认正确。

必须保持以下内容完全不变：

- 全部 133 个 `experiment_key`
- 全部 133 个 `experiment_config_sha256`
- 全部 `command_args`
- 全部 `expected_output_root`
- dataset / transfer / seed / variant / budget / group_mode
- Random 3-mask semantics

修复前先导出完整：

```text
experiment_key -> experiment_config_sha256
```

映射。

修复后必须逐项完全一致；任一 SHA / key 变化立即停止。

## 只修以下问题

### Node 1 — Office A→W

统一补：

```text
formal_baseline_protocol_revision:
OTTA_CONV_BASELINE_FORMAL_20260827_v1
```

不要使用其他自定义 formal-evaluation 字段代替。

### Node 3 — Office D→W

将：

```text
baseline_protocol_revision
```

统一为：

```text
formal_baseline_protocol_revision
```

值：

```text
OTTA_CONV_BASELINE_FORMAL_20260827_v1
```

### Node 6 — VisDA dense + out

当前 scientific SHA 已确认正确，不得重新定义实验。

只补全 experiment-level provenance。

每个 experiment entry 应通过 repository 当前 canonical config / identity builder 恢复完整：

```text
protocol_revision
implementation_revision
protocol_track
formal_baseline_protocol_revision
source_checkpoint_revision
scientific_config
```

不要手写一套新的 `scientific_config`，不要改变 SHA。

同时给 Node 6 launcher 补与其他 formal launchers 一致的 preflight：

- exact condition count = 10
- budgets only `.0005/.001/.002`
- no `.005`
- no filter variants
- no Conv-LBI
- scientific identity unique
- output root unique
- protocol / implementation revision correct
- Random seed=2026 + 3 child masks

## Static verification only

修完后检查全部 8 nodes：

```text
total scientific conditions = 133
experiment_key unique = 133
experiment_config_sha256 unique = 133
expected_output_root unique = 133
```

并确认：

```text
before key->SHA mapping == after key->SHA mapping
```

然后运行：

```text
bash -n <all 8 launchers>
git diff --check
```

不要运行训练。
不要 GPU probing。
不要 FINALIZE。
不要重新生成正确的 Node 0 / 2 / 4 / 5 / 7。

写：

```text
experiment_logs/conv_baseline_formal_prelaunch_audit_20260827/PRELAUNCH_REPAIR.md
```

记录：

- 修了哪些 metadata
- 133 个 SHA 是否全部保持
- 8 个 launcher static checks
- 是否 `READY_TO_LAUNCH`

最后只输出：

```text
READY_TO_LAUNCH: YES/NO
SHA_CHANGED: 0/非0
```

然后停止。
