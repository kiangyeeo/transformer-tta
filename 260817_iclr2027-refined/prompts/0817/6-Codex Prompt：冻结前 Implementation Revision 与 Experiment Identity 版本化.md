请只修改目录：

`260817_iclr2027-refined/`

旧目录仅作只读参考，不要修改。

这是**冻结前最后一次工程修改**。本次只做 implementation revision / experiment identity 版本化，确保当前 refined 实现生成的所有新实验都拥有新的 scientific identity，绝不能误复用旧实验结果。

**不要修改任何算法实现、loss、metric、BN policy、budget、LBI、baseline、数据流、超参数或训练逻辑。**

不要运行任何 Office / VisDA / 真实模型实验。

---

## 1. 增加统一 implementation revision

在合适的 experiment identity 模块中定义唯一常量，例如：

```python
IMPLEMENTATION_REVISION = "iclr2027_refined_20260817_v1"
```

这个 revision 表示当前即将冻结的 refined implementation。

不要为不同 variant 定义不同 revision。

---

## 2. revision 必须进入 scientific config 和 hash

将：

```python
"implementation_revision": IMPLEMENTATION_REVISION
```

加入所有实验的 scientific config，使其参与：

```text
experiment_config_sha256
```

或当前仓库用于判断 scientific identity 的等价 hash。

要求所有 variant 都生效，包括：

```text
source_only
full_dense
module_dense
module_random
module_magnitude
module_saliency
module_lbi
```

目标是：

$$
H(\text{current refined config})
\neq
H(\text{old implementation config})
$$

即使 dataset、seed、budget 和其他实验超参数完全相同。

尤其检查 `source_only` 和 `full_dense`，它们也必须获得新的 identity。

---

## 3. 不要通过特殊 resume 逻辑强制重跑

不要写：

```python
if old_result:
    force_rerun = True
```

也不要删除旧结果。

应当完全依靠新的 scientific identity：

```text
implementation_revision
    ↓
new scientific config
    ↓
new config sha256
    ↓
old completed run 不再匹配
```

现有 resume / skip-completed 机制应继续正常工作。

---

## 4. revision 写入实验产物元数据

确保 `implementation_revision` 能在后续结果中直接追溯。

根据仓库现有结构，将其写入适用的：

- experiment plan entry；
- run / experiment manifest；
- status / result metadata；
- summary CSV / JSON；
- experiment identity metadata；
- finalize / aggregate 结果。

优先复用已有 scientific config / identity metadata 的传播路径，不要新增重复的独立版本系统。

不要为了加入一个字段大规模重构文件格式。

---

## 5. 不要只 bump schema version

`schema_version` 与 `implementation_revision` 语义不同：

- schema version：文件/API 格式；
- implementation revision：scientific implementation semantics。

本次必须让 `implementation_revision` **直接参与 scientific identity/hash**。

如果现有 schema 本身无需改变，就不要为了本次任务额外修改 schema version。

---

## 6. README

同步更新：

`260817_iclr2027-refined/README.md`

增加一个简短的冻结实现说明，例如：

```text
Frozen implementation revision:
iclr2027_refined_20260817_v1
```

并说明该 revision 对应当前 refined implementation，包括已经确定的核心语义，例如：

- corrected LBI Eq.(5) old-state z update；
- strict integer global budget；
- global FC sparse selection；
- strict-budget rollback；
- unified support threshold；
- masked-delta initialization；
- Stage-2 mask-out value freezing；
- Stage-2 fixed LR；
- sparse optimizer mask-out momentum reset；
- Office overall accuracy；
- PU evaluation BN-buffer restore。

这里仅做文档记录，**不要再次修改上述算法代码**。

---

## 7. Tests

增加/更新轻量测试，至少验证：

### A. revision 进入 scientific config

对任意实验：

```python
assert scientific_config["implementation_revision"] == \
    "iclr2027_refined_20260817_v1"
```

### B. 所有 variant 都携带 revision

至少覆盖：

```text
source_only
full_dense
module_dense
module_random
module_magnitude
module_saliency
module_lbi
```

### C. revision 改变 hash

构造除 revision 外完全相同的两个 scientific config：

```text
revision=v0
revision=iclr2027_refined_20260817_v1
```

验证：

```python
sha_old != sha_new
```

### D. 旧 completed run 不会被误复用

使用轻量 mock/temp fixture 构造：

- 一个旧 revision / 无 revision 的 completed experiment；
- 当前 refined revision 的同 dataset/seed/variant 实验。

确认现有 status/resume 逻辑不会把旧 run 判成当前实验的 `skipped_completed`。

### E. 同 revision 的正常 resume 仍有效

同一个：

```text
implementation_revision
+
scientific config
```

已有 completed result 时，原有 skip/resume 行为仍应正常工作。

---

## 8. 回归检查

确保现有轻量测试继续通过，尤其：

- LBI tests；
- budget diagnostics；
- SHOT-OTTA smoke tests；
- experiment engineering；
- launcher；
- resume/checkpoint；
- summary/finalize；
- existing identity/hash tests。

可以运行：

```text
相关 unit/smoke tests
compileall / py_compile
git diff --check
```

**不要运行真实 Office / VisDA 实验。**

---

## 9. 严格禁止

本次不要修改：

- `core/lbi/engine.py` 中任何算法行为；
- LBI budget/stopping；
- support threshold；
- Random/Magnitude/Saliency selection；
- optimizer update semantics；
- SHOT loss；
- pseudo-label；
- Office/VisDA metric；
- BN adaptation/evaluation policy；
- batch size；
- seed；
- experiment budget；
- YAML 中任何数值实验超参数。

如果为了 metadata propagation 必须修改 trainer/planner 等文件，只允许增加或传递 `implementation_revision`，不要改变 scientific behavior。

---

完成后汇报：

1. 修改了哪些文件；
2. `IMPLEMENTATION_REVISION` 的唯一 source of truth 在哪里；
3. revision 如何进入 scientific config 和 SHA；
4. 哪些 artifacts / summaries 会记录 revision；
5. 如何验证 `source_only` 和 `full_dense` 也获得新 identity；
6. 如何验证旧 completed runs 不会被误复用；
7. 跑了哪些轻量测试及结果；
8. 明确确认本次没有修改任何算法或实验数值超参数。

验证全部通过后，当前 `260817_iclr2027-refined/` 即作为后续重新跑实验的冻结实现。