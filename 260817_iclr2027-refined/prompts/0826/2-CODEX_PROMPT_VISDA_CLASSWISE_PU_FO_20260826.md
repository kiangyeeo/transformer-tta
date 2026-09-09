# Codex Prompt — Expand VisDA class-wise audit to PU + FO

只做已有 artifact 的结果整理。**不跑任何实验，不 retry/rerun/resume，不修改算法、protocol 或已有 summary.json。**

Repo:
`/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined`

结果来源必须严格分开：

LBI formal:
`experiment_logs/visda_fc_lbi_formal_seed2026_20260825`

Baselines:
`experiment_logs/visda_fc_lbi_seed2026_search_20260819`

已有 FO-only 输出：
`VISDA_CLASSWISE_COMPARISON.md`
`VISDA_CLASSWISE_COMPARISON.csv`

## 目标

把现有 FO-only class-wise audit 扩展为 **PU + FO 两套完整 class-wise 结果**。

必须整理以下 15 个 method/budget rows：

1. source_only
2. module_dense
3. full_dense
4. module_random: `.0005/.001/.002`
5. module_magnitude: `.0005/.001/.002`
6. module_saliency: `.0005/.001/.002`
7. formal LBI: `.0005/.001/.002`

LBI 只能从 formal root 读取；baselines 只能从 baseline root 读取。
禁止拿 tuning LBI 代替 formal LBI。

## 必须实际从每个 summary.json 读取

共同：
- `class-names`

PU：
- `PU-Acc-per-class`
- `PU-mean-class-Acc`
- `PU-overall-Acc`
- `PU-worst-class-name`
- `PU-worst-class-Acc`
- `PU-class-std`

FO：
- `FO-Acc-per-class`
- `FO-mean-class-Acc`
- `FO-overall-Acc`
- `FO-worst-class-name`
- `FO-worst-class-Acc`
- `FO-class-std`

如果字段缺失，明确报 missing；**不要猜值、不要从其他字段推断 class name。**

## 输出

### A. `VISDA_CLASSWISE_COMPARISON_PU_FO.md`

至少包含：

1. provenance 表：
`method / budget / run_id / summary.json absolute path / PU mAcc / PU overall / FO mAcc / FO overall`

2. **12-class PU accuracy 总表**
行=15 个 method/budget rows  
列按 `class-names` 原始顺序：
`aeroplane, bicycle, bus, car, horse, knife, motorcycle, person, plant, skateboard, train, truck`

3. **12-class FO accuracy 总表**
同样 15 行 × 12 classes。

4. **PU hard-class 表**
`method / budget / knife / person / truck / PU worst-class name / PU worst-class acc`

5. **FO hard-class 表**
`method / budget / knife / person / truck / FO worst-class name / FO worst-class acc`

6. 对三个 formal LBI budget，分别计算 PU 和 FO：
- truck vs Source
- knife vs Source
- truck vs same-budget best sparse
- knife vs same-budget best sparse
- mAcc vs same-budget best sparse
- truck vs Module-Dense
- knife vs Module-Dense

same-budget best sparse 仍按对应阶段的 **mAcc** 在 Random/Magnitude/Saliency 中选：
- PU comparison 用 PU mAcc 选 PU best sparse
- FO comparison 用 FO mAcc 选 FO best sparse

7. 最后只回答 artifact 支持的事实：
- PU 下 Source / sparse / Module-Dense / Full-Dense 的 knife、truck 是否也低？
- FO 下同样结论是什么？
- 三个 formal LBI 的 PU worst class 分别是谁？
- 三个 formal LBI 的 FO worst class 是否仍全部是 truck？
- 从 PU→FO，knife/truck/person 哪些明显改善或恶化？
- LBI 相比 same-budget best sparse，在 PU 与 FO 下分别改善/恶化哪些 hard classes？

**只陈述事实，不做机制猜测。**

### B. `VISDA_CLASSWISE_COMPARISON_PU_FO.csv`

一行一个 method/budget，至少包含：
- method
- budget
- PU mAcc / PU overall / PU worst name / PU worst acc / PU class std
- 12 个 PU class accuracies
- FO mAcc / FO overall / FO worst name / FO worst acc / FO class std
- 12 个 FO class accuracies

## 校验

- 确认 15/15 summary.json 被实际读取。
- 确认 15 个 summary 的 `class-names` 顺序完全一致；若不一致立即停止并报告。
- 对每行重新计算 12-class mean，与 summary 中 PU/FO mean-class accuracy 对比；允许仅浮点误差。
- 不覆盖原 FO-only 文件；生成上面的两个新文件。
- 最后打印两个输出文件路径，以及实际读取的 15 个 summary.json 绝对路径。
