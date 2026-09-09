只做结果整理，不跑任何实验，不修改代码/协议。

Repo:
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined

结果来源必须严格区分：

LBI formal results：
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_formal_seed2026_20260825

Baselines：
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/visda_fc_lbi_seed2026_search_20260819

目标：
检查 VisDA-C formal/baseline results 的 12-class FO accuracy，重点判断 truck 的低准确率是否 baseline 也存在。

请整理：

1. source_only
2. module_dense
3. full_dense
4. module_random: rho=.0005/.001/.002
5. module_magnitude: rho=.0005/.001/.002
6. module_saliency: rho=.0005/.001/.002
7. formal LBI: rho=.0005/.001/.002

重要：
- LBI 只能从 `visda_fc_lbi_formal_seed2026_20260825` 读取。
- Baselines 只能从 `visda_fc_lbi_seed2026_search_20260819` 读取。
- 不允许拿 tuning LBI 代替 formal LBI。
- 不允许根据 FINALIZE.md 猜 class name。
- 必须实际读取每个 run 的 `summary.json`：
  - `class-names`
  - `FO-Acc-per-class`
  - `FO-mean-class-Acc`
  - `FO-overall-Acc`
  - `FO-worst-class-name`
  - `FO-worst-class-Acc`

禁止：
- 不训练
- 不 retry/rerun/resume
- 不改已有 artifact
- 不做机制猜测

输出：

A. `VISDA_CLASSWISE_COMPARISON.md`

包含：

1. Provenance 表：
method / budget / run_id / summary.json path / FO mAcc / FO overall

2. 12-class FO accuracy 总表：
行 = method/budget
列 =
aeroplane, bicycle, bus, car, horse, knife,
motorcycle, person, plant, skateboard, train, truck

3. truck 专表：
method / budget / truck accuracy / FO mAcc /
worst-class name / worst-class accuracy

4. 对 LBI 的三个 budget 分别计算：
- truck vs Source
- truck vs 同 budget best sparse baseline
- truck vs Module-Dense
- FO mAcc 对应差值

其中“同 budget best sparse baseline”按 FO mAcc 在
Random / Magnitude / Saliency 中选最好者。

5. 明确回答：
- Source 的 truck accuracy 是多少？
- Random/Magnitude/Saliency 的 truck 是否也低？
- Module-Dense 的 truck 是多少？
- Full-Dense 的 truck 是多少？
- Full-Dense 是否明显改善 truck？
- 三个 formal LBI 的 worst class 是否都是 truck？
- 三个 LBI 相比各自 best sparse，是改善还是恶化 truck？

只写 artifact 能支持的事实。

B. `VISDA_CLASSWISE_COMPARISON.csv`

保存全部方法的 12-class FO accuracies。

最后：
把两个输出文件，以及本次实际读取的所有 `summary.json` 的绝对路径列表一起给我。