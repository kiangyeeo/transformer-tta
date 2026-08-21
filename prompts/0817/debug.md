只修改：

/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/

不要运行任何真实实验。

当前三个 Office overnight PREPARE 存在确定的 output_root / runs_root mismatch。

train.py / plan_experiments.py 使用：

WORKSPACE_ROOT =
/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI

当前 machine matrices 写：

output_root:
experiment_logs/shot_otta_office_seed2026_stage1_20260818/runs

因此 plan 中 resolved expected_output_root 错误地变成：

/inspire/.../PJ/Split-LBI/experiment_logs/...

但三个 noninteractive .sh 给 launcher 的 --runs-root 是：

/inspire/.../PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/shot_otta_office_seed2026_stage1_20260818/runs

导致 child run 即使完成，launcher 也无法在 runs_root 下发现 summary，
最终将 experiment 判为 failed 并 exit 1。

请做以下修复：

1. 修正 Machine 1/2/3 的 baseline 和 Stage-1 matrices，
   output_root 必须解析到：

   /inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined/experiment_logs/shot_otta_office_seed2026_stage1_20260818/runs

   优先直接使用该绝对路径，避免 WORKSPACE_ROOT 歧义。

2. 重新生成全部 6 个 plans：
   - baseline_machine1: 24
   - baseline_machine2: 24
   - baseline_machine3: 24
   - stage1 machine1 budget .0005: 48
   - stage1 machine2 budget .001: 48
   - stage1 machine3 budget .002: 48

3. 不改变任何 scientific identity/settings。
   output_root 不属于 scientific identity，因此重新生成后每个实验的
   experiment_key / scientific hash 应与修复前保持一致。
   验证这一点。

4. 三个 .sh 的 RUNS_ROOT 保持在 refined repo 内：
   .../260817_iclr2027-refined/experiment_logs/.../runs

5. 给 tools/run_experiments_multi_gpu.py 增加 engineering fail-fast guard：
   在启动任何 child process 前，验证 plan 中每个 experiment 的
   expected_output_root 都位于传入 --runs-root 之下。
   如不一致直接 RuntimeError。
   这是纯路径/launcher safety check，不得修改 scientific behavior。

6. 新增 regression test：
   - 三个 baseline plan 的 expected_output_root 全部 under RUNS_ROOT
   - 三个 Stage-1 plan 同样如此
   - 人工构造 mismatch 时 launcher preflight 必须 fail before execution

7. 更新三个 PREPARE.md，记录此次修复和正确 resolved output path。

8. 只运行：
   - planner regeneration
   - planner dry-run
   - launcher dry-run
   - path-consistency regression
   - bash -n
   - git diff --check
   - lightweight tests as needed

不要运行真实 Office/VisDA experiment。

9. 检查是否存在错误位置：

/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/experiment_logs/shot_otta_office_seed2026_stage1_20260818/

只报告其中是否存在这次失败产生的 orphan outputs。
不要删除，不要迁移，不要复用，因为它在唯一允许修改目录之外。

完成后明确报告：
- 六个 plan count
- 三个 shell runs_root
- 每个 plan 的 expected_output_root prefix
- scientific hashes 是否保持不变
- path guard test 是否通过
- 未运行真实实验