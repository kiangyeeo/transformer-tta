# P2 — Conv-LBI Correctness / Contract Tests

你现在继续在：

260817_iclr2027-refined/

中工作。

当前 Conv Group-LBI 主代码已经完成重构和人工 review。

本轮任务：

> 把 Conv protocol 中已经定义好的 scientific semantics 转成自动化 correctness / contract tests，并运行这些轻量测试。

本轮只做 correctness。

不要：
- 不跑 Office-31；
- 不跑 VisDA-C；
- 不运行真实 target adaptation；
- 不做 accuracy experiment；
- 不做 budget pilot；
- 不搜 rho；
- 不搜 alpha/kappa/nu/omega/stage2_lr；
- 不生成正式实验矩阵；
- 不冻结 Conv final implementation revision；
- 不修改 FC protocol；
- 不因为测试方便而改变 scientific semantics。

==================================================
0. 先读规范和当前实现
==================================================

完整阅读：

Conv protocol：

260817_iclr2027-refined/protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_DRAFT_v0.md

重点检查 protocol 中 correctness / contract requirements，尤其 Section 22。

FC frozen protocol：

260817_iclr2027-refined/protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md

当前 Conv / shared implementation：

260817_iclr2027-refined/core/lbi/groups.py
260817_iclr2027-refined/core/lbi/engine.py
260817_iclr2027-refined/core/lbi/state.py
260817_iclr2027-refined/core/lbi/diagnostics.py

260817_iclr2027-refined/shot_otta/trainer.py
260817_iclr2027-refined/shot_otta/config.py
260817_iclr2027-refined/shot_otta/artifacts.py

260817_iclr2027-refined/experiment_identity.py
260817_iclr2027-refined/protocol_constants.py
260817_iclr2027-refined/train.py

260817_iclr2027-refined/tools/plan_experiments.py
260817_iclr2027-refined/tools/summarize_runs.py

现有 tests 也先读：

260817_iclr2027-refined/tests/

不要先假设实现正确。
测试应该直接锁死 protocol semantics。

==================================================
1. 测试原则
==================================================

新增 Conv-specific correctness tests。

优先：
- CPU-only；
- tiny synthetic tensors；
- deterministic；
- 单测执行快；
- 不加载 Office / VisDA；
- 不下载 checkpoint；
- 不依赖 GPU；
- 不跑完整 SHOT adaptation。

只有 candidate-scope / ResNet layer4 shape contract
确实需要真实模型结构时，
才允许构造 repository 当前使用的 ResNet architecture。

不要加载 pretrained weights。

如果测试发现现有代码与 protocol 不一致：

1. 先明确指出 failure；
2. 做最小必要修复；
3. 不改变无关代码；
4. 修复后重新运行对应 test；
5. 最终汇报修了什么以及为什么。

==================================================
2. Test A — Conv candidate scope
==================================================

锁死 controlled Conv candidate 只有：

netF.layer4.0.conv1.weight
netF.layer4.0.conv2.weight
netF.layer4.0.conv3.weight

netF.layer4.1.conv1.weight
netF.layer4.1.conv2.weight
netF.layer4.1.conv3.weight

netF.layer4.2.conv1.weight
netF.layer4.2.conv2.weight
netF.layer4.2.conv3.weight

检查：

candidate tensor 数 = 9

candidate scalar 总数：

12,845,056

并检查每一层 exact shape。

确保不存在：
- layer1/2/3；
- BN；
- netB；
- netC；
- Conv bias。

==================================================
3. Test B — out_channel grouping
==================================================

对于 synthetic 4D tensor：

W ∈ R^[Cout,Cin,Kh,Kw]

检查 out_channel：

group g=o
W[o,:,:,:]

必须满足：

1. group 数 = Cout；
2. 所有 scalar 被覆盖一次；
3. group 之间无重叠；
4. broadcast group mask 后 shape 与 W 完全一致；
5. 同一个 output channel 中所有 scalar 得到相同 mask。

真实 layer4 candidate 总 group 数必须：

9216

==================================================
4. Test C — filter_connection grouping
==================================================

检查：

group g=(o,i)
W[o,i,:,:]

必须满足：

1. group 数 = Cout * Cin；
2. 所有 scalar 被覆盖一次；
3. group 之间无重叠；
4. group mask broadcast 正确；
5. 同一个 (o,i) kernel 内所有 Kh*Kw scalar mask 相同。

真实 layer4 candidate 总 group 数必须：

6,553,600

整个测试不得 Python loop 枚举 6.5M groups。

==================================================
5. Test D — Group-Lasso prox
==================================================

锁死：

Gamma_g
=
kappa *
(1 - 1 / ||Z_g||_2)_+
*
Z_g

至少测试：

A. zero norm

Z_g = 0
→ Gamma_g = 0
→ 不出现 NaN / Inf

B. ||Z_g||_2 < 1

→ Gamma_g = 0

C. ||Z_g||_2 = 1

→ Gamma_g = 0

D. ||Z_g||_2 > 1

检查输出方向与 Z_g 相同，
并检查：

||Gamma_g||_2
=
kappa (||Z_g||_2 - 1)

允许正常 floating tolerance。

不要用 epsilon 修改数学 threshold。

==================================================
6. Test E — corrected old-state update
==================================================

构造 tiny deterministic tensors：

Theta_delta^k
Gamma^k
gradient
alpha
kappa
nu

手工计算：

c^k =
(Theta_delta^k - Gamma^k) / nu

Theta_delta^(k+1)
=
Theta_delta^k
-
alpha*kappa*(gradient + c^k)

Z^(k+1)
=
Z^k + alpha*c^k

验证 engine 的 Z update 使用的是：

Theta_delta^k, Gamma^k

而不是：

Theta_delta^(k+1), Gamma^k

这个 test 必须能明确区分 old-state 与旧 Conv new-state bug。

==================================================
7. Test F — tau support
==================================================

固定：

tau = 1e-4

构造 group norm：

tau - epsilon
tau
tau + epsilon

检查：

norm < tau
→ inactive

norm == tau
→ active

norm > tau
→ active

即：

M_g = 1[||Gamma_g||_2 >= tau]

禁止 gamma != 0 语义。

==================================================
8. Test G — global integer budget
==================================================

对多个 synthetic Conv tensors 构造一个 global group pool。

检查：

K_G = floor(rho * total_group_count)

必须是所有 tensor 合并后的 global K。

明确验证不会出现：

per-layer floor
ceil
minimum-one-per-layer
budget tolerance

设计一个例子，使：

global floor(rho * sum G_l)

明显不等于：

sum floor(rho * G_l)

从而测试能够真正抓住 per-layer budget bug。

==================================================
9. Test H — strict Stage-1 rollback
==================================================

构造可控制 support trajectory，例如：

0 → 2 → 4 → 7

令：

K_G = 5

预期：

support=4 是 latest feasible state。

下一步 7 > 5：

- overshoot candidate 不 commit；
- final selected_group_count = 4；
- rollback_used = true；
- final selected_group_count <= K_G；
- 不允许 top-K trim 成 5。

另测试：

0 → 2 → 5

应直接接受 exact K：

selected_group_count = 5
rollback_used = false

==================================================
10. Test I — K_G = 0 edge case
==================================================

构造：

rho > 0
但 floor(rho * |G|) = 0

检查：

- Stage 1 不进行无意义迭代；
- final support 为空；
- selected_group_count = 0；
- 不产生 overshoot / top-K 行为。

==================================================
11. Test J — Stage-2 masked-delta initialization
==================================================

构造：

Theta_base
Theta_delta
group mask M

检查：

Theta_init
=
Theta_base + M * Theta_delta

对于 off-mask coordinate：

Theta_init == Theta_base

必须确认不会：

Theta_base + dense Theta_delta

==================================================
12. Test K — Stage-2 strict off-mask freeze
==================================================

必须使用：

SGD
momentum = 0.9
weight_decay = 0.001
nesterov = true

构造 selected / unselected coordinates。

至少执行 1 个 optimizer step。

验证：

selected coordinate 可以改变。

off-mask coordinate：

optimizer step 后必须 exact 等于 pre-step value。

这个 test 必须能证明：

仅 gradient masking 不够，
而当前 restore-off-mask 逻辑确实阻止：
weight decay / momentum / Nesterov
移动未选参数。

==================================================
13. Test L — omega persistent writeback
==================================================

验证 Conv：

Theta_(t+1)
=
(1-omega) Theta_t
+
omega Theta_refined

对于 selected coordinate：
符合 interpolation。

对于 off-mask coordinate：
必须 exact 保持 Theta_t，
不能因为 float32：

(1-omega)x + omega*x

产生 1-ULP drift。

==================================================
14. Test M — next-batch restart
==================================================

模拟连续两个 online batches。

检查 batch t 完成后：

persistent model 可以保留更新。

进入 batch t+1：

Theta_delta^0 = 0
Gamma^0 = 0
Z^0 = 0

上一 batch 的 local LBI state 不得跨 batch 继承。

==================================================
15. Test N — Random global exact-K
==================================================

分别测试：

out_channel
filter_connection

检查：

- global pool；
- selected_group_count == K_G；
- 不按 layer 分配 K；
- 不存在 minimum-one-per-layer；
- formal seed = 2026；
- 3 deterministic child masks；
- 同 seed 重跑 mask 完全一致。

还要验证：

三个 child masks 的 group count 一样，
但 selected_scalar_count 可以不同。

parent summary 应正确保存 scalar count / scalar ratio 的：
mean
std
min
max

不能用最后一个 child mask 冒充 parent statistics。

==================================================
16. Test O — Magnitude global top-K
==================================================

手工构造多个 layers/groups，
让 group norm 排序可明确预期。

score：

s_g = ||W_g||_2

检查：

global top-K_G

不是：

per-layer top-K。

最好设计一个例子，使 global top-K
全部落在少数 layers，
从而确保代码没有 minimum-one-per-layer。

==================================================
17. Test P — Saliency global top-K
==================================================

构造：

W
grad

手工计算：

s_g = ||W_g * grad_g||_2

检查：

- group-wise score 正确；
- global top-K_G 正确；
- exact-K；
- deterministic tie behavior；
- 不发生 per-layer selection。

不需要运行真实 backward graph，
可以直接使用 synthetic gradient。

==================================================
18. Test Q — group vs scalar diagnostics
==================================================

构造 group size 不相同的例子。

例如 selected group 数一样，
但对应 scalar 数不同。

检查明确区分：

selected_group_count

selected_scalar_count

realized_group_ratio
=
selected_group_count / total_group_count

realized_scalar_ratio
=
selected_scalar_count / candidate_scope_param_count

stage1_support_count
必须是 group count。

stage1_support_ratio
必须是 group ratio。

selected_param_count
如果继续保留兼容字段，
必须明确等于 selected scalar count，
不能等于 selected group count。

==================================================
19. Test R — utilization diagnostics
==================================================

给定每 batch：

n_t

以及：

K_G

检查：

u_t = n_t / K_G

summary 正确计算：

group_utilization_min
group_utilization_mean
group_utilization_p05
group_utilization_below_95_count
group_utilization_below_95_fraction

这里只测试 diagnostics。

不要新增：
hard90
或任何尚未冻结的 tuning gate。

==================================================
20. Test S — Conv scientific identity
==================================================

验证 Conv scientific identity 至少区分：

protocol_track
Conv protocol id
candidate scope
group_mode
group partition semantics
requested_budget
LBI tuple
SHOT config
seed
dataset
transfer

同时验证：

out_channel
和
filter_connection

scientific hash 不同。

不同 requested_budget 的 hash 不同。

不同 LBI tuple 的 hash 不同。

==================================================
21. Test T — FC scientific identity regression
==================================================

这是必须做的。

确认加入 Conv 后，以下现有 FC formal variant：

source_only
full_dense
module_dense
module_random
module_magnitude
module_saliency
module_lbi

scientific identity 构造没有改变。

如果 repository 中已有 frozen expected hash / regression mechanism，
直接复用。

不要为了让 test 通过去更新 FC expected hash。

如果当前 FC hash 与 frozen expected 不一致：
视为 regression failure。

==================================================
22. Test U — Conv config fail-closed
==================================================

检查 sparse Conv：

缺 requested_budget
→ 必须报错

不能：
- 自动继承 FC .0005/.001/.002；
- 自动给默认 rho。

错误 group_mode
→ 必须报错。

检查 Conv-LBI：

允许用户只显式提供当前真正需要调的：

alpha
kappa
nu
omega
stage2_lr

其他已冻结 protocol constants 自动继承，例如：

support_threshold = 1e-4
stage1_max_steps = 3000
stage2_steps = 1
delta_nonzero_tolerance = 1e-12

如果用户 override frozen constant 为不一致值：
必须 fail closed。

==================================================
23. Test V — controlled Conv BN / parameter freeze
==================================================

检查 Conv controlled family：

只有 9 个 layer4 Conv weight
允许 persistent change。

所有：
BN affine
BN running_mean
BN running_var
netB
netC
其他 layer

保持 frozen。

BN module 必须处于 protocol 要求的 frozen/eval behavior。

不需要真实数据 forward。

==================================================
24. Test W — PU read-only state
==================================================

如果可以用当前 abstraction 做 tiny synthetic test：

保存 persistent model state。

执行一次模拟 PU evaluation。

检查：

persistent trainable parameters
optimizer-related persistent state
BN state

均没有变化。

如果当前 PU 代码无法不依赖真实 dataloader 完成 tiny test，
不要为了这个 test 大改 pipeline。

这种情况请明确汇报：
“PU read-only contract 仍由已有 FC regression 覆盖，
Conv-specific tiny test 暂未独立实现。”

==================================================
25. 新测试文件组织
==================================================

不要把所有内容塞进一个超长 test。

建议按职责拆，例如：

tests/conv_group_contract_test.py
tests/conv_lbi_dynamics_test.py
tests/conv_budget_contract_test.py
tests/conv_stage2_contract_test.py
tests/conv_sparse_baselines_test.py
tests/conv_diagnostics_identity_test.py

具体文件数可以按现有 repository 风格调整。

每个 test 文件：
- 能单独运行；
- 输出清楚 PASS / failure；
- deterministic；
- CPU-friendly。

==================================================
26. 运行新增 Conv tests
==================================================

新增完成后运行所有 Conv correctness tests。

不要运行任何真实 dataset experiment。

如果某个 test 失败：
先定位原因。

如果是 implementation bug：
最小修复后重跑。

如果是 test 假设超出 protocol：
修 test，不要擅自改变 protocol。

==================================================
27. 再跑已有 FC regression
==================================================

Conv tests 全通过以后，再跑已有轻量 FC regression：

python tests/lbi_smoke_test.py
python tests/protocol_alignment_smoke_test.py
python tests/implementation_revision_smoke_test.py
python tests/per_batch_efficiency_smoke_test.py
python tests/random_multimask_summary_smoke_test.py

以及 repository 当前已有的：
budget diagnostics / planner / summary
相关轻量 smoke tests。

不要运行 Office / VisDA adaptation。

==================================================
28. 不要做 rho pilot
==================================================

本轮 rho 仍然：

TBD

不要在测试结束后自动开始：

0.0005
0.001
0.002
0.005

或任何其他 budget。

测试只验证：

给定 rho 时，
数学 semantics 是否正确。

==================================================
29. 最终汇报
==================================================

完成后给我：

1. 新增了哪些 test 文件；
2. 每个 test 对应 protocol 哪条 contract；
3. 每项测试 PASS / FAIL；
4. 如果发现 implementation bug：
   - bug 是什么；
   - 为什么违反 protocol；
   - 修改了哪个文件；
   - 如何修复；
5. Conv group count / candidate scalar contract 结果；
6. old-state / prox / tau / strict-budget / rollback 结果；
7. Stage-2 masked init / off-mask freeze / omega writeback 结果；
8. Random / Magnitude / Saliency global exact-K 结果；
9. group/scalar diagnostics 与 utilization 结果；
10. Conv config / identity 结果；
11. FC regression 全部结果；
12. 明确说明：
    - 没有运行 Office；
    - 没有运行 VisDA；
    - 没有跑 accuracy experiment；
    - 没有做 rho pilot；
    - 没有搜参；
13. 是否已经可以进入 Conv implementation revision freeze。

不要自行进入下一阶段。