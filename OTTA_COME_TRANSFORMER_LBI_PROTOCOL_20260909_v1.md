# OTTA_COME_TRANSFORMER_LBI_PROTOCOL_20260909_v1

**状态：** 实现规格草案 v1，供后续开发审阅；目标数学与验收要求明确，尚无 COME-Transformer 实现验收或正式结果。  
**范围：** non-distilled DeiT-S + COME-OTTA + QK/VO/FFN Group Split-LBI；Office-31、VisDA-C。  
**日期：** 2026-09-09。  
**形式参考：** [SHOT Transformer frozen protocol](OTTA_TRANSFORMER_LBI_PROTOCOL_20260901_FROZEN_v1.md)。  
**交接依据：** [COME implementation handoff](COME_TRANSFORMER_IMPLEMENTATION_HANDOFF_FROM_RESNET_20260909.md)。  
**本次只读审阅的仓库 HEAD：** f1e680f6bee0bbb4d53135b47fdfb67a1696bd94。  
**待实现 revision 命名：** come_transformer_baseline_20260909_v1 / come_transformer_sparse_lbi_20260909_v1；仅预留，不代表代码已存在或已通过验证。

> 使用已有 SHOT-Transformer substrate，只将 adaptation objective 替换为当前稳定 COME current-logit objective。source、stream、candidate/groups、budget、optimizer family、corrected LBI、PU/FO 保持对应 Transformer 语义。

---

## 1. 来源优先级、范围与冲突裁决

基准/结构语义按当前用户要求、[AGENTS.md](AGENTS.md)、SHOT Transformer frozen protocol 和 [refined FC semantic parent](protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md) 执行。COME 数学按本次 stable handoff 与当前 objective.py；旧 direct-exp 公式仅作等价性测试 reference，不作为生产执行路径。

直接参考：

- [FC COME baseline protocol](260817_iclr2027-refined/protocol/come-otta/OTTA_COME_BASELINE_PROTOCOL_20260907_v1.md)，revision come_otta_baseline_20260908_v2。
- [FC COME sparse/LBI protocol](260817_iclr2027-refined/protocol/come-otta/OTTA_COME_LBI_PROTOCOL_20260907_v1.md)，revision come_otta_sparse_lbi_20260908_v3。
- [stable objective](260817_iclr2027-refined/come_otta/objective.py)、[dense trainer](260817_iclr2027-refined/come_otta/trainer.py)、[sparse trainer](260817_iclr2027-refined/come_otta/sparse_trainer.py)、[LBI adapter](260817_iclr2027-refined/come_otta/lbi.py)。

本地 objective.py 记录的 official COME commit 为 **409a19b71f62c765b1a5be62347a9455524ec176**；这是参考代码的 provenance，本次没有另行在线审计官方仓库。

| 交接中的 ResNet 示例/易混淆项 | 本 Transformer 裁决 |
|---|---|
| Stage-2 SGD、momentum=.9、wd=.001、Nesterov | 继承 Transformer one local **AdamW** step，见 §13 |
| dense scheduler_step_count=1 | 当前 Transformer fixed LR、无 scheduler，因此真实 scheduler_step_count=0，不能人为补一次调用 |
| singleton skip before everything | formal stream 保留 Amazon tail65/完整 PU；unexpected singleton 在所有状态转换前报错，skip仅非正式兼容测试 |
| scalar support threshold=1e-4 | Transformer normalized paired-group ||Γ_g||₂/√768≥1e-4 |
| ResNet BN adaptive/frozen | 对应 Transformer variant 原有 eval/LN/scope，不改为 norm-only，不激活 dropout |
| “到3000仍不正常停” | 当前 Transformer 任何实际3000-step batch均 invalid，含 exact-K/rollback；runner级失效边界如实披露 |
| COME tau=1 看似 identity | forward value 近似 identity，backward 不是；norm.detach 不得删除 |
| ResNet core/lbi | 使用现有 Transformer shared paired-group engine，避免 scalar/SGD 默认值 |
| COME objective tuning | 无 host objective grid；仅后续 LBI search，见 §17 |
| SHOT formal已完成的模板状态 | 本文仅计划开发、测试、搜索门槛，不声称已有 COME formal结果 |

这是一组 matched-scope COME-based OTTA variants，不声称复现官方 benchmark 的所有网络、optimizer 或 norm-only adaptation。full_dense 是本项目的 non-head unrestricted reference。不能新增官方 norm-only baseline 后与 candidate_dense 混用同一身份。

---

## 2. Benchmark、stream 与尾批

| 设置 | Office-31 | VisDA-C |
|---|---|---|
| Transfer | A→D、A→W、D→A、D→W、W→A、W→D | synthetic train→real validation |
| 类别数 C | 31 | 12 |
| Online outer batch / FO batch | 64 / 64 | 256 / 256 |
| Workers / formal seed | 4 / 2026 | 4 / 2026 |
| Target stream | 固定随机排列，一次遍历，drop_last=false | 同左 |
| PU/FO 主指标 | sample-level overall accuracy | fixed-12-class mAcc |

同一 (dataset, transfer, seed) 的所有 COME variant 复用同一 raw sample permutation、batch boundaries 和 augmentation RNG。保存 sample-index 顺序、列表 hash、order hash 与 batch-size 序列；不能仅凭相同 seed 声称 stream 一致。禁止 full-target 非因果聚类、未来样本预提特征和图像 replay。COME 适配仅使用当前 batch。

COME 输入分辨率为 224，每个在线样本使用一个 view，transform 为：
Resize((256,256), bilinear) → RandomCrop(224) → RandomHorizontalFlip(0.5) → ToTensor → ImageNet normalization。
mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]。FO 使用 bilinear resize + CenterCrop(224)。source training 的 bicubic 与 TTA 的 bilinear 差异沿用并披露，不在某个 host 中单独修正。

**尾批裁决：** target Amazon 的 2,817 张图采用 **43×64 + 65** 的 online batching，所有图进入 PU；FO 保持普通 BS64、drop_last=false。DSLR、Webcam 与 VisDA 保留普通尾批。不能直接搬用 ResNet loader 的 singleton-drop 或 FO batch-size×3。

交接强调的“actual BS=1 必须在一切 adaptation 前阻断”保留为防御契约：检查先于 COME objective、selector、scheduler、optimizer、LBI restart、Stage-1/2、omega 和 PU。正常 formal stream 不应产生 singleton；如果产生，报 stream_protocol_mismatch 并终止，不能静默少评一个样本。专门的非正式 singleton compatibility test 可采用 ResNet early-skip（上述计数全部为 0，完整 FO 仍含该样本），必须标记非正式，不混入 Transformer 主表。不能对已合并的 65-sample batch 再 skip。

---

## 3. Source、architecture mapping 与参数作用域

### 3.1 完全复用 source W0

以下路径均以 /home/nas3/biod/wangkangyi/ 为根：

| Transfer | Source-trained best checkpoint |
|---|---|
| A→D、A→W | checkpoints/source_models/office31/amazon.pth |
| D→A、D→W | checkpoints/source_models/office31/dslr.pth |
| W→A、W→D | checkpoints/source_models/office31/webcam.pth |
| VisDA train→validation | checkpoints/source_models/visda-c/train.pth |

复用 transformer/source_only/model.py 的 load_frozen_source_model 和 manifest/hash 校验。本地构造 pretrained=False；不隐式下载、不重训 source、不重置已训练 head、不用 ImageNet initialization 或 .last.pth 替代 W0。路径、revision 和 SHA-256 必须与相同 transfer 的 SHOT source 一致。

本次文档未重新读取大 checkpoint 或验证 hash；实现/运行时必须真实核验。记录绝对路径、SHA-256、manifest、best epoch、source validation metric、source seed/config、Git commit/dirty state 与 Python/PyTorch/torchvision/timm/CUDA 环境。缺失 provenance 不得捏造。source revision 使用 Transformer 实际 manifest/identity，不得照抄 ResNet 的 nips2026_shot_otta_uda_source_v1。

### 3.2 Architecture mapping

| ResNet 交接中的概念 | Transformer 对应 |
|---|---|
| source F/B/C | 已有完整 timm DeiT source model + direct classifier |
| netB 分类前特征 | 原 classifier 输入的 384 维 pre-logits representation |
| netC | 已训练 Linear(384,C)，所有本协议 variant 均冻结 |
| FC bottleneck / layer4 Conv | 不移植；使用 §4 的 QK/VO/FFN candidate |
| BN adaptive/frozen | 不移植；沿用 Transformer eval、LayerNorm 与 scope |
| ResNet scalar/out-channel LBI | 不移植 group 轴；使用 Transformer paired groups |

Backbone 固定为 non-distilled deit_small_patch16_224.fb_in1k：12 blocks、d=384、MLP hidden=1536。参考 SHOT 冻结环境 timm=1.0.28；运行前核验实际版本。禁止新增 bottleneck、distillation token/head、head averaging 或改变 pooling。

COME 直接沿用原模型的 logits 路径：

~~~python
logits = model(inputs)
~~~

验证 logits=[B,C]，且同一 checkpoint、输入与状态下，COME 初始 logits 与原 source model 输出相等或满足预声明的浮点误差界。保持原 classifier、pooling 和 normalization；COME objective 不需要额外 feature-extraction API。

### 3.3 Dense 与 controlled scope

| Variant | 可更新参数 | Model mode |
|---|---|---|
| source_only | 无；全部 parameter/buffer 精确不变 | eval |
| full_dense | 所有 DeiT 参数，除 classifier/head weight/bias | eval |
| candidate_dense | §4 的 12 个 weight tensor | eval |
| group_random / group_magnitude / group_saliency / group_lbi | 同一 candidate 内当前选中 paired groups | eval |

full_dense 包含 attention/MLP biases、LayerNorm、patch embedding、cls token、pos_embed、final norm，是 unrestricted reference，不是 matched-support selector。controlled family 冻结全部 bias/LN、blocks 0–8、patch embedding、cls token、pos_embed、final norm、head。eval 不禁止 autograd；适配前向仍按 scope 求导。禁止 broad optimizer parameter group 意外纳入 off-scope 参数。

---

## 4. Candidate、paired groups 与预算

复用 transformer/candidate_dense/ 与 transformer/group_random/groups.py。exact names、shapes、顺序、parameter identity 与 scalar coverage 全部一致，不能仅比较总参数量。

| 每个 blocks.9 / blocks.10 / blocks.11 中的 suffix | Shape |
|---|---|
| attn.qkv.weight | [1152,384] |
| attn.proj.weight | [384,384] |
| mlp.fc1.weight | [1536,384] |
| mlp.fc2.weight | [384,1536] |

共 12 tensors、5,308,416 scalars。qkv 保持物理 fused，Q/K/V 为逻辑 row slices。按 block→QK/VO/FFN→coordinate 分配 canonical group ID 0…6911：

$$
G_{l,p}^{QK}=\{W_{Q,l}[p,:],W_{K,l}[p,:]\},\quad
G_{l,p}^{VO}=\{W_{V,l}[p,:],W_{O,l}[:,p]\},
$$
$$
G_{l,p}^{FFN}=\{W_{1,l}[p,:],W_{2,l}[:,p]\}.
$$

每 block 为 384 QK、384 VO、1536 FFN；总共 6912 groups，每组 768 unique scalars。groups 无重叠且完整 partition candidate。检查 tied/shared storage、重复 parameter identity 和重复计数，不靠模糊 prefix 抓参数。

本文用 **C** 表示类别数、**K_G** 表示结构预算，对应 requested_group_count：

| rho_struct | K_G=floor(rho×6912) | exact-K_G active scalars |
|---:|---:|---:|
| 0.0005 | 3 | 2304 |
| 0.001 | 6 | 4608 |
| 0.002 | 13 | 9984 |

全局预算，没有 ceil/slack、per-block/type quota、minimum-one、top-K fill。Random/Magnitude/Saliency 每次 exact K_G；LBI 合法 rollback 后可 ≤K_G。报告 requested rho、selected/6912 和 selected×768/5308416；历史 update union 不受单批 K_G 限制。这是 sparse adaptation，不是 physical pruning，不自动产生 FLOP/推理加速收益。

---

## 5. COME objective：精确数学与符号

输入只有当前模型、当前 target batch 的 **current logits** z∈R^(B×C)，类别 C=31（Office）或12（VisDA）。无需 pseudo-label、teacher、memory bank、feature geometry、source-logit anchor、class prototypes、EMA target 或 target GT。现有 SHOT 的 pseudo CE、entropy/diversity/clustering 全部不进入 COME objective。

为避免命名撞车：class_count=C（交接的 K）、structural budget=K_G、come.tau=1、lbi.tau_g=1e-4。

### 5.1 Constrained logits

对每个样本：

$$
r_i=\|z_i\|_2,\qquad
\bar z_i=\frac{z_i}{r_i}\operatorname{sg}(r_i)\tau,\quad \tau=1.
$$

固定 p=2、norm dimension=last class dimension、keepdim=true：

~~~python
norm = torch.norm(logits, p=2, dim=-1, keepdim=True)
constrained = logits / norm * norm.detach() * 1.0
~~~

**无 norm epsilon/clamp**。不能写 norm+1e-6、clamp_min、normalize(...,eps=...)、nan_to_num，也不能令 constrained=logits 或去掉 detach。

在 r>0、tau=1 时，forward 与 z 数学上相同，但 Jacobian 为 I−zzᵀ/r²，区别于 identity。loss/gradient 等价测试必须保留此 backward path，不能只测 forward values。

### 5.2 Stable log-domain subjective opinion

直接公式仅用于说明：

$$
e_{ic}=\exp(\bar z_{ic}),\quad S_i=\sum_c e_{ic}+C,\quad
b_{ic}=e_{ic}/S_i,\quad u_i=C/S_i.
$$

生产执行用数学等价的稳定形式：

$$
\log S_i=\operatorname{logsumexp}(\bar z_{i1},\ldots,\bar z_{iC},\log C),
$$
$$
b_{ic}=\exp(\bar z_{ic}-\log S_i),\qquad
u_i=\exp(\log C-\log S_i),\qquad
o_i=[b_{i1},\ldots,b_{iC},u_i].
$$

~~~python
log_c = constrained.new_full((constrained.shape[0], 1), math.log(float(class_count)))
log_strength = torch.logsumexp(
    torch.cat((constrained, log_c), dim=1), dim=1, keepdim=True
)
belief = torch.exp(constrained - log_strength)
uncertainty = torch.exp(log_c - log_strength)
opinion = torch.cat((belief, uncertainty), dim=1)
~~~

不回退 direct exp(constrained)。VisDA 上的真实故障链为 exp overflow→inf/inf→NaN belief/loss；日志 diagnostics 也不能为输出 evidence 而重新引入直接指数溢出。记录 log_evidence=constrained、log_strength 等稳定量即可。

### 5.3 Opinion entropy

固定 entropy_epsilon=1e-7：

$$
o_{\epsilon,ij}=o_{ij}+10^{-7},\qquad
L_{\mathrm{COME}}=-\frac1B\sum_i\sum_{j=1}^{C+1}
o_{\epsilon,ij}\log o_{\epsilon,ij}.
$$

~~~python
entropy_input = opinion + 1e-7
entropy = -(entropy_input * torch.log(entropy_input)).sum(dim=1)
loss = entropy.mean()
~~~

加 epsilon 后 **不 renormalize**；也不只在 log 内加 epsilon、不改成 clamp entropy_input。未加 epsilon 的 opinion sum≈1；加后 sum≈1+(C+1)×1e-7，不应强制它再等于1。不是 ordinary C-class softmax entropy，不加入 SHOT diversity 或 confidence filtering。

C 必须是 adaptation classifier output dimension，断言 logits.ndim=2、logits.shape[1]==class_count、C∈{31,12}（由 dataset 决定）。禁止沿用 ImageNet 1000 或把当前 predicted-class count 当 C。

---

## 6. 数值稳定与 failure behavior

CPU objective contracts 必须先于 Transformer GPU 集成：

| 场景 | 必须验证 |
|---|---|
| Moderate finite logits，C=31/12 | stable vs direct reference 的 constrained/belief/uncertainty/loss/gradient close |
| 大但有限且 norm 可表示的 logits | direct exp reference 溢出；stable belief/uncertainty/loss/gradient 有限 |
| tau=1 detach contract | forward 与 raw logits close；gradient 与保留 stop-gradient 的 reference 一致，并区别于 identity shortcut |
| 零 norm / 非有限 logits或gradient | 明确抛错并 invalid，不自动加 epsilon/clamp/nan_to_num |
| class-count/shape mismatch | objective 前拒绝，不能默认 C=1000 |
| epsilon treatment | 同时在乘数与 log 输入中加1e-7，未做 renormalization |

log-domain 解决 evidence exponential overflow，不保证任意极端 FP32 输入的 L2 norm 都可表示。若 norm 计算本身非有限，按 fail-loudly 处理，不能承诺所有 finite logits 无条件 safe。

每次 objective、gradient、Δ/Γ/Z、refined/persistent state 都要检查 finite。异常写 invalid_reason、batch/stage、真实调用计数；不运行后续 batch/FO，不跳过坏 batch 后继续充当完整 run。修改数学稳定化约束必须有新 protocol revision；当前无 p/tau/C/evidence-clamp 搜索维度。

---

## 7. 输入、方法状态与 objective 所有权

COME 使用单个 online transformed current batch；PU 复用该输入 tensor，FO 另用 CenterCrop。适配直接计算 current-logit objective，不构造多视图任务、图传播、memory 或 frozen pseudo-target。

| 状态 | 生命周期 |
|---|---|
| persistent model | 跨 batches；每 transfer/Random child 从 W0 开始 |
| host AdamW moments | dense/sparse non-LBI 跨 batches |
| Random/Magnitude masks | run-local static |
| Saliency mask/current gradients | batch-local，once selection 后同一步使用 |
| Δ/Γ/Z/M、Stage-2 optimizer | 仅 LBI，batch-local |
| COME p/tau/C/epsilon | immutable objective config，无 adaptive state |
| target labels | 仅 PU/FO metric bookkeeping |

COME closure 应是 backbone-independent current-logit objective；architecture-specific 仅 model forward/scope/group builder。形式为：

~~~python
logits = model(current_inputs)
result = come_loss(logits, class_count=dataset_class_count)
return result.loss, result.diagnostics
~~~

诊断从当前结果 detach 得到，不再调用 objective 或写 method state。closure 无 optimizer/scheduler/writeback。full_dense、candidate_dense、sparse、Stage-1 和 Stage-2 都调用同一 stable objective，而不是各实现一份公式。

---

## 8. Dense COME：current-state one-step contract

full_dense/candidate_dense 的每个 valid outer batch：

~~~text
validate batch before all adaptation transitions
→ correct Transformer eval/scope
→ persistent host AdamW.zero_grad
→ fresh current model forward
→ stable COME objective
→ backward
→ exactly one optimizer.step
→ separate read-only post-update PU
~~~

使用 §10 的 shared AdamW：lr=1e-5、betas=.9/.999、eps=1e-8、wd=.01；optimizer 跨 batches 持续，不每批重新创建。constant LR、**无 scheduler**。因此计数为 objective_call_count=1、optimizer_step_count=1、scheduler_step_count=0。不能为匹配 ResNet 交接表的 scheduler=1 伪造调用/状态；也不把 SHOT loss logits当COME PU。

Dense COME 直接保留 host optimizer 更新后的参数，不进行 EMA、teacher moving average 或 omega 插值。full_dense 除 head 外全可更新，candidate_dense 只12 tensors；mode/LN policy见 §3/14。两条dense不按稀疏rho重复运行。

---

## 9. Sparse non-LBI：selector 与 gradient reuse

### 9.1 Random

三条真实独立 child trajectories，support seeds=202600/202601/202602。每 child fresh 相同 W0、optimizer、stream/augmentation generator；只 support 不同。每 budget global exact K_G，mask整stream不变。

继承 SHOT Transformer frozen 的 cross-budget nested-prefix：每 seed 生成同一 canonical 6912 permutation，取前 K_G。记录 child index/seed/group IDs/hash/source/stream与真实execution artifact，不能仅metadata写num_random_masks=3而只跑一次。

### 9.2 Magnitude

W0 上 CPU float64 的 paired-group ||W_g||₂，全局 top-K_G，canonical ID tie break；stream前算一次，随后静态。COME objective 不参与构建。不改成sum|W|或每batch当前参数magnitude。

### 9.3 Saliency

每个 valid outer batch 在当前 Θ_t 对同一 current batch **一次** COME forward/backward：

$$
s_g=\|(W\odot\nabla_W L_{\mathrm{COME}}(B_t;\Theta_t))_g\|_2.
$$

按 global top-K_G/canonical ID tie break 构造完整paired mask；不改成 gradient norm、不scalar选取、不按group/chunk重复objective。

selection 前后的 model parameters/buffers、CPU/CUDA RNG、module modes、objective config/state 必须相同。此时复用同一次已计算 gradient，执行一个strict masked host AdamW step。selection不能重新forward、消费RNG或更新normalization，不能把额外 backward 隐去后仍声称objective_call_count=1。

Random/Magnitude 同样一次current objective/backward后masked step。所有non-LBI sparse都是 persistent host optimizer，one masked step/batch，无LBI Stage-2、无EMA/omega。历史selected坐标的更新可保留；本batch off-mask禁止新增变化。

### 9.4 正常成功 batch 计数

| 路径 | objective calls | support selection | host steps | scheduler steps | omega / native EMA |
|---|---:|---:|---:|---:|---|
| dense | 1 | 0 | 1 | 0 | 0 / 0 |
| Random/Magnitude | 1 | stream前1次 | 1 | 0 | 0 / 0 |
| Saliency | 1 | 本batch1次 | 1 | 0 | 0 / 0 |

若为显存分块，逻辑 full-objective 是当前B上的sample mean，n_r/B加权累积，禁止每chunk step/selection。当前优先沿用SHOT单batch路径；任何分块需先测等价、记录physical forward count，不能用它改变上述逻辑计数。

---

## 10. Strict sparse AdamW 与 off-mask exactness

COME dense/sparse 共用 AdamW：lr=1e-5、betas=[0.9,0.999]、eps=1e-8、weight_decay=0.01、scheduler=none、AMP=false。不同 selector 不得分别调 base LR；每 outer batch 执行一个 host optimizer step。Stage-2 的 LR 由 COME-LBI tuple 冻结，其余 AdamW 设置相同。

每次 sparse step 必须复用/等价于 transformer/group_random/optimizer.py 的 strict_masked_adamw_step：

1. step 前 snapshot parameter values，清零当前 off-mask exp_avg/exp_avg_sq，mask gradients。
2. 执行一次 AdamW step。
3. 从 snapshot 精确恢复 off-mask values，再清零并断言 off-mask moments 为 0。
4. 连续选中 coordinates 保留历史 moments；离开 support 的 moments 清零，重新进入时不恢复陈旧 moments。

Adam scalar step 是 tensor 级 bookkeeping，沿用 shared helper，不虚构逐 coordinate step，也不为清 off-mask state 把整个 tensor 的 moments/step 每批重置。LBI Stage-2 例外：optimizer 本身为 batch-local fresh instance。

仅 mask gradients 不充分，weight decay 和 moments 均可造成 drift。动态 mask 只控制本次变化；off-mask anchor 是本次 step/outer batch 的 persistent 值，不能强行恢复 source W0。

COME-LBI 的 omega 写回后再显式恢复 off-mask base，防止对本来相等的值做插值产生 1 ULP 差异。off-scope tensors 直接保留。验收用 torch.equal/byte-level hash，不以 allclose 代替 exact invariance。

---

## 11. Shared Group Split-LBI：restart 与 corrected dynamics

### 11.1 实际复用边界

Transformer 实现实际位于 transformer/group_lbi/engine.py::GroupSplitLBIEngine。交接中的 260817_iclr2027-refined/core/lbi/engine.py 是 ResNet 共享引擎，默认 scalar/SGD，不能原样替换前者。

COME-LBI 必须复用 **现有 Transformer corrected LBI 引擎**，通过 COME current-logit closure 接入。可以在回归保护下抽公共核心，但不另写 COME 私有 LBI 数学，也不为目录名统一移植 FC optimizer/group 数值。现有 SHOT closure 行为必须保留。

现 Transformer engine 消费可微 scalar loss，并内部调用 autograd.grad/backward。COME closure 返回保留计算图的 loss，由引擎执行求导；不能在 closure 内先 backward，再将 detached scalar 交给引擎重复求导。若后续采用分块 gradient accumulator，必须显式区分已累积梯度与可微 loss 两种接口，并针对同一 COME batch 验证等价性。

### 11.2 Batch-local restart

进入 batch 的 persistent model 为 Θ_t。candidate 上初始化：

$$
\Delta^0=0,\quad \Gamma^0=0,\quad Z^0=0,\quad M^0=0.
$$

仅重启局部变量，不每批重载 source；跨 batch 保留的是 omega 更新后的 persistent model。COME-LBI 不使用 persistent host optimizer、host scheduler 或 EMA。

### 11.3 Current-state objective 与 old-state update

令 L_COME,t 为 §5 定义的当前 batch current-logit objective：

$$
L(\Delta,\Gamma)=L_{\mathrm{COME},t}(\Theta_t+\Delta)
+\frac{1}{2\nu}\|\Delta-\Gamma\|_2^2,
$$
$$
g^k=\nabla_\Delta L_{\mathrm{COME},t}(\Theta_t+\Delta^k),\quad
c^k=(\Delta^k-\Gamma^k)/\nu,
$$
$$
\Delta^{k+1}=\Delta^k-\alpha\kappa(g^k+c^k),\qquad
Z^{k+1}=Z^k+\alpha c^k,
$$
$$
\Gamma_g^{k+1}
=\kappa\left(1-\frac{\lambda_{\mathrm{prox}}}{\|Z_g^{k+1}\|_2}\right)_+
Z_g^{k+1}.
$$

c^k 必须由旧 Δ/Γ 计算；Z 不能使用新 Δ。unweighted group lasso J(Γ)=Σ_g||Γ_g||₂，prox_lambda=1。零 norm 安全返回零 group，不改变阈值规则。

每个 Stage-1 candidate 都 fresh forward/objective/full gradient；禁止缓存旧 logits/loss/gradient。临时 candidate 参数不能留在 persistent model，异常也要恢复 batch base。COME closure 仅计算 loss 与只读 diagnostics，不执行 optimizer、scheduler 或参数写回。

---

## 12. Support、strict rollback 与 3000-step invalidation

唯一 formal support：

$$
M_g=\mathbf1[\|\Gamma_g\|_2/\sqrt{768}\ge10^{-4}].
$$

lbi.tau_g=1e-4；count、budget、rollback、final mask、utilization、Stage-2 init 都用同一定义。Γ!=0、非零 Δ、非零梯度不是 formal support。

初始全零状态是 feasible checkpoint：

| 新 count S | 动作 |
|---|---|
| S<K_G | 保存最新 feasible Δ/Γ/Z/M/group IDs 与对应 step，继续 |
| S=K_G | 接受并停止 |
| S>K_G | 丢弃 overshoot，恢复最近 feasible 全状态并停止 |

不能仅恢复 mask 而保留 overshoot Δ/Z，不能 top-K trim/fill，不能接受 K_G+1。合法 rollback 可返回 S=0 或 S<K_G，与 max-step failure 分开报告。

stage1_max_steps=3000 固定。**任何 batch 实际 Stage-1 执行步数达到 3000，即使该步恰好 exact-K 或 rollback，也按当前 Transformer runner 规则使整 run invalid：**

~~~yaml
valid_lbi_run: false
result_validity: invalid
invalid_reason: stage1_3000_step_hit
termination_batch_index: <actual index>
~~~

不做该 batch PU，不处理后续 batch，不做 FO，不让 partial metric 进入正式结果/排名；不能提高 cap 救配置。

对齐 SHOT parent 的 runner-level invalidation：现 engine 在 run_batch 内已执行 Stage-2/omega，然后 runner 检查 3000。因此 invalidating batch 可能已有 Stage-2 compute；状态与计时只保留失败诊断，不能存成可继续的 completed-batch checkpoint，整个 run 仍作废。本文不声称它避免了 Stage-2。未来改成 Stage-1 内提前短路，须明确新实现 revision 与计数，不隐瞒实际执行边界。

---

## 13. Stage-2 与唯一 persistent omega writeback

从最后 feasible 状态初始化：

$$
\Theta_{t,2}^{0}=\Theta_t+M^\star\odot\Delta^\star.
$$

off-mask dense delta 不得进入参数。重新计算 **Stage-2 当前状态**的完整 COME objective，仅执行 **one masked local AdamW step**：

| 项 | 固定值 |
|---|---|
| optimizer lifetime | fresh instance / outer batch；batch 后丢弃 |
| betas / eps / weight decay | [0.9,0.999] / 1e-8 / 0.01 |
| steps / scheduler / AMP | 1 / none / off |
| stage2_lr | dataset×method×budget 冻结 tuple 的 LR |
| off-mask protection | §10 的 exact values + moments 保护 |

交接的“one SGD step、momentum=.9、wd=.001、Nesterov”为 ResNet 数值，Transformer 使用上述 AdamW。若 COME 分 chunk 计算，先按 n_r/B 累积完整当前 batch 的 objective gradient，再仅 step 一次。

得到 refined Θ̃_t 后：

$$
\Theta_{t+1}=(1-\omega)\Theta_t+\omega\widetilde\Theta_t.
$$

仅 support 内允许 persistent change，写回后再次 exact restore off-mask base。每成功 batch：omega_writeback_count=1、native_ema_commit_count=0、host_optimizer_persistent_step_count=0、host_scheduler_step_count=0。然后独立 PU，并丢弃 Δ/Γ/Z/M、Stage-2 optimizer 和局部 task。omega 改变未来梯度/support，不是报告时的输出插值。

---

## 14. Model mode、precision 与 RNG

所有 variant 沿用 SHOT Transformer 的 model.eval()，适配时启用必要梯度；Dropout、attention/MLP dropout、DropPath/stochastic depth inactive。source manifest 要求 drop_rate=drop_path_rate=0。LayerNorm affine 是否可更新按 §3 scope，而不是按 train/eval 推断；没有 BN running stats 不等于可忽略 LN parameters。

formal TTA 为 FP32/non-AMP；不得为显存或稳定性临时开 AMP、改变 gradient scale/support threshold/norm clamp。source training AMP 不改变此规则。

分离 stream、online augmentation 与 Random support RNG。PU/FO、selector 排序、诊断不可推进 adaptation 所需 CPU/CUDA/Python/NumPy RNG；必要时 snapshot/restore，但不能掩盖错误 state transition。相同状态、输入和 RNG 下 logits/gradient 可复现。worker/prefetch、resume 也须检查 sample-index/view hash；mask 生成不消耗 augmentation RNG。

---

## 15. PU / FO 与 target-label 边界

PU 必须在本 batch 的 host optimizer 更新或 LBI omega 写回完成之后，使用同一缓存 online input tensor，单独 fresh forward。不能复用 COME loss logits；每张原图仅计一次。

PU/FO 可读但不可写 parameters、buffers、optimizer、scheduler、LBI state、RNG 或任何未来新增 method state。evaluation 接口不能调用 COME adaptation closure、optimizer 或 omega 写回。PU 使用 no_grad/inference、eval；评估后 state hash/计数不变。

完整 stream 成功结束后 freeze final model/method state，使用独立 full-target CenterCrop loader 做 FO。FO 只进行 inference，不调用 COME objective、参数更新或 omega 写回，也不消费 online augmentation generator。在线随机 crop 与 FO center crop 不同，因此 source-only PU/FO 数值不必相同；只在输入 tensor 与 evaluation 条件相同时要求 prediction 一致。

Target labels 在 runner 的 metric 分支隔离。COME objective、selector、Stage-1/2、writeback 接口不得接收 ground-truth label 或带 label 的完整 batch object。置乱/替换 target labels 后，adapted model、support 与 optimizer state 必须不变。允许只读 offline reporting；若将来用 labeled FO 选超参数，必须先在独立 search protocol 披露用途、范围和固定规则。PU 为 report-only，不作为调参信号。

Office 每 transfer 累积 correct/total，不做 mean(batch_acc)；六 transfer 等权平均，不跨 transfer pool samples。VisDA 在全 stream/full FO 累积 class_correct[12]/class_total[12]，固定 12-class macro，同时保留 overall、12 class accuracies、worst class/name、class std。缺类不得改为 observed-class mean；短 smoke 可标 diagnostic/incomplete，formal full-target class coverage 不全应报错。

JSON/CSV 保留未预先舍入的数值。表格可显示四位小数，ranking 不能读格式化字符串。accuracy unit（fraction 或 percent）必须写明且聚合一致。

---

## 16. 有效性、support 与 collapse diagnostics

正式 run 要求完整 target stream、PU/FO 样本计数正确、无 NaN/Inf/runtime exception、source/stream/scope/seed/config 身份一致、每批 S≤K_G、无 cap hit、日志完整。失败 run 写明确 invalid_reason，不将 partial mean 伪装成完整结果。

每 batch 记录 K_G、selected S、u=S/K_G、selected group IDs/hash、QK/VO/FFN 和 block 分布、Stage-1 executed steps、last feasible step、stop reason、rollback、Stage-1/2 runtime。聚合 selected min/mean/max，utilization min/mean/p05，低于 90%/95% 次数与比例、exact-K rate、cap/rollback/failure rate，steps mean/max。

util95 是 u≥.95 的 batch 比例；Office 如聚合此诊断，pool 六 transfer 的 batch numerator/denominator，区别于 accuracy 的六 transfer 等权平均。合法 rollback underutilization 不自动 invalid；无额外 hard 90%/95% eligibility gate，也不因低 utilization top-K 修补。选择/排序规则另见 §17。

保存 historical active-group union，以及最终 relative-to-source parameter delta support；二者区别于单批 selected support。delta_nonzero_tolerance=1e-12 仅为诊断，不替代 Gamma normalized threshold 或 off-mask exact test。

COME 至少保留 predicted-class histogram、predicted class count、dominant-class ratio、mean softmax entropy、opinion entropy 与 mean uncertainty mass。用这些区分一般退化、类别失衡、单类 collapse、数值失败；不能仅因 smoke accuracy 高低调算法，也不能事后发明 collapse 阈值挑结果。

---

## 17. Hyperparameters、kappa=1 方向与 formal gate

COME host objective **无待调参数**：p=2、tau=1、C=dataset classes、entropy_epsilon=1e-7，no norm epsilon/clamp、stop-gradient、stable log-domain、no-renormalization固定。不能开tau/p/K/evidence-clamp grid。

Transformer Group-LBI 固定candidate/budget、prox_lambda=1、normalized tau_g=1e-4、cap=3000、Stage-2 AdamW一步/无scheduler/non-AMP。六个 COME profiles 均 TBD：

| Dataset | rho | (alpha,kappa,nu,omega,stage2_lr) |
|---|---:|---|
| Office-31 | .0005 | TBD |
| Office-31 | .001 | TBD |
| Office-31 | .002 | TBD |
| VisDA-C | .0005 | TBD |
| VisDA-C | .001 | TBD |
| VisDA-C | .002 | TBD |

Office同budget一个tuple用于六transfers，VisDA独立；不复制FC/SHOT winners，不per-transfer/seed/batch/class调参。

按交接§38，后续搜索**建议主线**：

$$
h_{\mathrm{S1}}=(\alpha,\nu),\quad \kappa=1,
$$
$$
\alpha\in\{0.025,0.05,0.10,0.125,0.15,0.20\},\quad
\nu\in\{0.25,0.50,1.00\}.
$$

这是18个Stage-1候选的建议域，本次不启动、不声称完成搜索。COME优先从kappa=1开始，不主动扩大kappa维度；同时须另行检查kappa=1 constrained tuning是否基本保留已有SHOT Transformer性能。不能倒改SHOT已冻结的kappa>1 winners。

omega×stage2_lr 仍需downstream joint calibration，暂不作为predictor target；范围、shortlist、ranking、validation/target-label使用方式须先写独立search protocol。改变omega/LR会改变未来Stage-1 dynamics，因此每个calibration配置重新验证finite、budget、cap与完整stream，不能沿用anchor有效性。

PU仅report。若search使用FO target labels，需事前披露离线选择规则，不能混入online objective，也不能在看完formal matrix后扩grid。没有额外hard utilization gate；合法rollback underfill报告即可。

当前formal COME-LBI必须blocked，直至search protocol和六tuples冻结；formal=true遇debug/provisional/fallback tuple直接拒绝。baseline也需自己的实现/tests/smoke验收，不因ResNet baseline可formal就直接放行Transformer。

---

## 18. Runtime / GPU-memory protocol

COME 的 measurement unit 是原始 **online outer batch**；计算 chunk 或 Stage-1 inner step 仅作为批内诊断单位。

计时在 DataLoader yield、CPU decode/增强和输入 H2D 完成后开始，包含完整 adaptation + 独立 PU。CUDA 在区间边界 synchronize；预先 reset peak-memory statistics。排除 DataLoader wait、CPU preprocessing、初始 H2D、checkpoint I/O、JSON/summary I/O。COME forward/objective/backward、动态 selector、masked optimizer step，以及 LBI 的 Stage-1/2、mask 构造与 omega 写回均属于 adaptation，不能只计 optimizer 时间。

每批记录：

~~~text
batch_index, raw_batch_size
adapt_runtime_sec, pu_runtime_sec, online_runtime_sec
lbi_stage1_runtime_sec, lbi_stage2_runtime_sec
peak_gpu_memory_allocated_bytes/mb
peak_gpu_memory_reserved_bytes/mb
~~~

COME 可额外记录 objective/selector/optimizer/writeback 子阶段计时及分块设置，所有子阶段须对应实际执行路径。CPU 增强与初始 H2D 不属于主 compute timer，其开销在 preprocessing_runtime、H2D_runtime 与 wall_runtime 中单独记录。

优先在计时前将完整当前 batch 输入传到设备，仅分块控制 activation 内存。如实现需要分段 H2D，则须暂停/单独核算该 H2D，明确 timer revision，不能因 chunk 大小改变主 timer 的包含项。chunk 设置以及 peak 中已驻留的输入、model、optimizer/LBI tensors 均真实记录。

run-level 保留 mean/std/median/p95 online batch runtime、Σonline runtime、adapt/PU totals、Stage-1/2 shares。主 memory 为所有 batch 的 **max peak allocated**；reserved 是诊断。FO_runtime 和 wall_runtime 单独报告。

Random 顶层 accuracy 为三 child mean、population std（ddof=0）。可比 runtime 是 mean single-child batch/stream/FO cost；三 child 真实总开销另存 random_total_online_compute_runtime_sec / random_total_fo_eval_runtime_sec。GPU peak 为所有 child 所有 batch 的最大值，不能三张 GPU memory 相加。

正式效率比较按同 GPU 型号（参考 RTX 3090）、1 experiment process/GPU、同软件、precision、BS、workers、transform/stream；runtime_comparable=true 只能由匹配的 execution condition 支持。共享 GPU、不同硬件或 resume 不作为正式可比效率。准确率与效率有效性分别记录。

---

## 19. Checkpoint / resume 与文件边界

source_only、full_dense、candidate_dense、Random/Magnitude/Saliency：save_model=false，partial_resume=false，stream_checkpoint=false；中断则从 W0 重跑，Random child 不能继承另一 child 的 adapted state。

仅 group_lbi 支持 **完整成功 outer-batch 边界**的精确 resume；不能在 Stage-1、Stage-2、omega 写回过程中或 PU 前保存可恢复边界。恢复 persistent model、stream position/order、Python/NumPy/CPU/CUDA RNG、PU metric accumulators、全部 completed batch records、support union、provenance 和计数。

COME 还须核验 immutable objective config，并恢复 online augmentation generator 与 processed/loader counters。COME-LBI 没有额外 memory、teacher、EMA 或 persistent host optimizer，不得在 resume 时创建这些状态。当前 batch inputs/logits、Δ/Γ/Z/M 和 local Stage-2 optimizer 不跨边界保存。

resume 前 exact scientific identity 校验，日志无重复/缺失 batch，raw efficiency history 完整恢复。记录 runtime_resume_used、runtime_segment_count。invalid/cap-hit batch 不可成为 checkpoint。无最终 adapted-model checkpoint；正式效率使用 fresh non-resumed runs。

所有资产位于 /home/nas3/biod/wangkangyi/；数据 datasets/、lists datasets/image_lists/、环境 envs/lbi/、临时 tmp/，缓存使用已确认 hf-cache/pip-cache/conda-pkgs。不得写大文件到 HOME 或系统 /tmp，不硬编码未确认 HF mirror，不提交 checkpoint/dataset/cache/log。本轮仅创建协议文档，未执行训练、安装或实验。

---

## 20. Revision、scientific identity 与 artifacts

COME 必须有独立 Transformer protocol / baseline implementation / sparse-LBI implementation revision；本文预留命名见文档开头，不能冒用 SHOT 或 ResNet revision，也不能声称尚未实现的 revision 已验证。记录本次参考代码 Git，与未来实际实现 Git 分开。

Scientific payload 至少包含：

~~~text
method, protocol_revision, implementation_revision, Git commit/dirty state
source path/revision/SHA256/manifest/config hash
dataset/transfer/source/target/classes/backbone/timm
formal seed, stream/list/order hashes, online/FO batching, singleton policy
preprocessing/view/RNG/precision/model-mode policy
variant, candidate exact names/shapes/order/count, group definition/order/count/size
rho, integer K_G, selector and Random child protocol
host objective and all method constants
host optimizer/lifetime/step unit/LR/scheduler
full LBI tuple, prox_lambda/tau_g/cap, Stage-2 optimizer, writeback mode
PU/FO definitions and accuracy aggregation
~~~

COME-specific payload 必须包含 official commit provenance、p/tau/C/epsilon、detach/norm/no-clamp/log-domain/no-renormalization，以及实际使用的 objective chunk 设置。使用独立 namespace，明确区分 come.tau 与 lbi.tau_g，并单独记录 lbi.alpha。

修改科学字段必须改变 scientific_config_sha256；更换 source revision/hash 必须改变身份。仅 output path、checkpoint cadence 或等价 resume engineering 不制造新科学实验。efficiency-only revision 与 scientific revision 分离；如执行方式实际改变数值/随机语义，则不再属于纯 efficiency 改动。

resolver、manual builder、runner、resume、aggregate、finalize 必须得到相同身份，特别测 sparse 不能 fallback 到 dense revision。每个 aggregate row 可追溯 raw artifact/log，不能只在 train 主入口正确。

保留 plan/effective config、manifest.json、metrics.jsonl、summary.json/results.json、status/finalize summary、artifact_path/log_path，以及实际存在的 config_path/summary_path。不存在的文件不要填虚构路径。每个 Random child 独立目录、模型状态生命周期和日志，顶层 summary 链接三条真实 child execution。

---

## 21. 计划比较矩阵与 formal gates

| Variant | 每 transfer 是否按预算重复 | 更新/selector |
|---|---|---|
| source_only | 否 | 无更新 |
| full_dense | 否 | non-head unrestricted reference |
| candidate_dense | 否 | 全 candidate |
| group_random | 每 rho；3 real children | source-static exact K_G |
| group_magnitude | 每 rho | W0 paired-group L2 exact K_G |
| group_saliency | 每 rho | 当前 batch COME objective 的 saliency |
| group_lbi | 每 rho | batch-local support ≤K_G |

COME 矩阵中，每 transfer 为 3 non-budgeted + 4×3 sparse =15 top-level identities；Office 90、VisDA 15，共 105。Random 的额外 child 使全量从头执行为每 transfer 21、共 **147 child/run executions**。复用已有 source-only artifact 时，必须核验 source/stream/online-view/metric 身份完全一致并明示引用，不能仅凭相同 W0 复用旧 PU。

该表是**计划矩阵，不是已完成实验**。先 CPU/synthetic contracts，再单 GPU 5-batch real-data smoke；全部通过后才具备后续矩阵运行条件。本协议撰写不授权本轮启动实验。

COME baseline formal gate 要求相应实现/tests/smoke/provenance 完整、无 debug-limit、完整 stream。COME-LBI 额外要求独立 search protocol 与六个 dataset×budget tuples 冻结；未完成时 formal=true 必须 fail closed。debug/provisional tuple 不得 fallback 成 formal。配置冻结后 fresh W0 正式重跑，不能把 tuning/smoke artifact 改名充当正式结果。

---

## 22. COME-LBI orchestration 与开发顺序

### 22.1 每个 valid outer batch 的顺序

~~~text
validate incoming current batch
→ Θ_t snapshot
→ local Δ/Γ/Z/M restart once
→ Stage-1: current candidate forward/COME/backward each step
→ corrected old-state coupling + paired prox + normalized support
→ exact budget stop OR strict rollback
→ base + masked feasible delta
→ fresh current-state COME at Stage-2
→ one fresh local masked AdamW step
→ discard local optimizer
→ omega-only persistent writeback once
→ exact off-mask/off-scope audit
→ runner validity gate（任何3000步invalid，不PU/后续/FO）
→ successful batch: read-only PU
→ release local state
~~~

每个成功batch：

~~~text
local_restart_count = 1
support_discovery_count = 1
stage2_optimizer_instance_count = 1
stage2_optimizer_step_count = 1
omega_writeback_count = 1
host_optimizer_persistent_step_count = 0
host_scheduler_step_count = 0
native_ema_commit_count = 0
objective_call_count = stage1_steps_completed + 1
~~~

计数里的Stage-1 steps包括被reject的overshoot尝试，不能用last_feasible_step替代executed count；+1为Stage-2 fresh objective。PU/FO不调用COME adaptation closure。失败/异常记录实际calls；cap-hit照§12的真实engine边界处理，不能把失败路径伪装为成功batch。

COME没有固定pseudo-target；Stage-1唯一重复计算对象是同一inputs在不同candidate下的current logits/opinion/loss。参数变了就必须重新求gradient，不能把一次outer gradient缓存给所有Stage-1 steps。

### 22.2 分阶段开发

| 阶段 | 交付/门槛 |
|---|---|
| P0 Objective | standalone stable objective；moderate equivalence、large-logit finite、detach、no-renormalization、fail-loudly contracts |
| P1 Dense | full_dense/candidate_dense；same source/stream/scope/mode、one objective/step、no scheduler、PU/FO read-only |
| P2 Sparse | Random三真实child → Magnitude静态 → Saliency current-state一次与gradient reuse audit |
| P3 LBI | existing Transformer engine + COME closure；fresh objective each candidate、strict rollback、one local AdamW、omega-only |
| P4 Review/smoke | CPU tests后，单GPU Office D→A与VisDA各5 valid batches，测试counts/finite/scope/state与数值；Amazon tail另测 |
| P5 Search/formal | 另行冻结search/tuples，fresh W0正式运行；本次不执行 |

数学objective保持backbone-independent。Transformer代码负责已有loader/model/candidate/group/mode；dense/sparse/LBI各自明确optimizer lifetime。避免复制FC trainer；不得让LBI借用host optimizer moments或scheduler。

代码审阅应检查：若COME closure换回原SHOT closure，除objective-specific diagnostics/identity之外，是否恢复当前SHOT Transformer execution semantics。公共 helper 必须保持 COME 的 current-batch objective、一次更新及对应 state lifetime。

---

## 23. 必须通过的 correctness / contract tests

以下为后续实现验收清单，本次文档工作未执行或声称已通过训练测试。

| ID | 测试及必须证明的行为 |
|---|---|
| C01 | same source checkpoint/hash/input下COME initial logits与SHOT一致；local-only，错误head/source/hash/last拒绝 |
| C02 | C=31/12，对应logits dimension；拒绝1000、wrong shape、invalid class_count |
| C03 | Moderate logits的stable/direct formula constrained/belief/u/loss/gradient等价 |
| C04 | 有限large logits使direct exp overflow；stable opinion/loss/gradient finite；不能只测loss |
| C05 | tau=1仍保留stop-gradient；正确Jacobian/gradient不同于identity shortcut |
| C06 | no norm clamp/epsilon；zero-norm/nonfinite主动失败，不nan_to_num、不跳batch继续 |
| C07 | opinion+1e-7后不renormalize；entropy用同一加epsilon值作乘数与log输入 |
| C08 | objective只依赖current logits/C；无pseudo-label、SHOT loss component、teacher、memory/source anchor；标签置乱不影响adaptation |
| C09 | dense每batchobjective=1/host step=1/scheduler=0；persistent AdamW state跨batch，不每batchfresh |
| C10 | full_dense非head全scope与controlled12tensors分别正确；LN/head/token/pos_embed/patch_embed按scope保护 |
| C11 | 6912groups×768、candidate5308416，no alias/overlap且coverage exact，QK/VO/FFN paired masks正确 |
| C12 | floor K_G=3/6/13，无slack/quota；paired support/scalar count匹配 |
| C13 | Random真实三fresh children，seeds202600/1/2、same views/stream、nested-prefix、exactK_G；顶层mean/std正确 |
| C14 | Magnitude W0 CPU-float64 pairedL2、static mask、canonical tie，不L1/每batch重选 |
| C15 | Saliency pairedL2(W*g) once/outer；selection不改model/RNG/norm/objective；复用原gradient，objective仍1次 |
| C16 | Dropout/DropPath inactive；所有variant eval/non-AMP；same-input/current-state determinism |
| C17 | 非LBIstrict masked AdamW off-mask exact及moment clearing；连续选中moments保留/离开清零 |
| C18 | 每Stage-1 fresh current-state COME：用可控changing-candidate例子证明logits/grad不是缓存 |
| C19 | old-state Z正确，首步Δ=Γ=0时Z不增；group prox零/阈值/active、normalized tau_g边界 |
| C20 | strict rollback恢复完整last feasible Δ/Γ/Z/M/IDs；无top-K修补；合法underfill/empty support |
| C21 | 任何3000-step run invalid，含exact-K/rollback；无PU/later batch/FO/成功checkpoint；实际invalidating Stage-2计数如实保留 |
| C22 | Stage-2 masked delta排除dense off-mask；fresh local AdamW实例一次/step一次、fresh current-state objective |
| C23 | LBI local restart/discovery/omega各1，host persistent optimizer/scheduler/nativeEMA全0；objective calls=executed S1+1 |
| C24 | omega后off-mask/off-scope精确相等，测试1ULP敏感值；不能仅allclose |
| C25 | PU在persistent writeback后独立forward；PU/FO的params/buffers/optimizer/RNG/method states完全不变 |
| C26 | Office sample-level、六transfer等权；VisDA fixed12macro+diagnostics；真实PU/FO总sample counts |
| C27 | Amazon 43×64+65，formal unexpected singleton fail；非正式early-skip objective/selector/optimizer/LBI/PU等全0 |
| C28 | collapse hist/count/dominant ratio/softmax entropy/opinion entropy/u均存在且diagnostics无额外objective/state mutation |
| C29 | LBI completed-boundary resume完整state/RNG/metrics/history；半批/identity mismatch拒绝 |
| C30 | method/objective/precision/group/source/LBI变更改变scientific hash；所有builder/resolver/aggregate/finalize保持新revision |
| C31 | formal gate拒绝provisional/missing tuple/debug-limit；baseline完整child/stream检查；raw floats不先round |
| C32 | SHOT source/dense/selector/LBI existing tests与smoke无回归；shared helper保持COME单视图、current-logit objective与对应更新次数 |

参考 tests/transformer_*_test.py 和FC tests/come_stable_objective_contract_test.py、come_formal_baseline_contract_test.py、come_sparse_lbi_contract_test.py。要求数值、调用次数、state transitions均验证，不只测shape/config strings。

---

## 24. Freeze control 与验收报告

本v1是开发规格草案。后续实现报告应列：

- Source identity与initial-logit equality；scope/model mode/non-AMP。
- Moderate/large-logit loss与gradient测试结果、detach位置、C/p/tau/epsilon/no-clamp/no-renormalization。
- dense/sparse每batchobjective/optimizer/scheduler计数；Random三child证据。
- Saliency gradient reuse前后model、CPU/CUDA RNG、normalization/objective state audit。
- LBI每candidate重算、old-state coupling、rollback、cap-hit、Stage-2 fresh step、host optimizer=0、omega-only。
- 所有off-mask/off-scope exact、PU/FO read-only、target-label隔离、collapse diagnostics。
- Runtime/memory边界、resume/identity/finalize、shared engine无复制、SHOT回归与5-batch smoke。
- 尚未冻结的search/六tuples、formal gate状态；不将debug结果列作paper结果。

COME稳定log-domain必须始终保留；不能因为small-data smoke未溢出而回退direct exp。任何objective、scope、stream、precision、threshold、rollback、optimizer/writeback或metric变更，都需要新scientific revision并重新验证。旧SHOT/FC/COME artifacts不重写，不借用其implementation label标记新Transformer结果。

---

## 25. 一页式实现摘要

| 项 | COME-Transformer v1 开发规格 |
|---|---|
| 迁移范围 | SHOT Transformer substrate + COME current-logit objective |
| Source/backbone | 同source-trained non-distilled DeiT-S；head冻结 |
| Stream | seed2026；Office64/VisDA256；Amazon tail65；单视图/完整PU/FO |
| Constraint | p2、come.tau1、class-axis L2、norm.detach、无norm epsilon/clamp |
| Opinion | logsumexp(constrained logits,log C)，稳定belief/u |
| Entropy | opinion+1e-7后entropy mean，不renormalize |
| C | Office31/VisDA12，严格等于logits类别维 |
| 不含机制 | pseudo-target/teacher/memory/source anchor/额外 entropy-diversity loss |
| Dense/sparse host | persistent AdamW1e-5，一objective/step每batch，scheduler0 |
| Selectors | Random三真实nested children；W0 pairedL2；current COME pairedL2(W*g)并gradient reuse |
| Candidate/budget | 12tensors、6912×768；rho.0005/.001/.002→3/6/13 |
| LBI | shared corrected paired engine；每candidate重算；old-state Z；strict rollback |
| Support/cap | normalized Gamma norm≥1e-4；prox_lambda1；任何3000步invalid |
| Stage-2/writeback | one fresh local masked AdamW step；omega only；host steps0；off-mask exact |
| Mode/precision | eval，Dropout/DropPath inactive，non-AMP |
| Tuning/formal | COME objective无grid；kappa1方向待验证；search/六tuples未冻结，LBI formal blocked |

---

## 附录 A. COME 交接逐项覆盖索引

| 交接章节 | 本协议落实位置 |
|---|---|
| 0–2 迁移哲学、只换objective、输入依赖 | §1、§3–5、§7、§22 |
| 3–4 constraint/detach/no norm clamp | §5.1、§6、C05–C06 |
| 5–7 stable opinion、entropy、dataset K | §5.2–5.3、§6、C02–C07 |
| 8–10 substrate、candidate、source | §1–4、§11.1、C01/C10–C12 |
| 11 dense计数 | §8；scheduler从ResNet1到Transformer0的显式裁决见§1 |
| 12–14 model mode/RNG/LN/head | §3、§9.3、§14、C10/C15–C16 |
| 15–16 sparse selectors、persistent optimizer | §9–10、C13–C17 |
| 17–20 LBI无host optimizer、outer顺序、fresh objective、无pseudo-target | §7、§11–13、§22.1、C18/C22–C23 |
| 21–24 corrected dynamics、group prox、strict rollback、cap | §11–12、C19–C21 |
| 25–27 Stage-2、omega、两个tau | §5、§10、§12–13；SGD到AdamW裁决见§1 |
| 28 singleton | §2、C27：保留before-all边界，formal继承Transformer tail65 |
| 29–30 PU/FO与labels | §15、C08/C25–C26 |
| 31 stable objective tests | §6、C02–C07 |
| 32 Transformer contracts A–E | §3–4、§9–14、§20、§23 |
| 33 collapse diagnostics | §16、C28 |
| 34 dense/sparse/LBI分离 | §7–10、§13、§22 |
| 35–36 formal gate/revisions | §17、§20–21、C29–C31 |
| 37–38 objective无调参、kappa1/alpha-nu/downstream calibration | §17 |
| 39 开发P0–P4 | §22.2 |
| 40 十五个错误 | §5–14、§17、§23对应数值/调用/state tests |
| 41 最终checklist | §23–25；ResNet-specific SGD/scheduler/singleton变体见§1–2 |
| 42–43 代码组织与换回SHOT的审阅原则 | §7、§11.1、§22.2、C32 |

**文档结束。**
