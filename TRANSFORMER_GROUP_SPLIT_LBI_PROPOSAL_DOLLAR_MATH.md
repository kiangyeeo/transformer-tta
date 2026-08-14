# Transformer Group Split-LBI Proposal

## 0. Proposal Summary

本 proposal 的目标是在现有 FC + Conv Split-LBI 框架中加入 Transformer，使三种架构共享同一条核心原则：**固定 pretrained source model，通过 target-specific delta 进行 TTA，并由 Split-LBI 动态发现真正需要发生变化的 structural support。** 三者的差别不在优化框架，而在 support unit 的定义：FC 以单个连接为基本单位，Conv 以 filter/channel group 为基本单位，Transformer 以 Q-K、V-O、FFN 的内部 representation coordinate 为基本单位。

Transformer 部分不采用 LoRA，不引入额外 rank 变量，而是直接对原始 Transformer weight 同形状的 full-rank delta 做 Group Split-LBI。这样可以最大程度复用现有 FC/Conv 的算法、代码和实验 protocol，减少 ICLR 截止前的额外开发成本。

## 1. FC、Conv、Transformer 的统一框架

三种架构统一写为：

$$
W^{\mathrm{TTA}} = W_0 + \Delta W.
$$

其中，`W0` 为 pretrained source weight，`Delta W` 为 target-specific delta。Split-LBI 使用与 `Delta W` 同形状的 sparse proxy `Gamma` 和 dual variable `Z`，通过 target adaptation loss 产生的梯度轨迹发现稀疏 support；Stage 1 负责 structure discovery，Stage 2 固定 support 后进行 masked adaptation。

| 架构 | Candidate space | Delta | 基本 support unit | 稀疏形式 | 结构含义 |
|---|---|---|---|---|---|
| FC | bottleneck FC | full-rank delta | scalar weight | element-wise | connection plasticity |
| Conv | 后部 Conv blocks | full-rank delta | filter/channel group | group-wise | feature/channel plasticity |
| Transformer | 后部 Transformer blocks | full-rank delta | QK/VO/FFN coordinate group | group-wise | representation-coordinate plasticity |

**相同点：** 三者都固定 source model，以 target-specific delta 表示 adaptation；都使用 Split-LBI 的 `Delta W / Gamma / Z` 动力学；都通过 sparsity budget 提前停止 structure discovery；都在相同 architecture 内、相同 budget 下与 Random、Magnitude、Saliency 做 matched-support comparison；TTDA 和 OTTA 的整体生命周期保持一致。

**不同点：** FC 的 support unit 是单个连接；Conv 的 support unit 具有局部卷积结构；Transformer 的 support unit 跨两个相互耦合的线性映射，需要利用 attention/FFN 的内部坐标对应关系进行 paired grouping。因此，Transformer 的主要新增内容不是新的优化器，而是新的 **architecture-aware group definition**。

## 2. Transformer Backbone 与实验范围

### 2.1 Backbone

主实验建议使用 **non-distilled DeiT-Small/16-224**，Office-31 和 VisDA-C 均使用同一 backbone。选择原因如下：

1. DeiT-Small 是标准 ViT/DeiT 结构，包含 12 个 Transformer blocks，hidden dimension 为 384，MLP hidden dimension 为 1536，约 22M 参数，结构标准，便于定义 QK/VO/FFN group。
2. 相比 DeiT-Base，计算和显存成本明显更低；相比 DeiT-Tiny，模型容量更适合 VisDA-C，作为主结果更稳妥。
3. SPP 已在 DeiT 上验证 Q-K、V-Projection 和 MLP 的 paired structural design，因此本工作的 Transformer grouping 有直接的结构参考，但目标不同：SPP 用于 pruning pretrained structure，本工作只选择 **哪些 target-specific delta 允许发生变化**。
4. non-distilled 版本避免额外的 distillation token/head，使实现和结果解释更干净。

### 2.2 Candidate blocks

只开放最后 3 个 Transformer blocks，其他 blocks 和 classifier 按现有 TTA protocol 处理。该设置与 Conv 只开放 ResNet 后部 blocks 的思路一致：先限定 candidate retrieval space，再在其中由 Split-LBI 自动发现 sparse support。

“last 3 blocks” 当前作为工程 setting 使用，不把它作为需要证明最优的超参数；当前阶段不进行 last-1/3/6/all-block 大规模搜索。

### 2.3 8×3090 可行性

该设置可以由 8 张 24GB RTX 3090 承担。DeiT-Small 本体约 22M 参数；最后 3 个 blocks 中，与 Q、K、V、O、W1、W2 对应的主要线性权重约为 5.31M scalar parameters。Stage 1 额外维护 `Delta W`、`Gamma`、`Z` 三套同形状 tensor，FP32 下三者合计约 64MB；即使再考虑梯度和优化器状态，也远小于 24GB。实际显存主要由 activation、batch size 和实现方式决定，因此一张 3090 跑一个任务是较保守且可行的配置。

建议初始设置：

- AMP/mixed precision：开启；
- Office batch size：优先沿用现有 setting；
- VisDA-C batch size：优先沿用现有 setting，如显存不足再单独减小；
- 不使用 gradient checkpointing，除非 pilot 显示必要；
- 先按一张 GPU 一个任务并行；若实测 peak memory 有明显余量，再考虑一张 3090 同时跑两个任务。

## 3. Transformer Structural Groups

以 PyTorch `nn.Linear` 的 weight shape `[d_out, d_in]` 为准：一行对应一个 output coordinate，一列对应一个 input coordinate。

对于第 `l` 个 candidate Transformer block，需要适应的矩阵为：

$$
W_{Q,l},\; W_{K,l},\; W_{V,l},\; W_{O,l},\; W_{1,l},\; W_{2,l}.
$$

为每个矩阵定义同形状 target-specific delta：

$$
W_{m,l}^{\mathrm{TTA}}
=
W_{m,l}^{0}
+
\Delta W_{m,l},
\qquad
m\in\{Q,K,V,O,1,2\}.
$$

### 3.1 Q-K Group

Query 与 Key 在 attention score 中通过相同 latent coordinate 发生配对。对于第 `l` 个 block、第 `p` 个 attention coordinate，定义：

$$
G_{l,p}^{QK}
=
\left\{
\Delta W_{Q,l}[p,:],
\Delta W_{K,l}[p,:]
\right\}.
$$

若该 group inactive，则：

$$
\Delta W_{Q,l}[p,:]=0,
\qquad
\Delta W_{K,l}[p,:]=0.
$$

含义是：第 `p` 个 Q-K latent coordinate 继续使用 source-model behavior，但不允许产生 target-specific delta。

### 3.2 V-O Group

Value projection 的 output coordinate 与 output projection 的 input coordinate 一一对应，因此定义：

$$
G_{l,p}^{VO}
=
\left\{
\Delta W_{V,l}[p,:],
\Delta W_{O,l}[:,p]
\right\}.
$$

若该 group inactive，则：

$$
\Delta W_{V,l}[p,:]=0,
\qquad
\Delta W_{O,l}[:,p]=0.
$$

含义是：对应的 value-space coordinate 在进入和离开 attention inner space 两端都不发生 target-specific update。

### 3.3 FFN Group

FFN 中，W1 的第 `p` 个 output coordinate 与 W2 的第 `p` 个 input coordinate 对应同一个 hidden neuron，因此定义：

$$
G_{l,p}^{FFN}
=
\left\{
\Delta W_{1,l}[p,:],
\Delta W_{2,l}[:,p]
\right\}.
$$

若该 group inactive，则：

$$
\Delta W_{1,l}[p,:]=0,
\qquad
\Delta W_{2,l}[:,p]=0.
$$

因此，Transformer 的 candidate group set 为：

$$
\mathcal G
=
\mathcal G^{QK}
\cup
\mathcal G^{VO}
\cup
\mathcal G^{FFN}.
$$

## 4. Group Split-LBI

### 4.1 Variables

对所有 candidate Transformer weights 维护三类变量：

$$
\Delta W,\qquad \Gamma,\qquad Z.
$$

初始化为：

$$
\Delta W^0=0,
\qquad
\Gamma^0=0,
\qquad
Z^0=0.
$$

其中，`Delta W` 是 functional delta，直接负责 target adaptation；`Gamma` 是 sparse structural proxy，负责 support discovery；`Z` 是 dual variable，用于累积 target gradient signal。

### 4.2 Stage 1: Structure Discovery

对当前 target batch，使用：

$$
W_m^k
=
W_m^0
+
\Delta W_m^k
$$

计算现有 TTA objective：

$$
\mathcal L_{\mathrm{TTA}}^k.
$$

加入 Split coupling：

$$
\widetilde{\mathcal L}^k
=
\mathcal L_{\mathrm{TTA}}^k
+
\frac{1}{2\nu}
\sum_m
\left\|
\Delta W_m^k-\Gamma_m^k
\right\|_F^2.
$$

更新 functional delta：

$$
\Delta W_m^{k+1}
=
\Delta W_m^k
-
\alpha\kappa
\nabla_{\Delta W_m}
\widetilde{\mathcal L}^k.
$$

更新 dual variable：

$$
Z_m^{k+1}
=
Z_m^k
+
\frac{\alpha}{\nu}
\left(
\Delta W_m^k-\Gamma_m^k
\right).
$$

### 4.3 Group-Lasso Proximal

对于每个 structural group，从 `Z` 中提取对应 row/column slices，并拼接成 group-level dual variable。例如 Q-K group：

$$
Z_{l,p}^{QK}
=
\left[
Z_{Q,l}[p,:];
Z_{K,l}[p,:]
\right].
$$

执行 Group-Lasso proximal：

$$
\Gamma_g^{k+1}
=
\kappa
\left(
1-\frac{\lambda}{\|Z_g^{k+1}\|_2}
\right)_+
Z_g^{k+1}.
$$

其中：

$$
(x)_+
=
\max(x,0).
$$

VO 和 FFN 采用相同形式，只改变 group 内对应的 row/column slices。得到的 group-level `Gamma` 再映射回各个 matrix-shaped sparse proxy。

该过程的目标不是直接裁剪 Transformer，而是沿 target adaptation trajectory 逐渐发现哪些 representation coordinates 应保持 plastic。

## 5. Sparsity Budget：三个 rho

FC、Conv、Transformer 不要求使用完全相同的 numerical budget。统一的是 **budget 定义、汇报方式，以及每个 architecture 内的公平比较**。

### 5.1 Structural density

定义 structural density：

$$
\rho_{\mathrm{struct}}
=
\frac{
\#\left\{
g\in\mathcal G:
\|\Gamma_g\|_2>\tau
\right\}
}{
|\mathcal G|
}.
$$

它回答：**候选 structural units 中有多少比例被激活。**

`rho_struct` 是 Split-LBI 的主要 stopping budget，也是 Random、Magnitude、Saliency、Split-LBI 之间 matched-budget comparison 的公平性约束。

### 5.2 Local active-parameter ratio

定义：

$$
\rho_{\mathrm{local}}
=
\frac{
\#\text{active scalar delta parameters}
}{
\#\text{candidate scalar delta parameters}
}.
$$

它回答：**在预先开放的 retrieval space 内，真正允许更新的 scalar parameters 占多少。**

FC 中，如果每个 group 就是一个 scalar，则：

$$
\rho_{\mathrm{local}}
=
\rho_{\mathrm{struct}}.
$$

Conv 中不同 group size 可能不同，因此两者不一定相等。

对于当前 DeiT-Small grouping，每个 QK、VO、FFN group 都包含：

$$
2d
=
2\times384
=
768
$$

个 scalar delta，因此在这一 Transformer setting 下：

$$
\rho_{\mathrm{local}}
=
\rho_{\mathrm{struct}}.
$$

### 5.3 Global active-parameter ratio

定义：

$$
\rho_{\mathrm{global}}
=
\frac{
\#\text{active scalar delta parameters}
}{
\#\text{whole-model parameters}
}.
$$

它回答：**相对于整个 pretrained model，最终有多少比例参数获得 target-specific plasticity。**

`rho_global` 主要用于跨 FC/Conv/Transformer 的统一汇报，而不作为 Split-LBI stopping criterion。

### 5.4 DeiT-Small 的 group 数量

对于 DeiT-Small：

$$
d=384,
\qquad
d_{\mathrm{ffn}}=1536.
$$

每个 block 的 group 数为：

$$
N_{QK}=384,
\qquad
N_{VO}=384,
\qquad
N_{FFN}=1536.
$$

因此：

$$
N_{\mathrm{group/block}}
=
384+384+1536
=
2304.
$$

如果只开放最后 3 个 blocks，则：

$$
N_{\mathrm{group}}
=
2304\times3
=
6912.
$$

初始 pilot 优先尝试：

$$
\rho_{\mathrm{struct}}
\in
\{0.001,\;0.003,\;0.005\}.
$$

三档 budget 分别约对应：

$$
6912\times0.001\approx7,
$$

$$
6912\times0.003\approx21,
$$

$$
6912\times0.005\approx35
$$

个 active groups。若最高档仍明显 under-capacity，再扩展到：

$$
\rho_{\mathrm{struct}}=0.01.
$$

当前不建议一开始做更大范围 sweep。

### 5.5 跨架构如何比较

FC、Conv、Transformer **不是三个 competing methods**，而是同一个 Split-LBI framework 的三种 structural instantiations，因此不比较三者的 absolute accuracy 谁更高。

跨架构主要比较 Split-LBI 相对 matched-budget baseline 的增益。例如：

$$
\Delta_{\mathrm{Random}}
=
\mathrm{Acc}_{\mathrm{SplitLBI}}
-
\mathrm{Acc}_{\mathrm{Random}},
$$

以及：

$$
\Delta_{\mathrm{Saliency}}
=
\mathrm{Acc}_{\mathrm{SplitLBI}}
-
\mathrm{Acc}_{\mathrm{Saliency}}.
$$

同时统一报告：

$$
\rho_{\mathrm{struct}},
\qquad
\rho_{\mathrm{local}},
\qquad
\rho_{\mathrm{global}}.
$$

目标是证明：**在 FC、Conv、Transformer 三种不同 structural granularity 下，Split-LBI 都能比 matched-budget naive/static selectors 更有效地发现 target-specific support。**

## 6. Stage 2: Masked Structural Adaptation

Stage 1 达到目标 structural density 后，对每个 group 定义：

$$
M_g
=
\mathbf 1
\left[
\|\Gamma_g\|_2>\tau
\right].
$$

将 group mask 映射回各矩阵。

Q-K：

$$
M_Q[p,:]
=
M_K[p,:]
=
M_{l,p}^{QK}.
$$

V-O：

$$
M_V[p,:]
=
M_{l,p}^{VO},
\qquad
M_O[:,p]
=
M_{l,p}^{VO}.
$$

FFN：

$$
M_1[p,:]
=
M_{l,p}^{FFN},
\qquad
M_2[:,p]
=
M_{l,p}^{FFN}.
$$

Stage 2 不再使用 sparsity penalty，只固定 mask 并继续优化 target adaptation objective：

$$
\Delta W_m^{k+1}
=
\Delta W_m^k
-
\eta
\left(
\nabla_{\Delta W_m}
\mathcal L_{\mathrm{TTA}}
\odot M_m
\right).
$$

最终模型为：

$$
W_{\mathrm{final}}
=
W_0
+
\Delta W_{\mathrm{masked}}.
$$

这里**不 prune source Transformer，也不声称降低 base-model inference FLOPs**；稀疏的是 target-specific adaptation support。

## 7. TTDA 与 OTTA

### 7.1 TTDA

沿用当前 Non-Restart Split-LBI：

1. 在完整 target dataset 上进行 Stage-1 structure discovery；
2. 当 structural density 达到目标 budget 时停止；
3. 固定 structural mask；
4. 在相同 target dataset 上进行 Stage-2 masked adaptation；
5. 输出：

$$
W_{\mathrm{final}}
=
W_0
+
\Delta W_{\mathrm{masked}}.
$$

### 7.2 OTTA

沿用当前 Restart Split-LBI：

1. streaming batch 到达；
2. 以当前模型为基准，reset 当前 batch 的 `Delta W / Gamma / Z`；
3. 在当前 batch 上进行 Stage-1 group support discovery；
4. 达到目标 structural budget 后生成 mask；
5. 在当前 batch 上进行 masked adaptation；
6. 将更新后的模型传给下一 batch：

$$
W_t
\rightarrow
W_{t+1}.
$$

因此 OTTA 中每个 batch 都可以重新发现当前 distribution shift 对应的 QK、VO、FFN support：

$$
M_t^{QK},
\qquad
M_t^{VO},
\qquad
M_t^{FFN}.
$$

## 8. Baselines 与必要消融

Transformer 部分尽量复用 FC/Conv 的同一套 selectors：

- **Random**：随机选择相同数量 structural groups；
- **Magnitude**：按 pretrained weight 的 group L2 norm 排序；
- **Saliency**：先计算 element-wise first-order saliency，再在 group 内聚合；
- **Split-LBI**：通过 trajectory 累积信号动态发现 support。

Saliency 的 element-wise score 可写为：

$$
S_m
=
\left|
W_m
\odot
\nabla_{W_m}
\mathcal L_{\mathrm{TTA}}
\right|.
$$

随后在每个 structural group 内取 L2 norm，得到 group score。

所有方法使用相同 candidate blocks、相同 group definition、相同 structural budget。

只增加一个低成本且关键的 Transformer-specific ablation：

- **Independent Split-LBI**：Q/K/V/O/W1/W2 各自独立成组；
- **Paired Split-LBI**：采用 Q-K、V-O、FFN paired groups。

该消融用于回答：Transformer architecture-aware coupling 是否比 architecture-agnostic sparsity 更有效。

## 9. 最小实验配置

为控制 ICLR 前的开发和计算成本，Transformer 分支只承担 **architecture generalization**，不重复验证所有 TTA objectives。

主设置：

- Dataset：Office-31、VisDA-C；
- Backbone：non-distilled DeiT-Small/16-224；
- Candidate space：last 3 Transformer blocks；
- TTA objective：优先沿用现有 SHOT；
- Protocol：TTDA + OTTA；
- Selectors：Random / Magnitude / Saliency / Split-LBI；
- 主 budget：由 pilot 在 structural density 的 0.001 / 0.003 / 0.005 三档中确定；
- 额外 ablation：Independent vs Paired，仅在一个代表性 setting 做完整比较。

当前不新增：

- 新数据集；
- CLIP/LLM；
- 第二个 Transformer backbone；
- LoRA/AdaLoRA/SoRA pipeline；
- rank sweep；
- 大规模 block-scope sweep；
- Transformer 上的 SHOT++/IST/TENT/EATA 全量重复实验。

## 10. 实现注意事项

DeiT/timm 通常将 Q/K/V 合并为一个 `qkv` Linear，其 weight shape 为：

$$
[3d,d].
$$

因此实现时逻辑切分为：

- Q rows：`[0:d]`
- K rows：`[d:2d]`
- V rows：`[2d:3d]`

attention output projection 的 weight shape 为：

$$
[d,d].
$$

MLP 第一层 weight shape 为：

$$
[4d,d].
$$

MLP 第二层 weight shape 为：

$$
[d,4d].
$$

算法层面仍然维护同一个 `qkv` tensor，只在 `group_definition / group_prox / mask_scatter` 时按上述 row ranges 建立 QK/VO groups。这样无需修改 Transformer forward 结构，也便于最大程度复用现有 Split-LBI optimizer。

建议把 FC、Conv、Transformer 的结构差异统一抽象为：

```text
get_groups(delta, gamma, z, architecture)
group_prox(groups)
compute_structural_density(groups)
build_mask(groups)
```

其中：

```text
FC          -> singleton groups
Conv        -> filter/channel groups
Transformer -> paired QK/VO/FFN groups
```

## 11. Pilot 与 Go/No-Go

在扩展到全部 Office transfers 和 VisDA-C 主实验前，先完成一个小规模 pilot。建议优先跑一个 Office transfer + VisDA-C，单 seed，只检查：

1. `rho_struct` 能否稳定从 0 增长到目标 budget，而不是不激活或瞬间全激活；
2. Split-LBI 是否在 matched budget 下优于 Random / Saliency；
3. Paired grouping 是否至少不弱于 Independent grouping；
4. 24GB 3090 的 peak memory 和每步 wall-clock 是否满足预期。

只有 pilot 通过后再扩到完整 TTDA/OTTA 实验，避免在 formulation 或实现尚未稳定时消耗大量 GPU 时间。

## 12. Takeaway

Transformer extension 不应成为一套独立的新算法，而应作为 Split-LBI 在第三类模型结构上的自然实例：

$$
\text{FC: connection plasticity}
\rightarrow
\text{Conv: feature/channel plasticity}
\rightarrow
\text{Transformer: representation-coordinate plasticity}.
$$

三者共享同一个 target-specific delta 与 Split-LBI dynamics，只改变 structural group 的定义。实验公平性在每个 architecture 内通过 matched structural density 保证；跨 architecture 统一报告：

$$
\rho_{\mathrm{struct}},
\qquad
\rho_{\mathrm{local}},
\qquad
\rho_{\mathrm{global}},
$$

并比较 Split-LBI 相对 Random、Magnitude、Saliency 的收益，而不比较三种 backbone 的 absolute accuracy。
