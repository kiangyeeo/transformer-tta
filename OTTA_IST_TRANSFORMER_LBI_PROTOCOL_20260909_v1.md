# OTTA_IST_TRANSFORMER_LBI_PROTOCOL_20260909_v1

**状态：** 实现规格草案 v1，供后续开发审阅；科学接口与验收要求明确，尚无 IST-Transformer 实现验收或正式结果。  
**范围：** non-distilled DeiT-S + IST-OTTA + QK/VO/FFN Group Split-LBI；Office-31、VisDA-C。  
**日期：** 2026-09-09。  
**形式参考：** [SHOT Transformer frozen protocol](OTTA_TRANSFORMER_LBI_PROTOCOL_20260901_FROZEN_v1.md)。  
**交接依据：** [IST implementation handoff](IST_TRANSFORMER_IMPLEMENTATION_HANDOFF_FROM_RESNET_20260909.md)。  
**本次只读审阅的仓库 HEAD：** f1e680f6bee0bbb4d53135b47fdfb67a1696bd94。  
**待实现 revision 命名：** ist_transformer_baseline_20260909_v1 / ist_transformer_sparse_lbi_20260909_v1；仅预留，不代表代码已存在或已通过验证。

> 将 IST 的 8-view → pre-adaptation features/probabilities → PLCA once → causal memory once → fixed task → native IST / sparse / Group-LBI → single persistent writeback 接到现有 Transformer substrate。IST 扩展包含完整状态机，不能简化为只换一个 loss。

---

## 1. 来源优先级、边界与冲突裁决

基准/结构语义按当前用户要求、[AGENTS.md](AGENTS.md)、SHOT Transformer frozen protocol 和 [refined FC semantic parent](protocol/shot-otta_fc/OTTA_FC_LBI_PROTOCOL_20260817_v1.md) 执行；IST-specific 语义由交接、当前 FC IST protocol/mechanism 共同确定。历史官方 benchmark 配置/旧实现不替换本项目已冻结 source、scope、stream 或 optimizer。

直接参考：

- [FC IST baseline protocol](260817_iclr2027-refined/protocol/ist-otta/OTTA_IST_BASELINE_PROTOCOL_20260906_v1.md)，revision ist_otta_p1_baseline_20260906_v4。
- [FC IST sparse/LBI protocol](260817_iclr2027-refined/protocol/ist-otta/OTTA_IST_LBI_PROTOCOL_20260906_v1.md)，revision ist_otta_sparse_lbi_20260906_v2。
- [IST trainer](260817_iclr2027-refined/ist_otta/trainer.py)、[objective](260817_iclr2027-refined/ist_otta/objective.py)、[PLCA](260817_iclr2027-refined/ist_otta/plca.py)、[memory](260817_iclr2027-refined/ist_otta/memory.py)、[data](260817_iclr2027-refined/ist_otta/data.py)、[EMA](260817_iclr2027-refined/ist_otta/ema.py)。

| 容易冲突的内容 | 本协议明确采用的规则 |
|---|---|
| 交接 §4 将原图 hard/soft target 重复到 8 views | **按当前 FC protocol §9 与 trainer 实际行为，全部 8B views 各自提取 F/Q、产生 view-level targets**，见 §5–7；不复制 reference-view target |
| ResNet singleton skip | Transformer 主流保持 Amazon 65 尾批、完整 PU；unexpected singleton 在任何状态转换前报错，compatibility skip 仅非正式测试，见 §2 |
| ResNet SGD / polynomial schedule | Transformer host 与 Stage-2 AdamW；fixed LR、无 scheduler，见 §8/10/13 |
| native iters=1 与 SHOT one step | IST native 遍历完整 8B task 一轮，多个 inner steps；LBI Stage-2 仍恰好一步 |
| ResNet core/lbi 路径 | 共享现有 Transformer paired-group engine，不复制数学、不导入 scalar/SGD 默认值，见 §11 |
| “memory 所有状态随 batch 重启” | 只重启 Δ/Γ/Z；IST past-only memory 跨 batch 持续 |
| SHOT 现成 winners | 不是 IST winners；六 tuple 待单独搜索冻结 |
| SHOT 已完成 formal 的模板措辞 | 仅借用协议结构；本 IST 文档不声称调参、测试或 formal 已完成 |

上述 target 粒度是本次从文字/代码差异中作出的明确裁决，而非遗漏交接要求。若后续希望改成 raw-sample target 广播，必须另立 method revision，不能与当前 IST reference 混报。

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

同一 (dataset, transfer, seed) 的所有 variant 复用同一 raw sample permutation、batch boundaries 和相应方法的 augmentation RNG。保存 sample-index 顺序、列表 hash、order hash 与 batch-size 序列；不能仅凭相同 seed 声称 stream 一致。禁止 full-target 非因果聚类、未来样本预提特征和图像 replay。IST 的 past-feature memory 按方法章节另行定义。

IST 输入分辨率为 224，reference view 与每个 adaptation view 的 transform 为：
Resize((256,256), bilinear) → RandomCrop(224) → RandomHorizontalFlip(0.5) → ToTensor → ImageNet normalization。
mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]。FO 使用 bilinear resize + CenterCrop(224)。source training 的 bicubic 与 TTA 的 bilinear 差异沿用并披露，不在某个 host 中单独修正。

**尾批裁决：** target Amazon 的 2,817 张图采用 **43×64 + 65** 的 online batching，所有图进入 PU；FO 保持普通 BS64、drop_last=false。DSLR、Webcam 与 VisDA 保留普通尾批。不能直接搬用 ResNet loader 的 singleton-drop 或 FO batch-size×3。

交接强调的“actual BS=1 必须在一切 adaptation 前阻断”保留为防御契约：检查先于 view materialization、objective、PLCA/memory、selector、scheduler、optimizer、LBI restart、EMA/omega 和 PU。正常 formal stream 不应产生 singleton；如果产生，报 stream_protocol_mismatch 并终止，不能静默少评一个样本。专门的非正式 singleton compatibility test 可采用 ResNet early-skip（上述计数全部为 0，完整 FO 仍含该样本），必须标记非正式，不混入 Transformer 主表。不能对已合并的 65-sample batch 再 skip。

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

需要 feature 的 adapter 使用原模型的 pre-logits 路径：

~~~python
tokens = model.forward_features(x)
features = model.forward_head(tokens, pre_logits=True)
logits = model.get_classifier()(features)
~~~

验证 features=[N,384]、logits=[N,C]，且 adapter logits 与 model(x) 在相同输入/状态下相等或满足预声明的浮点误差界。不能把 [N,197,384] token matrix、logits、attention map 或中间 block 当 feature。此为待实现接口契约，现 SHOT runner 直接调用 model(inputs)，本文不声称仓库已有统一 features/logits API。

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

## 5. IST objective 与固定 method constants

### 5.1 不可隐式调节的机制

| 字段 | 值 |
|---|---:|
| ist.extend / iters | 8 / 1 |
| ist.plca.repeat / k / gamma / mode | 1 / 50 / 3 / l2 |
| ist.plca.propagation_alpha | 0.99 |
| solver maxiter / rtol / atol | 20 / 1e-6 / 0 |
| memory capacity | 最近 10000 **view entries** |
| hard_ce_weight / soft_kl_weight | 1 / 1 |
| native parameter EMA m | 0.9；仅 non-LBI |
| native inner scientific BS | Office 64 / VisDA 256 |

PLCA k=50 是 graph 邻居参数，不是类别数 C 或结构预算 K_G；propagation_alpha=.99 不是 LBI alpha。不加入 SHOT 的 0.3 pseudo CE、entropy/diversity、confidence threshold、teacher 或 source-logit anchor。

### 5.2 View-level hard/soft targets 与 loss

当前 raw batch B_t 产生 N_t=8|B_t| 个 adaptation views，按 raw sample→view index 排列。在 Θ_t 上计算每个 view 的 F_t、pre-correction Q_t，PLCA 输出每个 view 的 corrected hard target ŷ_j。q_j 是该 view **correction 前**的 softmax；不能用 corrected one-hot、propagated soft scores 或平均 reference prediction 替代。

$$
\mathcal D_t=\{(x_j^{(v)},\hat y_j,q_j)\}_{j=1}^{N_t},
$$
$$
L_{\mathrm{IST},t}(\theta)
=\frac1{N_t}\sum_{j=1}^{N_t}
\left[-\log p_\theta(\hat y_j\mid x_j^{(v)})
+\sum_{c=1}^{C}q_{jc}\log\frac{q_{jc}}{p_{\theta,c}(x_j^{(v)})}\right].
$$

实现 CE 用 mean；KL 为 KL(q||p)，等价于 kl_div(log_softmax(logits), soft_targets, reduction="batchmean")，不是 reduction="mean"（会额外除类别数），也不是 KL(p||q)。hard/soft target detach、无梯度；views 与 targets 一经建立，当前 outer batch 内固定。预测随 native updates/Stage-1/Stage-2 状态变化重算。

---

## 6. Raw views、feature 与 causal PLCA/memory

### 6.1 Decode / augmentation contract

每 raw sample decode 成 RGB 一次，从同一 raw image 生成 1 个 cached PU reference view + 8 个独立 adaptation views。拒绝将已 normalize tensor 转回 PIL，拒绝对 reference/每个 adaptation view 各重复 decode。reference 不属于 adaptation task，不参与 PLCA/memory。FO 是另一个独立 evaluation pass。

IST 使用独立 CPU generators：reference seed=2026+10000、adaptation seed=2026+20000、inner-order seed=2026+30000；Random mask seeds 独立。增强按 §2 冻结，保留实际 crop/flip 顺序与 view hash；三个 Random children 的 views 与 inner-order 完全相同。

这些是 IST-specific view RNG。IST 沿用 Transformer 底座的 raw order、boundaries 与基础 transform，并按本节独立生成 reference/adaptation views。IST 内所有比较 variant 必须使用同一 RNG discipline；不得因 selector/Stage-1 forward 次数改变下一 batch 的增强。

### 6.2 Pre-adaptation inference exactly once（逻辑任务）

在任何参数更新之前，用 eval/no_grad 的 Θ_t 对全部 N_t adaptation views 做一次逻辑 pre-inference，可按 memory chunk 前向。得到 F_t=[N_t,384]、Q_t=[N_t,C]，detach。

F_t 是 §3 的原 classifier 输入；不新增 feature L2 normalization、token pooling 或额外 bottleneck。PLCA 的 mode=l2 指距离，不是要求把 feature normalize 成单位向量。检查 initial SHOT/IST logits equality、feature determinism 与 batch dimension。

“once”是完整 task 处理一次，不要求所有 N_t 一次物理 forward；须分别记录 logical pre_inference_count 与 physical forward/chunk count。

### 6.3 PLCA 数学与求解参考

使用 immutable M_{t-1} snapshot，其 batch IDs 必须严格早于 t。graph nodes 为 past memory entries + 当前全部 N_t views；memory labels 是过去 corrected one-hot，current labels 是 Q_t。禁止预读未来 features。

复用当前 plca.py 的数学：

1. 用平方 L2 distances 做 exact kNN；L2 的 neighbor convention 为 min(k+2, graph_node_count)，即通常 52，不擅自改成 50、cosine、approximate neighbors 或删除 self convention。
2. 每行 distances 以 row max（数值 guard 1e-8）归一化，relation=max(1-d/dmax,0)^gamma；参考实现令第一个 neighbor relation=1，再按 gamma=3 取幂。
3. 构建 directed affinity W，使用 W+Wᵀ 对称化和 D^(-1/2)(W+Wᵀ)D^(-1/2) normalization，degree guard=1e-8。
4. 初始 label matrix 按 class-column sum 归一化（guard=1e-8），求解 (I-.99 A)S=Y_norm；CG 初始解为 0，maxiter=20，relative rtol=1e-6、atol=0，不能把 rtol 当 absolute tolerance。
5. repeat=1，argmax S 得 hard labels/one-hot，截取 current-view 部分作为结果。保留参考 CG 的 denominator/residual 数值 guards（1e-12），任何新 solver/precision/neighbor tie 行为必须经等价测试并记录 revision。

不因 Transformer feature scale 临时调 PLCA k/gamma/alpha。distance_chunk_size 只控制 exact distance computation 的内存，不是 graph 截断。固定实现和 canonical input order，测试小 graph/duplicate features/zero class mass，不能默默用另一种图算法。

### 6.4 Memory commit exactly once

PLCA 完成后、self-training 之前，向 M_{t-1} append 当前 pre-adaptation F_t 与 corrected one-hot；CPU float32 detached clones、相应 processed outer-batch IDs。保留最近 10000 entries，FIFO 淘汰单位是 view，不是 raw sample/batch。历史 feature 不在未来参数下刷新，不把 soft Q_t、target GT、reference view 或 adapted features 写入。

每个 processed outer batch 仅一次 commit；PLCA 查询先于本 batch commit，不允许 current entries 被当作 past 重复输入。loader_batch_index 与 processed_batch_index 分开，memory commit_count 与成功进入状态机的 processed index 一致。异常 run 整体 invalid，不能将半批已写 memory 当成成功 resume 边界。

memory 允许存过去的 detached feature/label，不允许存 raw images 做 replay。每个新 transfer/Random child 从空 memory 开始。

---

## 7. IST outer-batch 状态机与所有权

~~~text
validate raw outer batch / stream boundary
→ decode once, cache reference + 8 views
→ save Θ_t anchor（non-LBI EMA independent clone）
→ pre-infer all 8B views at Θ_t
→ snapshot past-only memory
→ PLCA exactly once
→ memory commit exactly once
→ freeze task: views + view-level hard/soft targets
→ native dense/sparse training OR Group-LBI
→ exactly one persistent writeback: native EMA OR omega
→ off-mask/off-scope exact audit
→ separate PU on cached B reference views
→ release local task; move to next outer batch
~~~

| 状态 | 生命周期 |
|---|---|
| persistent model、causal memory | 跨 outer batches；每 run/child 重置 |
| native host AdamW moments | 仅 non-LBI，跨 native inner steps 和 outer batches |
| reference/adaptation/inner-order RNG | 独立且可恢复；不被 selection/PU 推进 |
| cached views、F/Q、corrected targets、EMA batch anchor | batch-local；不可 alias 可变 model state |
| Δ/Γ/Z/M、local Stage-2 optimizer | 仅 LBI、batch-local |
| target GT | 仅 metric bookkeeping，不能进入上表 adaptation state |

成功 batch 的公共计数：pre_inference_task_count=1、plca_call_count=1、memory_commit_count=1、pu_forward_task_count=1。物理 chunk forward/backward 数单独记录。

---

## 8. Non-LBI native IST：完整 traversal 与一次 EMA

full_dense/candidate_dense/Random/Magnitude/Saliency 采用 §10 的 **persistent host AdamW**，constant lr=1e-5、scheduler_step_count=0；不是从 ResNet 移植 polynomial schedule。

固定 task 经独立 inner-order generator 产生 permutation，iters=1 表示完整 N_t views 遍历一轮。native scientific inner batch size I=64（Office）/256（VisDA），每 inner minibatch 重新 zero_grad → current predictions → mean CE+KL → backward → 一次 AdamW step。尾 inner batch 保留。

因此 inner_optimizer_steps=ceil(N_t/I)：普通 Office B=64 时为 8；Amazon B=65 时 N=520，为 9（最后 inner batch=8）；普通 VisDA B=256 时为 8。不能把“one outer batch”误写成只有一个 host step，也不能把 native inner BS 改成 memory chunk size。

如一个 native inner minibatch 仍需 micro-chunk，按该 inner minibatch 的样本数加权累积梯度，然后仅在这个 native minibatch 结束时 step；不得因显存改 native scientific inner BS/step 数。

完整 traversal 得 Θ̃_t 后，恰好一次 native parameter moving average：

$$
\Theta_{t+1}=0.9\Theta_t+0.1\widetilde\Theta_t.
$$

anchor 为进入该 outer batch 的独立 clone，不是 mutable state_dict alias。仅作用于可更新 scope，frozen parameters/buffers 原样保留；sparse 后再恢复 off-mask Θ_t。host optimizer moments 按原 native 语义继续保留，不随参数 EMA 做平均/重置。不存在额外 teacher prediction ensemble，也不每 inner step EMA。LBI 禁用 native EMA，不能 omega 后再乘 .1。

---

## 9. IST Sparse baselines 与 full-objective accumulation

### 9.1 Random / Magnitude

Random 从 canonical pool 均匀 exact K_G，静态整条 stream；三条真实 child execution 的 support seeds=202600/202601/202602，each fresh W0、memory、optimizer、EMA、view/order generators。继承 SHOT frozen protocol 的 **cross-budget nested prefixes**：同 seed 的 6912 permutation 取前 K_G。不是三个 formal seeds，不能只生成三个 metadata masks 而实际跑一次。

Magnitude 在 source W0 上 CPU float64 计算 paired-group ||W_g||₂，canonical group-ID tie break，global exact top-K_G，stream 中不刷新。IST objective 不参与该 mask 的构建。

### 9.2 Saliency once per outer batch

在 Θ_t 和已固定 D_t 上，先计算完整 N_t objective gradient：

$$
s_g=\|(W\odot\nabla_W L_{\mathrm{IST},t}(\Theta_t))_g\|_2.
$$

global top-K_G、canonical tie break；本 outer batch 的全部 native inner updates 使用此同一个 mask。不是 gradient-only，也不按 view/chunk/inner optimizer step 重选。

scoring 不更新参数、memory、EMA 或 optimizer。完成 selection 后清掉 scoring gradients，native traversal 按当前 inner minibatch fresh backward。**不能把完整-task saliency gradient 当成 native 每个 inner minibatch 的 gradient 复用。** 下一 outer batch 才重新选；历史参数更新继续保留。

### 9.3 分块保持 full objective

Saliency full scoring、每个 LBI Stage-1 gradient、LBI Stage-2 均以完整 N_t task 为单位。若 chunk r 有 n_r 个 view：

$$
\nabla L_t=\sum_r\frac{n_r}{N_t}\nabla L_r.
$$

zero_grad once → 每 chunk mean loss 乘 n_r/N_t 后 backward → 得完整 gradient → 才做一次 support/动力学/Stage-2 action。不能等权相加 chunk means；末 chunk 不等大必须正确加权。

所有 chunks 求导期间参数、fixed task、normalization/stochastic state 不变。禁止每 chunk PLCA、memory commit、support discovery、optimizer/EMA/omega update；不能为省内存只取一个 view或缓存旧 logits。objective_chunk_size/pre_inference_chunk_size/distance_chunk_size 是实现配置，需先 synthetic 等价验证、单 GPU 验证后记录并冻结；不改变 native scientific inner BS。不能随正式 accuracy 调这些值。

---

## 10. Strict sparse AdamW 与 off-mask exactness

IST dense/sparse 共用 AdamW：lr=1e-5、betas=[0.9,0.999]、eps=1e-8、weight_decay=0.01、scheduler=none、AMP=false。不同 selector 不得分别调 base LR。native inner-step 数按 §8 的完整 task traversal 确定。Stage-2 的 LR 由 IST-LBI tuple 冻结，其余 AdamW 设置相同。

每次 sparse step 必须复用/等价于 transformer/group_random/optimizer.py 的 strict_masked_adamw_step：

1. step 前 snapshot parameter values，清零当前 off-mask exp_avg/exp_avg_sq，mask gradients。
2. 执行一次 AdamW step。
3. 从 snapshot 精确恢复 off-mask values，再清零并断言 off-mask moments 为 0。
4. 连续选中 coordinates 保留历史 moments；离开 support 的 moments 清零，重新进入时不恢复陈旧 moments。

Adam scalar step 是 tensor 级 bookkeeping，沿用 shared helper，不虚构逐 coordinate step，也不为清 off-mask state 把整个 tensor 的 moments/step 每批重置。LBI Stage-2 例外：optimizer 本身为 batch-local fresh instance。

仅 mask gradients 不充分，weight decay 和 moments 均可造成 drift。动态 mask 只控制本次变化；off-mask anchor 是本次 step/outer batch 的 persistent 值，不能强行恢复 source W0。

EMA/omega 后再显式恢复 off-mask base，防止对本来相等的值做插值产生 1 ULP 差异。off-scope tensors 直接保留。验收用 torch.equal/byte-level hash，不以 allclose 代替 exact invariance。

---

## 11. Shared Group Split-LBI：restart 与 corrected dynamics

### 11.1 实际复用边界

Transformer 实现实际位于 transformer/group_lbi/engine.py::GroupSplitLBIEngine。交接中的 260817_iclr2027-refined/core/lbi/engine.py 是 ResNet 共享引擎，默认 scalar/SGD，不能原样替换前者。

IST-LBI 必须复用 **现有 Transformer corrected LBI 引擎**，通过 IST objective/gradient adapter 接入。可以在回归保护下抽公共核心，但不另写 IST 私有 LBI 数学，也不为目录名统一移植 FC optimizer/group 数值。现有 SHOT closure 行为必须保留。

现 Transformer engine 消费可微 scalar loss，并内部调用 autograd.grad/backward；FC IST full-gradient accumulator 已 backward 后返回 detached scalar。两种接口不能直接拼接，否则重复 backward 或丢失图。IST 接入必须显式支持：可微 loss closure，或返回/填充完整 gradient map 的 accumulator；只对前者由 engine 自动求导。两路径须针对同一固定 IST task 测等价性。

### 11.2 Batch-local restart

进入 batch 的 persistent model 为 Θ_t。candidate 上初始化：

$$
\Delta^0=0,\quad \Gamma^0=0,\quad Z^0=0,\quad M^0=0.
$$

仅重启局部变量，不每批重载 source。IST causal memory 按方法规则持续，不能随 restart 清空。LBI 不使用 persistent host optimizer、host scheduler 或 native EMA。

### 11.3 Current-state objective 与 old-state update

令 L_IST,t 为 §5 定义的当前 outer batch 完整固定 task objective：

$$
L(\Delta,\Gamma)=L_{\mathrm{IST},t}(\Theta_t+\Delta)
+\frac{1}{2\nu}\|\Delta-\Gamma\|_2^2,
$$
$$
g^k=\nabla_\Delta L_{\mathrm{IST},t}(\Theta_t+\Delta^k),\quad
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

每个 Stage-1 candidate 都 fresh forward/objective/full gradient；禁止缓存旧 logits/loss/gradient。临时 candidate 参数不能留在 persistent model，异常也要恢复 batch base。task closure 不执行 optimizer、memory、EMA、scheduler 或写回。

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

off-mask dense delta 不得进入参数。重新计算 **Stage-2 当前状态**的完整 IST task objective，仅执行 **one masked local AdamW step**：

| 项 | 固定值 |
|---|---|
| optimizer lifetime | fresh instance / outer batch；batch 后丢弃 |
| betas / eps / weight decay | [0.9,0.999] / 1e-8 / 0.01 |
| steps / scheduler / AMP | 1 / none / off |
| stage2_lr | dataset×method×budget 冻结 tuple 的 LR |
| off-mask protection | §10 的 exact values + moments 保护 |

交接的“one SGD step、momentum=.9、wd=.001、Nesterov”为 ResNet 数值，Transformer 使用上述 AdamW。IST 分 chunk 时，先累积完整 8B objective gradient，再仅 step 一次。

得到 refined Θ̃_t 后：

$$
\Theta_{t+1}=(1-\omega)\Theta_t+\omega\widetilde\Theta_t.
$$

仅 support 内允许 persistent change，写回后再次 exact restore off-mask base。每成功 batch：omega_writeback_count=1、native_ema_commit_count=0、host_optimizer_persistent_step_count=0、host_scheduler_step_count=0。然后独立 PU，并丢弃 Δ/Γ/Z/M、Stage-2 optimizer 和局部 task。omega 改变未来梯度/support，不是报告时的输出插值。

---

## 14. Model mode、precision 与 RNG

所有 variant 沿用 SHOT Transformer 的 model.eval()，适配时启用必要梯度；Dropout、attention/MLP dropout、DropPath/stochastic depth inactive。source manifest 要求 drop_rate=drop_path_rate=0。LayerNorm affine 是否可更新按 §3 scope，而不是按 train/eval 推断；没有 BN running stats 不等于可忽略 LN parameters。

IST formal TTA 为 FP32/non-AMP；不得为显存或稳定性临时开 AMP、改变 gradient scale 或 support threshold。PLCA 的数值 guards 按 §6.3 固定，不能临时修改。source training AMP 不改变此规则。

分离 stream、view、inner-order、Random support RNG。PU/FO、selector 排序、诊断不可推进 adaptation 所需 CPU/CUDA/Python/NumPy RNG；必要时 snapshot/restore，但不能掩盖错误 state transition。相同状态、输入和 RNG 下 feature/logits/gradient 可复现。worker/prefetch、resume 也须检查 sample-index/view hash；mask 生成不消耗 augmentation RNG。

---

## 15. PU / FO 与 target-label 边界

PU 必须在本 batch 的唯一 persistent writeback 完成之后，使用当前 raw batch 对应的缓存 reference/online view，单独 fresh forward。不能复用 loss logits，不能将 8-view IST predictions 当 8B 个 PU 样本。每张原图仅计一次。

PU/FO 可读但不可写 parameters、buffers、memory、optimizer、scheduler、EMA、LBI state、RNG 或任何未来新增 method state。evaluation 接口不能调用 adaptation/PLCA/target construction。PU 使用 no_grad/inference、eval；评估后 state hash/计数不变。

完整 stream 成功结束后 freeze final model/method state，使用独立 full-target CenterCrop loader 做 FO。FO 不适配，不刷新 memory，不更新 teacher/EMA，不消费 online view generator。在线随机 crop 与 FO center crop 不同，因此 source-only PU/FO 数值不必相同；只在输入 tensor 与 evaluation 条件相同时要求 prediction 一致。

Target labels 在 runner 的 metric 分支隔离。objective、PLCA、memory、fixed task、selector、Stage-1/2、writeback 接口不得接收 ground-truth label 或带 label 的完整 batch object。置乱/替换 target labels 后，adapted state/support/memory 必须不变。允许只读 offline reporting；若将来用 labeled FO 选超参数，必须先在独立 search protocol 披露用途、范围和固定规则。PU 为 report-only，不作为调参信号。

Office 每 transfer 累积 correct/total，不做 mean(batch_acc)；六 transfer 等权平均，不跨 transfer pool samples。VisDA 在全 stream/full FO 累积 class_correct[12]/class_total[12]，固定 12-class macro，同时保留 overall、12 class accuracies、worst class/name、class std。缺类不得改为 observed-class mean；短 smoke 可标 diagnostic/incomplete，formal full-target class coverage 不全应报错。

JSON/CSV 保留未预先舍入的数值。表格可显示四位小数，ranking 不能读格式化字符串。accuracy unit（fraction 或 percent）必须写明且聚合一致。

---

## 16. 有效性、support 与 collapse diagnostics

正式 run 要求完整 target stream、PU/FO 样本计数正确、无 NaN/Inf/runtime exception、source/stream/scope/seed/config 身份一致、每批 S≤K_G、无 cap hit、日志完整。失败 run 写明确 invalid_reason，不将 partial mean 伪装成完整结果。

每 batch 记录 K_G、selected S、u=S/K_G、selected group IDs/hash、QK/VO/FFN 和 block 分布、Stage-1 executed steps、last feasible step、stop reason、rollback、Stage-1/2 runtime。聚合 selected min/mean/max，utilization min/mean/p05，低于 90%/95% 次数与比例、exact-K rate、cap/rollback/failure rate，steps mean/max。

util95 是 u≥.95 的 batch 比例；Office 如聚合此诊断，pool 六 transfer 的 batch numerator/denominator，区别于 accuracy 的六 transfer 等权平均。合法 rollback underutilization 不自动 invalid；无额外 hard 90%/95% eligibility gate，也不因低 utilization top-K 修补。选择/排序规则另见 §17。

保存 historical active-group union，以及最终 relative-to-source parameter delta support；二者区别于单批 selected support。delta_nonzero_tolerance=1e-12 仅为诊断，不替代 Gamma normalized threshold 或 off-mask exact test。

IST 至少保留 predicted-class histogram、predicted class count、dominant-class ratio、mean softmax entropy，以及 correction/target/memory 诊断。用这些区分一般退化、类别失衡、单类 collapse、数值失败；不能仅因 smoke accuracy 高低调算法，也不能事后发明 collapse 阈值挑结果。

---

## 17. 固定项、搜索边界与未解决配置

IST objective、PLCA/solver、memory、8 views、native iters/EMA、candidate/budgets、AdamW substrate、tau_g/prox_lambda/cap/Stage-2 step 数已作为本开发规格固定；本次不调参。

六个 IST-Transformer LBI profile 均 **TBD**：

| Dataset | rho | (alpha,kappa,nu,omega,stage2_lr) |
|---|---:|---|
| Office-31 | .0005 | TBD |
| Office-31 | .001 | TBD |
| Office-31 | .002 | TBD |
| VisDA-C | .0005 | TBD |
| VisDA-C | .001 | TBD |
| VisDA-C | .002 | TBD |

Office 每 budget 一个 tuple 跨六 transfers 共享；VisDA 独立。禁止 per-transfer/seed/class/batch tuning，禁止复制 FC 或 SHOT 已选 tuple 为 IST winner。

IST 后续 search proposal 可先考察 kappa=1、Stage-1 预测/搜索 (alpha,nu)，alpha∈{.025,.05,.10,.125,.15,.20}、nu∈{.25,.50,1}，omega×stage2_lr 做 downstream calibration。这里只列为 **候选起点**，不代表 IST 已冻结搜索；kappa=1 的适用性仍须验证。该候选设置不修改已有 SHOT-Transformer 冻结配置或结果。

独立 search protocol 须先写明：搜索空间、固定常数、pilot 范围、标注数据用途、validity 过滤、shortlist/ranking/tie break、omega×LR calibration、六 tuples 与 freeze artifact、fresh formal rerun。PU 禁止用于选择。改变 omega/LR 后须重新评估完整 online dynamics 与 Stage-1 validity，不能继承原 anchor 的有效性。当前文档不沿用 SHOT 已完成的 A1–A8 搜索/FO-margin winners。

工程待验证项：feature adapter equality、accumulator 接口、各 chunk size、PLCA 实际运行耗时/显存、新 revision/配置/identity plumbing。它们不是允许开发者随意改 objective 的入口。实现与 5-batch smoke 通过后交付并停止，不自动进入参数搜索。

---

## 18. Runtime / GPU-memory protocol

measurement unit 是原始 **online outer batch**，不是 IST view/chunk 或 Stage-1 inner step。

计时在 DataLoader yield、CPU decode/增强和输入 H2D 完成后开始，包含完整 adaptation + 独立 PU。CUDA 在区间边界 synchronize；预先 reset peak-memory statistics。排除 DataLoader wait、CPU preprocessing、初始 H2D、checkpoint I/O、JSON/summary I/O。IST 的 pre-adaptation inference、PLCA graph/solver、memory 操作、Saliency、native traversal/EMA 或 LBI/omega 属于 adaptation，不能只计 optimizer 时间。

每批记录：

~~~text
batch_index, raw_batch_size
adapt_runtime_sec, pu_runtime_sec, online_runtime_sec
lbi_stage1_runtime_sec, lbi_stage2_runtime_sec
peak_gpu_memory_allocated_bytes/mb
peak_gpu_memory_reserved_bytes/mb
~~~

IST 额外记录 adaptation_view_count、pre_adaptation_runtime、plca_runtime、memory_runtime、selector_runtime、native_self_training_runtime、writeback_runtime 和分块设置。CPU 增强虽不属于主 compute timer，仍保留 view_materialization_runtime、H2D_runtime 与 wall_runtime，不能把 8-view 额外 operational cost 隐去。memory 数据调入/操作属于方法开销，要记录其测量边界，不与初始 batch H2D 混淆。

优先在计时前缓存完整输入 views 到设备，仅分块控制 activation 内存。如实现需要分段 H2D，则须暂停/单独核算该 H2D，明确 timer revision，不能因 chunk 大小改变主 timer 的包含项。chunk 设置和 peak 中已驻留的 views/model/memory 均真实记录。

run-level 保留 mean/std/median/p95 online batch runtime、Σonline runtime、adapt/PU totals、Stage-1/2 shares。主 memory 为所有 batch 的 **max peak allocated**；reserved 是诊断。FO_runtime 和 wall_runtime 单独报告。

Random 顶层 accuracy 为三 child mean、population std（ddof=0）。可比 runtime 是 mean single-child batch/stream/FO cost；三 child 真实总开销另存 random_total_online_compute_runtime_sec / random_total_fo_eval_runtime_sec。GPU peak 为所有 child 所有 batch 的最大值，不能三张 GPU memory 相加。

正式效率比较按同 GPU 型号（参考 RTX 3090）、1 experiment process/GPU、同软件、precision、BS、workers、transform/stream；runtime_comparable=true 只能由匹配的 execution condition 支持。共享 GPU、不同硬件或 resume 不作为正式可比效率。准确率与效率有效性分别记录。

---

## 19. Checkpoint / resume 与文件边界

source_only、full_dense、candidate_dense、Random/Magnitude/Saliency：save_model=false，partial_resume=false，stream_checkpoint=false；中断则从 W0 重跑，Random child 不能继承另一 child 的 adapted state。

仅 group_lbi 支持 **完整成功 outer-batch 边界**的精确 resume；不是 mid-PLCA、mid-Stage-1、mid-Stage-2 或 PU 前。恢复 persistent model、stream position/order、Python/NumPy/CPU/CUDA RNG、method-specific persistent state、PU metric accumulators、全部 completed batch records、support union、provenance 和计数。

IST 还必须恢复 causal memory（features、corrected one-hot、batch IDs、capacity、commit_count）、reference/adaptation generators、inner-order generator（如实例化）、processed/loader counters。LBI native EMA 必须 disabled；不能恢复一个多余 EMA 或 host optimizer。局部 views/hard/soft task、Δ/Γ/Z/M 和 local Stage-2 optimizer 不跨边界保存。

resume 前 exact scientific identity 校验，日志无重复/缺失 batch，raw efficiency history 完整恢复。记录 runtime_resume_used、runtime_segment_count。invalid/cap-hit batch 不可成为 checkpoint。无最终 adapted-model checkpoint；正式效率使用 fresh non-resumed runs。

所有资产位于 /home/nas3/biod/wangkangyi/；数据 datasets/、lists datasets/image_lists/、环境 envs/lbi/、临时 tmp/，缓存使用已确认 hf-cache/pip-cache/conda-pkgs。不得写大文件到 HOME 或系统 /tmp，不硬编码未确认 HF mirror，不提交 checkpoint/dataset/cache/log。本轮仅创建协议文档，未执行训练、安装或实验。

---

## 20. Revision、scientific identity 与 artifacts

IST 必须有独立 Transformer protocol / baseline implementation / sparse-LBI implementation revision；本文预留命名见文档开头，不能冒用 SHOT 或 ResNet revision，也不能声称尚未实现的 revision 已验证。记录本次参考代码 Git，与未来实际实现 Git 分开。

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

IST-specific payload 必须包含 feature point、view-level targets、8 views、PLCA/solver/memory、native inner BS/order/iters、EMA、objective chunk 设置。使用独立 namespace，区分 ist.plca.propagation_alpha 与 lbi.alpha，并单独记录 lbi.tau_g。

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
| group_saliency | 每 rho | 当前完整 IST task 的 saliency |
| group_lbi | 每 rho | batch-local support ≤K_G |

IST 矩阵中，每 transfer 为 3 non-budgeted + 4×3 sparse =15 top-level identities；Office 90、VisDA 15，共 105。Random 的额外 child 使全量从头执行为每 transfer 21、共 **147 child/run executions**。IST source_only 使用本协议的 reference-view RNG；复用已有 source-only artifact 时，必须核验 source/stream/reference-view/metric 身份完全一致并明示引用，不能仅凭相同 W0 复用旧 PU。

该表是**计划矩阵，不是已完成实验**。先 CPU/synthetic contracts，再单 GPU 5-batch real-data smoke；全部通过后才具备后续矩阵运行条件。本协议撰写不授权本轮启动实验。

IST baseline formal gate 要求相应实现/tests/smoke/provenance 完整、无 debug-limit、完整 stream。IST-LBI 额外要求独立 search protocol 与六个 dataset×budget tuples 冻结；未完成时 formal=true 必须 fail closed。debug/provisional tuple 不得 fallback 成 formal。配置冻结后 fresh W0 正式重跑，不能把 tuning/smoke artifact 改名充当正式结果。

---

## 22. 建议实现顺序与代码职责

| 阶段 | 交付/进入下一步的条件 |
|---|---|
| P0 | source/feature/logits/scope/mode/stream mapping（本协议 §2–4）；建立 SHOT 回归基线 |
| P1 | backbone-independent IST mechanism：复用 PLCA/memory/objective/target math；Transformer adapter 仅连接 feature/logits/scope |
| P2 | full_dense + candidate_dense，验证 native inner-step/EMA/memory 状态机 |
| P3 | CPU contracts 后单 GPU Office D→A、seed2026、5 valid outer batches；不据 smoke accuracy 调算法 |
| P4 | Random → Magnitude → Saliency；真实三 child、exact budget/full scoring |
| P5 | existing Group-LBI engine objective/gradient adapter；不复制数学 |
| P6 | contracts + 独立 code review；再测全部 variants 的 5-batch real-data smoke，增加 VisDA 内存可行性与 Amazon 尾批测试 |
| P7 | 交付审阅报告并停止；另行冻结 search protocol 后才搜索 |

推荐职责分离为 mechanism（PLCA/memory/targets）、architecture adapter（source/features/logits/scope）、host orchestration（outer state machine）、shared Transformer LBI、metrics/artifacts。FC tree 是参考，不要求本轮改动；不要复制整套 ResNet trainer 或另建 ist_transformer_special_lbi 数学副本。

**正常成功 batch 的调用计数契约：**

| 路径 | full-task saliency gradient | host optimizer steps | native EMA | LBI discovery / Stage-2 optimizer instance / Stage-2 step / omega |
|---|---:|---:|---:|---|
| dense / Random / Magnitude | 0 | ceil(8B/I) | 1 | 0 / 0 / 0 / 0 |
| Saliency | 1 | ceil(8B/I) | 1 | 0 / 0 / 0 / 0 |
| Group-LBI | 0 | 0 | 0 | 1 / 1 / 1 / 1 |

所有上述路径 PLCA=1、memory commit=1、pre-inference logical task=1；scheduler steps=0。LBI full-objective evaluations=Stage-1 executed steps+1；最后 +1 是 Stage-2。不同 chunk 的物理 forward/backward 不增加 discovery 或 objective logical task count。失败/cap-hit 单独按真实执行记录，不能强行满足成功 batch 表。

---

## 23. 必须通过的 correctness / contract tests

测试要求是后续实现验收清单，本次文档工作没有执行或声称已通过这些训练测试。

| ID | 测试及必须证明的行为 |
|---|---|
| I01 | 同 checkpoint/input 下 adapter logits≈model(x)≈初始 SHOT logits；feature=[N,384]、head=[384,C]、non-distilled |
| I02 | local-only loading：无下载，拒绝 ImageNet/last/wrong source/hash/head；source-only state 精确不变 |
| I03 | 每 raw decode=1、reference=1、adapt views=8、sample/view index 对齐；normalize tensor 输入拒绝 |
| I04 | 三 RNG streams 可复现/互不干扰；三 Random children 的 raw order、views、inner order 相同 |
| I05 | view-level F/Q/ŷ，N=8B；构造同原图不同 view predictions，证明未误广播 reference label |
| I06 | PLCA exactly once；past+current graph、K+2 convention、relative CG criterion、repeat=1 与参考一致；不加 feature normalization |
| I07 | memory commit once；stored pre-adapt F + corrected one-hot、detach、FIFO 10000 view entries、strict past batch IDs；future/current snapshot 拒绝 |
| I08 | CE+KL(q||p) batchmean/权重正确；hard/soft targets batch 内 immutable、soft 为 pre-correction |
| I09 | 标签替换/置乱不改变 PLCA、memory、gradient、support、adapted model；接口无 GT |
| I10 | 普通 B64 native steps=8；Amazon B65 steps=9；全 task 遍历一次且无遗漏/重复；EMA once、anchor clone、不重置 host moments |
| I11 | full_dense/LN/non-head scope 与 controlled 12 tensors exact；全部 off-scope values/buffers 精确不变 |
| I12 | groups=6912、size=768、candidate=5308416、no overlap/alias、coverage exact；分别测试 QK/VO/FFN paired slices |
| I13 | floor budgets=3/6/13，无 slack/quota；Random exact K_G、三真实 fresh execution、nested-prefix reproducibility |
| I14 | Magnitude source-static paired L2、CPU float64/canonical tie；不能变成 L1 或 current-state magnitude |
| I15 | Saliency 是完整 N objective 的 paired L2(W*g)，once/outer；全 native traversal mask 不变；scoring grads 不充当 inner grads |
| I16 | unequal-tail chunks 的 loss/full gradients 与不分块一致；native-inner accumulation 与 full-task accumulation 分母分别正确 |
| I17 | Group prox 零/阈值/active 数值；normalized support 边界 >=1e-4；old-state Z（首步 Δ=Γ=0 时 Z 保持零） |
| I18 | overshoot rollback 恢复 Δ/Γ/Z/M 全部与 last feasible step；不 top-K trim/fill；合法 underfill/empty support |
| I19 | 3000 步 exact-K、rollback、underfill 三情形均 invalid；无 PU/later batch/FO、无成功 checkpoint；实际 Stage-2 compute 如实记录 |
| I20 | Stage-2 base+masked Δ；fresh local AdamW exactly one step、全 N objective fresh gradient；不能 per chunk step |
| I21 | off-mask AdamW values exact、exp_avg/exp_avg_sq 为零；动态 mask 连续选中 state 保留、离开后清零 |
| I22 | 每批 local restart；LBI host optimizer/scheduler/native EMA=0；omega once；EMA+omega 双写被拒绝 |
| I23 | omega/EMA 后 off-mask/off-scope exact，包含会触发 1 ULP rounding 的值；不能只 allclose |
| I24 | PU 在 EMA/omega 后 cached reference 独立 forward；PU/FO model/memory/optimizer/EMA/RNG 等 state 不变 |
| I25 | Office weighted correct/total、六 transfer equal mean；VisDA fixed-12 macro/诊断；完整 PU/FO 样本覆盖 |
| I26 | Amazon 43×64+65；unexpected formal singleton fail；非正式 early-skip 所有状态/PU计数=0，FO仍完整 |
| I27 | LBI boundary resume 恢复 memory/三 RNG/metric/history；拒绝半批、身份错配；与不中断轨迹对齐 |
| I28 | scientific field 改变 hash；sparse revision 在 resolver/manual builder/resume/aggregate/finalize 不 fallback 到 dense |
| I29 | formal gate 拒绝 unresolved tuple/debug-limit/partial run；Random 顶层必须三 child 完整 |
| I30 | SHOT 现有 source/dense/selector/LBI tests 与 smoke 回归不变；新 adapter 与旧 scalar closure 不重复 backward |

可参考现有 tests/transformer_*_test.py、FC tests/ist_otta_p1_contract_test.py、tests/ist_otta_sparse_lbi_contract_test.py；不是复制 shape tests 后就算完成。真实 GPU smoke 需报告 finite、scope、PLCA/memory/EMA/discovery/Stage-2/writeback 次数与 PU mutation check，不能仅报告“all tests passed”。

---

## 24. Freeze control 与交付验收报告

本 v1 为开发规格草案，不具有 SHOT frozen v1“实验已完成”的状态。实施前审阅明确本协议列出的裁决；实施后使用真实 revision/Git/test evidence 完成实现冻结，不修改既有 SHOT protocol/artifacts。

实现报告至少列：

- Source path/hash、initial-logit equality；feature point/shape、classifier/LN/stochastic-layer scope。
- Dense variants、processed batches、8-view 数、native inner steps、PLCA/memory/EMA commits。
- Random 三条真实 child artifact；candidate/group count、K_G、Magnitude static、Saliency once/outer。
- LBI discoveries、Stage-1 steps/cap/rollback/utilization、Stage-2 step count、native EMA=0、omega=1、off-mask exact。
- Precision/chunk strategy、显存/计时边界、PU/FO mutation audit、label leakage test。
- SHOT regression、共享引擎路径、resume/identity/finalize 一致性、5-batch smoke evidence。
- 未完成的 search protocol、六 tuples 和 formal gate 状态，不将其隐藏在默认 config。

科学语义、target 粒度、候选/预算、更新步数、writeback、precision 或 stream 改变须新 revision。不能依据正式结果再修改本 v1 的定义。

---

## 25. 一页式实现摘要

| 项 | IST-Transformer v1 开发规格 |
|---|---|
| Source/backbone | SHOT 相同 source-trained DeiT-S，non-distilled |
| Stream | seed2026；Office64/VisDA256；Amazon tail65；完整 PU/FO |
| Task | 每 raw 1 reference + 8 independent views，view-level hard/soft targets |
| Features | 原 classifier 前 384-D pre-logits，不额外 normalize |
| PLCA/memory | once/outer；k50,gamma3,l2,alpha.99,repeat1；CG20/rtol1e-6/atol0；past-only FIFO10000 view entries |
| Objective | CE + KL(q_pre||p)，权重1/1，targets固定、predictions重算 |
| Native dense/sparse | persistent AdamW 1e-5；完整8B task遍历1轮；inner BS64/256；EMA .9一次 |
| Random/Magnitude/Saliency | 3 real nested-prefix children / W0 paired L2 / full-task W*g paired L2 once/outer |
| Candidate / budgets | 12 tensors，6912×768；rho=.0005/.001/.002→K_G=3/6/13 |
| LBI | shared corrected engine；old-state Z；local restart；strict rollback |
| Support / cap | normalized Gamma norm≥1e-4；prox_lambda1；任何3000步run invalid |
| Stage-2 | full-task fresh gradient；one local masked AdamW step；无scheduler |
| Persistent LBI | omega only；native EMA disabled；off-mask exact |
| Precision/mode | eval，Dropout/DropPath inactive，non-AMP |
| Formal status | 实现/tests/smoke待做；六LBI tuples与search待冻结 |

---

## 附录 A. IST 交接逐项覆盖索引

以下编号对应交接文档章节，便于 review 时逐条核查，本文所有相关条款应合并阅读。

| 交接章节 | 本协议落实位置 |
|---|---|
| 0–1 任务与 source of truth | §1–4，§24 |
| 2–4 状态机、method constants、batch 顺序 | §5–7；target 粒度差异显式见 §1 |
| 5–7 dense、EMA、PU/FO | §8、§13、§15 |
| 8–13 substrate、F/B/C、feature、mode/LN/distillation | §3、§6.2、§14 |
| 14–17 8-view 显存、full accumulation、outer unit、重算 | §5、§9.3、§11、§18 |
| 18–21 shared engine、Stage-2 one step、禁止double writeback、sparse native | §8、§10–13 |
| 22–25 Random真实三child、Magnitude、Saliency | §9、§18、§21–23 |
| 26–29 candidate、alias、off-scope、AMP | §3–4、§10、§14、I11–I13/I21–I23 |
| 30–32 RNG、label isolation、past-only memory | §6、§14–15、I04/I07/I09 |
| 33 provenance/fallback | §20、I28–I29 |
| 34 contracts A–J | §23 I01–I30；singleton adaptation boundary见 §2/I26 |
| 35 开发 P0–P7 | §17、§21–22、§24 |
| 36–37 共享/architecture-specific职责 | §3、§11.1、§22 |
| 38 十二个高风险坑 | §6–14、§23对应状态/次数/梯度测试 |
| 39 验收报告 | §24 |
| 40–42 最终验收、迁移原则、先smoke后停止 | §21–25 |

**文档结束。**
