# Transformer Split-LBI TTA：项目工作约定

> 最近一次状态更新：2026-08-17；最近一次代码、论文与方案审计：2026-08-15。这个文件位于仓库根目录，供 Codex 和后续开发者进入项目时自动读取。实现状态发生变化时，必须在同一提交中更新本文件。

## 1. 项目目标

本项目研究用 Split-LBI 发现稀疏的 test-time adaptation（TTA）更新 support。核心表示为

\[
W^{\mathrm{TTA}} = W_0 + \Delta W,
\]

其中 `W0` 是在 source domain 上训练完成且在一次比较中保持完全一致的模型，`DeltaW` 是 target-specific update。

历史论文 `26445_Test_Time_Adaptation_via (1).pdf` 只在 SHOT 的 bottleneck FC 层上做逐元素稀疏更新。本仓库当前也只实现了这一条稀疏路径。当前阶段的主任务是新增 non-distilled DeiT-Small/16-224，并在最后 3 个 Transformer blocks 上实现 QK、VO、FFN structural groups，以及 matched-budget Random、Magnitude、Saliency 和 Group Split-LBI 实验。

不要把 proposal 中的设计目标误写成已经完成的代码，也不要假设仓库中已有卷积结构化更新或 TTDA。

## 2. 信息来源与适用范围

用户在当前任务中的明确要求始终优先。其余资料不是一条全局优先级链，而是分别对不同主题负责：

- 当前已经实现的行为：以代码和测试为准；
- Transformer 的目标设计：以 `TRANSFORMER_GROUP_SPLIT_LBI_PROPOSAL_DOLLAR_MATH.md` 为准，但它不证明功能已经实现；
- 服务器路径和磁盘边界：以 `catalog.md` 为准；
- 服务器上会动态变化的训练/产物状态：以受 Git 跟踪的 `SERVER_STATE.md` 为准；
- 旧实验、旧算法叙述和旧表格：以论文 PDF 为历史证据；
- 已由用户裁决的跨文件工程约定：记录在本 `AGENTS.md`。

若资料不一致，必须明确写成“证据”“推断”“建议”或“待裁决”，不能静默选择其一。需要改变旧 FC 数值语义时，保留 legacy 路径和回归测试。

## 3. 当前真实状态

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| ResNet/VGG + SHOT OTTA | 已有 | `train.py`、`shot_otta/` |
| FC dense/random/magnitude/saliency | 已有 | 候选参数硬编码为 `netB.bottleneck.weight/bias`，且按 scalar 选择 |
| FC Split-LBI | 已有 | `core/lbi/` 使用逐元素 prox 和逐元素 support |
| Conv dense update | 部分已有 | `full_dense` 更新 `netF + netB`，因此会更新 ResNet Conv；这不是 Conv group sparse update |
| Conv filter/channel Group Split-LBI | 未发现 | 当前 Git 仅有 `master/origin/master`，仓库和历史中没有相应实现 |
| DeiT source trainer | 已实现；4 个 W0 已在服务器产出 | `train_source_deit.py` 全量微调 non-distilled DeiT-S，直接 `Linear(384,31/12)` head；2026-08-17 用户确认 Office-31 三域和 VisDA-C train 的 `.pth` 均位于 catalog 约定路径，本地尚未核验 manifest/hash |
| DeiT/Transformer TTA backbone | 部分已有 | 已有严格本地 W0 加载和 TTDA source-only 直接 logits 评测；OTTA、可训练 TTDA 和 structural adaptation 尚未接入 |
| Transformer structural groups | 未实现 | 没有 group registry、group prox、group mask/scatter |
| OTTA | 已实现 | 单遍 target stream；每个 batch 适配后预测，模型状态传给后续 batch |
| TTDA | 仅 source-only control | `evaluate_deit_ttda.py` 支持零适配的完整 target dataset 评测；任何 TTDA adaptation 生命周期仍未实现 |
| TENT/EATA/CoTTA 等独立 TTA 方法 | 未实现 | 现有 baseline 是同一 SHOT objective 下的更新/选择 baseline |
| Source-domain trainer | ResNet/VGG + DeiT 可用 | legacy SHOT 三文件路径保持不变；DeiT 使用新单文件 schema、source-only validation 与每 epoch resume state |
| Office-31/VisDA-C 数据与 W0 | 仅服务器持有 | 数据、image lists 和 4 个 source checkpoints 均不进 Git；checkpoint 完成状态为用户确认，精确 hash/指标待从服务器 manifest 归档 |

如果学姐的 Conv 工作确实存在，先请求具体分支、仓库、补丁或文件；不得凭 proposal 首段推断它已经合入。

## 4. 现有仓库结构

```text
transformer-tta/
├── AGENTS.md                    # 本文件：持久项目上下文和开发约定
├── README.md                    # 给人的简明入口
├── catalog.md                   # 服务器目录；不要擅自改写用户路径
├── SERVER_STATE.md              # 受 Git 跟踪的服务器动态状态与核验边界
├── requirements.txt            # 所有运行时 Python 依赖必须记录在这里
├── train.py                     # 现有 SHOT-OTTA 单次实验入口
├── evaluate_deit_ttda.py        # DeiT TTDA source-only 零适配评测入口
├── train_source_deit.py         # DeiT-S source-domain 训练统一入口
├── experiment_identity.py       # legacy FC 与 DeiT TTDA source-only 实验身份
├── shot_otta/                   # 数据、模型、loss、trainer、artifact
├── core/lbi/                    # 逐元素 Split-LBI engine/state/diagnostics
├── source_training/             # SHOT 风格 ResNet/VGG source trainer
│   ├── deit_config.py           # Transformer source 协议校验与路径解析
│   ├── deit_data.py             # image-list、固定分层 source split、transform
│   ├── deit_model.py            # local-only safetensors 与单文件 W0 loader
│   └── deit_trainer.py          # full-model fine-tune、resume、artifact
├── visda_otta/                  # VisDA 指标和适配辅助代码
├── configs/                     # 单次运行配置
├── experiments/                 # 实验矩阵
├── tools/                       # planner、launcher、status、summary
├── scripts/                     # 维护脚本
├── tests/                       # 合成 smoke/工程测试
├── TRANSFORMER_GROUP_SPLIT_LBI_PROPOSAL_DOLLAR_MATH.md
├── SOURCE_TRAINING_DEIT_PROTOCOL.md
└── 26445_Test_Time_Adaptation_via (1).pdf
```

可直接复用的工程骨架包括 SHOT objective、image-list 数据读取、artifact、planner/launcher/status/summary，以及 LBI 的阶段生命周期。不能直接复用的是硬编码的 FC candidate、逐元素 support/prox、ResNet/VGG model factory、FC-only identity/output path 和 OTTA-only protocol。

## 5. Transformer 目标规范

除非用户更新 proposal，主实验固定为：

- Backbone：`deit_small_patch16_224.fb_in1k`，non-distilled，ImageNet-1K 初始化，224 输入；
- 模型结构：12 blocks，hidden dimension `d=384`，FFN dimension `1536`；
- Candidate space：最后 3 个 blocks；
- 主分组：paired QK、paired VO、paired FFN；
- QK group `p`：Q 的第 `p` 个输出 row + K 的第 `p` 个输出 row；
- VO group `p`：V 的第 `p` 个输出 row + output projection 的第 `p` 个输入 column；
- FFN group `p`：`fc1` 的第 `p` 个输出 row + `fc2` 的第 `p` 个输入 column；
- timm 的 fused `qkv.weight` shape 为 `[3d, d]`，必须通过逻辑 Q/K/V slice 分组，不能把整张 qkv 当成一个 group；
- 每个 block 有 `384 + 384 + 1536 = 2304` groups，最后 3 blocks 共 `6912` groups；
- 每个 paired group 含 `2d = 768` scalar weights；
- 主结构预算 `rho_struct ∈ {0.001, 0.003, 0.005}`，约对应 `{7, 21, 35}` 个 active groups；`0.01` 只作为必要时的扩展；
- proposal 明确冻结的 matched selectors：Random、Magnitude、Saliency、Group Split-LBI；它们必须使用同一 candidate space、同一 group definition 和同一 group budget；
- 强烈建议的 controls：Source-only、Full-dense、Candidate-dense；它们不是 proposal 中的四个 matched sparse selectors；
- 独立 Q/K/V/O/W1/W2 slice groups 仅作为 paired-vs-independent ablation，不作为另一套主方法；
- 数据集：Office-31 六个有向迁移任务和 VisDA-C `train -> validation`；
- 协议目标：TTDA + OTTA。先做 OTTA vertical slice，因为当前工程只支持 OTTA。

结构预算必须报告至少三种量：active groups / candidate groups、active scalars / candidate scalars、active scalars / whole model。不能把它们统称为一个 `sparsity`。TTDA 的一次 support 与 OTTA 的 per-batch support 也不能混用；OTTA 还要报告历史 active-group union 和最终相对 source model 的非零 support。

proposal 明确定义的 structural groups 只包含 qkv/proj/MLP 的 Q/K/V/O/W1/W2 weight slices。bias、LayerNorm、class token、position embedding、classifier 和可能的 bottleneck 是否冻结或在结构化 selector 之外更新，仍是待裁决项；它们不得被无意计入 structural denominator。

## 6. 开发前必须裁决的算法问题

### 6.1 最终 delta 的稀疏语义

先区分三个量：`W_source` 是 source checkpoint，`W_t` 是 OTTA stream 在 batch `t` 到达前的累计模型，`delta_t` 是当前 batch 的局部增量。当前 engine 每次调用都把当时的模型快照作为 `W_base`；TTDA 初始时它通常是 `W_source`，OTTA 时则是 `W_t`。

proposal 的最终方程暗示严格的 `W_new = W_base + delta_masked`。当前 FC engine 的 Stage 2 却从 `W_base + dense theta_delta` 初始化，只 mask gradient；带 weight decay 的 SGD 仍可能改变 mask 外元素，最终还会把整张 refined tensor 混回。因此当前实现是“稀疏 support discovery + masked refinement”，不是严格保证当前 batch 的 mask 外增量为零。

以下是待用户选择的两个方案，不是已经批准同时实现的需求：

- 方案 A：保持当前 `legacy_dense_init` 语义，用于严格延续旧 FC 方法；
- 方案 B：采用 `strict_masked_delta`，每次 step 后恢复/投影当前 batch mask 外参数，最终只写回 active groups。

无论选择哪一项，都要把 `delta_semantics` 写入 config、manifest、实验 identity 和汇总表。若选择方案 B，strict 约束只作用于当前 `delta_t`，不能抹掉先前 batch 已写入 `W_t` 的合法更新；即使每个 batch 恰好 K 个 groups，累计的 `W_T - W_source` 也可能覆盖远多于 K 个 groups。用户裁决前不要据此扩展两套 Transformer 主实验。

### 6.2 还未冻结的定义

以下内容没有被当前 proposal/代码共同、无歧义地确定；正式实验前必须写入配置并冻结：

- group prox 中 `lambda` 与 support threshold `tau`；
- tiny budget 的取整规则，以及 Split-LBI 一步跨过目标 K 时是 exact-K top score 还是回滚；
- DeiT source training v1 已由 `SOURCE_TRAINING_DEIT_PROTOCOL.md` 冻结：直接 `Linear(384,C)`、无 bottleneck、全模型 AdamW 微调、Office 100/VisDA 10 epochs、global batch 64、固定分层 90/10 source split、只按 source validation 选 W0。修改这些定义必须升级协议并同步配置，不能只改代码默认值；
- source training、可训练 TTA 和评估的 resize/crop、bicubic interpolation、crop percentage、augmentation、ImageNet mean/std 和 input normalization；当前 TTDA source-only 已冻结为 source v1 validation transform，但不能自动扩展为 adaptation 协议；
- Transformer 的 dropout/DropPath policy。当前 FC trainer 会把 `netF` 置为 train；DeiT 通常应保持 deterministic eval，除非显式研究 stochastic adaptation；
- bias、LayerNorm、class/position tokens、classifier 和可选 bottleneck 的冻结/更新策略；
- TTDA 的数据生命周期、epoch 数、模型选择和 evaluation timing；
- offline validation protocol：source validation、held-out transfer、target validation split、固定全 budget 汇报，或沿用旧代码的 target-accuracy oracle selection；
- `source_training_seed`、`tta_stream_seed` 和 selector/random-mask seed 的关系，以及正式三 seed 是否共享一个固定 W0；
- Transformer source checkpoint 已采用 versioned 单文件格式并有 round-trip/hash 测试；TTDA source-only loader 已接入，legacy ResNet/VGG 仍保留三文件格式；其他 TTA 路径尚未接入新 schema；
- 其他 Transformer 正式实验的 output root；当前只冻结了 source-only control 的独立目录。

不要通过代码默认值隐式决定这些实验定义。

### 6.3 已知工程风险

- 在线循环会静默跳过 batch size 为 1 的尾批；这是旧 BN workaround，对 DeiT/LayerNorm 未必合理；
- stream checkpoint 目前只允许 `VISDA-C + module_lbi`，且不能在昂贵的单个 Stage-1 batch 中途恢复；
- Stage 1 每个 step 都执行完整 forward/backward，当前上限可到 3000；DeiT 的首要瓶颈可能是时间而不是显存，pilot 必须记录 steps 分布与 wall time；
- source trainer 的 Office overall accuracy 与 target trainer 的 Office macro-per-class accuracy 口径不同；正式表必须统一或同时明确报告；
- artifact 的 Git 元数据目前从仓库父目录查询，服务器上可能得到空 commit；Transformer 实验前要验证 manifest 中的 commit/dirty state；
- 根目录已有最小 `.gitignore` 排除 Python cache、临时目录、runs、datasets 和 checkpoints；新增 artifact 目录时同步更新，但不要忽略需要版本控制的实验定义或冻结协议文件。

## 7. 推荐实现边界

保持现有 FC 路径可回归，不建议直接把所有 Transformer 条件继续堆进 `shot_otta/trainer.py`。推荐按以下边界扩展；文件名可在实现时微调，但职责不可重新混在一起：

```text
shot_otta/backbones/deit.py
    DeiT-S wrapper、CLS feature、严格本地 checkpoint 加载

shot_otta/adaptation/candidates.py
    architecture -> candidate parameters / structural groups

shot_otta/adaptation/groups/base.py
    GroupSpec / GroupOperator：prox、support、mask、scatter、计数

shot_otta/adaptation/groups/fc.py
    singleton group，封装并回归当前 scalar 行为

shot_otta/adaptation/groups/deit.py
    last-3 block paired/independent QK、VO、FFN groups

shot_otta/adaptation/selectors.py
    统一 group Random、Magnitude、Saliency，保证 matched K

configs/shot_otta_deit.yaml
experiments/shot_otta_deit_pilot.yaml
tools/build_image_lists.py
tests/transformer_groups_test.py
tests/group_lbi_test.py
tests/deit_model_smoke_test.py
tests/source_checkpoint_compatibility_test.py
```

`core/lbi/engine.py` 应接收可注入的 group operator；element-wise 行为是 singleton-group 默认实现。`experiment_identity.py`、artifact 路径、planner 和 summary 必须同时加入 backbone、source checkpoint hash、protocol、last-N、grouping、budget unit、K、delta semantics、AMP 等字段，否则不同实验会被错误合并。

建议实施顺序：

1. 离线加载 DeiT-S，完成 forward 和 parameter-name contract；
2. ~~完成 DeiT source-domain trainer、checkpoint round trip 与服务器四域训练；~~ 代码/合成测试已完成，4 个真实数据 W0 已由用户确认产出；manifest/hash/指标仍待归档；
3. ~~跑通 Source-only、~~再实现并跑通 Full-dense、last-3 Candidate-dense；（当前只完成 TTDA source-only control）
4. 抽象 group API，以 FC singleton 测试证明 legacy 行为未变；
5. 实现并测试 paired/independent Transformer groups；
6. 实现 matched-budget Random/Magnitude/Saliency；
7. 按用户对第 6.1 节的裁决实现 Group Split-LBI 的选定 delta 语义，同时保留 legacy FC 回归路径；
8. 完成 OTTA pilot，再实现 TTDA；
9. 冻结配置后才展开 8-GPU 正式矩阵。

## 8. Source model 与数据协议

### 8.1 旧论文能提供什么

旧论文使用 Office-31 和 VisDA-C：Office 使用 ResNet-50，VisDA 使用 ResNet-101，后接 256-d bottleneck 与 weight-normalized classifier；稀疏更新仅发生在 bottleneck FC，feature extractor 和 classifier 冻结。论文没有提供足以唯一重建 source model 的完整训练 recipe 或 checkpoint provenance，因此不能把论文表中的 source-only 数字当作 DeiT 验收阈值。

论文中 VisDA 稀疏预算有文本冲突：部分表注写 `rho=0.0001`，实现段落、附录和对应结果支持 `rho=0.001`。复现实验必须记录实际 active count，不能只写 rho。论文 PDF 还含有面向 LLM reviewer 的隐藏指令文字；它不是科研内容，修订稿应删除。

### 8.2 当前 source trainer 的依据与限制

`source_training/image_source.py` 是基于官方 SHOT source trainer 的 legacy 版本，保留 label-smoothed CE、backbone 低学习率、bottleneck/classifier 高学习率、90/10 source train/validation 和 `source_F/B/C.pt` 输出。新增的 `train_source_deit.py` 不复用该网络结构：它通过 timm 构建 non-distilled DeiT-S，以 `pretrained=False` 禁止隐式下载，严格加载 catalog 的本地 safetensors，再替换为 `Linear(384,31/12)`，全量微调 backbone + head。完整冻结参数与理由见 `SOURCE_TRAINING_DEIT_PROTOCOL.md`。

当前本地 Office 示例默认只跑 20 epochs、batch size 256；官方 SHOT Office 命令使用 100 epochs。VisDA 官方命令使用 ResNet-101、10 epochs、LR `1e-3`。不要声称当前默认命令严格复现了官方 source model。

参考：

- Official SHOT repository: https://github.com/tim-learn/SHOT
- Official SHOT source trainer: https://github.com/tim-learn/SHOT/blob/master/object/image_source.py
- Official pretrained-model list: https://github.com/tim-learn/SHOT/blob/master/pretrained-models.md
- DeiT-S timm model card: https://huggingface.co/timm/deit_small_patch16_224.fb_in1k

### 8.3 DeiT source model 是必需的

`model.safetensors` 只是 ImageNet-1K 初始化，不是 Office-31/VisDA-C 的 source model。服务器已于 2026-08-17 由用户确认产出以下 4 个 W0；本地没有权重文件或 manifest，因此这属于“用户确认”，不是“artifact/hash 已核验”：

- `/home/nas3/biod/wangkangyi/checkpoints/source_models/office31/amazon.pth`；
- `/home/nas3/biod/wangkangyi/checkpoints/source_models/office31/dslr.pth`；
- `/home/nas3/biod/wangkangyi/checkpoints/source_models/office31/webcam.pth`；
- `/home/nas3/biod/wangkangyi/checkpoints/source_models/visda-c/train.pth`。

正式 TTA 前仍必须满足：

- 同一 source checkpoint 下，所有 selector、budget 和 TTA stream seeds 必须读取完全相同的 W0，并记录文件 SHA-256；除非另有显式 source-seed study，不要让每个 TTA seed 暗中对应不同 W0；
- source training 与 target adaptation 使用同一个 model/head schema；
- source checkpoint 保存架构、类别数、pretrained 文件 hash、训练配置、seed 和 state dict；
- target labels 永远不得进入 adaptation loss、pseudo-label 生成或在线 early stopping。它们能否用于离线 budget/超参选择尚未裁决；现有 `analyze_*` 工具确实按 FO target accuracy 排候选，这属于 oracle-style selection，必须与最终声明的 validation protocol 一起保留、改写或明确披露。

当前 Transformer checkpoint 已按 catalog 使用每个 domain 一个 `.pth`，内部保存完整 DeiT `state_dict` 和版本化 metadata；相邻 manifest 保存 checkpoint SHA-256，`.last.pth` 保存 optimizer/scaler/RNG 供断线恢复。legacy loader 继续使用每个 domain 的 `source_F.pt/source_B.pt/source_C.pt` 三文件，两者有意不兼容。Transformer TTA 接入时必须调用 `source_training.deit_model.load_deit_source_checkpoint`，不得重新加载 ImageNet 权重。

### 8.4 数据 list contract

现有 loader 不扫描图片目录，而是读取：

```text
<data_root>/office/{amazon,dslr,webcam}_list.txt
<data_root>/VISDA-C/{train,validation}_list.txt
```

每行是 `image_path integer_label`。代码把 `image_path` 直接交给 PIL，不会自动拼接 `data_root`，所以服务器 list 使用绝对路径。`tools/build_image_lists.py` 已实现并测试，会为 Office 三域/VisDA 两域验证一致的 class directory 集合、写绝对路径并保存 `class_to_idx.json`。原始数据不随仓库提供，必须在服务器 `/home/nas3/biod/wangkangyi/datasets/image_lists/{office31,visda-c}/` 实际生成。新 source config 使用 catalog 的 `office31`/`visda-c` 命名；legacy OTTA 的 `office`/`VISDA-C` alias 仍待 Transformer adapter 统一。

## 9. 服务器资源与路径约束

服务器有 8 张 RTX 3090 24GB，但主盘空间已满。所有环境、缓存、临时文件、代码、数据和 checkpoint 只能位于用户划定的根目录：

```text
/home/nas3/biod/wangkangyi/
```

已知路径以 `catalog.md` 为准：

```text
repo:          /home/nas3/biod/wangkangyi/transformer-tta/
datasets:      /home/nas3/biod/wangkangyi/datasets/
pretrained:    /home/nas3/biod/wangkangyi/checkpoints/deit_small_patch16_224.fb_in1k/
source models: /home/nas3/biod/wangkangyi/checkpoints/source_models/
TTDA source-only results: /home/nas3/biod/wangkangyi/results/transformer_ttda_source_only/
conda env:     /home/nas3/biod/wangkangyi/envs/lbi/
HF cache:      /home/nas3/biod/wangkangyi/hf-cache/
pip cache:     /home/nas3/biod/wangkangyi/pip-cache/
conda pkgs:    /home/nas3/biod/wangkangyi/conda-pkgs/
conda home:    /home/nas3/biod/wangkangyi/conda-home/
temporary:     /home/nas3/biod/wangkangyi/tmp/
```

服务器命令应显式设置 `HF_HOME`、`TORCH_HOME`、`PIP_CACHE_DIR`、`CONDA_PKGS_DIRS`、`TMPDIR` 等到上述根目录内，禁止默认写入 `$HOME`、系统 `/tmp` 或主盘缓存。TTA evaluation 必须先以不下载 pretrained weight 的方式构建模型，再加载本地 source checkpoint；不得因 `pretrained=True` 隐式联网。

用户已确认 DeiT TTDA source-only control 的正式结果根目录为 `/home/nas3/biod/wangkangyi/results/transformer_ttda_source_only/`。这一裁决只覆盖该七任务零适配基线；后续 Transformer sparse/dense、OTTA 或 TTDA adaptation 的正式 output root 仍需单独冻结。旧代码的相对 `output.root: runs` 约定不得自动套用到新实验。

开发采用 local-first：本地完成静态实现、CPU/synthetic tests 和配置生成，再上传服务器。服务器先做单卡、单任务、短 stream pilot；确认数值、速度、显存和 artifact 后，才按“一张卡一个独立进程”扩展到 8 卡。launcher 默认只有 GPU 0--3 和 4 workers，正式执行需显式传 0--7 与 8 workers。

## 10. 依赖管理

任何新 Python import 都必须在同一变更中加入 `requirements.txt`；不能只在服务器上临时 `pip install`。新增依赖前先检查标准库或现有依赖能否完成任务。

当前本地权重是 `.safetensors`。根据用户要求，`requirements.txt` 已提前列入计划中的本地加载栈 `timm` 和 `safetensors`，即使 Transformer 模块尚未合入。服务器兼容组合验证后再锁定版本，并记录 Python、CUDA、PyTorch、torchvision、timm 和 safetensors 版本。不要自动改用 Hugging Face Transformers 版 DeiT，因为 proposal 的 fused qkv 分组依赖 timm 参数布局。

服务器安装应使用 catalog 中的 pip/conda cache 路径。环境验证通过后生成可复现 lock/snapshot；不要根据开发机当前混杂环境盲目锁版本。

## 11. 实验与可复现性约定

每个正式 run 的 config/manifest 至少记录：

- Git commit 和 dirty state；
- dataset、source、target、protocol、`source_training_seed`、`tta_stream_seed`、selector/random-mask seed、stream order；
- backbone 完整名、实现库和版本；
- ImageNet pretrained 文件路径与 hash；
- source checkpoint 路径、schema 与 hash；
- candidate blocks、完整 candidate parameter names；
- grouping、paired/independent、是否含 bias/LN/head；
- selector、budget unit、rho、目标 K、实际 K；OTTA 还要记录 per-batch K、历史 group union 和最终 source-relative support；
- `tau`、`lambda`、LBI step size、max steps、到达 budget 的实际 steps；
- 本次实验选定的 `delta_semantics`（例如 `legacy_dense_init` 或 `strict_masked_delta`）；
- optimizer、LR、weight decay、batch size、AMP、model mode policy；
- 指标口径：OTTA 记录 PU/FO，TTDA source-only 只记录单次完整 target 的 `Acc`；同时记录 peak GPU memory、wall time；
- Office overall 与 macro-per-class 的明确区别；VisDA 使用 12-class macro-per-class。

先做最小 pilot：Office `A -> D` 与 VisDA `train -> validation`、一个 seed、OTTA、三个主 budget；先跑 Source-only、Full-dense、Candidate-dense，再跑 selectors。对一个代表任务做 paired-vs-independent。pilot 可以用 target accuracy 做工程 go/no-go 检查，但这不自动构成无偏的超参选择。展开正式矩阵前必须冻结 validation protocol：使用独立验证依据，或预先承诺报告全部 budgets；若沿用 target benchmark accuracy 选主 budget，则必须明确标为 oracle-style selection，不能同时声称 labels 只用于最终一次评估。

现有 proposal 的“baseline”主要指 matched structural selectors，不等于已经实现 TENT/EATA/CoTTA 等方法。若后续需要广泛 TTA method baselines，应另列 scope、官方实现、protocol 和算力预算，不能把论文表格中的异构数字直接并到本实验表。

## 12. 测试门槛

原有 smoke/engineering 脚本在 2026-08-14 审计时强制 CPU 均通过；`tests/deit_source_training_test.py` 覆盖 source trainer，`tests/deit_ttda_source_only_test.py` 以 fake DeiT、31 类合成图片和临时 manifests 覆盖零适配、固定类指标、尾批、七任务 planner、checkpoint 合约与汇总。真实 timm checkpoint forward 与 CUDA 评测只允许在服务器环境验证。Windows PowerShell：

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
Get-ChildItem tests/*_test.py | ForEach-Object { python $_.FullName }
```

Linux：

```bash
CUDA_VISIBLE_DEVICES=-1 bash -c 'for f in tests/*_test.py; do python "$f" || exit 1; done'
```

GPU 可见时 `tests/lbi_smoke_test.py` 的 fake model/loader 留在 CPU，而 trainer 会把输入移到 CUDA，当前会产生 device mismatch。这是测试 fixture 问题，不代表真实 LBI 已通过 GPU 集成测试。

Transformer 合入前至少增加：

- 6912 group、每组 768 scalar、无重叠且覆盖预期 slices 的精确测试；
- fused qkv/Q/K/V 与 proj/fc1/fc2 gather-scatter 测试；
- paired group 同时 mask 两侧的测试；
- group prox 数值测试和 singleton-group legacy 回归；
- Random/Magnitude/Saliency/LBI exact matched-K 测试；
- strict mode 下当前 batch 相对 `W_t` 的 mask 外 `delta_t` 为零、且不会抹掉历史更新的测试；
- local-only safetensors load，禁止网络访问；（source 路径已有严格加载与合成测试，仍需真实 timm 权重 smoke）
- source checkpoint save/load/hash round trip；（已有 CPU 合成测试和用户确认的服务器真实 W0，仍需归档服务器 manifest/hash 并做 TTA loader smoke）
- CPU tiny forward/backward 与服务器 GPU AMP/peak-memory pilot；
- identity 能区分 backbone/grouping/protocol/delta semantics 的测试。

当前 Office planner dry-run 可生成 7 个实验；VisDA planner 会因缺少 `experiment_logs/CURRENT_VISDA_BATCH_SIZE.txt` 失败。先修复/显式配置这个冻结值，再声明 VisDA pipeline 可运行。

## 13. 未来 Codex/开发者的操作规则

- 首先读取本文件、`SERVER_STATE.md`、`catalog.md` 和 Transformer proposal；涉及旧实验时再查 PDF。
- 开始修改前运行 `git status --short --branch`；`catalog.md` 可能含用户未提交修改，不得覆盖。
- 保留 legacy FC 行为和测试；不要把语义变化伪装成重构。
- 不把 datasets、checkpoints、runs、cache 或 PDF render 临时文件提交到 Git。
- 不在未确认的服务器 output path 上启动大规模实验。
- 不自动下载模型或数据；所有输入使用 catalog 中的本地路径并校验 hash。
- target labels 不得进入 adaptation；离线 budget/超参选择严格遵循并披露已冻结的 validation protocol，不能由 analyzer 默认行为暗中决定。
- 每引入一个包，同步更新 `requirements.txt`；每新增实验轴，同步更新 identity、manifest 和 summary。
- 实验服务器很慢：先静态/单元测试，再单卡小批量 profile，最后并行；记录速度和显存，不能仅凭理论显存估算。
- 遇到论文、proposal、代码三者冲突时，写明证据、推断和建议，并请求用户裁决真正影响论文方法定义的部分。
