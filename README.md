# Transformer Split-LBI Test-Time Adaptation

这个仓库研究通过 Split-LBI 为 test-time adaptation（TTA）发现稀疏更新 support。旧论文以 SHOT 为目标函数，在 ResNet 的 256-d bottleneck FC 上做逐元素更新；仓库的 legacy model factory 还额外支持 VGG。当前开发目标是在 ImageNet-1K 预训练的 non-distilled DeiT-Small/16-224 上实现结构化 Transformer 更新。

## 先读什么

- [`AGENTS.md`](AGENTS.md)：完整项目现状、实现边界、服务器约束、实验协议和待裁决问题。Codex 会从根目录自动读取这个文件。
- [`TRANSFORMER_GROUP_SPLIT_LBI_PROPOSAL_DOLLAR_MATH.md`](TRANSFORMER_GROUP_SPLIT_LBI_PROPOSAL_DOLLAR_MATH.md)：Transformer QK/VO/FFN 分组和实验设计。
- [`catalog.md`](catalog.md)：服务器上的代码、数据、模型、环境和缓存路径。
- [`26445_Test_Time_Adaptation_via (1).pdf`](26445_Test_Time_Adaptation_via%20%281%29.pdf)：只做 bottleneck FC 稀疏更新的旧稿，作为历史依据。

## 当前状态

| 模块 | 状态 |
| --- | --- |
| ResNet/VGG + SHOT OTTA | 已实现 |
| FC scalar Dense/Random/Magnitude/Saliency/Split-LBI | 已实现 |
| ResNet Conv dense update | `full_dense` 间接包含 |
| Conv filter/channel Group Split-LBI | 当前仓库未发现 |
| DeiT-S backbone/source training | 未实现 |
| Transformer paired QK/VO/FFN groups | 未实现 |
| TTDA | 未实现 |

因此，本仓库是可复用的 FC-OTTA 工程骨架，不是已经完成的 FC + Conv + Transformer 框架。特别注意：当前 LBI Stage 2 从相应的当前模型加 dense local delta 初始化，最终更新不保证当前 mask 外 local delta 严格为零；Transformer proposal 的最终方程暗示 strict masked delta，但与现有 engine 不同，尚待裁决。OTTA 每批即使严格限制 K 个 groups，跨 stream 的累计 support 也可能大于 K。

验证协议也尚未冻结：target labels 不能进入 adaptation，但现有分析脚本会按 target FO accuracy 选择候选。正式实验前必须决定使用独立验证依据、报告全部 budgets，还是明确披露 oracle-style selection。

## 目录

```text
train.py                 现有单次 SHOT-OTTA 入口
shot_otta/               data/model/loss/trainer/artifact
core/lbi/                现有逐元素 Split-LBI
source_training/         SHOT 风格 ResNet/VGG source trainer
visda_otta/              VisDA 指标与辅助逻辑
configs/                 单次运行配置
experiments/             实验矩阵
tools/                   planner/launcher/status/summary
tests/                   synthetic smoke 与工程测试
```

## 现有代码快速检查

安装 `requirements.txt` 后，可以只解析配置而不访问数据：

```bash
python train.py --config configs/shot_otta.yaml --variant full_dense --dry-run
```

当前测试应先强制走 CPU，因为 `tests/lbi_smoke_test.py` 的 fake fixture 在 CUDA 可见时存在 device mismatch：

```bash
CUDA_VISIBLE_DEVICES=-1 bash -c 'for f in tests/*_test.py; do python "$f" || exit 1; done'
```

## 数据与 source model

仓库不包含 Office-31、VisDA-C、image-list 或 source checkpoint。现有 loader 期望旧 SHOT 风格的 `office/*_list.txt`、`VISDA-C/*_list.txt`，而服务器原始图片采用 `catalog.md` 中的 `office31/`、`visda-c/` 目录；正式运行前需要生成并验证 list。

`catalog.md` 中的 `deit_small_patch16_224.fb_in1k/model.safetensors` 只是 ImageNet-1K 初始化。Office-31 仍需训练 Amazon/DSLR/Webcam 三个 source models，VisDA-C 仍需在 synthetic train 上训练一个 12-class source model。所有 selector 比较必须使用同一个、带 hash 的 W0。

`source_training/image_source.py` 基于 SHOT source trainer，能生成 legacy `source_F.pt/source_B.pt/source_C.pt`，但只支持 ResNet/VGG，不能直接用于 DeiT。Transformer 的 source trainer 和 checkpoint schema 是第一阶段工作。

## 服务器约束

服务器有 8 张 RTX 3090 24GB，主盘空间不足。代码、数据、模型、环境、cache 和临时文件必须留在：

```text
/home/nas3/biod/wangkangyi/
```

精确路径见 `catalog.md`。模型加载不得隐式联网；`HF_HOME`、`TORCH_HOME`、`PIP_CACHE_DIR`、`CONDA_PKGS_DIRS`、`TMPDIR` 等必须显式指向该根目录。当前 catalog 未定义正式结果目录，大规模实验前需先确认并冻结 output root。

## 依赖规则

任何新增 Python import 都必须在同一变更中写入 `requirements.txt`。计划中的 timm 本地加载路线需要 `timm` 和 `safetensors`；在代码真正引入它们时，同步加入并锁定与服务器 PyTorch/CUDA 兼容的版本。
