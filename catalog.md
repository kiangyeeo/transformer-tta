/home/nas3/biod/wangkangyi/
│
├── transformer-tta/                 # 你的项目代码，Git/Codex 只管这里
│   ├── configs/                     # 实验配置
│   │   ├── office31/
│   │   └── visdac/
│   │
│   ├── src/                         # 核心实现
│   │   ├── models/                  # DeiT wrapper、delta 参数化
│   │   ├── lbi/                     # Split-LBI / Group-LBI
│   │   ├── tta/                     # TTDA / OTTA 流程
│   │   ├── datasets/                # dataset loader
│   │   └── utils/
│   │
│   ├── scripts/                     # 一键训练/测试脚本
│   ├── train_source.py              # source-domain model 训练
│   ├── train_ttda.py                # TTDA
│   ├── train_otta.py                # OTTA
│   ├── requirements.txt
│   └── README.md
│
├── datasets/                        # 所有数据，不放进项目 repo
│   ├── office31/
│   │   ├── amazon/
│   │   ├── dslr/
│   │   └── webcam/
│   │
│   └── visda-c/
│       ├── train/                   # synthetic source
│       └── validation/              # real target
│
├── checkpoints/                     # 所有模型权重
│   │
│   ├── deit_small_patch16_224.fb_in1k/
│   │   ├── model.safetensors        # ImageNet-1K pretrained DeiT-S
│   │   └── config.json
│   │
│   └── source_models/               # 用 source domain 训练后的 W0
│       ├── office31/
│       │   ├── amazon.pth
│       │   ├── dslr.pth
│       │   └── webcam.pth
│       │
│       └── visda-c/
│           └── train.pth
│
├── outputs/                         # 建议新建，统一放实验结果
│   ├── ttda/
│   └── otta/
│
├── envs/
│   └── lbi/                         # 当前 conda 环境
│
├── hf-cache/                        # HuggingFace cache
├── pip-cache/
├── conda-pkgs/
├── conda-home/
├── tmp/
└── miniconda3/