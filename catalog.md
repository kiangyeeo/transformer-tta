/home/nas3/biod/wangkangyi/
│
├── transformer-tta/                 # 你的项目代码，Git/Codex 只管这里
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
├── envs/
│   └── lbi/                         # 当前 conda 环境
│
├── hf-cache/                        # HuggingFace cache
├── pip-cache/
├── conda-pkgs/
├── conda-home/
├── tmp/
└── miniconda3/