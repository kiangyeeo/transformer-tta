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
│   └── source_models/               # 用 source domain 训练后的 W0；以下 4 个 .pth 已于 2026-08-17 由用户确认存在
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
├── results/
│   ├── transformer_ttda_source_only/ # 已确认的 DeiT TTDA source-only 正式结果根目录
│   ├── transformer_otta_source_only/ # DeiT OTTA source-only 配置采用的独立结果根目录
│   └── transformer_otta_full_dense/  # DeiT OTTA full/candidate-dense 配置采用的默认结果根目录（未冻结）
└── miniconda3/

状态说明：以上 4 个 source-model `.pth` 路径与训练配置中的
`checkpoint.output_path` 一致。权重文件留在服务器，不复制进 Git；本地仓库尚未核验
对应 manifest 中的 SHA-256、best epoch 和 source-validation 指标。动态状态详见
`SERVER_STATE.md`。

DeiT TTDA source-only 的正式输出根目录已由用户确认冻结为
`/home/nas3/biod/wangkangyi/results/transformer_ttda_source_only/`。其中 `runs/`
保存逐实验 artifact，`launcher_logs/` 保存调度日志，`plans/` 保存固定七任务计划，
`summary/` 保存最终汇总表。

DeiT OTTA source-only 配置和运行指令采用
`/home/nas3/biod/wangkangyi/results/transformer_otta_source_only/`，避免与 TTDA
identity、run 和汇总产物混放。该路径是本次实现的独立默认值；它不代表后续可训练
OTTA 或结构化 adaptation 的正式结果根目录已经冻结。

DeiT OTTA full-dense/candidate-dense baseline 的配置默认结果根目录为
`/home/nas3/biod/wangkangyi/results/transformer_otta_full_dense/`。它只是本次
实现给出的默认值，尚未被冻结为可训练 OTTA 的正式结果根目录。
