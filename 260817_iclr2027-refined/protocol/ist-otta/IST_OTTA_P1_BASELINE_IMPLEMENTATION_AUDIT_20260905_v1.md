# IST-OTTA P1 Baseline Implementation Audit

## A. 实现结果

### 新增文件

- `configs/ist_otta_p1_baseline_20260905_v1.yaml`：三个 P1 dense variant 的唯一 formal 配置入口。
- `ist_otta/config.py`：P0 常量、formal dataset 设置和 dense-only fail-closed 校验。
- `ist_otta/data.py`：fixed outer stream 上的 raw-image loader、PU reference view 缓存和 8-view materializer。
- `ist_otta/plca.py`：L2 robust PLCA（K=50、gamma=3、repeat=1、0.99 graph propagation）。
- `ist_otta/memory.py`：max_len=10000、按 outer batch 单次提交的 causal memory。
- `ist_otta/ema.py`：独立 clone anchor、每个 outer batch 单次 m=0.9 moving average。
- `ist_otta/trainer.py`：IST hard CE + soft KL online trainer、PU/FO 和 artifacts。
- `shot_otta/candidates.py`：SHOT/IST 共用的 frozen FC 与 full-layer4 9-Conv candidate 定义。
- `tests/ist_otta_p1_contract_test.py`：P1 targeted contract tests。

### 修改文件

- `train.py`：注册三个 IST variant，并按 `method=IST` 分派至 IST resolver/trainer。
- `experiment_identity.py`：加入 IST scientific identity、variant policy 和 method-specific fields。
- `protocol_constants.py`：加入 IST P0 protocol/P1 implementation revision。
- `shot_otta/data.py`：仅抽取原 fixed-order 与 order-record 逻辑供两种方法共用，SHOT loader 行为不变。
- `shot_otta/trainer.py`：仅改为导入共用 candidate 常量，SHOT variant/loss/loop 语义不变。
- `shot_otta/artifacts.py`：增加明确分隔的 `.../IST/<variant>/seed_2026/` 输出层级。

### 三个 dense variants

- `ist_full_dense`：`netF + netB` trainable，`netC` frozen；inner adaptation 使用 native/full train behavior。
- `ist_fc_module_dense`：仅 `netB.bottleneck.weight/bias` trainable；全部 BN parameter/running state frozen。
- `ist_conv_module_dense`：仅 frozen full-layer4 candidate 中 9 个 Conv weight trainable；全部 BN parameter/running state frozen。

### Per-outer-batch execution order

1. fixed seed-2026 stream 给出 raw identities/boundary；从 raw image 缓存一个 PU view并独立生成 8 个 IST views。
2. 保存 `netF/netB` 独立 old-state clone；以 adaptation 前模型和 eval behavior 计算 view features/soft predictions。
3. robust PLCA 只读取 current views 和 `M_{t-1}`，输出 corrected hard targets。
4. current features/corrected one-hot labels 向 memory commit 一次。
5. scheduler 以该 outer batch 推进一步；`iters=1` 遍历临时 8-view set，所有 inner mini-batch 共用同一 LR，优化 `hard CE + soft KL`。
6. 对 `netF/netB` 执行一次 m=0.9 outer EMA；非浮点 counter 不做浮点插值。
7. 对已缓存 PU reference view 做 eval/no-grad forward；不触发 parameter、BN、memory、optimizer 或 EMA 更新。
8. 丢弃临时 views；stream 结束后将 final F/B/C 设为 eval，以 no-grad 完成 FO。

## B. 基于原 IST 主动改了什么

| 改动 | Controlled comparison 原因 | 保留的 IST semantics |
|---|---|---|
| 直接调用 refined `load_source_models`，使用相同 SHOT `netF→netB→netC` 与相同 `source_F/B/C`；manifest 额外记录每个 checkpoint SHA256 | 排除 backbone、head 和 source initialization confound | IST adaptation mechanism 不变；`netC` 始终 frozen |
| 使用 refined fixed seed-2026 target order；Office BS64、VisDA BS256、drop_last=false、one pass | 固定 sample identity、顺序和 outer-batch membership | 每个 incoming batch 内仍做 IST multi-view self-training |
| 使用 common SGD substrate与 outer-batch scheduler：Office 0.01、VisDA 0.001，F×0.1、B×1.0，momentum 0.9、WD 0.001、Nesterov | 排除 optimizer/LR 时间轴 confound，防止 extend=8 将 schedule 加速 8 倍 | `extend=8`、`iters=1` 与 hard/soft objective 不变 |
| target loader 保留 raw path；PU/reference 与 adaptation RNG 分离并由 formal seed 固定 offset 派生 | 禁止 normalized tensor 的不可逆 image reconstruction，并记录 augmentation provenance | 每个 raw sample 独立产生 8 个 stochastic IST views |
| PU/FO 调用 refined dataset metric：Office overall accuracy；VisDA fixed-12-class mAcc | 统一 plasticity/stability 与 metric 口径 | adaptation 后 PU、final frozen FO |
| 增加 FC/Conv controlled dense scope和全 BN freeze；full 保留 native behavior | 与后续 matched sparse/LBI candidate pool 对齐，阻止 mask 外隐形 BN adaptation | IST PLCA/memory/loss/EMA 均保留 |
| method、variant、P0/P1 revision、source revision、stream hash、RNG、IST config进入 scientific config/manifest/summary | 与当前 refined artifact identity 一致，可区分 SHOT/IST | IST-specific config集中、无旧 parser fallback |
| PLCA 数值后端改为 chunked PyTorch kNN、sparse affinity 和 CG | 去除旧 distributed/FAISS/SciPy工程耦合，并允许直接进入 refined single-runner；float32 提高稳定性 | L2 K+2 neighbor construction、gamma relation、symmetric normalized graph、0.99 propagation、repeat=1 和 hard correction不变 |

## C. 相对旧 `IST-OTTA` 修复了什么

- 旧 target `shuffle=True`/独立 loader 改为与 SHOT 共用的 fixed seed-2026 permutation、相同 batch boundary 和 stream hash。
- 删除 `Normalize -> tensor -> ToPILImage -> IST augmentation` 路径；materializer 只接受 raw PIL/path，并显式拒绝 tensor。
- 不继承旧 parser LR 或历史 Office LR 记录；dataset LR 与 F/B multiplier 均由 P0 common substrate解析并进入 scientific identity。
- EMA 在每个 incoming batch 开始用 `detach().clone()` 建立独立 anchor，修复 shallow-reference first-batch no-op 风险；每 batch只允许 commit 一次。
- 旧工程自己的 evaluation、output layout 和 provenance 被 current PU/FO metric、artifact writer、scientific hash、source revision/checkpoint hash 和 stream/RNG record替代。
- memory 增加 batch id/order invariant，读取时拒绝 current/future state，提交时拒绝重复或跳号 batch。
- P1 resolver 对 LBI、group mode、random mask、selection seed和非 full budget全部 fail closed，不存在静默走回旧 IST/IST-LBI 的入口。

## D. Contract tests

测试文件：`tests/ist_otta_p1_contract_test.py`。

覆盖项：

1. IST/SHOT 引用同一个 source loader、相同 F/B/C paths/state，并在 deterministic input 上得到相同初始化 logits。
2. fixed seed-2026 order、stream hash 和 outer-batch boundaries。
3. PLCA 只能接收 immutable past-memory snapshot；memory 拒绝重复、跳号和 future commit。
4. raw materializer 源码无 `ToPILImage`，并拒绝 tensor input。
5. memory 每个 outer batch 单次 commit。
6. EMA 每个 outer batch 单次 commit、anchor storage 独立、first-batch EMA 非 no-op、整数 counter不插值。
7. `netC` state不变。
8. FC controlled step 只允许 bottleneck weight/bias 改变，BN buffers不变。
9. Conv controlled step 只允许共享定义的 9 个 layer4 Conv weights 改变，BN buffers不变。
10. PU 前后 model state、optimizer、memory、EMA anchor不变。
11. FO 前后 model state不变。
12. method/variant/source revision/IST config/scientific hash/stream hash进入 artifact provenance。
13. full variant 的 F/B trainability 与 native BN train behavior。

**implemented, not executed in P1 code-only phase**

本阶段只执行了 Python 静态编译、config-only `--dry-run` 解析和 `git diff --check`；未执行上述 contract tests、任何 smoke test或任何数据/模型运行。

## E. 尚未处理

```text
Random/Magnitude/Saliency: not implemented
FC-LBI: not implemented
Conv out-channel Group-LBI: not implemented
IST-LBI restart granularity: deferred
IST EMA vs LBI omega composition: deferred
hyperparameter search/formal runs: not started
```
