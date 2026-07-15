# SYCL GPTQ INT4 iGPU 后端实现与调优记录

本文记录 KTransformers 在 Linux + Intel 集成显卡上实现和优化 `SYCL_GPTQ_INT4`
MoE 专家后端的过程。测试模型为 Qwen3.5-35B-A3B-GPTQ-Int4，工作负载重点是
decode 阶段的 `M=1` 稀疏专家计算。

最后更新：2026-07-15

## 最终结论

本轮优化已经达到预期。最终固定 1000-token decode 测试结果为：

```text
Run 1: Total: 54270ms | TTFT: 115ms | TPOT: 54.2ms | In: 1 | Out: 1000
Run 2: Total: 53547ms | TTFT: 112ms | TPOT: 53.5ms | In: 1 | Out: 1000
```

平均 TPOT 为 `53.85 ms`，约等于 `18.6 token/s`。同一次服务端测试中的模型内部计时为：

```text
[KT model timing] mode=decode calls=2000 avg_total=53.170ms
avg_layers=53.154ms avg_tokens=1.00 deep=0
```

40 层 MoE decode 日志在 `calls=1900` 时的平均值为：

```text
avg_total=0.659ms/layer
avg_gate_up_submit=0.058ms/layer
avg_down=0.582ms/layer
avg_active=7.35
```

pipeline 模式下，`avg_gate_up` 主要记录异步提交时间；尚未完成的 gate/up 执行会在依赖它的
down 阶段等待，因此性能判断应以 `avg_total` 和最终 TPOT 为主。

最终采用的组合是：

- gate/up 融合，并对 active experts 异步提交。
- down projection 异步提交。
- gate/up 到 down 使用 event dependency 形成小 kernel pipeline。
- gate/up 和 down 都使用 packed INT4 subgroup kernel，subgroup size 为 16。
- GPTQ 权重在加载时重排为 output-major 布局。
- 每层专家权重使用连续 USM slab。
- down kernel 每个 work-group 放置两个 subgroup，即同时处理两行输出。
- 权重保留在 shared USM；device USM、Q8/DP4A 和 active-expert 单 launch 默认关闭。

## 测试环境与工作负载特点

主要测试环境：

- Linux。
- Intel oneAPI DPC++/C++ 2026.1。
- Intel iGPU，SYCL Level Zero backend。
- Qwen3.5-35B-A3B-GPTQ-Int4。
- `hidden_size=2048`、`moe_intermediate_size=512`、`group_size=128`。
- 每个 token 最多路由 8 个专家，实测 active expert 平均约 7.3。

decode 阶段不是大矩阵 GEMM，而是大量 `M=1` GEMV：

```text
gate/up: N=512,  K=2048
down:    N=2048, K=512
```

这种工作负载主要受以下因素限制：

- INT4 权重读取和内存访问合并效率。
- 数百个小 kernel 的提交、排队和依赖管理成本。
- 稀疏专家权重在共享内存中的访问局部性。
- iGPU 与 CPU 共享物理内存时，不同 USM allocation 类型带来的映射和调度差异。

因此，大 batch 或更高理论整数算力并不一定更快；布局、访问模式和提交粒度更关键。

## 从 llama.cpp SYCL 后端借鉴的设计

对比了 llama.cpp 的 Linux SYCL MMVQ 和 MoE 实现，重点包括：

- 对量化权重做一次性 reorder，使 subgroup lane 读取连续数据。
- 将量化 block 中的不同字段组织为更适合 device 读取的 SoA/重排布局。
- Q8 激活量化与 DP4A 向量点积。
- 在 MoE 路径中把 expert 作为 ND-range 维度，并提供 fused expert-ID GEMV。
- 使用 subgroup reduction，让一个 subgroup 协作计算一行输出。

KTransformers 没有直接照搬全部策略，而是逐项验证：

- 权重 reorder 对本项目收益最大，正式保留。
- 连续专家权重 slab 和 subgroup 内多输出行有稳定收益，正式保留。
- Q8 + DP4A 在当前 GPTQ 布局和 iGPU 上没有收益，默认关闭。
- active experts 单 kernel 会破坏当前访问局部性并放大调度问题，默认关闭。
- Level Zero immediate command list 和 in-order queue 均未带来收益。

这说明 llama.cpp 的优化思路可以借鉴，但 kernel 粒度和量化格式必须结合本项目的模型形状重新调优。

## 核心实现

### 1. Decode 分阶段 hook 与计时

`kt-kernel/operators/avx2/moe_base.hpp` 为派生后端提供：

- `use_fused_gate_up_decode()`
- `decode_gate_up_activation(...)`
- `use_fused_down_decode()`
- `decode_down_projection(...)`

同时通过 `KT_MOE_DECODE_TRACE_EVERY` 输出：

```text
[MOE decode] layer=... avg_total=... avg_setup=... avg_pack=...
avg_gate_up=... avg_activation=... avg_down_pack=... avg_down=...
avg_merge=... avg_active=... gate_up_fused=...
```

这些 hook 让 SYCL 后端可以替换 decode 的 gate/up、activation 和 down 流程，同时不改变其他
AVX2/AMX 后端。

### 2. Gate/up 融合和异步 pipeline

每个 active expert 的 gate 和 up projection 在同一 kernel 内计算，共享输入读取，并直接完成
SwiGLU/SiLU activation，输出到 down projection 的输入缓冲。这样消除了：

- gate 和 up 两次独立 kernel 提交。
- 中间结果回到 CPU。
- CPU activation。
- activation 后再次 pack 到 down 输入。

所有 active expert kernel 先异步 submit。down kernel 依赖对应 expert 的 gate/up event，不需要先等待
所有 gate/up 完成，从而形成细粒度 pipeline。

### 3. Packed subgroup kernel

GPTQ INT4 权重以一个 `uint32_t` 保存 8 个 4-bit 值。packed kernel 让 subgroup lane 每次读取一个
完整 packed word，再在寄存器中解包 8 个 nibble，减少循环和地址计算。

最终 gate/up 和 down 都使用：

```bash
KT_SYCL_INT4_GATE_UP_PACKED=1
KT_SYCL_INT4_GATE_UP_PACKED_SUBGROUP=16
KT_SYCL_INT4_DOWN_PACKED=1
KT_SYCL_INT4_DOWN_PACKED_SUBGROUP=16
```

### 4. Output-major 权重重排

原始 GPTQ 布局为：

```text
qweight: [K/8, N]
scales:  [K/group_size, N]
```

加载到 SYCL buffer 时重排为：

```text
qweight: [N, K/8]
scales:  [N, K/group_size]
```

对于 `group_size=128` 和 SG16，一个 group 正好包含 16 个 packed `uint32_t`。重排后 16 个
subgroup lane 可以读取同一输出行中连续的 16 个 word，避免原布局中以 `N` 为跨度的离散访问。

这是后半段调优中最重要的优化。启用后，一次非定长完整模型测试从约 `112 ms TPOT` 降到：

```text
Total: 91341ms | TTFT: 3199ms | TPOT: 68.9ms | In: 4 | Out: 1283
```

对应环境变量：

```bash
KT_SYCL_INT4_WEIGHT_REORDER=1
```

### 5. 连续专家权重 slab

每一层的 gate、up、down 专家权重分别放入连续 USM slab，各 expert 的 `BufferB` 绑定到 slab
中的固定 offset。该设计减少了大量独立 USM allocation，改善了映射、页表和权重地址管理。

需要区分：连续权重 slab 只是存储优化，不代表把所有 active experts 合并为一个 kernel。最终路径仍然
保留每个 expert 的小 kernel 异步提交。

对应环境变量：

```bash
KT_SYCL_INT4_CONTIGUOUS_WEIGHTS=1
KT_SYCL_INT4_EXPERT_BATCH=0
```

### 6. Down WG2

down projection 使用 SG16。最终让一个 work-group 包含两个 subgroup，每个 subgroup 负责一行输出：

```bash
KT_SYCL_INT4_DOWN_WG_ROWS=2
```

这减少了极小 work-group 的数量，并保持相邻输出行的权重访问和 activation cache 局部性。WG4、WG8、
WG16 和 WG32 在微基准中没有继续改善，WG2 是当前稳定默认值。

## 测试方法的修正

早期测试让模型自由生成到 EOS，不同轮次的输出 token 数可能相差很大，TPOT 会受到生成内容、频率变化、
热机状态和统计区间影响。因此早期结果只能用于发现数量级变化，不能用于小于几个百分点的 A/B。

后续新增 `perf-log/fixed-decode.py`，固定：

```text
temperature=0
seed=0
ignore_eos=true
max_tokens=1000
repeats=2
```

运行方式：

```bash
python perf-log/fixed-decode.py --max-tokens 1000 --repeats 2
```

后续所有小幅优化都应使用该方法，并同时比较：

- 两次客户端 TPOT。
- `[KT model timing]` 的长期累计值。
- 40 层 `[MOE decode]` 的平均值。
- active expert 数量是否接近。

## 优化实验汇总

下表中的早期 TPOT 来自不同输出长度，只用于展示优化方向；最终 fixed decode 才是严格结果。

| 实验 | 代表结果 | 结论 |
| --- | --- | --- |
| per-expert gate/up 融合、async down、event pipeline、fast SiLU | 早期 TPOT 从约 `214 ms` 降至约 `128 ms` | 有效，作为后续基线 |
| packed gate/up 与 packed down | TPOT 进入约 `112 ms` 区间 | 有效，保留 |
| output-major weight reorder | 单次 TPOT `68.9 ms` | 最大新增收益，保留 |
| 连续专家权重 slab | 单次 TPOT 约 `55.7 ms` | 有效，保留 |
| down WG2 | 非定长测试约 `51-53 ms`；fixed decode 平均 `53.85 ms` | 有稳定收益，保留 |
| output-lane：一 lane/输出 | TPOT 约 `55.5 ms` | 退化，关闭 |
| output-lane：四 lanes/输出 | TPOT 约 `54.4 ms` | 无稳定收益，关闭 |
| active experts 单 launch | 修正调度异常后 TPOT 仍约 `71.5 ms` | 明显慢于 per-expert async，关闭 |
| all-device USM weights | down 变快但 gate/up 明显变慢 | 无净收益，关闭 |
| down-only device USM | fixed decode 平均 `53.60 ms`，模型内部 `53.164 ms` | 与纯 shared 等价，关闭 |
| Level Zero immediate command list | 多专家微基准 `0.454 -> 0.531 ms` | 慢约 17%，关闭 |
| in-order queue | 微基准略慢 | 关闭 |
| 静态固定形状 kernel | 通用与静态 WG2 都约 `0.285 ms` | 已受权重带宽限制，无收益 |
| FP16 scale 存储 | down-only 微基准约 2% 收益 | 整机预期低于 1%，未集成 |

## Q8 + DP4A 组合实验

早期单独启用 Q8 激活量化没有积极效果。为了确认它是否需要和其他优化组合才会生效，后续重新实现了
只针对最大瓶颈 down projection 的组合路径：

- output-major reorder。
- contiguous weights。
- down WG2。
- 一次 batched device-side BF16 -> Q8 activation quantization。
- SYCL `dot_acc` DP4A。

专用微基准严格模拟：

```text
active=8, N=2048, K=512, group_size=128
```

结果：

```text
BF16 WG2 down:          约 0.282-0.288 ms
Q8 DP4A down（不量化）: 约 0.289 ms
Q8 quant + DP4A down:   约 0.295-0.297 ms
```

数值误差：

```text
max_abs = 0.059437
mean_abs = 0.010370
relative L2 = 0.006408
```

Q8 路径慢约 3%-5%。原因是当前 output-major INT4 down 已主要受权重带宽限制，DP4A 减少的是算术，
并没有减少主要权重读取量；动态量化、INT4 sign unpack 和额外 dependency 又增加了开销。

因此实验代码保留用于研究，但默认关闭：

```bash
KT_SYCL_INT4_DOWN_Q8=0
KT_SYCL_INT4_GATE_UP_Q8=0
```

不能仅凭“使用了 DP4A”判断会更快，必须把 activation quantization、kernel submit 和最终 wait 全部纳入
端到端计时。

## Shared USM 与 device USM 的最终 A/B

纯 shared 权重配置：

```bash
KT_SYCL_INT4_DEVICE_WEIGHTS=0
KT_SYCL_INT4_GATE_UP_DEVICE_WEIGHTS=0
KT_SYCL_INT4_DOWN_DEVICE_WEIGHTS=0
```

fixed decode：

```text
54.2 ms
53.5 ms
平均 53.85 ms
KT model timing @2000 calls = 53.170 ms
```

混合配置（gate/up shared、down device）：

```text
53.9 ms
53.3 ms
平均 53.60 ms
KT model timing @2000 calls = 53.164 ms
```

客户端只差 `0.25 ms`，约 0.47%，模型内部长期累计值只差 `0.006 ms`，属于测试波动。

分层日志还显示：

```text
纯 shared: avg_total=0.659ms, gate submit=0.058ms, down=0.582ms
混合 USM: avg_total=0.668ms, gate submit=0.087ms, down=0.562ms
```

down device USM 本身更快，但会间接增加 gate/up 提交和调度时间，最终完全抵消。因此正式默认值继续使用
纯 shared USM。

## 当前推荐配置

`perf-log/35b-build-sycl-int4.sh` 当前默认值已经对应最终组合。关键配置为：

```bash
KT_SYCL_INT4_DECODE_MODE=per_gemm
KT_SYCL_INT4_PER_GEMM_KERNEL=subgroup
KT_SYCL_INT4_PER_GEMM_SUBGROUP=32

KT_SYCL_INT4_GATE_UP_FUSE=1
KT_SYCL_INT4_GATE_UP_BATCH=0
KT_SYCL_INT4_GATE_UP_ASYNC=1
KT_SYCL_INT4_GATE_UP_PACKED=1
KT_SYCL_INT4_GATE_UP_PACKED_SUBGROUP=16

KT_SYCL_INT4_DOWN_ASYNC=1
KT_SYCL_INT4_DOWN_PACKED=1
KT_SYCL_INT4_DOWN_PACKED_SUBGROUP=16
KT_SYCL_INT4_DOWN_WG_ROWS=2

KT_SYCL_INT4_GATE_UP_DOWN_PIPELINE=1
KT_SYCL_INT4_WEIGHT_REORDER=1
KT_SYCL_INT4_CONTIGUOUS_WEIGHTS=1
KT_SYCL_INT4_SPECIALIZE=1
KT_SYCL_INT4_FAST_SILU=1

KT_SYCL_INT4_EXPERT_BATCH=0
KT_SYCL_INT4_DOWN_OUTPUT_LANES=0
KT_SYCL_INT4_GATE_UP_Q8=0
KT_SYCL_INT4_DOWN_Q8=0

KT_SYCL_INT4_DEVICE_WEIGHTS=0
KT_SYCL_INT4_GATE_UP_DEVICE_WEIGHTS=0
KT_SYCL_INT4_DOWN_DEVICE_WEIGHTS=0

KT_SYCL_QUEUE_PROFILING=0
KT_SYCL_QUEUE_IN_ORDER=0
SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
SYCL_UR_USE_IMMEDIATE_COMMANDLISTS=0
```

启动服务：

```bash
bash perf-log/35b-build-sycl-int4.sh
```

性能测试建议保持 `KT_TIMING=0`。需要分析分层耗时时再启用：

```bash
KT_TIMING=1 bash perf-log/35b-build-sycl-int4.sh
```

## 构建与扩展同步

使用 oneAPI 构建：

```bash
source /opt/intel/oneapi/setvars.sh
cmake --build \
  kt-kernel/build/temp.linux-x86_64-cpython-311/kt_kernel.kt_kernel_ext_Release \
  -j 8
```

本地开发时要确认 Python 实际加载的是新构建的扩展。如果使用源码目录中的 `.so`，需要把构建产物同步到
`kt-kernel/python/`，并做一次 import 检查，避免服务仍加载旧 kernel。

## 日志索引

本轮主要日志位于 `perf-log/`：

- `q8-scalar-fixed.log`：早期 Q8 scalar。
- `q8-dp4a.log`：早期 Q8 DP4A。
- `down-nibble.log`、`down-packed.log`：down packed 对比。
- `gate-up-packed.log`：gate/up packed。
- `contiguous-only.log`：连续权重 slab。
- `down-wg2.log`：down WG2。
- `down-output-lanes.log`、`down-output-lanes4.log`：output-lane 实验。
- `device-weights.log`、`hybrid-device-weights.log`：device 和混合 USM。
- `shared-fixed-server.log`、`shared-fixed-client.log`：最终纯 shared 固定测试。
- `hybrid-fixed-client.log`：混合 USM 固定测试。

## 调优经验总结

1. 先优化内存布局，再优化算术指令。output-major reorder 的收益远大于单独 Q8/DP4A。
2. 微基准必须覆盖完整依赖链。只测 DP4A kernel 而不测量化会得出错误结论。
3. 对小于 2% 的变化必须固定 token 数，并同时看模型内部累计计时。
4. iGPU 上更少的 kernel 不一定更快。单个 active-expert batch kernel 可能失去权重局部性并触发调度退化。
5. 单阶段变快不代表端到端变快。down device USM 的收益被 gate/up 的退化抵消。
6. 保留实验开关但默认关闭失败路径，便于后续在不同 iGPU、驱动和 oneAPI 版本上重新验证。

当前约一半 decode 时间已经不在 SYCL MoE 内部：`0.659 ms/layer * 40` 约为 `26.4 ms/token`，
而模型内部 decode 总计约 `53.2 ms/token`。如果未来继续优化整体性能，应优先重新拆解 attention、其他层和
框架调度开销，而不是继续假设瓶颈全部位于专家 GEMV。
