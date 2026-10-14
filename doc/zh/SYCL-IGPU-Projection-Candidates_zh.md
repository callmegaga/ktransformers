# iGPU 第二阶段：向量 GEMV、DP4A 与 XMX 同调度比较

本阶段在第一版 `SYCLXMXGPTQInt4_MOE` 执行器内新增投影候选，不改变 Python 方法名，默认仍是原来的 XMX 路径。目的不是预先宣布某种指令最快，而是把相同权重布局、路由和同步下的候选分开比较。

第三阶段的完整执行链诊断与 decode down/merge 融合见 [执行链优化说明](SYCL-IGPU-Pipeline_zh.md)。

## 候选与数值边界

- `xmx`：第一版 DPAS 实现，精度由 `KT_SYCL_XMX_MODE=fp16/int8` 决定。
- `vector`：BF16 激活直接加载成 FP32，INT4 按 group 解包后在向量单元做 FP32 FMA。这里的 W4A16 指 16-bit 激活存储，不表示 FP16 乘法/累加。它不引入额外 INT8 激活量化，也不受 FP16 激活最大值限制。
- `dp4a`：group-wise S8 激活，INT4 在寄存器中展开成 S8，向量 DP4A 点积、INT32 组内累加、FP32 跨组累加。ESIMD `dp4a` 的第一个参数是累加器，与某些其它 API 的参数顺序不同。

- `xmx-split`：保留 XMX 精度/布局，多个 work-items 分割 K groups，SLM 合并 FP32 partial 后执行激活/输出。这是并行组织候选，不是另一种指令集。

所有候选读取 `[expert, N/16, K/8, 16]` packed INT4；无常驻 INT8 或 FP16 权重副本。向量 work-item 拥有 16 个输出，用 1/2/4/8 个 work-items 分割 K groups，并通过 SLM 合并 partial。第一版 gate/up 激活融合、输入复用、设备端专家合并、CPUInfer 提交路径保持共用。

## 选择开关

| 开关 | 默认 | 值 |
|---|---|---|
| `KT_SYCL_XMX_GATE_KERNEL` | `xmx` | `xmx` / `xmx-split` / `vector` / `dp4a` / `inherit`，多 token gate/up |
| `KT_SYCL_XMX_DOWN_KERNEL` | `xmx` | 同上，多 token down |
| `KT_SYCL_XMX_DECODE_GATE_KERNEL` | 继承 gate 设置 | 同上，单 token gate/up |
| `KT_SYCL_XMX_DECODE_DOWN_KERNEL` | 继承 down 设置 | 同上，单 token down |
| `KT_SYCL_XMX_VECTOR_SPLITS` | `4` | 向量 K 分割数：1/2/4/8 |
| `KT_SYCL_XMX_GATE_SPLITS` | 继承 VECTOR_SPLITS | vector/DP4A/xmx-split 的 gate/up 分割数 |
| `KT_SYCL_XMX_DOWN_SPLITS` | 继承 VECTOR_SPLITS | vector/DP4A/xmx-split 的 down 分割数 |
| `KT_SYCL_PROFILE` | `0` | `1` 启用 queue profiling，用于独立投影 benchmark |

`KT_SYCL_XMX_FUSED_QUANT=prefill` 可以与 vector/DP4A decode 共存，因为融合只影响多 token 路径。全阶段融合 `=1` 要求 decode gate 仍为未分割的 INT8 XMX；不兼容的选择会报错，不会静默忽略。混合模式每个投影单独判断是否需要量化，例如 vector gate + INT8 XMX down 只量化中间激活。

decode 对照示例（其余模型参数同第一版）：

```bash
KT_SYCL_INT4_BACKEND=xmx \
KT_SYCL_XMX_MODE=int8 \
KT_SYCL_XMX_FUSED_QUANT=prefill \
KT_SYCL_XMX_NUMA_DISPATCH=1 \
KT_SYCL_XMX_DECODE_GATE_KERNEL=vector \
KT_SYCL_XMX_DECODE_DOWN_KERNEL=xmx \
KT_SYCL_XMX_GATE_SPLITS=2 \
KT_SYCL_PROFILE=0 \
ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
python -m sglang.launch_server ...
```

`...` 需要替换为第一版文档中的完整参数。该配置是对照候选，不是已经证明优于默认配置的推荐。

## 不修改 llama.cpp 子模块的构建

新增 opt-in 构建开关 `KTRANSFORMERS_LLAMA_BUILD_COPY=ON`。它将 vendored llama.cpp 复制到构建目录，应用项目已有的 MXFP4/NumPy 补丁，并优先使用副本头文件；原子模块不会变 dirty。

```bash
source /opt/intel/oneapi/setvars.sh --force
CPUINFER_USE_SYCL=1 CPUINFER_LLAMA_BUILD_COPY=ON CPUINFER_PARALLEL=2 \
  python -m pip install --no-build-isolation ./kt-kernel
```

已有 CMake 构建可显式传 `-DKTRANSFORMERS_LLAMA_BUILD_COPY=ON`。默认 OFF 保持原构建行为；更换 vendored llama.cpp 版本或源文件后，应清理构建目录的 `patched_deps/llama.cpp`，避免复用旧副本。

## 独立正确性测试

```bash
source /opt/intel/oneapi/setvars.sh --force
icpx -fsycl -O2 kt-kernel/test/sycl/test_vector_projection.cpp -o /tmp/test_vector_projection
ONEAPI_DEVICE_SELECTOR=level_zero:gpu /tmp/test_vector_projection

KT_TEST_SYCL_XMX=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  python -m pytest -q kt-kernel/test/test_sycl_xmx_hardware.py
```

独立测试覆盖负值 nibble/激活、DP4A 精确结果、group=32/128、split=1/2/4/8、gate/up/down、尾部行、路由索引和 SwiGLU clamp。硬件集成测试覆盖 mixed projections 和 command graph 动态输入。向量与 XMX 的运算/舍入顺序不同，不要求与浮点参考逐位一致。

## 单独测量投影

```bash
icpx -fsycl -O3 kt-kernel/bench/bench_sycl_projections.cpp -o /tmp/bench_sycl_projections
KT_SYCL_PROFILE=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  /tmp/bench_sycl_projections --repeat 50 > /tmp/projections-hot.csv
KT_SYCL_PROFILE=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  /tmp/bench_sycl_projections --rows 1 --active 1 --rotate --repeat 60 > /tmp/projections-rotate.csv
```

`wall_us` 包含 submit/wait；`device_us` 是 event 的 command_start/end 差，不包含输入量化、路由、合并或 CPUInfer。`--rotate` 轮换专家，每个候选前做 64MiB 读写来置换缓存（不计入计时）；它是人为缓存压力，不保证每一级缓存都严格 cold。候选顺序在每轮随机打乱。不要同时运行其它 GPU 测试、推理服务或编译。

## 完整 MoE 与交错比较

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
KT_SYCL_XMX_GATE_SPLITS=2 KT_SYCL_XMX_DOWN_SPLITS=1 \
KT_SYCL_XMX_FUSED_QUANT=0 KT_SYCL_XMX_GRAPH=0 \
KT_SYCL_XMX_DECODE_GATE_KERNEL=inherit KT_SYCL_XMX_DECODE_DOWN_KERNEL=inherit \
python kt-kernel/bench/bench_sycl_xmx_moe.py \
  --backends xmx-int8,dp4a,vector,dp4a-xmx,vector-xmx,packed \
  --rows 1,64,512 --route-sets 32 --repeat 40 --check --output /tmp/full-moe.json

ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
KT_SYCL_XMX_GATE_SPLITS=2 KT_SYCL_XMX_DOWN_SPLITS=1 \
python kt-kernel/bench/bench_sycl_decode_interleaved.py \
  --samples 100 --rotate --output /tmp/decode-interleaved.json
```

完整 MoE 时间包含量化、激活、down、合并和 CPUInfer submit/sync。交错测试保留多个执行器，使用相同输入并随机打乱执行顺序，保存 p10/p90 和所有样本。多个执行器的权重副本会改变缓存工作集，因此它也是交叉验证，不应单独替代整模型测试。

初步投影观察（356H，2048/512，group=128，60 samples，旋转专家+64MiB cache displacement）：

| 情况 | XMX INT8 | 最快非 XMX 候选 |
|---|---:|---:|
| gate/up，1 活跃专家 | 36.1 us | DP4A split=8，27.4 us |
| gate/up，2 活跃专家 | 41.9 us | DP4A split=4，39.5 us |
| gate/up，4 活跃专家 | 66.1 us | DP4A split=2，62.7 us |
| gate/up，8 活跃专家 | 112.4 us | DP4A split=2，122.5 us |
| down，8 活跃专家 | 80.4 us | vector split=1，126.4 us |

这支持按阶段和规模比较，**不支持直接宣称非 XMX decode 更快**。完整随机 MoE 的 vector gate + XMX down 一度更快，但端到端首轮没有复现稳定收益；目前保留第一版默认分派，不加入未经验证的自动阈值。

全部本机结果在 `/home/wy/work/kt-perf-log/igpu-v2/`，服务 CSV/JSON 在其相邻 `perf/`。排除的并发测试和错误 DP4A 参数试验数据不用于最终结论。


随机交错完整 MoE decode 测试（150 samples、轮换专家，gate split=2/down split=1）：

| 活跃专家数 | 全 XMX INT8 (ms) | vector gate + XMX down (ms) | split-K XMX gate + XMX down (ms) |
|---|---:|---:|---:|
| 1 | 0.0806 | 0.0893 | 0.0681 |
| 2 | 0.0962 | 0.1014 | 0.0811 |
| 4 | 0.1417 | 0.1356 | 0.1233 |
| 8 | 0.2270 | 0.2359 | 0.2086 |

交错结果没有复现顺序测试中向量 gate 对 8 专家的优势，说明仅靠顺序测试会误判。split-K XMX gate 在该单层测试中降低约 8%–16% 的时延；端到端结果见下节，不能将该比例直接当作整模型加速。

8 专家的独立投影测试：未分割/2-way/4-way/8-way INT8 XMX gate 分别约 111.46/93.75/101.72/112.60 us，down 以未分割的约 80.83 us 最快。因此只测试 decode gate 的 2-way split，保留 prefill 和 down 原路径，不统一增加 split。

第一轮整模型测试（每项 3 次、最多 300 输出 tokens）：

| decode 投影 | 1K decode tok/s | 4K decode tok/s | 7.5K decode tok/s |
|---|---:|---:|---:|
| XMX/XMX | 37.1 | 36.5 | 35.6 |
| vector/XMX | 36.7 | 35.7 | 34.8 |
| DP4A/XMX | 36.1 | 35.6 | 34.7 |
| vector/vector | 34.1 | 33.5 | 32.3 |

prefill 均为原 XMX 路径，输入吞吐约 755–794 tok/s。向量和 DP4A 未证明端到端收益，因此不会自动替换默认 XMX。GPU kernel 减少并不等于整模型更快，真实路由、缓存、系统频率以及 CPU/NVIDIA/iGPU 同步仍是测量的一部分。


## 端到端反序对照与当前选择（2026-10-14）

使用相同的第二阶段二进制，先测 split-K，再测未分割 XMX；每档 5 次请求，
每次 300 输出 tokens。模型为本机 `Qwen3.5-35B-A3B-GPTQ-Int4`，
prefill chunk=512，`NUMA_DISPATCH=1`、`FUSED_QUANT=prefill`，
iGPU command graph、host quantization 和 queue profiling 均关闭。
仅单 token gate/up 改为 2-way split，down 与多 token 路径不变。

| Prompt 档位 | 未分割 XMX decode (tok/s) | split-K gate decode (tok/s) | 中位数变化 |
|---|---:|---:|---:|
| 1K | 37.09 | 37.52 | +1.14% |
| 4K | 36.22 | 36.97 | +2.05% |
| 7.5K | 35.58 | 35.85 | +0.74% |

prefill 输入吞吐分别为未分割 760.3/793.1/790.7 tok/s、split 764.6/790.4/788.4 tok/s，
没有更换 prefill kernel。这里的 prefill 指实际 prompt tokens / TTFT，包含请求和框架开销，
不是独立矩阵乘吞吐。1K/4K/7.5K 的实际输入分别是 1029/4101/7685 tokens；
decode 使用服务返回的 usage，排除首输出 token。

这轮没有并发 GPU 测试或编译干扰，但仍不是随机交错的整模型试验。
例如 7.5K 的每次 decode 结果范围为未分割 35.27–36.20、split 35.71–35.89 tok/s，
范围重叠；5 次样本不足以证明跨工作负载、跨运行条件的稳定提升。
原脚本 prompt 为重复 `word` 的合成输入，真实对话/代码生成会有不同专家路由。
所以当前结论是：**2-way split-K gate 有小幅正向观测，保留为 opt-in；
不启用自动 vector/DP4A 选择，也不改变第一版默认 XMX 路径。**

在第一版完整启动命令上追加以下配置可复现这个候选：

```bash
export KT_SYCL_XMX_GATE_KERNEL=xmx
export KT_SYCL_XMX_DOWN_KERNEL=xmx
export KT_SYCL_XMX_DECODE_GATE_KERNEL=xmx-split
export KT_SYCL_XMX_DECODE_DOWN_KERNEL=xmx
export KT_SYCL_XMX_GATE_SPLITS=2
export KT_SYCL_XMX_DOWN_SPLITS=1
export KT_SYCL_XMX_FUSED_QUANT=prefill
```

其它开关使用第一版推荐配置。要回退，只需设置
`KT_SYCL_XMX_DECODE_GATE_KERNEL=xmx`。若原配置使用 `FUSED_QUANT=1`，
必须改为 `prefill` 或 `0`，因为 split-K gate 不支持输出量化融合。
原始请求与逐次数据见 `igpu-v2/e2e-control-*.log`、`e2e-summary.json` 和 `split-control-comparison.json`。

## 最终数值与构建检查

- 硬件集成、分派、SwiGLU guards 与 loader 回归：**80 passed / 3 skipped**。
  三项 skip 都是有意不支持的 `FUSED_QUANT=1` 与非未分割 XMX decode gate 组合；
  不兼容组合的构造错误另有检查。另覆盖 decode-only 覆盖不影响 fused prefill、
  host quant 开关和执行器构造后环境变化的组合。
- 外部 `perf.py` 回归：**15 passed**，包括新开关的指纹和显式覆盖。
- 独立 signed DP4A、vector gate/down、group=32/128、split=1/2/4/8 检查通过；
  signed DPAS 和 FP16/INT8 XMX split projection 检查通过。
- 真实模型第 0 层权重，随机 BF16 输入和路由，rows=1/17/64 检查通过。
  对 BF16 浮点参考的 relative RMSE：vector 约 0.42%–0.45%；
  原 INT8 XMX、split-K XMX 和 DP4A 均约 1.37%–1.50%。
  这些是算子级数值检查，**不是整模型困惑度或下游任务精度验收**；
  该顺序测试的耗时也不用于候选性能排序。
- CMake 副本模式重复 configure/build 成功；已有补丁识别为已应用，不重复打补丁。
  构建与当前安装扩展的 SHA256 一致，`third_party/llama.cpp` 无工作区修改。

完整原始记录、测试日志、源文件/二进制指纹位于
`/home/wy/work/kt-perf-log/igpu-v2/`。第二、三阶段代码通过后续 checkpoint commit 保存；不改外部 OpenVINO/llama.cpp 源码。

## 后续优化优先级

下一轮优先定位完整 MoE 的路由构造、量化、kernel 提交/等待、CPU/NVIDIA/iGPU 同步开销，
以及多层动态专家工作集的缓存影响。已有数据显示单层 kernel 收益远大于整模型收益，
不能继续只用 hot-cache 投影微基准判断改动。后续需要更真实的 prompt 与路由、交替顺序重复运行，
并将数值/模型质量检查和计时分开执行；暂不依据活跃专家数量加入未经验证的自动分派阈值。
