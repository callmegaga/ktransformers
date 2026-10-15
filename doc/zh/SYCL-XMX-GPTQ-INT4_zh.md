# Panther Lake iGPU 专用 GPTQ INT4 XMX 后端（实验性）

第二阶段的向量 GEMV/DP4A 候选、同调度比较和无子模块修改构建选项见 [投影候选说明](SYCL-IGPU-Projection-Candidates_zh.md)。第一版默认计算路径保持不变。

第三阶段的分段测量、NVIDIA pinned 数据交接对照与 down/merge 融合候选见 [执行链说明](SYCL-IGPU-Pipeline_zh.md)。

第四阶段的整专家融合与当前平台交付配置见 [整专家说明](SYCL-IGPU-Fused-Expert_zh.md)。

第五阶段的 prefill 数据复用、独立投影分块与 OpenVINO 对照见 [prefill 优化说明](SYCL-IGPU-Prefill_zh.md)。

该后端仍运行在 KTransformers 的混合推理框架中：NVIDIA GPU 执行 attention、dense 算子和配置在 CUDA 上的专家；Intel iGPU 执行其余 routed experts。它不是纯 iGPU 模型执行器，也没有把 OpenVINO 或 llama.cpp 作为运行时依赖。

旧 SYCL 后端仍为默认值。新后端必须显式选择，便于回退和比较。

## 选择后端

先按照 [SYCL 安装教程](SYCL-GPTQ-INT4-Tutorial_zh.md) 构建启用 `CPUINFER_USE_SYCL=1` 的扩展，并加载 oneAPI 环境。

已安装旧扩展时需要重新构建，避免继续加载旧的 site-packages 二进制：

```bash
source /opt/intel/oneapi/setvars.sh --force
CPUINFER_USE_SYCL=1 CPUINFER_FORCE_REBUILD=1 CPUINFER_PARALLEL=2 \
  python -m pip install --no-build-isolation ./kt-kernel
```

```bash
source /opt/intel/oneapi/setvars.sh --force
python -c "from kt_kernel import kt_kernel_ext; assert hasattr(kt_kernel_ext.moe, 'SYCLXMXGPTQInt4_MOE')"
```

沿用 `--kt-method SYCL_GPTQ_INT4`，通过环境变量选择实现：

| 环境变量 | 值 | 含义 |
|---|---|---|
| `KT_SYCL_INT4_BACKEND` | `legacy`（默认） | 原有 SYCL 后端 |
| `KT_SYCL_INT4_BACKEND` | `xmx` | 专用 XMX 执行器 |
| `KT_SYCL_XMX_MODE` | `fp16`（XMX 默认） | INT4 按 tile 反量化成 FP16，FP16 DPAS，FP32 累加 |
| `KT_SYCL_XMX_MODE` | `int8` | 激活按 group 动态量化，原生 S8×S4 DPAS，INT32 组内累加、FP32 跨组累加 |
| `KT_SYCL_XMX_ROWS` | `1` / `2` / `4` / `8`（默认） | 多 token 的 DPAS 行 tile，供实测调优 |
| `KT_SYCL_XMX_GRAPH` | `0`（默认）/ `1` | 单 token 流水线缓存 command graph |
| `KT_SYCL_XMX_HOST_QUANT` | `0`（默认）/ `1` | decode 输入由 CPU 量化，省去一个 GPU dispatch，实验性 |
| `KT_SYCL_XMX_FUSED_QUANT` | `0`（默认）/ `1` / `prefill` | 中间 Q8 量化融合；`prefill` 只在多 token 开启，要求 INT8 和 group=128，实验性 |
| `KT_SYCL_XMX_NUMA_DISPATCH` | `0`（默认）/ `1` | 对比直接提交与现有绑核 NUMA worker 提交，实验性 |
| `ONEAPI_DEVICE_SELECTOR` | `level_zero:gpu` | 使用 Level Zero GPU |

`int8` 模式目前是本机测试中更快的 XMX 路径，但有新增激活量化误差，因此不自动替换旧后端，也不自动替换 `fp16` 模式。未知的后端/模式值会报错，不会悄悄回退。

启动示例：

```bash
KT_SYCL_INT4_BACKEND=xmx \
KT_SYCL_XMX_MODE=int8 \
ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
SGLANG_DISABLE_CUDNN_CHECK=1 \
SGLANG_MAMBA_CONV_DTYPE=float16 \
python -m sglang.launch_server \
  --model /home/wy/models/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --kt-weight-path /home/wy/models/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --kt-method SYCL_GPTQ_INT4 \
  --kt-cpuinfer 10 --kt-threadpool-count 1 --kt-num-gpu-experts 8 \
  --tensor-parallel-size 1 --dtype float16 --attention-backend triton \
  --chunked-prefill-size 512 --max-prefill-tokens 4096 \
  --max-running-requests 1 --max-total-tokens 8192 --context-length 8192 \
  --mem-fraction-static 0.9 --disable-radix-cache \
  --disable-shared-experts-fusion --watchdog-timeout 1200 --trust-remote-code
```

日志应包含 `Created SYCL_XMX_GPTQ_INT4_MOE mode=int8`（或 `mode=fp16`）。构造标记只表示选择了该类；仍需确认服务成功启动和请求完成。

## 实现与限制

- 目标平台为 Intel Core Ultra 7 356H 的 Panther Lake iGPU（`intel_gpu_ptl_u`），oneAPI 2026.1.1；其他平台尚未验证。
- 单个 CPUInfer threadpool/NUMA part；不支持将同一专家拆到多个 NUMA part。
- 对称 GPTQ INT4，`desc_act=false`，group size 为正的 32 倍数；hidden/intermediate 必须整除 group size，并按 16 对齐。
- 权重常驻 packed INT4，无全模型 INT8/FP16 解包副本。布局为 `[expert, output_tile16, K/8, output_lane16]`。
- GPTQ offset-binary nibble 在整数 DPAS 输入处通过 XOR 转换为有符号 S4。
- 整数 partial 每个 group 单独应用 activation scale 和 weight scale，不能跨 group 先累加再统一缩放。
- 当前路由任务表由 CPU 建立；gate/up 融合激活、down 投影、加权合并在 iGPU 上批量执行。输入不按专家重复复制；每个原始 token 只量化一次。
- 工作区在构造时预分配，避免 forward 期间 SYCL 分配。32GB 系统加载完整模型时可能有较高内存压力；需要记录 swap/内存状态。
- 默认在 CPUInfer 任务线程运行，可选在现有绑核 NUMA worker 上提交。NUMA worker 内异常会先捕获，再在任务线程重新抛出；无效请求和运行时异常可由 `CPUInfer.sync()` 报告。
- `fp16` 对超出 FP16 数值范围的激活有溢出风险。INT8 有量化误差；目前没有完成模型级困惑度或任务精度验收。
- 支持 mask、physical-to-logical 权重映射、无效/重复专家 ID 和 incremental 输出。mask 为加载时配置，不支持运行中动态迁移专家。
- 可选单 token command graph 按 top-k 和有效路由行数缓存，专家 ID/输入/路由权重仍从固定地址读取。30 次改变输入与路由的测试与直接执行逐位一致；首轮端到端测试未显示收益，因此默认关闭。设备端路由和访存流水线仍可优化。

## 正确性和单层性能测试

独立 DPAS/投影测试，不需要模型或 Python 扩展：

```bash
source /opt/intel/oneapi/setvars.sh --force
icpx -fsycl -O2 kt-kernel/test/sycl/test_xmx_dpas.cpp -o /tmp/test_xmx_dpas
ONEAPI_DEVICE_SELECTOR=level_zero:gpu /tmp/test_xmx_dpas
```

该测试精确比较 M=1/2/4/8 的有符号整数点积，并测试不同 group scale 和不足 8 行的 tile。

单层完整 MoE 对照（latency 包含 CPUInfer submit/sync，不是仅 GPU kernel 时间）：

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:gpu python kt-kernel/bench/bench_sycl_xmx_moe.py \
  --backends legacy,packed,xmx-fp16,xmx-int8 \
  --rows 1,64,512 --repeat 12 --check --output /tmp/moe-comparison.json

ONEAPI_DEVICE_SELECTOR=level_zero:gpu python kt-kernel/bench/bench_sycl_xmx_moe.py \
  --backends xmx-fp16,xmx-int8 --experts 16 --hidden 256 --intermediate 128 \
  --topk 4 --rows 1,2,7,17 --check --edge-cases --output /tmp/moe-edge.json
```

使用真实第 0 层权重：

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:gpu python kt-kernel/bench/bench_sycl_xmx_moe.py \
  --model-path /home/wy/models/Qwen3.5-35B-A3B-GPTQ-Int4 --layer 0 \
  --backends legacy,packed,xmx-fp16,xmx-int8 --rows 1,17,64 \
  --repeat 8 --check --output /tmp/moe-real-layer.json
```

测试使用同一份权重、输入和路由，CPU FP32 参考在阶段边界进行 BF16 舍入。融合实现与 CPU 的 gate/up 舍入位置不同，故误差不能简单归因于某一种计算精度。

## 2026年10月14日首轮结果

硬件：356H、Intel iGPU、RTX 5060 Laptop、约 30GiB 可见系统内存。模型尺寸：256 experts，top-k=8，hidden=2048，intermediate=512，group=128，10 CPUInfer threads。

随机权重、完整单层 MoE，12 次计时的中位数：

| 后端 | 1 token (ms) | 64 tokens (ms) | 512 tokens (ms) |
|---|---:|---:|---:|
| 旧 SYCL | 0.604 | 14.293 | 122.496 |
| packed VNNI | 0.189 | 6.351 | 21.438 |
| XMX FP16 | 0.841 | 8.199 | 29.892 |
| XMX INT8 | 0.415 | 5.350 | 15.189 |

INT8 XMX 对 512-token 单层测试约为旧 SYCL 的 8.1 倍速度、packed VNNI 的 1.41 倍速度；单 token 的 isolated latency 仍慢于 packed VNNI。不要从单层 latency 直接推算端到端 decode。

真实第 0 层权重、随机 BF16 输入：XMX FP16 相对 RMSE 约 0.43%–0.45%，XMX INT8 约 1.32%–1.40%，packed VNNI 约 1.37%–1.42%。这是算子数值对照，不是模型质量评测。

本机端到端短测（512-token chunk、CUDA Graph 开启、1K 输入、最多 64 输出）：packed 为 423 tok/s 输入吞吐、35.5 tok/s decode；XMX INT8 为 705 tok/s、38.1 tok/s。这只是短测，不是完整 sweep 或跨框架最优性能结论。

完整端到端对照由 `/home/wy/work/kt-perf-log/perf.py` 执行：

```bash
python /home/wy/work/kt-perf-log/perf.py \
  --backend sycl-int4,packed,sycl-xmx-int8 --repeats 2
```

该脚本支持 `sycl-xmx`（FP16）和 `sycl-xmx-int8`，保存实际后端标记、CSV、服务日志和逐请求 JSON。吞吐使用服务返回的 tokenizer usage，而不是 SSE chunk 数；decode 分子扣除 TTFT 中已包含的首个 token。`prefill token/s` 是输入 token 数/TTFT，仍包含首 token 计算、传输及服务开销，不是纯 prefill kernel 吞吐。新结果不能直接与旧 chunk-count 口径的 CSV 比较。


完整 sweep 已完成（每项 2 次请求，取中位数，最多 300 输出 tokens）：

| 后端/指标 | 1K | 2K | 4K | 7.5K |
|---|---:|---:|---:|---:|
| 旧 SYCL 输入 tok/s | 87.3 | 109.6 | 117.6 | 123.5 |
| packed VNNI 输入 tok/s | 509.1 | 577.0 | 579.2 | 578.4 |
| XMX INT8 输入 tok/s | 706.8 | 736.3 | 742.0 | 739.8 |
| 旧 SYCL decode tok/s | 24.9 | 25.0 | 23.8 | 24.1 |
| packed VNNI decode tok/s | 38.0 | 35.8 | 35.9 | 36.4 |
| XMX INT8 decode tok/s | 37.5 | 36.9 | 36.9 | 35.8 |

XMX INT8 的输入吞吐比 packed 高约 28%–39%，decode 大致持平，不构成每个场景都最快的结论。实际输入长度为 1029、2053、4101、7685 tokens，均返回 300 completion tokens。

后续优化组合（prefill 融合量化、NUMA worker 提交，3 次请求中位数）在 1K/4K/7.5K 的输入吞吐为 747.0/788.4/782.5 tok/s，decode 为 37.5/36.4/35.8 tok/s。它是当前完成测试的较快 prefill 配置；decode 仍不稳定超过 packed。建议显式启用以下设置，并保留质量/性能回归对照：

```bash
KT_SYCL_INT4_BACKEND=xmx \
KT_SYCL_XMX_MODE=int8 \
KT_SYCL_XMX_FUSED_QUANT=prefill \
KT_SYCL_XMX_NUMA_DISPATCH=1 \
KT_SYCL_XMX_GRAPH=0 \
KT_SYCL_XMX_HOST_QUANT=0 \
ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
python -m sglang.launch_server ...
```

`...` 需替换为本文件上面的完整服务参数，不能直接作为可执行命令。所有实验优化开关仍默认关闭，以免改变原有数值和调度行为。

额外测试（graph=1，每项 3 次）：1K/4K/7.5K 输入吞吐为 704.0/740.2/741.6 tok/s，decode 为 37.2/36.7/35.8 tok/s，未超过直接执行，故 graph 保持 opt-in。

M tile 扫描（512-token 单层、15 次）：M=1/2/4/8 中位数分别为 32.75/24.25/18.11/15.16 ms，因此当前默认保留 M=8。isolated decode 会受 cache/频率影响，需要结合端到端结果判断。

中间量化融合的单层测试（20 次）：512-token 中位数由 15.09 ms 降至 14.69 ms；对 1K/2K/4K/7.5K 的端到端输入吞吐为 750.7/776.4/786.5/789.5 tok/s，但统一融合的 decode 为 36.0/34.7/34.4/35.2 tok/s，低于未融合版本。因此提供 `KT_SYCL_XMX_FUSED_QUANT=prefill`，独立选择 prefill 融合而不改变 decode 路径；是否使用以完整测试结果为准，不自动打开实验设置。

固定英文、中文和代码片段的 teacher-forced logprob smoke test（573 个评分 token）：旧 SYCL、packed、XMX INT8 的 mean NLL 分别为 0.72425、0.72809、0.72978。XMX 相对旧 SYCL 的平均绝对 logprob 差为 0.01392，最大为 1.11123；不能只看平均值宣称质量等价。这些片段包含重复文本，不是标准数据集的模型质量结论。

WikiText-2 test 的前 4096 个 tokens（16 个独立 256-token 段，评分 4080 个 tokens）：旧 SYCL/packed/XMX INT8 的困惑度分别为 10.82014/10.81997/10.83152。XMX 相对旧 SYCL 的平均绝对 logprob 差为 0.05024，最大为 1.43895。这是固定截断样本对照，未覆盖完整 WikiText-2、长上下文或下游任务，不能作为完整精度验收。512-token 段的 logprob 请求曾因 NVIDIA logits 缓冲区 OOM 失败，故质量测试使用 256-token 段，与性能 sweep 分开执行。
