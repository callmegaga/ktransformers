# iGPU 第三阶段：完整执行链测量与 decode down/merge 融合

本阶段接续 [投影候选对照](SYCL-IGPU-Projection-Candidates_zh.md)，不改第一版默认分派，
不修改 `third_party/llama.cpp` 或外部 OpenVINO/llama.cpp 源码。
目的在于区分算子计算、提交/等待、NUMA 调度和 NVIDIA 数据交接，
避免继续只凭 hot-cache GEMV 微基准决定整模型优化。

## 分段测量接口

| 开关 | 默认 | 说明 |
|---|---|---|
| `KT_SYCL_XMX_TRACE` | `0` | `host`：主机路由/copy/quant/submit/wait/转换；`device`：另记 SYCL event 时间 |
| `KT_SYCL_XMX_TRACE_EVERY` | `0` | 非零时每 N 次调用输出本层本阶段的累计 `KT_SYCL_PIPELINE` JSON |
| `KT_SYCL_PROFILE` | `0` | `device` trace 需要在进程首次创建 SYCL queue 前设为 `1` |
| `KT_SYCL_XMX_DECODE_MERGE` | `separate` | `fused`：单 token down 与专家加权合并融合；实验选项 |

Python 原生执行器增加 `profile_snapshot()` 与 `reset_profile()`。
调用 snapshot/reset 前应先 `CPUInfer.sync()`，避免把正在执行的调用分到两个统计窗口。
默认没有计时、event timestamp 查询或日志输出；仍有轻量开关分支。

主机字段是同一个 steady_clock 的时间差。设备字段只用 SYCL event 的
command_start/end，**不将主机/CUDA/device 时间戳相减**。
全部事件在最终完成后统一查询，避免每段 wait 打断原依赖链。
`host_wait_us` 是主机阻塞时间，不是额外的 GPU 同步耗时：
它与设备执行重叠，不能与 `device_kernels_us` 相加。
wrapper 包含 NUMA 调度、执行器、BF16 输出转换及 trace 统计开销；
wall 还包含 CPUInfer 队列/唤醒和 Python 提交。

`device` trace 与 command graph 不兼容，会显式报错；
graph 可用 `host` trace 做整次 submit/wait 测量，不伪造 graph 内部节点时间。
诊断插桩、累计统计与可选日志会影响耗时，因此性能排序另用 `TRACE=0 PROFILE=0`。

## 完整链 benchmark

新增 `kt-kernel/bench/bench_sycl_pipeline.py`。所有候选共用权重/输入/路由，
每轮随机交错执行，保存原始样本、p10/p90 和中位数。
多执行器驻留会改变缓存工作集，这仍是诊断/对照，不能替代整模型测试。

```bash
source /opt/intel/oneapi/setvars.sh --force
# 设备分段：quant -> gate/up -> middle quant -> down -> merge
ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
python kt-kernel/bench/bench_sycl_pipeline.py \
  --trace device --rows 1,17,64 --rotate --samples 100 \
  --output /tmp/pipeline-device.json

# NVIDIA pinned D2H -> stream callback -> CPUInfer -> iGPU -> H2D
# 不开启 trace；CUDA event 时钟只测自身 chain，不与 SYCL event 时钟相减
ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
python kt-kernel/bench/bench_sycl_pipeline.py \
  --trace 0 --transport cuda --rows 1 --rotate --samples 300 \
  --variants baseline,split,fused-merge,split-fused-merge \
  --output /tmp/pipeline-cuda.json
```

CUDA 模式使用实际 pinned buffers 和 CPUInfer stream callbacks；
检查输出有限、候选 relative RMSE，以及 H2D 后设备输出与 pinned 输出一致。
该 benchmark 不包含真实 attention/dense 并行工作或完整 CUDA graph，
也不从 wall 与 wrapper 的差值推断精确的 D2H/H2D 各段耗时。

整模型诊断可在原启动命令添加：

```bash
export KT_SYCL_XMX_TRACE=device
export KT_SYCL_PROFILE=1
export KT_SYCL_XMX_GRAPH=0
export KT_SYCL_XMX_TRACE_EVERY=100
```

解析服务日志：

```bash
python kt-kernel/bench/summarize_sycl_pipeline.py /tmp/server.log \
  --output /tmp/pipeline-summary.json
```

记录是累计值，不可将所有日志条目相加。汇总同时给出最后累计记录、
相邻末两次记录的差分和逐层明细。前者包含首次 JIT/初始化；
后者排除更早的初始化，但 N>1 时可能遗漏最后不足 N 次调用的尾部。

## 2026-10-14 本机诊断

平台：Core Ultra 7 356H / `intel_gpu_ptl_u` / RTX 5060 Laptop，oneAPI 2026.1.1。
40 层 Qwen3.5-35B-A3B 的 1K 合成 prompt 诊断，原未分割 INT8 XMX、
NUMA dispatch=1、prefill quant fusion。**以下用于定位，不用于和未插桩吞吐比较。**

末次 100 decode 调用差分，40 层平均每层每次约：

| 部分 | 时间 |
|---|---:|
| 主机路由构造 | 1.26 us |
| 主机输入 copy | 0.28 us |
| 主机 submit（所有 kernels） | 10.97 us |
| gate/up 设备计算 | 109.39 us |
| down 设备计算 | 73.70 us |
| 输入量化 + 中间量化 | 7.55 us |
| 设备专家合并 | 5.10 us |
| 设备依赖链 gaps | 4.32 us |
| 执行器 host 总时延 | 210.44 us |
| TP wrapper host 总时延 | 247.44 us |

这些区间有重叠，不能直接求和。稳态计算仍以投影为主；
路由优化的上限较小。wrapper 与 executor 的约 37 us 差额包含 NUMA 调度、
BF16 转换和诊断统计，不应全部宣称为可消除的同步开销。
首次 layer0 decode 的累计 submit/JIT 开销很大，所以不能拿全程均值推断稳态瓶颈。
本次 `TRACE_EVERY=100` 未收集足够多 prefill 调用，没有 prefill 逐层统计。

## decode down/merge 融合候选

`KT_SYCL_XMX_DECODE_MERGE=fused` 为每个 N16 block 创建一组 8 个 work-items，
分别处理 decode 的最多 8 个 routed rows，将 down 输出按原路径舍入到 BF16 后，
在 SLM 中按原 top-k slot 顺序做 routing weight 合并。
不使用浮点原子，不缓存路由，不添加 INT8/FP16 常驻权重副本。
无效/屏蔽槽位被跳过，重复 expert 槽位保留各自权重，空路由仍输出零。

这样将 decode 链从 5 次 kernel 提交变为 4 次，
避免 down 中间缓冲区的写入与 merge 的读回，但增加 SLM 和 barrier。
只支持 top-k<=8、未分割 XMX down；gate/up 可独立使用 split-K。
多 token/prefill 完全沿用 separate merge。配置不支持时构造报错，不静默忽略。

300 samples、旋转专家、关闭 profiling 的 pinned CUDA 链初次交错结果：

| 候选 | wall 中位数 | CUDA chain 中位数 |
|---|---:|---:|
| baseline | 275.22 us | 261.42 us |
| split gate | 251.54 us | 237.92 us |
| fused merge | 266.08 us | 252.43 us |
| split gate + fused merge | 249.55 us | 235.90 us |

fused merge 相比 baseline 的单层正向观测约 3.3%；
在 split gate 基础上仅约 0.8%。设备分段也显示 barrier 会抵消部分节省，
不能把少一次 kernel 提交等同于大幅端到端收益。默认保持 separate。

原始记录在 `/home/wy/work/kt-perf-log/igpu-v3/`，第二阶段安装扩展备份在
`second-stage-backup/`。整模型对照结果与最终回归检查另见该目录 README。


## 同二进制整模型对照

关闭 trace/profiling，5 次请求/档，300 输出 tokens，prefill chunk=512、
NUMA dispatch=1、prefill quant fusion，iGPU graph/host quant 关闭。
顺序为组合候选、仅 split、baseline；所有配置用相同安装二进制。

| Prompt | 原 XMX | 仅 split gate | split gate + fused merge |
|---|---:|---:|---:|
| 1K | 37.12 tok/s | 37.69 tok/s | 37.93 tok/s |
| 4K | 36.62 tok/s | 37.00 tok/s | 37.26 tok/s |
| 7.5K | 35.69 tok/s | 36.23 tok/s | 36.33 tok/s |

组合候选相对原 XMX 约 +1.75%–2.19%，相对仅 split 的额外观测仅 +0.27%–0.71%。
逐次范围有重叠，顺序试验不能排除频率/温度变化，不据此宣布融合普遍更快。

新增 `perf.py --prompt-file` 测试，用保存的非重复词代码说明 prompt 反转顺序
（baseline 先、组合候选后），5 次请求，实际输入 97 tokens、输出 300 tokens：
原 XMX **38.43 tok/s**，组合候选 **39.02 tok/s**，观测约 **+1.55%**。
两组生成内容并不逐字相同，所以实际专家路由也可能不同；
这是第二种工作负载的性能 smoke，不是固定路由归因或质量等价证明。
两组 TTFT 分别约 222/236 ms；prefill kernel 未改，这个短 prompt 未观测到 TTFT 改善。

在后续 300 samples 的关闭插桩 pinned CUDA 复测中，baseline/split/fused/组合/direct
的 wall 中位数分别 282.99/269.47/281.62/266.39/285.34 us。
融合单独收益从前轮约 3.3% 缩小到约 0.5%，进一步说明需要保守解释微小差异。
原有 NUMA dispatch 继续保留；direct 路径在此没有胜出，不改通用 TaskQueue/WorkerPool。

### 可选启用与回退

在第一版 INT8 XMX 完整启动命令上添加：

```bash
export KT_SYCL_XMX_GATE_KERNEL=xmx
export KT_SYCL_XMX_DOWN_KERNEL=xmx
export KT_SYCL_XMX_DECODE_GATE_KERNEL=xmx-split
export KT_SYCL_XMX_GATE_SPLITS=2
export KT_SYCL_XMX_DECODE_DOWN_KERNEL=xmx
export KT_SYCL_XMX_DECODE_MERGE=fused
export KT_SYCL_XMX_FUSED_QUANT=prefill
export KT_SYCL_XMX_TRACE=0
export KT_SYCL_XMX_TRACE_EVERY=0
export KT_SYCL_PROFILE=0
```

只回退本轮融合设 `KT_SYCL_XMX_DECODE_MERGE=separate`；
再设 `KT_SYCL_XMX_DECODE_GATE_KERNEL=xmx` 就恢复原投影路径。
当前自动/默认分派仍不变，以上是实验选项。

### 最终检查

- 硬件集成、分派/guards/loader 与 trace 汇总回归：**115 passed / 3 skipped**。
  覆盖 FP16/INT8、graph on/off、top-k=1/2/8、量化融合 0/1/prefill、动态重复/无效/空路由、incremental、
  decode-only 与 prefill、无效参数、top-k>8 拒绝和 trace queue guard。
- `perf.py`：**17 passed**，覆盖新环境指纹及自定义 prompt 内容哈希。
- 独立 device trace 检查通过：5/4 commands、累积/清空与 graph guard。
- 真实第 0 层 rows=1/17/64、mask/physical permutation/duplicate/invalid route 与
  SwiGLU clamp/alpha 检查通过；数值检查不替代整模型质量验收。

构建/安装二进制指纹、所有源码/测试指纹与原始结果见 `igpu-v3/validation-manifest.json`。
上述结果为第三阶段完成时的记录；第二、三阶段代码通过后续 checkpoint commit 保存。
测试服务已停止，`third_party/llama.cpp` 无修改。
