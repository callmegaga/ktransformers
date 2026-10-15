# Panther Lake iGPU 整专家融合与本机性能配置

后续 prefill 优化及更新的交付配置见 [prefill 说明](SYCL-IGPU-Prefill_zh.md)；下述性能记录为第四阶段历史数据。

适用对象为当前 Core Ultra7 356H / `intel_gpu_ptl_u` 平台与
Qwen3.5-35B-A3B-GPTQ-Int4，H=2048、专家 I=512、group=128。
这里的性能最优指**本机完成测量的候选中较优配置**，不是所有实现/框架的全局最优证明。
原有跨平台 CPU、legacy SYCL 与第一版 XMX 路径仍可回退。

第二/三阶段 checkpoint：主仓库 `cacf1a8`，外部性能脚本 `ca59619`。
之前投影与执行链对照见 [投影候选](SYCL-IGPU-Projection-Candidates_zh.md) 和
[执行链说明](SYCL-IGPU-Pipeline_zh.md)。

## 整专家融合

新开关 `KT_SYCL_XMX_FUSED_EXPERT=1` 在单 token 时选择专用 kernel：

1. 一个 work-group 负责一个 routed expert row，32/64 个 ESIMD work-items；
2. 原 packed W4×group-wise S8 使用原生 S4×S8 XMX DPAS，组内 INT32 累加，组间 FP32 恢复 scale；
3. gate/up 可使用 1/2-way K split，SLM 合并 partial 后执行原 SwiGLU；
4. 按原路径舍入到 BF16，再在 SLM 内生成 group-wise S8 中间激活；
5. 同 work-group 执行 down，结果按原路径舍入到 BF16；
6. 独立设备 kernel 按原 top-k slot 顺序加权合并，没有浮点原子或跨 work-group barrier。

常规 decode 的 input quant / gate / middle quant / down / merge 五次提交
缩减为 input quant / fused expert / merge 三次。
中间激活与 Q8/scales 驻留 SLM，避免其全局写回/读回；没有额外常驻权重副本。
多 token/prefill 沿用原 M8 XMX + prefill quant fusion，不切换到该固定维度 kernel。

此模式要求 INT8、H=2048/I=512/group=128、XMX 或 split-XMX decode gate、未分割 XMX down、
gate split<=2，并要求 `KT_SYCL_XMX_DECODE_MERGE=separate`。
不兼容配置会显式报错，不静默覆盖选择。原 `FUSED_QUANT=prefill` 只影响多 token 路径。
设备诊断中的 gate event 包含整专家计算，不能把它当成单独 gate/up。

4-way 整专家候选未通过独立数值检查，**不进入可选实现**；失败记录被排除。
这不影响第二阶段的小 work-group 4-way projection 测试。

## 初步同输入对照

150 samples，随机交错、旋转专家，8 routed experts，包含完整 CPUInfer 提交/同步。
诊断模式的 device span：原 XMX 202.44 us，上一阶段 split+fused-merge 186.16 us，
整专家 1-way 151.34 us，整专家 2-way 137.65 us。
诊断 event 查询有额外开销，最终性能另用 trace/profile 关闭的路径。

300 samples 的无插桩 pinned NVIDIA D2H/callback/iGPU/H2D 完整链：

| 候选 | wall 中位数 | CUDA chain 中位数 |
|---|---:|---:|
| 原 XMX | 283.03 us | 266.64 us |
| 上一阶段 split gate + fused merge | 262.27 us | 245.90 us |
| 整专家 1-way | 214.24 us | 198.21 us |
| 整专家 2-way | 198.25 us | 182.24 us |

多个执行器的驻留工作集与完整 40 层不同，不能把这些比例直接当作整模型加速。

原始数据与 excluded 记录：`/home/wy/work/kt-perf-log/igpu-v4/`。

## 安装与启动

加载 oneAPI 后，在 `kt-kernel` Python 环境构建；副本模式避免修改 llama.cpp 子模块：

```bash
source /opt/intel/oneapi/setvars.sh --force
CPUINFER_USE_SYCL=1 CPUINFER_LLAMA_BUILD_COPY=ON CPUINFER_FORCE_REBUILD=1 \
CPUINFER_PARALLEL=2 python -m pip install --no-build-isolation ./kt-kernel
bash kt-kernel/scripts/launch_sycl_igpu_qwen35b.sh
```

启动脚本显式固定已测量配置，忽略同名旧环境设置；可用 `KT_MODEL_PATH`、`KT_PYTHON`、
`KT_HOST`、`KT_PORT` 修改路径/地址。默认绑定 localhost，额外参数追加给 SGLang。
脚本检查扩展 implementation revision，拒绝旧扩展静默忽略整专家开关。
该脚本是当前硬件/模型专用配置，不是跨模型默认。

只回退整专家融合：`KT_SYCL_XMX_FUSED_EXPERT=0`（手动启动命令），
可保留 split gate；再设 `KT_SYCL_XMX_DECODE_GATE_KERNEL=xmx` 恢复原 decode 投影。
回退不需要 checkout 或重装，旧 kernel 与执行器路径仍保留。

## 第一轮整模型对照（同一二进制）

trace/profile 关闭，prefill chunk=512，5 次请求/档，300 输出 tokens。
顺序为整专家候选、上一阶段较优配置、原 XMX；三者使用相同安装扩展。

| Prompt | 原 XMX | split gate + fused merge | 整专家 2-way |
|---|---:|---:|---:|
| 1K | 37.31 tok/s | 38.13 tok/s | **39.72 tok/s** |
| 4K | 36.36 tok/s | 37.20 tok/s | **39.38 tok/s** |
| 7.5K | 35.63 tok/s | 36.59 tok/s | **38.23 tok/s** |

相对原 XMX 的观测约 +6.5%–8.3%，相对上一阶段较优配置约 +4.2%–5.9%。
prefill 未改，无该融合的 prefill 加速结论。服务顺序、温度/频率和生成路由仍会影响结果，
需要结合反序自然 prompt、相同输入 kernel 数值验证和模型输出检查解释。


反序的非重复词代码说明 prompt（实际 97 输入/300 输出 tokens，5 次请求）为
原 XMX **37.94 tok/s**、整专家 **40.88 tok/s**，观测约 **+7.74%**。
这是第二种工作负载的对照，不是固定专家路由归因；生成路径可能改变。

固定英文/中文/代码三条 chat prompt 的 64-token greedy/logprob 检查：
输入 token logprobs 无差异，三条输出的 64 个 token IDs 均相同；
同前缀 chosen-token logprob 平均绝对差约 0.00265/0.00869/0.00437，
最大差约 0.0465/0.2408/0.0562。概率并不逐位相同。
这是有限模型输出 smoke，**不是完整困惑度/下游任务质量等价证明**。
原始检查记录保存于 `igpu-v4/decode-output-check.json`。


## 调度补测与最终配置

整专家候选的无插桩 pinned 链，300 次交错测试：默认/host quant/graph/
host quant+graph 的 wall 中位数为 197.00/197.34/194.63/194.14 us。
随后 graph=1 先、graph=0 后，各档 3 次整模型补测：

| 模式 | 1K decode | 4K decode | 7.5K decode |
|---|---:|---:|---:|
| graph=1 | 40.56 tok/s | 38.53 tok/s | 37.97 tok/s |
| graph=0 | 40.25 tok/s | 39.26 tok/s | 38.24 tok/s |

graph 的小幅单层收益未一致转化为整模型优势，因此交付选择 **graph=0、host quant=0**。
不按 prompt 长度加入未经验证的自动切换。原 staged 和实验路径继续保留。

该交付配置是目前本机上述测试中较优且通过验证的配置；未来仍可能有更好的实现。

## 交付验证记录

- 硬件/分派/loader/trace-summary/启动配置回归：**131 passed / 3 skipped**；
  skip 为不兼容的全阶段量化融合与非原 gate 组合，另有拒绝检查。
- 外部 `perf.py`：**18 passed**，新开关指纹、自定义 prompt 哈希与旧扩展检查覆盖。
- 真实模型第 0 层 rows=1/17/64 数值检查通过；整专家物理映射/屏蔽/重复/无效/空路由、
  host quant 与 graph 动态输入、SwiGLU alpha/clamp、incremental 检查通过。
- 专用启动脚本已实际启动服务、核验 `fused_expert=1` 与 split2，完成请求后停止服务。
- CMake 副本模式重复构建与独立 DPAS/vector/trace 检查日志、二进制和源码哈希见
  `/home/wy/work/kt-perf-log/igpu-v4/validation-manifest.json`。

默认 legacy 与 staged XMX 选择不被全局更改；交付脚本显式选择已测配置。
`third_party/llama.cpp` 保持干净。详细原始数据保存而不把模型、二进制或大量服务日志提交到源码库。
