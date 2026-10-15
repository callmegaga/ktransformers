# Panther Lake iGPU prefill：M/N 复用与独立投影分块

接续 [整专家 decode 融合](SYCL-IGPU-Fused-Expert_zh.md)。目标是在保持已测约40tok/s
decode 的前提下，改善 Qwen3.5-35B-A3B-GPTQ-Int4 的长 prompt TTFT，
追赶本机官方 OpenVINO IR 的约1027tok/s 输入吞吐。
本页的比较针对当前 Core Ultra7 356H / Intel b0a0 / RTX5060 Laptop 平台；
KT 为混合推理，OpenVINO GPU.0 为纯 Intel GPU 路径，不宣称跨框架/模型的普遍最优。

## 为什么需要不同的 prefill kernel

此前诊断显示4K请求40层的 gate/up、down 累计约2.34/1.87s，CPU路由和提交远小于它。
只把 chunk512改为1024没有明显改善。现有 decode 的整专家融合不适合多 token prefill。
本阶段不改变 group-wise S8 激活量化、S4 权重、BF16 输出舍入与路由语义，
只改善矩阵分块与数据复用。

新增 `gptq_int4_xmx-prefill.hpp` 的 N32 kernel：

- 每个 ESIMD work-item 计算两个 N16 输出块，复用同一次激活加载和行映射；
- 多个 DPAS repeat8 共享权重加载，允许 M16/M32 实验；
- gate/up 在同一个 work-item 内保持融合，并可与 SwiGLU + BF16舍入 + 中间量化融合；
- N32 的四个 work-items 通过 SLM 合并一个128值量化组的最大值；
- gate 与 down 使用各自的 Tile 列表和 M，输出 routed row 索引保持不变；
- 全部读取原来的 packed 权重，无额外常驻 FP16/INT8 权重副本，无新运行时依赖。

独立实验表明 M16 对 gate 与 down 的效果不同，M32 gate明显回退。
因此选用 **gate M8N32 + down M16N32**，而不是简单统一增加 M。
小 token batch、尾部行、重复 expert 槽位、无效/空路由仍正确。
单 token decode 保持第四阶段整专家实现，不经过新的 prefill kernel。

## 环境开关

| 开关 | 后端默认 | 可选值/约束 |
|---|---|---|
| `KT_SYCL_XMX_PREFILL_ROWS` | 继承旧 `KT_SYCL_XMX_ROWS` | 1/2/4/8/16/32；大于8要求INT8原XMX prefill |
| `KT_SYCL_XMX_PREFILL_DOWN_ROWS` | 继承 PREFILL_ROWS | 同上；与gate不同时要求INT8原XMX prefill |
| `KT_SYCL_XMX_PREFILL_N` | 16 | 16/32；32要求INT8原XMX prefill、输出维度%32=0、M<=16 |

原 Python/C++ 后端默认分派保持兼容，不自动按未验证阈值切换。
专用交付脚本显式设置经过本机测试的 M8N32/M16N32：

```bash
bash kt-kernel/scripts/launch_sycl_igpu_qwen35b.sh
```

手动启动命令附加：

```bash
export KT_SYCL_XMX_PREFILL_ROWS=8
export KT_SYCL_XMX_PREFILL_DOWN_ROWS=16
export KT_SYCL_XMX_PREFILL_N=32
```

保持 `KT_SYCL_XMX_FUSED_QUANT=prefill` 与原 INT8 decode 配置。
只回退本阶段设置 N=16、gate/down rows均=8；无需重新安装或 checkout。
新扩展 implementation_revision=5，启动脚本和 perf.py 会拒绝过旧扩展忽略新开关。

## 构建与正确性

原有 oneAPI、SYCL 构建方式不变，`CPUINFER_LLAMA_BUILD_COPY=ON` 避免污染 llama.cpp 子模块。
无需链接 oneDNN；虽然做过oneDNN GPU可用性探测，但当前胜出实现为原生ESIMD/XMX。

```bash
source /opt/intel/oneapi/setvars.sh --force
CPUINFER_USE_SYCL=1 CPUINFER_LLAMA_BUILD_COPY=ON CPUINFER_FORCE_REBUILD=1 \
CPUINFER_PARALLEL=2 python -m pip install --no-build-isolation ./kt-kernel

icpx -fsycl -fp-model=precise -O2 kt-kernel/test/sycl/test_prefill_n32.cpp -o /tmp/test_prefill_n32
ONEAPI_DEVICE_SELECTOR=level_zero:gpu /tmp/test_prefill_n32
```

独立测试必须使用与扩展一致的 `-fp-model=precise`，否则编译器收缩浮点表达式可能导致末位舍入差异。
检查 M1/2/4/8/16、group32/128、不同专家/scale、反向token映射、tail33、gate/down、
SwiGLU alpha/clamp，及融合中间量化的S8结果/scale，与原路径逐位比较通过。
集成回归147passed/3skipped；真实模型第0层rows1/17/64/512参考检查通过。
这不替代完整数据集PPL或下游任务质量评估。

## 候选比较与首轮端到端结果

完整 MoE 随机交错、旋转专家、50samples，rows512中位时延：

| 配置 | 完整 MoE 时延 |
|---|---:|
| 原 M8N16 |15.27ms|
| gate/down M8N32 |11.61ms|
| gate M8N32 + down M16N32 |10.68ms|

同一组权重与输入，改善约30%。这不是整模型加速比例。

整模型首轮每档3次、32输出tokens，chunk512、trace/profile关闭，prefill定义为input_tokens/TTFT：

| 配置 |1K|4K|7.5K|
|---|---:|---:|---:|
| 昨日交付 M8N16（今日复测）|752.5|785.2|777.5|
| gate/down M8N32 |1041.8|1097.6|1094.8|
| gate M8N32 + down M16N32 |1082.0|1144.3|1140.8|
| OpenVINO unlimited预算（今日3次复测）|782.9|1027.4|1026.5|

单位tok/s，实际输入1029/4101/7685tokens。OpenVINO为另一量化产物（NNCF非对称INT4），
KT为对称GPTQ，权重与路由不完全一致。OV对照16输出tokens，表中KT候选32；
TTFT发生于decode之前，但这些不是严格的固定路由/固定二进制跨框架归因实验。
OpenVINO的原脚本并没有使用报告中写的512chunk；unlimited是其原实际配置。

这一轮表明已测长prompt输入吞吐超过该OV参考，但后续还需同二进制长输出和反序控制。
详细数据在 `/home/wy/work/kt-perf-log/igpu-prefill-v5/`，OpenVINO调查在
`/home/wy/work/kt-perf-log/openvino-analysis/20261015/`，不修改原始用户报告。
