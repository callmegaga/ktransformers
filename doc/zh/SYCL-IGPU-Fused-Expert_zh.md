# Panther Lake iGPU 整专家融合与本机性能配置

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
