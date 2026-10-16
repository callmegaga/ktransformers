# Panther Lake 混合推理：CUDA dense decode 研究

2026-10-16，接续 [长上下文复核与关键路径调查](SYCL-IGPU-Validation_zh.md)。
平台仍为 Core Ultra 7 356H / Intel `b0a0` / RTX 5060 Laptop，
模型为 Qwen3.5-35B-A3B-GPTQ-Int4，40 层、top-k8、H2048、I512、group128。
**这是 NVIDIA + Intel iGPU + CPU 的整模型混合推理优化，不是新的纯 iGPU kernel。**

## 为什么继续研究 NVIDIA dense

前一轮 CUPTI 数据显示 decode graph 中 FP16 cuBLAS GEMV 值得进一步验证。
本轮列出实际运行的 unquantized linear / LM head 形状：

|每 token 调用数|权重 N×K|作用|
|---:|---:|---|
|30|8192×2048|linear attention QKV|
|30|4096×2048|linear attention Z|
|40|2048×4096|attention output|
|10|9216×2048|full attention QKV / gate|
|40|1024×2048|shared expert gate/up|
|40|2048×512|shared expert down|
|40|256×2048|router|
|60|32×2048|small control projections|
|1|248320×2048|LM head|

这些被探针覆盖的 FP16 权重逻辑用量合计约 **3.61 GiB / token**。
这是矩阵形状与调用数的乘积，**不是硬件计数器测得的 DRAM 流量**；缓存会影响实际读取。
逐算子的同步 CUDA event 探针用于确认形状，不以其带插桩耗时作为生产吞吐或关键路径比例。

## 实现与限制

SGLang 新增 `layers/kt_cuda_dense.py` 与 lazy-import 的 `kt_cuda_dense_kernel.py`：

- 针对 M=1、contiguous CUDA FP16 输入/权重，每 CTA 计算 4 个输出，完整 K 方向归约；
  **乘法与归约均为 FP32，最终转换回原 FP16 输出**。不打包/重算/量化权重。
- 只替换实测的较大 dense projection 形状；保留 router N256 与小投影的原路径。
- LM head 可使用同一 FP32-reduction GEMV；另保留 pad4 实验分派，
  复制 4 行 activation 调用原 FP16 GEMM，再取第一行。pad4 不复制权重。
- 多 token prefill、非连续输入、其它 dtype、梯度训练、autocast 等回退原实现；
  LM head 的 LoRA、显式 FP32 与 RL BF16 分支仍按原代码执行。
- 原生 expert MoE、iGPU v6、GPTQ/S8 激活量化、BF16 舍入边界与 GPU experts=8 放置不变。

检查一份生产形状 N8192/K2048 的实际 PTX：有 `mul.rn.f32` 与 FP32 add，
没有 FP16 multiply 或 MMA 指令。该记录不是寄存器硬件计数器或吞吐证明。
尽管未降低算术精度，**归约顺序改变，所以不宣称与 cuBLAS 逐位相同**。

### CUDA graph 捕获时机是硬约束

本轮最初尝试通过 stock `--forward-hooks` factory 替换算子。
但 `model_runner.py` 先调用 `init_device_graphs()`，随后才注册普通 forward hooks，
因此替换只影响 eager 调用，**不会修改已经捕获的 decode graph**。
首轮 `e2e-native/`、`e2e-both-pad4/` 已标为无效 decode 对照，不能作为优化无效或有效的证据。

随后在独立实验进程中通过环境门控的 local `sitecustomize` 提前替换，确认有收益后，
才接入实际 `UnquantizedLinearMethod.apply` / `_compute_lm_head`，使其在正式 graph capture 时生效。
**生产交付不依赖 sitecustomize 或 monkeypatch。** 环境开关必须在启动服务前选择，
运行中改开关不会重写 graph，切换 profile 需要重启服务。

## 算子与真实输入验证

- CUDA regression **37 passed**：FP32 reference、尾部 mask、bias、训练/autocast 回退、
  原 linear / LM head 特殊分支、CUDA graph replay 后变更输入、实际 248320 vocab head。
- 在 native 模型真实输入与驻留权重上，对候选做 **5,000 次 shadow 检查**，始终把 native 输出
  送回模型，避免候选的后续输入漂移影响比较。所有候选结果有限。
- 原/新完整输出的最坏 relative RMSE 约 `3.10e-4`，单独 head-row 最坏约 `9.92e-6`。
  每种形状另外抽取 128 个输出，用 FP32 matmul 作 reference；这不是全数据集质量验证。

### 长提示词不能标成全部通过

相同 GPU8 放置、seed20261016，7 条提示词，native / both-row 每条各重复 3 次，
每次生成 64 tokens。保留原重复 word 压力测试，同时检查长英文、中文和代码。

- 三条短英文/中文/代码，各 6 次 greedy token 全部相同，检查的 input logprob 也完全一致。
- 长代码各 6 次 greedy token 相同，但 input logprob 仍不稳定。
- 原压力测试 native 自身出现 index47 分歧；候选 3 次都匹配 native 已观察到的序列。
- 长英文两边各出现 2 种序列，候选均匹配 native 已观察到的序列。
- 长中文两边各出现 2 种序列；候选有 1 次序列不属于这 3 次 native 样本，首次分歧 index43。
  不能据此断言质量下降，也不能因为 native 也波动就将它忽略。
- 所有输出 logprob 有限，但长文本 input logprob 的重复差异仍显著；没有进行完整 PPL 或下游评测。

因此新 dense 路径 **保留 opt-in**，不代替已验证 v6 的默认 native dense 配置。

## 原生 GPU trace 验证

相同 4K prefill + 32 输出请求，独立 CUPTI13 采集：

- native：graph 中新 row GEMV 为 0 次。
- both-row：graph 中新 row GEMV 为 **6,112 次 = 32 × 191**，与上表被替换的调用数一致。
- graph launch 间隔中位数：native 约 25.49ms，both-row 约 23.90ms。
- graph kernel 时间之和：约 549.62 → 528.83ms；kernel interval union：约 516.53 → 477.09ms。

这证明候选真正进入 decode graph，不是只改了 Python 开关。
插桩数据不替代无 profiling 的最终 A/B；kernel 时间和与 union 都不是 CPU/iGPU/NVIDIA 的
共同关键路径。仍不能将 wall time 减去这些数值后的剩余项全部归为 CPU 或 iGPU。

## 最终无 profiling 的整模型复测

同一安装版本与二进制，先 both-row 后 native，每档 5 次，每次生成 300 tokens。
seed20261016、GPU experts8、CPUInfer10、chunk512、iGPU v6；
没有 hooks、logprob 提取或 profiling。配置校验确认唯一环境差别为
`SGLANG_KT_CUDA_DENSE=native → both-row`。以下为中位数，实际输入为 1029 / 4101 / 7685 tokens：

|输入档位|原 v6 decode tok/s|新候选 decode tok/s|提升|TOPT ms/tok|Prefill 原→新 tok/s|
|---|---:|---:|---:|---:|---:|
|1K|40.42|43.86|+8.51%|24.74→22.80|1415.73→1421.56|
|4K|39.37|41.90|+6.44%|25.40→23.87|1558.69→1520.87|
|7.5K|38.58|40.79|+5.72%|25.92→24.51|1537.76→1525.02|

这里是 **v6 基础上的额外 decode 收益**，不是重复计算上一轮 v5→v6 的 prefill 提升。
Prefill 没有稳定收益；本轮 4K 中位数反而下降约 2.43%，7.5K 下降约 0.83%。
顺序对照不能完全隔离温度、功耗和频率，所以不宣称所有输入无回退。
较早 3×128 的消融与正式 5×300 的 decode 改善方向一致；pad4 head 单独替换则没有稳定增益，
本轮推荐候选为 both-row，而不是默认用更多 TensorCore 运算就更快。

最终 Intel 硬件/调度/启动/qlen 回归 **203 passed / 3 expected skipped**；
CUDA dense **37 passed**，perf.py **25 passed**。

后续优先补齐 CPUInfer 任务队列、主机 callback、SYCL 等待和跨流同步的关键路径边界测量，
而不是把 kernel 时间之和从 wall time 中减掉后直接归因。
若进一步压缩 dense 权重以减少带宽，需要单独声明存储/算术格式与质量预算；本轮没有做这类精度变更。

## 使用

```bash
# 保守默认：iGPU v6 + 原 NVIDIA dense
bash kt-kernel/scripts/launch_sycl_igpu_qwen35b.sh

# 本轮推荐的可选 dense decode 候选（需要安装 matching third_party/sglang）
KT_IGPU_DENSE_PROFILE=both-row bash kt-kernel/scripts/launch_sycl_igpu_qwen35b.sh

# 只替换 projection，不替换 LM head，便于消融
KT_IGPU_DENSE_PROFILE=linear bash kt-kernel/scripts/launch_sycl_igpu_qwen35b.sh
```

支持 `native / linear / head-row / head-pad4 / both-row / both-pad4`。
启动脚本拒绝拼写错误并检查已安装的 SGLang 是否包含新实现，不让旧包静默忽略新配置。
直接启动 SGLang 时，对应开关为 `SGLANG_KT_CUDA_DENSE`；perf.py 对所有专家后端记录该开关
并计入配置指纹，避免将 CUDA dense 差异错误归因于 AVX2 / VNNI / iGPU 专家后端。
Intel 扩展仍为 revision6，二进制没有改变，不需要为本轮 Python/Triton 改动重新编译 SYCL。
`third_party/llama.cpp` 与外部 OpenVINO / llama.cpp 参考仓库未修改。

全部实验、排除记录和复现脚本：`/home/wy/work/kt-perf-log/igpu-v8-dense/`。
尚未证明达到硬件极限，也没有把不同量化产物的 OpenVINO 报告作为这轮微架构归因的依据。
