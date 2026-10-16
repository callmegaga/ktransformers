# Panther Lake iGPU：长上下文复核与下一阶段瓶颈

后续 CPUInfer/callback 测量见 [调度研究](SYCL-IGPU-Scheduling_zh.md)。

后续研究见 [2026-10-16 CUDA dense decode](SYCL-IGPU-CUDA-Dense_zh.md)。
原验证记录及非确定性限制保留。

2026-10-15，接续 [v6 prefill 向量读取](SYCL-IGPU-Prefill-Loads_zh.md)。
平台与模型不变：Core Ultra 7 356H / Intel `b0a0` / RTX 5060 Laptop，
Qwen3.5-35B-A3B-GPTQ-Int4。本文的整模型执行仍是 **CPU / NVIDIA / Intel iGPU 混合推理**，
不是将整个模型放到 iGPU 上运行。

## 1. 修复独立的异步 qlen 所有权缺陷

`kt-kernel/python/experts_base.py` 的临时 buffer 在 batch shape 改变时会被替换。
原生 `forward_task` 只保存 CPU qlen 标量的裸指针，不持有 Python tensor；
qlen 不参与 DMA，所以 pinned allocator 的 stream 事件不会替它延长生命周期。
CPU-only 弱引用测试在 callback 消费前复现了 tensor 已释放的问题。

修复以 `(batch_size, buffer_depth)` 为 key 缓存小型、不可变的 qlen tensor，
不保留整份大型 staging buffer，也不新增逐层同步事件。
5 项 CPU-only 回归通过。该修复在共用的 Python MoE wrapper 中，对 CPU 与 iGPU 后端均有意义。

**不能把这项缺陷直接等同于长文本分歧的全部原因**：修复后 v5 自身仍能复现不同输出。

## 2. v5 基线并非逐位确定

原重复 `word` 的 4118-token chat 压力提示词保留，没有从验收中删掉：

- 原 v5 同一服务 5 次重复出现 2 种 greedy 输出，首次分歧 index 47（第 48 个输出 token）。
- qlen 修复后，v5 / v6 同配置重复仍分别出现多个输出。
- cuBLAS `CUBLAS_WORKSPACE_CONFIG=:4096:8` 对照也未消除分歧。
- 逐层同步探针中，第 3 层 attention、post-attention norm、router logits、top-k
  的输入与结果均相同，专家模块之后开始不同。forward 后的 input 可能已经被
  `inplace=True` 的 NVIDIA MoE 改写，不能误认为它就是 forward 前的原始输入。
- RoPE 模块由多个 full-attention 层共享，单一模块 hook 会多次触发；不能用共享 hook
  的最后一条记录来归因“第 3 层 RoPE 出错”。

### 固定输入的 NVIDIA 专家重放

保存真实第 3 层的输入、路由和常驻 GPTQ 权重，单独运行 NVIDIA Marlin，**不调用 iGPU**。
4096 个 top-k slot 中 146 个落在这组 GPU 专家上。原路径固定输入重放 20 次，
20 个输出 hash 不同；相对首个结果的最大绝对差约 `3.81e-6`。
强制 non-atomic 路径仍非逐位确定，最大绝对差约 `1.91e-6`。

这证明 NVIDIA 专家腿本身就有末位数值波动，不能将单次 v5/v6 greedy 分歧直接归因于
新 iGPU 内核精度退化。近似相等的路由候选可能放大微小差异。
这里没有宣称所有跨设备耦合问题均已排除，也没有把“差异很小”当成完整质量评估。

### 排除 NVIDIA 路由专家的整模型对照

在 **两边都为 GPU experts=0**、其余配置相同的条件下，v5/v6 各重复 5 次上述长提示词。
10 次的输出 token、检查的输入/输出 logprob、output top-5 logprob 全部完全一致。
这是与此前 40 层、520 次实际 iGPU 调用、353,157,120 个 FP32 输出逐字节一致相互独立的证据。

注意：GPU0 与 GPU8 的专家计算精度/放置不同；这里只比较同为 GPU0 的 v5 和 v6，
**不声称 GPU0 与正常 GPU8 配置数值等价**，也不以该对照的耗时作为生产性能。

### 补充长英文 / 中文 / 代码检查

实际输入分别为 4246 / 4200 / 4195 tokens，每个 profile 每条提示词重复 3 次，
每次 64 输出 tokens。全部输出 logprob 有限。中文和代码各 6 次生成 token 全部相同；
英文 v5 / v6 各出现 2 种输出，首次分歧 index 37，两边都有同配置重复分歧。

尾部 63 个可用 input logprob **并不稳定**：v5 自身相对首轮的平均绝对差最高约 1.40，
v6 相对 v5 首轮最高约 1.73。本阶段没有把这些差异解释成已验证无害，
也没有将“finite logprob”或中文/代码 token 一致写成完整精度验收。
原重复词压力测试、常规长文的失败/非确定记录都保留。
相同 greedy 输出前缀上的 output logprob 平均绝对差最大约 0.0121（v5 自身）/
0.0148（v6 相对 v5 首轮），与上述 input logprob 统计分开报告，不能混为同一指标。

### 交付选择

交付脚本默认启用 v6，`KT_IGPU_PREFILL_PROFILE=v5` 保留回退。
依据是受控 iGPU 逐字节等价、同 GPU0 放置的端到端逐位对照，以及同二进制的吞吐收益，
不是因为混合 GPU8 模型通过了逐位确定性或完整 PPL 验收。
没有改变 GPTQ 权重、S8 activation 量化、BF16 舍入、GPU experts=8 放置或 decode 算法。
底层通用 C++ 开关默认仍保守保留，变化仅限这个本机定制的交付入口。

## 3. 未推广的“跳过无效 NVIDIA 专家”实验

KT hybrid 把 CPU 路由 remap 成 `-1`，原 Marlin 包装层再将其替换为专家 0、权重 0。
结果正确，但仍为这些 slot 做 GEMM。原生 aligner / EP-aware Marlin 支持跳过 `-1`，因此试验：

1. 保留 `-1`，打开原生 EP skip；对未写入的 slot 清零，再执行原有 epilogue。
2. 加 masked SiLU 与 masked FP32 顺序归约，不读取无效 slot，省掉两次大缓存清零。

包含实际 H2048 / I512 / E8 / group128、无效及重复路由、NaN 无效权重、空 shape、
inplace、CUDA graph replay 在 sparse / empty 路由间切换的 21 项测试通过。
512 行约 3% 有效 slot 的独立微基准从约 1040µs 降至 179µs。

**但整模型没有稳定改善**：清零版本的 4K/7.5K prefill 约退化 3%；
masked 版本按相反顺序复测，各档结果涨跌不一致，decode 也无稳定收益。
顺序测试仍会受功耗、频率和温度影响；不能将独立 kernel 的数倍收益外推为整模型收益。

因此该路径仅保存为实验 patch，**没有合入生产 SGLang，也没有改动 submodule 指针**。
`third_party/sglang`、`third_party/llama.cpp` 保持干净；OpenVINO / llama.cpp 参考仓库未改动。

## 4. 原生 NVIDIA profiling：下一步不应只盯 XMX

使用本机 CUDA 13.2 SDK 的 CUPTI injection 独立采集，不替换 PyTorch 或其 bundled CUDA。
`CUDA_PROFILER` start/stop 覆盖一次 4K prefill + 32 输出请求。
共捕获 71,391 个 kernel，其中 51,328 个来自 graph；32 次 graph launch，间隔中位数约 25.19ms。

Graph kernel 时间之和约 550.28ms，其中两个主要 FP16 cuBLAS GEMV 合计约 394.42ms，
Marlin MoE 约 46.98ms。Nongraph kernel 时间之和约 976.61ms，其中 dense GEMM 约 363.75ms、
Marlin MoE 约 295.36ms、attention 约 91.64ms。

**这些是 NVIDIA kernel 时间的和，不是整模型时延占比**；允许不同 stream 重叠。
解析器还报告 kernel interval union：graph 约 517.16ms，nongraph 约 975.30ms。
它们也只是 NVIDIA 设备的 kernel busy 区间，不等于 CPU/iGPU/NVIDIA 的共同关键路径。
当前 collector 没拿到完整 graph host-node 事件，不能把剩余 wall time 简单归为 CPU 或 iGPU。
Nongraph/graph 在这次 capture 大致对应 prefill/decode，但不是通用分类规则。

下一阶段应继续验证 NVIDIA dense GEMV、跨设备传输和异步调度的关键路径，
不能承诺单纯更换另一种 iGPU 指令或继续扩大 XMX tile 就会提高整模型 decode。
当前没有证明达到硬件极限，也没有覆盖所有模型、上下文和并发形状。

## 5. 最终同二进制复测与交付状态

qlen 修复已安装，恢复原生产 NVIDIA 路径，先 v6 后 v5，各档 5 次、每次 300 输出 tokens。
chunk512、CPUInfer10、GPU experts8、固定 seed20261015；没有 hooks、logprob 或 profiling。
唯一运行配置差异为 `PREFILL_LOAD=block→gather`、`PREFILL_MERGE=scalar→vector`。
prefill 为实际输入 tokens / TTFT，decode 排除首个输出 token，以下均为中位数：

|输入档位（实际 tokens）|v5 prefill tok/s|v6 prefill tok/s|提升|v5→v6 TTFT|v5→v6 decode tok/s|
|---|---:|---:|---:|---:|---:|
|1K (1029)|1076.36|1399.58|+30.03%|0.956→0.735s|40.46→40.49|
|4K (4101)|1133.22|1534.33|+35.40%|3.619→2.673s|39.08→39.43|
|7.5K (7685)|1130.62|1525.25|+34.90%|6.797→5.039s|38.31→38.52|

这里的约 30–35% 是 **已有 v6 算法相对 v5 的最终复测结果**，不是本轮又叠加了 35% 的新加速。
decode 没有稳定的大幅变化。顺序服务对照没有完全消除温度/功耗/频率影响，
不将单次或独立 kernel 峰值解释为整模型硬件利用率，也不拿该混合执行数据宣称纯 iGPU 框架胜负。

最终 Intel 硬件、调度、方法守卫、启动及 qlen 回归为 **196 passed / 3 expected skipped**；
perf.py 回归 **22 passed**。已排除的 SGLang sparse 原型另有 **21 passed**，不计入生产测试数。
扩展仍为 `implementation_revision=6`，二进制 SHA256：
`973c389550412e8bc9eb64948de3691d4ff463f0f21ccee752b966178de0c3b0`，没有为本轮诊断替换内核精度。

```bash
# 当前最快的本机已测交付 profile（默认 v6）
bash kt-kernel/scripts/launch_sycl_igpu_qwen35b.sh

# 保留 v5 回退；量化/专家放置/decode 相同
KT_IGPU_PREFILL_PROFILE=v5 bash kt-kernel/scripts/launch_sycl_igpu_qwen35b.sh
```

## 数据与复现

脚本、原始日志、实际输入快照、对照结果和排除实验保存在：
`/home/wy/work/kt-perf-log/igpu-v7/`。

- `diagnose_prefill.py`：同服务重复原长提示词、可选层级探针。
- `check_real_prompts.py`：长英文/中文/代码 logprob 与 greedy 重复检查，不是吞吐测试。
- `replay_real_marlin.py`：真实权重和固定路由的 NVIDIA-only 重放；生产路径可直接运行，
  额外 sparse 对照需安装归档实验 patch。
- `build_cupti_trace.sh` / `profile_native.py` / `summarize_cupti.py`：原生 GPU 诊断。
- `sparse-marlin-experiment.patch`：已排除的优化；不默认安装。

有限 logprob/greedy 检查不替代完整 PPL 或下游任务评估。
