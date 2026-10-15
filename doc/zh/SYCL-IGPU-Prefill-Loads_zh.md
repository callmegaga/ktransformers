# Panther Lake iGPU prefill v6：路由向量读取与专家合并

接续 [prefill M/N复用](SYCL-IGPU-Prefill_zh.md)。在当前 Core Ultra7 356H / Intel b0a0 /
RTX5060 Laptop 的 Qwen3.5-35B-A3B-GPTQ-Int4 混合推理配置上，进一步优化数据供给。
**本阶段没有改变GPTQ权重、group-wise S8激活量化、BF16舍入边界或decode算法。**
目标是继续逼近本机性能上限，而不是宣称已经达到硬件极限或所有工作负载上的最优。

**2026-10-15 后续复核：交付入口默认使用 v6，可显式回退 v5。**
原长文本分歧在 v5 自身也可复现；固定输入的 NVIDIA-only Marlin 重放独立复现数值波动，
而同为 GPU experts=0 的 v5/v6 各 5 次整模型检查结果逐位相同。
详细证据、限制和未推广的实验见 [长上下文复核与瓶颈](SYCL-IGPU-Validation_zh.md)。
这不代表整个混合框架逐位确定，也不是完整 PPL / 下游质量验收。

## 实现

- `gptq_int4_xmx-prefill.hpp`：保留 gate M8N32、down M16N32 和现有DPAS计算；
  可选用masked dword gather，一次组织多行32字节的激活向量，替代每行独立的小块读取。
  token行偏移与有效行mask在K循环外计算，尾部无效行保持零，不越界访问。
- `gptq_int4_xmx-kernels.hpp`：新增ESIMD向量专家合并，每个work-item处理连续256个FP32输出；
  hidden只对齐128时使用128宽，其它情况回退原合并。保持原slot累加顺序，CPU最终BF16转换与
  incremental语义不变。
- 两个优化可独立开关，便于同一二进制的2×2归因。单token decode不进入新prefill路径。
- 不新增常驻权重副本、不增加激活打包workspace、不引入运行时依赖，不修改llama.cpp子模块。

## 开关与启动

| 开关 | 底层后端默认 | 交付入口 v6 profile |
|---|---|---|
| `KT_SYCL_XMX_PREFILL_LOAD` | `block` | `gather` |
| `KT_SYCL_XMX_PREFILL_MERGE` | `scalar` | `vector` |

`gather`要求已经选择合法的INT8 XMX N32 prefill；非法值或不兼容配置会明确报错。
其它已有约束不变。扩展 `implementation_revision=6`；交付脚本按profile检查v5/v6最低扩展版本，perf.py也进行功能版本预检，
避免旧扩展静默忽略新配置。

```bash
KT_IGPU_PREFILL_PROFILE=v6 bash kt-kernel/scripts/launch_sycl_igpu_qwen35b.sh
```

不设置`KT_IGPU_PREFILL_PROFILE`时使用v6；设置`KT_IGPU_PREFILL_PROFILE=v5`回退。非法profile会报错。
脚本会按所选profile覆盖继承的旧开关。自行启动sglang时，设置以下两个值即可
只回退本阶段，保持v5的gate M8N32/down M16N32及融合decode：

```bash
export KT_SYCL_XMX_PREFILL_LOAD=block
export KT_SYCL_XMX_PREFILL_MERGE=scalar
```

## 完整MoE链路归因

相同权重、输入和路由，随机交错顺序，50个样本，512行，CPU transport，device profiling开启。
以下是诊断数据，不是无profiling的整模型TTFT。

| 输入读取 / 合并 | 完整MoE中位时延 |
|---|---:|
| block / scalar（v5） |10.55ms|
| gather / scalar |6.79ms|
| block / vector |10.31ms|
| gather / vector（v6） |6.54ms|

完整链路时延减少约38%。同组设备event均值：gate约5.95→3.36ms、down约3.08→1.91ms、
merge约0.570→0.342ms。CPU等待与GPU时间存在包含关系，不能相加。
17/64行仍未获得同样幅度的收益；不能把大prefill的优势泛化到所有小专家批量。

## 最终整模型对照

同一v6二进制，先新配置后v5配置，各档5次，每次300输出tokens；chunk512、GPU experts8、
CPUInfer10、单请求、context8192，关闭trace/profile和SYCL graph。
唯一配置差别是上面两个prefill开关。表中为中位数，实际输入1029/4101/7685tokens。

| 输入档位 | v5 prefill | v6 prefill | 吞吐提升 | v5→v6 TTFT | v5→v6 decode |
|---|---:|---:|---:|---:|---:|
|1K|1075.84tok/s|1411.96tok/s|+31.24%|0.956→0.729s|40.06→40.78tok/s|
|4K|1142.30tok/s|1532.57tok/s|+34.17%|3.590→2.676s|39.23→39.45tok/s|
|7.5K|1129.68tok/s|1527.99tok/s|+35.26%|6.803→5.029s|38.31→38.41tok/s|

没有观察到明显decode回退；小幅波动不能解释为本阶段改进了decode。
顺序实验不能完全消除温度、功耗、频率等因素，数据只覆盖本机这组工作负载。
本阶段主要使用KT自身同二进制对照，不以不同量化产物的OpenVINO对比作微架构归因。

## 已排除的方向与硬件诊断

- **输入预打包**：独立consumer看起来可能更快，但包含GPU打包event后，512行完整链路约
  7.60ms，慢于不打包gather的约6.75ms；17/64行也更差。未纳入生产实现。
- **继续增大M**：gather下M16 gate仍慢于M8，M8 down慢于M16；保持独立M8/M16。
- **N64及256 GRF**：独立候选未胜出。汇编转储确认N64 M8 gate候选有约3.3KiB GRF spill；
  强制256 GRF后也没有得到更快的组合。不能假定更大的分块或寄存器配置一定更快。
- **显式L1/L2缓存提示**：独立交错测试没有稳定优势，未纳入生产实现。
- `bench_sycl_limits.cpp`测得独立S8×S4 DPAS约32–35TOPS，256MiB流式拷贝总读写带宽约
  116–117GB/s。它们是独立微基准，不是整模型可同时达到的上限，也不是模型利用率的证明。
- VTune GPU event收集报错；PyTorch的CUPTI报`CUPTI_ERROR_INVALID_DEVICE`，导出的torch trace
  只有CPU事件。不能据此声称拿到了完整NVIDIA/iGPU重叠时间线或硬件计数器利用率。

因此，后续大幅decode提升必须继续调查整条CPU/NVIDIA/iGPU关键路径，不能仅凭iGPU专家
kernel事件相加或CPU trace就判断剩余瓶颈。prefill仍可研究更好的复用、形状分派及访存流水，
但当前结果不支持承诺下一轮的固定加速比例。

## 验证与数据位置

硬件集成与调度/启动/诊断回归190passed、3expected skipped，包含非零权重、重复/无效专家、
尾部17/33/65、非incremental及incremental输出的原路径逐位对照。
独立N32测试检查M1/2/4/8/16、group32/128、反向行映射和融合量化结果/scale；
向量合并检查0/2/8路由和16/32/64/128/256/512向量宽度。必须使用生产一致的
`-fp-model=precise`；merge测试另需`-std=c++20`。

完整实验与复现脚本位于 `/home/wy/work/kt-perf-log/igpu-prefill-v6/`。
`e2e-comparisons.json`记录最终A/B结果，`checkpoint/`保存二进制与哈希；失败/排除实验单独保留，
不与成功结果混合。固定提示词校验与真实层权重检查另见该目录的验收记录；有限测试不替代完整
数据集PPL或下游质量评估。

### 整模型长文本检查：不能写成“全部通过”

英文/中文/代码三条短提示词，新旧配置64个greedy输出IDs全部相同，输入logprobs平均差为0。
新增4118-token提示词，在固定seed=20261015后仍在第48个输出token分歧；该严格检查未通过。
请求只取最后64个输入logprobs，避免全输入logits在8GB NVIDIA显卡上OOM，但保留完整长prompt。

进一步以v5对v5做相同seed的重复控制：长输入logprobs也并非逐位确定，平均绝对差约0.216；
新旧配置约0.0515。这个结果只能说明原执行链存在数值非确定性，**不能完整归因greedy分歧，
也不能据此证明新旧模型质量完全等价**。

为区分算子差异与执行链差异，额外编译临时shadow诊断扩展：每次真实prefill调用，在完全相同的
BF16输入、路由与常驻权重上重放v5，与候选输出做FP32逐字节比较。覆盖全部40层、520次调用、
353,157,120个FP32输出，无差异。该诊断不用于性能计时；随后恢复原源码、构建对象及已测二进制，
构建/安装/性能checkpoint的SHA256均为
`973c389550412e8bc9eb64948de3691d4ff463f0f21ccee752b966178de0c3b0`。

这些证据支持没有发现iGPU算子精度回归，但不替代整条混合执行链的进一步排查。
**本节为最初的保守 rollout 记录**：当时保留 v5 默认，未标记整模型质量验收完成。
后续重复控制及 NVIDIA-only / GPU0 对照补齐后，交付入口改为 v6；
完整混合模型的数值非确定性和质量评估限制仍保留，不把原失败记录改写成“全部通过”。
