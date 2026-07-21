# CPU-iGPU MoE 动态调度开发记录

## 1. 目标

在 Intel Core Ultra 7 265K 上，为 GPTQ INT4 MoE 专家计算增加 CPU AVX-VNNI-256 与 Intel iGPU SYCL 的协同调度：

- CPU 空闲时优先使用吞吐更高的 AVX-VNNI-256 路径。
- CPU 被其他进程占用时，将部分专家计算迁移到通常空闲的 iGPU。
- CPU 与 iGPU 尽量共享同一份专家权重，避免同时保留 packed INT4 和展开 INT8 两套完整权重。
- 保留当前 NVIDIA GPU 专家放置与 CPUInfer/CUDA stream 并行机制。
- 分别优化 decode 和 prefill，并关注尾延迟而不只看平均吞吐。

## 2. 开发环境

记录日期：2026-07-17

| 项目 | 配置 |
|---|---|
| 开发分支 | `vnni-sycl-scheduling-dev` |
| 基线提交 | `8f5e288 [feat](sycl): finish and test sycl gptq int4` |
| CPU | Intel Core Ultra 7 265K，8 个 P-core + 12 个 E-core，无 SMT |
| CPU ISA | AVX2、AVX-VNNI-256、FMA |
| iGPU | Intel Arrow Lake-U Graphics，Level Zero，shared USM |
| dGPU | NVIDIA GeForce RTX 4060 Ti 16 GB |
| 系统内存 | 46.4 GiB，单 NUMA 节点 |
| 代表模型 | Qwen3.5-35B-A3B-GPTQ-Int4 |
| 量化约束 | symmetric GPTQ INT4，`desc_act=false`，重点验证 `group_size=128` |

工作树中的 `perf-log/` 和 `report/` 是本机已有的未跟踪实验资料。本开发过程不会修改或清理它们。

## 3. 已知基线

以下数据来自改动前已有的端到端结果。两个后端都使用 TP=1、24 个 NVIDIA GPU experts、8 个 CPUInfer worker 和单请求执行。

| 后端 | Decode，约 1K context | Decode，约 8K context | Prefill，约 1K | Prefill，约 8K |
|---|---:|---:|---:|---:|
| AVX-VNNI-256 fused | 27.2 token/s | 26.8 token/s | 235.3 token/s | 616.9 token/s |
| SYCL GPTQ INT4 | 17.5 token/s | 16.7 token/s | 75.2 token/s | 88.4 token/s |

注意：已有 VNNI prefill 使用 `chunked_prefill_size=4096`，SYCL 使用 `512`，因此 prefill 数据不能直接用于计算加速比或调度阈值。Decode 数据表明空闲状态下应默认优先 VNNI，但仍需在统一配置和受控 CPU 压力下重新测量交叉点。

## 4. 当前内存布局

SYCL 后端在加载时把 GPTQ `qweight [K/8, N]` 和 `scales [K/group_size, N]` 转为 output-major：

```text
qweight [N, K/8]
scales  [N, K/group_size]
```

它们被放入 `sycl::malloc_shared` 分配的连续 shared-USM。现有日志中 Qwen3.5 每层 256 个专家约占 408 MiB，40 层约为 15.9 GiB。

当前 AVX-VNNI-256 后端把 INT4 在加载时展开为 `int8 [N, K]`，另存 FP32 scales 和 INT16 weight sums。若直接同时实例化 VNNI 和 SYCL，主权重会同时存在 packed INT4 与 unpacked INT8 两份，相对原始 packed qweight 约为三倍，无法满足本机全模型内存目标。

计划采用 SYCL 的 output-major packed INT4 作为唯一主权重布局。CPU VNNI kernel 在寄存器中按块展开 nibble，并复用 shared-USM 中的 scales；只额外保存体积较小的 weight sums。

## 5. 实施阶段

### 阶段 A：基线与实验工具

- [x] 固化硬件、分支和已有端到端基线。
- [x] 在相同 shape、路由和线程数下测量 AVX2、VNNI、SYCL 单层耗时。
- [x] 增加计算型和内存型背景负载，测出 CPU/iGPU 路径的性能交叉点。
- [ ] 记录 RSS/PSS、shared-USM 分配量、CPU PSI 和每设备执行时间。

### 阶段 B：共享 packed 权重

- [x] 抽取 output-major packed GPTQ INT4 权重存储和只读 view。
- [x] 实现 CPU packed AVX-VNNI-256 GEMM，避免持久化展开 INT8 权重。
- [x] 对 decode、稀疏 prefill、稠密 prefill 做数值一致性测试。
- [x] 比较 packed-VNNI 与现有 unpacked-VNNI 的性能和内存占用。

验收条件：主权重只有一份 packed INT4；CPU 与 iGPU 后端可绑定同一组权重指针；相对现有量化参考的误差不超过对应后端既有阈值。

### 阶段 C：固定比例异构执行

- [x] 新增独立于 NVIDIA `gpu_experts_mask` 的 CPU/iGPU runtime target map。
- [x] 根据 routed row 数把专家任务拆分给 CPU 和 iGPU。
- [x] decode 通过按需创建的 iGPU worker 与 CPU 路径并发，最后等待并合并 partial output。
- [x] 支持 `0%`、`25%`、`50%`、`75%`、`100%` iGPU 固定比例。

验收条件：所有固定比例输出正确；`0%` 和 `100%` 分别回归 CPU-only 和 iGPU-only；中间比例不存在重复计算或漏算专家。

### 阶段 D：动态控制器

- [x] 采样 CPUInfer 绑定核心的 busy 比例，并减去本进程 CPU 时间，同时结合 CPU PSI。
- [x] 使用 CPU/iGPU 实际完成时间的 EWMA 更新成本模型。
- [ ] 以预测 makespan `max(T_cpu, T_igpu)` 为优化目标分配任务。
- [x] decode 与 prefill 使用独立模型、滞回阈值和最小保持周期。
- [x] 修复并扩展 CPUInfer active worker 限制接口。

验收条件：CPU 空闲时性能接近 VNNI-only；计算型背景负载下优于 VNNI-only；内存型背景负载下能够识别共享带宽瓶颈，避免无效频繁切换。

## 6. 实验矩阵

固定所有模型和服务参数后，分别执行：

| 背景负载 | 强度 | CPU/iGPU 比例 |
|---|---|---|
| 无负载 | 0 | 100/0、75/25、50/50、25/75、0/100、动态 |
| 计算型 | 4、8、12、20 workers | 同上 |
| 内存型 | 低、中、高带宽 | 同上 |
| 混合型 | 计算 + 内存 | 同上 |

每组至少记录 warmup 后的 decode token/s、TTFT、TOPT p50/p95/p99、CPU 专家时间、iGPU 专家时间、同步等待时间、进程 RSS/PSS、系统内存带宽和 CPU PSI。

## 7. 实验记录

### 2026-07-17：初始化

- 建立开发记录和阶段验收条件。
- 确认当前分支没有已跟踪文件改动。
- 确认已有 SYCL 后端使用 shared-USM，但 VNNI 与 SYCL 的持久化权重布局不同。
- 下一步：补充可重复的单层基准，随后实现 shared packed weight view 和 packed-VNNI kernel。

### 2026-07-17：共享权重与固定比例原型

实现内容：

- 增加 output-major packed GPTQ INT4 的 AVX-VNNI-256 kernel。每次从 packed INT4 中加载并在寄存器内展开 32 个权重，使用 `vpdpbusd` 完成点积。
- SYCL shared-USM 权重块中增加 INT16 weight sums，CPU packed-VNNI 通过只读 view 直接绑定 `qweight`、`scales` 和 `weight_sums`，不再分配持久化 INT8 展开副本。
- 增加 `CPUiGPUGPTQInt4_MOE`。按 routed row 数选择完整专家，避免同一个专家在 CPU 和 iGPU 两端重复执行；decode 使用常驻 iGPU worker 与 CPU 并发。
- prefill 暂时串行。原因是现有 SYCL prefill 路由和 CPU packed-VNNI 路径都会使用同一个可变 CPU worker pool，在 submit/wait 拆分前不能安全并发。
- 新增 `bench_gptq_int4_backends.py`，统一权重、路由、shape、线程数、warmup、迭代次数和 JSONL 输出。

正确性命令：

```bash
PYTHONPATH="$PWD/kt-kernel/python:${PYTHONPATH:-}" \
KT_TEST_SYCL_GPTQ_INT4=1 \
python -m pytest -q -s kt-kernel/test/per_commit/test_moe_gptq_int4_accuracy.py
```

最终结果：`10 passed`。packed-VNNI 与已有 unpacked-VNNI 输出完全一致；混合后端的 0% 和 100% 分别与 packed-VNNI 和 SYCL 输出一致。各中间比例相对量化参考的误差仍在现有后端阈值内；新增的 3 个用例覆盖动态默认值、固定比例环境配置和非法 policy。

统一单层微基准参数：16 experts、top-k 8、hidden 2048、intermediate 512、group size 128、8 CPU workers、50 次计时、10 次 warmup。原始数据保存在本次开发会话的 `/tmp/kt_gptq_baseline.jsonl`。

| 后端 | Decode qlen=1 mean | Prefill qlen=128 mean | `load_weights` RSS 增量 |
|---|---:|---:|---:|
| AVX2 | 0.508 ms | 60.540 ms | 26.8 MiB |
| 原有 unpacked VNNI | 0.192 ms | 14.463 ms | 51.8 MiB |
| SYCL | 0.976 ms | 24.089 ms | 1.5 MiB（runtime 已提前初始化，不能代表完整 USM） |
| packed VNNI | 0.202 ms | 18.550 ms | 26.1 MiB |

packed-VNNI 相对 unpacked-VNNI 的 decode 慢约 5.2%，prefill 慢约 28.3%，但将此 shape 的持久化 CPU 权重增量从约 51.8 MiB 降到约 26.1 MiB。该取舍满足 CPU/iGPU 共用一份主权重的内存目标，但后续需要继续优化 prefill 的 nibble unpack 和缓存访问。

固定比例混合基准使用相同参数，原始数据保存在 `/tmp/kt_gptq_hybrid.jsonl`：

| iGPU routed-row 目标比例 | Decode mean | Decode p95 | Prefill mean |
|---:|---:|---:|---:|
| 0% | 0.281 ms | 0.486 ms | 18.820 ms |
| 25% | 0.602 ms | 1.073 ms | 21.248 ms |
| 50% | 0.477 ms | 0.637 ms | 22.260 ms |
| 75% | 0.767 ms | 1.072 ms | 23.151 ms |
| 100% | 0.673 ms | 0.815 ms | 24.361 ms |

结论：

- 空载 decode 下中间比例均不优于 CPU-only；动态策略必须默认稳定选择 CPU，不能把 50/50 作为默认分配。
- 0% 混合封装比直接 packed-VNNI 多约 0.079 ms decode 均值，主要来自每次构造两份 expert-id map、两个 FP32 partial output 和最终 merge；后续应为 0%/100% 增加直通路径。
- 当前 prefill 为串行执行，比例越高延迟基本越大。在安全实现并发前，动态 prefill 应优先在 CPU-only 与 iGPU-only 间切换，而不是使用中间比例。
- 中间比例按完整专家分配，实际 routed-row 比例受当次路由离散性影响。decode top-k 8 时只有 8 个活跃专家，这也是不同固定比例延迟不严格单调的原因之一。
- 混合实例的 `load_weights` RSS 增量约 44 KiB 是在 SYCL runtime 和 shared-USM 已经建立后的差值，不能作为完整权重内存证据。共享关系由 CPU BufferB view 直接绑定同一组 USM 指针保证，压力实验还需单独记录进程峰值和 USM 分配量。

### 2026-07-17：外部负载与动态控制器

负载工具：

- 新增 `cpu_background_load.py`，每个进程绑定到指定 CPU。计算型负载循环执行缓存内 SHA-256，内存型负载循环执行两份大 buffer 的 `memmove`。
- `bench_gptq_int4_backends.py` 可直接启动和回收负载进程组，并记录 worker PID/CPU、绑核 busy、CPU/memory PSI、观测时长、动态最终比例和控制器调试统计。
- 修复了负载工具第一次实现中的退出死锁：signal handler 不再直接重入 multiprocessing Event 锁，超时清理以整个新进程组为单位执行。

固定比例压力结果表明，中间比例不适合作为当前默认策略。下表来自 50 次计时、10 次 warmup；计算负载与 CPUInfer 的 CPU 0-7 重叠：

| 负载 | iGPU 比例 | Decode mean | Decode p95 | Prefill mean |
|---|---:|---:|---:|---:|
| compute 4 workers | 0% | 0.601 ms | 4.179 ms | 27.340 ms |
| compute 4 workers | 50% | 0.962 ms | 2.912 ms | 32.800 ms |
| compute 4 workers | 100% | 0.762 ms | 0.856 ms | 24.331 ms |
| compute 8 workers | 0% | 0.841 ms | 5.160 ms | 52.178 ms |
| compute 8 workers | 50% | 5.901 ms | 6.153 ms | 56.118 ms |
| compute 8 workers | 100% | 0.784 ms | 0.959 ms | 31.859 ms |

结论：4-worker 时 decode 的 CPU 平均值仍较好，但 iGPU 尾延迟更稳定，prefill 已应切换 iGPU；8-worker 时 decode 和 prefill 都应切换 iGPU。混合比例只要保留 CPU 分量，就会等待被系统调度延迟的 CPU 尾部，无法获得预期的 `max(T_cpu, T_igpu)` 收益。

内存型压力对 CPU 与 iGPU 的共享 DRAM 都有影响：

| 负载 | 路径 | Decode mean | Decode p95 | Prefill mean |
|---|---|---:|---:|---:|
| memory 1 worker | CPU | 0.281 ms | 0.514 ms | 19.471 ms |
| memory 1 worker | iGPU | 0.963 ms | 1.255 ms | 24.646 ms |
| memory 4 workers | CPU | 0.389 ms | 1.899 ms | 27.318 ms |
| memory 4 workers | iGPU | 1.212 ms | 1.606 ms | 29.294 ms |

因此第一版动态控制器采用二元设备策略，而不是默认拆分专家：

- 所有层共享一个 50 ms 采样线程。采样器只观察 CPUInfer 实际绑定的 CPU，将这些 CPU 的系统 busy ticks 减去本进程 CPU ticks，再与 CPU PSI `some` stall 比例取最大值并做 EWMA。这样不会把框架自身正常占满 CPU 误判为外部负载。
- decode 使用 load hysteresis：低于 0.20 回 CPU，高于 0.60 切 iGPU，最小保持 4 次。短 decode 的单次服务时间受 Linux scheduler quantum 影响过大，不用于覆盖该决策。
- prefill 低于 0.05 回 CPU，高于 0.25 建立 high-load epoch；每个 epoch 先采 3 次 CPU，再至少采 10 次 iGPU 以越过 SYCL 冷启动，之后比较每 routed-row 服务时间 EWMA，低阈值之前不会重复校准。
- 0%/100% 使用直通路由，只保留一份 FP32 partial output。expert-id map 和 partial output 改为 CPUInfer 执行线程的 `thread_local` scratch，所有层复用，避免大 prefill 后每层长期保留两份输出。中间比例的 iGPU worker 改为首次需要并发时才创建。

最终控制器微基准使用 100 次计时、20 次 warmup：

| 负载 | Decode 选择 / mean / p95 | Prefill 选择 / mean | 设备切换次数（decode / prefill） |
|---|---|---|---:|
| 无负载 | CPU / 0.211 ms / 0.265 ms | CPU / 18.598 ms | 0 / 0 |
| compute 4 workers | CPU / 0.802 ms / 4.244 ms | iGPU / 25.258 ms | 0 / 1 |
| compute 8 workers | iGPU / 0.689 ms / 1.077 ms | iGPU / 32.959 ms | 1 / 1 |
| memory 4 workers | CPU / 0.662 ms / 4.108 ms | iGPU / 28.449 ms | 0 / 3 |

对照实验中，compute 8 的固定 CPU decode 为 3.000 ms mean / 9.018 ms p95，固定 iGPU 为 0.853 ms / 1.338 ms；动态选择 iGPU 后为 0.689 ms / 1.077 ms。compute 4 的固定 iGPU prefill 为 24.683 ms，动态为 25.258 ms。结果存在进程启动相位、频率和调度量子造成的跨进程波动，因此控制器选择与尾延迟比单次绝对均值更有参考价值。

内存 4-worker prefill 的动态结果 28.449 ms，仍比同轮固定 CPU 25.705 ms 慢约 10.7%。本机暴露 `uncore_imc_free_running/data_read` 和 `data_write` PMU，但 `perf_event_paranoid=4`，普通进程不能读取，不能把 IMC 带宽作为默认信号。当前通过服务时间反馈减少错误迁移，但不能在很少的 prefill 调用内可靠识别所有共享带宽场景；这是下一轮需要解决的主要残余问题。

生产接入：

- 新增 method `CPU_IGPU_GPTQ_INT4`，复用 `GPTQSafeTensorLoader`，默认 `KT_CPU_IGPU_POLICY=dynamic`。
- `KT_CPU_IGPU_POLICY=fixed KT_CPU_IGPU_RATIO=<0..1>` 可复现固定比例。
- `KT_CPU_IGPU_DECODE_LOAD_LOW/HIGH`、`KT_CPU_IGPU_PREFILL_LOAD_LOW/HIGH`、`KT_CPU_IGPU_LOAD_EWMA_ALPHA`、`KT_CPU_IGPU_LOAD_SAMPLE_MS` 和两个 `MIN_DWELL` 环境变量可用于端到端调参。
- 修复 `WorkerPool::set_restricted_worker_count(count)` 忽略 `count` 以及线程总数字段未初始化的问题，并向 Python 暴露 worker count 与实际绑定 CPU 列表。

原始 JSONL：`/tmp/kt_gptq_compute_load_20260717.jsonl`、`/tmp/kt_gptq_memory_load_20260717.jsonl`、`/tmp/kt_gptq_dynamic_v5_20260717.jsonl`。这些是本机开发会话的临时原始数据；上述表格是需要长期保留的结果摘要。

### 2026-07-17：端到端论文实验脚本

新增 `kt-kernel/bench/bench_cpu_igpu_e2e.py`，在完全相同的 SGLang 参数下比较：

- `vnni-only`：`kt-method=GPTQ_INT4`，强制 `KT_GPTQ_INT4_BACKEND=avxvnni` 和 fused VNNI。
- `vnni-sycl-dynamic`：`kt-method=CPU_IGPU_GPTQ_INT4`，启用 oneAPI、Level Zero iGPU 和动态 policy。

两组均禁用 radix cache 和 CUDA graph，使用相同的 24 个 NVIDIA GPU experts、8 个 CPUInfer P-core workers、prefill chunk 和请求顺序。脚本支持 `none`、`compute:N`、`memory:N` 负载矩阵，负载固定到指定 CPU；每个 server repetition 会交替后端启动顺序，场景和请求顺序由固定 seed 随机化。

每次请求记录实际/估算 token 数、TTFT、prefill token/s、稳态 decode token/s、TOPT、端到端时间、输出 SHA-256、绑核 busy、CPU/memory PSI 和服务进程组峰值 RSS/PSS。输出包括：

- `manifest.json`：硬件、git、命令、环境、运行状态和日志索引。
- `samples.jsonl`：逐请求立即落盘的原始样本。
- `summary.csv`：mean、p50、p95、标准差和 bootstrap 95% CI。
- `comparisons.csv`：只使用两后端都成功的成对样本，给出动态调度相对 VNNI-only 的吞吐或延迟 speedup、成对 bootstrap 95% CI，以及输出 SHA-256 完全匹配率。
- `report.md`：自动生成的论文实验摘要，包含后端均值、置信区间、speedup、输出一致率、完整服务生命周期的进程组峰值 RSS/PSS 和原始结果文件链接。

正式计时请求的 prompt nonce 不含 backend 名称，同一个 `(server repetition, load, workload, request repetition)` 在两个后端使用逐字节相同的 prompt；`samples.jsonl` 同时记录 `prompt_sha256` 供事后审计。warmup prompt 允许随 backend 不同，因为 warmup 不参与性能和正确性统计。

脚本对服务和负载生成器都使用独立进程组；启动、健康检查、ready JSON 解析或实验中途失败时会回收整个进程组。每个 scenario 在 `manifest.json` 中单独标记 `running`、`ok`、`error` 或 `interrupted`，失败信息会立即持久化；未指定 `--fail-fast` 时继续执行后续 scenario。

进程组内存监控在服务 `Popen` 成功后立即启动，因此 server-level RSS/PSS 峰值覆盖模型加载、GPTQ 权重打包和稳态请求；每个 scenario 仍单独记录进入该场景后的峰值。早期 smoke 的监控是在健康检查后启动，不能用于分析加载峰值，正式实验必须使用当前版本。

源码构建的 Python 包位于 `kt-kernel/build/lib.*`，而 `kt-kernel/python` 是 `kt_kernel` 的包内容，不能直接作为包父目录加入 `PYTHONPATH`。端到端脚本会自动寻找 `build/lib.*`，也可通过 `--kt-kernel-package-root` 显式指定；扩展更新后应先同步 Python 文件：

```bash
cd kt-kernel
python setup.py build_py
cd ..
```

两种被测后端都加载同一个含 SYCL 的扩展，因此即使 VNNI-only 不向 iGPU 派发计算，加载扩展时仍需要 oneAPI 动态库。脚本现在为两端使用相同的 oneAPI runtime，并在创建实验目录前验证 `kt_kernel` 的实际加载路径和 `CPUiGPUGPTQInt4_MOE` 符号。若服务、scenario、请求或任一预期成对样本缺失，manifest 标记为 `complete_with_errors` 且命令以非零状态退出，避免把空 CSV 当成有效实验。

第一次真实 smoke 在该预检加入前执行，两个服务都在模型构造阶段失败：VNNI-only 因缺少 oneAPI 动态库而被 SGLang 报告为 `kt_kernel is not installed`，动态后端则加载了 `site-packages` 中不含新 method 的旧包。失败记录保存在 `artifacts/cpu-igpu-e2e/smoke-20260717/`，没有产生性能样本；这不是性能实验结果。

第二次 smoke 的 VNNI 两个场景成功，动态后端在第一层加载时因 `SYCL GPTQ INT4 requires scale tensors` 中止。原因是 `NativeMoEWrapper` 使用逐专家 mmap tensor 指针，而混合后端直接持有的 SYCL TP 部件当时只接受连续 tensor 指针。SYCL TP 现已同时支持两种 `GeneralMOEConfig` 布局；准确性测试中的混合后端 `ratio=0/0.5/1` 改用生产逐专家布局，三条路径均通过。失败记录位于 `artifacts/cpu-igpu-e2e/smoke-fixed-20260717/`，其中仅有 VNNI 样本，不能用于后端对比。

第三次 smoke 中动态后端已完成 40 层权重加载并通过健康检查，但 runner 在请求前拒绝了服务：原先校验依赖 C++ `printf` 的 packed-backend 文本，stdout 连接管道时采用块缓冲，健康检查时标记尚未 flush。后端构造现在由 Python logger 输出稳定的 `KT_SELECTED_MOE_BACKEND=...` 标记，runner 改为校验该标记。失败记录位于 `artifacts/cpu-igpu-e2e/smoke-paired-20260717/`，同样只有 VNNI 样本。

第四次 smoke 的动态后端完成 15 次校准和两个负载场景的全部请求，计算负载下短、长 prompt 的 decode 分别约为 17.91 和 19.14 token/s；同轮 VNNI 样本因 GPTQ 稳定标记被误加到 RAWINT4 分支而未执行，前一轮 VNNI 在相同负载下约为 1.78--2.62 token/s，因此这些跨运行数字只能作为调试信号，不能作为论文 speedup。动态服务退出时又发现 `Popen.communicate()` 与 tee 线程并发读取 stdout 会触发 `EBADF`，进程组回收已改为 `SIGTERM -> wait -> SIGKILL -> wait`，由 tee 线程独占 stdout。该轮记录位于 `artifacts/cpu-igpu-e2e/smoke-complete-20260717/`。

最终 paired smoke 位于 `artifacts/cpu-igpu-e2e/smoke-final-20260717/`：两个 server 和四个 scenario 均为 `ok`，8/8 请求成功；四组成对 prompt SHA-256 全部相同，输出 SHA-256 完全匹配率均为 100%。该轮只使用 1 次服务启动、每条件 1 次请求和 3 次 warmup，bootstrap 区间退化为单点，仅用于证明端到端链路闭环，严禁作为论文结果。调试数值如下：

| 负载 / 工作负载 | Prefill speedup | Decode speedup | E2E speedup |
|---|---:|---:|---:|
| compute-8 / p1-o8 | 0.812x | 8.272x | 2.265x |
| compute-8 / p1024-o8 | 1.130x | 9.687x | 1.448x |
| none / p1-o8 | 1.342x | 1.249x | 1.292x |
| none / p1024-o8 | 0.804x | 1.333x | 0.824x |

其中 speedup 对吞吐定义为 dynamic/VNNI，对延迟定义为 VNNI/dynamic。无负载长 prompt 的 0.804x 和 compute-8 短 prompt prefill 的 0.812x 表明 prefill 策略仍需正式多重复实验校准；不能只报告 decode 的高提升。该 smoke 完成后，server-level 内存监控起点又提前到 `Popen` 后，因此上述 artifact 的内存字段不用于加载峰值分析，后续正式实验使用当前脚本重新采集。

先验证实验矩阵而不启动模型：

```bash
python kt-kernel/bench/bench_cpu_igpu_e2e.py --dry-run
```

建议先执行小规模端到端 smoke test：

```bash
python kt-kernel/bench/bench_cpu_igpu_e2e.py \
  --loads none,compute:8 \
  --workloads 1:32,1024:64 \
  --request-repetitions 1 \
  --scenario-warmups 15 \
  --output-dir artifacts/cpu-igpu-e2e/smoke
```

面向消费级端侧多任务的论文实验使用合成计算 worker，不依赖具体编译器或仿真软件。保留两档 Linux CFS 优先级：负 `nice` 的高优先级背景是调度方法和后端优化的主实验，`nice=0` 是验证结论不依赖优先级设置的对照实验。`--load-affinity free` 不给背景 worker 绑核，由 Linux 在全部在线 P/E core 上自由调度；worker 数表示背景并发度而不是名义 CPU 百分比。

主实验建议使用 `nice=-5`，至少 3 次独立服务启动、每个条件 5 次请求：

```bash
python kt-kernel/bench/bench_cpu_igpu_e2e.py \
  --load-affinity free \
  --load-nice -5 \
  --loads none,compute:1,compute:2,compute:4,compute:8,compute:12,compute:16,compute:20 \
  --workloads 1:300,1024:300,4096:300,8192:300 \
  --server-repetitions 3 \
  --request-repetitions 5 \
  --scenario-warmups 15 \
  --output-dir artifacts/cpu-igpu-e2e/paper-high-priority
```

默认优先级对照使用相同矩阵，只改变 `nice` 和输出目录：

```bash
python kt-kernel/bench/bench_cpu_igpu_e2e.py \
  --load-affinity free \
  --load-nice 0 \
  --loads none,compute:1,compute:2,compute:4,compute:8,compute:12,compute:16,compute:20 \
  --workloads 1:300,1024:300,4096:300,8192:300 \
  --server-repetitions 3 \
  --request-repetitions 5 \
  --scenario-warmups 15 \
  --output-dir artifacts/cpu-igpu-e2e/paper-default-priority
```

负 `nice` 需要实验用户具备 `CAP_SYS_NICE` 或系统配置允许的 `RLIMIT_NICE`。runner 会在加载 35B 模型前启动一个短 worker，核对每个 worker 的实际 `nice`；权限不足或内核未应用请求值时直接终止，不能静默退回 `nice=0`。实验不使用 `SCHED_FIFO`/`SCHED_RR` 实时策略，因为它们可能造成推理进程和桌面服务饥饿，不符合日常多任务场景。

2026-07-17 本机权限探测：当前开发用户的 `RLIMIT_NICE=0`。`free + nice=0 + 2 workers` 验证成功，两个 worker 的 allowed CPU 均为 `0-19`，请求值和实际值均为 0；`free + nice=-5 + 1 worker` 在 `setpriority` 处返回 `PermissionError(13)`。因此当前环境只能执行默认优先级对照，高优先级主实验标记为“等待系统权限配置”，在权限配置完成并重新通过 preflight 前不得采集论文数据。

机理对照实验继续使用固定 P-core 重叠负载，并保持与主实验相同的 `nice=-5`：

```bash
python kt-kernel/bench/bench_cpu_igpu_e2e.py \
  --load-affinity pinned \
  --load-nice -5 \
  --load-cpus 0-7 \
  --loads none,compute:2,compute:4,compute:6,compute:8,compute:16 \
  --workloads 1:300,1024:300,4096:300 \
  --server-repetitions 3 \
  --request-repetitions 5 \
  --scenario-warmups 15 \
  --output-dir artifacts/cpu-igpu-e2e/paper-pcore-control
```

高优先级自由调度主实验回答“前台计算任务优先获得 CPU 时，动态 CPU-iGPU 调度能否保持推理性能”，默认优先级对照回答该收益是否仍存在于普通并发条件，固定 P-core 对照回答“加速是否来自直接 P-core 竞争”。`summary.csv`、`comparisons.csv` 和 `report.md` 均记录 `load_affinity` 与 `load_nice`，不能跨 affinity 或 nice 合并统计。`scenario-warmups=15` 不是普通 kernel warmup：它保证动态 prefill policy 在正式计时前完成 3 次 CPU 和至少 10 次 iGPU 服务成本校准。

自由模式下，每个 sample 同时记录全部在线 CPU 的 `cpu_busy_fraction`、CPU 0-7 目标集合的 `target_cpu_busy_fraction`、逐核 `cpu_busy_by_cpu` 和系统 CPU PSI。这样可以判断背景 worker 在低并发时是否主要落在空闲 E-core，以及并发升高后何时开始挤占 CPUInfer 的 P-core。背景 worker 的 ready JSON 记录每个进程的继承 affinity、请求 `nice` 和实际 `nice`；本机 2-worker 验证显示二者均允许在 CPU 0-19 上迁移。

背景任务完成时间不是本研究的优化目标，但不能因此忽略背景负载是否真正建立。论文结果至少同时报告实际 CPU busy、CPU PSI 和背景 worker 的有效优先级；否则“推理更快”可能只是背景 worker 没有获得预期 CPU 时间。调度器与后端参数先在高优先级主实验上确定，默认优先级对照只复用已冻结的参数，避免对测试集二次调参。

上述稀疏并发点用于昂贵的 35B 多 workload 主实验。如果论文需要画完整的并发响应曲线，可另用单一短 workload 扫描 `none,compute:1,...,compute:20`；不要把 21 个并发级别与所有 prompt 长度做全笛卡尔积。

### 2026-07-18：连接手动启动引擎的性能客户端

增加 `kt-kernel/bench/bench_running_server.py`。该脚本只连接已启动的 OpenAI-compatible SGLang 服务，不负责启动、修改或停止引擎，也不自动创建背景负载。它从服务端 streaming usage 获取真实输入/输出 token 数，保存逐请求 `samples.jsonl`、聚合 `summary.csv`、`manifest.json` 和 `report.md`。

指标定义固定为：TTFT 是发送请求到收到第一个非空输出 token 的时间；prefill throughput 为输入 token 数除以 TTFT；decode throughput 使用首、末输出 token 之间的 `completion_tokens - 1` 个区间；TPOT 是同一区间的平均耗时；`output_phase_ms` 是首 token 到末 token 的总输出阶段时间；TTLT 是请求开始到末 token；E2E 还包括末 token 后结束 SSE stream 的开销。本文不使用含义不统一的“TOFT”缩写；若其本意是 TOPT/TPOT，使用 `tpot_ms`，若指总输出阶段则使用 `output_phase_ms`。

手动启动服务并确认健康后执行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label vnni-only__nice-minus5__compute8 \
  --workloads 1:300,1024:300,4096:300,8192:300 \
  --warmups 15 \
  --repetitions 5 \
  --output-dir artifacts/running-server-bench/vnni-only__nice-minus5__compute8
```

切换后端后保持 `seed`、workload、warmup 和 repetition 完全相同，只更改 `run-label` 与输出目录。服务启动参数必须包含 `--disable-radix-cache`；客户端也会让长 prompt 从首部开始变化，以进一步避免跨请求 prefix cache。请求固定使用 `temperature=0`、`seed=20260718`、`ignore_eos=true`，默认拒绝缺少 server usage 或未生成指定 token 数的样本。动态 CPU-iGPU 后端保留 15 次 warmup 完成控制器校准。

为手动实验增加统一启动脚本 `perf-log/35b-test-cpu-igpu.sh`。默认使用本地 `kt-kernel/build/lib.*` 和仓库内 SGLang，加载 oneAPI 后检查 method、扩展符号和 Level Zero GPU，并自动把服务日志写入 `artifacts/server-logs/`。两种被测后端使用同一组服务参数：

```bash
# CPU-iGPU 动态调度
bash perf-log/35b-test-cpu-igpu.sh dynamic

# VNNI-only 对照
bash perf-log/35b-test-cpu-igpu.sh vnni-only
```

默认 endpoint 为 `http://127.0.0.1:30100`，与手动 benchmark 客户端一致。可通过环境变量覆盖配置，例如 `PORT=30101 KT_CPUINFER=8 LOG_FILE=/tmp/dynamic.log bash perf-log/35b-test-cpu-igpu.sh dynamic`。`DRY_RUN=1` 只打印最终命令；`PREFLIGHT_ONLY=1` 会实际加载 oneAPI 并验证设备、Python package 和扩展符号，但不加载模型权重。脚本只设置引擎后端，不改变引擎 nice；高优先级背景 worker 仍应在独立终端启动，避免把推理服务本身提升为高优先级。

2026-07-18 本机真实 preflight：两种模式均成功加载 `kt-kernel/build/lib.linux-x86_64-cpython-311/kt_kernel`。dynamic 模式确认 Level Zero iGPU 可见且 `CPUiGPUGPTQInt4_MOE` 存在；vnni-only 模式确认 `AVXVNNI256GPTQInt4_MOE` 存在。该检查没有加载 35B 权重，不构成端到端启动或性能结果。

### 2026-07-18：dynamic 无负载完整手动测试

结果位于 `artifacts/running-server-bench/dynamic-no-load-full-20260718/`，服务日志为 `artifacts/server-logs/20260718_103018-dynamic.log`。服务后端标记为 `KT_SELECTED_MOE_BACKEND=CPU_IGPU_GPTQ_INT4`；25/25 请求成功，全部使用 server usage，且每次严格生成 300 tokens。5 次重复的结果如下：

| Workload | 实际 prompt tokens | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 1 | 11.33 [10.79, 11.63] | 29.58 [29.50, 29.66] | 88.48 [85.94, 93.04] | 33.81 [33.71, 33.90] |
| p1024-o300 | 1046 | 169.60 [168.84, 170.04] | 29.55 [29.47, 29.63] | 6167.53 [6152.04, 6195.46] | 33.85 [33.75, 33.94] |
| p2048-o300 | 2070 | 169.14 [168.71, 169.39] | 29.52 [29.44, 29.62] | 12238.38 [12219.55, 12269.14] | 33.87 [33.76, 33.96] |
| p4096-o300 | 4118 | 168.34 [167.75, 168.83] | 29.58 [29.51, 29.64] | 24462.63 [24398.53, 24550.41] | 33.81 [33.74, 33.89] |
| p8192-o300 | 8214.2 | 169.17 [168.98, 169.34] | 29.45 [29.27, 29.58] | 48556.54 [48508.12, 48604.96] | 33.96 [33.80, 34.15] |

1K--8K 的 TTFT 随实际输入 token 数近似线性增长，packed CPU 路径的端到端 prefill 吞吐稳定在约 168--170 token/s。不能直接用旧 `perf.py` 的 2K/4K/8K VNNI 数据计算回退：旧 VNNI 启动未禁用 radix cache，benchmark 又按 1K、2K、4K、8K 顺序发送具有相同前缀的 prompt，后续请求可能复用前一请求的 KV cache；旧表 8K TTFT 低于 4K 也是明显异常信号。旧 1K 是该序列中第一个长 prompt，相对较少受此前短 `hi` 请求影响；其 235.3 token/s 与本轮 169.6 token/s 相差约 27.9%，方向与单层 packed-VNNI prefill 回退一致，但在用当前客户端完成同条件 vnni-only 配对前仍不作为正式归因结果。

### 2026-07-18：vnni-only 无负载同条件配对

VNNI 结果位于 `artifacts/running-server-bench/vnni-only-no-load-full-20260718/`，服务日志为 `artifacts/server-logs/20260718_120338-vnni-only.log`。日志确认后端为 `GPTQ_INT4:AVXVNNI256GPTQInt4_MOE`；25/25 请求成功，token 数、workload、seed、warmup 和 repetition 与 dynamic 完全一致。VNNI-only 结果如下：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] |
|---|---:|---:|---:|---:|
| p1-o300 | 7.16 [5.87, 7.97] | 21.72 [20.24, 23.03] | 145.22 [125.55, 177.34] | 46.35 [43.46, 49.82] |
| p1024-o300 | 215.32 [211.98, 218.59] | 24.21 [24.07, 24.31] | 4859.35 [4785.43, 4936.19] | 41.31 [41.14, 41.55] |
| p2048-o300 | 221.00 [218.54, 222.53] | 24.19 [24.10, 24.28] | 9367.63 [9302.19, 9473.52] | 41.34 [41.18, 41.50] |
| p4096-o300 | 217.40 [208.54, 222.52] | 23.10 [21.26, 24.07] | 18973.82 [18506.33, 19786.87] | 43.61 [41.55, 47.55] |
| p8192-o300 | 222.10 [220.55, 223.63] | 23.48 [23.33, 23.62] | 36987.16 [36727.25, 37245.94] | 42.58 [42.34, 42.87] |

以 throughput 使用 dynamic/VNNI、E2E latency 使用 VNNI/dynamic 定义 speedup，同条件均值比较为：

| Workload | Packed dynamic prefill speedup | Packed dynamic decode speedup | Packed dynamic E2E speedup |
|---|---:|---:|---:|
| p1-o300 | 1.583x | 1.362x | 1.373x |
| p1024-o300 | 0.788x | 1.221x | 1.057x |
| p2048-o300 | 0.765x | 1.220x | 0.972x |
| p4096-o300 | 0.774x | 1.281x | 0.926x |
| p8192-o300 | 0.762x | 1.254x | 0.847x |

1K--8K packed prefill 比 unpacked VNNI 低约 21%--24%，与单层微基准由延迟换算得到的约 22% throughput 回退一致。Decode 则高约 22%--28%；一个待验证的机理解释是全模型 decode 每 token 流式读取大量专家权重，更小的 packed INT4 表示减少内存流量，而 prefill 对同一专家存在 routed-row 复用，运行时 nibble 解包和计算开销占主导。该解释仍需要硬件计数器或工作集/shape sweep 佐证，不能仅由端到端数据断言。

25 对请求的 prompt SHA-256 全部一致；输出 SHA-256 为 23/25 一致（92%）。两个 mismatch 均来自 p1-o300（repetition 3、4），1K--8K 的 20/20 输出完全一致。短 prompt 的生成分叉需要保留为数值一致性风险，不能报告为全矩阵 100%。本轮每个后端只有一次服务启动，可用于开发决策，正式论文统计仍需多个独立 server repetition。

### 2026-07-18：无特权的相对优先级实验方案

本机启用了 `kernel.sched_autogroup_enabled=1`。如果引擎和背景负载从不同终端 session 启动，仅设置进程 nice 可能只在各自 autogroup 内生效，不能保证跨组的 CPU 份额关系。主实验因此改用同一 user systemd 层级下的两个 transient scope：背景负载保持 `nice=0, CPUWeight=100`，推理引擎使用 `nice=5, CPUWeight=33`。CPUWeight 的相对比例约为 3.03:1；进程 nice 权重中 `nice=0` 对 `nice=5` 约为 3.06:1。这与原先背景 `nice=-5` 对引擎 `nice=0` 的约 3.05:1 接近，但不需要 `CAP_SYS_NICE`，也更符合“普通前台任务 + 主动让步的推理服务”这一消费级部署场景。

启动脚本新增两个 profile：

```bash
# 默认优先级对照：引擎 nice=0, CPUWeight=100
ENGINE_PRIORITY=normal ./perf-log/35b-test-cpu-igpu.sh dynamic

# 主实验：引擎 nice=5, CPUWeight=33
ENGINE_PRIORITY=low ./perf-log/35b-test-cpu-igpu.sh dynamic
```

背景负载也应放入独立的默认权重 scope，避免终端 autogroup 干扰：

```bash
systemd-run --user --scope --quiet -p CPUWeight=100 \
  python kt-kernel/bench/cpu_background_load.py \
  --kind compute --workers 8 --affinity free --nice 0
```

论文表述应称为“推理服务低优先级/前台负载相对优先”，不能再称背景任务具有系统级高优先级。原 `nice=-5` 方案保留为可选敏感性实验，不作为主实验环境。正式自动化 runner 尚需增加 engine scope/CPUWeight 参数；在该功能完成前，上述配置先用于手动高负载 smoke。

### 2026-07-18：dynamic 低优先级引擎与 compute-8 smoke

结果位于 `artifacts/running-server-bench/dynamic-engine-low-compute8-smoke-20260718/`。动态引擎使用 `nice=5`，背景计算负载包含 8 个 `nice=0` worker；6/6 请求成功。3 次重复的结果如下：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] |
|---|---:|---:|---:|---:|
| p1-o300 | 10.26 [10.22, 10.35] | 24.03 [24.01, 24.05] | 97.45 [96.63, 97.87] | 41.62 [41.58, 41.65] |
| p1024-o300 | 169.55 [168.82, 170.10] | 24.06 [23.98, 24.11] | 6169.32 [6149.18, 6196.01] | 41.56 [41.47, 41.71] |

相对 dynamic 无负载均值，p1 prefill 下降约 9.4%，p1024 prefill 下降约 0.03%，两种 workload 的 decode 分别下降约 18.8% 和 18.6%。这说明动态后端在该负载下仍保持了约 24 token/s decode，且长 prompt prefill 基本未变；但这只是同一后端的性能保持率，不是相对 VNNI-only 的 speedup。

实验结束后，在引擎和背景负载均继续运行时进行了现场核对。本机 CPU 0--7 的最高频率为 5.5--5.6 GHz，对应 8 个 P 核；CPU 8--19 的最高频率为 4.6 GHz，对应 12 个 E 核。8 个背景 worker 当时分别驻留在 CPU 0--7，每个进程约占用 100% CPU；连续两秒 `mpstat -P ALL` 也显示 CPU 0--7 均为 100% busy，而大部分 CPU 8--19 处于空闲。引擎 scheduler 的实际进程 nice 和终端 autogroup nice 均为 5，背景 worker 的实际 nice 和独立 autogroup nice 均为 0。该快照只能说明请求结束后的采样时刻存在直接 P-core 占用，不能证明计时请求期间持续存在相同竞争；后续主动推理期间的采样确认 worker 会迁移到 E-core，因此不再使用这条结束后快照排除 E-core 调度解释。

该观测是在计时请求完成后采集，能用于定位当前 smoke，但不能替代论文实验中的请求期间连续遥测。当前手动客户端也没有保存调度器的 CPU load、最终 iGPU ratio 和 switch count。按照当前策略，绑定核负载超过高阈值后 decode 会选择 iGPU；prefill 则先采集 3 次 CPU 和 10 次 iGPU 服务时间，再选择较快设备。因此本次约 24 token/s decode 和基本不变的 1K prefill 与动态迁移生效相符，但在增加调度器 telemetry 前不能仅凭吞吐量断言实际分流路径。

下一项必要对照是在完全相同的 `engine nice=5 + background nice=0 + compute:8` 条件下运行 VNNI-only，保持 workload、15 次 warmup、seed 和 repetition 相同。只有比较 `dynamic@compute8` 与 `vnni-only@compute8`，才能判断动态调度是否降低了竞争造成的性能损失；正式实验还应在每个请求期间记录逐核 busy、CPU PSI、调度器 load、iGPU ratio 和切换次数。

### 2026-07-18：VNNI-only 低优先级引擎与 compute-8 配对 smoke

结果位于 `artifacts/running-server-bench/vnni-only-engine-low-compute8-smoke-20260718/`，服务日志为 `artifacts/server-logs/20260718_132734-vnni-only.log`。日志和运行时状态确认引擎使用 `profile=low, nice=5, CPUWeight=33`，后端为 `GPTQ_INT4:AVXVNNI256GPTQInt4_MOE`；原来的 8 个 `nice=0` compute worker 保持运行，但该 smoke 没有记录请求期间的实际落核。6/6 请求成功，3 次重复结果如下：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] |
|---|---:|---:|---:|---:|
| p1-o300 | 6.48 [2.68, 8.52] | 20.19 [18.42, 22.54] | 204.03 [115.67, 372.92] | 50.16 [44.36, 58.09] |
| p1024-o300 | 205.88 [199.82, 215.62] | 22.24 [22.02, 22.38] | 5086.22 [4851.12, 5234.73] | 44.97 [44.69, 45.42] |

与相同负载下的 dynamic smoke 使用均值直接比较：

| Workload | Dynamic/VNNI prefill | Dynamic/VNNI decode | VNNI/dynamic TTFT | VNNI/dynamic TPOT | VNNI/dynamic E2E |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 1.583x | 1.190x | 2.094x | 1.205x | 1.212x |
| p1024-o300 | 0.824x | 1.082x | 0.824x | 1.082x | 0.997x |

两组的 6/6 prompt SHA-256 完全一致。p1024 的输出为 3/3 完全一致，p1 为 0/3；后者延续了短 prompt 对数值微扰敏感的现象。p1 的 VNNI 样本存在一个 TTFT 372.9 ms、decode 17.2 token/s 的慢样本，N=3 的置信区间很宽，因此短 prompt speedup 只能作为调试信号。p1024 相对稳定：dynamic 的 decode 快约 8.2%，但 prefill 慢约 17.6%；dynamic E2E 18595.45 ms，VNNI E2E 18532.78 ms，两者仅差约 0.34%。按照本轮均值，dynamic 多出的约 1083.1 ms TTFT 可由每个 decode interval 节省的约 3.413 ms 抵消，交叉点约为 319 个输出 token；这是由当前两条均值外推的工作负载特定结果，不应视为固定常数。

相对各自无负载基线，p1024 dynamic 保留约 99.97% prefill 和 81.44% decode throughput；VNNI-only 保留约 95.62% prefill 和 91.86% decode throughput。dynamic 在负载下的绝对 decode 仍高于 VNNI-only，但其相对空载的 decode 降幅更大。因此本轮可以支持“dynamic 在 compute-8 下提供更高的绝对 decode throughput”，尚不能支持“动态控制器具有更高的性能保持率”这一更强结论。

更重要的是，当前两组不是纯调度器消融：VNNI-only 使用已有 unpacked VNNI 后端，dynamic 使用共享 packed INT4 布局及 packed VNNI/SYCL 后端。二者差值同时包含权重布局、CPU kernel 和调度策略的影响。论文实验矩阵必须至少增加：

- `legacy-vnni-only`：现有产品基线；
- `packed-cpu-fixed`：混合后端固定 `KT_CPU_IGPU_POLICY=fixed, KT_CPU_IGPU_RATIO=0`，用于隔离调度收益；
- `igpu-fixed`：混合后端固定 ratio 1，作为设备端点和控制器选择依据；
- `dynamic`：待评价的动态策略。

主结果可报告 dynamic 相对 legacy VNNI 的系统级收益，但调度创新点必须使用 dynamic 相对 packed CPU fixed、fixed iGPU 及两者逐条件 oracle 的收益或 regret。完成这些启动模式和调度 telemetry 后再执行昂贵的完整负载矩阵，避免采集无法支撑归因的论文数据。

### 2026-07-18：VNNI-only compute-8 全 workload 开发基线

结果位于 `artifacts/running-server-bench/vnni-only-engine-low-compute8-full-20260718/`，与前述 smoke 复用同一个 low-priority VNNI 服务和持续运行的 compute-8 背景负载。25/25 请求成功，使用 15 次 warmup、5 次重复、seed 20260718 和 2000 次 bootstrap。结果如下：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] |
|---|---:|---:|---:|---:|
| p1-o300 | 7.96 [7.67, 8.27] | 23.11 [22.91, 23.28] | 125.90 [121.49, 130.36] | 43.28 [42.96, 43.66] |
| p1024-o300 | 212.20 [209.55, 214.40] | 22.66 [21.75, 23.36] | 4930.19 [4878.78, 4992.86] | 44.22 [42.65, 46.07] |
| p2048-o300 | 214.99 [214.24, 215.74] | 22.74 [22.10, 23.39] | 9628.56 [9594.91, 9660.92] | 44.03 [42.77, 45.43] |
| p4096-o300 | 216.24 [215.18, 217.24] | 22.51 [21.96, 23.08] | 19044.04 [18954.05, 19134.12] | 44.47 [43.42, 45.51] |
| p8192-o300 | 217.01 [215.86, 218.18] | 22.15 [21.62, 22.97] | 37853.71 [37651.89, 38052.85] | 45.20 [43.62, 46.27] |

相对 VNNI-only 无负载全 workload 基线，长 prompt 的变化为：

| Workload | Prefill throughput 变化 | Decode throughput 变化 | E2E latency 变化 |
|---|---:|---:|---:|
| p1024-o300 | -1.45% | -6.40% | +5.46% |
| p2048-o300 | -2.72% | -5.99% | +4.90% |
| p4096-o300 | -0.53% | -2.56% | +1.02% |
| p8192-o300 | -2.29% | -5.67% | +3.32% |

p1 在本轮已经稳定，但其 compute-8 均值反而好于此前波动很大的无负载 VNNI 单次服务结果，因此不使用该跨启动比值解释负载效应。长 prompt 表明：compute-8 自由调度负载曾在相邻 smoke 的现场观测中直接占满 P-core 0--7，但本轮没有请求期间逐核记录；legacy VNNI 的 prefill 仅下降约 0.5%--2.7%，decode 下降约 2.6%--6.4%。这说明端到端路径对 CPU 算力竞争的敏感度低于早期预期，可能还受到 NVIDIA GPU experts、attention、内存带宽和各阶段非专家开销限制；具体归因需要 fixed packed CPU 和逐阶段 telemetry，不能仅凭本轮数据确定。

本轮完成后现场短暂出现过一个约占用单核的 `colord-sane` 进程，但检查时进程已经退出，无法证明它在 13:44:40--13:56:34 的计时区间内运行。当前手动客户端没有进程清单或逐请求 CPU 遥测，因此本轮仍定位为开发基线而不是论文正式样本。下一轮 dynamic 必须保持背景负载、priority profile、workload、warmup、repetition、seed 和 bootstrap 完全一致，生成 N=5 的直接配对结果。

### 2026-07-18：主动推理期间的 P/E-core 迁移核查

在 dynamic compute-8 full 运行期间，于 14:19:59--14:20:03 连续采样自由调度背景 worker。8 个实际计算子进程均允许使用 CPU 0--19；采样的大部分时刻中，1 个 worker 位于 P-core 0，另外 7 个分别位于 E-core 10--14、18、19，其中一个随后迁移到 E-core 16。同期 `mpstat -P ALL 1 3` 显示 P-core 1--7 的 `%nice` 接近 100%，对应固定绑定在这些核上的 `nice=5` CPUInfer 线程；上述 E-core 的 `%usr` 接近 100%，对应 `nice=0` 背景 worker。推理阶段结束后再次读取时，8 个背景 worker 又回到了 CPU 0--7。

这说明 `compute:8 + affinity=free` 的实际行为随推理阶段变化：CPUInfer 忙时，Linux 调度器倾向于把可迁移的前台 worker 放到空闲 E-core，而不是让它们与固定 P-core CPUInfer 线程共享 runqueue；CPUInfer 空闲时，worker 又迁回高性能 P-core。此前请求结束后的单次快照恰好捕获了后一状态，因此不能代表计时期间的竞争。用户观察到 E-core 高占用是正确的，也能够解释 legacy VNNI 在 compute-8 下只出现较小性能下降。

实验设计据此分为两条互补路径：

- free-affinity 继续作为消费级混合核系统的主场景，评价操作系统自然迁移任务时的端到端表现；
- pinned `--affinity pinned --cpus 0-7` 作为机理对照，强制背景任务与 CPUInfer 在相同 P-core 上竞争。

自由调度主场景必须继续覆盖 `compute:12/16/20`。当 runnable 背景任务超过空闲 E-core 容量后，才更可能形成持续 P-core 竞争。正式 runner 需要在每个请求期间连续记录背景 PID 的 `processor`、逐核 busy 和调度器 telemetry；单次 `ps`、任务管理器截图或请求结束后的 `mpstat` 都不能用于判定整段请求的落核。当前 CPU load monitor 只对 CPUInfer 绑定的 0--7 计算外部 busy，并结合系统 CPU PSI；当背景 worker 被迁移到 E-core 且系统仍有空闲核时，控制器可能观测不到高 P-core contention。是否因此保持 CPU 路径，必须通过 iGPU ratio 和 switch count 证实。

### 2026-07-18：dynamic 与 VNNI-only compute-8 全 workload 配对

dynamic 结果位于 `artifacts/running-server-bench/dynamic-engine-low-compute8-full-20260718/`。25/25 请求成功，参数与 VNNI-only full 完全相同：low-priority engine、持续 compute-8 free-affinity 背景、15 次 warmup、5 次重复、seed 20260718 和 2000 次 bootstrap。dynamic 结果如下：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] |
|---|---:|---:|---:|---:|
| p1-o300 | 9.58 [8.80, 10.12] | 23.98 [23.71, 24.16] | 105.13 [98.87, 114.53] | 41.71 [41.40, 42.20] |
| p1024-o300 | 164.51 [163.83, 165.02] | 24.08 [23.93, 24.21] | 6358.43 [6338.52, 6385.92] | 41.54 [41.30, 41.82] |
| p2048-o300 | 163.41 [162.66, 164.18] | 24.07 [23.96, 24.14] | 12667.71 [12609.09, 12725.97] | 41.55 [41.42, 41.74] |
| p4096-o300 | 163.88 [163.10, 164.65] | 23.83 [23.62, 24.05] | 25129.29 [25015.23, 25257.38] | 41.96 [41.59, 42.33] |
| p8192-o300 | 164.74 [164.17, 165.31] | 24.16 [24.13, 24.20] | 49861.58 [49701.07, 50046.53] | 41.39 [41.32, 41.44] |

按 `(workload, repetition)` 配对后，25/25 prompt SHA-256 一致。下表 speedup 对 throughput 定义为 dynamic/VNNI，对 E2E latency 定义为 VNNI/dynamic，因此大于 1 始终表示 dynamic 更好：

| Workload | Prefill speedup [paired 95% CI] | Decode speedup [paired 95% CI] | E2E speedup [paired 95% CI] | 输出 SHA-256 |
|---|---:|---:|---:|---:|
| p1-o300 | 1.204 [1.068, 1.302] | 1.038 [1.019, 1.054] | 1.039 [1.019, 1.056] | 2/5 |
| p1024-o300 | 0.775 [0.769, 0.786] | 1.063 [1.020, 1.108] | 0.967 [0.941, 0.995] | 5/5 |
| p2048-o300 | 0.760 [0.756, 0.764] | 1.058 [1.032, 1.087] | 0.908 [0.893, 0.924] | 4/5 |
| p4096-o300 | 0.758 [0.753, 0.762] | 1.059 [1.029, 1.083] | 0.858 [0.850, 0.865] | 5/5 |
| p8192-o300 | 0.759 [0.757, 0.762] | 1.091 [1.050, 1.117] | 0.825 [0.818, 0.830] | 5/5 |

dynamic 在全部长 prompt 上把 decode throughput 提高约 5.8%--9.1%，但 prefill throughput 降低约 22.5%--24.2%。在固定输出 300 tokens 时，p1 的 E2E 提高约 3.9%，p1024 反而慢约 3.5%，p2048/p4096/p8192 分别慢约 10.1%/16.5%/21.2%。若使用本轮均值并假设 TPOT 随输出长度不变，p1024/p2048/p4096/p8192 的 dynamic E2E 交叉点约为 534/1230/2431/3156 个输出 token；该线性外推忽略 KV 长度增长，只用于说明 prefill 与 decode 的权衡，不作为正式预测模型。

相对 dynamic 自己的无负载基线，compute-8 下长 prompt prefill 保留约 96.6%--97.4%，decode 仅保留约 80.6%--82.0%，E2E latency 增加约 6.0%--15.3%。结合主动推理期间的落核观测，背景 worker 大多迁移到 E-core，而 dynamic 的 P-core CPUInfer 线程仍持续工作；约 24 token/s 的 decode 与高负载切换到 iGPU 的历史微基准相符，但当前 artifact 没有记录 ratio，仍不能把该路径当作已证实事实。

输出一致性为 21/25（84%）：3 个 mismatch 来自 p1，另一个来自 p2048 repetition 4；其余长 prompt 为 19/20 一致。所有请求均生成 300 tokens，但 mismatch 的输出字符数不同，说明是生成轨迹分叉而非请求提前结束。该风险需继续记录，不能把当前后端描述为 bitwise deterministic。

工程结论：当前 dynamic 在 compute-8 free-affinity 场景中实现了稳定的 decode 优势，但尚未实现长 prompt、300 输出的系统级 E2E 加速。主要障碍是共享 packed INT4 CPU prefill 相对 legacy unpacked VNNI 的约 24% 回退，而不是 compute-8 额外造成的 prefill 退化。继续采集完整负载矩阵之前，先增加 `packed-cpu-fixed`、`igpu-fixed` 启动模式和调度 telemetry：只有比较 dynamic 与同布局 fixed CPU/iGPU，才能判断控制器迁移本身是否正确；随后再决定优先优化 packed prefill kernel、控制器阈值或 phase/objective 设计。

### 2026-07-18：固定端点与请求级调度 telemetry

`perf-log/35b-test-cpu-igpu.sh` 新增两个消融模式：

- `packed-cpu-fixed`：仍使用 `CPU_IGPU_GPTQ_INT4` 和共享 packed 权重，设置 `KT_CPU_IGPU_POLICY=fixed, KT_CPU_IGPU_RATIO=0`；
- `igpu-fixed`：使用完全相同的 method 和权重，设置 fixed ratio 1。

这两个模式与 dynamic 的权重表示、加载路径和内存所有权相同，不创建第二份专家权重。`legacy-vnni-only` 继续使用原有 GPTQ VNNI 后端作为系统基线。实际 preflight 已确认两个 fixed 模式均从 `kt-kernel/build/lib.linux-x86_64-cpython-311/kt_kernel` 加载 `CPUiGPUGPTQInt4_MOE`，并可见 Level Zero GPU；preflight 不加载 35B 权重。

增加 `kt-kernel/python/scheduler_telemetry.py`。仅当启动时显式设置 `SCHEDULER_TELEMETRY_FILE` 时，launcher 才导出 telemetry 环境变量；默认关闭，不影响普通运行。启用后，指定代表层（默认 layer 0）在每次 MoE 完成后使用已有 pybind 方法记录：

- phase、qlen、当前 iGPU ratio 和控制器 CPU load；
- CPU/iGPU ms-per-row 服务时间 EWMA 与样本数；
- switch count 和 high-load epoch；
- wall/monotonic timestamp、PID、layer 和单调 sequence。

writer 使用单次 append JSONL，不修改 C++ kernel，也不复制权重。选择一个代表层是为了把采样开销限制为每个模型 step 一条事件；它不能证明全部层的策略状态，论文表述必须明确这一限制。后续若发现跨层策略分歧，再增加低频全层聚合采样，而不是默认每层每 token 写文件。

`bench_running_server.py` 新增 `--scheduler-telemetry-file`。客户端在每个正式请求开始前记录文件 offset，请求结束后只读取新增事件，因此不会把 server warmup 或前一请求混入当前样本。输出新增：

- `scheduler-telemetry.jsonl`：增加 workload、repetition 和 request index 后的逐事件原始数据；
- `samples.jsonl`：每请求、分 prefill/decode 的 ratio/load mean、min、max、final，事件数、switch first/final/delta 和 high-load fraction；
- `summary.csv`/`report.md`：每 workload 的代表层 prefill/decode ratio 和 CPU load 聚合。

客户端现在默认在每个请求窗口前后读取 `/proc/stat` 和 `/proc/pressure/cpu`，记录全系统 busy/user/nice/system fraction、CPU PSI some/full，以及每个逻辑 CPU 的 busy/user/nice/system 映射。这里 `nice` 可以近似识别 `nice=5` 引擎工作，`user` 包含 `nice=0` 背景任务；逐核映射用于分析 P/E-core 迁移。该观测是整段请求的时间平均，优于单点 `ps`，但仍不能给出 worker 的逐时刻迁移轨迹。

本次只修改 shell/Python，已执行 `python setup.py build_py` 同步到本地 build，无需重新编译 C++ 扩展。聚焦回归结果为 `40 passed`；`ruff` 在当前环境未安装，因此使用 `py_compile`、`bash -n` 和 `git diff --check` 完成静态检查。

下一轮先运行 compute-8 下的 packed CPU fixed smoke，不立即扩展 worker 数。引擎终端：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/packed-cpu-fixed-compute8.jsonl \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh packed-cpu-fixed
```

测试终端：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label packed-cpu-fixed__engine-low__compute8__smoke \
  --workloads 1:300,1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/packed-cpu-fixed-compute8.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/packed-cpu-fixed-engine-low-compute8-smoke-20260718
```

该 smoke 的 ratio 应严格保持 0；若不是 0，则 fixed 配置或 telemetry 关联存在错误，不能继续性能归因。随后以相同方式运行 `igpu-fixed` 和重新运行带 telemetry 的 `dynamic`，三组同布局结果用于计算 dynamic 相对两个端点和逐 workload oracle 的 regret。

### 2026-07-18：packed CPU fixed compute-8 smoke

结果位于 `artifacts/running-server-bench/packed-cpu-fixed-engine-low-compute8-smoke-20260718/`，服务日志为 `artifacts/server-logs/20260718_145413-packed-cpu-fixed.log`，server telemetry 源文件为 `artifacts/server-telemetry/packed-cpu-fixed-compute8.jsonl`。6/6 请求成功，结果如下：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] | E2E ms mean |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 9.54 [8.66, 10.43] | 24.35 [24.25, 24.40] | 106.81 [95.92, 123.38] | 41.07 [40.97, 41.24] | 12385.85 |
| p1024-o300 | 168.70 [166.62, 169.91] | 24.37 [24.32, 24.41] | 6200.82 [6156.09, 6277.87] | 41.04 [40.96, 41.11] | 18470.63 |

代表层共捕获 1806 条事件：configured ratio 和实际 ratio 的唯一值均为 0，policy 唯一值为 fixed，switch count 始终为 0，验证 fixed CPU 配置及请求关联正确。事件由 3 条 prefill 和 1803 条 decode 组成。p1024 每个请求为 1 条 qlen>1 prefill 加 300 条 qlen=1 decode；p1 的初始单 token prompt forward 也满足 qlen=1，因此和输出 step 一起被内核策略归为 301 条 decode，不能把 p1 telemetry 中的全部 decode 事件都解释为生成 token。

fixed policy 不构造 `CPULoadMonitor`，因此 scheduler telemetry 中的 `cpu_load=0` 表示“该模式未启用控制器负载监视”，不是系统空载。请求窗口的独立 `/proc` telemetry 显示：

| Workload | 全 CPU busy | user (`nice=0`) | nice (`nice=5`) | CPU PSI some | CPU PSI full |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 82.11% | 40.34% | 41.45% | 0.106% | 0% |
| p1024-o300 | 83.08% | 40.26% | 41.84% | 0.118% | 0% |

全机约有 8 个逻辑核等价的 normal-priority 背景计算和约 8 个逻辑核等价的 low-priority 引擎工作，但 CPU PSI some 只有约 0.1%。逐核数据中，背景 user 时间主要分布到 E-core，CPUInfer nice 时间主要占据 P-core，说明 compute-8 free-affinity 仍以迁移隔离为主，而不是严重 runqueue 等待。

与此前未启用 telemetry 的 dynamic 运行相比，fixed CPU 的 decode 高约 1.1%--1.3%，E2E 高约 0.7%--1.7%；但两个 dynamic artifact 的 p1024 prefill 相对 fixed 分别高约 0.5%和低约 2.4%，存在跨启动/时段漂移，且采样开销不完全对称。因此当前只能判断 fixed CPU 与 dynamic 处于同一性能区间，不能据此宣称控制器错误迁移。一个更符合现有证据的假设是 dynamic 在 compute-8 下保持 ratio 0；需要带 telemetry 的 dynamic 重跑确认。

相对 legacy VNNI compute-8 full 均值，p1024 packed CPU fixed 的 prefill 为 0.795x，decode 为 1.076x，E2E 为 0.983x。也就是说，同样不使用 iGPU 时，packed 布局本身已经带来约 20.5% prefill 回退和约 7.6% decode 提升；这正式确认此前 dynamic/VNNI 的主要阶段权衡不能归因于调度器。

下一项运行 `igpu-fixed` smoke，并保持背景负载、priority、workload 和 telemetry 参数不变：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/igpu-fixed-compute8.jsonl \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh igpu-fixed
```

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label igpu-fixed__engine-low__compute8__smoke \
  --workloads 1:300,1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/igpu-fixed-compute8.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/igpu-fixed-engine-low-compute8-smoke-20260718
```

该组所有 ratio 应严格为 1。完成后再以相同 telemetry 设置重跑 dynamic，避免拿未观测的旧 dynamic 结果做最终控制器归因。

### 2026-07-18：iGPU fixed compute-8 smoke

结果位于 `artifacts/running-server-bench/igpu-fixed-engine-low-compute8-smoke-20260718/`，服务日志为 `artifacts/server-logs/20260718_150820-igpu-fixed.log`。6/6 请求成功，1806 条代表层事件的 configured/actual ratio 唯一值均为 1，policy 为 fixed，switch count 始终为 0，验证 iGPU fixed 端点正确。结果如下：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] | E2E ms mean |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 6.57 [4.83, 7.51] | 20.18 [19.96, 20.48] | 158.72 [131.92, 206.89] | 49.56 [48.82, 50.48] | 14975.90 |
| p1024-o300 | 89.04 [88.92, 89.18] | 20.52 [20.26, 20.67] | 11747.95 [11729.10, 11763.46] | 48.74 [48.38, 49.36] | 26321.39 |

以 packed CPU fixed/iGPU fixed 定义端点 speedup，大于 1 表示 CPU 更好，配对 bootstrap 结果为：

| Workload | CPU prefill speedup [95% CI] | CPU decode speedup [95% CI] | CPU E2E speedup [95% CI] | 输出 SHA-256 |
|---|---:|---:|---:|---:|
| p1-o300 | 1.453 [1.017, 2.155] | 1.206 [1.191, 1.232] | 1.209 [1.193, 1.239] | 1/3 |
| p1024-o300 | 1.895 [1.868, 1.909] | 1.188 [1.179, 1.203] | 1.425 [1.414, 1.439] | 3/3 |

compute-8 free-affinity 下，packed CPU 在 prefill、decode 和 E2E 三个目标上均严格支配 iGPU；当前条件的逐 workload oracle 明确为 ratio 0。代表层近期服务时间 EWMA 也一致：packed CPU prefill/decode 约为 0.022/0.037 ms per routed row，iGPU 约为 0.063/0.090 ms per routed row。该服务时间只来自 layer 0，不能直接当作全模型阶段延迟，但可用于解释端点排序。

iGPU fixed 的请求窗口 CPU telemetry 为：

| Workload | 全 CPU busy | user (`nice=0`) | nice (`nice=5`) | system | CPU PSI some |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 48.63% | 40.44% | 5.72% | 2.47% | 0.07% |
| p1024-o300 | 51.72% | 39.64% | 7.84% | 4.23% | 3.83% |

与 CPU fixed 的约 41.5%--41.8% nice fraction 相比，iGPU fixed 把大部分 CPUInfer 计算移出 CPU，但推理反而变慢，说明 compute-8 下节省 CPU 资源不是 E2E 最优点。p1024 的 CPU PSI some 上升到约 3.8%，尽管总 busy 只有约 52%；可能与 SYCL host/runtime 线程的局部 affinity 或等待有关，不能用全机 busy 单独解释，后续需要结合逐核数据和动态控制器观测。

下一步重新运行带 telemetry 的 dynamic smoke。引擎终端：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-compute8.jsonl \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

测试终端：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic__engine-low__compute8__telemetry-smoke \
  --workloads 1:300,1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-compute8.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-engine-low-compute8-telemetry-smoke-20260718
```

正确控制器应在本条件下使 prefill/decode ratio 都保持 0，并接近 packed CPU fixed 性能。若选择 ratio 1，则动态策略相对 oracle 存在明确的正 regret（性能损失）；若 ratio 0 但性能仍显著偏离 CPU fixed，则应检查 telemetry I/O、频率/热状态和跨启动漂移，而不是先调负载阈值。

### 2026-07-18：dynamic compute-8 telemetry smoke 分析

本轮计时结果写入 `artifacts/running-server-bench/dynamic-engine-low-compute8-telemetry-smoke-20260718/`，服务端实际为 dynamic：`artifacts/server-logs/20260718_152842-dynamic.log` 明确记录 `Backend: dynamic`、`Method: CPU_IGPU_GPTQ_INT4` 和 `KT_SELECTED_MOE_BACKEND=CPU_IGPU_GPTQ_INT4`。结果如下：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] | E2E ms mean |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 9.97 [9.76, 10.27] | 24.12 [23.97, 24.22] | 100.35 [97.35, 102.50] | 41.45 [41.29, 41.72] | 12494.97 |
| p1024-o300 | 167.54 [164.57, 169.23] | 23.48 [23.11, 23.89] | 6244.18 [6180.83, 6355.79] | 42.59 [41.85, 43.27] | 18978.50 |

但该 artifact 的实验元数据无效：manifest、report 和 sample 内的 `run_label` 仍为上一轮 `igpu-fixed__engine-low__compute8__smoke`，`scheduler_source` 也错误地指向 `igpu-fixed-compute8.jsonl`。因此客户端关联到 0 条事件，而 dynamic 服务实际把 1952 条事件写入 `artifacts/server-telemetry/dynamic-compute8.jsonl`。性能计时与服务日志时间窗口一致，可用于本轮工程诊断；该目录不能直接作为论文原始 artifact，正式结果必须使用正确参数重跑。

使用每个 sample 的 wall-clock 开始时间和 `e2e_ms` 重建窗口后，6 个正式请求与服务端事件一一对应，共 1806 条：p1024 每请求 1 条 prefill 和 300 条 decode，p1 每请求 301 条 qlen=1 事件。所有 1806 条事件的实际 iGPU ratio 都严格为 0，high-load fraction 为 0，正式请求期间没有设备迁移。服务启动和 benchmark warmup 的其余 146 条事件中曾出现 9 条 ratio 1 事件，decode policy 完成两次切换后回到 CPU；这些事件不属于正式计时窗口。

| Workload / phase | 事件数 | iGPU ratio mean/min/max | controller CPU load mean | load max | high-load fraction | switch count |
|---|---:|---:|---:|---:|---:|---:|
| p1 decode | 903 | 0 / 0 / 0 | 0.272% | 8.609% | 0 | 2 -> 2 |
| p1024 prefill | 3 | 0 / 0 / 0 | 0.627% | 1.175% | 0 | 0 -> 0 |
| p1024 decode | 900 | 0 / 0 / 0 | 0.731% | 26.437% | 0 | 2 -> 2 |

独立 `/proc` 请求窗口显示 p1/p1024 全 CPU busy 为 82.54%/83.08%，其中 normal-priority user 为 40.65%/40.38%，low-priority engine nice 为 41.46%/41.62%，CPU PSI some 为 0.259%/0.453%。controller 的低 load 与全机高 busy 并不矛盾：controller 估计的是 CPUInfer 所绑定 P-core 上扣除本引擎后的外部竞争，背景 compute-8 大多运行在 E-core；这再次证明 compute-8 free-affinity 主要形成 P/E 隔离，尚未构成需要 iGPU 接管的 P-core 竞争场景。

以 packed CPU fixed 为本条件 oracle，按 `(workload, repetition)` 配对，吞吐 speedup 定义为 dynamic/CPU，延迟 speedup 定义为 CPU/dynamic，大于 1 表示 dynamic 更好：

| Workload | Prefill speedup [paired 95% CI] | Decode speedup [paired 95% CI] | E2E speedup [paired 95% CI] | 输出 SHA-256 |
|---|---:|---:|---:|---:|
| p1-o300 | 1.045 [0.936, 1.270] | 0.991 [0.988, 0.993] | 0.991 [0.991, 0.992] | 1/3 |
| p1024-o300 | 0.993 [0.988, 0.996] | 0.964 [0.947, 0.982] | 0.973 [0.960, 0.985] | 3/3 |

p1024 的 dynamic decode 比 CPU fixed 低约 3.6%，E2E 高约 2.75%，但这不是错误选择 iGPU 导致的 device-selection regret，因为正式窗口 ratio 始终为 0。两个条件各只有一次独立服务启动，N=3 请求级 bootstrap 没有覆盖跨启动、温度和频率方差，不能把区间解释为完整实验置信度。`CPULoadMonitor::acquire` 使用进程级共享 weak singleton，40 个 MoE 层共用一个采样线程，也排除了每层单独轮询 `/proc` 的假设。剩余候选是 dynamic policy 的轻量原子读/分支、负载监控线程、频率/热状态及跨启动漂移；需要正确元数据的重复启动实验区分，而不是根据这一个 N=3 artifact 调阈值。

为避免再次生成此类伪完整结果，`bench_running_server.py` 现在要求指定的 scheduler telemetry 在 warmup 期间真实增长，并要求每个成功的正式请求至少关联一条事件。旧文件虽然存在但本轮没有新增内容时，benchmark 会以 failed 状态结束。新增陈旧文件回归测试后，`test_running_server_bench.py` 为 12 passed，`py_compile` 和 `git diff --check` 通过。

下一步先保持当前 dynamic 服务和 compute-8 背景不变，用正确的 label、telemetry 路径和新目录重跑一次，以生成可追溯的对照 artifact：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic__engine-low__compute8__telemetry-smoke-rerun \
  --workloads 1:300,1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-compute8.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-engine-low-compute8-telemetry-smoke-rerun-20260718
```

该重跑用于修复证据链，不用于扩大结论。随后应把 free-affinity 背景负载提升到 12、16、20 workers：12 workers 对应填满 E-core 的边界，16 和 20 workers 才预计逐步直接竞争 P-core。先对 16 或 20 workers 做三端点 smoke，确认 iGPU 何时优于 CPU，再扩展完整负载矩阵。

### 2026-07-18：有效的 dynamic compute-8 telemetry 重跑

重跑结果位于 `artifacts/running-server-bench/dynamic-engine-low-compute8-telemetry-smoke-rerun-20260718/`。manifest 的 run label、dynamic telemetry 源路径和输出目录一致，6/6 请求成功，客户端正确关联 1806 条正式请求事件；该 artifact 可用于后续开发比较。结果如下：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] | E2E ms mean |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 10.42 [10.37, 10.48] | 23.71 [23.40, 24.03] | 95.98 [95.45, 96.66] | 42.19 [41.62, 43.31] | 12711.02 |
| p1024-o300 | 167.95 [166.19, 168.84] | 23.78 [23.46, 24.04] | 6228.49 [6195.05, 6293.92] | 42.07 [41.59, 42.62] | 18806.01 |

1806 条事件的 policy 唯一值为 dynamic，configured/actual iGPU ratio 唯一值都为 0。p1 decode、p1024 prefill 和 p1024 decode 的 controller load 均值分别为 0.479%、1.936% 和 0.469%，最大值分别为 16.82%、5.35% 和 12.54%，high-load fraction 全部为 0。decode switch count 在正式窗口内保持 `4 -> 4`；相对上一轮的 2 次累计切换，新增切换发生在本次 15 次 warmup 中，未污染正式请求。系统 `/proc` telemetry 的 p1/p1024 busy 为 82.90%/84.08%，user 为 40.94%/40.99%，nice 为 41.41%/41.86%，CPU PSI some 为 0.57%/0.47%，与 compute-8 的 P/E 隔离判断一致。

相对 packed CPU fixed，配对结果为：

| Workload | Prefill speedup [paired 95% CI] | Decode speedup [paired 95% CI] | E2E speedup [paired 95% CI] | 输出 SHA-256 |
|---|---:|---:|---:|---:|
| p1-o300 | 1.092 [1.004, 1.327] | 0.974 [0.952, 0.990] | 0.974 [0.946, 0.993] | 0/3 |
| p1024-o300 | 0.996 [0.980, 1.013] | 0.976 [0.965, 0.985] | 0.982 [0.970, 0.994] | 2/3 |

因此本次 p1024 的 dynamic/CPU fixed prefill 基本持平，decode 低约 2.44%，E2E latency 高约 1.82%。相对前一轮同一 dynamic 服务的诊断计时，新一轮 p1024 prefill 为 1.002x、decode 为 1.012x、E2E speedup 为 1.009x；这些区间均跨 1，说明两轮 dynamic 自身存在约 1% 级请求/时段波动。两次 dynamic 都在同一服务进程内，而 CPU fixed 来自另一服务启动，当前仍不能把约 2% 差距完全归因于动态控制开销。正式论文实验必须把“服务启动”作为重复层级或至少交替启动后端，不能只对同一启动内的请求做 bootstrap。

compute-8 的控制结论已经复现：CPU fixed 是该条件的设备 oracle，dynamic 正确保持 CPU，iGPU 不应参与。继续增加 compute-8 重复不会验证异构调度收益。下一轮先测试 free-affinity compute-20 最大压力；若 CPU/iGPU 端点排序仍不反转，再检查负载模型和 iGPU kernel，而不是立即采集 12/16 的大矩阵。

操作顺序：先在原背景负载终端按 `Ctrl+C` 停止 compute-8，在原引擎终端按 `Ctrl+C` 停止当前 dynamic 服务；然后启动 compute-20 背景：

```bash
python kt-kernel/bench/cpu_background_load.py \
  --kind compute \
  --workers 20 \
  --affinity free \
  --nice 0
```

第一项运行 packed CPU fixed 端点。引擎终端：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/packed-cpu-fixed-compute20.jsonl \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh packed-cpu-fixed
```

测试终端：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label packed-cpu-fixed__engine-low__compute20__smoke \
  --workloads 1:300,1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/packed-cpu-fixed-compute20.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/packed-cpu-fixed-engine-low-compute20-smoke-20260718
```

compute-20 下 CPU fixed 可能显著变慢，仍保持 900 秒客户端 timeout 和 1200 秒引擎 watchdog。完成该端点后，以同一背景进程依次运行 iGPU fixed 和 dynamic，三组都完成后再判断负载转折点。

### 2026-07-18：packed CPU fixed compute-20 smoke

结果位于 `artifacts/running-server-bench/packed-cpu-fixed-engine-low-compute20-smoke-20260718/`，服务日志为 `artifacts/server-logs/20260718_155107-packed-cpu-fixed.log`。6/6 请求成功，1806 条代表层事件的 policy 均为 fixed，configured/actual ratio 均为 0，switch count 均为 0，排除了 fixed 配置错误。结果如下：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] | E2E ms mean |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 0.62 [0.57, 0.68] | 1.36 [1.35, 1.39] | 1627.06 [1477.14, 1740.31] | 733.28 [719.44, 740.35] | 220877.30 |
| p1024-o300 | 82.73 [79.12, 86.45] | 1.37 [1.35, 1.38] | 12660.20 [12099.66, 13221.18] | 731.22 [725.12, 738.63] | 231296.83 |

请求窗口全 CPU busy 为 99.87%--99.89%，normal-priority user 为 75.67%--75.97%，low-priority engine nice 为 22.78%--23.07%，CPU PSI some 为 40.57%--40.73%。按核类型聚合，P-core busy 约 99.8%，其中 user 约 47.2%--48.4%、nice 约 51.1%--52.2%；E-core busy 约 99.9%，user 约 94.4%--94.6%。这证明 20 个 free-affinity normal-priority worker 已经填满 E-core 并直接竞争 P-core，且引擎存在显著 runnable stall，不再是 compute-8 的迁移隔离场景。

相对 compute-8 的 packed CPU fixed：

| Workload | Prefill throughput retention | Decode throughput retention | TTFT multiplier | TPOT multiplier | E2E multiplier |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 6.47% | 5.60% | 15.23x | 17.86x | 17.83x |
| p1024-o300 | 49.04% | 5.61% | 2.04x | 17.82x | 12.52x |

p1024 prefill 主要是持续计算，获得约一半 P-core 时间后吞吐也约减半；decode 由大量短 MoE kernel、线程池唤醒和层间同步组成，对调度时间片及最慢 worker 更敏感，即使 P-core nice time 仍约 51%，吞吐也只剩约 5.6%。因此不能用平均 CPU time share 线性预测 decode latency。该阶段差异是论文可建模的重要现象：控制目标需要包含 contention-induced synchronization penalty，而不只是 CPU utilization。

下一项保持当前 compute-20 背景进程不变，仅在引擎终端停止 packed CPU fixed，并启动 iGPU fixed：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/igpu-fixed-compute20.jsonl \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh igpu-fixed
```

服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label igpu-fixed__engine-low__compute20__smoke \
  --workloads 1:300,1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/igpu-fixed-compute20.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/igpu-fixed-engine-low-compute20-smoke-20260718
```

预期 iGPU fixed 的 MoE 计算受 CPU 竞争影响较小，但 SYCL host/runtime、attention 和采样仍需 CPU，因此不能假设它完全保持 compute-8 性能。完成后以 CPU/iGPU fixed 的阶段和 E2E 排序定义 compute-20 oracle。

### 2026-07-18：prefill/decode 竞争敏感性与优先级实验原则

compute-20 下 p1024 prefill throughput 从 168.70 降至 82.73 token/s，实际已经下降 50.96%，TTFT 放大 2.04x；它只是没有像 decode 一样下降 94.39%、TPOT 放大 17.82x。两者的差异不能只用平均 CPU utilization 解释。

对长 prefill，可用连续服务模型近似：

\[
L_{\mathrm{prefill}}(s)
= L_{\mathrm{other}} + \frac{W_{\mathrm{cpu}}}{s},
\qquad
D_{\mathrm{prefill}}(s)
= (1-\alpha)+\frac{\alpha}{s},
\]

其中 \(s\) 是 CPUInfer 相对空闲基线获得的有效 CPU 服务率，\(\alpha\) 是基线 prefill latency 中受 CPU 服务率影响的比例。compute-20 请求窗口的 P-core nice time 约为 51%，与 p1024 prefill 约 49% 的吞吐保持率同量级，因此长 prefill 基本表现为持续计算按 CPU 服务时间份额缩放。

decode 则需要加入调度等待和同步尾延迟：

\[
L_{\mathrm{decode}}
\approx \sum_{\ell=1}^{N_{\mathrm{layer}}}
\left[
\max_j\left(\frac{C_{\ell j}}{s_j}+Q_{\ell j}\right)
+ B_\ell
\right],
\]

其中 \(C_{\ell j}\) 是第 \(\ell\) 层第 \(j\) 个 CPU worker 的计算量，\(Q_{\ell j}\) 是被 normal-priority 背景任务抢占后等待调度的时间，\(B_\ell\) 是线程池/层间屏障开销。decode 的单次 kernel 很短，8 个 worker 中任意一个成为 straggler 都会阻塞该层；该效应在多层逐 token 重复后累积。因此平均仍获得约 51% P-core nice time，并不意味着 decode 能保持 51% 吞吐。CPU PSI some 约 40.6% 是这一 runnable stall 的系统级旁证。

继续降低引擎优先级在技术上会扩大异构调度收益。Linux CFS 的典型 task weight 中，nice 0/5/10 分别约为 1024/335/110；在理想的一对一持续竞争下，nice 5 和 nice 10 任务相对一个 nice 0 任务的理论 CPU share 约为 24.7% 和 9.7%。但本机实测 P-core nice time 约 51%，说明 free-affinity 迁移、异构核 capacity、cgroup/autogroup 和阶段性 runnable 状态会使实际份额偏离简单权重公式，所以论文必须报告实测 share/PSI，不能用 nice 值直接代替负载强度。

现场检查引擎 systemd scope 确认 `CPUWeight=33`，对应 cgroup v2 文件为 `cpu.weight=33`、`cpu.weight.nice=5`，引擎进程的实际 nice 也是 5，因此观测偏差不是启动参数未生效。背景 worker 位于 GNOME Terminal 的独立 scope，且采用 free affinity；Linux hybrid-capacity placement 和分层调度组共同决定它在 P/E-core 的实际 runnable 分布。后续优先级消融应同时记录 nominal nice/CPUWeight 和请求窗口的 P-core user/nice share，分析时以后者作为有效 CPU 服务率的观测量。

实验设计上，`ENGINE_PRIORITY=low`（当前 nice 5、CPUWeight 33）继续作为主实验设置，因为它表达“前台编译/仿真优先、后台推理让步”的真实端侧策略。nice 0/5/10 可作为优先级敏感性消融；不能在看到结果后只选择使 dynamic 优势最大的 nice 值作为主设置。核心评价仍是 dynamic 相对同条件 CPU fixed、iGPU fixed 和逐条件 oracle 的 regret，而不是单独最大化 dynamic/CPU-only speedup。当前 compute-20 已经产生 17.8x decode slowdown，足以验证调度价值，无需先进一步降低优先级；下一步仍先完成 iGPU fixed 和 dynamic 三端点闭环。

### 2026-07-18：iGPU fixed compute-20 smoke

结果位于 `artifacts/running-server-bench/igpu-fixed-engine-low-compute20-smoke-20260718/`，服务日志为 `artifacts/server-logs/20260718_162155-igpu-fixed.log`。6/6 请求成功，1806 条代表层事件的 policy 均为 fixed，configured/actual ratio 均为 1，switch count 均为 0。结果如下：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] | E2E ms mean |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 6.73 [2.71, 10.11] | 14.91 [14.28, 15.88] | 201.13 [98.89, 368.63] | 67.23 [62.97, 71.52] | 20304.71 |
| p1024-o300 | 83.60 [74.76, 88.75] | 14.16 [13.19, 15.50] | 12587.11 [11786.05, 13991.78] | 70.93 [64.50, 75.81] | 33795.73 |

相对同条件 packed CPU fixed，iGPU fixed 的配对 speedup 为：

| Workload | Prefill speedup [paired 95% CI] | Decode speedup [paired 95% CI] | E2E speedup [paired 95% CI] | 输出 SHA-256 |
|---|---:|---:|---:|---:|
| p1-o300 | 10.90 [4.01, 17.60] | 10.93 [10.35, 11.75] | 10.88 [10.24, 11.76] | 1/3 |
| p1024-o300 | 1.010 [0.865, 1.122] | 10.36 [9.74, 11.11] | 6.84 [6.42, 7.17] | 3/3 |

p1024 prefill 两个端点在统计上近似持平，不能根据 1.0% 点估计差异宣称 iGPU 更快；decode 和 300-token E2E 则由 iGPU 严格支配。compute-20 的 phase oracle 因而是：prefill 两端近似等价，decode 选择 ratio 1；该 workload 的 E2E oracle 为 iGPU fixed。

iGPU fixed 请求窗口的全 CPU busy 仍为 99.1%--99.6%，但 engine nice fraction 只有 4.0%--6.5%，CPU PSI some 为 13.9%--17.4%，显著低于 CPU fixed 的约 40.6%。相对 compute-8 iGPU fixed，p1024 prefill 保留 93.9%，decode 保留 69.0%，E2E latency 增加约 28.4%。这说明 SYCL host/runtime、attention 和采样仍受 CPU 竞争影响，但 offload 避免了 CPUInfer 8-worker 屏障的 17.8x decode 放大。

下一项保持 compute-20 背景进程不变，仅停止 iGPU fixed 引擎并启动 dynamic：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-compute20.jsonl \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic__engine-low__compute20__smoke \
  --workloads 1:300,1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-compute20.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-engine-low-compute20-smoke-20260718
```

验收标准：正式 decode 事件应主要保持 ratio 1，decode throughput 应接近 14.16 token/s；prefill 选择 0 或 1 都可能具有很小的端点 regret。若 decode 保持 ratio 0，则当前 load signal/0.60 高阈值未捕获同步型竞争；若频繁 0/1 震荡，则需调整 hysteresis/dwell；若 ratio 1 但性能明显低于 iGPU fixed，则需检查切换冷启动或跨层策略不同步。

### 2026-07-18：dynamic compute-20 smoke

结果位于 `artifacts/running-server-bench/dynamic-engine-low-compute20-smoke-20260718/`，服务日志为 `artifacts/server-logs/20260718_163200-dynamic.log`。6/6 请求成功，1806 条代表层正式事件的 policy 均为 dynamic，configured ratio 均为 0，actual ratio 均为 1。prefill/decode high-load fraction 均为 100%；策略在 warmup 中完成一次 CPU->iGPU 切换，正式窗口 switch count 始终为 `1 -> 1`，没有震荡。结果如下：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] | E2E ms mean |
|---|---:|---:|---:|---:|---:|
| p1-o300 | 8.03 [6.32, 10.63] | 14.40 [14.31, 14.50] | 133.81 [94.11, 172.57] | 69.44 [68.99, 70.32] | 20896.92 |
| p1024-o300 | 88.74 [85.96, 90.46] | 14.80 [13.88, 15.27] | 11792.59 [11563.49, 12168.34] | 67.72 [65.47, 72.03] | 32041.11 |

p1 decode、p1024 prefill 和 p1024 decode 的 controller load 均值分别为 81.90%、73.87% 和 81.64%，正式窗口 actual ratio 均为 1。系统 busy 为 98.7%--99.5%，nice fraction 为 4.2%--6.7%，CPU PSI some 为 13.6%--16.8%，与 iGPU fixed 的 host-side 资源特征一致。

dynamic 相对两个 fixed 端点的配对 speedup 如下；吞吐使用 dynamic/fixed，E2E 使用 fixed/dynamic：

| Workload | 对比端点 | Prefill speedup [95% CI] | Decode speedup [95% CI] | E2E speedup [95% CI] | 输出 SHA-256 |
|---|---|---:|---:|---:|---:|
| p1-o300 | CPU fixed | 13.00 [9.17, 18.49] | 10.56 [10.43, 10.72] | 10.57 [10.47, 10.71] | 2/3 |
| p1-o300 | iGPU fixed | 1.193 [0.749, 2.926] | 0.966 [0.912, 1.017] | 0.972 [0.921, 1.028] | 2/3 |
| p1024-o300 | CPU fixed | 1.073 [0.994, 1.143] | 10.82 [10.07, 11.25] | 7.22 [6.79, 7.50] | 2/3 |
| p1024-o300 | iGPU fixed | 1.062 [1.019, 1.150] | 1.045 [0.985, 1.155] | 1.055 [1.001, 1.104] | 2/3 |

compute-20 下控制器成功识别高竞争并选择正确设备，消除了 CPU fixed 的同步型 decode 崩溃；相对 phase/E2E oracle 的 regret 在当前测量精度下可视为 0。dynamic 点估计比 iGPU fixed 快约 4%--6%，但正式阶段两者都执行 ratio 1 的同一 kernel 路径，动态策略不可能凭空产生这部分 kernel speedup。请求级 bootstrap 没有覆盖独立服务启动，故该差异应归于跨启动、频率/热状态和背景落核漂移，不能宣称 dynamic 超越 oracle。这一结果进一步支持后续采用多次交替服务启动或分层 bootstrap。

目前两个边界条件均正确：compute-8 选择 CPU，compute-20 选择 iGPU。下一步寻找切换边界，先做 compute-16；为缩短 CPU fixed 极慢条件的 smoke，只保留论文主要工作负载 `p1024-o300`。先停止当前 dynamic 引擎和 compute-20 背景，再启动 16 workers：

```bash
python kt-kernel/bench/cpu_background_load.py \
  --kind compute \
  --workers 16 \
  --affinity free \
  --nice 0
```

第一项启动 packed CPU fixed：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/packed-cpu-fixed-compute16.jsonl \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh packed-cpu-fixed
```

服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label packed-cpu-fixed__engine-low__compute16__smoke \
  --workloads 1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/packed-cpu-fixed-compute16.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/packed-cpu-fixed-engine-low-compute16-smoke-20260718
```

compute-16 预计填满 12 个 E-core 后有约 4 个背景 worker 竞争 P-core；由于 decode 受最慢 worker 支配，它可能在平均 P-core share 仍较高时就出现非线性下降。完成 CPU/iGPU fixed 和 dynamic 三端点后，再决定是否在 compute-12 与 16 之间增加更细粒度负载点。

### 2026-07-18：packed CPU fixed compute-16 smoke

结果位于 `artifacts/running-server-bench/packed-cpu-fixed-engine-low-compute16-smoke-20260718/`，服务日志为 `artifacts/server-logs/20260718_193446-packed-cpu-fixed.log`。3/3 请求成功，903 条正式事件均为 fixed、ratio 0、switch count 0。结果为：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] | E2E ms mean |
|---|---:|---:|---:|---:|---:|
| p1024-o300 | 136.91 [128.06, 142.47] | 4.49 [3.77, 5.75] | 7657.14 [7341.71, 8168.09] | 230.60 [173.87, 265.16] | 76606.61 |

全 CPU busy 为 99.86%，user/nice fraction 为 62.07%/35.81%，CPU PSI some 为 25.59%。按核类型聚合，P-core busy 99.85%，其中 user 16.87%、nice 82.01%；E-core busy 99.87%，其中 user 92.19%、nice 5.01%。这与 16 个 free-affinity worker 主要填满 12 个 E-core、仅部分时间竞争 P-core 的预期一致。

相对 compute-8 packed CPU fixed，compute-16 的 p1024 prefill/decode throughput retention 分别为 81.15%/18.44%，TTFT/TPOT/E2E multiplier 分别为 1.23x/5.62x/4.15x。引擎仍获得约 82% P-core nice time，prefill throughput 也保持约 81%，但 decode 只保持约 18%，说明少数 P-core 上的间歇性抢占已经通过最慢 worker 和屏障产生强非线性放大。compute-8、16、20 的 CPU fixed decode retention 依次约为 100%、18.4%、5.6%，而 prefill 为 100%、81.2%、49.0%。

下一项保持当前 compute-16 背景进程不变，仅停止 packed CPU fixed 引擎并启动 iGPU fixed：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/igpu-fixed-compute16.jsonl \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh igpu-fixed
```

服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label igpu-fixed__engine-low__compute16__smoke \
  --workloads 1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/igpu-fixed-compute16.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/igpu-fixed-engine-low-compute16-smoke-20260718
```

若 iGPU fixed decode 明显高于 4.49 token/s，则 compute-16 的 decode oracle 已经反转；prefill 仍需单独比较，不能从 decode 推断。

### 2026-07-18：iGPU fixed compute-16 smoke 与 phase-composed oracle

结果位于 `artifacts/running-server-bench/igpu-fixed-engine-low-compute16-smoke-20260718/`，服务日志为 `artifacts/server-logs/20260718_194909-igpu-fixed.log`。3/3 请求成功，903 条正式事件均为 fixed、ratio 1、switch count 0：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] | E2E ms mean |
|---|---:|---:|---:|---:|---:|
| p1024-o300 | 82.92 [82.00, 83.68] | 21.05 [20.69, 21.35] | 12615.15 [12499.77, 12755.85] | 47.51 [46.83, 48.34] | 26822.19 |

相对同条件 CPU fixed，iGPU/CPU prefill throughput 为 0.606x [0.576, 0.653]，decode throughput 为 4.69x [3.67, 5.66]，iGPU E2E speedup 为 2.86x [2.26, 3.26]；3/3 输出 SHA-256 相同。compute-16 的 phase oracle 已明确分裂：prefill 选择 CPU，decode 选择 iGPU。

iGPU fixed 的系统 busy 为 88.79%，user/nice fraction 为 75.16%/9.45%，CPU PSI some 为 6.41%。相对 CPU fixed 的 99.86% busy、35.81% nice 和 25.59% PSI，offload 显著降低了 CPUInfer contention，但其 prefill kernel 本身仍比该负载下 CPU 慢约 39.4%。

用 CPU fixed 的 TTFT 与 iGPU fixed 的 output phase 构造仅用于验收的 phase-composed oracle：

\[
L^*_{\mathrm{phase}}
\approx L^{\mathrm{CPU}}_{\mathrm{TTFT}}
+ L^{\mathrm{iGPU}}_{\mathrm{output}}
= 7.657\ \mathrm{s} + 14.207\ \mathrm{s}
= 21.864\ \mathrm{s}.
\]

该值不是一次实测请求，未包含真实阶段切换的状态与冷启动开销；它是当前两个 fixed 端点可组合出的目标下界。配对 bootstrap 估计为 21.864 s [21.464, 22.334]，相对 CPU fixed 和 iGPU fixed 的潜在 E2E speedup 分别为 3.50x [2.87, 4.04] 和 1.227x [1.194, 1.248]。如果 dynamic 实际达到该区间，就能证明 phase-aware 调度优于任何单一静态端点，而不只是随负载在两个端点间选择。

下一项保持 compute-16 背景不变，仅停止 iGPU fixed 并启动 dynamic：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-compute16.jsonl \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic__engine-low__compute16__smoke \
  --workloads 1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-compute16.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-engine-low-compute16-smoke-20260718
```

理想验收条件为 prefill actual ratio 0、decode actual ratio 1，prefill/decode 分别接近 136.91/21.05 token/s，E2E 接近 21.86 s。若两个阶段都选择 1，控制器虽会接近 iGPU fixed，但会损失 CPU prefill 优势；若两个阶段都选择 0，则 decode 策略未识别同步型竞争；若实现 0/1 分阶段选择但 E2E 明显高于组合 oracle，则差值可归因于切换冷启动、状态迁移或跨层不同步。

### 2026-07-18：dynamic compute-16 smoke 与跨层状态疑点

结果位于 `artifacts/running-server-bench/dynamic-engine-low-compute16-smoke-20260718/`，服务日志为 `artifacts/server-logs/20260718_195958-dynamic.log`。3/3 请求成功，903 条代表层正式事件的 policy 为 dynamic、configured ratio 为 0、actual ratio 均为 1；代表层 prefill/decode load 均值为 75.09%/78.33%，high-load fraction 均为 100%，switch count 始终为 1。结果为：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] | E2E ms mean |
|---|---:|---:|---:|---:|---:|
| p1024-o300 | 103.55 [98.85, 108.60] | 20.11 [18.56, 20.89] | 10116.53 [9631.68, 10581.43] | 49.89 [47.86, 53.89] | 25033.66 |

dynamic 相对 CPU fixed 的 prefill/decode/E2E speedup 分别为 0.756x [0.705, 0.848]、4.48x [3.63, 5.28]、3.06x [2.51, 3.39]；相对 iGPU fixed 分别为 1.249x [1.190, 1.298]、0.955x [0.869, 1.010]、1.071x [0.996, 1.113]。两个端点与 dynamic 的输出 SHA-256 均为 3/3 一致。dynamic 已明显优于 CPU fixed，并在点估计上优于最佳静态 iGPU fixed，但 E2E 仍比 21.864 s phase-composed oracle 高 14.50%。

现有代表层数据与系统性能不完全一致：layer 0 prefill ratio 为 1，若所有层都走相同 iGPU 路径，prefill 应接近 iGPU fixed 的 82.92 token/s；实际 103.55 位于 CPU 136.91 和 iGPU 82.92 之间，且三次请求都保持该中间区间。可能解释包括：

- 各 MoE 层拥有独立 `PolicyState`，layer 0 已选择 iGPU，但其他层仍有一部分选择 CPU，形成跨层混合；
- dynamic/iGPU fixed 的独立服务启动和 free-affinity 落核产生了较大的时段漂移；
- 各层切换/服务时间 EWMA 不一致，造成未观测的层间分歧。

只记录 layer 0 已不足以完成归因。新增诊断选项 `SCHEDULER_TELEMETRY_LAYER=all`：默认值仍为 0，不改变正式 benchmark；all 模式让每个 MoE 层写独立 layer/sequence 事件。已同步 Python build，无需编译 C++；scheduler telemetry、launcher 和 running-server 聚焦测试共 23 passed，`py_compile`、`bash -n` 和 `git diff --check` 通过。

下一项保持 compute-16 背景不变，停止当前 dynamic 后以全层诊断模式重启：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-compute16-all-layers.jsonl \
SCHEDULER_TELEMETRY_LAYER=all \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

服务就绪后只运行一个短输出请求：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic__engine-low__compute16__all-layers-diagnostic \
  --workloads 1024:32 \
  --warmups 15 \
  --repetitions 1 \
  --seed 20260718 \
  --bootstrap-samples 100 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-compute16-all-layers.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-engine-low-compute16-all-layers-diagnostic-20260718
```

该运行只用于统计每层 prefill/decode ratio、load、service EWMA 和 switch count，不能用于性能比较，因为全层 JSONL 写入带来额外 I/O。若 prefill 层同时存在 ratio 0/1，应把 phase policy 从每层独立状态改为模型级共享决策，或显式把跨层分配建模为受控比例；若所有层均为 1，则跨服务启动重复和 P/E 落核漂移成为首要解释。

### 2026-07-18：compute-16 全层策略诊断

诊断 artifact 位于 `artifacts/running-server-bench/dynamic-engine-low-compute16-all-layers-diagnostic-20260718/`，服务日志为 `artifacts/server-logs/20260718_215239-dynamic.log`。`p1024-o32, N=1` 请求成功，捕获 40 层共 1320 条正式事件：40 条 prefill、1280 条 decode。由于全层 JSONL I/O，本轮 99.10/19.39 token/s 不用于性能结论。

正式 prefill 中 39/40 层实际 ratio 为 1，只有 layer 0 为 0，平均 ratio 0.975；decode 中 1269/1280 个层事件 ratio 为 1，11 个 ratio 0 事件分别来自 layer 0、1、14、27、29--35 的首个 decode step，之后所有层均为 1，平均 ratio 0.991。所有层 high-load epoch 均为 true。因此跨层确有短暂分歧，但 current dynamic 在正式请求中实质上仍是近似全 iGPU，无法用 2.5% CPU prefill 层解释相对 iGPU fixed 约 25% 的 prefill 点估计提升；跨服务启动和背景落核漂移仍是主要解释。

全服务生命周期中每层捕获 17 次 prefill。最终局部 service EWMA 判断 11/40 层 CPU 更快、29/40 层 iGPU 更快；prefill switch count 为 2 的也正好有 11 层。但全模型 fixed 端点显示 CPU prefill 比 iGPU 快 1.65x，说明逐层、在不同自干扰状态下采集的 ms-per-routed-row 不是可靠的全模型 TTFT 反事实估计。当前控制器存在两个需要后续改进的问题：

- per-layer policy state 会在切换边界产生暂时不同步；
- local service EWMA 的测量条件随 offload 决策改变，存在 endogenous load/self-interference，局部最优不必等于全模型 phase 最优。

telemetry 现在额外记录 `policy_igpu_ratio`（C++ phase policy state），并保留 `igpu_ratio` 表示本次实际执行路径，用于区分状态切换与实际调用；聚焦测试仍为 23 passed，Python build 已同步，无需编译 C++。

在修改控制器前，先实测强制 phase split，验证 `CPU prefill + iGPU decode` 是否能接近 21.864 s 组合下界。保持 compute-16 背景不变，停止当前全层 dynamic 服务后运行：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/forced-phase-split-compute16.jsonl \
SCHEDULER_TELEMETRY_LAYER=0 \
KT_CPU_IGPU_PREFILL_LOAD_LOW=0.99 \
KT_CPU_IGPU_PREFILL_LOAD_HIGH=1.0 \
KT_CPU_IGPU_DECODE_LOAD_LOW=0.0 \
KT_CPU_IGPU_DECODE_LOAD_HIGH=0.01 \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

该阈值组合只用于消融：prefill load 几乎必然落在 low 以下并固定 CPU，decode 在 warmup 后进入 high epoch 并固定 iGPU。服务就绪后运行正常代表层性能测试：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label forced-phase-split__engine-low__compute16__smoke \
  --workloads 1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/forced-phase-split-compute16.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/forced-phase-split-engine-low-compute16-smoke-20260718
```

必须先验证代表层 prefill/decode actual ratio 为 0/1；若成立，实测 E2E 与 21.864 s 的差值就是 phase switching、背景重排和非可组合系统状态造成的真实 gap。该 forced policy 不是最终算法，也不能作为 dynamic 方法结果，只作为 phase oracle 消融。

### 2026-07-18：forced phase split compute-16 实测

结果位于 `artifacts/running-server-bench/forced-phase-split-engine-low-compute16-smoke-20260718/`，服务日志为 `artifacts/server-logs/20260718_220313-dynamic.log`。3/3 请求成功。代表层 prefill actual/policy ratio 均为 0；decode policy ratio 900/900 为 1，实际路径 897/900 为 1，每个请求首个 decode step 的 actual ratio 为 0，随后稳定为 1。结果如下：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] | E2E ms mean |
|---|---:|---:|---:|---:|---:|
| p1024-o300 | 144.33 [138.95, 151.34] | 20.01 [18.24, 20.91] | 7256.34 [6911.49, 7528.06] | 50.18 [47.83, 54.84] | 22260.66 |

prefill 与 CPU fixed 处于同一区间，decode 与 iGPU fixed 处于同一区间。forced split 相对 CPU fixed、iGPU fixed 和默认 dynamic 的 E2E speedup 分别为 3.44x [2.84, 3.83]、1.205x [1.112, 1.257] 和 1.125x [1.116, 1.130]；三组对比的输出 SHA-256 都是 3/3 一致。实测 E2E 只比 21.864 s phase-composed oracle 高 1.81%，说明 phase 切换和非可组合系统状态的实际 gap 很小，`CPU prefill + iGPU decode` 确实优于两个单端点。

系统 busy/user/nice/PSI some 分别为 91.53%/72.80%/16.34%/8.78%，介于 CPU fixed 与 iGPU fixed 之间，符合两个阶段使用不同资源的预期。这是当前最直接的 phase-aware 系统收益证据，但 forced 阈值是 oracle 消融，不能冒充最终自适应方法。

下一步测试 dynamic v2 candidate：prefill 暂时固定为 CPU；decode 不再用 0.0/0.01 强制切换，而使用 0.10/0.20 的负载迟滞自动识别同步竞争。保持 compute-16 背景不变，停止 forced 服务后运行：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v2-candidate-compute16.jsonl \
SCHEDULER_TELEMETRY_LAYER=0 \
KT_CPU_IGPU_PREFILL_LOAD_LOW=0.99 \
KT_CPU_IGPU_PREFILL_LOAD_HIGH=1.0 \
KT_CPU_IGPU_DECODE_LOAD_LOW=0.10 \
KT_CPU_IGPU_DECODE_LOAD_HIGH=0.20 \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v2-candidate__engine-low__compute16__smoke \
  --workloads 1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v2-candidate-compute16.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v2-candidate-engine-low-compute16-smoke-20260718
```

验收条件仍为 prefill ratio 0、decode ratio 1、E2E 接近 forced split。若 compute-16 成功，还不能立即修改默认值；必须用相同 v2 参数回归 compute-8，验证低竞争下 decode 最终回到 ratio 0，并比较其切换次数和低负载开销。

### 2026-07-18：dynamic v2 candidate compute-16

结果位于 `artifacts/running-server-bench/dynamic-v2-candidate-engine-low-compute16-smoke-20260718/`，服务日志为 `artifacts/server-logs/20260718_221329-dynamic.log`。3/3 请求成功。代表层 prefill actual/policy ratio 均为 0；decode policy ratio 900/900 为 1，actual ratio 897/900 为 1，每个请求首 step 后稳定在 iGPU。0.10/0.20 decode 迟滞在 compute-16 下能够自动进入 high-load epoch，switch count 在正式窗口稳定为 1。结果如下：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] | E2E ms mean |
|---|---:|---:|---:|---:|---:|
| p1024-o300 | 147.76 [145.12, 149.80] | 20.97 [20.68, 21.19] | 7080.16 [6982.46, 7207.91] | 47.70 [47.19, 48.34] | 21341.97 |

v2 相对 CPU fixed、iGPU fixed、默认 dynamic 和 forced split 的 E2E speedup 分别为 3.59x [2.84, 4.05]、1.257x [1.240, 1.270]、1.173x [1.132, 1.211] 和 1.043x [1.002, 1.116]；所有比较的输出 SHA-256 均为 3/3 一致。v2 相对 CPU fixed 的 prefill/decode speedup 为 1.079x/4.67x，相对 iGPU fixed为 1.782x/0.996x，说明两个阶段分别落入目标端点的性能区间。

v2 E2E 点估计比先前构造的 21.864 s phase oracle 低 2.4%，但两者来自不同服务启动，oracle 也只是跨端点组合估计，因此不能宣称突破下界；正确结论是 v2、forced split 和 phase-composed oracle 已进入同一性能区间。系统 busy/nice/PSI some 为 91.63%/16.54%/9.18%，与 forced split 一致。

下一步做必要的 compute-8 回归。停止当前 v2 服务和 compute-16 背景，启动 8 workers：

```bash
python kt-kernel/bench/cpu_background_load.py \
  --kind compute \
  --workers 8 \
  --affinity free \
  --nice 0
```

然后用完全相同的 v2 参数启动引擎：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v2-candidate-compute8-regression.jsonl \
SCHEDULER_TELEMETRY_LAYER=0 \
KT_CPU_IGPU_PREFILL_LOAD_LOW=0.99 \
KT_CPU_IGPU_PREFILL_LOAD_HIGH=1.0 \
KT_CPU_IGPU_DECODE_LOAD_LOW=0.10 \
KT_CPU_IGPU_DECODE_LOAD_HIGH=0.20 \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v2-candidate__engine-low__compute8__regression \
  --workloads 1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v2-candidate-compute8-regression.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v2-candidate-engine-low-compute8-regression-20260718
```

低负载验收条件是 prefill/decode 正式 ratio 都为 0，decode 接近 packed CPU fixed 的 24.37 token/s。允许 warmup 中发生探索切换，但正式窗口不得持续选择 iGPU 或频繁震荡；通过后再把 v2 参数固化为新默认并补 compute-12/20 回归。

### 2026-07-18：dynamic v2 compute-8 回归与默认值固化

结果位于 `artifacts/running-server-bench/dynamic-v2-candidate-engine-low-compute8-regression-20260718/`。3/3 请求成功，903 条正式事件的 prefill/decode actual ratio 和 `policy_igpu_ratio` 全部为 0。decode switch count 在正式窗口恒为 2，表示 warmup 中完成一次 CPU->iGPU 探索和一次 iGPU->CPU 返回；正式请求没有继续切换或震荡。结果如下：

| Workload | Prefill token/s mean [95% CI] | Decode token/s mean [95% CI] | TTFT ms mean [95% CI] | TPOT ms mean [95% CI] | E2E ms mean |
|---|---:|---:|---:|---:|---:|
| p1024-o300 | 168.89 [168.32, 169.31] | 24.34 [24.30, 24.39] | 6193.42 [6177.94, 6214.43] | 41.09 [41.01, 41.15] | 18478.97 |

controller prefill/decode load 均值只有 0.021%/0.070%，decode max 1.74%，high-load fraction 为 0。相对 packed CPU fixed，v2 的 prefill/decode/E2E speedup 为 1.001x [0.993, 1.016]、0.999x [0.997, 1.001]、1.000x [0.997, 1.003]，3/3 输出 SHA-256 一致。v2 在 compute-8 下与 CPU oracle 统计上等价，较低的 decode 阈值没有造成正式阶段误 offload。

基于 compute-8 和 compute-16 两端验证，将 v2 固化为当前实验默认：

- decode load hysteresis：0.10/0.20，替代 0.20/0.60；
- prefill CPU-biased hysteresis：0.99/1.0，替代局部 EWMA 容易误判的 0.05/0.25；
- 环境变量覆盖接口保持不变，后续仍可做阈值敏感性实验。

默认值已同步到 `GeneralMOEConfig` C++ 源码、`python/utils/amx.py` 和 `perf-log/35b-test-cpu-igpu.sh`，相应默认值和 launcher 测试已更新。43 个 benchmark/telemetry/launcher 聚焦测试通过；加载 oneAPI 环境后 2 个 scheduler config per-commit 测试通过；`py_compile`、`bash -n` 和 `git diff --check` 通过。Python build 已同步，启动脚本会显式导出 v2 值，因此后续服务重启无需手工写四个阈值。C++ header 的默认值将在下一次扩展编译时进入二进制；当前 Python/SGLang 路径会在构造后显式赋值，不影响后续实验。

下一轮补切换边界和高负载回归：先测 compute-12，确认填满 E-core 的边界仍选择 CPU；再测 compute-20，确认 v2 保持 CPU prefill/iGPU decode。两项均先用 p1024-only smoke，完成后再决定正式负载网格和独立服务启动重复次数。

compute-12 操作顺序：停止当前 v2 服务和 compute-8 背景，启动 12 workers：

```bash
python kt-kernel/bench/cpu_background_load.py \
  --kind compute \
  --workers 12 \
  --affinity free \
  --nice 0
```

使用新默认启动 dynamic，不再传阈值覆盖：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v2-default-compute12.jsonl \
SCHEDULER_TELEMETRY_LAYER=0 \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v2-default__engine-low__compute12__smoke \
  --workloads 1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v2-default-compute12.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v2-default-engine-low-compute12-smoke-20260718
```

compute-12 验收条件是正式 prefill/decode ratio 均为 0、无震荡，性能接近 compute-8 CPU 端点。若 decode 已切到 iGPU，则需要补 CPU/iGPU fixed compute-12 端点，判断是正确提前切换还是 0.10/0.20 阈值过敏。

### 2026-07-18：compute-12 双稳态与 v2 边界失败

结果位于 `artifacts/running-server-bench/dynamic-v2-default-engine-low-compute12-smoke-20260718/`，服务日志为 `artifacts/server-logs/20260718_223122-dynamic.log`。3/3 请求均完成，但 decode 呈现两个离散模态，不能用均值 10.19 token/s 表示稳定性能：

| Request / repetition | Prefill token/s | Decode token/s | E2E ms | Decode actual ratio | Load mean | Switch |
|---|---:|---:|---:|---:|---:|---:|
| request 0 / r2 | 155.25 | 5.07 | 65661.14 | 0.000 | 0.068 | 2 -> 2 |
| request 1 / r0 | 147.29 | 19.93 | 22104.70 | 0.987 | 0.785 | 2 -> 3 |
| request 2 / r1 | 148.06 | 5.56 | 60796.08 | 0.000 | 0.064 | 3 -> 4 |

prefill 三次都稳定 ratio 0。request 0 走 CPU decode 时，controller load 仅约 6.8%，始终低于 0.20 high threshold，decode 只有 5.07 token/s。request 1 的第 4 个 decode step 出现 32.6% load spike，策略切到 iGPU 后 load 升至约 80%，并以 19.93 token/s 保持。request 2 首 step 继承 iGPU policy state，但实际执行路径为 CPU；随后 load 降到 9.94% 并触发 iGPU->CPU，余下请求保持 CPU，decode 退回 5.56 token/s。

逐核数据也显示两个平衡点：CPU decode 请求的 P-core user/nice 约为 12%/87.5%，iGPU decode 请求约为 62.0%/36.5%。free-affinity 背景 worker 会随引擎设备选择重新落核。当前监控量近似为：

\[
z_t = \max\left(
B^{P}_{\mathrm{busy},t} - U_{\mathrm{process},t}(a_t),
\mathrm{PSI}_t
\right),
\]

其中动作 \(a_t\) 会改变本进程 CPU ticks 和背景任务落核，所以观测 \(z_t\) 不是动作无关的外生负载。CPU 动作使 process ticks 较高、扣减后的 load 偏低，从而继续选择 CPU；iGPU 动作使 process ticks 降低、观测 load 升高，从而继续选择 iGPU，形成 action-dependent bistability。这解释了为什么继续微调单一 load threshold 不能保证稳定最优。

v2 compute-12 边界回归失败，因此 0.10/0.20 只能保留为实验 candidate，不能视为最终默认策略。下一版 decode 控制器应利用已经采集的 CPU/iGPU service cost 做在线两臂比较，并把 load 主要用于触发重采样，而不是直接决定设备；同时需要处理 inactive arm 样本陈旧和周期探索。

修改控制器前先补齐 compute-12 fixed endpoints。保持 12-worker 背景不变，停止 dynamic 后启动 packed CPU fixed：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/packed-cpu-fixed-compute12.jsonl \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh packed-cpu-fixed
```

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label packed-cpu-fixed__engine-low__compute12__smoke \
  --workloads 1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/packed-cpu-fixed-compute12.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/packed-cpu-fixed-engine-low-compute12-smoke-20260718
```

完成 CPU fixed 后保持背景进程不变，再运行 iGPU fixed。两个端点用于确认 compute-12 的真实 phase/E2E oracle，并为 service-cost controller 选择校准样本数和切换 margin。

### 2026-07-18：compute-12 packed CPU fixed 端点

结果位于 `artifacts/running-server-bench/packed-cpu-fixed-engine-low-compute12-smoke-20260718/`。3/3 请求成功，scheduler telemetry 确认 prefill/decode 的实际 iGPU ratio 均为 0，且无切换：

| Workload | N | Prefill token/s | Decode token/s | TTFT ms | TPOT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o300 | 3 | 151.72 [150.16, 153.47] | 4.85 [4.51, 5.51] | 6895.04 [6815.45, 6966.06] | 208.21 [181.33, 221.80] | 69148.57 |

prefill 很稳定，并与 dynamic v2 的 150.20 token/s 接近，说明 dynamic v2 在 compute-12 三次 prefill 都正确保持 CPU 路径。固定 CPU decode 的 4.51、4.51、5.51 token/s 与 dynamic 的两个 CPU 慢请求 5.07、5.56 token/s 属于同一性能区间；而 dynamic 切至 iGPU 的请求达到 19.93 token/s，因此先前的两个离散模态确实对应不同设备路径，不是普通请求噪声。

该端点内部仍有约 22% 的 decode 极差，表明 free-affinity 背景任务与引擎竞争会带来操作系统调度波动；当前 N=3 只用于控制器诊断，最终论文统计仍需增加重复次数和独立服务启动次数。请求窗口内系统 CPU busy 均值为 99.70%，normal-priority user 为 59.33%，低优先级引擎 nice 为 38.39%，CPU PSI some 为 6.01%。这组数据同时说明，整体 CPU busy 接近 100% 并不能直接给出 CPU MoE 的可用服务能力。

保持当前 12-worker 背景进程不变，停止 packed CPU fixed 服务后启动 iGPU fixed：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/igpu-fixed-compute12.jsonl \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh igpu-fixed
```

服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label igpu-fixed__engine-low__compute12__smoke \
  --workloads 1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/igpu-fixed-compute12.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/igpu-fixed-engine-low-compute12-smoke-20260718
```

### 2026-07-18：compute-12 iGPU fixed 端点与 phase reference

结果位于 `artifacts/running-server-bench/igpu-fixed-engine-low-compute12-smoke-20260718/`。3/3 请求成功，scheduler telemetry 确认 prefill/decode 的实际 iGPU ratio 均为 1：

| Workload | N | Prefill token/s | Decode token/s | TTFT ms | TPOT ms | Output phase ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1024-o300 | 3 | 86.01 [82.12, 91.83] | 18.42 [18.36, 18.53] | 12189.20 [11390.46, 12737.14] | 54.29 [53.97, 54.46] | 16231.66 | 28420.94 |

compute-12 的两个静态端点具有相反的 phase 优势：

| Phase | CPU fixed | iGPU fixed | 较优设备与倍率 |
|---|---:|---:|---:|
| Prefill | 151.72 token/s | 86.01 token/s | CPU，1.764x |
| Decode | 4.85 token/s | 18.42 token/s | iGPU，3.801x |

用 CPU fixed 的平均 TTFT 与 iGPU fixed 的平均 output phase 组成实验参考：

\[
T_{\mathrm{phase\mbox{-}ref}}
= 6895.04 + 16231.66
= 23126.70\ \mathrm{ms}.
\]

该 phase-composed reference 相对 CPU fixed E2E 为 2.990x，相对 iGPU fixed E2E 为 1.229x；换言之，正确的 CPU-prefill/iGPU-decode 组合预计比最佳静态端点再减少约 18.6% E2E。dynamic v2 唯一的快模态请求为 22104.70 ms，比拼接参考还低约 4.4%，原因是它本次 iGPU output phase 为 15002.85 ms，快于独立 iGPU fixed 均值。由于 reference 来自不同服务启动和不同请求的均值，它是 phase 组合参考而不是严格不可超越的 oracle。

iGPU fixed decode 的三次结果仅为 18.36、18.37、18.53 token/s，显著比 CPU fixed 稳定。请求窗口的系统 CPU busy 均值从 CPU fixed 的 99.70% 降至 72.19%，CPU PSI some 从 6.01% 降至 1.93%。这说明把专家 decode 移至 iGPU 不仅提高推理速度，也释放了 CPU 调度容量；后续可将 CPU contention reduction 作为论文的次级系统指标。

compute-12 fixed endpoints 证明 phase-aware 调度目标成立，但 v2 的 load-only decode 决策不能稳定达到该目标。v3 应保留 CPU-biased prefill，并把 decode 改为 CPU/iGPU 服务代价的显式校准、比较和重采样。实现时需要避免直接使用高 alpha 的单层 EWMA：固定端点的 layer-0 CPU `ms/row` 在 compute-12 正式窗口内跨越约 0.04--4.42，而 iGPU 约为 0.10--0.15；CPU 的阻塞尖峰是真实成本，但短窗口会因是否命中尖峰而误判。

### 2026-07-18：v3 跨层 service-cost decode controller

v3 已实现于 `kt-kernel/operators/cpu_igpu_service_scheduler.hpp`，并接入 `kt-kernel/operators/sycl/gptq_int4_cpu_igpu-moe.hpp`。它只替换 dynamic decode 决策；fixed 路径和现有 CPU-biased prefill 保持不变，共享 packed INT4 权重布局也没有变化，因此不会增加第二份专家权重。

旧控制器每层独立维护 EWMA，单层是否恰好遭遇一次抢占就可能改变该层的设备选择。v3 让同一 WorkerPool 下的 MoE 层共享 decode 决策，并把一个 decode step 内各层的设备时间聚合为一个 phase-level 服务代价样本：

\[
c_{d,t}=
\frac{\sum_{l\in\mathcal{L}} T_{l,d,t}}
     {\sum_{l\in\mathcal{L}} R_{l,d,t}},
\qquad d\in\{C,G\},
\]

其中 $T_{l,d,t}$ 是第 $t$ 个 decode step 中第 $l$ 层在设备 $d$ 上的实测时间，$R_{l,d,t}$ 是对应的有效 expert rows。控制器对每个 active arm 更新：

\[
\hat c_{d,t}=(1-\alpha)\hat c_{d,t-1}+\alpha c_{d,t}.
\]

完成 CPU/iGPU 各 $K$ 个初始样本后，带相对 margin $m$ 的决策为：

\[
a_{t+1}=
\begin{cases}
G, & (1+m)\hat c_{G,t}<\hat c_{C,t},\\
C, & (1+m)\hat c_{C,t}<\hat c_{G,t},\\
a_t, & \text{otherwise}.
\end{cases}
\]

这使 load 不再直接决定设备。CPU 为 active arm 时，其服务代价恶化会直接与已校准的 iGPU 代价比较；iGPU 为 active arm 时，仅在 CPU 样本超过 inactive horizon，或在同一 iGPU action 下观测到 load 相对校准基线明显下降时，才重新采集 CPU 样本。这样 load 的角色是 change detector，而不是 action-dependent threshold policy。

首轮默认参数为：

| 参数 | 环境变量 | 默认值 |
|---|---|---:|
| cost EWMA | `KT_CPU_IGPU_COST_EWMA_ALPHA` | 0.20 |
| switch margin | `KT_CPU_IGPU_DECODE_SWITCH_MARGIN` | 0.10 |
| 每端校准样本 | `KT_CPU_IGPU_DECODE_CALIBRATION_SAMPLES` | 4 |
| minimum dwell | `KT_CPU_IGPU_DECODE_MIN_DWELL` | 4 |
| phase-boundary load grace | `KT_CPU_IGPU_DECODE_LOAD_REPROBE_GRACE` | 16 steps |
| inactive horizon | `KT_CPU_IGPU_DECODE_REPROBE_INTERVAL` | 1024 steps |
| same-arm load drop | `KT_CPU_IGPU_DECODE_LOAD_REPROBE_DELTA` | 0.15 |

telemetry 新增 `exploration`，benchmark 会生成 `scheduler_{phase}_exploration_fraction`。`high_load_epoch` 继续保留用于兼容 v1/v2 artifact，但 v3 decode 不再使用它。正式窗口期望 exploration fraction 为 0；若校准延伸进正式请求，必须把对应性能标为含探索开销，不能和稳态端点混用。

当前实现按 layer leader 边界聚合一次串行 decode round，针对本研究 `--max-running-requests 1` 的单请求场景。未来若扩展并发请求，必须为聚合器增加 request/sequence key，避免不同请求的 layer duration 被合并；在此之前不能把 v3 结论外推到并发 serving。

实现验证：独立 C++ 状态机、launcher、telemetry、running-server benchmark 和 E2E benchmark 共 44 个测试通过；scheduler Python 配置 3 个测试通过；SYCL 扩展增量编译和真实 pybind 字段导入通过；launcher `PREFLIGHT_ONLY=1` 确认从本地 build 目录加载 `CPUiGPUGPTQInt4_MOE`。编译仍报告项目既有的 abstract interface non-virtual destructor 警告，本次未新增编译错误。build 目录的扩展和 Python 文件已经同步，下一轮无需再次运行完整 `./install.sh`。

保持 compute-12 的 12-worker 背景任务不变，停止当前 iGPU fixed 服务后启动 v3 dynamic：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v3-service-cost-compute12.jsonl \
SCHEDULER_TELEMETRY_LAYER=0 \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

启动输出必须包含 `decode_policy=service-cost`。服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v3-service-cost__engine-low__compute12__smoke \
  --workloads 1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260718 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v3-service-cost-compute12.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v3-service-cost-engine-low-compute12-smoke-20260718
```

compute-12 smoke 验收条件：prefill ratio 为 0，decode ratio 为 1，正式 exploration fraction 为 0，switch count 在正式三次请求间不增长；Prefill 接近 151.72 token/s，Decode 接近 18.42--19.93 token/s，E2E 接近 23.13 s 的 phase-composed reference。通过后依次回归 compute-8、16、20。

### 2026-07-19：v3 compute-12 性能通过，但边界重采样未通过

结果位于 `artifacts/running-server-bench/dynamic-v3-service-cost-engine-low-compute12-smoke-20260718/`。3/3 请求均稳定处于快模态，没有再次出现 v2 的 60 s 以上慢请求：

| Workload | N | Prefill token/s | Decode token/s | TTFT ms | TPOT ms | Output phase ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1024-o300 | 3 | 145.32 [139.80, 150.44] | 19.71 [19.12, 20.23] | 7204.50 [6952.88, 7482.16] | 50.76 [49.42, 52.30] | 15176.92 | 22381.48 |

E2E 相对 packed CPU fixed 为 3.090x，相对 iGPU fixed 为 1.270x，并比 23126.70 ms 的 phase-composed reference 低 3.22%。Prefill 比 CPU fixed 均值低约 4.2%，但 decode 比 iGPU fixed 均值高约 7.0%，最终平均 E2E 与 v2 唯一快请求的 22104.70 ms 只差约 1.25%，同时消除了请求级双稳态。从性能和 phase 选择方向看，跨层 service-cost controller 有效。

不过正式 telemetry 未满足稳定性验收：

| Request | Prefill ratio | Decode ratio | Decode exploration | Switch delta |
|---|---:|---:|---:|---:|
| 0 | 0.0000 | 0.9933 | 0.0000 | 1 |
| 1 | 0.0000 | 0.9833 | 0.0133 | 2 |
| 2 | 0.0000 | 0.9833 | 0.0133 | 2 |

后两次请求各包含 4 个正式 CPU probe steps，且 switch count 持续增加。最终跨层 iGPU cost 稳定在约 0.0856 ms/row，而重新采集的 CPU cost 为 0.343--0.791 ms/row，所以这些 probe 不改变最优选择，只增加了不必要的尾延迟。首个请求开头也有 2 个非 exploration CPU steps，说明 warmup 期间已经因同一问题反复切换；正式首事件的 switch count 已达到 12。

触发序列表明，CPU prefill 与 iGPU decode 的 phase transition 会使 load monitor 在 decode 开头暂时落至约 0.05--0.06，随后用约 10--16 个 step 恢复到约 0.8。旧 v3 在恢复完成前便将该瞬态解释为背景负载下降，触发 CPU 重采样。这不是 service-cost 比较错误，而是 change detector 没有建模 phase-boundary transient。

修正已加入：prefill leader 通知共享 decode controller 结束上一轮聚合，并设置 16-step load-reprobe grace。grace 期间继续更新 CPU/iGPU service cost，但禁止 load-drop 触发；inactive horizon 仍然有效。若低 load 只来自 phase transition，它会在 grace 内恢复而不探索；若背景负载确实持续下降，grace 结束后仍满足 delta 条件，CPU 重采样照常发生。对应默认参数为 `KT_CPU_IGPU_DECODE_LOAD_REPROBE_GRACE=16`。

同时修复 `bench_running_server.py` 的指标列表遗漏：下一次 summary/report 会正确汇总 prefill/decode exploration fraction，而不是在 Markdown 表中显示 `NA`。合成测试覆盖瞬态恢复和持续 load drop，两种序列均通过；修正后的 SYCL 扩展已增量编译并同步至 build 目录，无需再次执行完整安装。

保持 12-worker 背景任务不变，停止当前旧 v3 服务并启动带 phase grace 的 v3：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v3-grace-compute12.jsonl \
SCHEDULER_TELEMETRY_LAYER=0 \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

启动输出必须包含 `load_grace=16`。服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v3-grace__engine-low__compute12__smoke \
  --workloads 1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v3-grace-compute12.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v3-grace-engine-low-compute12-smoke-20260719
```

复测验收条件不变，但此次要求更严格：三次正式请求的 decode ratio 必须均为 1.0000，decode exploration fraction 均为 0，switch delta 均为 0，且首尾 switch count 相同。满足后才进入 compute-8/16/20 回归。

### 2026-07-19：v3 phase grace compute-12 稳态通过

结果位于 `artifacts/running-server-bench/dynamic-v3-grace-engine-low-compute12-smoke-20260719/`。3/3 请求成功：

| Workload | N | Prefill token/s | Decode token/s | TTFT ms | TPOT ms | Output phase ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1024-o300 | 3 | 150.29 [147.82, 151.66] | 20.93 [20.67, 21.10] | 6960.92 [6901.30, 7020.53] | 47.80 [47.36, 48.56] | 14290.85 | 21251.83 |

E2E 相对 packed CPU fixed 为 3.254x，相对 iGPU fixed 为 1.337x，相对 phase-composed reference 为 1.088x（latency 降低 8.11%），相对无 phase grace 的 v3 初版为 1.053x（latency 降低 5.05%）。Prefill 已回到 CPU fixed 的同一性能区间，Decode 也稳定处于 iGPU 快端点区间。这是目前 compute-12 的最佳稳定结果。

控制状态满足正式窗口稳定性要求：三次 prefill ratio 均为 0，decode exploration fraction 均为 0，decode switch delta 均为 0，switch count 全程保持 3。共享 CPU cost 保持 0.6303 ms/row，CPU samples 全程保持 8；iGPU cost 从 0.0921 收敛到 0.0838 ms/row，iGPU samples 每个请求增加约 300。由此可知正式 decode 没有执行 CPU probe，phase-boundary grace 成功消除了跨请求误触发。

旧 telemetry 的瞬时 `igpu_ratio` 均值仍显示 0.9967，因为每个请求首个 qlen=1 事件读取到该层异步 prefill 留下的 `current_igpu_ratio_=0`，而同一事件的共享 policy ratio 已为 1。CPU samples 不增加、iGPU samples 连续增加也证明这不是实际 CPU decode。该字段此前被称为 actual ratio 不够严谨；异步 CPUInfer 中，forward snapshot 可能在 telemetry sync 时被另一个已排队任务覆盖。

为论文测量修正观测定义：C++ 现按 prefill/decode phase 分别维护累计 execution calls $N_p$ 与累计 ratio units $S_p$。相邻 telemetry 事件的实际执行比例定义为：

\[
\rho^{\mathrm{exec}}_{p,t}
=\frac{\Delta S_{p,t}}
       {10^6\,\Delta N_{p,t}}.
\]

新的 `igpu_ratio` 使用该差分值；旧瞬时值保留为 `igpu_ratio_snapshot`，并新增 `execution_calls_delta`。phase 分离可避免迟到的 prefill task 污染 decode 比例。修正后的扩展和 Python build 已同步，telemetry 单元测试与真实 pybind binding 验证通过。该改动只影响观测，不改变 v3 设备决策和本轮性能结论。

下一步回归 compute-8。先停止当前 dynamic 服务和 12-worker 背景，再启动 8 workers：

```bash
python kt-kernel/bench/cpu_background_load.py \
  --kind compute \
  --workers 8 \
  --affinity free \
  --nice 0
```

启动 v3 dynamic：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v3-grace-compute8.jsonl \
SCHEDULER_TELEMETRY_LAYER=0 \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v3-grace__engine-low__compute8__regression \
  --workloads 1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v3-grace-compute8.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v3-grace-engine-low-compute8-regression-20260719
```

compute-8 验收条件：prefill/decode execution ratio 均为 0，正式 exploration fraction 和 switch delta 均为 0；跨层 CPU cost 应低于 iGPU cost，Prefill/Decode 接近 packed CPU fixed 的 168.70/24.37 token/s，E2E 接近 18.47 s。通过后再回归 compute-16 和 compute-20。

### 2026-07-19：v3 compute-8 回归通过

结果位于 `artifacts/running-server-bench/dynamic-v3-grace-engine-low-compute8-regression-20260719/`：

| Workload | N | Prefill token/s | Decode token/s | TTFT ms | TPOT ms | Output phase ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1024-o300 | 3 | 169.38 [168.50, 169.90] | 24.53 [24.51, 24.55] | 6175.50 [6158.45, 6192.55] | 40.76 [40.72, 40.79] | 12188.18 | 18363.74 |

相对 packed CPU fixed，Prefill/Decode/E2E speedup 分别为 1.004x、1.007x、1.006x，属于同一性能区间。系统 CPU busy/user/nice/PSI some 分别为 83.11%/40.21%/41.86%/0.06%，再次复现 compute-8 下背景任务主要使用 E-core、引擎主要使用 P-core 的低竞争状态。

新版 execution telemetry 完整通过：三次请求的 prefill/decode ratio、exploration fraction 和 switch delta 均为 0；每个请求恰好记录 300 个 decode execution calls，单事件 delta 恒为 1。switch count 正式窗口恒为 2，对应 warmup 中 CPU->iGPU 校准及 iGPU->CPU 返回。CPU cost 稳定为约 0.0365 ms/row，明显低于冷校准 iGPU cost 0.2580 ms/row；CPU samples 每请求增加 300，iGPU samples 固定为 4。v3 在低负载下正确选择 CPU，且正式阶段没有探索开销。

下一步回归 compute-16。停止当前 dynamic 服务和 8-worker 背景，启动 16 workers：

```bash
python kt-kernel/bench/cpu_background_load.py \
  --kind compute \
  --workers 16 \
  --affinity free \
  --nice 0
```

启动 v3 dynamic：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v3-grace-compute16.jsonl \
SCHEDULER_TELEMETRY_LAYER=0 \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v3-grace__engine-low__compute16__regression \
  --workloads 1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v3-grace-compute16.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v3-grace-engine-low-compute16-regression-20260719
```

compute-16 验收条件：prefill execution ratio 为 0、decode execution ratio 为 1，正式 exploration fraction 和 switch delta 为 0；CPU decode cost 应高于 iGPU cost，Prefill 接近 137--148 token/s，Decode 接近 21 token/s，E2E 接近 21.3--21.9 s。

### 2026-07-19：v3 compute-16 性能通过，但 16-step grace 仍不足

结果位于 `artifacts/running-server-bench/dynamic-v3-grace-engine-low-compute16-regression-20260719/`：

| Workload | N | Prefill token/s | Decode token/s | TTFT ms | TPOT ms | Output phase ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1024-o300 | 3 | 147.32 [142.92, 150.38] | 20.68 [20.50, 20.87] | 7103.45 [6955.62, 7224.52] | 48.36 [47.82, 48.78] | 14459.96 | 21563.47 |

性能达到 compute-16 的预期区间，说明高竞争下选择 iGPU decode 的方向正确；但控制稳定性没有通过，不能仅凭均值把该点判为最终结果：

| Request | Decode execution ratio | Decode exploration | Switch delta | CPU samples 首/尾 |
|---|---:|---:|---:|---:|
| 0 | 1.0000 | 0.0000 | 0 | 4 / 4 |
| 1 | 0.9800 | 0.0133 | 2 | 4 / 6 |
| 2 | 0.9300 | 0.0133 | 2 | 6 / 21 |

后两次请求均发生了不必要的 iGPU->CPU->iGPU 切换。每次 decode 开头，CPU load 从约 0.25 开始恢复，随后才升至约 0.8；compute-16 下背景 worker 的恢复/迁移时间超过了原来的 16-step phase grace，因此控制器仍把 phase-boundary transient 误判为背景负载真实下降。

旧重采样窗口只有 4 个 CPU 样本。切到 CPU 后，背景 worker 尚未完成迁移，前几个 CPU 样本出现约 0.046--0.053 ms/row 的虚假低值，远低于请求 0 保存的 0.8566 ms/row。控制器过早结束 exploration，并暂时把 CPU 当作更快设备；随着更多 CPU 样本进入，代价才回升并重新切回 iGPU。这解释了 request 2 虽然 exploration 只有 4 steps，CPU 实际执行却达到 21 steps。问题属于 change detector 和重采样窗口不足，不是 iGPU service cost 本身不稳定；iGPU cost 在请求末保持约 0.084 ms/row。

为提高跨负载点稳定性，默认参数调整为：

| 参数 | 环境变量 | 旧值 | 新值 | 目的 |
|---|---|---:|---:|---|
| phase-boundary grace | `KT_CPU_IGPU_DECODE_LOAD_REPROBE_GRACE` | 16 | 64 | 覆盖 compute-16 下较慢的 load 恢复瞬态 |
| CPU reprobe samples | `KT_CPU_IGPU_DECODE_REPROBE_SAMPLES` | 4（与初始校准共用） | 32 | 重采样期间持续标记 exploration，避免用迁移前的少量快样本决策 |
| inactive horizon | `KT_CPU_IGPU_DECODE_REPROBE_INTERVAL` | 1024 | 4096 | 避免 300-token 稳态请求因周期到期产生无意义探测 |
| same-arm load drop | `KT_CPU_IGPU_DECODE_LOAD_REPROBE_DELTA` | 0.15 | 0.25 | 降低 load 波动导致的误触发 |

初始启动校准仍为 CPU/iGPU 各 4 个样本，以控制冷启动开销；只有运行期 CPU 重采样使用 32 个样本。状态机在重采样窗口采满前保持 exploration，不会再依据前 4 个迁移瞬态样本切换到 CPU。64/32/4096/0.25 是当前稳定性候选参数，还必须通过 compute-16、compute-20 和动态负载下降实验，才能作为论文的最终参数；尤其需要验证更长 grace/horizon 不会使真实负载下降后的返回 CPU 响应过慢。

修正后的 SYCL 扩展已完成增量编译并同步到 Python build。状态机、launcher、telemetry、running-server benchmark 和 E2E benchmark 共 44 项测试通过，scheduler Python 配置 3 项测试通过。真实 pybind 默认值读取为 `64/32/4096/0.25`，launcher `PREFLIGHT_ONLY=1` 也确认加载本地 `CPUiGPUGPTQInt4_MOE` 并导出相同参数，无需重新执行完整 `./install.sh`。

保持当前 16-worker 背景任务不变，手动停止旧服务后启动修正版 dynamic：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v3-stable-compute16.jsonl \
SCHEDULER_TELEMETRY_LAYER=0 \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

启动输出必须包含 `calibration=4 reprobe_samples=32 reprobe_interval=4096 load_delta=0.25 load_grace=64`。服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v3-stable__engine-low__compute16__regression \
  --workloads 1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v3-stable-compute16.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v3-stable-engine-low-compute16-regression-20260719
```

compute-16 复测验收条件：三次 prefill execution ratio 均为 0，decode execution ratio 均为 1.0000，正式 exploration fraction 和 switch delta 均为 0，且每次请求恰好有 300 个 decode execution calls。性能应保持在本轮约 147/20.7 token/s、21.6 s E2E 的区间。通过后保持相同参数回归 compute-20，再进行负载从高到低的动态变化实验。

### 2026-07-19：v3 stable compute-16 回归通过

结果位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-compute16-regression-20260719/`：

| Workload | N | Prefill token/s | Decode token/s | TTFT ms | TPOT ms | Output phase ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1024-o300 | 3 | 145.85 [144.89, 147.12] | 20.98 [20.92, 21.02] | 7172.22 [7135.75, 7219.03] | 47.67 [47.58, 47.83] | 14254.02 | 21426.31 |

性能保持在 compute-16 的预期区间。相对 16-step grace 版本，Decode 从 20.68 提高到 20.98 token/s，E2E 从 21563.47 降至 21426.31 ms；更重要的是，请求级控制状态完全稳定：

| Request | Decode calls | Decode execution ratio | Decode exploration | Switch delta | CPU samples 首/尾 |
|---|---:|---:|---:|---:|---:|
| 0 | 300 | 1.0000 | 0.0000 | 0 | 10 / 10 |
| 1 | 300 | 1.0000 | 0.0000 | 0 | 10 / 10 |
| 2 | 300 | 1.0000 | 0.0000 | 0 | 10 / 10 |

三次 prefill execution ratio 也均为 0。正式窗口 switch count 恒为 3，CPU cost 和 samples 分别保持约 0.5520 ms/row 和 10；iGPU cost 从 0.0850 收敛至 0.0840 ms/row，iGPU samples 随每次请求连续增加。即使每次 decode 开头 load 仍暂时降至约 0.20--0.26，64-step grace 也成功阻止了 phase-boundary 误探测。由此确认 `64/32/4096/0.25` 修正解决了 compute-16 的请求级不稳定问题。

下一步回归 compute-20。手动停止当前 dynamic 服务和 16-worker 背景任务，然后启动 20 workers：

```bash
python kt-kernel/bench/cpu_background_load.py \
  --kind compute \
  --workers 20 \
  --affinity free \
  --nice 0
```

启动相同参数的 v3 dynamic：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v3-stable-compute20.jsonl \
SCHEDULER_TELEMETRY_LAYER=0 \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v3-stable__engine-low__compute20__regression \
  --workloads 1024:300 \
  --warmups 15 \
  --repetitions 3 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v3-stable-compute20.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v3-stable-engine-low-compute20-regression-20260719
```

compute-20 验收重点仍是控制稳定性：decode execution ratio 应为 1.0000，正式 exploration fraction 和 switch delta 应为 0，每次请求应有 300 个 decode calls。Prefill 当前仍由 CPU 执行，其性能用于观察完全竞争下的退化程度，不预设必须达到 compute-16 的数值。通过该稳态端点后，再设计高负载运行中减少/停止背景 workers 的动态变化实验，测量控制器从 iGPU 返回 CPU 的响应延迟和切换收益。

### 2026-07-19：v3 stable compute-20 回归通过

结果位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-compute20-regression-20260719/`：

| Workload | N | Prefill token/s | Decode token/s | TTFT ms | TPOT ms | Output phase ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1024-o300 | 3 | 145.82 [141.06, 152.81] | 16.76 [16.31, 17.36] | 7181.43 [6991.26, 7371.60] | 59.72 [57.20, 61.70] | 17856.58 | 25038.07 |

Decode 控制状态通过全部验收条件：三次请求均有 300 个 decode execution calls，iGPU execution ratio 均为 1.0000，exploration fraction 和 switch delta 均为 0。CPU samples 全程保持 4，CPU cost 保持 2.1484 ms/row；iGPU cost 三次请求末分别约为 0.0842、0.0833 和 0.0857 ms/row。因此 `64/32/4096/0.25` 在 compute-8/12/16/20 四个稳态点均没有正式窗口误探测。

相对 compute-16，Prefill 均值几乎不变，但 Decode 从 20.98 降至 16.76 token/s（下降约 20.1%），E2E 从 21426.31 增至 25038.07 ms（延迟增加约 16.9%）。系统 CPU busy 从 91.92% 增至 98.63%，CPU PSI some 从 9.12% 增至 21.59%。与此同时，跨层 iGPU MoE service cost 仍保持约 0.084 ms/row，与 compute-16 基本相同。这说明满核竞争主要拖慢 CPU 上的非 MoE 计算、任务提交和运行时调度，而不是 iGPU 专家计算本身；论文中不能把端到端 Decode 的全部下降都归因于 iGPU 内核。

本轮 report 中 Prefill iGPU ratio 显示 0.3333 是观测伪影。request 0 的 prefill event 出现 `execution_calls_delta=0`，旧 writer 在没有已完成调用时回退到异步 snapshot 1.0；后续两个事件分别携带 2 和 1 个已完成 prefill calls，execution ratio 均为 0。按有效 execution calls 重算，三个 prefill calls 实际全部为 CPU。telemetry 定义已修正：没有已完成调用时写 `igpu_ratio=null`；benchmark 忽略零调用事件，并按 `execution_calls_delta` 加权 execution ratio，同时输出每 phase 的 execution calls。原始 artifact 已包含 counter delta，因此本轮无需重跑。

下一步验证真正的运行期响应，而不是再增加稳态点。`bench_running_server.py` 新增受控高->低负载测量：在指定输出 token 处向显式传入的 `cpu_background_load.py` 父 PID 发送 `SIGTERM`，继续完成同一请求，并用 `monotonic_ns` 对齐 scheduler telemetry。artifact 会记录首次 CPU execution 和结束 exploration、稳定选择 CPU 的 delay calls/delay ms。为防误操作，该模式会验证 PID 命令，只允许单 workload、单 repetition，并要求 scheduler telemetry。

保持当前 compute-20 背景任务和 dynamic 引擎运行，不要提前停止背景任务。当前背景父 PID 为 `771441`；运行前可用 `ps -fp 771441` 再确认。执行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v3-stable__engine-low__compute20-to-none__transition-smoke \
  --workloads 1024:600 \
  --warmups 15 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v3-stable-compute20.jsonl \
  --stop-background-pid 771441 \
  --stop-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v3-stable-engine-low-compute20-to-none-transition-smoke-20260719
```

这条命令会自动停止 20-worker 背景父进程及其 workers，背景窗口应自行退出。初步验收条件：signal 前执行比例保持 iGPU=1；signal 后 load 持续下降并触发恰好一轮 32-call CPU exploration；随后 policy 和 execution ratio 稳定为 CPU=1，直到第 600 个输出 token。若没有返回 CPU，需要区分 load EWMA 未越过 0.25 delta、CPU 重采样 cost 仍高于 iGPU，或 transition 对齐错误，不能直接缩短 grace/horizon。

### 2026-07-19：compute20->none 运行期切换 smoke 通过

结果位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-compute20-to-none-transition-smoke-20260719/`。背景负载在第 150 个输出 token 被 benchmark 成功停止，单次请求结果为：

| Workload | N | Prefill token/s | Decode token/s | TTFT ms | TPOT ms | Output phase ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 152.12 | 24.50 | 6876.09 | 40.81 | 24447.88 | 31324.03 |

控制器完成了预期的完整状态转换：

| 指标 | 结果 |
|---|---:|
| signal 前 decode calls | 150 |
| signal 后首次 CPU execution | 8 calls / 395.02 ms |
| CPU exploration | 精确 32 calls |
| signal 后结束 exploration、稳定 CPU | 40 calls / 1494.77 ms |
| 全请求 iGPU / CPU calls | 158 / 442 |
| switch count | 1 -> 2，仅一次切换 |
| 最终 execution / policy ratio | CPU=1 / CPU=1 |

时序与控制器参数严格一致：停止背景后的前 8 calls 仍使用 iGPU，load EWMA 从约 0.81 下降到 0.23；第 9 个 call 开始 CPU exploration，此时 load 约 0.175。sequence 1345--1376 共 32 个事件均为 `exploration=true`；sequence 1377 的 CPU samples 达到 32，`exploration=false`，之后直到第 600 个 token 均保持 CPU。CPU cost 在探索窗口从约 0.0387 收敛到 0.0366 ms/row，最终约 0.0365 ms/row；保留的 iGPU cost 为 0.0845 ms/row，所以完成重采样后选择 CPU 有明确的 service-cost 依据，而不是仅由 load threshold 强制切换。

telemetry 分段时间轴也与端点实验一致。首个 decode telemetry 间隔包含约 3.31 s 的 phase-boundary 异步间隔，不能纳入稳态 token rate；排除该已知边界后，signal 前高负载 iGPU 段约为 16.81 calls/s，与 compute-20 稳态的 16.76 token/s 一致。稳定切回 CPU 后为约 29.07 calls/s，接近无背景负载 dynamic 的约 29.45 token/s。这解释了整段请求最终达到 24.50 token/s：前 158 calls 承担高负载/检测开销，后 442 calls 使用更快的无负载 CPU 路径。

该结果证明三个机制同时成立：load-drop change detector 能识别真实负载下降；32-sample reprobe 能得到稳定 CPU service cost；控制器能在同一请求内从 iGPU 返回 CPU并获得端到端收益。但当前仅为 N=1 smoke，不能据此报告置信区间或最终响应分布。论文实验至少需要 5 次独立 high->low cycle，报告 detection delay、settle delay、exploration calls、switch count 和分段 token rate 的均值、标准差及 bootstrap 95% CI；还需要对称的 none->compute20 实验，验证背景任务突然启动时 CPU active cost 恶化能否及时触发 CPU->iGPU。

### 2026-07-19：compute20->none transition cycle 2 通过

结果位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-compute20-to-none-transition-cycle02-20260719/`：

| Workload | N | Prefill token/s | Decode token/s | TTFT ms | TPOT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 145.75 | 24.58 | 7176.68 | 40.68 | 31542.99 |

cycle 2 严格复现 cycle 1 的 call 级状态序列：signal 前 150 calls 全部使用 iGPU；signal 后等待 8 calls、424.78 ms 开始 CPU execution；随后恰好 32 calls 为 exploration；第 40 call、1529.76 ms 结束 exploration 并稳定使用 CPU。全请求仍为 158 iGPU calls 和 442 CPU calls，正式 switch count 从 3 增至 4，仅发生一次 iGPU->CPU 切换，最终 execution/policy ratio 均为 CPU=1。

cycle 1 结束时 switch count 为 2、控制器处于 CPU；重新启动 compute-20 背景并执行本轮 warmup 后，正式请求的初始 switch count 已为 3、execution ratio 为 iGPU=1。这表明 warmup 期间控制器先完成了一次 CPU->iGPU 恢复，正式请求又完成 iGPU->CPU 返回。同一服务实例能够连续双向切换，不依赖进程重启或重新加载权重。不过此次 CPU->iGPU 发生在 warmup 窗口，尚未测量其响应延迟。

前两轮汇总如下；分段 rate 均排除了首个已知 phase-boundary 异步间隔：

| 指标 | Cycle 1 | Cycle 2 | Mean | Sample stdev |
|---|---:|---:|---:|---:|
| End-to-end Decode token/s | 24.501 | 24.583 | 24.542 | 0.058 |
| First CPU delay calls | 8 | 8 | 8 | 0 |
| First CPU delay ms | 395.02 | 424.78 | 409.90 | 21.04 |
| Settle delay calls | 40 | 40 | 40 | 0 |
| Settle delay ms | 1494.77 | 1529.76 | 1512.27 | 24.74 |
| High-load iGPU calls/s | 16.807 | 17.166 | 16.986 | 0.254 |
| Low-load CPU calls/s | 29.074 | 29.036 | 29.055 | 0.026 |

两轮的 detection/settle call 数完全一致，时间标准差分别约为 21.0/24.7 ms，低负载 CPU 分段 rate 的标准差仅 0.026 calls/s，初步重复性良好。但 N=2 仍不生成置信区间；继续完成 cycle 3--5 后统一生成 bootstrap 95% CI。

为简化后续操作，`cpu_background_load.py` 的 ready JSON 现新增 `parent_pid`。启动 cycle 3 的 20-worker 背景后，直接使用输出中的该值作为 benchmark 的 `--stop-background-pid`，不再需要通过 `pgrep` 区分父进程和 20 个 workers。

### 2026-07-19：compute20->none transition cycle 3 通过

结果位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-compute20-to-none-transition-cycle03-20260719/`。单次请求 Prefill/Decode/TTFT/TPOT/E2E 分别为 148.75 token/s、24.56 token/s、7031.84 ms、40.72 ms/token 和 31425.94 ms。

cycle 3 在 signal 后 6 calls、317.55 ms 开始 CPU execution，仍精确执行 32-call exploration，并在第 38 call、1419.13 ms 稳定选择 CPU。全请求为 156 iGPU calls 和 444 CPU calls；正式 switch count 从 5 增至 6，仅一次切换，最终 execution/policy ratio 均为 CPU=1。相对前两轮的 8-call detection，本轮提前 2 calls，原因是 50 ms load sampler 与 signal 的相位不同；exploration 长度仍固定为 32，所以 settle call 同步从 40 降到 38。这属于预期的观测/采样离散性，不是控制状态抖动。

三轮汇总更新为：

| 指标 | Cycle 1 | Cycle 2 | Cycle 3 | Mean | Sample stdev |
|---|---:|---:|---:|---:|---:|---:|
| End-to-end Decode token/s | 24.501 | 24.583 | 24.555 | 24.546 | 0.042 |
| First CPU delay calls | 8 | 8 | 6 | 7.333 | 1.155 |
| First CPU delay ms | 395.02 | 424.78 | 317.55 | 379.12 | 55.35 |
| Settle delay calls | 40 | 40 | 38 | 39.333 | 1.155 |
| Settle delay ms | 1494.77 | 1529.76 | 1419.13 | 1481.22 | 56.55 |
| High-load iGPU calls/s | 16.807 | 17.166 | 16.940 | 16.971 | 0.182 |
| Low-load CPU calls/s | 29.074 | 29.036 | 29.029 | 29.047 | 0.024 |

三轮均满足单次切换、32-call exploration 和最终 CPU 稳态，且端点分段 rate 高度一致。继续 cycle 4--5；达到 N=5 后再固定 high->low 的正式统计表，并转向受控 none->compute20 测量。

### 2026-07-19：compute20->none transition cycle 4 通过

结果位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-compute20-to-none-transition-cycle04-20260719/`。Prefill/Decode/TTFT/TPOT/E2E 分别为 152.28 token/s、25.03 token/s、6868.73 ms、39.96 ms/token 和 30802.16 ms。

客户端在收到第 150 个输出 token 时发送 signal，但同一时刻服务端 telemetry 已完成 151 个 decode calls；这是一项正常的单步流水并行差异。控制器在 signal 后 9 calls、489.12 ms 开始 CPU execution，精确执行 32-call exploration，并在第 41 call、1601.18 ms 稳定选择 CPU。全请求为 160 iGPU calls 和 440 CPU calls，正式 switch count 从 7 增至 8，仅一次切换，最终 ratio 为 CPU=1。不能简单用客户端 stop token 150 推导服务端 device calls，正式统计必须以对齐后的 `monotonic_ns` 和 execution counters 为准。

四轮汇总为：

| 指标 | C1 | C2 | C3 | C4 | Mean | Sample stdev |
|---|---:|---:|---:|---:|---:|---:|
| End-to-end Decode token/s | 24.501 | 24.583 | 24.555 | 25.028 | 24.667 | 0.243 |
| First CPU delay calls | 8 | 8 | 6 | 9 | 7.750 | 1.258 |
| First CPU delay ms | 395.02 | 424.78 | 317.55 | 489.12 | 406.62 | 71.19 |
| Settle delay calls | 40 | 40 | 38 | 41 | 39.750 | 1.258 |
| Settle delay ms | 1494.77 | 1529.76 | 1419.13 | 1601.18 | 1511.21 | 75.69 |
| High-load iGPU calls/s | 16.807 | 17.166 | 16.940 | 18.116 | 17.257 | 0.591 |
| Low-load CPU calls/s | 29.074 | 29.036 | 29.029 | 28.923 | 29.016 | 0.065 |

cycle 4 的整体 Decode 较高，主要对应 signal 前 high-load 段 rate 提高到 18.116 calls/s，而切换后的 CPU 稳态仍为 28.923 calls/s，未出现控制器异常。完成 cycle 5 后统一计算 bootstrap CI。

### 2026-07-19：compute20->none transition N=5 正式汇总

cycle 5 位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-compute20-to-none-transition-cycle05-20260719/`。本轮 Prefill/Decode/TTFT/TPOT/E2E 分别为 154.83 token/s、25.88 token/s、6755.86 ms、38.64 ms/token 和 29899.68 ms。signal 后 7 calls、372.11 ms 开始 CPU execution，精确执行 32-call exploration，在第 39 call、1477.26 ms 稳定选择 CPU；正式 switch count 从 9 增至 10，仅一次切换，最终 CPU ratio 为 1。

五轮原始 artifact 已由可复现工具 `kt-kernel/bench/report_load_transition_cycles.py` 聚合。正式结果目录为 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-compute20-to-none-transition-n5-20260719/`，包含 `cycles.csv`、`summary.csv`、`manifest.json` 和 `report.md`。聚合参数为 10000 次 percentile bootstrap、seed 20260719；high-load 分段 rate 明确定义为排除首个 prefill/decode phase-boundary interval 后的 execution calls/s。

N=5 正式统计：

| 指标 | Mean | Sample stdev | Bootstrap 95% CI |
|---|---:|---:|---:|
| End-to-end Decode token/s | 24.9098 | 0.5827 | [24.5392, 25.4348] |
| First CPU delay calls | 7.6000 | 1.1402 | [6.6000, 8.4000] |
| First CPU delay ms | 399.72 | 63.55 | [350.29, 451.48] |
| Settle CPU delay calls | 39.6000 | 1.1402 | [38.8000, 40.4000] |
| Settle CPU delay ms | 1504.42 | 67.29 | [1454.01, 1558.62] |
| High-load iGPU calls/s | 17.7944 | 1.3061 | [16.9318, 18.9506] |
| Low-load CPU calls/s | 29.0164 | 0.0560 | [28.9658, 29.0539] |
| Low/high segment speedup | 1.6373x | 0.1143 | [1.5383x, 1.7157x] |

五轮均满足以下不变量：CPU exploration 恰好 32 calls；正式窗口仅发生一次 iGPU->CPU switch；请求结束时 execution/policy ratio 均为 CPU=1；背景父进程及 workers 均被 benchmark 正常停止。First/settle delay 的 call 数波动仅来自 50 ms load sampler 与客户端 signal 的相位差；一旦触发，固定 32-call 重采样窗口没有变化。low-load CPU rate 的标准差只有 0.056 calls/s，说明切换后的端点稳定。

high-load iGPU 分段 rate 在五轮中为 16.807--19.944 calls/s，波动高于 CPU 端点。这与满核背景下 CPU runtime/提交线程竞争有关，应在论文中作为消费级多任务系统噪声报告，而不能只选择较快轮次。尽管如此，每轮 CPU 低负载端点都更快，分段 speedup 的 bootstrap 下界仍为 1.538x，因此运行中返回 CPU 的收益具有一致方向。

这组结果可作为 high->low 机制和响应时间的正式实验。下一项为对称的 none->compute20：请求开始时控制器稳定使用 CPU，在指定输出 token 启动 20-worker normal-priority 背景负载，测量 CPU active service cost 恶化后首次/稳定切换 iGPU 的 calls、ms 和分段性能。该方向不应触发 CPU reprobe；预计依靠 active CPU EWMA 与已保存 iGPU cost 直接切换。

`bench_running_server.py` 已实现受控 none->compute20 模式：`--start-compute-background-workers 20` 与 `--start-background-after-output-tokens 150` 会在客户端收到第 150 个输出 token 时非阻塞启动 `compute/free/nice=0` 背景负载。benchmark 同时记录进程 launch 和全部 workers ready 的 `monotonic_ns`，请求结束后自动停止父进程及整个进程组。为保证低负载初态，该模式发现任何现存 `cpu_background_load.py` 时都会拒绝启动；start 与 stop transition 参数也互斥。

transition telemetry 已泛化为 CPU/iGPU target，可分别输出 launch 后首次 iGPU execution、稳定 iGPU policy 的 calls/ms，以及相对 workers ready 的延迟。1-worker 真实生命周期预检的 ready delay 约 25.1 ms，进程组正常清理；完整相关测试 51 项、scheduler 配置测试 3 项通过。

当前 cycle 5 已使控制器停在 CPU，背景进程也已退出。保持引擎运行，执行 none->compute20 smoke：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v3-stable__engine-low__none-to-compute20__transition-smoke \
  --workloads 1024:600 \
  --warmups 15 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v3-stable-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-smoke-20260719
```

验收条件：launch 前 execution ratio 为 CPU=1；20 workers ready 后 CPU active cost 明显上升；控制器最多进行一次 CPU->iGPU switch，最终 execution/policy ratio 为 iGPU=1；不应出现 CPU reprobe exploration。若 CPU cost 在剩余 450 calls 内始终未超过带 10% margin 的 iGPU cost，则应报告为 service-cost 不满足，而不是强制按 load 切换。

### 2026-07-19：none->compute20 运行期切换 smoke 通过

结果位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-smoke-20260719/`。20 个 `compute/free/nice=0` workers 在第 150 个客户端输出 token 启动，父 PID 为 811578，全部 workers 在 launch 后 59.10 ms ready，并在请求结束后由 benchmark 正常清理。单次请求结果：

| Workload | N | Prefill token/s | Decode token/s | TTFT ms | TPOT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 169.17 | 18.41 | 6183.12 | 54.31 | 38717.30 |

正式窗口起始 150 个服务端 decode calls 全部使用 CPU。launch 后继续执行 14 个 CPU calls，并在第 14 call、1174.47 ms 直接切换到 iGPU；相对 workers ready 的延迟为 1115.37 ms。first iGPU execution 与 settled iGPU 是同一事件，没有 exploration；正式 switch count 从 10 增至 11，仅一次切换。全请求为 164 CPU calls、436 iGPU calls，最终 execution/policy ratio 均为 iGPU=1。

该动作明确由 active CPU service cost 驱动。launch 前 CPU cost 约 0.0365 ms/row；背景进程启动后，前 13 个 CPU rounds 的 EWMA 仍不超过约 0.0697 ms/row。第 14 个受竞争 round 将 CPU cost 推高到 0.2075 ms/row，超过带 10% margin 的已保存 iGPU cost（切换前约 0.0875 ms/row），控制器随即选择 iGPU。切换事件的 load 约 0.425，但 v3 不按 load threshold 直接决定设备；本轮 exploration fraction 为 0 也证明没有经过 load-triggered CPU reprobe。

telemetry 分段 rate 为：launch 前无负载 CPU 约 28.963 calls/s；launch 后、切换前受竞争 CPU 约 15.779 calls/s；切换后的高负载 iGPU 约 16.637 calls/s。后两段的端到端 calls rate 提升约 5.44%，小于 MoE service cost 的差距，因为 CPU 上的非 MoE 和 runtime 计算在满核背景下仍是共同瓶颈。切换后的 rate 与 compute-20 稳态 16.76 token/s 同一区间，端点方向正确。

该 smoke 证明 CPU active cost 恶化可以在同一请求内直接触发 CPU->iGPU，不需要额外 exploration，也不依赖 load threshold。与 high->low 的 32-call CPU reprobe 相比，low->high 的动作更直接，但等待真实 CPU cost 恶化证据使本轮响应约为 1.17 s；后续 N=5 应报告这一延迟分布。

重复 none->compute20 时必须恢复 CPU 初态。本轮结束后控制器停在 iGPU，而默认 warmup 只有 8 decode tokens，小于 64-step phase grace，无法触发 load-drop CPU reprobe。benchmark 已新增 warmup 终态校验：start-background 模式若正式测量前不是 `execution=CPU, policy=CPU, exploration=false` 会拒绝运行。cycle 2--5 使用至少一次 128-token 无负载 warmup，例如 `--warmups 3 --warmup-output-tokens 128`；这会在正式请求前完成 iGPU->CPU 恢复并留下足够的 CPU 稳态样本。

### 2026-07-19：none->compute20 transition cycle 2 通过

结果位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-cycle02-20260719/`。本轮使用 3 次 128-token 无负载 warmup；正式请求起始 switch count 为 12，而上一轮结束为 11，说明 warmup 已完成 iGPU->CPU 恢复。benchmark 的 CPU 初态校验通过。

20 workers 在 48.43 ms ready。正式请求前 150 个 decode calls 使用 CPU；launch 后仅继续执行 3 个 CPU calls，并在 735.50 ms（相对 ready 为 687.07 ms）直接切换 iGPU。first/settled iGPU 为同一事件，exploration 为 0；正式 switch count 从 12 增至 13，仅一次切换。全请求为 153 CPU calls 和 447 iGPU calls，最终 execution/policy ratio 均为 iGPU=1。CPU cost 从 0.0365 上升至 0.4178 ms/row，iGPU cost 最终约 0.0846 ms/row，service-cost 方向明确。

前两轮汇总：

| 指标 | Smoke | Cycle 2 | Mean |
|---|---:|---:|---:|
| End-to-end Decode token/s | 18.411 | 18.040 | 18.226 |
| Workers ready ms | 59.10 | 48.43 | 53.77 |
| First iGPU delay calls | 14 | 3 | 8.5 |
| First iGPU delay ms | 1174.47 | 735.50 | 954.99 |
| Ready-to-first iGPU ms | 1115.37 | 687.07 | 901.22 |
| Pre-launch low-load CPU calls/s | 28.963 | 28.970 | 28.967 |
| Post-switch high-load iGPU calls/s | 16.637 | 16.352 | 16.494 |

两轮端点 rate 很稳定，但 CPU->iGPU 响应从 14 calls 降至 3 calls。该差异不是误切换：两轮都无 exploration、只有一次正式 switch，最终均为 iGPU。它反映了背景 worker 启动后首批 CPU rounds 的竞争强度和调度位置差异；这正是 N=5 需要估计的响应分布。cycle 3--5 继续使用 `--warmups 3 --warmup-output-tokens 128`，无需手动管理背景进程。

### 2026-07-19：none->compute20 transition cycle 3 通过

结果位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-cycle03-20260719/`。Prefill/Decode/TTFT/TPOT/E2E 分别为 169.68 token/s、19.24 token/s、6164.40 ms、51.98 ms/token 和 37298.26 ms。

20 workers 在 52.19 ms ready。正式请求从 CPU 稳态开始，launch 后继续执行 8 个 CPU calls，在 615.52 ms（相对 ready 563.33 ms）直接切到 iGPU；无 exploration，正式 switch count 从 14 增至 15，仅一次切换。全请求为 158 CPU calls 和 442 iGPU calls，最终 execution/policy ratio 均为 iGPU=1。CPU cost 从 0.0364 上升到 0.1665 ms/row，iGPU cost 最终约 0.0836 ms/row。

三轮汇总：

| 指标 | C1 | C2 | C3 | Mean | Sample stdev |
|---|---:|---:|---:|---:|---:|
| End-to-end Decode token/s | 18.411 | 18.040 | 19.240 | 18.564 | 0.614 |
| Workers ready ms | 59.10 | 48.43 | 52.19 | 53.24 | 5.41 |
| First iGPU delay calls | 14 | 3 | 8 | 8.333 | 5.508 |
| First iGPU delay ms | 1174.47 | 735.50 | 615.52 | 841.83 | 294.26 |
| Ready-to-first iGPU ms | 1115.37 | 687.07 | 563.33 | 788.59 | 289.69 |
| Pre-launch low-load CPU calls/s | 28.963 | 28.970 | 28.893 | 28.942 | 0.043 |
| Post-switch high-load iGPU calls/s | 16.637 | 16.352 | 17.438 | 16.809 | 0.563 |

cycle 3 的整体 Decode 较高，同时来自更短的切换延迟和更快的高负载 iGPU 端点；不是 measurement/control failure。继续 cycle 4--5，保持相同恢复 warmup 和自动背景参数。

### 2026-07-19：none->compute20 transition cycle 4 通过

结果位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-cycle04-20260719/`。Prefill/Decode/TTFT/TPOT/E2E 分别为 169.68 token/s、18.66 token/s、6164.64 ms、53.59 ms/token 和 38264.15 ms。

20 workers 在 58.17 ms ready。正式请求以 CPU 开始，launch 后执行 7 个 CPU calls，在 1219.53 ms（相对 ready 1161.36 ms）直接切到 iGPU。exploration 为 0，正式 switch count 从 16 增至 17，仅一次切换；全请求为 157 CPU calls 和 443 iGPU calls，最终 execution/policy ratio 均为 iGPU=1。CPU cost 从 0.0365 上升至 0.4949 ms/row，iGPU cost 最终约 0.0859 ms/row。

四轮汇总：

| 指标 | C1 | C2 | C3 | C4 | Mean | Sample stdev |
|---|---:|---:|---:|---:|---:|---:|
| End-to-end Decode token/s | 18.411 | 18.040 | 19.240 | 18.661 | 18.588 | 0.504 |
| Workers ready ms | 59.10 | 48.43 | 52.19 | 58.17 | 54.47 | 5.06 |
| First iGPU delay calls | 14 | 3 | 8 | 7 | 8.000 | 4.546 |
| First iGPU delay ms | 1174.47 | 735.50 | 615.52 | 1219.53 | 936.26 | 305.60 |
| Ready-to-first iGPU ms | 1115.37 | 687.07 | 563.33 | 1161.36 | 881.78 | 301.14 |
| Pre-launch low-load CPU calls/s | 28.963 | 28.970 | 28.893 | 28.916 | 28.936 | 0.037 |
| Post-switch high-load iGPU calls/s | 16.637 | 16.352 | 17.438 | 17.220 | 16.912 | 0.504 |

四轮均保持 CPU 初态、无 exploration、一次 CPU->iGPU switch 和最终 iGPU 稳态。First iGPU delay 的标准差约 306 ms，显著高于 workers ready 时间的约 5 ms 标准差，说明方差来自 workers ready 后 CPU 调度竞争如何落到首批 engine rounds，而不是背景进程创建速度。完成 cycle 5 后生成 none->compute20 的正式 bootstrap 报告。

### 2026-07-19：none->compute20 transition N=5 正式汇总

cycle 5 位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-cycle05-20260719/`。本轮 Prefill/Decode/TTFT/TPOT/E2E 分别为 169.43 token/s、18.54 token/s、6173.49 ms、53.95 ms/token 和 38490.27 ms。正式请求仍从 CPU 开始，无 exploration，仅一次 CPU->iGPU switch，最终 iGPU ratio 为 1。

cycle 5 的背景启动过程出现重要系统噪声：20 workers 全部 ready 耗时 2385.62 ms，明显高于前四轮的 48--59 ms。控制器在 launch 后 35 calls、3248.19 ms 切换，但相对 all-workers-ready 只需 1 call、862.57 ms。该轮不能删除，因为进程创建受端侧系统调度影响本身属于研究场景；但不能把 3.248 s 全部归因于调度器。正式报告同时保留 process launch 和 workers ready 两个时间原点，并将 ready->iGPU 作为主要控制响应指标。

双向聚合工具已扩展为统一的 direction/target schema，并从原始 telemetry 计算 ready 后的 execution calls。N=5 正式目录为 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-n5-20260719/`，包含五轮明细、10000 次 bootstrap 汇总和 manifest。

none->compute20 N=5 结果：

| 指标 | Mean | Sample stdev | Bootstrap 95% CI |
|---|---:|---:|---:|
| End-to-end Decode token/s | 18.5775 | 0.4368 | [18.2386, 18.9579] |
| Launch->iGPU calls | 13.4000 | 12.7004 | [5.6000, 24.4000] |
| Launch->iGPU ms | 1398.64 | 1067.26 | [760.32, 2339.92] |
| Workers ready ms | 520.70 | 1042.53 | [51.44, 1453.45] |
| Ready->iGPU calls | 5.0000 | 4.5277 | [1.8000, 8.6000] |
| Ready->iGPU ms | 877.94 | 260.93 | [672.67, 1083.21] |
| Pre-transition low-load CPU calls/s | 28.9294 | 0.0351 | [28.9022, 28.9566] |
| Post-transition high-load iGPU calls/s | 17.0022 | 0.4809 | [16.6113, 17.3649] |

Workers-ready 中位数为 58.17 ms；520.70 ms 的均值和宽 CI 由 cycle 5 的真实长尾拉高。相比之下，ready->iGPU 延迟的 95% CI 明显更窄。五轮都满足：正式初态 CPU=1；exploration=0；只发生一次 CPU->iGPU switch；终态 iGPU=1；请求结束后 20 workers 全部清理。由此可将 none->compute20 的控制机制和响应分布判为通过。

`post/pre rate=0.5877` 不能解释为调度导致性能下降，因为 transition 同时把系统从无负载变为满核负载；两个分段不是同一外部条件。当前实验只证明动态控制器在竞争出现后切到 service-cost 更优的 iGPU arm。要量化“调度带来的净收益”，下一步必须在完全相同的 none->compute20 时间序列下补 static CPU 和 static iGPU counterfactual，比较 transition 后分段吞吐、整体 Decode 和尾延迟；不能只拿动态前后的两个端点相除。

为支持静态反事实，`bench_running_server.py` 新增逐输出 chunk 的 `monotonic_ns` 记录，并输出客户端 pre/post-transition token/s。`--transition-static-baseline` 允许 fixed backend 在没有 scheduler telemetry 时使用同一 managed background transition；仍保留单 workload、单 repetition、无现存背景进程和自动进程组清理约束。定向测试通过。

先测 packed CPU fixed。手动停止当前 dynamic 引擎，确认没有背景负载，然后启动：

```bash
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh packed-cpu-fixed
```

服务就绪后执行同一 none->compute20 序列：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label packed-cpu-fixed__engine-low__none-to-compute20__transition-smoke \
  --workloads 1024:600 \
  --warmups 15 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --transition-static-baseline \
  --fail-fast \
  --output-dir artifacts/running-server-bench/packed-cpu-fixed-engine-low-none-to-compute20-transition-smoke-20260719
```

该请求后 450 tokens 始终使用 CPU，可能持续数分钟；不能因变慢提前中断。主要结果是 client post-transition token/s 和 E2E。完成 CPU fixed 后再以相同命令测 iGPU fixed，仅替换引擎模式、run label 和输出目录。最后需要补一轮带新 client timestamp 的 dynamic transition，才能做完全同口径的三策略分段比较。

### 2026-07-19：packed CPU fixed none->compute20 静态反事实完成

结果位于 `artifacts/running-server-bench/packed-cpu-fixed-engine-low-none-to-compute20-transition-smoke-20260719/`。请求在第 150 个输出 token 启动 20 个 `compute/free/nice=0` workers；全部 workers 在 46.76 ms 内 ready，父进程和 workers 均在请求结束后由 benchmark 正常清理。固定 CPU 引擎不会进行设备切换，因此后 450 tokens 始终承受满核竞争。

| Workload | N | Prefill token/s | Overall Decode token/s | TTFT ms | TPOT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 168.08 | 6.43 | 6223.40 | 155.50 | 99367.79 |

客户端按同一个 background launch 时间点切分后，负载前 CPU decode 为 29.820 token/s，负载后固定 CPU decode 为 5.105 token/s。即满核 normal-priority 背景负载使低优先级 CPU 推理吞吐下降 82.88%，负载前/后速度比为 5.84x。整请求 CPU busy 为 93.60%，其中 user 55.34%、nice 36.23%、system 2.03%；CPU PSI `some` 为 39.82%，确认该慢速段确实处于显著 CPU runnable contention，而不是背景负载未生效。

这组数据是动态调度净收益所需的 CPU static counterfactual，并强烈表明在 compute20 下继续使用 CPU 不合理。不过当前只能得出方向性结论，不能直接把 `5.105 token/s` 与前述 dynamic telemetry 的 `17.002 calls/s` 相除：前者是客户端 token timestamp 口径，旧 dynamic 五轮是在新增 timestamp 记录之前完成，二者测量层次不同。严格三策略比较还需要相同序列的 iGPU fixed，以及一轮启用新客户端分段时间戳的 dynamic。

下一步测量 iGPU fixed。手动停止当前 packed CPU fixed 引擎，确认没有遗留背景进程，然后启动：

```bash
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh igpu-fixed
```

服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label igpu-fixed__engine-low__none-to-compute20__transition-smoke \
  --workloads 1024:600 \
  --warmups 15 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --transition-static-baseline \
  --fail-fast \
  --output-dir artifacts/running-server-bench/igpu-fixed-engine-low-none-to-compute20-transition-smoke-20260719
```

iGPU fixed 的负载前段也会始终使用 iGPU，因此它不是低负载最优策略，但可隔离 compute20 条件下固定 iGPU 的客户端吞吐。主要检查 client post-transition token/s、整体 Decode、E2E、CPU/PSI，以及背景进程是否成功清理。

### 2026-07-19：iGPU fixed none->compute20 静态反事实完成

结果位于 `artifacts/running-server-bench/igpu-fixed-engine-low-none-to-compute20-transition-smoke-20260719/`。请求仍在第 150 个输出 token 启动 20 个 `compute/free/nice=0` workers；全部 workers 在 89.43 ms 内 ready，并在请求结束后正常清理。固定 iGPU 引擎全程不切换设备。

| Workload | N | Prefill token/s | Overall Decode token/s | TTFT ms | TPOT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 85.16 | 14.61 | 12282.51 | 68.47 | 53295.39 |

客户端分段吞吐为：负载前固定 iGPU 17.090 token/s，负载后固定 iGPU 13.935 token/s，compute20 使 iGPU 路径下降 18.46%。这说明 iGPU 并非完全不受 CPU 负载影响：主机 runtime、提交线程和非 MoE CPU 计算仍参与端到端 decode；但下降幅度远小于固定 CPU 的 82.88%。整请求 CPU busy 为 66.38%，user/nice/system 分别为 56.23%、6.46% 和 3.69%，CPU PSI `some` 为 7.12%。

相同 transition 条件下的两个静态端点对比：

| 指标 | packed CPU fixed | iGPU fixed | iGPU/CPU 或收益 |
|---|---:|---:|---:|
| Client pre-transition token/s | 29.820 | 17.090 | 低负载 CPU 快 1.745x |
| Client post-transition token/s | 5.105 | 13.935 | 高负载 iGPU 快 2.730x |
| Overall Decode token/s | 6.431 | 14.605 | iGPU fixed 快 2.271x |
| E2E ms | 99367.79 | 53295.39 | iGPU fixed 缩短 46.37% |

两项静态反事实分别支持动态策略的两端决策：低负载应使用 CPU，高负载应使用 iGPU。与此同时，iGPU fixed 的 Prefill 只有 85.16 token/s，而 packed CPU fixed 为 168.08 token/s；因此论文必须分别报告 Prefill/TTFT 与 Decode/TPOT，不能用单一吞吐掩盖固定 iGPU 的首 token 代价。

最后补一轮带客户端分段 timestamp 的 dynamic，形成严格同口径三方比较。手动停止当前 iGPU fixed 引擎后启动新 dynamic；启动脚本会删除同名旧 telemetry 文件，当前文件名也尚未被使用：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v3-client-segment-compute20.jsonl \
SCHEDULER_TELEMETRY_LAYER=0 \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v3-stable__engine-low__none-to-compute20__transition-client-segment-smoke \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v3-client-segment-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-client-segment-smoke-20260719
```

验收条件与 N=5 dynamic 相同：正式请求以 CPU 开始，负载启动后不进行 exploration，最多一次 CPU->iGPU switch，终态为 iGPU；同时本轮必须在 `samples.jsonl` 中得到非空的 `transition_client_pre_tps` 和 `transition_client_post_tps`。最终主要净收益定义为 dynamic post-transition client token/s 相对 packed CPU fixed 的 5.105 token/s；dynamic 相对 iGPU fixed 的价值则由低负载 pre-transition、整体 Decode、TTFT 和 E2E 共同说明。

### 2026-07-19：dynamic 客户端分段 smoke 与 token 口径修正

结果位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-client-segment-smoke-20260719/`。服务端整体指标有效：Prefill/Decode/TTFT/TPOT/E2E 分别为 161.67 token/s、15.92 token/s、6470.11 ms、62.81 ms/token 和 44094.88 ms。Prefill 全部使用 CPU；decode 共 600 execution calls，其中 164 CPU、436 iGPU，正式窗口 iGPU ratio 为 0.7267。

20 workers 在 49.27 ms ready。负载启动后控制器继续执行 13 个 CPU calls，在 launch 后 1791.61 ms、workers-ready 后 1742.34 ms 直接切换 iGPU。无 exploration，正式 switch count 只增加 1，first/settled iGPU 是同一事件，最终 execution/policy ratio 均为 iGPU=1。切换前 CPU EWMA cost 约 0.044 ms/row；第 13 个受竞争 CPU round 将其推高到 0.442 ms/row，超过带 margin 的 iGPU 保存 cost 0.294 ms/row，因而触发 service-cost switch。该响应比既有 N=5 的 ready->iGPU 均值 877.94 ms 更慢，应作为额外长尾样本保留。

本轮客户端原始分段输出为 pre=26.934、post=13.960 token/s。按该近似值，dynamic post 相对 packed CPU fixed 的 5.105 token/s 为 2.734x，并与 iGPU fixed 的 13.935 token/s 相差仅 +0.18%；整体 Decode 相对 CPU/iGPU fixed 分别为 2.476x 和 1.090x，E2E 分别缩短 55.62% 和 17.26%。dynamic Prefill 仅比 packed CPU fixed 低 3.81%，却比 iGPU fixed 快 1.898x。因此策略同时接近低负载 CPU 的 TTFT/Prefill 和高负载 iGPU 的 decode 端点。

但检查 artifact 后发现：服务端 usage 为 600 completion tokens，客户端只有 598 个非空 SSE text chunks。旧版 transition 客户端分段以非空 chunk 时间戳作为 token 时间戳，因此本轮 pre/post 数值存在至多约 0.33% 的计数偏差；它不改变 2.734x 的方向和量级，但不能作为论文中的“严格 token/s”最终值。两个 static smoke 均恰好为 600 chunks/600 tokens，所以其分段值不受此问题影响。

根因是部分 token 可产生空文本增量，或一个文本增量覆盖多个累计 token。benchmark 已改用本地 SGLang 支持的 `stream_options.continuous_usage_stats=true`：每个流事件读取服务端累计 `completion_tokens`，按 token 增量记录时间戳并在精确第 150 token 触发背景负载；空文本和批量 token 增量都被计入。`stream_chunks` 继续保留为诊断字段，同时新增 `stream_tokens`；transition 请求若未获得与最终 completion tokens 等长的 token 时间戳会直接失败。新增空文本和多 token 流事件测试，`test_running_server_bench.py` 共 20 项通过。

该修正仅修改 benchmark 客户端，无需重启或重新编译当前 dynamic 引擎。当前背景进程已清理；控制器虽停在 iGPU，但 3 次 128-token 无负载 warmup 会恢复 CPU，并由初态校验确认。使用新目录重跑 token-exact smoke：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v3-stable__engine-low__none-to-compute20__transition-token-exact-smoke \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v3-client-segment-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-token-exact-smoke-20260719
```

新 artifact 必须满足 `stream_tokens=completion_tokens=600`；`stream_chunks` 可以小于 600。该轮通过后再决定是否直接进入三策略 N>=5 正式重复实验。

### 2026-07-19：dynamic token-exact smoke 通过

结果位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-token-exact-smoke-20260719/`。新客户端口径生效：`stream_tokens=completion_tokens=600`，本轮 `stream_chunks` 也恰好为 600；负载在精确第 150 个服务端累计输出 token 启动，telemetry 对应 150 个 transition 前和 450 个 transition 后 decode calls。背景 20 workers 在 61.44 ms ready，并在请求结束后全部清理。

| Workload | N | Prefill token/s | Overall Decode token/s | TTFT ms | TPOT ms | E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| p1024-o600 | 1 | 169.83 | 18.08 | 6159.26 | 55.32 | 39296.23 |

客户端精确分段为：transition 前 29.596 token/s，transition 后 16.013 token/s。控制器在 launch 后 3 calls/621.52 ms、workers-ready 后 560.07 ms 直接切换 iGPU；无 exploration，switch count 只增加 1，最终 execution/policy ratio 均为 iGPU=1。CPU cost 在低负载时约 0.0376 ms/row，受竞争 round 将其推高到 0.3450 ms/row，超过保存的 iGPU cost 0.0888 ms/row，因此切换符合 service-cost 判据。本轮 153 CPU calls、447 iGPU calls，decode iGPU ratio 为 0.745。

三策略 token-exact smoke 对比：

| 指标 | packed CPU fixed | iGPU fixed | Dynamic | Dynamic vs CPU | Dynamic vs iGPU |
|---|---:|---:|---:|---:|---:|
| Client pre-transition token/s | 29.820 | 17.090 | 29.596 | -0.75% | +73.18% |
| Client post-transition token/s | 5.105 | 13.935 | 16.013 | 3.137x | +14.91% |
| Overall Decode token/s | 6.431 | 14.605 | 18.077 | 2.811x | 1.238x |
| Prefill token/s | 168.075 | 85.162 | 169.826 | +1.04% | 1.994x |
| TTFT ms | 6223.40 | 12282.51 | 6159.26 | -1.03% | -49.85% |
| E2E ms | 99367.79 | 53295.39 | 39296.23 | -60.45% | -26.27% |

该 N=1 结果同时接近低负载 CPU 端点并在高负载后切到 iGPU，方向和机制均通过。Dynamic post 比本次 iGPU fixed 高 14.91% 不能解释为调度器使同一 iGPU kernel 加速：两者切换后使用相同设备路径，差异还包含 CPU runtime 竞争相位、iGPU/内存频率和单次系统噪声。该列只能作为 smoke observation，正式论文结论必须来自每策略 N>=5 的均值、方差和 bootstrap CI。

`report_load_transition_cycles.py` 已扩展客户端 pre/post token/s 与 post/pre 比值的 bootstrap 汇总，并在新 artifact 声明 `stream_tokens` 时强制校验其等于 `completion_tokens`；旧 artifact 没有该字段时仍可聚合并标记为 `NA`。cycle report 同时保留 scheduler calls/s 和 client token/s，避免混淆内部控制速率与端到端吞吐。旧 none->compute20 N=5 兼容性验证通过；实验工具定向测试共 48 项通过。

当前 token-exact smoke 作为正式 dynamic cycle 1。保持当前 dynamic 引擎运行，继续 cycle 2；3 次 128-token warmup 会将上一轮结束的 iGPU 状态恢复至 CPU，并由 benchmark 初态检查验证：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v3-stable__engine-low__none-to-compute20__transition-token-exact-cycle02 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v3-client-segment-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-token-exact-cycle02-20260719
```

以完全相同参数完成 dynamic cycle 2--5 后生成 token-exact N=5；随后分别重启 packed CPU fixed 和 iGPU fixed，各执行 5 个 token-exact static cycles。实验顺序和所有 cycle 均保留，不能按结果快慢筛选样本。

### 2026-07-19：dynamic token-exact cycle 2 通过

结果位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-token-exact-cycle02-20260719/`。本轮仍满足 `stream_tokens=completion_tokens=stream_chunks=600`；transition 精确切分为 150/450 decode calls。Prefill/Decode/TTFT/TPOT/E2E 分别为 169.58 token/s、18.25 token/s、6168.06 ms、54.81 ms/token 和 38997.08 ms。

20 workers 在 43.22 ms ready。控制器在 launch 后 3 calls/840.06 ms、workers-ready 后 1 call/796.84 ms 直接切到 iGPU；无 exploration，正式窗口只有一次 switch，最终 execution/policy iGPU ratio 均为 1。全请求仍为 153 CPU calls 和 447 iGPU calls，与 cycle 1 完全一致。客户端 pre/post 分段分别为 29.585 和 16.191 token/s。

前两轮 token-exact 中间统计：

| 指标 | Cycle 1 | Cycle 2 | Mean | Sample stdev |
|---|---:|---:|---:|---:|
| Overall Decode token/s | 18.077 | 18.246 | 18.161 | 0.120 |
| E2E ms | 39296.23 | 38997.08 | 39146.66 | 211.53 |
| Client pre token/s | 29.596 | 29.585 | 29.591 | 0.008 |
| Client post token/s | 16.013 | 16.191 | 16.102 | 0.126 |
| Launch->iGPU ms | 621.52 | 840.06 | 730.79 | 154.54 |
| Ready->iGPU ms | 560.07 | 796.84 | 678.46 | 167.42 |

两轮客户端端点的变异很小，而响应毫秒数有可见波动；这与此前结论一致，即吞吐端点稳定、切换延迟受背景 worker 与引擎 CPU round 的调度相位影响。N=2 的 bootstrap 区间没有正式推断意义，只用于发现异常；继续原样采集 cycle 3--5。

cycle 3 命令：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v3-stable__engine-low__none-to-compute20__transition-token-exact-cycle03 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v3-client-segment-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-token-exact-cycle03-20260719
```

### 2026-07-19：dynamic token-exact cycle 3 通过

结果位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-token-exact-cycle03-20260719/`。`stream_tokens=completion_tokens=stream_chunks=600`，客户端 transition 仍在精确第 150 token 触发。Prefill/Decode/TTFT/TPOT/E2E 分别为 169.58 token/s、18.00 token/s、6168.06 ms、55.55 ms/token 和 39441.01 ms。

本轮 worker 启动出现中等长尾：all-ready 为 446.47 ms。控制器在 launch 后 5 calls/1062.45 ms、workers-ready 后 1 call/615.99 ms 切换 iGPU；无 exploration、一次 switch、终态 iGPU=1。客户端 pre/post 为 29.183/15.976 token/s。

scheduler telemetry 本轮为 599 个 decode execution calls，其中 signal 前 149、signal 后 450；前两轮为 600 和 150/450。客户端累计 token 仍严格为 600，差异来自第一个输出 token 可由 prefill 产生，以及 telemetry phase 边界事件相对请求窗口的归类相位，不是输出丢失。聚合与论文端到端结果以服务端 completion tokens 和客户端累计 token 时间戳为准；scheduler calls 仅用于控制响应解释。

前三轮中间统计：Overall Decode `18.108 ± 0.125 token/s`，Client pre `29.455 ± 0.235 token/s`，Client post `16.060 ± 0.115 token/s`，Ready->iGPU `657.63 ± 123.76 ms`（均为 mean ± sample stdev）。三轮 ready 后均只跨 1 个 scheduler execution call，说明毫秒方差主要是该受竞争 CPU round 本身的时长，不是控制器额外 hold-off。

cycle 4 命令：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v3-stable__engine-low__none-to-compute20__transition-token-exact-cycle04 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v3-client-segment-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-token-exact-cycle04-20260719
```

### 2026-07-19：dynamic token-exact cycle 4 通过

结果位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-token-exact-cycle04-20260719/`。`stream_tokens=completion_tokens=stream_chunks=600`，transition 为 150/450 calls。Prefill/Decode/TTFT/TPOT/E2E 分别为 170.11 token/s、18.77 token/s、6149.08 ms、53.28 ms/token 和 38064.22 ms。

20 workers 在 44.51 ms ready。控制器在 launch 后 3 calls/886.60 ms、ready 后 1 call/842.09 ms 切换；无 exploration、一次 switch、终态 iGPU=1。客户端 pre/post 为 29.616/16.739 token/s。

本轮整体 Decode 较高主要来自切换后的高负载 iGPU 端点更快，而不是控制器响应更早：ready->iGPU 的 842.09 ms 反而是当前四轮最大值。该样本是实际 runtime/内存频率和背景竞争相位波动，应完整保留，不能归因于调度器使同一 iGPU kernel 加速。

前四轮中间统计：Overall Decode `18.273 ± 0.345 token/s`，Client pre `29.495 ± 0.208 token/s`，Client post `16.230 ± 0.352 token/s`，Ready->iGPU `703.75 ± 136.81 ms`（mean ± sample stdev）。四轮均无 exploration、一次 CPU->iGPU switch、终态 iGPU=1，且 all-workers-ready 后都只跨 1 个 scheduler execution call。

cycle 5 命令：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v3-stable__engine-low__none-to-compute20__transition-token-exact-cycle05 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v3-client-segment-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-token-exact-cycle05-20260719
```

### 2026-07-19：dynamic v3 cycle 5 暴露 stale-arm 往返切换

结果位于 `artifacts/running-server-bench/dynamic-v3-stable-engine-low-none-to-compute20-transition-token-exact-cycle05-20260719/`。token 口径和请求本身有效：`stream_tokens=completion_tokens=stream_chunks=600`，客户端 pre/post 为 29.339/15.709 token/s；Prefill/Decode/TTFT/TPOT/E2E 为 169.96 token/s、17.76 token/s、6154.23 ms、56.30 ms/token 和 39877.96 ms。20 workers 在 52.53 ms ready，背景进程正常清理。

但本轮不能并入“稳定 v3 N=5”：正式窗口 switch count 从 12 增至 15，共 3 次设备切换。事件链为：

| Sequence | 动作 | CPU cost ms/row | iGPU cost ms/row | 说明 |
|---:|---|---:|---:|---|
| 5520 | CPU->iGPU | 0.1013 | 0.0850 | 负载启动后正确选择 iGPU |
| 5673 | iGPU->CPU | 0.1013（inactive stale） | 0.1215 | 用较早、较低负载下的 CPU cost 与当前 iGPU cost 直接比较 |
| 5677 | CPU->iGPU | 0.1346 | 0.1215 | 仅 4 个 CPU calls 后实测证明 CPU 仍更慢 |

第二、三次切换无 exploration。根因不是 `min_dwell=4` 太短，而是 contextual service cost 的适用条件缺失：CPU estimate 的采样负载与当前 iGPU estimate 的采样负载不同，却被当作同一上下文下的可比成本。单纯增大 dwell 只会延后错误回切，不能消除偏差。cycle 5 作为稳定性反例永久保留，不能删除或用额外“较好 cycle 5”替换；此前 cycle 1--4 仍是有效 v3 observations，但不再宣称为正式 N=5。

### 2026-07-19：dynamic v4 引入 load-context cost guard

控制器新增独立参数 `cost_load_match_delta`，默认 0.10。令两个 arm 的成本 EWMA 和对应采样负载 EWMA 为 `(C_c, L_c)` 与 `(C_g, L_g)`。CPU 活跃时，若 `C_g(1+m)<C_c`，仍可直接切 iGPU，以保持 low->high 的快速响应；iGPU 活跃时，历史 CPU cost 只有同时满足以下条件才允许直接回切：

\[
C_c(1+m)<C_g, \qquad |L_c-L_g|\le\delta_{match}.
\]

若负载上下文不可比，则拒绝用 inactive CPU estimate 直接决策，继续使用 iGPU，直至已有的 load-drop 或 periodic-staleness reprobe 获取新 CPU 样本。该约束与研究场景的非对称性一致：CPU contention 是主要变化源，空闲 iGPU 是稳定 fallback；同时仍允许负载可比时根据真实 service cost 返回 CPU。

配置链路新增 `KT_CPU_IGPU_DECODE_COST_LOAD_MATCH_DELTA=0.10`，覆盖 `GeneralMOEConfig`、Python 环境解析、pybind、C++ scheduler 和启动日志。telemetry 的原有前 8 个 debug 字段保持不变，并追加 `cpu_sample_load`、`igpu_sample_load`，使实验可直接审计上述公式是否满足；writer 仍兼容旧 7/8 字段 payload。

新增 C++ 状态机回归精确覆盖 cycle 5 模式：iGPU 活跃成本变差，但 inactive CPU 样本来自差异超过阈值的负载，控制器必须保持 iGPU 且 switch count 不增加。原有“初始校准返回 CPU”“CPU active cost 恶化直接切 iGPU”“load drop 后 CPU reprobe”测试全部保留。实验工具、状态机、启动器和 telemetry 共 56 项通过；Python scheduler 环境定向测试 3 项通过。

v4 修改涉及 C++ header、`GeneralMOEConfig` 和 pybind，必须重新编译安装并重启引擎。v4 验证从新的 smoke/N=5 系列重新开始，不能与 v3 cycle 1--5 混合计算置信区间。首轮应复现完全相同的 none->compute20 600-token transition，并检查：只有一次 CPU->iGPU switch；后续即使 iGPU cost 短时升高，只要 `|cpu_sample_load-igpu_sample_load|>0.10` 就不得直接回 CPU；终态 iGPU=1。

已执行 `./install.sh kt-kernel --no-clean`，`kt_kernel_ext` 完整编译至 100% 并成功重新安装。安装后的 `MOEConfig.cpu_igpu_decode_cost_load_match_delta` 为 0.1000000015；dynamic launcher dry-run 正确打印 `cost_load_delta=0.10`。构建前运行的旧 dynamic 服务在安装完成后已不在进程表中，未由调试脚本发送停止信号；当前也没有遗留 `cpu_background_load.py`。

启动 v4：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v4-load-context-compute20.jsonl \
SCHEDULER_TELEMETRY_LAYER=0 \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

服务就绪后执行 v4 smoke：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v4-load-context__engine-low__none-to-compute20__transition-token-exact-smoke \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v4-load-context-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v4-load-context-engine-low-none-to-compute20-transition-token-exact-smoke-20260719
```

### 2026-07-19：v4 首次 smoke 在 warmup 终态校验失败

失败 artifact 保留于 `artifacts/running-server-bench/dynamic-v4-load-context-engine-low-none-to-compute20-transition-token-exact-smoke-20260719/`。命令已经使用 `--warmup-output-tokens 128`，所以原始错误中的“increase to at least 128”不是根因，只是旧 benchmark 的通用提示。正式 measured sample 为 0，未启动背景 workers，也没有产生可纳入性能统计的数据。

新增 arm-load telemetry 使根因可以直接定位。新引擎的最初 CPU decode cost EWMA 依次约为 2.1986、1.7663、1.4204、1.1438 ms/row；iGPU 最初 4 个样本从 0.6350 衰减至 0.4044 ms/row。完成旧默认 4+4 samples 后，控制器认为 iGPU 更快并停止 CPU 采样；随后 iGPU 经数十 rounds 才收敛至约 0.09--0.10 ms/row，而 CPU estimate 永久停留在冷启动的 1.1438。此时继续增加 warmup token 无法恢复 CPU，因为 periodic reprobe interval 为 4096，且无负载环境没有 0.25 load-drop 事件。

这是独立于 stale-arm guard 的第二个问题：固定 4-sample 初始校准窗口无法覆盖大模型专家权重冷页和首次 kernel/runtime 开销。将初始 `calibration_samples` 从 4 提升到 32；CPU 与 iGPU 各执行 32 个初始 rounds，共 64 rounds，仍小于既有 128-token warmup。EWMA alpha=0.2 时，早期 4 个冷样本的剩余权重会在后续 28 个样本中衰减到 `0.8^28` 量级，从而不再主导设备选择。

新增 C++ 回归模拟每个 arm 前 4 个冷样本、后 28 个稳态样本，并验证 64-round 校准结束后正确返回 CPU。benchmark warmup 失败信息也改为输出 execution/policy ratio、exploration、两端 samples 和 cost，不再机械建议增加 token。状态机、launcher 和 benchmark 28 项通过，Python scheduler 配置 3 项通过。

已再次执行 `./install.sh kt-kernel --no-clean`，完整构建和安装成功。安装后的 `MOEConfig` 验证为 `calibration_samples=32`、`cost_load_match_delta=0.10`；launcher dry-run 同样显示 `calibration=32`。当前 PID 861611 是安装前启动的 4-sample 旧进程，必须手动停止并启动新进程。

停止旧引擎后，以新的实验标识启动 cal32 v4：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v4-cal32-load-context-compute20.jsonl \
SCHEDULER_TELEMETRY_LAYER=0 \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

服务就绪后执行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v4-cal32-load-context__engine-low__none-to-compute20__transition-token-exact-smoke \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v4-cal32-load-context-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v4-cal32-load-context-engine-low-none-to-compute20-transition-token-exact-smoke-20260719
```

warmup 验收除 CPU/CPU/non-exploring 外，还应看到两端 samples 均至少 32，CPU cost 已收敛到低于 iGPU cost的量级。正式请求仍要求精确 600 token、一次 CPU->iGPU switch、无额外回切、终态 iGPU=1。

### 2026-07-19：dynamic v4 cal32 smoke 通过，暴露局部 cost 响应盲区

结果位于 `artifacts/running-server-bench/dynamic-v4-cal32-load-context-engine-low-none-to-compute20-transition-token-exact-smoke-20260719/`。冷启动校准修正有效，warmup 终态校验通过；正式请求的 `stream_tokens=completion_tokens=stream_chunks=600`，背景负载在精确第 150 个输出 token 后启动。Prefill/Decode/TTFT/TPOT/E2E 分别为 167.57 token/s、17.72 token/s、6242.17 ms、56.45 ms/token 和 40053.80 ms。

本轮客户端 transition 前后端点分别为 29.833 和 15.616 token/s。与相同 none->compute20 静态反事实对比，低负载端点基本复现 packed CPU fixed 的 29.820 token/s；高负载端点相对 packed CPU fixed 的 5.105 token/s 提升 `3.059x`，相对 iGPU fixed 的 13.935 token/s 提升 `1.121x`。这说明同一请求内先使用 CPU、受竞争后切换 iGPU 的方向正确。不过当前仅 N=1，只能作为功能 smoke 和响应机制观测，不能作为正式置信区间或稳定收益结论。

控制状态验收全部通过：20 个 normal-priority workers 在 46.33 ms 内 ready；正式窗口无 exploration，switch count 只增加 1，终态 execution/policy iGPU ratio 均为 1。600 个 decode execution calls 中 199 次在 CPU、401 次在 iGPU。请求结束时 CPU/iGPU cost 样本负载约为 0.4507/0.8232，差值 0.3725 大于 `cost_load_match_delta=0.10`，控制器没有使用上下文不匹配的历史 CPU cost 回切，说明 v4 guard 阻止了 v3 cycle 5 的 stale-arm 往返模式。

本轮仍出现较长的高负载响应：从背景 launch 到第一次 iGPU execution 为 49 calls/4240.99 ms，从 all-workers-ready 到 iGPU 为 4194.66 ms。逐事件 telemetry 表明，负载启动后的前 48 个 CPU MoE rounds 中，CPU cost EWMA 大部分仍在约 0.04--0.05 ms/row，低于保存的 iGPU cost 0.09857 ms/row；第 49 个 CPU round 才出现可见阻塞，使 CPU cost EWMA 从 0.04377 跃升至 0.31794 ms/row，控制器在同一事件立即切换 iGPU。因此该延迟不是 dwell、exploration 或 guard 造成的，而是 layer-0 MoE 局部 service cost 对端到端 CPU pipeline 竞争存在观测盲区。

暂不依据单个长响应样本立即增加 token-level 控制器。先在同一 v4-cal32 引擎、相同 workload 和 transition 参数下完成 cycle 2--5，估计长响应的出现频率及方差；v4 数据必须单独聚合，不与 v3 cycle 混合。若多轮仍稳定出现数秒级响应，再引入客户端 token interval 或 scheduler execution interval 的 change-point 信号，并将其作为独立消融项。

cycle 2 命令：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v4-cal32-load-context__engine-low__none-to-compute20__transition-token-exact-cycle02 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v4-cal32-load-context-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v4-cal32-load-context-engine-low-none-to-compute20-transition-token-exact-cycle02-20260719
```

### 2026-07-19：dynamic v4 cal32 cycle 2 通过，响应长尾未复现

结果位于 `artifacts/running-server-bench/dynamic-v4-cal32-load-context-engine-low-none-to-compute20-transition-token-exact-cycle02-20260719/`。`stream_tokens=completion_tokens=stream_chunks=600`，正式请求无 exploration、只有一次 CPU->iGPU switch、终态 iGPU ratio=1。Prefill/Decode/TTFT/TPOT/E2E 分别为 170.42 token/s、18.62 token/s、6137.62 ms、53.72 ms/token 和 38315.87 ms。

客户端 transition 前后端点为 29.789/16.559 token/s。20 workers 在 46.82 ms ready；从 launch 到第一次 iGPU execution 为 3 calls/453.25 ms，从 workers-ready 到 iGPU 为 406.43 ms。600 个 decode calls 中 153 次 CPU、447 次 iGPU。与 smoke 的 49 calls/4194.66 ms 相比，本轮未复现数秒级响应长尾。

切换事件同样证明控制器检测到 service-cost 信号后没有额外等待。切换前一个 CPU round 的 CPU cost EWMA 为 0.03824 ms/row；下一个受阻 round 将其推至 0.24270 ms/row，高于保存的 iGPU 0.08628 ms/row，控制器在该事件立即切换。请求末 CPU/iGPU sample load 为 0.0187/0.8034，负载上下文显著不匹配，v4 guard 保持 iGPU 且没有回切。

前两轮仅作中间观测：客户端 pre 平均 29.811 token/s，post 平均 16.087 token/s；workers-ready 到 iGPU 分别为 4194.66 和 406.43 ms，均只发生一次切换。两轮响应时延差异很大，支持“竞争相位长尾”而非固定 hold-off 的判断，但 N=2 仍不足以估计概率，继续采集 cycle 3--5。

cycle 3 命令：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v4-cal32-load-context__engine-low__none-to-compute20__transition-token-exact-cycle03 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v4-cal32-load-context-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v4-cal32-load-context-engine-low-none-to-compute20-transition-token-exact-cycle03-20260719
```

### 2026-07-19：dynamic v4 cal32 cycle 3 通过，再次快速切换

结果位于 `artifacts/running-server-bench/dynamic-v4-cal32-load-context-engine-low-none-to-compute20-transition-token-exact-cycle03-20260719/`。`stream_tokens=completion_tokens=stream_chunks=600`，无 exploration、一次 CPU->iGPU switch、终态 iGPU ratio=1。Prefill/Decode/TTFT/TPOT/E2E 分别为 170.12 token/s、18.75 token/s、6148.78 ms、53.32 ms/token 和 38089.07 ms。

客户端 transition 前后端点为 29.621/16.722 token/s。20 workers 在 50.70 ms ready；从 launch 到 iGPU 为 2 calls/178.31 ms，从 workers-ready 到 iGPU 仅 127.61 ms。600 个 decode calls 中 152 次 CPU、448 次 iGPU。切换事件前 CPU cost 为 0.03716 ms/row；受竞争 round 将其提高至 0.09595 ms/row，此时保存的 iGPU cost 为 0.08492 ms/row，满足带 margin 的 iGPU 优选条件并在同一事件切换。

三轮中间统计：Overall Decode 平均 18.362 token/s，客户端 pre/post 平均 29.748/16.299 token/s。workers-ready 到 iGPU 分别为 4194.66、406.43 和 127.61 ms；三轮均无 exploration、只有一次 CPU->iGPU switch、没有回切。cycle 2--3 的快速响应已经表明 4.19 秒不是固定控制延迟，但当前长尾出现计数仍为 1/3，继续采集 cycle 4--5 后再决定是否增加额外信号。

cycle 4 命令：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v4-cal32-load-context__engine-low__none-to-compute20__transition-token-exact-cycle04 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v4-cal32-load-context-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v4-cal32-load-context-engine-low-none-to-compute20-transition-token-exact-cycle04-20260719
```

### 2026-07-19：dynamic v4 cal32 cycle 4 通过，出现中等切换延迟

结果位于 `artifacts/running-server-bench/dynamic-v4-cal32-load-context-engine-low-none-to-compute20-transition-token-exact-cycle04-20260719/`。`stream_tokens=completion_tokens=stream_chunks=600`，无 exploration、一次 CPU->iGPU switch、终态 iGPU ratio=1。Prefill/Decode/TTFT/TPOT/E2E 分别为 170.21 token/s、19.09 token/s、6145.50 ms、52.37 ms/token 和 37515.45 ms，是目前 v4 四轮中 Overall Decode 最高的一轮。

客户端 transition 前后端点为 29.673/17.079 token/s。20 workers 在 63.95 ms ready；从 launch 到 iGPU 为 18 calls/1346.08 ms，从 workers-ready 到 iGPU 为 1282.13 ms。600 个 decode calls 中 168 次 CPU、432 次 iGPU。正式窗口仍只有一次切换，未发生 v3 式往返。

本轮再次呈现局部 cost 延迟响应：切换前两个 CPU rounds 的 cost 为 0.04523 和 0.04481 ms/row，仍低于保存的 iGPU 0.08495 ms/row；第 18 个负载后 round 将 CPU cost 推至 0.19480 ms/row，控制器同一事件切换。与 smoke 的 49 calls/4194.66 ms 属于同一机制，但长尾程度较轻。

四轮中间统计（mean ± sample stdev）：Overall Decode `18.545 ± 0.588 token/s`，客户端 pre `29.729 ± 0.099 token/s`，post `16.494 ± 0.624 token/s`，workers-ready 到 iGPU `1502.71 ± 1860.82 ms`。四轮 post 端点范围仅为 15.616--17.079 token/s，而 ready->iGPU 范围达到 127.61--4194.66 ms；主要不稳定来源是切换检测时延，而不是切换后 iGPU 服务能力。继续采集最后一个 cycle 5 后形成 v4 N=5 汇总。

cycle 5 命令：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v4-cal32-load-context__engine-low__none-to-compute20__transition-token-exact-cycle05 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v4-cal32-load-context-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v4-cal32-load-context-engine-low-none-to-compute20-transition-token-exact-cycle05-20260719
```

### 2026-07-19：dynamic v4 cal32 cycle 5 暴露 load-drop reprobe 误触发

结果位于 `artifacts/running-server-bench/dynamic-v4-cal32-load-context-engine-low-none-to-compute20-transition-token-exact-cycle05-20260719/`。token 口径仍有效：`stream_tokens=completion_tokens=stream_chunks=600`。Prefill/Decode/TTFT/TPOT/E2E 为 170.15 token/s、15.21 token/s、6147.53 ms、65.74 ms/token 和 45528.36 ms；客户端 pre/post 为 29.661/13.098 token/s，post 明显低于前四轮。

本轮不能作为“稳定 v4 N=5”的第五个成功样本。正式窗口 switch count 增加 3，exploration fraction 为 0.0533，即 32/600 decode calls 被强制放回 CPU 探测。事件链为：

| Request call | Sequence | 动作 | CPU cost ms/row | iGPU cost ms/row | CPU/iGPU sample load | 说明 |
|---:|---:|---|---:|---:|---:|---|
| 154 | 4512 | CPU->iGPU | 0.47808 | 0.08614 | 0.0664/0.0014 | 背景负载启动后的正确切换 |
| 542 | 4900 | iGPU->CPU probe | reset | 0.08442 | 0/0.5802 | load-drop 条件误触发，开始 32-round reprobe |
| 574 | 4932 | CPU->iGPU | 0.50955 | 0.08442 | 0.4278/0.5802 | 高负载 CPU probe 明显更慢，返回 iGPU |

此次误触发不是 4096-round periodic staleness。iGPU sample-load reference 在高负载阶段曾达到约 0.833，随后 EWMA 暂降至约 0.580，刚好满足原条件 `L_ref-L_g>=0.25`；但 20 个背景 workers 始终在运行，当前负载仍远非低负载。原控制器把“相对峰值下降”错误等价为“背景负载撤销”。v4 cost load-context guard 只阻止 inactive CPU cost 直接回切，无法阻止 `should_reprobe_cpu()` 主动清空 CPU estimate 并强制 probe。

已生成五轮诊断汇总 `artifacts/running-server-bench/dynamic-v4-cal32-load-context-none-to-compute20-n5-diagnostic-20260719/`。N=5 的 Overall Decode 为 `17.878 ± 1.576 token/s`，bootstrap 95% CI `[16.420, 18.863]`；客户端 pre/post 为 `29.716 ± 0.091` 和 `15.815 ± 1.612 token/s`，post bootstrap 95% CI `[14.398, 16.832]`；ready->iGPU 为 `1329.40 ± 1657.46 ms`。该集合用于诊断 v4 的收益与失败模式，不能标记为最终稳定控制器结果。

v5 将 load-drop probe 改为相对下降与绝对负载双门控。令 iGPU 高负载参考、当前 EWMA 和可探测负载上限分别为 `L_ref`、`L_g`、`L_probe`：

\[
\operatorname{reprobe}_{drop}
= [L_{ref}-L_g\ge\Delta_L]
\land [L_g\le L_{probe}]
\land [G=0],
\]

其中 `G` 表示 phase-boundary grace 尚未结束。`L_probe` 复用 decode high-load threshold 0.20；因此 cycle 5 的 `0.833->0.580` 不会 probe，而真实 compute20->none 在 load EWMA 降到 0.20 以下后仍能获取新 CPU 样本。periodic staleness probe 保持独立，不受该门控影响。

### 2026-07-19：dynamic v5 实现 reprobe absolute-load guard

`ServiceCostConfig` 新增内部参数 `load_reprobe_max`，由现有 `cpu_igpu_decode_load_high` 注入，默认 0.20，不增加新的用户环境变量。load-drop reprobe 现在同时要求 phase grace 结束、相对 reference 下降至少 0.25、当前 iGPU sample-load 不高于 0.20。periodic staleness 仍独立触发，避免绝对门控导致永久不再校准 CPU。

telemetry 在兼容旧 7/8/10 字段 payload 的基础上追加为 12 字段：`igpu_reference_load` 和 `reprobe_reason`。reason 取值 0/1/2，分别表示无 probe、load-drop probe、periodic probe；probe 开始时保留触发 reference load，因而可以直接审计 `L_ref-L_g` 与 `L_g<=L_probe`。启动器日志新增 `load_probe_max=0.20`。

状态机新增两个关键反例：`0.8->0.5` 的相对下降虽然超过阈值，但当前仍高于 0.20，必须保持 iGPU 且不 exploration；继续降到 0.0 后必须触发 reason=1 的 CPU probe。另有短 interval 用例验证 periodic probe 仍触发 reason=2。状态机、telemetry、启动器定向测试 13 项通过；完整实验工具 56 项通过；加载 oneAPI runtime 后 Python scheduler 配置 3 项通过。`bash -n` 和 `git diff --check` 均通过。

已执行 `./install.sh kt-kernel --no-clean`，最终 C++ extension 编译并安装成功。安装后核验：writer 接受 12 字段且包含 `reprobe_reason`；`MOEConfig.cpu_igpu_decode_load_high=0.20`、calibration samples=32；launcher dry-run 打印 `load_probe_max=0.20`。当前 PID 870689 启动于最终安装之前，仍加载 v4 代码，必须手动停止并启动新进程。

停止旧引擎后启动 v5：

```bash
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v5-reprobe-ceiling-compute20.jsonl \
SCHEDULER_TELEMETRY_LAYER=0 \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

启动日志必须包含 `load_probe_max=0.20`。服务就绪后执行 v5 smoke：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v5-reprobe-ceiling__engine-low__none-to-compute20__transition-token-exact-smoke \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v5-reprobe-ceiling-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-none-to-compute20-transition-token-exact-smoke-20260719
```

验收要求：精确 600 token；正式窗口只有一次 CPU->iGPU switch；exploration=0；终态 iGPU=1；即使 `L_ref-L_g>=0.25`，只要 `L_g>0.20` 就不得出现 reason=1。none->compute20 稳定性验证后，还必须补做 compute20->none 方向，确认真实低负载下 reason=1 的 CPU reprobe 未被阻断。

### 2026-07-19：dynamic v5 reprobe-ceiling smoke 行为通过

结果位于 `artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-none-to-compute20-transition-token-exact-smoke-20260719/`。`stream_tokens=completion_tokens=stream_chunks=600`，正式窗口 switch count 只增加 1、exploration=0、全部 `reprobe_reason=0`、终态 iGPU ratio=1。新增 12-field telemetry 已在真实引擎中正常落盘。Prefill/Decode/TTFT/TPOT/E2E 为 161.86 token/s、16.65 token/s、6462.23 ms、60.06 ms/token 和 42439.08 ms。

客户端 pre/post 为 29.025/14.590 token/s。20 workers 在 84.23 ms ready；从 launch 到 iGPU 为 15 calls/1941.23 ms，从 workers-ready 到 iGPU 为 1857.00 ms。600 个 decode calls 中 165 次 CPU、435 次 iGPU。切换后的纯 iGPU telemetry rate 约 15.06 calls/s，低于 v4 前四轮的 16.31--17.28 calls/s；同时 prefill 与 pre-transition CPU 也比此前偏慢，当前 N=1 更符合整机频率/竞争状态波动，不能归因于 v5 新增的一个 reprobe 条件判断。

本轮 `igpu_reference_load` 最终约 0.861，iGPU sample load 最终约 0.852；正式窗口没有出现 `L_ref-L_g>=0.25` 的事件。因此本轮证明 v5 已正确加载且没有无故回探，但没有在端到端环境中实际命中“相对下降足够大、绝对负载仍高”的反事实门控分支。继续使用同一 v5 引擎采集 cycle 2--5，观察该分支和性能方差；不得与 v4 聚合。

v5 cycle 2 命令：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v5-reprobe-ceiling__engine-low__none-to-compute20__transition-token-exact-cycle02 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v5-reprobe-ceiling-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-none-to-compute20-transition-token-exact-cycle02-20260719
```

### 2026-07-19：dynamic v5 cycle 2 行为通过，iGPU 稳态仍偏慢

结果位于 `artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-none-to-compute20-transition-token-exact-cycle02-20260719/`。`stream_tokens=completion_tokens=stream_chunks=600`；正式窗口一次 CPU->iGPU switch、exploration=0、全部 reprobe reason=0、终态 iGPU=1。Prefill/Decode/TTFT/TPOT/E2E 为 170.41 token/s、16.38 token/s、6138.22 ms、61.04 ms/token 和 42703.86 ms。

客户端 pre/post 为 28.380/14.370 token/s。20 workers 本轮 ready 较慢，为 623.30 ms；从 launch 到 iGPU 为 3 calls/794.86 ms，但从 all-ready 到 iGPU 仅 171.56 ms，说明控制器检测响应很快，进程创建/调度占据了大部分 launch 延迟。600 个 decode calls 中 154 次 CPU、446 次 iGPU。

切换后纯 iGPU telemetry rate 为 14.64 calls/s，仍低于 v4 前四轮的 16.31--17.28 calls/s；这解释了整体 Decode 偏低，且与错误 CPU reprobe 无关。v5 前两轮 Overall Decode 平均 16.516 token/s，客户端 pre/post 平均 28.702/14.480 token/s，ready->iGPU 分别为 1857.00/171.56 ms。两轮均只有一次切换且无 exploration，继续采集 cycle 3--5 判断 iGPU 稳态下降是否持续。

v5 cycle 3 命令：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v5-reprobe-ceiling__engine-low__none-to-compute20__transition-token-exact-cycle03 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v5-reprobe-ceiling-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-none-to-compute20-transition-token-exact-cycle03-20260719
```

### 2026-07-19：dynamic v5 cycle 3 行为通过，iGPU 稳态回升

结果位于 `artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-none-to-compute20-transition-token-exact-cycle03-20260719/`。`stream_tokens=completion_tokens=stream_chunks=600`；正式窗口一次 CPU->iGPU switch、exploration=0、全部 reprobe reason=0、终态 iGPU=1。Prefill/Decode/TTFT/TPOT/E2E 为 170.12 token/s、17.52 token/s、6148.68 ms、57.07 ms/token 和 40333.10 ms。

客户端 pre/post 为 29.102/15.483 token/s。20 workers 在 69.70 ms ready；从 launch 到 iGPU 为 8 calls/1022.15 ms，从 all-ready 到 iGPU 为 952.44 ms。600 个 decode calls 中 158 次 CPU、442 次 iGPU。切换后的纯 iGPU telemetry rate 回升到 15.78 calls/s，高于 v5 前两轮的 15.06/14.64 calls/s，但仍低于 v4 前四轮范围。

V5 三轮中间统计（mean ± sample stdev）：Overall Decode `16.851 ± 0.597 token/s`，客户端 pre `28.836 ± 0.397 token/s`，post `14.814 ± 0.589 token/s`，ready->iGPU `993.67 ± 843.48 ms`。三轮均一次切换、零 exploration、无 reprobe；当前 post 均值比静态 iGPU fixed 的 13.935 token/s 高约 6.3%。继续 cycle 4--5 后再做正式聚合。

v5 cycle 4 命令：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v5-reprobe-ceiling__engine-low__none-to-compute20__transition-token-exact-cycle04 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v5-reprobe-ceiling-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-none-to-compute20-transition-token-exact-cycle04-20260719
```

### 2026-07-19：dynamic v5 cycle 4 行为通过，稳态重复性良好

结果位于 `artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-none-to-compute20-transition-token-exact-cycle04-20260719/`。`stream_tokens=completion_tokens=stream_chunks=600`；正式窗口一次 CPU->iGPU switch、exploration=0、全部 reprobe reason=0、终态 iGPU=1。Prefill/Decode/TTFT/TPOT/E2E 为 169.86 token/s、17.50 token/s、6157.96 ms、57.14 ms/token 和 40386.42 ms，与 cycle 3 几乎一致。

客户端 pre/post 为 28.795/15.488 token/s。20 workers 在 47.41 ms ready；从 launch 到 iGPU 为 29 calls/2286.48 ms，从 all-ready 到 iGPU 为 2239.06 ms。600 个 decode calls 中 179 次 CPU、421 次 iGPU。切换后的纯 iGPU telemetry rate 为 15.75 calls/s，与 cycle 3 的 15.78 calls/s 接近；检测延迟虽有长尾，但没有错误回切或 probe。

V5 四轮中间统计（mean ± sample stdev）：Overall Decode `17.013 ± 0.585 token/s`，客户端 pre `28.825 ± 0.325 token/s`，post `14.983 ± 0.588 token/s`，ready->iGPU `1305.02 ± 928.47 ms`。四轮均一次切换、零 exploration、无 reprobe；post 均值比静态 iGPU fixed 的 13.935 token/s 高约 7.5%。继续最后一个 cycle 5，之后生成 V5 独立 N=5 聚合并执行 compute20->none 反向恢复验证。

v5 cycle 5 命令：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v5-reprobe-ceiling__engine-low__none-to-compute20__transition-token-exact-cycle05 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v5-reprobe-ceiling-compute20.jsonl \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-none-to-compute20-transition-token-exact-cycle05-20260719
```

### 2026-07-19：dynamic v5 none->compute20 N=5 稳定性汇总

cycle 5 位于 `artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-none-to-compute20-transition-token-exact-cycle05-20260719/`。`stream_tokens=completion_tokens=stream_chunks=600`；正式窗口一次 CPU->iGPU switch、exploration=0、全部 reprobe reason=0、终态 iGPU=1。Prefill/Decode/TTFT/TPOT/E2E 为 170.18 token/s、18.07 token/s、6146.51 ms、55.33 ms/token 和 39290.76 ms。客户端 pre/post 为 29.474/16.021 token/s，ready->iGPU 为 13 calls/832.80 ms，纯 iGPU 稳态为 16.05 calls/s。

五轮已由 `report_load_transition_cycles.py` 使用 5000 次 percentile bootstrap、seed 20260719 独立聚合，结果位于 `artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-none-to-compute20-n5-20260719/`。所有 cycle 均为精确 token 时间戳、一次切换、零 exploration，没有复现 V4 cycle 5 的强制 CPU reprobe。

| Metric | Mean | Sample stdev | Bootstrap 95% CI |
|---|---:|---:|---:|
| Prefill token/s | 168.4857 | 3.7070 | [165.1720, 170.2578] |
| Decode token/s | 17.2253 | 0.6937 | [16.6634, 17.7381] |
| E2E ms | 41030.65 | 1475.82 | [39926.83, 42134.46] |
| Client pre token/s | 28.9552 | 0.4040 | [28.6378, 29.2640] |
| Client post token/s | 15.1904 | 0.6887 | [14.6377, 15.7002] |
| First iGPU delay calls | 13.6000 | 9.7877 | [6.4000, 22.0000] |
| Ready->iGPU ms | 1210.57 | 831.35 | [592.23, 1828.91] |

相对现有 N=1 静态 smoke，V5 平均 Decode 相比 packed CPU fixed 提升 167.8%、相比 iGPU fixed 提升 17.9%；post 端点分别提升 197.6% 和 9.0%；E2E 分别减少 58.7% 和 23.0%。pre 端点相对 CPU fixed 低 2.9%，但相对 iGPU fixed 高 69.4%，符合低负载优先 CPU 的设计。上述比较足以用于开发验收，但静态基线当前仅 N=1，论文正式表格必须在相同运行时段补齐 CPU fixed/iGPU fixed N>=5，不能把这些比例直接当作最终置信结论。

下一步验证 V5 compute20->none。先在独立终端启动背景负载：

```bash
python kt-kernel/bench/cpu_background_load.py \
  --kind compute \
  --workers 20 \
  --affinity free \
  --nice 0
```

记录 ready JSON 中的 `parent_pid`，保持该终端运行。随后将 `<PARENT_PID>` 替换为实际值执行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v5-reprobe-ceiling__engine-low__compute20-to-none__transition-token-exact-smoke \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v5-reprobe-ceiling-compute20.jsonl \
  --stop-background-pid <PARENT_PID> \
  --stop-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-compute20-to-none-transition-token-exact-smoke-20260719
```

反向 smoke 验收要求：signal 前 iGPU=1；负载 EWMA 降到 0.20 以下后出现 `reprobe_reason=1`；CPU exploration 恰好 32 calls；正式窗口只发生一次 iGPU->CPU switch；最终 CPU ratio=1。该测试将直接证明 absolute-load guard 只抑制高负载误探测，不会阻断真实负载消失后的恢复。

### 2026-07-19：dynamic v5 compute20->none 反向 smoke 通过

结果位于 `artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-compute20-to-none-transition-token-exact-smoke-20260719/`。`stream_tokens=completion_tokens=stream_chunks=600`；benchmark 在精确第 150 个客户端输出 token 向 PID 901610 发送停止信号，20 个 workers 全部退出。Prefill/Decode/TTFT/TPOT/E2E 为 146.48 token/s、22.90 token/s、7141.10 ms、43.67 ms/token 和 33301.27 ms；客户端 pre/post 为 14.462/28.378 token/s。

V5 完成了预期反向状态链：signal 前 iGPU=1；signal 后 14 calls/731.54 ms 首次执行 CPU，此时 `igpu_reference_load=0.8706`、`igpu_sample_load=0.1685<load_reprobe_max=0.20`，`reprobe_reason=1`；随后精确执行 32-call CPU exploration，在第 46 call/1849.65 ms 结束探索并保持 CPU。正式窗口 switch count 仅增加 1，最终 iGPU ratio=0；全请求 164 iGPU calls、436 CPU calls，无 periodic reason=2。

CPU cost 在 exploration 后收敛到约 0.03720 ms/row，低于保存的 iGPU 0.08530 ms/row，因此最终选择 CPU 仍由实测 service cost 支持。请求结束 CPU cost 约 0.03740 ms/row、CPU sample load 接近 0；低负载 post 端点 28.378 token/s 接近 V5 none->compute20 N=5 的低负载 pre 均值 28.955 token/s。

绝对负载门控带来可量化的恢复延迟代价。V3 compute20->none N=5 首次 CPU/settle 平均为 7.6 calls/399.72 ms 和 39.6 calls/1504.42 ms；V5 smoke 为 14 calls/731.54 ms 和 46 calls/1849.65 ms，约增加 6.4 calls/332 ms 的检测等待以及 345 ms 的 settle 时间。该代价换取了高负载波动时不误 probe 的稳定性；当前整体 Decode 仍为 22.90 token/s，说明端到端收益保留。由于 V5 反向当前仅 N=1，继续 cycle 2--5 后再报告正式分布。

每个反向 cycle 都需重新启动背景父进程：

```bash
python kt-kernel/bench/cpu_background_load.py \
  --kind compute \
  --workers 20 \
  --affinity free \
  --nice 0
```

取得新的 `parent_pid` 后，cycle 2 使用：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v5-reprobe-ceiling__engine-low__compute20-to-none__transition-token-exact-cycle02 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v5-reprobe-ceiling-compute20.jsonl \
  --stop-background-pid <PARENT_PID> \
  --stop-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-compute20-to-none-transition-token-exact-cycle02-20260719
```

### 2026-07-19：dynamic v5 compute20->none cycle 2 通过

结果位于 `artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-compute20-to-none-transition-token-exact-cycle02-20260719/`。`stream_tokens=completion_tokens=stream_chunks=600`；Prefill/Decode/TTFT/TPOT/E2E 为 148.17 token/s、24.20 token/s、7059.45 ms、41.33 ms/token 和 31816.21 ms；客户端 pre/post 为 16.198/28.924 token/s。

signal 后 15 calls/741.81 ms 首次执行 CPU，触发点 `igpu_reference_load=0.8711`、`igpu_sample_load=0.1811<0.20`、reason=1；随后精确 32-call exploration，第 47 call/1823.91 ms settle。正式窗口仅一次 iGPU->CPU switch，最终 CPU=1，全请求 165 iGPU calls、435 CPU calls，无 periodic probe。

前两轮中间统计：Overall Decode 平均 23.546 token/s，客户端 high/low 端点平均 15.330/28.651 token/s；first CPU 平均 14.5 calls/736.67 ms，settle 平均 46.5 calls/1836.78 ms。两轮状态序列仅相差 1 call，绝对门控触发负载分别为 0.168/0.181，均严格低于 0.20。继续 cycle 3--5。

cycle 3 仍先启动新的 20-worker 背景父进程，随后将新 PID 代入：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v5-reprobe-ceiling__engine-low__compute20-to-none__transition-token-exact-cycle03 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v5-reprobe-ceiling-compute20.jsonl \
  --stop-background-pid <PARENT_PID> \
  --stop-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-compute20-to-none-transition-token-exact-cycle03-20260719
```

### 2026-07-19：dynamic v5 compute20->none cycle 3 通过

结果位于 `artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-compute20-to-none-transition-token-exact-cycle03-20260719/`。Prefill/Decode/TTFT/TPOT/E2E 为 142.28 token/s、23.57 token/s、7351.56 ms、42.43 ms/token 和 32764.27 ms；客户端 pre/post 为 15.512/28.468 token/s。

signal 后 12 calls/629.74 ms 首次执行 CPU，触发点 `igpu_reference_load=0.8744`、`igpu_sample_load=0.1687<0.20`、reason=1；精确 32-call exploration 后在第 44 call/1725.05 ms settle。正式窗口一次 iGPU->CPU switch、最终 CPU=1，全请求 163 iGPU calls、437 CPU calls，无 periodic probe。

三轮中间统计（mean ± sample stdev）：Overall Decode `23.555 ± 0.649 token/s`，客户端 high/low 端点 `15.391 ± 0.874` / `28.590 ± 0.293 token/s`；first CPU `13.667 ± 1.528 calls`、`701.03 ± 61.95 ms`；settle `45.667 ± 1.528 calls`、`1799.54 ± 65.78 ms`。三轮均 reason=1、32-call exploration、一次切换，继续 cycle 4--5。

cycle 4 仍需新的背景 PID，命令模板：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v5-reprobe-ceiling__engine-low__compute20-to-none__transition-token-exact-cycle04 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v5-reprobe-ceiling-compute20.jsonl \
  --stop-background-pid <PARENT_PID> \
  --stop-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-compute20-to-none-transition-token-exact-cycle04-20260719
```

### 2026-07-19：dynamic v5 compute20->none cycle 4 通过

结果位于 `artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-compute20-to-none-transition-token-exact-cycle04-20260719/`。Prefill/Decode/TTFT/TPOT/E2E 为 146.60 token/s、24.07 token/s、7135.12 ms、41.55 ms/token 和 32022.05 ms；客户端 pre/post 为 16.076/28.812 token/s。

signal 后 15 calls/786.43 ms 首次执行 CPU，触发点 `igpu_reference_load=0.8350`、`igpu_sample_load=0.1913<0.20`、reason=1；精确 32-call exploration 后在第 47 call/1872.48 ms settle。正式窗口一次切换、最终 CPU=1，全请求 165 iGPU calls、435 CPU calls，无 periodic probe。

四轮中间统计（mean ± sample stdev）：Overall Decode `23.683 ± 0.589 token/s`，客户端 high/low 端点 `15.562 ± 0.792` / `28.645 ± 0.264 token/s`；first CPU `14.000 ± 1.414 calls`、`722.38 ± 66.20 ms`；settle `46.000 ± 1.414 calls`、`1817.77 ± 64.92 ms`。完成 cycle 5 后生成反向 V5 N=5 bootstrap 汇总。

cycle 5 需最后一个新背景 PID，命令模板：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v5-reprobe-ceiling__engine-low__compute20-to-none__transition-token-exact-cycle05 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v5-reprobe-ceiling-compute20.jsonl \
  --stop-background-pid <PARENT_PID> \
  --stop-background-after-output-tokens 150 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-compute20-to-none-transition-token-exact-cycle05-20260719
```

### 2026-07-19：dynamic v5 compute20->none N=5 正式开发汇总

cycle 5 位于 `artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-engine-low-compute20-to-none-transition-token-exact-cycle05-20260719/`。Prefill/Decode/TTFT/TPOT/E2E 为 151.27 token/s、23.05 token/s、6915.00 ms、43.39 ms/token 和 32903.65 ms；客户端 pre/post 为 14.709/28.375 token/s。signal 后 14 calls/739.72 ms 在 `igpu_sample_load=0.1714<0.20` 触发 reason=1，精确 32-call exploration，第 46 call/1858.90 ms settle；正式窗口一次切换、最终 CPU=1。

五轮使用 10000 次 percentile bootstrap、seed 20260719 聚合，结果位于 `artifacts/running-server-bench/dynamic-v5-reprobe-ceiling-compute20-to-none-n5-20260719/`。每轮均使用精确 token 时间戳，reason=1、32 exploration calls、一次 iGPU->CPU switch、最终 CPU=1，无 reason=2。

| Metric | Mean | Sample stdev | Bootstrap 95% CI |
|---|---:|---:|---:|
| Prefill token/s | 146.9586 | 3.2497 | [144.3234, 149.3988] |
| Decode token/s | 23.5563 | 0.5838 | [23.0926, 24.0200] |
| E2E ms | 32561.49 | 622.88 | [32074.87, 33034.82] |
| Client high-load pre token/s | 15.3916 | 0.7846 | [14.7711, 16.0121] |
| Client low-load post token/s | 28.5914 | 0.2583 | [28.3948, 28.7922] |
| First CPU delay calls | 14.0000 | 1.2247 | [13.0000, 14.8000] |
| First CPU delay ms | 725.85 | 57.85 | [673.73, 766.53] |
| Settle CPU delay calls | 46.0000 | 1.2247 | [45.0000, 46.8000] |
| Settle CPU delay ms | 1826.00 | 59.16 | [1774.31, 1862.48] |
| Low-load CPU calls/s | 29.0287 | 0.2765 | [28.8210, 29.2379] |

与 V3 compute20->none N=5 相比，V5 first CPU 从 7.6 calls/399.72 ms 增至 14.0 calls/725.85 ms，分别增加 84.2%/81.6%；settle 从 39.6 calls/1504.42 ms 增至 46.0 calls/1826.00 ms，增加 16.2%/21.4%；整体 Decode 下降 5.4%，E2E 增加 5.0%。切回后的低负载 CPU rate 仅变化 0.04%，说明差异来自绝对门控的额外检测等待，不是 CPU 后端退化。

该结果给出清晰的稳定性-响应速度权衡：V3 恢复更快，但 V4 已证明仅依赖相对 load drop 会在持续高负载中误 probe；V5 要求当前负载低于 0.20，牺牲约 326 ms first-response 和 322 ms settle 均值，换取 none->compute20 N=5 中零错误 probe。V5 双向 N=5 均通过，候选控制器开发验收完成，后续不再根据单个样本改算法。

下一阶段进入论文正式实验。优先补齐同一时间段、相同 transition workload 下 packed CPU fixed 和 iGPU fixed 的 N>=5 静态反事实，再进行负载等级、prompt/output 长度和消融矩阵。当前静态基线仅 N=1，不能与动态 N=5 直接形成最终统计显著性结论。

为减少静态反事实的手工循环，`bench_running_server.py` 现允许且仅允许 `start-background + transition-static-baseline` 使用多 repetitions。`ManagedComputeBackground` 每轮 start 前重置 launch/ready/metadata/error 状态，每个 measured request 在第 150 token 创建一组全新的 20 workers，并在请求末自动清理。动态 transition 和使用外部 PID 的 high->low 模式仍强制单 repetition，防止破坏初始状态语义。benchmark/report 定向测试 25 项通过，完整实验工具 58 项通过，`py_compile` 与 `git diff --check` 通过；该工具修改不需要重新编译引擎。

论文静态基线先测 packed CPU fixed。手动停止当前 dynamic 引擎并启动：

```bash
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh packed-cpu-fixed
```

服务就绪后一次采集 N=5：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label packed-cpu-fixed__engine-low__none-to-compute20__transition-token-exact-n5 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 5 \
  --seed 20260719 \
  --bootstrap-samples 5000 \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --transition-static-baseline \
  --fail-fast \
  --output-dir artifacts/running-server-bench/packed-cpu-fixed-engine-low-none-to-compute20-transition-token-exact-n5-20260719
```

该测试在高负载 CPU 段较慢，预计耗时数分钟，不能因单轮变慢提前终止。五轮完成后停止 packed CPU fixed，引擎模式改为 `igpu-fixed`，使用同样参数采集 iGPU N=5。

### 2026-07-19：packed CPU fixed none->compute20 静态基线 N=5

结果位于 `artifacts/running-server-bench/packed-cpu-fixed-engine-low-none-to-compute20-transition-token-exact-n5-20260719/`。五轮均为精确 `600/600` token 时间戳，每轮在第 150 个输出 token 创建一组新的 20-worker、nice=0、free-affinity 计算负载，请求结束后均成功清理。该组与 V5 none->compute20 N=5 使用相同 engine nice、workload、transition token、warmup 和随机种子，可作为同协议静态反事实。

| Metric | Mean | Sample stdev | Bootstrap 95% CI |
|---|---:|---:|---:|
| Prefill token/s | 169.4313 | 0.7179 | [168.7702, 169.8872] |
| Decode token/s | 8.7935 | 0.8651 | [8.1983, 9.4434] |
| TTFT ms | 6173.68 | 26.28 | [6157.46, 6197.46] |
| TPOT ms/token | 114.57 | 10.90 | [105.00, 122.37] |
| E2E ms | 74804.21 | 6543.36 | [69196.26, 79473.55] |
| Client low-load pre token/s | 29.2159 | 0.0975 | [29.1445, 29.2877] |
| Client high-load post token/s | 7.1458 | 0.7633 | [6.6208, 7.8342] |
| Background ready ms | 207.29 | 356.00 | [45.51, 526.13] |

CPU request-window busy fraction 为 0.9141，CPU PSI some 为 0.3694，说明 compute20 确实形成持续 CPU 争用。五轮 post 端点均明显低于 pre；其中一轮 background ready 为 844.10 ms，其余约 43--52 ms。分段 post 从负载 launch token 开始计时，因此该长尾会让该轮 CPU 高负载区间稍短，并使 packed CPU 结果偏乐观，而不会夸大动态调度收益。V5 的同协议五轮也包含 623.30 ms 的 ready 长尾，启动扰动并非只出现在一个策略中。

使用两组各五个原始样本进行独立重采样、10000 次 percentile bootstrap、seed 20260719，V5 相对 packed CPU fixed 的效应量为：

| Comparison | Point estimate | Bootstrap 95% CI |
|---|---:|---:|
| Prefill token/s change | -0.56% | [-2.59%, +0.70%] |
| Overall Decode token/s improvement | +95.89% | [+79.88%, +112.73%] |
| E2E latency reduction | 45.15% | [40.66%, 48.88%] |
| Client low-load pre token/s change | -0.89% | [-2.03%, +0.19%] |
| Client high-load post token/s improvement | +112.58% | [+94.42%, +132.43%] |

Prefill 和低负载 pre 的 CI 跨 0，不能宣称存在显著差异；这正好表明 V5 基本保留 CPU 空闲时的性能。Decode、post 和 E2E 的 CI 均不跨 0，当前数据已支持“动态调度在负载突增时显著优于固定 CPU”这一开发阶段结论。N=5 仍属于小样本，正式论文可在冻结实验矩阵后增加重复次数。

`bench_running_server.py` 已将 `background_ready_delay_ms`、`transition_client_pre_tps` 和 `transition_client_post_tps` 纳入逐 workload bootstrap 汇总；报告的 Background Load Transition 区域现在显示所有 repetition 的均值和 CI，不再误显示 manifest 中第一轮样本。原始 `samples.jsonl` 未改写，仅从原始五轮数据重建了该产物的 `summary.csv` 和 `report.md`。

新增两项报告聚合回归测试后，完整实验工具测试集共 60 项通过，`py_compile` 通过；该修改只影响 benchmark 离线汇总，不需要重编译或重启引擎。

下一步采集同协议 iGPU fixed N=5。先停止当前 packed CPU fixed 引擎，然后启动：

```bash
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh igpu-fixed
```

服务就绪后运行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label igpu-fixed__engine-low__none-to-compute20__transition-token-exact-n5 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 5 \
  --seed 20260719 \
  --bootstrap-samples 5000 \
  --start-compute-background-workers 20 \
  --start-background-after-output-tokens 150 \
  --transition-static-baseline \
  --fail-fast \
  --output-dir artifacts/running-server-bench/igpu-fixed-engine-low-none-to-compute20-transition-token-exact-n5-20260719
```

iGPU fixed 完成后即可形成同一实验协议下 packed CPU fixed、iGPU fixed、V5 dynamic 三策略的 N=5 主对比，分别回答固定 CPU 的竞争退化、固定 iGPU 的空闲态代价，以及动态策略能否同时接近两个阶段的较优后端。

### 2026-07-19：iGPU fixed none->compute20 静态基线 N=5

结果位于 `artifacts/running-server-bench/igpu-fixed-engine-low-none-to-compute20-transition-token-exact-n5-20260719/`。启动环境已从运行进程确认：`KT_CPU_IGPU_POLICY=fixed`、`KT_CPU_IGPU_RATIO=1`、`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`。五轮均为精确 `600/600` token 时间戳，每轮背景负载均成功启动和清理，无残留 `cpu_background_load.py`。

| Metric | Mean | Sample stdev | Bootstrap 95% CI |
|---|---:|---:|---:|
| Prefill token/s | 83.3557 | 5.1090 | [79.9216, 87.7264] |
| Decode token/s | 13.9413 | 0.6177 | [13.4781, 14.4216] |
| TTFT ms | 12584.78 | 737.91 | [11995.63, 13095.97] |
| TPOT ms/token | 71.84 | 3.17 | [69.33, 74.27] |
| E2E ms | 55620.41 | 1762.50 | [54365.93, 57159.09] |
| Client low-load pre token/s | 16.3118 | 1.2286 | [15.2851, 17.1023] |
| Client high-load post token/s | 13.3315 | 0.7942 | [12.7218, 13.8819] |
| Background ready ms | 77.61 | 15.97 | [64.08, 89.31] |

iGPU fixed 相比 packed CPU fixed 明显减轻高负载 decode 退化，但固定使用 iGPU 的代价同样清晰：Prefill 约为 packed CPU 的一半，低负载 pre 也只有约 56%。这说明动态策略的目标不是简单地永久迁移到 iGPU，而是保留低负载 CPU 路径并在 CPU 竞争出现后切换。

### 2026-07-19：none->compute20 三策略 N=5 对比

新增 `kt-kernel/bench/report_transition_strategy_comparison.py`，可同时读取 fixed benchmark 的 `samples.jsonl` 与 dynamic 聚合产物的 `cycles.csv`，强制检查精确 token 时间戳，并输出规范化样本、逐策略 bootstrap 统计和候选策略相对固定基线的独立重采样效应量。真实结果位于 `artifacts/running-server-bench/three-strategy-none-to-compute20-n5-20260719/`，使用 10000 次 percentile bootstrap、seed 20260719。

| Strategy | N | Prefill token/s | Decode token/s | TTFT ms | E2E ms | Client pre token/s | Client post token/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| packed CPU fixed | 5 | 169.431 | 8.794 | 6173.68 | 74804.21 | 29.216 | 7.146 |
| iGPU fixed | 5 | 83.356 | 13.941 | 12584.78 | 55620.41 | 16.312 | 13.331 |
| dynamic V5 | 5 | 168.486 | 17.225 | 6210.72 | 41030.65 | 28.955 | 15.190 |

V5 相对 packed CPU fixed：

| Metric | Effect | Independent bootstrap 95% CI |
|---|---:|---:|
| Prefill token/s | -0.56% | [-2.57%, +0.69%] |
| Decode token/s | +95.89% | [+79.54%, +112.18%] |
| TTFT reduction | -0.60% | [-2.73%, +0.67%] |
| TPOT reduction | +49.26% | [+44.76%, +53.00%] |
| E2E reduction | +45.15% | [+40.61%, +48.87%] |
| Client low-load pre token/s | -0.89% | [-2.03%, +0.18%] |
| Client high-load post token/s | +112.58% | [+94.30%, +131.75%] |

V5 相对 iGPU fixed：

| Metric | Effect | Independent bootstrap 95% CI |
|---|---:|---:|
| Prefill token/s | +102.13% | [+91.79%, +111.65%] |
| Decode token/s | +23.56% | [+17.73%, +29.47%] |
| TTFT reduction | +50.65% | [+47.92%, +52.76%] |
| TPOT reduction | +19.09% | [+15.10%, +22.77%] |
| E2E reduction | +26.23% | [+23.41%, +28.97%] |
| Client low-load pre token/s | +77.51% | [+68.88%, +90.18%] |
| Client high-load post token/s | +13.94% | [+7.69%, +21.03%] |

这组数据已给出论文主张所需的基本形状：V5 的低负载性能与 packed CPU fixed 统计等价，在负载突增后显著优于 fixed CPU，并避免 fixed iGPU 的空闲态代价。三种策略中 V5 的整体 Decode 和 E2E 最优，且关键效应量 CI 均不跨 0。

但 `dynamic V5 post > iGPU fixed post` 不能直接解释为调度器使同一 iGPU kernel 加速。V5 在 telemetry layer 最终为 iGPU，不同策略稳定后本应更接近；当前差异可能来自运行顺序、封装在 post 窗口内的切换过程、各层独立决策、cache/频率/温度状态或 CPU-iGPU 共享功耗与内存系统。正式论文在解释该差异前必须做稳态复验，不能把这 13.94% 直接归因于控制器。

下一步采用 `iGPU-fixed A1 -> dynamic V5 B -> iGPU-fixed A2` 的 compute20 稳态分块实验。每个 block 都在 benchmark warmup 前启动 20-worker nice=0 背景负载，使用 `p1024-o600`、N=5；A1/A2 可估计时间漂移和运行顺序效应，B 与两侧 A 的均值比较可判断 V5 高负载优势是否可复现。新增比较器的两项测试加入后，完整实验工具测试集 62 项通过，`py_compile` 通过；`ruff` 未安装，因此未执行该可选检查。

当前 iGPU-fixed 引擎可以直接用于 A1。先在背景负载终端启动并保持运行：

```bash
python kt-kernel/bench/cpu_background_load.py \
  --kind compute \
  --workers 20 \
  --affinity free \
  --nice 0
```

确认 ready JSON 后，在性能测试终端执行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label igpu-fixed__engine-low__compute20-steady-a1-n5 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 5 \
  --seed 20260719 \
  --bootstrap-samples 5000 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/igpu-fixed-engine-low-compute20-steady-a1-n5-20260719
```

A1 完成后先在背景终端 `Ctrl-C` 停止负载，但暂不删除任何产物。核对结果和 CPU telemetry 后，再停止 iGPU-fixed 引擎并进入 dynamic V5 的 B block。

### 2026-07-19：compute20 稳态分块实验 iGPU-fixed A1 N=5

结果位于 `artifacts/running-server-bench/igpu-fixed-engine-low-compute20-steady-a1-n5-20260719/`。五轮均为精确 `600/600` token；请求窗口 CPU busy fraction 为 0.9902，CPU PSI some 为 0.1395，证明 compute20 在 warmup 和全部 measured request 期间持续形成接近满载的 CPU 争用。

| Metric | Mean | Sample stdev | Bootstrap 95% CI |
|---|---:|---:|---:|
| Prefill token/s | 71.3172 | 3.0342 | [68.8758, 73.6005] |
| Decode token/s | 13.1302 | 0.5276 | [12.7550, 13.5349] |
| TTFT ms | 14688.42 | 633.33 | [14202.50, 15174.34] |
| TPOT ms/token | 76.26 | 3.01 | [73.61, 78.41] |
| E2E ms | 60366.79 | 2066.10 | [58706.94, 61846.16] |

A1 steady Decode 相对 iGPU-fixed transition post 的变化为 -1.51%，独立 bootstrap 95% CI `[-6.56%, +4.54%]`，CI 跨 0。即固定 iGPU 的 transition post `13.3315 token/s` 与 compute20 稳态 `13.1302 token/s` 统计一致，之前观察到的 fixed iGPU post 较低不能用 transition 窗口偏差解释。

在尚未获得 dynamic steady B block 前，仅作诊断性比较：V5 transition post `15.1904 token/s` 相对 A1 steady Decode 高 15.69%，独立 bootstrap 95% CI `[10.35%, 21.09%]`。该差异继续存在，但跨 transition/steady 协议，不能替代 B block 的直接比较。

进入 dynamic V5 B block 前按顺序操作：先在背景负载终端 `Ctrl-C`，确认 20-worker 组退出；再停止 iGPU-fixed 引擎。随后启动带独立 telemetry 文件的 V5：

```bash
ENGINE_PRIORITY=low \
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v5-compute20-steady-b.jsonl \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

dynamic 服务就绪后，在背景终端重新启动 compute20，并保持运行：

```bash
python kt-kernel/bench/cpu_background_load.py \
  --kind compute \
  --workers 20 \
  --affinity free \
  --nice 0
```

确认 ready JSON 后执行 B block：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v5__engine-low__compute20-steady-b-n5 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 5 \
  --seed 20260719 \
  --bootstrap-samples 5000 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v5-compute20-steady-b.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v5-engine-low-compute20-steady-b-n5-20260719
```

B block 验收除端到端指标外，还要求五轮 scheduler telemetry 均显示 decode 已稳定在高负载策略，没有 exploration、周期性 reprobe 或跨请求漂移。B 完成后再决定是否必须执行 iGPU-fixed A2；若 B 与 A1 无显著差异，则 transition post 优势更可能来自切换/状态历史，若 B 仍显著更快，则必须检查全层后端分配和硬件频率/功耗。

### 2026-07-19：compute20 稳态分块实验 dynamic V5 B N=5

结果位于 `artifacts/running-server-bench/dynamic-v5-engine-low-compute20-steady-b-n5-20260719/`。五轮均为精确 `600/600` token；CPU busy fraction 为 0.9899、CPU PSI some 为 0.1726，与 A1 的 0.9902/0.1395 一样属于持续满载条件。

| Metric | Mean | Sample stdev | Bootstrap 95% CI |
|---|---:|---:|---:|
| Prefill token/s | 145.4116 | 6.7148 | [140.1985, 150.6246] |
| Decode token/s | 15.3957 | 0.6260 | [14.8502, 15.7950] |
| TTFT ms | 7205.67 | 332.87 | [6947.40, 7463.94] |
| TPOT ms/token | 65.04 | 2.75 | [63.27, 67.46] |
| E2E ms | 46166.17 | 1745.27 | [45118.35, 47707.91] |

V5 B 相对 iGPU-fixed A1 的独立 10000 次 bootstrap 效应量：

| Metric | Effect | Bootstrap 95% CI |
|---|---:|---:|
| Prefill token/s | +103.89% | [+94.50%, +114.32%] |
| Decode token/s | +17.25% | [+11.89%, +22.23%] |
| TTFT reduction | 50.94% | [48.47%, 53.33%] |
| TPOT reduction | 14.71% | [10.51%, 18.18%] |
| E2E reduction | 23.52% | [20.17%, 26.28%] |

B block 直接复现了 V5 在 compute20 下高于 fixed iGPU 的 Decode 性能，说明该差异不是 none->compute20 分段统计产生的假象。请求级 telemetry 同时满足稳定性验收：五轮各 600 个 decode calls，观测 layer 0 共 3000 个 decode 事件全部 `igpu_ratio=policy_igpu_ratio=1.0`、exploration=false、`reprobe_reason=0`；每轮 request 内 switch delta=0。Prefill 在 layer 0 全部使用 CPU，因此 V5 的 Prefill/TTFT 优势符合预期。

但当前只观测 `KT_CPU_IGPU_TELEMETRY_LAYER=0`。layer 0 decode 固定 iGPU 并不证明所有 MoE 层都固定 iGPU；Decode +17.25% 仍可能来自其他层选择 CPU/iGPU 混合组合，也可能来自 fixed/dynamic 的 prefill 历史对 cache、频率、温度或共享功耗状态的影响。A2 用于排除简单时间漂移，之后必须以 `SCHEDULER_TELEMETRY_LAYER=all` 做全层诊断，不能仅凭 layer 0 推断原因。

当前 dynamic B 的 compute20 父进程仍在运行。先在背景终端 `Ctrl-C` 停止负载，再停止 dynamic 引擎。随后启动 iGPU-fixed A2：

```bash
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh igpu-fixed
```

服务就绪后重新启动同一背景负载：

```bash
python kt-kernel/bench/cpu_background_load.py \
  --kind compute \
  --workers 20 \
  --affinity free \
  --nice 0
```

确认 ready 后执行 A2：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label igpu-fixed__engine-low__compute20-steady-a2-n5 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 5 \
  --seed 20260719 \
  --bootstrap-samples 5000 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/igpu-fixed-engine-low-compute20-steady-a2-n5-20260719
```

A2 完成后停止背景负载。正式 ABA 分析以 B 对 `(A1+A2)/2` 的 block-level fixed iGPU 参考值报告，同时单独报告 A1/A2 漂移，避免只选择对 V5 有利的一侧基线。

### 2026-07-19：compute20 稳态 iGPU-fixed A2 与 ABA 分层汇总

A2 位于 `artifacts/running-server-bench/igpu-fixed-engine-low-compute20-steady-a2-n5-20260719/`。CPU busy fraction 0.9910、CPU PSI some 0.1375，与 A1 的 0.9902/0.1395 高度接近，两个 fixed block 的背景竞争强度一致。

| Metric | Mean | Sample stdev | Bootstrap 95% CI |
|---|---:|---:|---:|
| Prefill token/s | 83.2590 | 2.1823 | [81.6010, 84.9350] |
| Decode token/s | 13.4941 | 0.6532 | [13.0223, 14.0250] |
| TTFT ms | 12570.17 | 331.78 | [12317.93, 12821.18] |
| TPOT ms/token | 74.24 | 3.53 | [71.35, 76.86] |
| E2E ms | 57041.82 | 2015.02 | [55474.77, 58601.01] |

A2 相对 A1 的 block 漂移：

| Metric | A2 vs A1 | Independent bootstrap 95% CI |
|---|---:|---:|
| Prefill token/s | +16.74% | [+12.38%, +21.47%] |
| Decode token/s | +2.77% | [-2.05%, +8.01%] |
| TTFT reduction | 14.42% | [11.00%, 17.76%] |
| TPOT reduction | 2.64% | [-2.11%, +7.28%] |
| E2E reduction | 5.51% | [1.70%, 9.00%] |

Decode/TPOT 漂移 CI 跨 0，A1/A2 的稳态 decode 可视为一致；Prefill/TTFT 存在明显 block 效应，进一步证明论文不能把所有 request 简单视为同分布独立样本。

新增 `kt-kernel/bench/report_blocked_strategy_comparison.py`：每个策略先按 engine block 分组，点估计对 block mean 等权；bootstrap 外层重采样 block、内层重采样被选 block 内的 request。ABA 可复现产物位于 `artifacts/running-server-bench/steady-compute20-aba-n5-20260719/`，使用 20000 次 hierarchical percentile bootstrap、seed 20260719。

| Strategy/block | N | Prefill token/s | Decode token/s | TTFT ms | TPOT ms | E2E ms | CPU busy | PSI some |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iGPU fixed A1 | 5 | 71.317 | 13.130 | 14688.42 | 76.26 | 60366.79 | 0.9902 | 0.1395 |
| dynamic V5 B | 5 | 145.412 | 15.396 | 7205.67 | 65.04 | 46166.17 | 0.9899 | 0.1726 |
| iGPU fixed A2 | 5 | 83.259 | 13.494 | 12570.17 | 74.24 | 57041.82 | 0.9910 | 0.1375 |

fixed iGPU 参考值为 A1/A2 block mean 的等权均值。V5 B 相对该参考：

| Metric | Fixed block reference | Dynamic V5 | Effect | Hierarchical bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Prefill token/s | 77.2881 | 145.4116 | +88.14% | [+70.11%, +109.63%] |
| Decode token/s | 13.3121 | 15.3957 | +15.65% | [+10.28%, +20.55%] |
| TTFT ms | 13629.29 | 7205.67 | 47.13% reduction | [41.24%, 52.33%] |
| TPOT ms/token | 75.2502 | 65.0425 | 13.57% reduction | [9.28%, 17.10%] |
| E2E ms | 58704.31 | 46166.17 | 21.36% reduction | [17.09%, 25.21%] |

ABA 结果排除了 fixed iGPU 的简单时间漂移解释，compute20 稳态 Decode 优势仍约 15.7%。但 fixed 有 2 个 blocks、dynamic 只有 1 个 block，当前 hierarchical CI 只能反映已观测的 fixed block 漂移与 B 内 request 波动，不能估计 dynamic 的 block 间方差。该结果足以驱动机制诊断；论文最终效应量至少再增加一个 dynamic block，最好采用 ABBA/BAAB 随机化 block 顺序。

下一步先做全层调度状态诊断，不把带全层 JSONL 写入开销的吞吐当成主性能结果。当前 A2 的 compute20 父进程仍在运行，先在背景终端 `Ctrl-C`，再停止 iGPU-fixed 引擎。启动全层 telemetry 的 V5：

```bash
ENGINE_PRIORITY=low \
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v5-compute20-all-layers.jsonl \
SCHEDULER_TELEMETRY_LAYER=all \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

服务就绪后重新启动 compute20：

```bash
python kt-kernel/bench/cpu_background_load.py \
  --kind compute \
  --workers 20 \
  --affinity free \
  --nice 0
```

确认 ready 后采集一个诊断请求：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v5__engine-low__compute20-steady-all-layers-diagnostic \
  --workloads 1024:300 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v5-compute20-all-layers.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v5-engine-low-compute20-steady-all-layers-diagnostic-20260719
```

该诊断的验收输出是逐层表：每层 prefill/decode 的 execution ratio、policy ratio、exploration、reprobe reason、switch delta、CPU/iGPU service cost。若存在 decode CPU 层，则 B 的优势可能来自跨层异构组合；若所有层 decode 均为 iGPU，则继续采集 iGPU 频率、CPU package 功耗/温度和 prefill 历史对照。

### 2026-07-19：compute20 全层 telemetry 诊断与 phase-fixed 消融

诊断请求位于 `artifacts/running-server-bench/dynamic-v5-engine-low-compute20-steady-all-layers-diagnostic-20260719/`。全层 JSONL 写入使 Prefill/Decode 降至 134.08/14.22 token/s，因此该吞吐不进入主性能比较。请求 CPU busy fraction 0.9910、PSI some 0.2102，仍是有效 compute20 满载条件。

新增 `kt-kernel/bench/report_scheduler_layers.py`，逐层聚合 execution ratio、policy ratio、exploration、reprobe、switch 和最终 service cost；可复现结果位于 `artifacts/running-server-bench/dynamic-v5-compute20-all-layers-analysis-20260719/`。

全层结果：

| Diagnostic | Result |
|---|---:|
| MoE layers | 40 |
| Telemetry events | 12040 |
| Prefill fully CPU layers | 40/40 |
| Decode fully iGPU layers | 40/40 |
| Decode calls per layer | 300 |
| Exploration calls | 0 |
| Reprobe calls | 0 |
| Request-window switch delta | 0 |
| Mean final CPU service cost | 1.29002 ms/row |
| Mean final iGPU service cost | 0.09352 ms/row |
| Layers with lower iGPU cost | 40/40 |

该结果排除了“V5 在其他层保留 CPU，跨层混合使 Decode 更快”的解释。V5 B 与 fixed iGPU 在 measured decode 中都对 40 层使用 iGPU；已知功能差异是 V5 在 prefill 对 40 层使用 CPU，而 fixed iGPU 在 prefill/decode 都使用 iGPU。因此最直接假设变为：CPU-prefill/iGPU-decode 的阶段映射改善 Prefill，并通过 cache、频率、温度、共享功耗或内存状态影响后续 decode。仅凭当前 telemetry 还不能区分这些硬件机制。

为隔离阶段映射与动态控制器，新增长期保留的消融后端 `phase-fixed`：固定 prefill ratio=0、decode ratio=1，`cpu_igpu_dynamic=false`，因此不执行负载监测后的选择、校准、探索或 reprobe。V5 代码和 V5 参数未修改。实现包含：

- `GeneralMOEConfig.cpu_igpu_prefill_ratio` 与 `cpu_igpu_decode_ratio`；负值向后兼容地继承原 `cpu_igpu_igpu_ratio`。
- pybind 和 `_configure_cpu_igpu_scheduler` 支持两个比例以及 `KT_CPU_IGPU_POLICY=phase-fixed`。
- `gptq_int4_cpu_igpu-moe.hpp` 在非动态模式按 qlen 选择固定 phase ratio。
- `35b-test-cpu-igpu.sh phase-fixed` 固定输出 `prefill_ratio=0 decode_ratio=1`，其他四种启动模式显式清理 phase override，避免环境泄漏。

构建记录：第一次执行不带 `CPUINFER_USE_SYCL=1` 的安装命令生成了 CPU-only 扩展，preflight 正确发现 hybrid symbol 缺失；随后仅加 `CPUINFER_USE_SYCL=1` 又因 shell 未加载 oneAPI 而在 CMake 配置阶段找不到 `icpx/dpcpp`。最终正确命令为：

```bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null && \
CPUINFER_USE_SYCL=1 ./install.sh kt-kernel --no-clean
```

最终 SYCL 扩展编译到 100% 并成功安装。安装后验证：`MOEConfig` 包含默认 -1.0 的两个 phase ratio 字段，`CPUiGPUGPTQInt4_MOE` symbol 存在，`phase-fixed` preflight 从本地 build 成功加载。实验工具 67 项通过，scheduler Python 配置 4 项通过，`py_compile`、`bash -n`、`git diff --check` 通过。构建仅出现项目既有的 llama/kvcache/abstract-interface 警告。

当前全层诊断的 compute20 父进程和安装前启动的 dynamic 服务仍是旧进程。先在背景终端 `Ctrl-C`，再停止旧 dynamic 引擎；随后启动新 phase-fixed：

```bash
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh phase-fixed
```

服务就绪后重新启动 compute20：

```bash
python kt-kernel/bench/cpu_background_load.py \
  --kind compute \
  --workers 20 \
  --affinity free \
  --nice 0
```

确认 ready 后执行 phase-fixed C block：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label phase-fixed__engine-low__compute20-steady-c-n5 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 5 \
  --seed 20260719 \
  --bootstrap-samples 5000 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/phase-fixed-engine-low-compute20-steady-c-n5-20260719
```

若 phase-fixed 与 V5 B 的 Decode/Prefill 统计一致，则 fixed-iGPU 的差距主要来自阶段映射或 phase history，而不是动态决策本身；V5 的贡献应表述为在线发现并维持该映射、处理负载变化，而不是让相同 iGPU decode kernel 加速。若 phase-fixed Decode 回落到 fixed-iGPU，则继续用频率/功耗和函数级 trace 查找动态路径的隐含状态差异。

### 2026-07-19：phase-fixed compute20 C block N=5

结果位于 `artifacts/running-server-bench/phase-fixed-engine-low-compute20-steady-c-n5-20260719/`。运行进程环境已确认 `KT_CPU_IGPU_POLICY=phase-fixed`、prefill ratio=0、decode ratio=1。CPU busy fraction 0.9908、PSI some 0.1672，与 A/B blocks 的满载条件一致。

| Metric | Mean | Sample stdev | Bootstrap 95% CI |
|---|---:|---:|---:|
| Prefill token/s | 143.7871 | 2.5002 | [141.8368, 145.7373] |
| Decode token/s | 16.3472 | 1.1969 | [15.3958, 17.2986] |
| TTFT ms | 7276.41 | 126.55 | [7177.63, 7375.18] |
| TPOT ms/token | 61.43 | 4.48 | [57.87, 65.00] |
| E2E ms | 44076.00 | 2603.52 | [42008.50, 46010.07] |

初步 blocked comparison 位于 `artifacts/running-server-bench/steady-compute20-phase-fixed-preliminary-20260719/`。phase-fixed 相对 A1/A2 等权 fixed-iGPU 参考：Prefill +86.04% `[+70.53%, +105.26%]`，Decode +22.80% `[+14.95%, +30.77%]`，E2E 减少 24.92% `[20.03%, 29.55%]`。因此 CPU-prefill/iGPU-decode 阶段映射本身已经能够解释 fixed-iGPU 与 V5 之间的主要差距。

phase-fixed 相对 V5 B：Prefill -1.12% `[-4.85%, +2.77%]`，Decode +6.18% `[-0.51%, +13.35%]`，E2E 减少 4.53% `[-0.64%, +9.71%]`；三项 CI 均跨 0，当前不能声称 phase-fixed 或 V5 显著更快。

但 B block 启用了同步 layer-0 scheduler JSONL，而 C 未启用 telemetry。writer 每个 decode token 执行 JSON 编码和 `os.write`，这会把观测开销混入所谓“动态控制器开销”。此前论文主对比中的 dynamic 也带 telemetry、fixed 不带，因此现有 dynamic 优势是保守估计，但不能用 B/C 差值量化控制器代价。

下一步补充关闭 telemetry 的 V5 D block。当前 phase-fixed 的 compute20 父进程仍在运行；先在背景终端 `Ctrl-C`，再停止 phase-fixed 引擎。启动时显式清空 telemetry 文件变量：

```bash
SCHEDULER_TELEMETRY_FILE= \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

服务就绪后重新启动 compute20：

```bash
python kt-kernel/bench/cpu_background_load.py \
  --kind compute \
  --workers 20 \
  --affinity free \
  --nice 0
```

确认 ready 后执行 D block，benchmark 命令中不传 `--scheduler-telemetry-file`：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v5__engine-low__compute20-steady-no-telemetry-d-n5 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 5 \
  --seed 20260719 \
  --bootstrap-samples 5000 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v5-engine-low-compute20-steady-no-telemetry-d-n5-20260719
```

若 D 与 phase-fixed C 等价，则 B/C 差异主要是 telemetry 开销，动态控制器稳态成本低于当前 N=5 检出能力；若 D 仍显著慢于 C，才继续优化共享 scheduler 每层 mutex/load-monitor 路径。

### 2026-07-19：V5 no-telemetry D block 与控制器开销消融

D 位于 `artifacts/running-server-bench/dynamic-v5-engine-low-compute20-steady-no-telemetry-d-n5-20260719/`。manifest 确认 `scheduler_telemetry_file=null`、event count=0；CPU busy fraction 0.9927、PSI some 0.1700，与 phase-fixed C 的 0.9908/0.1672 几乎相同。

| Metric | Mean | Sample stdev | Bootstrap 95% CI |
|---|---:|---:|---:|
| Prefill token/s | 142.0897 | 2.7401 | [140.0007, 144.2845] |
| Decode token/s | 16.3104 | 1.1576 | [15.4139, 17.2069] |
| TTFT ms | 7363.72 | 141.01 | [7255.87, 7471.57] |
| TPOT ms/token | 61.56 | 4.45 | [58.14, 64.99] |
| E2E ms | 44239.99 | 2784.02 | [42107.03, 46440.60] |

最终 blocked ablation 位于 `artifacts/running-server-bench/steady-compute20-controller-overhead-ablation-20260719/`，包含 A1/A2 fixed-iGPU、telemetry-on V5 B、phase-fixed C 和 no-telemetry V5 D，使用 20000 次 hierarchical bootstrap。

V5 D 相对 phase-fixed C：

| Metric | Effect | Hierarchical bootstrap 95% CI |
|---|---:|---:|
| Prefill token/s | -1.18% | [-3.11%, +0.91%] |
| Decode token/s | -0.22% | [-8.03%, +8.10%] |
| TTFT reduction | -1.20% | [-3.23%, +0.87%] |
| TPOT reduction | -0.21% | [-8.78%, +7.40%] |
| E2E reduction | -0.37% | [-7.38%, +6.00%] |

所有 CI 均跨 0，V5 与 phase-fixed 在 compute20 稳态统计等价。当前 N=5 无法检出动态控制器的额外稳态成本；不应继续依据点估计优化 shared scheduler mutex/load-monitor 路径。

V5 D 相对 A1/A2 fixed-iGPU 等权参考：Prefill +83.84% `[+68.44%, +103.00%]`、Decode +22.52% `[+14.73%, +30.24%]`、E2E 减少 24.64% `[19.56%, 29.36%]`。结合全层 telemetry，可将机制结论表述为：V5 在 compute20 下在线收敛到 CPU-prefill/iGPU-decode 阶段映射，并达到预先知道正确映射的 phase-fixed oracle 性能。

no-telemetry D 相对 telemetry-on B 的 Decode 点估计为 +5.94%，CI `[-0.84%, +12.76%]`；E2E 减少 4.17%，CI `[-1.39%, +9.62%]`。方向符合同步 JSONL 开销预期，但 N=5 下未达显著，不能给 telemetry 定量宣称固定开销比例。

论文实验协议据此冻结：主性能测量不启用 scheduler telemetry；相同配置另做诊断 pass，报告状态序列、切换延迟和 service cost。不得把 telemetry-on dynamic 与 telemetry-off fixed 的差值解释成纯策略效应；已有 transition N=5 是保守开发结果，正式论文主表需用 no-telemetry performance pass 重采。

下一步测无负载稳态，验证 phase-fixed 不是全局最优而 V5 会恢复 CPU decode。当前 dynamic D 的 compute20 父进程仍在运行；只在背景终端 `Ctrl-C` 停止负载，保留当前 no-telemetry dynamic 引擎。随后执行 E block，三次 128-token warmup 足以覆盖 V5 的 load-drop 检测和 32-call CPU reprobe：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v5__engine-low__none-steady-no-telemetry-e-n5 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 5 \
  --seed 20260719 \
  --bootstrap-samples 5000 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v5-engine-low-none-steady-no-telemetry-e-n5-20260719
```

E 的预期是 Prefill 接近 CPU phase、Decode 明显高于 phase-fixed/iGPU decode；若达成，则形成两状态 oracle-tracking 证据：无负载选择 CPU/CPU，compute20 选择 CPU/iGPU。

### 2026-07-19：V5 no-telemetry 无负载 E block N=5

E 位于 `artifacts/running-server-bench/dynamic-v5-engine-low-none-steady-no-telemetry-e-n5-20260719/`。manifest 确认 `scheduler_telemetry_file=null`，本次运行期间没有 `cpu_background_load.py` 进程。请求窗口 CPU busy fraction 为 0.4302、CPU nice fraction 为 0.4199、PSI some fraction 为 0.0002，说明观测到的 CPU 使用主要来自 nice=5 的推理引擎，系统不存在可测的 CPU 压力等待。

| Metric | Mean | Sample stdev | Bootstrap 95% CI |
|---|---:|---:|---:|
| Prefill token/s | 169.5466 | 0.1464 | [169.4314, 169.6617] |
| Decode token/s | 29.6862 | 0.0315 | [29.6650, 29.7124] |
| TTFT ms | 6169.40 | 5.33 | [6165.21, 6173.59] |
| TPOT ms/token | 33.6857 | 0.0357 | [33.6539, 33.7090] |
| E2E ms | 26347.19 | 19.18 | [26332.31, 26360.46] |

五个请求的 Decode 为 29.66--29.74 token/s，变异远小于 compute20 blocks。相对相同 dynamic/no-telemetry 的 compute20 D block，E 的 Prefill 点估计提高 19.32%，Decode 提高 82.01%，E2E 减少 40.44%。该差值同时包含外部负载状态变化和控制器策略变化，不能作为纯调度收益；它的作用是确认 V5 在负载解除后没有滞留于 compute20 的低吞吐状态。

E 与 D 共同给出了两状态 oracle-tracking 假设的第一半证据：V5 在 compute20 下达到 CPU-prefill/iGPU-decode 的 phase-fixed oracle，而在无负载下恢复到 29.69 token/s 的更快 Decode。由于 E 是 no-telemetry 性能 pass，不能仅根据吞吐宣称每层已经选择 CPU/CPU；最终论文应使用独立的轻量诊断 pass 确认执行比例，并把诊断结果与主性能数据分开报告。

下一步采集无负载 phase-fixed F block，直接检验固定 CPU-prefill/iGPU-decode 是否在无负载下劣于 V5。当前 dynamic 引擎 PID 958953 仍运行在端口 30100，nice=5；先在引擎终端 `Ctrl-C`，再启动不带 telemetry 的 phase-fixed：

```bash
SCHEDULER_TELEMETRY_FILE= \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh phase-fixed
```

保持没有背景负载，服务就绪后执行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label phase-fixed__engine-low__none-steady-f-n5 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 5 \
  --seed 20260719 \
  --bootstrap-samples 5000 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/phase-fixed-engine-low-none-steady-f-n5-20260719
```

F 的主要判据是 Decode/TPOT/E2E：若 Prefill 与 E 接近而 Decode 明显低于 29.69 token/s，则说明 phase-fixed 只是在 compute20 下的状态特定 oracle，V5 的价值是按负载在线选择不同 phase mapping。完成 F 后再补 packed-CPU fixed 无负载 G block，以检验 V5 E 是否达到 CPU/CPU oracle；正式效应量使用 E/F/G 的相同无负载 block，而不混用早期不同协议的数据。

### 2026-07-19：phase-fixed 无负载 F block N=5

F 位于 `artifacts/running-server-bench/phase-fixed-engine-low-none-steady-f-n5-20260719/`。manifest 确认 telemetry 关闭，运行时没有背景负载；CPU busy fraction 为 0.1452、CPU nice fraction 为 0.1152、PSI some fraction 为 0.0012。

| Metric | Mean | Sample stdev | Bootstrap 95% CI |
|---|---:|---:|---:|
| Prefill token/s | 168.2883 | 1.6116 | [166.8058, 169.2335] |
| Decode token/s | 19.4347 | 0.7793 | [18.8223, 20.0472] |
| TTFT ms | 6215.99 | 60.23 | [6181.38, 6271.38] |
| TPOT ms/token | 51.5201 | 2.0527 | [49.9020, 53.1381] |
| E2E ms | 37076.60 | 1198.95 | [36138.17, 38015.02] |

E/F 的 blocked comparison 位于 `artifacts/running-server-bench/steady-none-phase-fixed-ablation-20260719/`，使用 20000 次 hierarchical bootstrap。V5 E 相对 phase-fixed F：

| Metric | Effect | Hierarchical bootstrap 95% CI |
|---|---:|---:|
| Prefill token/s improvement | +0.75% | [+0.19%, +1.64%] |
| Decode token/s improvement | +52.75% | [+48.05%, +57.64%] |
| TTFT reduction | +0.75% | [+0.19%, +1.61%] |
| TPOT reduction | +34.62% | [+32.50%, +36.59%] |
| E2E reduction | +28.94% | [+27.09%, +30.67%] |

Prefill 仅有约 0.75% 差异，而 Decode 和 E2E 差异很大，符合两种配置都使用 CPU Prefill、但 V5 在无负载下恢复更快 Decode 后端的假设。结合 compute20 下 V5 D 与 phase-fixed C 统计等价，可得到负载依赖的 oracle-tracking 结果：CPU-prefill/iGPU-decode 是 compute20 的合适映射，却不是无负载的全局最优映射；动态控制器的贡献是在线选择并维持状态特定映射。

当前 E/F 各只有一个 block，hierarchical bootstrap 只能估计已观测 block 内的请求波动，不能估计跨时间 block 方差。因此这些结果可作为开发阶段强证据，但论文最终置信区间需要增加随机化或交错顺序的重复 blocks。

下一步补充无负载 packed-CPU fixed G block，检验 V5 E 是否达到 CPU/CPU oracle。当前 phase-fixed 引擎 PID 966330 仍在端口 30100 运行；先在引擎终端 `Ctrl-C`，再启动：

```bash
SCHEDULER_TELEMETRY_FILE= \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh packed-cpu-fixed
```

保持无背景负载，服务就绪后执行：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label packed-cpu-fixed__engine-low__none-steady-g-n5 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 5 \
  --seed 20260719 \
  --bootstrap-samples 5000 \
  --fail-fast \
  --output-dir artifacts/running-server-bench/packed-cpu-fixed-engine-low-none-steady-g-n5-20260719
```

G 的主要判据是 V5 E 与 packed-CPU G 的 Decode、TPOT 和 E2E 是否统计等价。若等价，则完整证据为：无负载 V5 达到 CPU/CPU oracle，compute20 V5 达到 CPU/iGPU phase-fixed oracle，并显著优于各状态下选错的固定策略。

### 2026-07-19：packed-CPU fixed 无负载 G block N=5

G 位于 `artifacts/running-server-bench/packed-cpu-fixed-engine-low-none-steady-g-n5-20260719/`。进程环境确认 `KT_CPU_IGPU_POLICY=fixed`、`KT_CPU_IGPU_RATIO=0`、telemetry 关闭；运行时没有背景负载。CPU busy fraction 为 0.4301、CPU nice fraction 为 0.4198、PSI some fraction 为 0.0002，几乎完全匹配 V5 E 的 0.4302/0.4199/0.0002。

| Metric | Mean | Sample stdev | Bootstrap 95% CI |
|---|---:|---:|---:|
| Prefill token/s | 169.8195 | 0.6242 | [169.2695, 170.2384] |
| Decode token/s | 29.8582 | 0.0753 | [29.8091, 29.9235] |
| TTFT ms | 6159.55 | 22.72 | [6144.52, 6179.40] |
| TPOT ms/token | 33.4918 | 0.0843 | [33.4202, 33.5468] |
| E2E ms | 26221.21 | 32.88 | [26196.79, 26246.08] |

E/F/G 三策略 comparison 位于 `artifacts/running-server-bench/steady-none-oracle-tracking-20260719/`，使用 20000 次 hierarchical bootstrap。V5 E 相对 packed-CPU G：

| Metric | V5 effect relative to packed CPU | Hierarchical bootstrap 95% CI |
|---|---:|---:|
| Prefill token/s improvement | -0.16% | [-0.42%, +0.17%] |
| Decode token/s improvement | -0.58% | [-0.81%, -0.38%] |
| TTFT reduction | -0.16% | [-0.42%, +0.17%] |
| TPOT reduction | -0.58% | [-0.82%, -0.39%] |
| E2E reduction | -0.48% | [-0.60%, -0.37%] |

V5 在该 block 达到 packed-CPU Decode oracle 的 99.42%，E2E oracle 的 99.52%。Decode/E2E 的请求内 CI 不跨 0，但 E/G 各只有一个时间 block，区间没有估计跨 block 的系统漂移，因此不能把 0.58% 宣称为稳定的动态控制器开销，也不能以此为依据继续调参。论文当前可表述为“V5 在无负载下达到 99.4% 的 CPU/CPU oracle Decode 性能，单 block 剩余差距低于 1%”；正式实验通过重复、交错 blocks 检验该差距是否持续，并预先设定 practical-equivalence margin。

结合 compute20 C/D 和无负载 E/F/G，目前状态特定 oracle-tracking 证据闭环：

| Load state | Selected/expected phase mapping | V5 relative result |
|---|---|---|
| none | CPU Prefill + CPU Decode | 99.42% of packed-CPU Decode oracle |
| compute20 | CPU Prefill + iGPU Decode | statistically indistinguishable from phase-fixed oracle at N=5 |

最后补一个不进入主性能表的无负载全层 telemetry 诊断，确认 V5 的 40 层实际执行比例。先在引擎终端停止当前 packed-CPU PID 970712，再启动 diagnostic dynamic：

```bash
ENGINE_PRIORITY=low \
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v5-none-all-layers.jsonl \
SCHEDULER_TELEMETRY_LAYER=all \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

保持无背景负载，服务就绪后执行一个诊断请求：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v5__engine-low__none-steady-all-layers-diagnostic \
  --workloads 1024:300 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260719 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v5-none-all-layers.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v5-engine-low-none-steady-all-layers-diagnostic-20260719
```

该诊断只用于确认 Prefill/Decode 的逐层 execution ratio、探索和切换状态；同步全层 JSONL 会扰动吞吐，因此诊断请求的性能数字不得与 E/F/G 比较。完成后开发阶段 V5 验证即可冻结，转入正式论文实验矩阵自动化。

### 2026-07-19：V5 无负载全层 telemetry 诊断与开发阶段冻结

诊断请求位于 `artifacts/running-server-bench/dynamic-v5-engine-low-none-steady-all-layers-diagnostic-20260719/`，逐层分析位于 `artifacts/running-server-bench/dynamic-v5-none-all-layers-analysis-rerun-20260719/`。请求 CPU busy fraction 为 0.4367、PSI some fraction 为 0.0007，仍是有效无负载状态。

诊断吞吐为 Prefill 168.11 token/s、Decode 27.66 token/s。它低于 no-telemetry E 的 169.55/29.69 token/s，方向符合每层每 token 同步 JSONL 编码和写入开销；该数值只用于验证机制，不进入主性能表，也不用于估计 telemetry 的固定开销。

逐层报告聚合 measured request 的 12040 个事件：

| Diagnostic | Result |
|---|---:|
| MoE layers | 40 |
| Decode fully CPU layers | 40/40 |
| Decode fully iGPU layers | 0/40 |
| Prefill fully CPU layers with execution samples | 39/39 |
| Prefill execution-sample coverage | 39/40 |
| Prefill CPU-policy layers | 40/40 |
| Decode calls per layer | 300 |
| Exploration calls | 0 |
| Reprobe calls | 0 |
| Request-window switch delta | 0 |
| Mean final CPU service cost | 0.03630 ms/row |
| Mean final iGPU service cost | 0.09334 ms/row |
| Layers with lower final iGPU cost | 0/40 |

layer 0 的 Prefill 事件记录了 `policy_igpu_ratio=0` 和 CPU snapshot，但 `execution_calls_delta=0`、实际 `igpu_ratio=null`，因此没有把缺测强行计为 CPU execution。其余 39 层的实际 ratio 全部为 0，同时 40 层 policy 全部为 0；结合 40/40 Decode CPU execution，可以严谨地确认无负载下 V5 收敛并稳定维持 CPU/CPU phase mapping。

该缺测同时暴露了 `report_scheduler_layers.py` 的报告层 bug：CSV 能写空值，但 Markdown 曾直接执行 `float(None)`。报告器现已改为输出 `n/a`、execution coverage、CPU-policy layer count 和 Decode fully-CPU count；新增缺失 Prefill execution sample 的回归测试。`test_report_scheduler_layers.py` 共 3 项通过，`py_compile` 和 `git diff --check` 通过。第一次失败生成的 `dynamic-v5-none-all-layers-analysis-20260719/` 是不完整调试产物，最终引用带 `-rerun` 的报告目录。

至此，V5 开发阶段的两状态机制与性能证据闭环：

1. 无负载：40/40 Decode 层执行 CPU，V5 Decode 达到 packed-CPU oracle 的 99.42%，单 block 剩余差距低于 1%。
2. compute20：40/40 Decode 层执行 iGPU，V5 与预先知道正确 CPU-prefill/iGPU-decode 映射的 phase-fixed oracle 在 N=5 下无可检出差异。
3. 错误固定映射代价明确：无负载固定 iGPU Decode 使 V5 相对收益为 +52.75% Decode；compute20 固定 packed CPU 时，V5 transition Decode 相对收益为 +95.89%。
4. V5 在持续高负载下没有 V4 false reprobe，在双向切换开发试验中能够恢复状态特定策略。

开发阶段据此冻结 V5 算法和阈值，不再根据单次或单 block 点估计调参。当前 diagnostic dynamic 引擎 PID 973919 仍在端口 30100 运行；完成记录后可在引擎终端 `Ctrl-C` 停止。下一阶段只做实验基础设施和正式论文测量：主性能 pass 全部 telemetry-off，诊断 pass 独立采集，采用随机化/交错 block 顺序和预先固定的负载、工作负载、重复次数及 practical-equivalence margin。

### 2026-07-20：V5 source-fingerprint freeze 与正式 steady-load pilot 协议

当前没有 SGLang 服务或 `cpu_background_load.py` 进程。开发分支为 `vnni-sycl-scheduling-dev`，HEAD 基线为 `8f5e2884548ee654db5d396bb6139af0d9a20fab`（`[feat](sycl): finish and test sycl gptq int4`）。当前 V5、实验工具和文档仍包含未提交修改/新文件，因此该 HEAD 不能单独标识实际实验代码；在正式提交创建前，采用 source-fingerprint freeze：每个 sweep manifest 同时记录 branch、HEAD、dirty 文件列表、tracked diff SHA-256，以及调度器、CPU/SYCL 算子、Python 配置、benchmark 和启动脚本等关键文件的逐文件 SHA-256。

冻结边界如下：

1. V5 算法、阈值、校准、迟滞、探索和 reprobe 逻辑冻结；pilot 和正式实验期间不根据结果调参。
2. 允许修复不改变被测策略语义的实验工具 bug，但必须记录修改并由新 source fingerprints 区分。
3. 主性能 pass 强制关闭 `SCHEDULER_TELEMETRY_FILE`；逐层机制诊断独立运行且不进入性能表。
4. 推理服务由用户启动/停止；sweep runner 只验证服务，不向服务 PID 发送信号。
5. runner 只停止自己创建的背景负载进程组；检测到外部 `cpu_background_load.py` 时拒绝运行。

新增 `kt-kernel/bench/bench_running_server_load_sweep.py` 作为运行中服务的正式 steady-load 编排器。它复用 `bench_running_server.py` 的请求、token、TTFT/TPOT、CPU telemetry 和 bootstrap 实现；每个负载点独立启动 `cpu_background_load.py`、等待结构化 ready、稳定后测量，并在正常完成、失败或 Ctrl-C 时清理其进程组。每个负载点前重新验证：

- 端口上仍是同一个 SGLang server PID；
- server nice=5；
- `--kt-method=CPU_IGPU_GPTQ_INT4`；
- 环境中的 policy/ratio 与命令行声明的 backend 一致；
- scheduler telemetry 为空；
- 没有外部背景负载进程。

根 manifest 记录硬件、Python、`lscpu`、`lspci`、SYCL/NVIDIA 设备探测、服务器命令/亲和性/公开调度环境、模型 `config.json` hash、实际 kt-kernel extension hash（可发现时）、源文件 fingerprints、随机顺序以及每个子 benchmark 的命令和背景进程 ready 元数据。根目录另生成合并 `summary.csv`、`report.md`、逐点日志和原始 benchmark artifacts；已存在的目录会被拒绝，防止覆盖。

第一轮只做筛选性 pilot，不作为最终跨 block 显著性结论：

| Protocol item | Frozen value |
|---|---|
| Strategies | dynamic V5, packed-cpu-fixed, igpu-fixed |
| Engine priority | nice=5 (`ENGINE_PRIORITY=low`) |
| Background load | compute, free affinity, nice=0 |
| Worker counts | 0, 4, 8, 12, 16, 20 |
| Workload | p1024-o600 |
| Warmup | 3 requests, p256-o128 |
| Measured requests | N=5 per load point |
| Bootstrap | 5000 samples |
| Seed | 20260720 |
| Load order | shuffled; realized `[0, 12, 20, 16, 8, 4]` |
| Load stabilization | 3 seconds before benchmark warmup |
| Scenario cooldown | 5 seconds after each point |
| Scheduler telemetry | disabled |
| Practical-equivalence margin | 2% |

pilot 的目标是确定性能交叉区、检查自动化和估计方差，不以单个 block 的 request-level CI 代替跨 block 方差。pilot 完成后再冻结重点负载点，并用不同 seed 的随机化/交错 blocks 做至少 3 次独立重复。

用户启动 dynamic V5：

```bash
SCHEDULER_TELEMETRY_FILE= \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

服务 ready 后运行第一个完整 pilot block：

```bash
python kt-kernel/bench/bench_running_server_load_sweep.py \
  --base-url http://127.0.0.1:30100 \
  --backend dynamic \
  --block-label pilot-b1 \
  --load-workers 0,4,8,12,16,20 \
  --load-order shuffled \
  --load-affinity free \
  --load-nice 0 \
  --load-stabilization-seconds 3 \
  --scenario-cooldown-seconds 5 \
  --workloads 1024:600 \
  --warmups 3 \
  --warmup-output-tokens 128 \
  --repetitions 5 \
  --seed 20260720 \
  --bootstrap-samples 5000 \
  --expected-server-nice 5 \
  --output-dir artifacts/running-server-sweeps/dynamic-pilot-b1-20260720
```

该 block 完成并检查 manifest 后，再依次停止/重启为 `packed-cpu-fixed` 和 `igpu-fixed`，保持完全相同参数，仅修改 `--backend` 与输出目录。不要同时启动三个引擎，也不要在一个引擎环境下只改结果标签。

正式 sweep 工具验证记录：新增 `kt-kernel/test/test_running_server_load_sweep.py`，覆盖 worker count 解析、确定性随机顺序、错误后端/优先级/telemetry 拒绝、server PID 变化拒绝、free-affinity nice=0 命令、背景负载启动失败清理、benchmark 命令禁止 telemetry，以及 dry-run 不创建目录。CPU+iGPU 实验工具与调度配置相关的 10 个测试文件共 80 项通过；两个新文件通过 Black、`py_compile`，启动脚本通过 `bash -n`，全仓 `git diff --check` 通过。环境没有安装 `ruff`，因此未执行 ruff lint。

执行了一个 `compute workers=1, affinity=free, nice=0, duration=0.2s` 的生命周期 smoke。ready 元数据显示父/子进程均应用 nice=0，允许 CPU 为 0--19；duration 到期后父子进程均退出。验证结束时没有 SGLang 或 `cpu_background_load.py` 残留进程。

dry-run 不访问服务、不创建输出目录，确认 seed=20260720 的实际负载顺序为 `[0, 12, 20, 16, 8, 4]`，每个点调用 telemetry-off 的 `bench_running_server.py`，输出目录分别使用 `runs/01-none` 至 `runs/06-compute4`。至此正式 steady-load pilot 基础设施完成；下一动作是用户启动 telemetry-off、nice=5 的 dynamic V5，然后执行上述 `dynamic-pilot-b1-20260720` sweep 命令。

### 2026-07-20：dynamic V5 steady-load pilot B1

结果位于 `artifacts/running-server-sweeps/dynamic-pilot-b1-20260720/`。根 manifest 状态为 complete，协议版本 `cpu-igpu-steady-v1`；6/6 负载点、30/30 measured requests 全部成功。实际顺序为 `[0, 12, 20, 16, 8, 4]`。server PID 44211 在所有点保持不变，nice=5，policy=dynamic、ratio=0、scheduler telemetry 为空；extension SHA-256 为 `bf38fa420c069ca981ac28ac7842131fc1b52b93fa86332b263b94457ed12ead`。sweep 结束后没有背景负载残留，dynamic 服务继续运行。

| Workers | Prefill token/s [95% CI] | Decode token/s [95% CI] | CPU busy | PSI some |
|---:|---:|---:|---:|---:|
| 0 | 167.04 [163.64, 169.04] | 29.48 [29.26, 29.71] | 0.4306 | 0.0003 |
| 4 | 169.07 [168.86, 169.29] | 24.16 [24.08, 24.23] | 0.6279 | 0.0002 |
| 8 | 164.83 [163.77, 165.89] | 22.41 [21.16, 23.69] | 0.6607 | 0.0027 |
| 12 | 129.13 [128.01, 130.24] | 21.06 [20.98, 21.14] | 0.7532 | 0.0162 |
| 16 | 98.02 [94.90, 100.57] | 19.66 [16.13, 21.46] | 0.9229 | 0.0854 |
| 20 | 79.55 [74.69, 84.75] | 13.45 [11.50, 14.63] | 0.9962 | 0.2030 |

均值随 worker count 总体下降，且 CPU busy/PSI 显示从无压力逐步进入饱和；compute20 的 0.9962 busy 和 0.2030 PSI some 是有效的持续满载状态。compute12 的 Decode 五次均在 20.96--21.21 token/s，稳定性很好；但三个负载点出现需要后续基线解释的混合状态：

- compute8：前三次约 21.14--21.18，后两次约 24.18--24.37 token/s；对应 CPU busy 从约 0.55 变为约 0.82。
- compute16：四次约 21.38--21.50，一次 12.61 token/s；慢样本的 CPU busy/PSI 同时更高。
- compute20：首次 9.60，随后四次 13.91--14.80 token/s；总体变异明显高于 compute12。

无负载首个 measured request 的 Prefill/Decode 为 160.23/29.07，低于其余四次，说明新启动服务的首个完整 p1024 请求仍可能带有 p256 warmup 未覆盖的冷态或系统噪声。pilot 不删除该样本，也不事后修改 warmup；该现象作为正式重复协议设计依据。

dynamic B1 的 compute20 Decode 低于此前独立 D block 的 16.31 token/s，同时本次 PSI some 0.2030 高于 D 的 0.1700。当前 telemetry-off 数据无法区分策略选择、负载历史、热/功耗稳态和随机系统竞争；按照预注册流程不修改 V5、不补挑选性 dynamic 点，也不把单个 block 用作最终结论。先以完全相同 source fingerprints 和 sweep 参数完成 packed CPU 与 fixed iGPU pilot，利用静态策略曲线判断异常区间更接近哪种执行路径；之后只对预先选定的交叉/高方差点做独立 telemetry 诊断。

下一步先停止 PID 44211 的 dynamic 服务，启动 telemetry-off packed CPU：

```bash
SCHEDULER_TELEMETRY_FILE= \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh packed-cpu-fixed
```

服务 ready 后运行：

```bash
python kt-kernel/bench/bench_running_server_load_sweep.py \
  --base-url http://127.0.0.1:30100 \
  --backend packed-cpu-fixed \
  --block-label pilot-b1 \
  --output-dir artifacts/running-server-sweeps/packed-cpu-fixed-pilot-b1-20260720
```

### 2026-07-20：packed-CPU fixed steady-load pilot B1

结果位于 `artifacts/running-server-sweeps/packed-cpu-fixed-pilot-b1-20260720/`。根 manifest complete，6/6 负载点、30/30 measured requests 成功；实际顺序与 dynamic B1 相同。server PID 60735、nice=5，policy=fixed、ratio=0、telemetry 为空；source fingerprints 与 dynamic B1 完全一致，extension SHA-256 仍为 `bf38fa420c069ca981ac28ac7842131fc1b52b93fa86332b263b94457ed12ead`。结束后没有背景负载残留。

| Workers | Prefill token/s | Decode token/s | CPU busy | PSI some |
|---:|---:|---:|---:|---:|
| 0 | 166.01 | 29.63 | 0.4356 | 0.0003 |
| 4 | 170.26 | 23.92 | 0.6250 | 0.0002 |
| 8 | 167.42 | 24.04 | 0.8252 | 0.0004 |
| 12 | 125.01 | 1.88 | 0.9992 | 0.0578 |
| 16 | 90.89 | 1.44 | 0.9994 | 0.2492 |
| 20 | 71.58 | 1.39 | 0.9990 | 0.4204 |

packed CPU 在 workers=0/4/8 下 Decode 稳定在约 24--30 token/s，但 workers=12 后出现断崖式下降，五个请求全部降至 1.71--2.11 token/s；workers=16/20 进一步稳定在约 1.4 token/s。高负载下 CPU busy 约 0.999，说明 nice=0 背景任务持续压制 nice=5 的 CPU Decode，而不是少数离群点。

dynamic V5 相对 packed CPU 的单 block 点估计：

| Workers | Prefill effect | Decode effect | Decode ratio | E2E reduction |
|---:|---:|---:|---:|---:|
| 0 | +0.62% | -0.49% | 1.00x | -0.22% |
| 4 | -0.70% | +0.97% | 1.01x | +0.63% |
| 8 | -1.54% | -6.81% | 0.93x | -6.54% |
| 12 | +3.30% | +1018.47% | 11.18x | +88.86% |
| 16 | +7.84% | +1264.35% | 13.64x | +90.04% |
| 20 | +11.14% | +863.97% | 9.64x | +86.72% |

这些是两个单 blocks 的点估计，尚未使用跨 block bootstrap，不作为最终显著性结论。但效应规模已经确定主要交叉区：workers=8 时 packed CPU 仍更快，dynamic 的混合状态带来约 6.8% Decode 损失；workers=12 时 packed CPU 已崩落，而 dynamic 仍为 21.06 token/s。动态调度的主要价值因此集中在背景负载越过 CPU 可用容量之后，下一轮正式矩阵应在 8--12 workers 之间加密负载点并做独立重复。

dynamic 在 workers=12/16/20 的 CPU PSI some 分别为 0.0162/0.0854/0.2030，明显低于 packed CPU 的 0.0578/0.2492/0.4204；这说明异构执行不仅提高推理速度，也减少推理与高优先级背景任务对 CPU 的共同争用。正式论文仍需测背景任务吞吐或完成时间，才能把该方向扩展成系统整体 QoS 结论。

下一步停止 packed CPU PID 60735，启动 telemetry-off fixed iGPU：

```bash
SCHEDULER_TELEMETRY_FILE= \
ENGINE_PRIORITY=low \
./perf-log/35b-test-cpu-igpu.sh igpu-fixed
```

服务 ready 后运行：

```bash
python kt-kernel/bench/bench_running_server_load_sweep.py \
  --base-url http://127.0.0.1:30100 \
  --backend igpu-fixed \
  --block-label pilot-b1 \
  --output-dir artifacts/running-server-sweeps/igpu-fixed-pilot-b1-20260720
```

### 2026-07-21：fixed-iGPU pilot B1 与三策略 steady-load 比较

fixed iGPU 结果位于 `artifacts/running-server-sweeps/igpu-fixed-pilot-b1-20260720/`。根 manifest complete，6/6 负载点、30/30 requests 成功；server PID 8321、nice=5，policy=fixed、ratio=1、telemetry 关闭。三策略的 sweep runner、service scheduler 和 extension SHA-256 完全一致，结束后没有背景负载残留。

| Workers | Prefill token/s | Decode token/s | E2E ms | CPU busy | PSI some |
|---:|---:|---:|---:|---:|---:|
| 0 | 97.45 | 16.62 | 46919.35 | 0.1156 | 0.0025 |
| 4 | 99.32 | 17.58 | 44603.17 | 0.3026 | 0.0278 |
| 8 | 99.83 | 18.20 | 43395.04 | 0.4934 | 0.0368 |
| 12 | 103.27 | 18.67 | 42210.39 | 0.6989 | 0.0206 |
| 16 | 101.68 | 18.89 | 41995.49 | 0.8947 | 0.0238 |
| 20 | 96.63 | 13.54 | 55087.10 | 0.9947 | 0.1294 |

fixed iGPU 并非与 CPU 负载完全独立：Decode 从无负载 16.62 上升到 workers=12/16 的约 18.7--18.9，随后在 workers=20 降至 13.54；Prefill 也在约 97--103 token/s 之间变化。这说明即使专家 kernel 固定在 iGPU，CPU 侧 attention/runtime、共享内存带宽、封装功耗和温度仍会影响端到端性能。无负载首个 p1024 Prefill 只有 80.44 token/s，其余约 100--103，再次确认正式协议需要在首个负载点前增加不计入 block 的全工作负载 conditioning。

新增 `report_load_sweep_comparison.py`，接受同一策略的多个 sweep blocks，按 worker count 聚合，并对 block 和 request 做 hierarchical bootstrap；它拒绝不完整 sweep、缺失负载点以及协议/source/extension identity 不一致。新增 2 项测试后，CPU+iGPU 实验相关回归集为 82 passed；新报告器通过 Black、`py_compile` 和 `git diff --check`。

三策略比较位于 `artifacts/running-server-sweeps/pilot-b1-three-strategy-comparison-20260721/`，使用 20000 次 bootstrap。报告中的“static oracle”严格指两个已观测固定策略中该指标更好的一个，不代表理论或 phase-aware 全局 oracle。当前每个策略只有一个 sweep block，CI 主要反映 request 内波动，不能估计跨 block 方差。

dynamic V5 的 static-oracle point attainment：

| Workers | Prefill attainment | Decode attainment | E2E attainment | Best static path |
|---:|---:|---:|---:|---|
| 0 | 100.62% | 99.51% | 99.78% | packed CPU |
| 4 | 99.30% | 100.97% | 100.64% | packed CPU |
| 8 | 98.46% | 93.19% | 93.86% | packed CPU |
| 12 | 103.30% | 112.78% | 115.49% | metric-dependent; dynamic beats both |
| 16 | 96.40% | 104.06% | 98.68% | fixed iGPU |
| 20 | 82.33% | 99.32% | 93.40% | fixed iGPU |

主要结论：

1. workers=0/4 时 dynamic 在预注册 2% practical margin 内贴近 CPU static oracle。
2. workers=8 时 dynamic Decode/E2E 分别落后 CPU static oracle 6.81%/6.54%，表明交叉区存在不必要 offload 或状态混合。
3. workers=12 时 dynamic Prefill 129.13、Decode 21.06，同时优于 packed CPU 和 fixed iGPU，证明 phase-aware 异构组合能够超过任一“全阶段固定设备”策略。
4. workers=16 时 dynamic 与 fixed iGPU 的 E2E 点估计相差 1.33%，但 dynamic 单次慢样本使 CI 很宽，需独立 blocks 判断。
5. workers=20 时 dynamic Decode 13.45，达到 fixed iGPU 的 99.32%；但 Prefill 79.55 仅达到 static oracle 的 82.33%，最终 E2E 落后 fixed iGPU 7.07%。因此极端持续负载下的主要缺口是 Prefill phase，而不是 Decode phase。
6. 相对 packed CPU，dynamic 在 workers=12/16/20 的 Decode 为 11.18x/13.64x/9.64x，E2E 减少 88.86%/90.04%/86.72%；研究方向的核心收益已经得到数量级证据。

pilot B1 是开发/筛选集。为了避免在发现问题后直接用同一数据宣称最终效果，暂不开始大规模重复。先做一个独立、非性能计分的 compute20 全层 telemetry 诊断，验证当前持续负载下是否仍为 CPU-Prefill/iGPU-Decode；若确认，则后续 V6 只针对极端负载 Prefill 保护，最终用不同 seed、增加 10-worker 点的独立 blocks 评价。V5 B1 原始结果永久保留，不覆盖、不从论文开发过程记录中删除。

先停止 fixed-iGPU PID 8321，启动全层 diagnostic dynamic：

```bash
ENGINE_PRIORITY=low \
SCHEDULER_TELEMETRY_FILE=artifacts/server-telemetry/dynamic-v5-pilot-compute20-all-layers.jsonl \
SCHEDULER_TELEMETRY_LAYER=all \
./perf-log/35b-test-cpu-igpu.sh dynamic
```

在背景终端启动持续 compute20：

```bash
python kt-kernel/bench/cpu_background_load.py \
  --kind compute \
  --workers 20 \
  --affinity free \
  --nice 0
```

确认 ready 后运行带 10 次 warmup 的热稳态诊断：

```bash
python kt-kernel/bench/bench_running_server.py \
  --base-url http://127.0.0.1:30100 \
  --run-label dynamic-v5__engine-low__compute20-all-layers-pilot-diagnostic \
  --workloads 1024:300 \
  --warmups 10 \
  --warmup-output-tokens 128 \
  --repetitions 1 \
  --seed 20260721 \
  --bootstrap-samples 500 \
  --scheduler-telemetry-file artifacts/server-telemetry/dynamic-v5-pilot-compute20-all-layers.jsonl \
  --fail-fast \
  --output-dir artifacts/running-server-bench/dynamic-v5-engine-low-compute20-all-layers-pilot-diagnostic-20260721
```

该请求仅用于 phase mapping、service cost、探索/reprobe/switch 状态分析；全层 telemetry 性能不得与 B1 telemetry-off 曲线比较。完成后先在背景终端 `Ctrl-C`，再停止 diagnostic engine。

### 2026-07-21：compute20 热稳态全层诊断与 V5 Prefill 根因

请求位于 `artifacts/running-server-bench/dynamic-v5-engine-low-compute20-all-layers-pilot-diagnostic-20260721/`，逐层分析位于 `artifacts/running-server-bench/dynamic-v5-compute20-all-layers-pilot-analysis-20260721/`。10 次 p256-o128 warmup 后采集一个 p1024-o300 请求，共 12040 个 measured telemetry events。请求窗口 CPU busy=0.9958、PSI some=0.2486，是有效的持续 compute20 热稳态。

诊断吞吐为 Prefill 71.79、Decode 14.82 token/s。它包含全层同步 JSONL 开销，不与 B1 telemetry-off 性能做数值比较；但 Prefill 与 B1 dynamic/packed CPU 的高负载方向一致。

| Diagnostic | Result |
|---|---:|
| Prefill actual CPU layers | 40/40 |
| Prefill CPU-policy layers | 40/40 |
| Decode actual iGPU layers | 40/40 |
| Decode iGPU-policy layers | 40/40 |
| Decode calls per layer | 300 |
| Exploration calls | 0 |
| Reprobe calls | 0 |
| Request-window switch delta | 0 |

请求级 Prefill mean load 为 0.6637，各层 Prefill 事件采样范围约 0.3582--0.8755；所有层 `high_load_epoch=false`、`switch_count=0`，每层已有 11 个 CPU samples、0 个 iGPU samples。Prefill CPU service sample 为约 0.0487--0.0672 ms/row，iGPU 尚未被采样，因此不能从 V5 telemetry 内部形成 Prefill 两臂成本比较。

代码根因明确：V5 默认 Prefill `load_low=0.99/load_high=1.00`。`choose_igpu_ratio()` 在 `load <= low` 时无条件选择 CPU 并清除 high-load epoch；本次 0.6637 远低于 0.99，所以即使系统 CPU busy 已达 99.6%，Prefill 也永远不会进入“3 次 CPU + 10 次 iGPU”的成本校准。Decode 使用独立 shared service-cost scheduler，已稳定选择 iGPU；逐层报告中的最终 CPU/iGPU 2.43786/0.08399 ms/row 是 Decode 成本快照，不得误写为 Prefill 成本。

因此 workers=20 的 E2E 缺口是可复现的 V5 设计限制，而非误切换：极端持续竞争时 Prefill 仍固定 CPU。不能直接把 Prefill 门限改成 0.6637，因为 workers=8 应继续使用 CPU、workers=12 的 dynamic CPU Prefill 仍优于 fixed iGPU；需要先测这些状态下同一内部 load signal 的分布，确定 guard band，再设计 V6。pilot B1 和 V5 source fingerprints 保持不变，V6 将使用独立 policy/启动入口，避免覆盖 V5 复现路径。

### 2026-07-21：学位论文优先的研究范围冻结

在完成三策略 B1 pilot 和 compute20 根因诊断后，研究目标调整为优先满足硕士学位论文完成要求，V6 与额外 workers=8/12/16 telemetry 诊断暂停。此前计划的 compute8 命令取消，不再执行。当前 V5 作为学位论文系统最终版本，算法和参数继续冻结；论文后续投稿或扩展研究再以 B1 作为开发集设计 V6，并使用独立 holdout blocks 评价。

该决策不会使现有数据失效。学位论文可使用的证据包括：VNNI/SYCL 算子实现与历史性能记录、共享权重布局、V5 调度方法、fixed CPU/fixed iGPU/phase-fixed 消融、无负载与 compute20 oracle tracking、双向切换、V4/V5 reprobe 稳定性、三策略六负载点 B1 曲线，以及无负载/compute20 全层 telemetry。B1 每策略只有一个 sweep block，论文应明确将 bootstrap CI 表述为 request-level 或单-block 不确定性，不声称已经估计跨 block 方差。

论文结论采用保守边界：V5 在 workers=0/4 接近 CPU static oracle，在 workers=12 显著优于两种全阶段固定策略，在 workers=12/16/20 相对 packed CPU 获得数量级 Decode 加速和 86.72%--90.04% E2E 降低；workers=8 的约 6.5% E2E 损失和 workers=20 的约 7.1% fixed-iGPU E2E 差距作为局限性报告，不再为消除这两个点继续调参。

从现在开始不再安排大规模硬件测试。下一阶段工作改为：

1. 建立论文结果索引，把可引用 artifact、协议、指标和结论映射到章节/图表。
2. 整理 VNNI、SYCL、共享布局和动态调度三个技术章节的公式、伪代码与实现说明。
3. 从现有 CSV/JSONL 生成论文主表和负载曲线图，不手工抄写数值。
4. 撰写实验威胁与局限性，包括单机/单模型、单 sweep block、CPU 混合核心、热/功耗状态和 telemetry 扰动。
5. 只有在论文初稿或导师审阅明确发现证据缺口时，再执行最小补充实验；不预先重跑全部矩阵。

记录本条时 compute20 背景父进程 PID 21760 和全层 diagnostic engine PID 20884 尚在运行。用户应先在背景终端 `Ctrl-C` 停止 PID 21760，再在引擎终端 `Ctrl-C` 停止 PID 20884；无需启动 compute8。
