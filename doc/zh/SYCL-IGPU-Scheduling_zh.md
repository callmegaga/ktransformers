# Panther Lake 混合推理：CPUInfer / callback / SYCL 时间线研究

2026-10-16，接续 [CUDA dense decode 优化](SYCL-IGPU-CUDA-Dense_zh.md)。
平台/模型/专家放置保持不变：Core Ultra 7 356H / Intel `b0a0` / RTX 5060 Laptop，
Qwen3.5-35B-A3B-GPTQ-Int4，GPU experts8、CPUInfer10、chunk512。
仍是 CPU / NVIDIA / Intel iGPU 混合推理，未改变权重或激活量化精度。

## 本轮先纠正“CPU 开销”的归因

旧 CUPTI capture 没有完整 graph host-node 活动，不能把 kernel 时间和从 wall 中减掉后，
将剩余全部归为 CPU 或 iGPU。本轮补齐：

- 以独立 `LD_PRELOAD` collector 拦截 **CPUInfer 扩展的** CUDA13 `cudaLaunchHostFunc`，
  包装实际回调、保留 graph replay 的 context；不在回调中调用 CUDA/CUPTI。
- `cpu_backend/scheduling_trace.h` 加可选外部 recorder hook，关联 queue enqueue/run/sync
  和 NUMA submit/run。仅在进程启动前设置 `KT_SCHED_TRACE=1` 且预加载 collector 时记录。
- 关闭时不查询时间、不包装 task、不分配 trace buffer；原队列/锁/NUMA 分派路径不变。
  有一次初始化和分支，并非承诺绝对零机器指令成本。
- collector 对 host clock 与 CUPTI 做显式校准，实测 bracket 约 1.13µs；同时检查
  host callback 的 REALTIME / MONOTONIC_RAW 时长差，防止把时钟跳变当作调度。
- 32 个 decode graph 每个均对应 40 次 submit callback、40 次 sync callback、40 个 queue job、
  40 个 NUMA forward；未丢记录。queue job 与 enqueue 以唯一 id 关联。

**限制**：collector 读取当前 CPUInfer 参数的已知前缀布局，只适用于记录的这版二进制；
不能当成任意 CUDA callback 的通用 tracing 库。没有修改/关联 SYCL device 时钟到 host 时钟，
Chrome trace 只包含校准后的 host / NVIDIA，SYCL 设备事件单独汇总。

## 稳态测量

无 SYCL device profiling 的 host trace 中，每层 queue dispatch 从 enqueue 开始到执行开始
中位约 7.67µs；NUMA 转交约 0.54µs；queue sync 完成通知约 2.93µs。
每 token 对应这些阶段的和约 0.33ms / 0.025ms / 0.16ms。

sync callback 时间和约 7.04ms/token，**大部分包含实际专家执行与等待，不是纯 CPU 开销**。
这些 scope 相互嵌套，不能全部相加。

另开 SYCL device profiling，用最后两条累计记录的差分去掉早期 JIT/初始化，
覆盖 40 层各 16 次稳态 decode，平均每层：

|阶段|平均时间|
|---|---:|
|iGPU kernel 时间和|149.73µs|
|其中 fused expert|139.59µs|
|input quant / merge|4.50 / 5.63µs|
|device span|152.09µs|
|host submit|9.29µs|
|host wait|152.44µs|
|host route / input copy / output copy|1.21 / 0.31 / 0.46µs|
|executor host wall|163.77µs|

host wait 与设备执行重叠，不能相加。带 profiling 的平均数不能与无插桩生产吞吐直接比较。
完整累计值含 JIT，尤其首次 submit，**不能用累计均值作稳态归因**。

### 真正值得跟进的是 D2H 后的 callback 派发间隙

在校准后时间线，graph 内按层有序的 3 次 DTOH 后到 submit callback 开始，
1280 组的中位间隔约 **73.06µs/层**。用严格的三次 DTOH/层数量和顺序做配对；
CUDA runtime stream handle 不是 CUPTI streamId，不能把它们直接当成相同 id。

这不是 memcpy 的实际执行时间，也不是已经进入 TaskQueue 后的唤醒；
它包含驱动/graph host-node 派发等因素，观测到这一间隔并不代表能完整回收全部时间。

校准 trace 中，NUMA host forward 与 NVIDIA kernel 的观测重叠非常小。
仍不能把未覆盖区间全部归为 callback：它还可能包含传输、事件调度与其它工作。

## 异构 CPU 核心：NUMA 亲和性不足以表达大小核分工

本机 Linux 拓扑（2026-10-16，本机 CPU 编号，不可照搬到其它机器）：

|逻辑 CPU|PMU 类别|最高频率|`cpu_capacity`|共享缓存分组|
|---|---|---:|---:|---|
|0–3|`cpu_core`，P 核|4.7GHz|1004–1024|各自 L2；与 4–11 共享 L3|
|4–11|`cpu_atom`，较高频组|3.7GHz|714|4–7 / 8–11 各共享 L2；与 P 核共享 L3|
|12–15|`cpu_atom`，较低频组|3.3GHz|637|共享 L2；sysfs 未列出该组的 L3|

证据来自 `/sys/bus/event_source/devices/{cpu_core,cpu_atom}/cpus` 和各 CPU 的
`cpufreq/cpuinfo_max_freq`、`cpu_capacity`、`cache/index*/shared_cpu_list`。
低频 Atom 组在实验中按实际拓扑命名，不仅凭频率推断具体微架构身份。
`cpu_capacity` 不是本后端 AVX2/VNNI/SYCL 提交性能的实测比例，不能直接换算吞吐。
本机 16 个逻辑 CPU 对应 16 个物理核、没有 SMT，且**全部在 NUMA node0**。
`KT_SYCL_XMX_NUMA_DISPATCH=1` 是“启用 NUMA 分派路径”，**不是选择 NUMA node1**。

上一轮 trace 中，实际 NUMA/SYCL forward 已在 P0；CUDA callback 和 TaskQueue
却经常在 Atom 核间迁移。因此“NUMA 放置正确”不代表整条提交链都在快核上。
下面仅固定 TaskQueue 到 P3 的负结果，**不能推出 P 核无用或所有绑核均无收益**：
尚未隔离 CUDA callback、SGLang 主线程、TaskQueue 忙等与 SYCL 执行线程的联合放置。
尤其不能把忙等线程与提交线程绑定到同一个核心，再把争用造成的退化归因于核心类型。

后续测试按角色分别设置并验证亲和性，比较分离 P 核与部分 Atom 分工；保留同进程
未绑定对照并随机交错，记录真实 TID、允许 CPU、采样落点、CPU 时间、频率和温度。
只观察温度/频率不等于测量封装功率；本用户当前无权读取 RAPL `energy_uj`。
该研究不增加默认绑核，不改变系统 governor、功率上限或全机 CPU 配置。

### 补充的联合绑核实测

同一存活 server、同一轻量 callback 线程标识 interposer，原 CUDA callback 调度路径、
GPU experts8、iGPU v6、**opt-in both-row dense**；每组每长度 3 次、256 输出 tokens，
6 种配置按轮随机交错，合计 36 次有效请求。标识器只包装原回调并在首次执行时命名，
不记录时钟/事件；仍有非零 wrapper 成本，所以只用于本组策略相对比较，
**不是新的无插桩生产吞吐基线，也不是 mapped-memory 原型的绑核结果**。

|配置|1K decode tok/s|4K decode tok/s|
|---|---:|---:|
|原亲和性|42.90|41.54|
|四个关键角色分别 P0/P1/P2/P3|41.57|40.47|
|执行 P0，队列 Atom4，回调 P3|42.06|40.91|
|执行 P0，队列 Atom4，回调 Atom5|43.13|41.80|
|执行 P0，队列 Atom12，回调 Atom13|43.73|42.59|
|执行组 Atom6，队列 Atom4，回调 P3|39.56|38.45|

除原亲和性对照外，SGLang 主线程都固定 P1。执行组包含实际 NUMA 主线程以及继承其
线程名/亲和性的 SYCL helper；没有把同名 helper 误认成另一条专家执行线程。
其余运行时/CPUInfer 子 worker 的亲和性不变；采样显示 CPUInfer 子 worker 无累计 CPU ticks，
所以这些结论也不能外推成 AVX2/VNNI CPU 专家计算的最佳绑核方案。

更严格的单因素比较是表中“执行 P0，队列 Atom4，回调 P3”与最后一行：
只将执行线程组由 P0 改至 Atom6，1K/4K decode 分别下降 **5.93% / 6.00%**。
另一方面，四个关键角色分别固定到四个 P 核，比原设置平均慢约 **3.1% / 2.6%**。
执行保留 P0、队列/回调移至 Atom12/13 的候选平均快约 **1.9% / 2.5%**。
这支持“按角色分工”，但不证明所有大核绑核无用，也不证明低频核心天然更适合所有通知任务。

73 个观测线程的身份/亲和性检查通过，结束前已恢复；所有 36 次均完整输出 256 tokens。
首个 baseline 1K prefill 有冷启动离群，**不从这组数据宣称 prefill 提升**。
所有配置采样的 iGPU 实际频率中位数均为 2450MHz；没有功率测量，不能把差异直接归因于
省电或 GPU 睿频。三轮、单请求、同一启动的结果仍需独立启动和热稳态复验。
没有把约 2% 的候选收益设为默认，也没有将其叠加到上一轮 memop 原型数字上。
下一阶段原生 signal bridge 应继续区分执行/提交、通知和忙等角色，而不是全进程只用大核。

数据、精简逐请求记录、亲和性恢复与 46 项实际结果审计、8 项协议单测、graph callback
标识自测和复现脚本：`/home/wy/work/kt-perf-log/igpu-v10-hybrid-affinity/`。

## 未采纳的调度“简化”

同一实验二进制、native CUDA dense，每档 3 次、128 输出 tokens：

|模式|1K decode|4K decode|
|---|---:|---:|
|原路径，NUMA dispatch1|40.70|39.79|
|TaskQueue 固定 CPU3，NUMA dispatch1|40.07|39.06|
|TaskQueue 固定 CPU0，NUMA dispatch0|31.66|33.64|
|TaskQueue 固定 CPU3，NUMA dispatch0|37.49|35.98|

这些配置中没有因为少一次转交就更快，直接执行反而明显退化；这不否定尚未测试的联合大小核放置。
固定 CPU 功能和 feature probe 已撤回，只保留实验 patch；没有增加默认 CPU 绑核或忙轮询。
Linux 原先的 NUMA worker 亲和性和 `KT_SYCL_XMX_NUMA_DISPATCH=1` 保留。

另以 3 个依赖小 kernel 的 fresh-process 随机重复检查 SYCL queue：
本机 default / 明确 V2 的 out-of-order约 11.3–11.4µs，in-order约 11.0–11.1µs；
legacy约 12µs。只有很小的独立收益，没有模型证据，不改 queue / adapter 默认。
`/proc/maps` 同时出现两个 adapter 库不代表两个都在执行，不能凭 maps 确认实际 adapter。
本轮只按显式环境设置和观测结果记录，不依赖不稳定的内部运行时开关作为交付优化。

## 新方向：mapped stream-memory handshake 原型

本机 CUDA13 driver 支持 `cuStreamWriteValue32` / `cuStreamWaitValue32`，且可捕获到 graph 后重复执行。
原型在**独立实验 interposer**中，将 CPUInfer 的回调替换成 mapped pinned signal 的握手：

1. 保留原 stream 上的 D2H。
2. stream 写入 request；CPU poller acquire 观察后执行**原 submit / sync 函数**。
3. poller release 写 completion；stream wait 后继续 H2D。
4. request / completion 双向 reset 握手，避免 graph replay 将旧完成值误判为新任务。

不更改专家计算、队列与 iGPU 核心。40 层独立链、每层模拟 160µs 工作、动态输入校验、
30 次交错采样，callback / mapped 原型中位数约 **8.32 / 7.41ms**。
这是可行性微基准，不等于整模型收益，也没有包含真实 attention/dense 竞争。

### 完整模型正式对照：有收益，但仍是原型

相同 binary、GPU8、iGPU v6、CUDA dense both-row，先 memop 后 callback，每档 5 次、
每次 300 输出 tokens；seed20261016，关闭所有 trace/profiling。
唯一环境差异为显式实验标记和加载 memop interposer。

|输入|原 callback decode tok/s|memop 原型 decode tok/s|提升|TOPT ms/tok|Prefill 原→新 tok/s|
|---|---:|---:|---:|---:|---:|
|1K|42.99|46.60|+8.40%|23.26→21.46|1417.70→1431.50|
|4K|42.28|45.01|+6.44%|23.65→22.22|1541.17→1564.09|
|7.5K|40.77|44.24|+8.54%|24.53→22.60|1541.68→1559.62|

这是相对 both-row 的额外约 6–8.5% decode 收益，不把它写成已安全交付的生产性能。
Prefill 小幅差别不能据此推广成稳定收益。顺序测试没有完整隔离温度/功耗/频率。

7 条提示词 × 3 次 × 64 输出的检查均输出有限 logprob；短 EN/ZH/code token 与原对照相同。
长压力/英文/代码仍出现多个序列，其中有此前两组有限 control 没出现的序列；
例如压力/代码相对 callback 首轮分别在 index37/index25 分歧。
这些检查不是 PPL、并发/取消/故障安全验收，不能因专家计算代码没变就假定协议完全正确。

独立 CUPTI 请求中，原型 graph launch 间隔中位约 22.19ms；仍捕获到 32 次 graph、
6,112 次 both-row GEMV。driver 记录有 1,440 次 write32 / 1,440 次 wait32 API。
这确认进程实际加载并调用了 memop 路径，但 collector 没有完整 graph 内部 memop 活动时间，
不能把这些 API 调用次数当成 replay 的实际执行次数，也不声称得到全部信号协议时间线。

同为 GPU experts=0、both-row、其余配置一致的额外控制中，原 callback / memop 各重复 3 次，
6 次检查的 input/output logprob、output top-5 logprob 和生成 token 逐位一致。
这里只比较相同 GPU0 放置的同步协议，不代表 GPU0 与正常 GPU8 放置数值等价，
也不证明并发/设备异常/取消安全。

### 为什么不直接合入生产

该原型只有单请求实验约束，仍有以下未完成项：

- 忙轮询线程的 CPU 占用、整机功耗和热稳态未完成评估。
- 动态 graph/node 生命周期与 stream 销毁复用、重捕获的清理还不完整。
- 原回调异常传播、取消、退出、超时和设备错误路径没有生产级处理。
- 单一 poller 若在 sync 阻塞，多独立 stream 的进度可能出现互相阻塞；多请求/多 CPUInfer 并发未验收。
- stream memory 操作的跨 stream 依赖必须保留 CUDA 可见事件；不能只靠主机信号偷偷建立设备依赖。

所以 **interposer 不安装到全局、不作为默认环境、不进入交付脚本**。
它仅作为下一阶段实现安全原生协议的证据和实验 checkpoint。
不要将 `LD_PRELOAD` 原型当作可长期服务的后端。

## 当前交付与验证

生产默认仍为 iGPU v6 + native CUDA dense + 原 CUDA callback 链。
上一轮 `KT_IGPU_DENSE_PROFILE=both-row` 仍是可选候选，质量限制不变。
本轮生产改动只增加可关闭的调度记录 hook 和 CPUInfer 队列线程名，没有修改计算 kernel 或同步协议。
Intel 扩展仍报告 implementation_revision6，新增 C feature symbol
`kt_cpuinfer_scheduling_revision()` 用于诊断脚本拒绝旧扩展，不改变 Python API。
本轮重建会改变 `.so` hash，但计算精度/专家放置不变。

已检查：原 TaskQueue exception/drain 测试；trace 开关各 102 个 task 的关联、异常保留与无调用检查；
ASAN / UBSAN / leak 检查；Intel/启动/qlen **203 passed / 3 expected skipped**；
CUDA dense **37 passed**；perf.py **27 passed**。

数据、时间线、原始记录、排除 patch、握手原型与复现脚本：
`/home/wy/work/kt-perf-log/igpu-v9-scheduling/`。
硬件极限和完整模型质量仍未证明；不能将本轮诊断结果解释成对所有 shape/模型/并发的保证。
