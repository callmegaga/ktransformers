# CPU-iGPU GPTQ INT4 动态调度后端

`CPU_IGPU_GPTQ_INT4` 后端让 MoE 模型中未放在 CUDA GPU 上的 GPTQ INT4 experts 复用同一份 packed 权重，并根据外部 CPU 竞争在 AVX-VNNI CPU 与 Intel iGPU 之间选择执行设备。

该后端只调度 MoE expert 计算。Attention、Embedding、LM Head，以及通过 `--kt-num-gpu-experts` 保留的 experts，仍使用原有执行路径。

## 支持范围

- Linux x86-64。
- 支持 AVX-VNNI-256 的 Intel CPU。
- 支持 Level Zero 和 shared USM 的 Intel iGPU。
- 对称 GPTQ INT4 MoE 权重，`sym=true`、`desc_act=false`。
- group size 为 32 的正整数倍，最大为 256；已重点验证 group size 128。
- 单 NUMA subpool、`tensor_parallel_size=1`。
- 主要面向 `--max-running-requests 1` 的端侧单请求推理。

CPU 和 iGPU 使用同一份 output-major packed `qweight`、`scales` 和 `weight_sums`。CPU 端只创建非拥有型视图，不保存第二份 expert 权重。

## 构建

先加载 Intel oneAPI 环境，再启用 SYCL 构建：

```bash
source /opt/intel/oneapi/setvars.sh
CPUINFER_USE_SYCL=1 ./install.sh
```

构建后检查扩展符号：

```bash
python - <<'PY'
import kt_kernel_ext.moe as moe
print(hasattr(moe, "CPUiGPUGPTQInt4_MOE"))
PY
```

预期输出为 `True`。Intel GPU 设备与 render node 权限的配置方法参见 [SYCL GPTQ INT4 后端教程](SYCL-GPTQ-INT4-Tutorial_zh.md)。

## 启动

在原有 SGLang 启动命令中使用：

```text
--kt-method CPU_IGPU_GPTQ_INT4
```

动态调度是默认策略，无需设置版本号或实验开关：

```bash
export KT_CPU_IGPU_POLICY=dynamic
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
```

其余模型路径、显存、CPUInfer worker 和 CUDA expert 参数与 `SYCL_GPTQ_INT4` 后端相同。

## 动态策略

后端使用 `qlen == 1` 判断 Decode，其余 forward 视为 Prefill。监控线程每 50 ms 从 `/proc/stat` 读取整机 CPU 时间，并排除推理进程自身消耗，形成外部 CPU busy 的 EWMA。

默认迟滞阈值如下：

| 阶段 | CPU 阈值 | iGPU 阈值 | minimum dwell |
| --- | ---: | ---: | ---: |
| Decode | `0.45` | `0.55` | 4 |
| Prefill | `0.65` | `0.75` | 2 |

负载不高于 CPU 阈值时选择 CPU，不低于 iGPU 阈值时选择 iGPU，处于两个阈值之间时保持原设备。minimum dwell 限制连续切换之间的最少决策次数。

Decode 状态按 MoE 层维护。Prefill 状态在使用同一 worker pool 的所有 MoE 层之间共享：最低层号在每个 forward group 更新一次选择，其余层读取同一结果，避免一次 Prefill 内出现层间设备分裂。长上下文使用 chunked Prefill 时，每个 chunk 可以重新决策。

## 配置

通常应保留默认值。不同硬件需要重新标定时，可使用以下环境变量：

| 环境变量 | 默认值 | 说明 |
| --- | ---: | --- |
| `KT_CPU_IGPU_DECODE_LOAD_LOW` | `0.45` | Decode 切回 CPU 的低阈值 |
| `KT_CPU_IGPU_DECODE_LOAD_HIGH` | `0.55` | Decode 切到 iGPU 的高阈值 |
| `KT_CPU_IGPU_PREFILL_LOAD_LOW` | `0.65` | Prefill 切回 CPU 的低阈值 |
| `KT_CPU_IGPU_PREFILL_LOAD_HIGH` | `0.75` | Prefill 切到 iGPU 的高阈值 |
| `KT_CPU_IGPU_LOAD_EWMA_ALPHA` | `0.25` | 外部 CPU busy 的 EWMA 系数 |
| `KT_CPU_IGPU_LOAD_SAMPLE_MS` | `50` | CPU 负载采样周期，最小为 10 ms |
| `KT_CPU_IGPU_DECODE_MIN_DWELL` | `4` | Decode 最小驻留决策数 |
| `KT_CPU_IGPU_PREFILL_MIN_DWELL` | `2` | Prefill 最小驻留决策数 |

阈值必须满足 `0 <= LOW < HIGH <= 1`，EWMA 系数必须位于 `(0, 1]`，minimum dwell 必须为正整数。

## 固定模式

固定模式主要用于回归测试、设备诊断和硬件标定。

固定使用 packed VNNI CPU：

```bash
export KT_CPU_IGPU_POLICY=fixed
export KT_CPU_IGPU_RATIO=0
```

固定使用 SYCL iGPU：

```bash
export KT_CPU_IGPU_POLICY=fixed
export KT_CPU_IGPU_RATIO=1
```

分别固定两个阶段，例如 CPU Prefill 与 iGPU Decode：

```bash
export KT_CPU_IGPU_POLICY=phase-fixed
export KT_CPU_IGPU_PREFILL_RATIO=0
export KT_CPU_IGPU_DECODE_RATIO=1
```

固定比例可以设置为 0 到 1 之间的值。中间值按 expert 的实际 routed rows 近似划分；Decode 中 CPU 与 iGPU 子任务可以并行执行。正式动态策略只选择 0 或 1，不执行在线探索或服务代价校准。

## 常见问题

### 提示后端未编译

确认使用 `CPUINFER_USE_SYCL=1` 重新构建，并且启动终端已加载 oneAPI 环境。

### 提示 CPU 不支持 AVX-VNNI-256

该后端没有 AVX2 回退，因为 CPU 计算路径依赖 packed AVX-VNNI kernel。可以改用 `SYCL_GPTQ_INT4` 或其它 CPU 后端。

### 动态策略始终选择 CPU

监控线程需要至少完成一次采样。还应确认背景任务确实占用 CPU，而不是处于 I/O 等待状态。该信号表示可观测的外部 CPU busy，不等同于 worker 数占总线程数的比例。

### iGPU 固定模式仍受 CPU 负载影响

SYCL runtime、路由、结果合并、共享内存带宽和整机功耗仍会使用或受制于 CPU，因此固定 iGPU 并不代表端到端推理与 CPU 完全隔离。
