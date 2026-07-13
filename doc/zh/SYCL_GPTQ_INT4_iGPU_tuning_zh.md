# SYCL_GPTQ_INT4 iGPU 后端调优记录

本文记录在 Intel iGPU 上为 `SYCL_GPTQ_INT4` MoE 专家后端做的一轮增量调优。目标是让
`/home/wy/Work/models/Qwen3.5-35B-A3B-GPTQ-Int4` 在 CPU + iGPU 混合推理框架中更稳定、更快地执行专家计算，并保留足够的诊断开关，便于后续继续优化。

日期：2026-07-13

## 背景

原始想法是把 MoE 专家计算从 CPU AVX2/VNNI 路径扩展到 CPU 集成显卡上。iGPU 理论上更适合矩阵计算，但本模型的 decode 阶段是 `M=1` 的大量小矩阵专家计算，实际瓶颈不只是算力，还包括：

- SYCL/OpenCL/Level-Zero 的 kernel 提交和排队延迟。
- iGPU 访问共享内存中的 GPTQ 权重的延迟。
- 大 fused kernel 在 Intel iGPU 上触发的调度等待。
- MoE top-k 专家稀疏访问导致 batch 化收益有限。

本轮调优后的稳定推荐路径是：保留小 kernel per expert，减少 CPU 中间拷贝和等待点，通过 async submit 和有限 pipeline 覆盖部分提交/执行延迟。

## 主要代码变动

### `kt-kernel/operators/avx2/moe_base.hpp`

新增 decode 分阶段计时：

- 环境变量：`KT_MOE_DECODE_TRACE_EVERY`
- 日志格式：

```text
[MOE decode] layer=... avg_total=... avg_setup=... avg_pack=...
avg_gate_up=... avg_activation=... avg_down_pack=... avg_down=...
avg_merge=... avg_active=... gate_up_fused=...
```

新增后端扩展 hook：

- `use_fused_gate_up_decode()`
- `decode_gate_up_activation(...)`
- `use_fused_down_decode()`
- `decode_down_projection(...)`

这些 hook 允许 SYCL 后端在 decode 阶段替换默认的 CPU activation / down pack / worker-pool GEMM 流程，同时不影响其他 AVX2/AMX 后端的默认行为。

### `kt-kernel/operators/sycl/gptq_int4_sycl-moe.hpp`

新增或强化的实验路径：

- `per_gemm + subgroup`：每个专家一个小 GEMM kernel，适配 decode 的 `M=1`。
- `gate_up_fuse`：把 gate GEMM、up GEMM 和 activation 合并到每个专家的小 kernel 内，直接写入 `down_ba_`。
- `gate_up_async`：每个 active expert 的 gate/up 小 kernel 全部 submit 后统一等待。
- `down_async`：每个 active expert 的 down 小 kernel 全部 submit 后统一等待，再拷回 CPU merge 缓冲。
- `gate_up_down_pipeline`：down kernel 依赖对应 expert 的 gate/up event 提交，避免所有 gate/up 完成后才开始 down。
- `fast_silu`：使用 `sycl::native::exp` 的 SiLU 近似路径。
- `gate_up_q8`：实验性 q8 激活 + int4 权重整数累加路径，当前验证为变慢，默认关闭。

同时保留了之前用于诊断的大 fused 路径、device scratch、device cache、queue profiling 等开关。它们对定位瓶颈有帮助，但不是当前推荐默认路径。

### `perf-log/35b-build-sycl-int4.sh`

脚本中加入并转发本轮调优相关环境变量。当前推荐默认值：

```bash
KT_SYCL_INT4_DECODE_MODE=per_gemm
KT_SYCL_INT4_PER_GEMM_KERNEL=subgroup
KT_SYCL_INT4_PER_GEMM_SUBGROUP=32
KT_SYCL_INT4_GATE_UP_FUSE=1
KT_SYCL_INT4_GATE_UP_BATCH=0
KT_SYCL_INT4_GATE_UP_ASYNC=1
KT_SYCL_INT4_DOWN_ASYNC=1
KT_SYCL_INT4_GATE_UP_DOWN_PIPELINE=1
KT_SYCL_INT4_GATE_UP_Q8=0
KT_SYCL_INT4_FAST_SILU=1
KT_SYCL_QUEUE_PROFILING=0
KT_SYCL_QUEUE_IN_ORDER=0
```

注意：当前主仓库把 `perf-log/` 识别为未跟踪的嵌套 Git 仓库。如果要提交 `35b-build-sycl-int4.sh`，需要在 `perf-log` 自己的仓库里提交，或者先明确调整仓库结构，避免在主仓库中误提交嵌入式 Git 仓库。

## 实验过程与结论

不同轮次的输入长度和输出长度并不完全一致，所以 TPOT 只能做近似比较；更稳定的判断依据是 `[MOE decode]` 的单层分项耗时。

| 阶段 | 关键配置/改动 | 代表结果 | 结论 |
| --- | --- | --- | --- |
| 大 fused kernel | 单层融合 gate/up/down/merge | `avg_total ~80ms/layer`，device kernel 约 `1ms`，但 host 可见等待约 `60-80ms` | iGPU/OpenCL/Level-Zero 对该大 kernel 调度很差，不适合作为主路径 |
| flat_i8 | 层内 flat 化 + int8 激活 | 约 `160ms/layer` | 变慢，放弃 |
| per-gemm subgroup | 每个专家小 kernel，SG=8/16/32 | SG=32 最好，TPOT 约 `214.5ms` | 小 kernel 更适合当前 iGPU decode |
| gate/up fused per expert | gate + up + activation 合为专家小 kernel | TPOT 约 `174.1ms`，`gate_up_act ~0.303ms/expert` | 明显有效，移除 CPU activation 和 down pack |
| gate/up batch single kernel | 把 active experts 批到一个 kernel | `gate_up ~88ms/layer` | 大 batch kernel 触发调度问题，保持关闭 |
| gate/up async | active experts 小 kernel 全部 submit 后等待 | TPOT 约 `161.3ms`，`gate_up ~1.76ms/layer` | 有效，减少串行等待 |
| down async | down 小 kernel 全部 submit 后等待 | TPOT 约 `131.4ms`，`avg_down 2.322ms -> 0.717ms` | 本轮最大收益点 |
| gate/up-down pipeline | down 依赖各自 gate/up event | TPOT 约 `129.4ms`，`avg_total ~2.275ms/layer` | 小幅收益，保留 |
| gate/up q8 | 动态 int8 激活 + int4 权重 | TPOT 约 `138.4ms`，`avg_total ~2.736ms/layer` | 变慢，默认关闭 |
| fast SiLU | `sycl::native::exp` | TPOT 约 `128.1ms`，`avg_total ~2.152ms/layer` | 小幅收益，默认开启 |

最终较稳定的一组日志：

```text
Total: 79589ms | TTFT: 3108ms | TPOT: 128.1ms | In: 4 | Out: 601
[MOE decode] layer=39 calls=600 avg_total=2.152ms avg_setup=0.051ms avg_pack=0.028ms avg_gate_up=0.099ms avg_activation=0.000ms avg_down_pack=0.000ms avg_down=1.972ms avg_merge=0.002ms avg_active=6.55 gate_up_fused=1.00
```

其中 pipeline 打开时，`avg_gate_up` 主要是 submit 时间，真实 gate/up 执行等待会被计入后续 `avg_down`。因此此模式下要重点比较 `avg_total` 和最终 TPOT，而不是单独看 `avg_gate_up`。

## 当前推荐启动方式

在 `perf-log/35b-build-sycl-int4.sh` 默认值已设置好的情况下，可直接运行：

```bash
cd /home/wy/Work/ktransformers/perf-log
./35b-build-sycl-int4.sh
```

如果要显式指定当前推荐配置：

```bash
KT_SYCL_INT4_GATE_UP_FUSE=1 \
KT_SYCL_INT4_GATE_UP_BATCH=0 \
KT_SYCL_INT4_GATE_UP_ASYNC=1 \
KT_SYCL_INT4_DOWN_ASYNC=1 \
KT_SYCL_INT4_GATE_UP_DOWN_PIPELINE=1 \
KT_SYCL_INT4_GATE_UP_Q8=0 \
KT_SYCL_INT4_FAST_SILU=1 \
./35b-build-sycl-int4.sh
```

## 已验证不推荐的路径

- `KT_SYCL_INT4_GATE_UP_BATCH=1`
  - 单个 batch kernel 变成约 `88ms/layer`，明显退化。
- 大 fused decode kernel
  - device kernel 时间不大，但排队/启动等待异常高。
- `KT_SYCL_INT4_GATE_UP_Q8=1`
  - 当前实现没有真正用到 DPAS/dp4a 矩阵指令，还额外引入动态量化和 scale 开销，实测变慢。
- `KT_SYCL_QUEUE_PROFILING=1`
  - 诊断有用，但可能显著放慢启动和推理，甚至触发 watchdog。

## 当前瓶颈判断

以 `TPOT ~128ms` 和 `avg_total ~2.15ms/layer` 粗算：

- MoE 专家计算约 `2.15ms * 40 = 86ms/token`。
- 总 TPOT 约 `128ms/token`。
- 剩余约 `40ms/token` 在 MoE 外部，包括 attention、shared experts/非专家层、SGLang 调度、Python/CUDA 侧同步等。

因此，后续继续优化 SYCL MoE 内部仍有空间，但边际收益会变小。下一阶段建议先把 MoE 外部耗时拆出来，确认总 TPOT 的剩余瓶颈。

## 后续优化方向

1. 分离 MoE 外部耗时
   - 增加或打开 attention、shared expert、scheduler 侧 timing。
   - 判断 `~40ms/token` 主要来自哪里。

2. 更深入的 iGPU dot 路径
   - 当前 q8 路径只是普通 int 累加，没有真正利用 DPAS/dp4a。
   - 如果继续做 q8/int4，需要研究 Intel SYCL `joint_matrix`/DPAS 或 OpenCL Intel subgroup 扩展。

3. down 输出合并迁移到 GPU
   - 当前 down 后仍然回到 CPU 做 router weighted merge。
   - 如果能在 GPU 上完成 down + weighted sum，可能减少回拷和 CPU merge，但要避免重新落入大 fused kernel 调度陷阱。

4. 专家权重布局
   - 当前权重布局沿用 GPTQ `[K/8, N]`。
   - 小 kernel 可能受内存访问模式限制，后续可考虑 per-expert 预重排，但需要权衡加载时间和内存占用。

5. 设备后端选择
   - `opencl:gpu` 与 `level_zero:gpu` 需要持续 A/B。
   - 本轮中 OpenCL 路径曾表现更稳定；Level-Zero immediate command list 对旧 fused 路径有小幅收益，但不是根本解。

## 提交建议

主仓库建议提交内容：

- `kt-kernel/operators/avx2/moe_base.hpp`
- `kt-kernel/operators/sycl/gptq_int4_sycl-moe.hpp`
- `doc/zh/SYCL_GPTQ_INT4_iGPU_tuning_zh.md`
- `doc/SUMMARY.md`

`perf-log/35b-build-sycl-int4.sh` 位于嵌套 Git 仓库 `perf-log/`，建议单独在该目录提交：

```bash
cd /home/wy/Work/ktransformers/perf-log
git status
git add 35b-build-sycl-int4.sh
git commit -m "Add SYCL INT4 iGPU launch defaults"
```

主仓库可参考提交信息：

```bash
git add kt-kernel/operators/avx2/moe_base.hpp \
        kt-kernel/operators/sycl/gptq_int4_sycl-moe.hpp \
        doc/zh/SYCL_GPTQ_INT4_iGPU_tuning_zh.md \
        doc/SUMMARY.md
git commit -m "Add SYCL GPTQ INT4 iGPU decode tuning path"
```

