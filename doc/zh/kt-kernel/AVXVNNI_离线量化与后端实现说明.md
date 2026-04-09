# AVX-VNNI 离线量化与后端实现说明

本文整理今天围绕 `kt-kernel` 完成的 AVX-VNNI 相关改动，目标是帮助理解这次功能扩展的设计思路、实现路径、测试方式，以及如何在真实模型上验证结果。

## 1. 背景与目标

这次工作的目标不是只增加一个新 kernel，而是把一条完整链路打通：

1. 为 CPU MoE 后端增加基于 AVX-VNNI-256 的 `INT8` / `INT4` 实现。
2. 支持把 `BF16`、`FP16`、`FP8` 模型离线量化为适合 AVX-VNNI 的 CPU 权重格式。
3. 让框架在推理时能直接加载离线量化后的 CPU experts，进行混合推理。
4. 保持接入方式尽量贴近现有 `AMXINT8` / `AMXINT4` 方案，降低改动面和维护成本。
5. 在当前机器没有 AMX 的前提下，先确保功能跑通，并验证相比旧路径至少能正确落到 VNNI 后端。

这次选择的真实模型是：

- 原始模型：`/home/wy/Work/models/Qwen3.5-35B-A3B-FP8`
- 离线量化 INT8 输出：`/home/wy/Work/models/Qwen3.5-35B-A3B-FP8-AVXVNNI_INT8-NUMA1`
- 离线量化 INT4 输出：`/home/wy/Work/models/Qwen3.5-35B-A3B-FP8-AVXVNNI_INT4-NUMA1`

## 2. 总体设计思路

整体设计遵循一个原则：尽量复用现有 AMX 的“上层接口”和“离线量化流程”，只在真正依赖 ISA 和数据布局的地方增加 AVX-VNNI 分支。

换句话说，这次改动分成三层：

1. **内核层**
   新增 AVX-VNNI INT8 / INT4 的 MoE kernel，实现真正的 CPU 计算与权重加载。

2. **运行时接线层**
   让 Python wrapper、`KTMoEWrapper`、CLI、SGLang 启动参数都认识 `AVXVNNI_INT8` / `AVXVNNI_INT4`。

3. **离线量化层**
   扩展 `convert_cpu_weights.py` 和 `merge_cpu_weights.py`，使其能把原始 BF16/FP16/FP8 experts 转成 AVX-VNNI 可直接加载的布局与元数据格式。

这样做的好处是：

- 上层调度逻辑几乎不需要重写。
- 新后端复用现有 `KTMoEWrapper` / `AMXMoEWrapper` 的工作流。
- 离线转换后的权重目录结构与 AMX 路径保持一致，便于推理时统一加载。

## 3. 内核层实现

新增了两个后端头文件：

- `kt-kernel/operators/avx2/int8_avxvnni-moe.hpp`
- `kt-kernel/operators/avx2/int4_avxvnni-moe.hpp`

这两个实现的核心思路是：

1. 复用现有 CPU MoE backend 的整体组织方式。
2. 保持与 AMX 离线量化语义一致，使离线转换脚本可以沿用同一套高层接口。
3. 针对 AVX-VNNI-256 做权重读取与计算路径适配。

这里的关键不是单纯“指令替换”，而是让 **权重布局、量化元数据、运行时 kernel 预期** 三者保持一致。否则即使指令集支持，也会出现以下问题：

- 权重能保存但不能正确加载
- 能加载但输出精度明显异常
- 算子能跑通但 decode/prefill 性能没有提升

## 4. C++ / Python 绑定接线

为了让新后端能被 Python 层和推理框架调用，做了以下接线：

### 4.1 C++ 导出

在 `kt-kernel/ext_bindings.cpp` 中注册并导出了两个新类：

- `AVXVNNI256Int8_MOE`
- `AVXVNNI256Int4_MOE`

这一步的作用是让 Python extension 能直接实例化对应 backend。

### 4.2 Python 能力探测与 wrapper

在 `kt-kernel/python/utils/amx.py` 中新增：

- `_HAS_AVXVNNI256_INT8_SUPPORT`
- `_HAS_AVXVNNI256_INT4_SUPPORT`

并扩展 `AMXMoEWrapper`，使其支持：

- `AVXVNNI_INT8`
- `AVXVNNI_INT4`

虽然类名里仍然叫 `AMXMoEWrapper`，但它实际上已经承担了“CPU 离线量化后端统一包装层”的职责，不再只服务于 AMX。

### 4.3 上层统一入口

在 `kt-kernel/python/experts.py` 中，`KTMoEWrapper` 也接入了：

- `AVXVNNI_INT8`
- `AVXVNNI_INT4`

这样做后，框架上层不需要知道底层究竟是 AMX 还是 AVX-VNNI，只需要传入 method 即可。

## 5. 离线量化实现

这次最重要的扩展之一，是让下面这些输入格式都能离线转换成 AVX-VNNI 友好的 CPU experts：

- `bf16`
- `fp16`
- `fp8`

### 5.1 convert_cpu_weights.py

`kt-kernel/scripts/convert_cpu_weights.py` 新增支持：

- `avxvnni_int8`
- `avxvnni_int4`

同时修复了一个重要问题：不仅要让 CLI 能识别新 method，还要让真实转换分支真的落到对应 backend，否则会出现“参数看似支持，实际没有执行”的假接线问题。

脚本输出时还会在 `config.json` 中写入量化元数据，例如：

```json
{
  "converted": true,
  "method": "avxvnni_int8",
  "backend": "AVXVNNI_INT8",
  "numa_count": 1
}
```

这份元数据的作用是：

- 让后续 merge 和运行时加载能识别权重格式
- 便于人工确认当前目录到底是不是 VNNI 版本

### 5.2 merge_cpu_weights.py

`kt-kernel/scripts/merge_cpu_weights.py` 增加了对以下前缀的识别：

- `AVXVNNI_INT8_*`
- `AVXVNNI_INT4_*`

因此离线量化输出的分层目录可以正确合并为最终 safetensors 目录，并补齐元数据。

### 5.3 为什么要离线量化，而不是直接在线算

原因和 AMX 路径一样，主要有三点：

1. **推理时延**
   在线把 BF16/FP16/FP8 转成 INT8/INT4 代价太大，尤其是 experts 数量很大时。

2. **布局优化**
   AVX-VNNI kernel 对数据排布有自己的偏好。离线阶段可以把权重提前整理成适合 kernel 的布局，而不是在推理时临时重排。

3. **工程可控性**
   量化、打包、merge、加载分阶段完成后，更容易测试、定位问题和做缓存复用。

## 6. CLI 与交互层改动

为了让新后端不仅能“代码里调用”，还能“用户侧正常使用”，这次也同步扩展了 CLI 相关入口：

- `kt-kernel/python/cli/commands/quant.py`
- `kt-kernel/python/cli/utils/quant_interactive.py`
- `kt-kernel/python/cli/utils/run_interactive.py`
- `kt-kernel/python/cli/completions/_kt`

这样做后，`kt` 命令行和自动补全都认识：

- `AVXVNNI_INT8`
- `AVXVNNI_INT4`

## 7. SGLang 侧兼容性处理

SGLang 服务启动时，本身 `server_args` 对 `--kt-method` 没有做严格枚举限制，因此新 method 可以直接透传。

但还有一个容易遗漏的点：

- `third_party/sglang/python/sglang/srt/layers/moe/benchmark_kt_ep.py`

这个 benchmark 脚本里原先的 `choices` 没有包含 AVX-VNNI 选项，因此已补充：

- `AVXVNNI_INT4`
- `AVXVNNI_INT8`

这一步虽然不影响主服务启动，但会影响基准测试和后续验证脚本的可用性。

## 8. 新增的测试

### 8.1 算子精度测试

新增：

- `kt-kernel/test/per_commit/test_moe_avxvnni_accuracy_int8.py`
- `kt-kernel/test/per_commit/test_moe_avxvnni_accuracy_int4.py`

用途：

- 用 PyTorch reference 作为基线
- 验证 AVX-VNNI INT8 / INT4 的前向输出误差是否在可接受范围内

### 8.2 离线量化与加载 roundtrip 测试

新增：

- `kt-kernel/test/per_commit/test_moe_avxvnni_offline_roundtrip.py`

用途：

- 先保存离线量化权重
- 再重新加载
- 验证整个保存/读取/执行链路是闭环的

### 8.3 转换脚本端到端测试

新增：

- `kt-kernel/test/per_commit/test_moe_avxvnni_convert_cpu_weights.py`

用途：

- 验证 `convert_cpu_weights.py`
- 验证 `merge_cpu_weights.py`
- 覆盖 `bf16` / `fp16` / `fp8` 输入路径

### 8.4 轻量 benchmark 脚本

新增：

- `kt-kernel/scripts/benchmark_moe_avxvnni.py`

用途：

- 在当前机器上快速测 decode / prefill
- 比较 VNNI 输出与 reference 的偏差
- 作为手工性能验证入口

## 9. 已完成的验证

### 9.1 Python 语法检查

已执行：

```bash
python -m py_compile ...
```

相关修改的 Python 文件通过语法检查。

### 9.2 单元/集成测试

已执行并通过：

```bash
pytest -q \
  kt-kernel/test/per_commit/test_moe_avxvnni_accuracy_int8.py \
  kt-kernel/test/per_commit/test_moe_avxvnni_accuracy_int4.py \
  kt-kernel/test/per_commit/test_moe_avxvnni_offline_roundtrip.py \
  kt-kernel/test/per_commit/test_moe_avxvnni_convert_cpu_weights.py
```

结果：

- `12 passed`

### 9.3 真实模型离线转换

已完成：

1. `FP8 -> AVXVNNI_INT8`
2. `FP8 -> AVXVNNI_INT4`

其中：

- INT8 最终目录约 `31G`
- INT4 最终目录约 `16G`
- 两者都成功 merge 成 21 个 safetensors 分片

### 9.4 真实模型服务启动验证

新增并验证了两个启动脚本：

- `30b-build-avxvnni-int8-fp8.sh`
- `30b-build-avxvnni-int4-fp8.sh`

验证结果：

- 服务可以成功启动
- `/model_info` 可访问
- `/v1/chat/completions` 返回 `200 OK`

更重要的是，日志中明确打印了：

```text
Created AVXVNNI256_INT8_MOE_TP
Created AVXVNNI256_INT4_MOE_TP
```

这说明运行时实际落到了新的 AVX-VNNI backend，而不是旧路径。

## 10. 精度结论

当前结果表明：

- `INT8` 的相对误差较小，行为与 AMXINT8 路径的预期一致
- `INT4` 的误差明显大于 `INT8`，但这与低比特量化本身的损失特性一致

需要注意的一点是：

- 在这类离线量化方案中，`INT8` 和 `INT4` 的精度主要由量化策略、scale 计算方式、分组方式、布局对齐方式共同决定
- 指令集本身不会直接改变“理论量化精度”
- 指令集真正影响的是执行方式和性能

所以如果 `AVXVNNI_INT8` 与 `AMXINT8` 使用相同量化语义，它们应当尽量接近；但 `INT4` 本身就会比 `INT8` 更敏感。

## 11. 这次实现里的一个重要经验

在纯 CPU 测试路径中，`KTMoEWrapper.forward(..., cuda_stream=0)` 不能简单视为绝对可靠，曾出现过输出为零的情况。

因此这次在做离线量化和真实模型验证时，更可信的路径是直接调用底层 backend：

- `wrapper.moe.forward(...)`

这说明：

- wrapper 层和 backend 层要分别验证
- 某些问题可能不是 kernel 精度问题，而是测试路径或调度路径问题

## 12. 当前测试体系中的一个缺口

虽然新增了 AVX-VNNI 离线量化测试，但目前项目的 CI 体系是靠 `register_cpu_ci(...)` 进行 AST 收集的。

目前：

- `test_moe_avxvnni_accuracy_int8.py` 已注册进 CPU CI
- `test_moe_avxvnni_accuracy_int4.py` 已注册进 CPU CI
- `test_moe_avxvnni_convert_cpu_weights.py` 还未注册
- `test_moe_avxvnni_offline_roundtrip.py` 还未注册

这意味着：

- 这两个离线量化相关测试现在可以手动跑
- 但默认不会进入项目现有的 PR CPU CI 链路

后续建议优先补上这部分。

## 13. 后续建议

如果继续沿这个方向演进，建议优先做以下几件事：

### 13.1 补齐测试注册

让离线量化和 roundtrip 测试正式进入 CPU CI，避免后续改动破坏转换链路。

### 13.2 增加 `resume-layer` 测试

这次真实 INT4 转换过程中实际用到了 `--resume-layer`，建议补一个小模型测试，验证中断续跑功能不会回归。

### 13.3 文档同步到 README

当前 README 主要描述 AMX 与现有后端，后续可以补充：

- `AVXVNNI_INT8`
- `AVXVNNI_INT4`
- 对应离线转换示例
- 对应启动方式

### 13.4 再看 decode 阶段优化

目前功能已打通，但 decode 阶段不一定有明显提升。后续如果要继续优化，应重点检查：

- 小 batch / 小 token 数量下的调度与访存开销
- kernel 访存布局是否完全匹配 VNNI256
- 是否存在 threadpool / NUMA / packing 开销掩盖了指令级收益

## 14. 总结

今天的工作，本质上完成了从“新增 AVX-VNNI kernel”到“真实模型可启动推理”的完整闭环：

1. 新增 AVX-VNNI INT8 / INT4 MoE backend
2. 扩展 Python wrapper 和 CLI
3. 扩展离线量化与 merge 脚本
4. 增加精度、roundtrip、转换测试
5. 在真实 Qwen3.5-35B-A3B-FP8 模型上完成 INT8 / INT4 离线量化
6. 用 SGLang 成功拉起真实服务并确认运行在 AVX-VNNI 后端

从工程角度看，这次不是“只加了两个 kernel 文件”，而是把 **kernel、权重格式、运行时接线、CLI、测试、真实模型验证** 一起补齐了。

如果后续继续学习代码，建议按下面顺序阅读：

1. `kt-kernel/operators/avx2/int8_avxvnni-moe.hpp`
2. `kt-kernel/operators/avx2/int4_avxvnni-moe.hpp`
3. `kt-kernel/ext_bindings.cpp`
4. `kt-kernel/python/utils/amx.py`
5. `kt-kernel/python/experts.py`
6. `kt-kernel/scripts/convert_cpu_weights.py`
7. `kt-kernel/scripts/merge_cpu_weights.py`
8. `kt-kernel/test/per_commit/test_moe_avxvnni_*.py`

按这个顺序看，会比较容易把“底层 kernel”与“上层工程接线”串起来。
