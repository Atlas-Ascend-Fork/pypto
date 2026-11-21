# AI CPU Kernel为什么没有block_dim、sk_cap和schedmode

## 问题现象描述

使用`@pypto_pro.language.jit(mode="aicpu_kernel")`时：

- `kernel[stream, block_dim]`启动形式被拒绝，报错包含`block_dim has no effect`：

```text
pypto_pro._errors.InvalidArgument: [jit.py:2695][PRO_RUNTIME]:ErrCode: F0000E! Enum: ExternalError::INVALID_ARGUMENT. Expected kernel[stream] for an aicpu kernel; a block_dim has no effect (the shim always launches <<<1>>>)
```

- 传入sk_cap、schedmode，或pipeline、tiling_key、datatype、sanitizer等选项时抛出NotSupported，报错信息逐项点名（同时传入多个不支持的选项时会全部列出）：

```text
pypto_pro._errors.NotSupported: [jit.py:2362][PRO_RUNTIME]:ErrCode: F00005! Enum: ExternalError::NOT_IMPLEMENTED_ERROR. jit() accepts no options for aicpu kernels (got: sk_cap)
```

## 可能原因

AI CPU Kernel运行在SoC控制CPU上，与AI Core Kernel的执行模型不同：

- **没有启动规模配置**：AI CPU Kernel单任务串行执行（下发固定为`<<<1>>>`），不存在多核切分，block_dim没有语义。若允许传入，会被校验后静默丢弃，因此前端直接拒绝，避免造成"已生效"的误解。
- **sk_cap依赖AI Core的SK基础设施**：SuperKernel融合的对象是AI Core子核函数的二进制，`__sk__`入口与`SK_BIND`记录是AI Core编译产物；cap位对应的DCCI、早启同步、扩核均为AI Core硬件机制。AI CPU工具链没有对应设施，AI CPU任务以独立任务形式与SK容器按Stream依赖衔接，不参与融合，详见[SuperKernel](../../programming_guide/pro/development/advanced_programming/superkernel.md#ai-cpu-kernel与superkernel的关系)。
- **schedmode是AI Core入口限定符**：`__schedmode__`写入AI Core Kernel的`__global__`入口，控制batch下发模式；AI CPU任务的下发由AI CPU运行时管理，没有对应限定符。
- **pipeline、tiling_key、datatype、sanitizer为Tile/SIMT专属机制**：AI CPU Kernel没有Tile、多核执行域与设备侧插桩设施，装饰时直接抛出NotSupported。

## 处理步骤

- 使用`kernel(...)`（当前Stream）或`kernel[stream](...)`（显式Stream）启动AI CPU Kernel；
- 移除报错点名的选项，AI CPU模式仅接受arch、name、compile_timeout、max_threads，完整能力边界参见[AI CPU编程范式](../../programming_guide/pro/programming_paradigm/AICPU/programming_paradigm.md)中的编程面说明；
- 需要多核并行、SuperKernel融合或batch调度的计算放在AI Core Kernel（SIMD/SIMT）中实现，AI CPU Kernel承担元数据规划等串行逻辑。
