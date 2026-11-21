# AI CPU编程范式

## 概述

AI CPU Kernel是运行在SoC控制CPU上的串行任务，与运行在AI Core上的SIMD/SIMT Kernel属于不同的执行域。PyPTO Pro通过pypto_pro.language.jit的mode参数声明AI CPU Kernel：

```python
import pypto_pro.language as pl


@pl.jit(mode="aicpu_kernel")
def metadata_kernel(
    seq_lens: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64, pl.Input],
    metadata: pl.Tensor[[pl.DYNAMIC], pl.DT_UINT32, pl.Output],
    num_heads: pl.DT_INT32,
):
    ...
```

AI CPU Kernel适合承担算子中的控制流密集、串行依赖强、计算量小的部分，典型场景包括：

- 元数据（metadata）规划：如注意力算子的多核任务切分表、调度表生成；
- 动态shape预处理：读取Device上的shape类数据并产出后续AI Core Kernel消费的索引或计划；
- 轻量标量计算与查表：不适合占用AI Core资源的串行逻辑。

数据密集计算仍应使用SIMD或SIMT编程，参见[SIMD编程范式](../SIMD/programming_paradigm.md)与[SIMT编程范式](../SIMT/programming_paradigm.md)。

## 执行模型

与AI Core Kernel相比，AI CPU Kernel的执行模型有以下差异：

| 维度     | AI Core Kernel（SIMD/SIMT）               | AI CPU Kernel                                                |
| -------- | ----------------------------------------- | ------------------------------------------------------------ |
| 执行硬件 | AI Core（AIC/AIV）                        | SoC控制CPU（aarch64）                                        |
| 并行方式 | SPMD多核并行，block_dim/grid配置启动规模  | 单任务串行执行，无启动规模配置                               |
| 数据访问 | GM与片上存储（UB、L1 Buffer等），Tile编程 | 仅GM，逐元素标量读写                                         |
| 下发方式 | `kernel[stream, block_dim](...)`        | `kernel(...)`或`kernel[stream](...)`                     |
| 任务状态 | 无返回值约定                              | 状态ABI：0为KERNEL_STATUS_OK，1为KERNEL_STATUS_PARAM_INVALID |

AI CPU Kernel直接读写Device内存，输入、输出Tensor必须位于NPU上；启动相对于Host异步，任务失败在同步时以Stream错误形式上报。

## 与SuperKernel的关系

SuperKernel是AI Core侧的二进制融合技术，AI CPU Kernel不生成`__sk__`入口、不参与融合，在图中作为独立的AI CPU任务与AI Core任务（含SuperKernel容器）按Stream依赖顺序衔接。详细说明参见[SuperKernel](../../development/advanced_programming/superkernel.md#ai-cpu-kernel与superkernel的关系)。

## 进一步阅读

- [Kernel核函数创建](../../aicpu_development/kernel_function.md)：AI CPU Kernel的声明、参数、装饰器选项、状态ABI与启动方式。
- [Tensor创建和操作](../../aicpu_development/tensor_creation_and_operations.md)：Tensor声明与numel/getval/setval标量读写操作面。
