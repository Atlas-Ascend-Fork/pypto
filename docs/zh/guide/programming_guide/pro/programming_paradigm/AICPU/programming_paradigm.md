# AI CPU编程概述

## 概述

AI CPU Kernel用于在运行时计算tiling分核策略、任务划分、运行时值依赖等调度信息。PyPTO Pro通过pypto_pro.language.jit的mode参数声明AI CPU Kernel：

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

AI CPU Kernel适用的典型场景包括：

- tiling分核与偏移计算：如注意力算子的多核任务切分表、调度表生成；
- 动态shape预处理：读取Device上的shape类数据并输出后续AI Core Kernel依赖的索引和控制流信息。

## 数据访问

AI CPU Kernel直接对Device GM内存读写进行计算，输入、输出Tensor必须位于Device GM上。

## 与SuperKernel的关系

SuperKernel是AI Core侧的二进制融合技术，AI CPU Kernel不生成`__sk__`入口、不参与融合，在图中作为独立的AI CPU任务与AI Core任务（含SuperKernel容器）按Stream依赖顺序衔接。详细说明参见[SuperKernel](../../development/advanced_programming/superkernel.md#ai-cpu-kernel与superkernel的关系)。

## 相关文档

- [Kernel核函数创建](../../aicpu_development/kernel_function.md)：AI CPU Kernel的声明、参数、装饰器选项、启动方式。
- [Tensor创建和操作](../../aicpu_development/tensor_creation_and_operations.md)：Tensor声明与numel/getval/setval标量读写等接口说明。
