# pypto_pro.language.numel

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->

## 功能说明

返回Tensor的元素总数，即所有维度大小的乘积。pypto_pro.language.numel是AI CPU Kernel专有接口：仅可在mode="aicpu_kernel"声明的Kernel中使用，AI Core Kernel中调用会被拒绝。元素总数按Tensor声明shape计算，声明中含动态维度（pypto_pro.language.DYNAMIC）时在运行时求值。

## 函数原型

```python
pypto_pro.language.numel(tensor: Tensor) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| tensor | 输入 | GM Tensor，支持动态维度声明。传入Tile会被拒绝。 |

## 返回值说明

返回DT_INT64类型的标量，等于Tensor所有维度大小的乘积。

## 约束说明

- 仅AI CPU Kernel支持，在其他模式的Kernel中调用抛出NotSupported；
- 仅接受Tensor参数，传入Tile会被拒绝。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(mode="aicpu_kernel")
def batch_row_plan(
    seq_lens: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input],
    plan: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Output],
):
    n = pl.numel(seq_lens)
    if n <= 0:
        return 1  # KERNEL_STATUS_PARAM_INVALID
    for b in pl.range(0, n):
        rows = pl.getval(seq_lens, b)
        pl.setval(plan, 2 * b + 1, rows)
    return  # KERNEL_STATUS_OK
```

完整示例参见[元数据规划算子快速入门（AI CPU）](../../../guide/quick_start/pro/metadata_aicpu.md)。
