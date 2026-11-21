# Tensor创建和操作

AI CPU Kernel没有片上存储与Tile编程，函数体使用标量编程面描述串行逻辑，数据访问以GM Tensor为中心。

## Tensor声明

使用[pypto_pro.language.Tensor](../../../../api/pro_api/SIMD-API/basic_data_structures/Tensor.md)声明Tensor参数，支持pypto_pro.language.DYNAMIC动态维度，方向标注（pypto_pro.language.Input、pypto_pro.language.Output）与AI Core Kernel一致：

```python
import pypto_pro.language as pl


@pl.jit(mode="aicpu_kernel")
def batch_row_plan(
    seq_lens: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input],
    plan: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Output],
):
    ...
```

完整类型语法与约束参见[pypto_pro.language.Tensor](../../../../api/pro_api/SIMD-API/basic_data_structures/Tensor.md)。

## Tensor操作

函数体内Tensor仅支持以下操作（函数调用形式）：

| 接口                                                                                           | 说明                                     |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------- |
| [pypto_pro.language.numel](../../../../api/pro_api/AICPU-API/numel.md)(t)         | 返回Tensor元素总数。                     |
| [pypto_pro.language.getval](../../../../api/pro_api/SIMD-API/transpose_and_element_access/getval.md)(t, i)    | 读取GM中第i个元素的标量值。              |
| [pypto_pro.language.setval](../../../../api/pro_api/SIMD-API/transpose_and_element_access/setval.md)(t, i, v) | 将标量值v写入GM中第i个元素。             |
| `t is None` / `t is not None`                                                              | 在位性判断，运行时None实参按空指针打包。 |

以下形式不支持：

- Tensor方法形式（如`t.size()`、`t.numel()`）：仅支持上表中的函数调用形式；
- 其他任何接收Tensor的接口（如pypto_pro.language.make_tensor、视图与搬运类接口）：解析阶段抛出NotSupported；
- Tile对象传入numel/getval/setval同样会被拒绝。

## Scalar参数与控制流

- Scalar参数以数据类型标注声明（如[pypto_pro.language.DataType](../../../../api/pro_api/SIMD-API/basic_data_structures/DataType.md)中的DT_INT32、DT_INT64），运行时按值传入；
- 标量运算支持算术、比较、逻辑运算（含整除与取模）与[pypto_pro.language.astype](../../../../api/pro_api/Utils-API/python_syntax_sugar/astype.md)类型转换；
- 控制流支持Python条件分支（if/elif/else）、while循环和[pypto_pro.language.range](../../../../api/pro_api/SIMD-API/controlflow/range.md)循环。
