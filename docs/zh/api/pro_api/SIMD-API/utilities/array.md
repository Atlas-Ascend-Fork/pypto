# pypto_pro.language.array

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

声明一个长度固定、可读写的标量数组，元素数据类型和长度均为编译期常量。数组为Kernel内部存储，`A[i]`读取指定元素并返回元素类型的标量，`A[i] = value`写入一个元素。适用于在Kernel内保存少量、长度为编译期常量、需要跨语句或跨循环迭代复用的标量（如计数器、临时索引）；若这些标量需要随结构体一起传递，可使用[pypto_pro.language.struct](./struct.md)的一维数组字段。

## 函数原型

```python
pypto_pro.language.array(size: int, /, *, dtype: Any) -> Any
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| size | 输入 | 编译期正整数，表示元素个数。取值必须为整数字面量或Kernel内常量。 |
| dtype | 输入 | 标量元素数据类型，例如pypto_pro.language.DT_INT32。支持pypto_pro.language.DT_INT8、pypto_pro.language.DT_INT16、pypto_pro.language.DT_INT32、pypto_pro.language.DT_INT64、pypto_pro.language.DT_UINT8、pypto_pro.language.DT_UINT16、pypto_pro.language.DT_UINT32、pypto_pro.language.DT_UINT64、pypto_pro.language.DT_FP16、pypto_pro.language.DT_FP32、pypto_pro.language.DT_BF16、pypto_pro.language.DT_BOOL；仅用于存储的低精度类型（如pypto_pro.language.DT_INT4、pypto_pro.language.DT_FP8E4M3FN）不支持。 |

## 约束说明

- 数组存储未初始化。Kernel必须保证每个元素先写入后读取。
- `A[0]`和`A[i]`均接受单个整数下标。常量下标越界会在编译期报错；运行时下标必须保证在`[0, size)`范围内。
- 下标必须为整数标量，不支持多维下标。
- 读取返回元素类型的标量。`A[i] = value`接受标量值：字面量会转换为元素类型，其他标量由代码生成阶段进行类型转换，行为与pypto_pro.language.setval一致。
- A = pypto_pro.language.array(...)必须将声明绑定到变量。将变量赋给另一个名字表示同一份存储的别名，通过任一名称写入都会修改同一数组。
- 每次调用pypto_pro.language.array(...)都会创建独立存储，即使参数完全相同。
- 在循环外声明的数组，其元素值在循环迭代之间保持；在循环体内声明的数组，其生命周期仅限该循环体。
- 数组存储仅在其所在核的代码体内有效。请勿跨核section使用同一数组，例如在pypto_pro.language.section_cube()中写入后再在pypto_pro.language.section_vector()中读取。
- 数组不能在pypto_pro.language.vector_function内使用。

## 返回值说明

返回绑定到Kernel内变量的数组值。

## 调用示例

### 别名与标量捕获

```python
import pypto_pro.language as pl


@pl.jit()
def alias_kernel(out: pl.Tensor[[3], pl.DT_INT32]):
    values = pl.array(8, dtype=pl.DT_INT32)
    values[0] = 7
    a = values[0]          # 捕获int32值7
    alias = values         # 指向同一份存储
    alias[0] = 9
    values[1] = a          # values[0]为9，values[1]为7
    with pl.section_vector():
        pl.setval(out, 0, a)
        pl.setval(out, 1, values[0])
        pl.setval(out, 2, values[1])
```

### 在一个循环中填充，在另一个循环中读取

```python
import pypto_pro.language as pl


@pl.jit()
def loop_kernel(out: pl.Tensor[[8], pl.DT_INT32]):
    values = pl.array(8, dtype=pl.DT_INT32)
    for i in pl.range(0, 8):
        values[i] = i * 2 + 1
    with pl.section_vector():
        for i in pl.range(0, 8):
            pl.setval(out, i, values[i])
```
