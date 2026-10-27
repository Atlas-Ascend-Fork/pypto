# pypto_pro.language.dynamic_input_dim

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

读取运行时变长（DYNAMIC）输入端口描述符中第i个张量指定维度的长度。带形状（dim != 0）的描述符为每个挂接的张量携带一组形状值，本接口按张量下标和维度下标取出对应形状值，返回INT64标量。

## 函数原型

```python
pypto_pro.language.dynamic_input_dim(x: Ptr, i: int, axis: int, /) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明                                                                                                                                                                                                       |
| ---- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| x    | 输入      | DYNAMIC输入端口的描述符指针，在Kernel签名中声明为[pypto_pro.language.Ptr](../basic_data_structures/Ptr.md)[dtype]。 |
| i    | 输入      | 张量下标，取值为[0, N)，N为Host侧挂接的张量个数；可以是编译期常量或运行时整型标量表达式。                                                                                                                  |
| axis | 输入      | 维度下标，整型标量表达式（编译期常量或运行时值）；合法范围为[0, dim)，其中dim为该张量的维数，仅在运行时可知。                                                                                              |

## 约束说明

- x是DYNAMIC输入端口的描述符（pl.Ptr[dtype]）；其指向的内存必须确实是框架传递的列表描述符，否则会按描述符布局解读，属未定义行为。
- 仅带形状（dim != 0）的描述符形态携带形状；dim == 0（纯指针形态）的描述符没有形状块，不属本接口的支持范围。
- axis必须落在[0, dim)内；越界（axis < 0或axis >= dim）时本接口在设备侧执行trap()终止执行，不会读取形状块之外的内存。
- 各张量可以独立分配，地址不必连续；形状与张量个数由描述符及Host侧共同提供。

## 返回值说明

返回第i个张量第axis维的长度，类型为INT64标量。

## 调用示例

下面的Kernel读取第1个张量各维的长度；dim由Host侧提供，用于遍历合法维度。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dynamic_input_dim_read(
    desc: pl.Ptr[pl.DT_FP32],
    out: pl.Ptr[pl.DT_INT64],
    dim: pl.DT_INT32,
):
    with pl.section_vector():
        out_t = pl.make_tensor(out, [1, 4], [4, 1], dtype=pl.DT_INT64)
        # axis 在 [0, dim) 内取值，越界会在设备侧 trap
        for axis in pl.range(0, dim):
            pl.setval(out_t, axis, pl.dynamic_input_dim(desc, 1, axis))
```
