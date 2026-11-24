# pypto_pro.language.ceildiv

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

计算两个标量的向上取整除法，即`res = ceil(lhs ÷ rhs)`，结果为不小于lhs/rhs的最小整数。整数操作数得到整数向上取整结果，浮点操作数得到浮点向上取整结果。操作数类型规则与整除运算符//一致。

## 函数原型

```python
pypto_pro.language.ceildiv(
    lhs: Union[int, float],
    rhs: Union[int, float],
    /,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 被除数（Python int、Python float或Kernel内整型或浮点型标量表达式），支持DT_INT8、DT_INT16、DT_INT32、DT_INT64、DT_UINT8、DT_UINT16、DT_UINT32、DT_UINT64、DT_FP16、DT_BF16、DT_FP32；DT_BOOL操作数按DT_INT64参与计算；无符号整型的运算规则与有符号整型相同。 |
| rhs | 输入 | 除数（Python int、Python float或Kernel内整型或浮点型标量表达式），支持的类型与lhs相同，编译期可知的0会被拒绝。 |

## 返回值说明

返回`ceil(lhs / rhs)`，为标量值表达式。设lhs、rhs参与计算的标量类型分别为A_TYPE、B_TYPE，结果类型OUT_TYPE按下表确定，与整除运算符//的提升规则一致：

| A_TYPE | B_TYPE | OUT_TYPE |
|---|---|---|
| 整型 | 整型 | 位宽较大一侧的类型；位宽相等（含符号不同，如DT_INT32×DT_UINT32）取A_TYPE |
| 整型 | 浮点型 | DT_FP32（两侧先转为DT_FP32再计算） |
| 浮点型 | 整型 | DT_FP32（同行规则，对称适用） |
| 浮点型 | 浮点型 | 位宽较大一侧的类型；位宽相等（DT_FP16×DT_BF16）取A_TYPE |
| DT_BOOL | 任意类型 | DT_BOOL按DT_INT64代入上述规则（如DT_BOOL×DT_INT16→DT_INT64、DT_BOOL×DT_FP16→DT_FP32） |

- 整型含DT_INT8/16/32/64、DT_UINT8/16/32/64；浮点型含DT_FP16、DT_BF16、DT_FP32。
- 裸Python int字面量按DT_INT64（数值超过INT64_MAX时按DT_UINT64）、裸Python float字面量按DT_FP32、Python DT_BOOL按DT_INT64代入上表。

## 约束说明

- **适用范围**：本接口仅用于标量向上取整除法，常用于循环边界、索引计算；针对Tile逐元素向上取整除法请使用[pypto.ceil_div](../../../tensor_api/operation/pypto-ceil_div.md)。
- **浮点精度与边界**：浮点操作数按DT_FP32运算，等价于`-((-lhs) // rhs)`；操作数含Inf、NaN时结果遵循Python浮点整除语义（如`Inf / 2`得NaN、`0.5 / Inf`得1）；用于循环边界或索引计算时请使用整数操作数。
- **整数结果溢出**：整型组合仅在数学结果超出OUT_TYPE表示范围时溢出，典型为lhs等于OUT_TYPE最小值且rhs=-1（如`ceil(-2147483648 / -1)`的数学值2147483648超出DT_INT32）。两侧均为编译期常量时，折叠以Python精确整型计算，结果无法表示时编译期报`OutOfRange`错误；操作数含运行期值时编译不报错，运行结果为硬件截断除法的溢出语义，不保证等于数学值，请先用[astype](./astype.md)提升位宽或规避该组合。
- **浮点结果溢出**：`|ceil(lhs / rhs)|`超出OUT_TYPE最大有限值（DT_FP32与DT_BF16约3.4e38，DT_FP16为65504）时，运行期结果溢出时返回相应符号的无穷。两侧均为编译期常量且折叠结果超出OUT_TYPE可表示范围时，编译期报`OutOfRange`错误。
- **向量函数内仅支持整数操作数**：VF域（pypto_pro.language.vector_function修饰的函数体）内不支持浮点操作数，编译期报错；请改用整数操作数，或将浮点除法移至Kernel层。整数操作数的pypto_pro.language.ceildiv在VF域内可用。

## 调用示例

```python
import pypto_pro.language as pl

@pl.jit()
def example_kernel(...):
    n_blocks = pl.ceildiv(total, block_size)

    for i in pl.range(n_blocks):
        ...
```
