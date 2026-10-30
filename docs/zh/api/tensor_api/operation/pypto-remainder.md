# pypto.remainder

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->
<!-- npu="9030" id10 -->
- Kirin9030处理器系列产品：不支持
<!-- end id10 -->
<!-- npu="x90" id11 -->
- KirinX90处理器系列产品：不支持
<!-- end id11 -->
<!-- npu="9050" id12 -->
- Kirin9050处理器系列产品：不支持
<!-- end id12 -->

## 功能说明

将input的每个元素和other中对应位置的元素进行取余运算，计算公式如下：

$$
res_i = input_i - other_i * floor(input_i / other_i)
$$

## 函数原型

```python
remainder(
    input: Union[Tensor, int, float],
    other: Union[Tensor, int, float],
    precision_type: PrecisionType = PrecisionType.HIGH_PRECISION
) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。<br>支持的类型为：Tensor、int、float。<br>Tensor支持的数据类型不同型号有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持1-4维，支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other  | 输入      | 源操作数。<br>支持的类型为：Tensor、int、float。<br>Tensor支持的数据类型不同型号有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持1-4维，支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| precision_type | 输入      | 精度模式枚举类型，用以控制取余计算的精度模式，具体定义为：[PrecisionType](../datatype/PrecisionType.md)。<br>默认为HIGH_PRECISION（高精度模式）。 |

## 返回值说明

返回输出Tensor，Shape为input和other广播后大小，数据类型和输入Tensor的数据类型相同。

## 约束说明

1. 当前不支持混合精度类型输入，即输入都是Tensor时数据类型都相同，输入有一个是标量时，Tensor的数据类型必须是对应的整数类型或浮点数类型；
2. 当input为整型数据类型时，**other不能含0**，整数取余的结果由芯片类型决定，可能为0或-1。
3. 若输入Tensor的数据类型为DT_INT32，数据范围超过\[-2^24, 2^24\]范围时不保证精度；
4. precision_type使用说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：支持
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：不支持，默认使用指令模式 `INTRINSIC`
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：不支持，默认使用指令模式 `INTRINSIC`
   <!-- end id6 -->
5. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。
6. Tensor数据类型说明：
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT系列产品：DT_FP32，DT_FP16，DT_BF16，DT_INT32，DT_INT16，DT_INT64。
   <!-- end id7 -->
   <!-- npu="A3" id8 -->
   - Atlas A3系列产品：DT_FP32，DT_FP16，DT_BF16，DT_INT32，DT_INT16。
   <!-- end id8 -->
   <!-- npu="910b" id9 -->
   - Atlas A2系列产品：DT_FP32，DT_FP16，DT_BF16，DT_INT32，DT_INT16。
   <!-- end id9 -->
7. 由于存在临时内存使用，设置TileShape时需保证输入Tile、输出Tile及临时空间的总占用小于可用UB。记TileShape的最后两维为H和W（一维TileShape取H为1），记align为256/sizeof\(output\)，W对齐到align记为WAlign：
   - 若input和other均为Tensor，则使用临时空间大小为`(WAlign + CeilAlign(WAlign / 8, 256) / sizeof(output) + 256) * sizeof(output)`字节
   - 若input和other之一不为Tensor，则使用临时空间大小为`(H + 1) * (WAlign + CeilAlign(WAlign / 8, 256) / sizeof(output) + 256) * sizeof(output)`字节

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如以下非广播示例，输入input和other的Shape均为[n]，输出为[n]，TileShape设置为[n1]，则n1用于切分n轴。

```python
pypto.set_vec_tile_shapes(16)
```

### 接口调用示例

以下为计算图构建上下文中的接口调用片段。pypto.tensor的首参为Shape，不会初始化示例数值；具体输入数据由调用方在执行时提供。高精度模式和指令模式示例使用相同的输入Shape。

```python
a = pypto.tensor([3], pypto.DT_FP32)
b = pypto.tensor([3], pypto.DT_FP32)
out = pypto.remainder(a, b)
```

当调用方提供下列输入数据时，结果示例如下：

```python
输入数据a:    [7.0, 8.0, 9.0]
输入数据b:    [-3.0, -3.0, -3.0]
输出数据out:  [-2.0, -1.0, 0.0]
```

### 高精度模式示例

```python
a = pypto.tensor([3], pypto.DT_FP16)
b = pypto.tensor([3], pypto.DT_FP16)
out = pypto.remainder(a, b, pypto.PrecisionType.HIGH_PRECISION)
```

### 指令模式示例

```python
a = pypto.tensor([3], pypto.DT_FP32)
b = pypto.tensor([3], pypto.DT_FP32)
out = pypto.remainder(a, b, pypto.PrecisionType.INTRINSIC)
```
