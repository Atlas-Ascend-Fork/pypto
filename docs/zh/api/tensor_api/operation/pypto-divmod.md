# pypto.divmod

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

将input的每个元素除以other中对应位置的元素，向下取整得到商，并按 `r = input - q * other` 计算余数：

$$
q_i = \left\lfloor \frac{input_i}{other_i} \right\rfloor, \qquad
r_i = input_i - q_i \cdot other_i
$$

非零除数时，余数 $r_i$ 与除数 $other_i$ 同号（或为0），与 `numpy.divmod` 的整数语义一致。仅支持整型输入，不支持浮点类型，也不进行数据类型提升。input和other支持广播到相同Shape。

与[pypto.floor_div](pypto-floor_div.md)、[pypto.remainder](pypto-remainder.md)相比，本接口一次返回商和余数，避免分别调用两个接口带来的重复除法和中间结果搬运；需要在一次计算中同时获得商和余数时，优先使用本接口。

## 函数原型

```python
divmod(
    input: Tensor,
    other: Tensor,
    precision_type: PrecisionType = PrecisionType.INTRINSIC
) -> Tuple[Tensor, Tensor]
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型不同型号有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持1-4维，支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other  | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor的数据类型必须与input相同，不同型号支持的数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>不支持空Tensor；Shape仅支持1-4维，支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。<br>除数为0属于未定义行为（undefined behavior），不保证商和余数的结果；调用方应保证other中的元素均不为0。 |
| precision_type | 输入 | 保留参数，整型实现不使用。<br>默认为INTRINSIC（指令模式），具体定义为：[PrecisionType](../datatype/PrecisionType.md)。 |

input和other的Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 返回值说明

返回quotient和remainder两个Tensor，数据类型与input相同，Shape为input和other广播后大小。

## 约束说明

- Tensor数据类型说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT系列产品：DT_INT8、DT_UINT8、DT_INT16、DT_INT32、DT_INT64、DT_UINT16、DT_UINT32、DT_UINT64；DT_UINT16走原生无符号路径。
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3系列产品：DT_INT8、DT_UINT8、DT_INT16、DT_INT32、DT_UINT16；DT_UINT16使用等价的精确模拟路径；不支持DT_INT64、DT_UINT32、DT_UINT64。
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2系列产品：DT_INT8、DT_UINT8、DT_INT16、DT_INT32、DT_UINT16；DT_UINT16使用等价的精确模拟路径；不支持DT_INT64、DT_UINT32、DT_UINT64。
   <!-- end id6 -->
- 支持多维度广播；不可广播时在构图阶段报错。
- 整型溢出仅可能出现在INT_MIN / -1场景：商按目标数据类型的回绕语义处理（如INT8_MIN / -1得到INT8_MIN），余数为0，不额外报错，与NumPy的整数语义一致。
- 由于存在临时内存使用，设置TileShape时需保证输入Tile、输出Tile及临时空间的总占用小于可用UB，否则编译阶段报错。记TileShape最后一维为W，其余维度大小的乘积为H（一维TileShape取H为1），记输出数据类型为dstType，align为max(32/sizeof(dstType), 8)，W对齐到align记为WAlign，则临时空间为`PlaneCount * H * WAlign * sizeof(TmpType)`字节，其中TmpType在dstType为DT_INT64时为DT_INT64，其余情况为DT_FP32。PlaneCount取值如下：
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT系列产品：DT_UINT16、DT_UINT32、DT_UINT64为2；DT_INT16为3；DT_INT8、DT_UINT8为5；DT_INT32、DT_INT64为4。
   <!-- end id7 -->
   <!-- npu="A3" id8 -->
   - Atlas A3系列产品：DT_INT16为3；DT_INT32、DT_UINT16为6；DT_INT8、DT_UINT8为5。
   <!-- end id8 -->
   <!-- npu="910b" id9 -->
   - Atlas A2系列产品：DT_INT16为3；DT_INT32、DT_UINT16为6；DT_INT8、DT_UINT8为5。
   <!-- end id9 -->

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如非广播示例，输入input和other的Shape均为[n]，输出为[n]，TileShape设置为[n1]，则n1用于切分n轴；n1应不大于n，且需结合UB容量选择。

```python
pypto.set_vec_tile_shapes(16)
```

如广播示例，输入input的Shape为[m, n]，other的Shape为[m, 1]，输出为[m, n]，TileShape设置为[m1, n1]，则m1、n1分别用于切分m、n轴；m1应不大于m，n1应不大于n。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

#### 基本示例

```python
a = pypto.tensor([2, 3], pypto.DT_INT32)
b = pypto.tensor([2, 3], pypto.DT_INT32)
q, r = pypto.divmod(a, b)
```

当调用方提供下列输入数据时，结果示例如下：

```python
输入数据a:    [[7, 8, 9], [-7, -8, -9]]
输入数据b:    [[3, 3, 3], [3, 3, 3]]
输出数据q:    [[2, 2, 3], [-3, -3, -3]]
输出数据r:    [[1, 2, 0], [2, 1, 0]]
```

#### 广播示例

```python
a = pypto.tensor([2, 3], pypto.DT_INT32)
b = pypto.tensor([3], pypto.DT_INT32)
q, r = pypto.divmod(a, b)
```

#### 无符号示例

```python
a = pypto.tensor([2, 3], pypto.DT_UINT16)   # Ascend 950PR&950DT系列产品走原生路径，Atlas A3系列产品和Atlas A2系列产品走精确模拟路径
b = pypto.tensor([2, 3], pypto.DT_UINT16)
q, r = pypto.divmod(a, b)
```

#### 运算符调用示例

```python
a = pypto.tensor([2, 3], pypto.DT_INT32)
b = pypto.tensor([2, 3], pypto.DT_INT32)
q, r = divmod(a, b)
```
