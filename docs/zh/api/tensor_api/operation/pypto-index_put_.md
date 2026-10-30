# pypto.index\_put\_

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
<!-- npu="9030" id4 -->
- Kirin9030处理器系列产品：支持
<!-- end id4 -->
<!-- npu="x90" id5 -->
- KirinX90处理器系列产品：支持
<!-- end id5 -->
<!-- npu="9050" id11 -->
- Kirin9050处理器系列产品：支持
<!-- end id11 -->

## 功能说明

根据索引indices将values的多个或多块数据更新到self中。如果accumulate参数为True，则表示在更新时，values和原本存储在相应位置的值进行累加；如果accumulate为False，则会直接覆盖原本的值。

## 函数原型

```python
index_put_(input: Tensor, indices: tuple, values: Tensor, accumulate: bool = False) -> None
```

## 参数说明

|   参数名   | 输入/输出 | 说明                                                                  |
|------------|-----------|----------------------------------------------------------------------|
|   input    |    输入   | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_INT8，DT_UINT8，DT_INT16，DT_UINT16，DT_INT32，DT_UINT32，DT_INT64，DT_UINT64，DT_BF16，DT_FP16，DT_FP32。<!-- npu="x90,9030,9050" id8 -->Kirin数据类型支持范围请参考：[Tensor数据类型说明](#约束说明)。<!-- end id8 --><br>不支持空Tensor，Shape仅支持1-4维，Shape Size不大于2147483647（即INT32_MAX）。 |
|  indices   |   输入    | Tensor类型的元组，每个Tensor表示一个维度的索引。<br>支持的类型为：tuple\[Tensor\]，每个Tensor均为一维，且维度相同。<br>Tensor支持的数据类型为：DT_INT8，DT_UINT8，DT_INT16，DT_UINT16，DT_INT32，DT_UINT32，DT_INT64，DT_UINT64。<!-- npu="x90,9030,9050" id9 -->Kirin数据类型支持范围请参考：[Tensor数据类型说明](#约束说明)。<!-- end id9 --><br>不支持空Tensor，tuple中Tensor的个数不大于input的维数。 |
|   values   |   输入    | 待更新到input中的值。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_INT8，DT_UINT8，DT_INT16，DT_UINT16，DT_INT32，DT_UINT32，DT_INT64，DT_UINT64，DT_BF16，DT_FP16，DT_FP32。<!-- npu="x90,9030,9050" id10 -->Kirin数据类型支持范围请参考：[Tensor数据类型说明](#约束说明)。<!-- end id10 --><br>不支持空Tensor，维数不大于input的维数。 |
| accumulate |   输入（可选）    | 累加参数，默认为False。<br>支持的类型为：bool。 |

## 返回值说明

对input进行原地操作，无返回值

## 约束说明

1. indices中的一维Tensor维度相同，不支持broadcast。indices中第i个Tensor中的值须小于input中第i-1维的Shape大小。当indices的选取会对同一个位置进行重复更新时，结果是未确定的。

2. values不支持broadcast，其第0维的shape须和indices中一维Tensor的shape相同。若values的维度大于等于2，那么其除第0维外的后i个维度（i>0）和input后i个维度的shape完全相同。

3. input的维度、indices中Tensor的个数和values的维度之间需满足：(input.shape.size) + 1 = (indices.size) + (values.shape.size)。

4. input和values的数据类型须相同。

5. viewshape为一维，针对indices中的每个一维Tensor和values的第0维进行切分，values的其它维度不做切分。

6. TileShape的维度不超过values的维度，针对indices中的每个一维Tensor和values进行切分。indices和values的TileShape大小总和不能超过UB内存的大小。

7. accumulate为True时，input和values的数据类型仅支持DT_BF16、DT_FP16、DT_FP32、DT_INT8、DT_INT16、DT_INT32。

8. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

<!-- npu="x90,9030,9050" id6 -->
9. Tensor数据类型说明：
   <!-- npu="9030" id12 -->
   - Kirin9030处理器系列产品：
      - input/values：DT_FP16，DT_FP32，DT_INT8，DT_INT16，DT_INT32
      - indices：DT_INT8，DT_INT16，DT_INT32，DT_INT64，DT_UINT8，DT_UINT16，DT_UINT32，DT_UINT64
   <!-- end id12 -->
   <!-- npu="x90" id13 -->
   - KirinX90处理器系列产品：
      - input/values：DT_FP16，DT_FP32，DT_INT8，DT_INT16，DT_INT32
      - indices：DT_INT8，DT_INT16，DT_INT32，DT_INT64，DT_UINT8，DT_UINT16，DT_UINT32，DT_UINT64
   <!-- end id13 -->
   <!-- npu="9050" id16 -->
   - Kirin9050处理器系列产品：
      - input/values：DT_FP16，DT_FP32，DT_INT8，DT_INT16，DT_INT32
      - indices：DT_INT8，DT_INT16，DT_INT32，DT_INT64，DT_UINT8，DT_UINT16，DT_UINT32，DT_UINT64
   <!-- end id16 -->
<!-- end id6 -->

<!-- npu="x90,9030,9050" id7 -->
10. accumulate支持说明：
    <!-- npu="9030" id14 -->
    - Kirin9030处理器系列产品：不支持，必须设置为False
    <!-- end id14 -->
    <!-- npu="x90" id15 -->
    - KirinX90处理器系列产品：不支持，必须设置为False
    <!-- end id15 -->
    <!-- npu="9050" id17 -->
    - Kirin9050处理器系列产品：不支持，必须设置为False
    <!-- end id17 -->
<!-- end id7 -->

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape的维度不超过values的维度，若TileShape的维度小于values的维度，则TileShape在切分时会自动补全后续维度与values的shape一致。

如输入input为[m, n, p]，输入indices为([t])，输入values为[t, n, p]，TileShape设置为[t1, n1, p1]，则t1用于切分t轴，n1用于切分n轴，p1用于切分p轴，m轴不切。

如输入input为[m, n, p]，输入indices为([t])，输入values为[t, n, p]，TileShape设置为[t1, n1]，则TileShape会自动补全为[t1, n1, p]，t1用于切分t轴，n1用于切分n轴，m轴和p轴不切。

如输入input为[m, n, p]，输入indices为([t], [t])，输入values为[t, p]，TileShape设置为[t1, p1]，则t1用于切分t轴，p1用于切分p轴，m轴和n轴不切。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([3, 3], pypto.DT_INT32)
indices0 = pypto.tensor([2], pypto.DT_INT32)
indices = (indices0, )
values = pypto.tensor([2, 3], pypto.DT_INT32)
accumulate = True
# accumulate is True
pypto.index_put_(x, indices, values, accumulate)
# accumulate is False(default)
pypto.index_put_(x, indices, values)
```

结果示例如下：

```python
输入数据x:      [[1 1 1],
                 [1 1 1],
                 [0 0 0]]
      indices:   ([1 2], )
      values:    [[0 1 0],
                  [0 2 0]]
原地更新后的x:   [[1 1 1],
                 [1 2 1],
                 [0 2 0]]               # accumulate is True
                 [[1 1 1],
                 [0 1 0],
                 [0 2 0]]               # accumulate is False
```
