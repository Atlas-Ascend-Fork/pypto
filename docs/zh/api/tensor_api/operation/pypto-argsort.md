# pypto.argsort

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
- Kirin9030处理器系列产品：不支持
<!-- end id4 -->
<!-- npu="x90" id5 -->
- KirinX90处理器系列产品：不支持
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin9050处理器系列产品：不支持
<!-- end id6 -->

## 功能说明

获取输入在指定轴按照升序或者降序进行排序后的索引。

## 函数原型

```python
argsort(input: Tensor, dim: Optional[int]=None, descending: bool=False) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 指定排序的维度。<br>支持负索引,取值范围为[-input.dim, input.dim - 1]; 默认值为None,表示最后一个轴。四维输入不支持第0轴(即dim=0或者dim=-4)。|
| descending | 输入      | 如果为True，按降序返回索引。如果为False，按升序返回索引。默认值为False，即按升序返回索引。 |

## 返回值说明

返回一个Tensor，为输入按照descending在dim轴进行排序后的索引。

## 约束说明

- 当前不支持ViewShape在dim轴切分，即要求ViewShape[dim] = InputShape[dim]。
- 当前仅支持TileShape在dim轴上为32的倍数，即要求TileShape[dim] % 32 = 0。
- 记TileShape各维大小乘积除以TileShape[dim]为R，CeilAlign\(x, n\)表示将x向上对齐到n的倍数，则当Shape较大场景`R * CeilAlign(ViewShape[dim], 32) >= 6144`时，排序轴的Tile切分数量小于128。
- 对于四维输入，暂不支持在第0轴上进行排序。
- 排序时遇到相同值时，采用稳定排序返回对应的索引。
- Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。
- 由于存在临时内存使用，设置TileShape时需保证输入Tile、输出Tile及临时空间的总占用小于可用UB。计算过程会将输入Tile转换为DT_FP32的中间结果。记TileShape各维大小乘积除以TileShape[dim]为R，ViewShape[dim]为K。当`R * CeilAlign(K, 32) < 6144`时，排序在UB中完成，使用临时空间大小为`(2 * R * CeilAlign(K, 32) + 2 * R * CeilAlign(K, 8) + 2 * CeilAlign(K, 32)) * sizeof(DT_FP32)`字节；当`R * CeilAlign(K, 32) >= 6144`时，排序在GM中完成，UB内临时空间可将上式中的K替换为`CeilAlign(TileShape[dim], 32)`进行估算。此外，接口内部会先将dim轴转置到最后一轴、排序后再转置还原，除上述临时空间外还会使用转置临时空间，参见[pypto.transpose](pypto-transpose.md)的约束说明。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输入input一致。

如输入input shape为[m, n, p]，dim为2，descending为True，输出为[m, n, p]，TileShape设置为[m1, n1, p1]，则m1，n1，p1分别用于切分m，n，p轴。

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### 接口调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.argsort(x, -1, True)
```

结果示例如下：

```python
输入数据x: [[1.0 2.0 3.0],
            [1.0 2.0 3.0]]
输出数据y: [[2, 1, 0],
            [2, 1, 0]]
```
