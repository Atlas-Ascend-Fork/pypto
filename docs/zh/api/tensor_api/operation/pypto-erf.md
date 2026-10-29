# pypto.erf

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

## 功能说明

计算输入Tensor中每个元素的误差函数，逐元素运算，计算公式如下：
$$
\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt
$$

## 函数原型

```python
erf(input: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的数据类型为：DT_FP32，DT_BF16，DT_FP16。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回Tensor类型。其Shape、数据类型与输入Tensor一致，其元素为输入Tensor对应元素的误差函数值。

## 约束说明

1. 输入Tensor和输出Tensor类型应该相同。
2. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。
3. 由于存在临时内存使用，TileShape大小有额外约束。input会被转换为DT_FP32后拷贝至UB，UB中的输入Tensor会先对最后一维做32字节对齐填充，再对倒数第二维做行填充；输出Tensor仅对最后一维做32字节对齐填充。假设TileShape为\[a,b,c,d\]，记 $d_{align}=CeilAlign(d, 8), k=d_{align}/8, p=\lceil8/k\rceil, c_{pad}=c+p-1$，则总的UB空间占用为：

   $$
   a*b*c_{pad}*d_{align}*sizeof(DT\_FP32)+4*a*b*c*d_{align}*sizeof(DT\_FP32) <= UB
   $$

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如输入input shape为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.erf(x)
```

结果示例如下：

```python
输入数据x: [-0.5461, 0.1347, -200.7266, -0.2746]
输出数据y: [-0.5603, 0.1511, -1.0000, -0.3023]
```
