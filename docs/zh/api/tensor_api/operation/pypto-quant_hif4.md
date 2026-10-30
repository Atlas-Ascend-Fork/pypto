# pypto.quant_hif4

## 产品支持情况

<!-- npu="960" id1 -->
- Ascend 960DT系列产品：支持
<!-- end id1 -->
<!-- npu="950" id2 -->
- Ascend 950PR&950DT系列产品：不支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- Atlas A3系列产品：不支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- Atlas A2系列产品：不支持
<!-- end id4 -->
<!-- npu="9030" id5 -->
- Kirin9030处理器系列产品：不支持
<!-- end id5 -->
<!-- npu="x90" id6 -->
- KirinX90处理器系列产品：不支持
<!-- end id6 -->

## 功能说明

将1-4维ND格式的高精度浮点Tensor量化为Hif4格式，返回量化结果和共享指数scale。

- 输入Tensor支持DT_BF16。
- 输出量化Tensor支持DT_HF4。
- scale Tensor的数据类型固定为uint_32。
- 仅支持对尾轴（`axis=-1`）量化。
- `scale_alg=0/1`控制量化功能中第4步的参数阈值为4/2(0)还是3.75/1.875(1)

若输入shape记为 $[d_0, d_1, ..., d_{n-1}]$，则：

- 量化结果`output`的shape与`input`相同。
- FP4使用逻辑元素shape：`DT_HF4`的两个元素打包到一个字节。外部Torch输出应使用尾轴减半的`torch.uint8`缓冲区，kernel输出参数声明为`DT_HF4`。

## 函数原型

```python
def quant_hif4(
    input: Tensor,
    quant_dtype: DataType = DataType.DT_HF4,
    scale_alg: int = 0,
    axis: int = -1,
) -> Tuple[Tensor, Tensor]:
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| input | 输入 | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_BF16。<br>仅支持`TILEOP_ND`格式；Shape仅支持1-4维。 |
| quant_dtype | 输入 | 量化后输出Tensor的数据类型。<br>支持：DT_HF4。 |
| scale_alg | 输入 | 控制量化功能中第4步的参数阈值。<br>0：表示4和2，1表示3.75和1.875。 |
| axis | 输入 | 指定量化轴。<br>支持最后一维（`-1`或`input.shape.size() - 1`）。|

## 返回值说明

返回一个二元组`(output, scale)`：

- `output`：量化后的Tensor，数据类型由`quant_dtype`指定，Shape与`input`相同。
- `scale`：缩放因子，数据类型为uint32，每64量化元素的缩放因子打包到1个uint32类型中。`scale`的形状为`input`尾轴对64上取整。例如，输入为[2,7,120]的输出`scale`的形状为[2,7,2]。

## 约束说明

1. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。
2. 若设置ViewShape和TileShape，维度必须与输入一致，且ViewShape和TileShape的尾轴需要64对齐。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过`set_vec_tile_shapes`设置TileShape。

TileShape的具体取值要求参见[约束说明](#约束说明)。

示例1：输入view shape为`[m, n]`，TileShape可设置为`[m1, n]`。

```python
pypto.set_vec_tile_shapes(4, 64)
```

示例2：输入shape为`[m, n]`，TileShape可设置为`[m1, n1]`。

```python
pypto.set_vec_tile_shapes(2, 128)
```

### 接口调用示例

```python
x = pypto.tensor([8, 64], pypto.DT_BF16)

# 默认配置：scale_alg = 0，尾轴量化
output, scale = pypto.quant_hif4(x)

# 显式指定scale_alg参数
output_alg_1, scale_alg_1 = pypto.quant_mx(
    x,
    pypto.DT_HF4,
    1,
    -1,
)
```

结果示例如下：

```python
Input x.shape: [8, 64]
Input x.dtype: DT_BF16
Output output.shape: [8, 64]
Output output.dtype: DT_HF4
Output scale.shape: [8, 1]
Output scale.dtype: uint32
```
