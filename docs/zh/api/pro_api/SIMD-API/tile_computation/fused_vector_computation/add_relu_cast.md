# pypto_pro.language.add_relu_cast

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

对两个源Tile的有效区域逐元素相加，对中间结果执行ReLU，再转换为目标数据类型：

$$
out_{i,j}=\operatorname{cast}_{mode}\left(\max\left(lhs_{i,j}+rhs_{i,j}, 0\right)\right)
$$

计算过程中，lhs同时作为加法和ReLU的中间结果缓冲：加法结果和ReLU结果依次写回lhs，再转换数据类型并写入out。因此，调用完成后lhs的原始内容被覆盖。

## 函数原型

```python
pypto_pro.language.add_relu_cast(
    out: Tile,
    lhs: Tile,
    rhs: Tile,
    *,
    target_type: DType,
    mode: RoundMode=RoundMode.CAST_ROUND,
) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | UB、RowMajor Tile；物理shape、valid_shape和layout须与lhs、rhs一致。dtype须与target_type一致，支持的目标类型见下表。 |
| lhs | 输出 | UB、RowMajor Tile；dtype须为DT_FP16、DT_FP32或DT_INT32，并与rhs一致。物理shape、valid_shape和layout须与rhs、out一致；调用后内容变为ReLU后的加法中间结果。 |
| rhs | 输入 | UB、RowMajor Tile；dtype、物理shape、valid_shape和layout须与lhs一致。 |
| target_type | 输入 | 编译期[pypto_pro.language.DataType](../../basic_data_structures/DataType.md)枚举值，必须等于out.dtype。该参数不会重新分配或改变out的类型；二者不一致属于不支持的用法。 |
| mode | 输入 | 可选，编译期[pypto_pro.language.RoundMode](../../basic_data_structures/RoundMode.md)枚举值，默认pypto_pro.language.RoundMode.CAST_ROUND。具体转换路径和舍入模式支持情况与[pypto_pro.language.cast](../type_conversion/cast.md)一致。 |

### 支持的数据类型转换

算术和ReLU均以lhs/rhs的数据类型执行，随后从该中间类型转换到target_type。未列出的组合不支持。

| lhs/rhs dtype | target_type（同时为out.dtype） |
|---|---|
| DT_FP16 | DT_FP32、DT_INT8、DT_UINT8、DT_INT16、DT_INT32、DT_HF8 |
| DT_FP32 | DT_FP32、DT_FP16、DT_BF16、DT_INT16、DT_INT32、DT_INT64、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8 |
| DT_INT32 | DT_FP32、DT_INT64、DT_UINT8、DT_INT16、DT_UINT16 |

舍入模式按中间类型到目标类型的转换路径解释。例如DT_FP16 -> DT_FP32不涉及舍入；DT_FP32 -> DT_FP16支持CAST_ODD；DT_FP16/DT_FP32 -> DT_HF8固定使用CAST_ROUND。完整规则参见[cast的约束说明](../type_conversion/cast.md#约束说明)。

## 返回值说明

无返回值。计算结果写入out。

## 约束说明

1. lhs会被覆盖，后续不能再依赖其调用前的值。rhs和out应使用与lhs互不重叠的UB区域；本接口不保证任意地址重叠时的结果。
2. 接口只定义valid_shape有效区域内的结果，不应依赖有效区域外的内容。
3. target_type和mode必须在编译期确定，不能使用运行时Scalar或Tensor动态选择。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def add_relu_cast_kernel(a: pl.Tensor[[64, 64], pl.DT_FP16], b: pl.Tensor[[64, 64], pl.DT_FP16],
                         out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt_in = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt_in, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt_in, addrs=0x2000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x4000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add_relu_cast(cur_out, cur_a, cur_b, target_type=pl.DT_FP32, mode=pl.RoundMode.CAST_ROUND)
        pl.store(out, cur_out, [0, 0])
```

结果示例如下：

<!-- pypto-doc-output:add_relu_cast:start -->
```bash
输入数据a：[[-2 -1.75 -1.5 -1.25 -1 -0.75 -0.5 -0.25 ...], [14 14.25 14.5 14.75 15 15.25 15.5 15.75 ...], [30 30.25 30.5 30.75 31 31.25 31.5 31.75 ...], [46 46.25 46.5 46.75 47 47.25 47.5 47.75 ...], ...]
输入数据b：[[3 2.875 2.75 2.625 2.5 2.375 2.25 2.125 ...], [-5 -5.125 -5.25 -5.375 -5.5 -5.625 -5.75 -5.875 ...], [-13 -13.125 -13.25 -13.375 -13.5 -13.625 -13.75 -13.875 ...], [-21 -21.125 -21.25 -21.375 -21.5 -21.625 -21.75 -21.875 ...], ...]
输出数据out：[[1 1.125 1.25 1.375 1.5 1.625 1.75 1.875 ...], [9 9.125 9.25 9.375 9.5 9.625 9.75 9.875 ...], [17 17.125 17.25 17.375 17.5 17.625 17.75 17.875 ...], [25 25.125 25.25 25.375 25.5 25.625 25.75 25.875 ...], ...]
```
<!-- pypto-doc-output:add_relu_cast:end -->
