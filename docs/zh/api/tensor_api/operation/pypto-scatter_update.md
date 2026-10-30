# pypto.scatter\_update

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

功能1：原地操作，将4维src根据2维索引index更新到4维input上，计算公式如下：

$$
input\left[\frac{\text{index}[i][j]}{\text{blockSize}}\right]\left[\text{index}[i][j] \% \text{blockSize}\right][0][\dots] = src[i][j][0][\dots]
$$

功能2：原地操作，将2维src根据2维index更新到2维input上，计算公式如下（其中s是index第二维的大小，即index.shape[1]）：

$$
input[[\text{index}[i][j]][\dots]] = src[i*s + j][\dots]
$$

## 函数原型

```python
scatter_update(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 支持的类型为：Tensor。<br>不同型号支持的数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>支持的维度：2维，4维<br>2维Shape[blockNum * blockSize, d]，4维Shape[blockNum, blockSize, 1, d]<br>不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 请保持默认值-2。 |
| index   | 输入      | input的一组索引。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_INT64，DT_INT32，DT_INT16。<br>支持的维度：2维<br>Shape[b, s] |
| src     | 输入      | src是一组更新值。<br>支持的类型为：Tensor。<br>不同型号支持的数据类型有所差异，详细请参见[约束说明](#约束说明)。数据类型和input保持一致<br>支持的维度：2维，4维<br>2维Shape[b * s, d]，4维Shape[b, s, 1, d]<br>不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回更新后的input，为inplace操作，非索引位置保留目标缓冲区原值。若输入、输出使用不同形参，调用时应绑定同一个实际缓冲区。

## 约束说明

1. broadcast约束：不支持broadcast。
2. 索引约束：当index中包含指向同一目标位置的重复索引时，写入顺序以及最终写入结果不保证确定性。
3. Tensor格式约束：Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。
4. ViewShape约束：2维场景下ViewShape为\[viewB \* s, d\]，4维场景下ViewShape为\[viewB, viewS, 1, d\]，尾轴d不做切分。2维场景下，\[viewB \* s, d\]针对src做切分，其第0维是index的第1维s的倍数，\[viewB, s\]针对index做切分。4维场景下，\[viewB, viewS, 1, d\]针对src做切分，\[viewB, viewS\]针对index做切分。
5. TileShape约束：
   - 2维场景下，TileShape为\[tileSize, tileD\]，设index.shape\[1\]为s。当TileShape\[0\]不大于s时，TileShape\[0\]必须是s的约数；当TileShape\[0\]大于s时，TileShape\[0\]必须是s的倍数。例如src为\[12, 64\]、index为\[3, 4\]时，TileShape可以为\[2, 64\]、\[4, 64\]或\[12, 64\]；当tileD小于d时，也可以取\[4, 32\]。
   - 4维场景下，TileShape为\[tileB, tileS, 1, tileD\]。
6. 输入input和输出均常驻gm，不占用ub空间。输入index和输入src需要按TileShape切分后搬入ub，两个输入的tile块大小之和应小于ub限制。
7. Tensor数据类型说明：
   <!-- npu="950" id6 -->
   - Ascend 950PR&950DT系列产品：DT_FP32，DT_FP16，DT_BF16，DT_FP8E4M3，DT_INT32，DT_INT16，DT_INT8。
   <!-- end id6 -->
   <!-- npu="A3" id7 -->
   - Atlas A3系列产品：DT_FP32，DT_FP16，DT_BF16，DT_INT32，DT_INT16，DT_INT8。
   <!-- end id7 -->
   <!-- npu="910b" id8 -->
   - Atlas A2系列产品：DT_FP32，DT_FP16，DT_BF16，DT_INT32，DT_INT16，DT_INT8。
   <!-- end id8 -->

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输入src一致。

若输入input为[blockNum \* blockSize, d]，dim为-2，输入index为[b, s]，输入src为[b \* s, d]，输出为[blockNum \* blockSize, d]，TileShape设置为[tileSize, tileD]，则tileSize用于切分bs轴，tileD用于切分d轴。

若输入input为[blockNum, blockSize, 1, d]，dim为-2，输入index为[b, s]，输入src为[b, s, 1, d]，输出为[blockNum, blockSize, 1, d]，TileShape设置为[tileB, tileS, 1, tileD]，则tileB用于切分b轴，tileS用于切分s轴，tileD用于切分d轴。

```python
pypto.set_vec_tile_shapes(16, 64)
```

### 接口调用示例

- 将2维src根据2维index更新到2维input上，以下使用move将更新结果写回同一个目标Tensor：

    ```python
    x = pypto.tensor([8, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([4, 3], pypto.DT_INT32)
    pypto.set_vec_tile_shapes(4, 3)
    x.move(pypto.scatter_update(x, -2, y, z))
    ```

    以下输入数据需由调用方提供；`pypto.tensor`仅声明Tensor，不会默认初始化为0。结果示例中的非索引位置保留输入值：

    ```python
    输入数据x:[[1 2 3],
               [4 5 6],
               [7 8 9],
               [10 11 12],
               [13 14 15],
               [16 17 18],
               [19 20 21],
               [22 23 24]]
    输入数据y:[[1 2],
               [4 5]]
    输入数据z:[[1 2 3],
               [4 5 6],
               [7 8 9],
               [10 11 12]]
    输出数据x:[[1 2 3],
               [1 2 3],
               [4 5 6],
               [10 11 12],
               [7 8 9],
               [10 11 12],
               [19 20 21],
               [22 23 24]]
    ```

- 将4维src根据2维索引index更新到4维input上，以下使用move将更新结果写回同一个目标Tensor：

    ```python
    x = pypto.tensor([2, 6, 1, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([2, 2, 1, 3], pypto.DT_INT32)
    pypto.set_vec_tile_shapes(1, 2, 1, 3)
    x.move(pypto.scatter_update(x, -2, y, z))
    ```

    以下输入数据需由调用方提供；`pypto.tensor`仅声明Tensor，不会默认初始化为0。结果示例中的非索引位置保留输入值：

    ```python
    输入数据x:[[
                 [[1 2 3]],
                 [[4 5 6]],
                 [[7 8 9]],
                 [[10 11 12]],
                 [[13 14 15]],
                 [[16 17 18]],
               ],
               [
                 [[19 20 21]],
                 [[22 23 24]],
                 [[25 26 27]],
                 [[28 29 30]],
                 [[31 32 33]],
                 [[34 35 36]],
               ]]
    输入数据y:[[1 8],
               [4 10]]
    输入数据z:[[
                 [[1 2 3]],
                 [[4 5 6]],
               ],
               [
                 [[7 8 9]],
                 [[10 11 12]],
               ]]
    输出数据x:[[
                 [[1 2 3]],
                 [[1 2 3]],
                 [[7 8 9]],
                 [[10 11 12]],
                 [[7 8 9]],
                 [[16 17 18]],
               ],
               [
                 [[19 20 21]],
                 [[22 23 24]],
                 [[4 5 6]],
                 [[28 29 30]],
                 [[10 11 12]],
                 [[34 35 36]],
               ]]
    ```
