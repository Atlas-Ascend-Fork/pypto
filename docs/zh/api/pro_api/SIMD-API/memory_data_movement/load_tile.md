# pypto_pro.language.load_tile

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

将GM Tensor中的数据搬入L1 Buffer或UB中的Tile。与[pypto_pro.language.load](load.md)不同，tile_offsets使用Tile块编号指定搬运位置：由order选中源操作数的Tensor维度按Tile块编号寻址，接口将对应编号乘以目的操作数的Tile中对应维度的大小，换算为绝对元素坐标；未被order选中的维度仍按绝对元素坐标寻址。

例如，对于shape=[64, 128]的二维Tile，不设置order时，tile_offsets=[2, 2]对应的绝对元素坐标为[128, 256]，等价于调用pypto_pro.language.load时传入offsets=[128, 256]。

![load_tile按块索引从GM搬入Tile](../../figures/load_tile_block_offset.jpg "load_tile按块索引从GM搬入Tile")

## 函数原型

```python
pypto_pro.language.load_tile(
    dst_tile: Tile,
    src_tensor: Tensor,
    /,
    tile_offsets: Sequence[int],
    *,
    order: Optional[List[int]] = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dst_tile | 输出 | 目的操作数，Tile类型，存储空间为L1 Buffer或UB，首地址必须按32字节对齐。接口按照该Tile的valid_shape搬运数据；支持的数据类型和分形组合详见[约束说明](#约束说明)。 |
| src_tensor | 输入 | 源操作数，Tensor类型，存储空间为GM。支持的数据类型和分形组合详见[约束说明](#约束说明)。 |
| tile_offsets | 输入 | 表示源Tensor各维度的Tile块编号，整数序列（列表或元组），每项可为整型常量或运行时整型标量表达式，长度须与源Tensor的维数相同。<br>- 不支持负数。<br>- order选中的维度按Tile块编号寻址，对应编号乘以目的操作数的Tile中对应维度的大小后得到绝对元素坐标。<br>- order未选中的维度按绝对元素坐标寻址。<br>- 换算后的绝对元素坐标必须位于源操作数的shape范围内；可通过pypto_pro.language.set_validshape设置有效形状，保证有效搬运范围不超过源操作数的shape范围。 |
| order | 输入 | 可选，维度映射，长度为2的编译期整数列表，指定目的操作数的Tile各维度对应的源操作数的Tensor维度索引。<br>- 两个维度索引必须互不重复且位于源操作数的Tensor维度范围内。<br>- 升序表示不转置，例如order=[0, 1]。<br>- 降序表示转置，例如order=[1, 0]，源操作数的Tensor分形描述由ND转为DN。<br>- 不配置时，指定源操作数的Tensor最后两维，且不转置。<br>- 搬运MX矩阵乘量化系数时，order中不能选择源操作数的Tensor最后一维；不设置order时，默认为除源操作数的Tensor最后一维外的最后两个维度。<br>- 当Tensor为1维时，不支持传入order参数。 |

## 约束说明

- 数据类型及分形约束：

  | 源 → 目的 | 分形要求 | 数据类型要求 |
  |---|---|---|
  | GM → UB | 源与目的分形必须相同，支持ND、DN、NZ。 | 源与目的数据类型位宽必须相同，支持DT_INT8、DT_UINT8、DT_FP16、DT_BF16、DT_INT16、DT_UINT16、DT_FP32、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP8E8M0、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
  | GM → L1 Buffer | ND → NZ。 | 源与目的数据类型位宽必须相同，支持DT_INT8、DT_UINT8、DT_FP16、DT_BF16、DT_INT16、DT_UINT16、DT_FP32、DT_INT32、DT_UINT32、DT_FP8E8M0、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
  | GM → L1 Buffer | DN → NZ。 | 源与目的数据类型位宽必须相同，支持DT_INT8、DT_UINT8、DT_FP16、DT_BF16、DT_INT16、DT_UINT16、DT_FP32、DT_INT32、DT_UINT32、DT_FP8E8M0、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8。 |
  | GM → L1 Buffer | DN → ZN、NZ → NZ。 | 源与目的数据类型位宽必须相同，支持DT_INT8、DT_UINT8、DT_FP16、DT_BF16、DT_INT16、DT_UINT16、DT_FP32、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP8E8M0、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
  | GM → L1 Buffer | ND → ND。 | 源与目的数据类型必须相同，仅支持DT_INT64、DT_UINT64。 |
  | GM → L1 Buffer | ND/DN → ZZ/NN，仅用于pypto_pro.language.matmul_mx或pypto_pro.language.matmul_mx_acc的量化系数搬运。 | 源与目的数据类型必须为DT_FP8E8M0。 |

- NZ搬运约束：

  - src_tensor声明为NZ时，dst_tile必须为NZ，且只支持从src_tensor的最后两个维度正序搬运，不支持通过降序order转置。NZ的物理排布和Tensor shape约束详见[TensorLayout](../basic_data_structures/TensorLayout.md)。
  - dst_tile的shape和valid_shape中，M必须按16对齐，N必须按src_tensor数据类型对应的C0对齐；tile_offsets换算成绝对元素偏移后，N方向的offset也必须按C0对齐。

- MX矩阵乘量化系数搬运约束：

  - 仅支持将DT_FP8E8M0的src_tensor搬入fractal为32、布局为ZZ或NN的L1 Buffer Tile，并作为pypto_pro.language.matmul_mx或pypto_pro.language.matmul_mx_acc的量化系数使用。
  - src_tensor的维数必须大于等于3，最后一维为长度等于2的物理phase轴，offsets中phase轴对应的偏移必须为0。显式设置order时，不能选择phase轴。

- Tile地址复用约束：

  开启auto_mutex时，如果连续两次pypto_pro.language.load_tile写入同一个UB或L1 Buffer地址，且两次搬运之间没有操作读取前一次搬入的数据，需要在两次搬运之间调用[pypto_pro.language.system.bar_mte2](../synchronization/bar_mte2.md)。pypto_pro.language.system.bar_mte2仅保证两次写操作的先后顺序；如果后续仍需使用前一次搬入的数据，应在复用地址前先读取或复制该数据。

## 返回值说明

无。

## 调用示例

### 按Tile块编号搬运

输入Tensor由4个[64, 64]数据块组成。循环变量tile_index依次取0～3，load_tile自动将其换算为第0维的绝对元素偏移0、64、128和192。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_tile_kernel(
    x: pl.Tensor[[256, 64], pl.DT_FP16],
    out: pl.Tensor[[256, 64], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    x_tiles = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    out_tiles = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])

    with pl.section_vector():
        for tile_index in pl.range(0, 4, 1):
            current_x = x_tiles.next()
            current_out = out_tiles.next()
            pl.load_tile(current_x, x, [tile_index, 0])
            pl.add(current_out, current_x, current_x)
            pl.store_tile(out, current_out, [tile_index, 0])
```

### 从高维Tensor中选择维度搬运

以下示例中，输入Tensor的维度依次为B、S、N、D。order=[1, 3]表示Tile的两个维度分别对应S、D，因此tile_offsets中的S、D按Tile块编号解释，B、N仍按绝对下标解释。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_tile_high_dim_kernel(
    x: pl.Tensor[[2, 128, 4, 64], pl.DT_FP16],
    out: pl.Tensor[[2, 128, 4, 64], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    x_tile = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    out_tile = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[1])

    with pl.section_vector():
        current_x = x_tile.current()
        current_out = out_tile.current()
        # 固定B=1、N=2；S方向块编号为1，对应绝对元素偏移64。
        pl.load_tile(current_x, x, [1, 1, 2, 0], order=[1, 3])
        pl.add(current_out, current_x, current_x)
        pl.store_tile(out, current_out, [1, 1, 2, 0], order=[1, 3])
```

### 转置搬运

转置场景同样通过降序order指定。以下示例使用shape=[64, 64]的方形Tile，tile_offsets=[1, 0]先换算为绝对元素坐标[64, 0]，再从普通ND GM Tensor转置搬入ZN Tile。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_tile_transpose_kernel(x: pl.Tensor[[128, 128], pl.DT_FP16]):
    tile_type = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.ZN,
    )
    x_l1 = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])

    with pl.section_cube():
        pl.load_tile(x_l1.current(), x, [1, 0], order=[1, 0])
```
