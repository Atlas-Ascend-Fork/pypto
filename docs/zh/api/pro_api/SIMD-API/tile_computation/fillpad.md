# pypto_pro.language.fillpad

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

填充Tile有效数据之外的padding区域，用于尾块数据在后续计算前的补齐。UB路径按src的有效区域保留数据并填充其余指定区域；L1 Buffer路径只对out补零，调用前须将有效数据装载到out。填充值及布局约束与Tile所在存储空间有关，mode用于选择源、目的Tile的地址和Shape关系。模式定义参见[pypto_pro.language.FillPadMode](../basic_data_structures/FillPadMode.md)。

## 函数原型

```python
pypto_pro.language.fillpad(out: Tile, src: Tile, *, mode: FillPadMode=FillPadMode.NORMAL) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的Tile，位于UB或L1 Buffer，保存填充结果。支持4、8、16、32位存储类型，具体布局、pad和模式要求参见[约束说明](#约束说明)。 |
| src | 输入 | 源Tile，须与out位于同一种Buffer，元素位宽与out一致。UB路径中，源有效Shape指定需要保留的数据范围；L1 Buffer路径不从src复制有效数据，初始化要求及补零边界参见[约束说明](#约束说明)。 |
| mode | 输入 | 可选，[pypto_pro.language.FillPadMode](../basic_data_structures/FillPadMode.md)枚举值，默认FillPadMode.NORMAL，须在编译期确定。 |

## 返回值说明

无。

## 约束说明

### UB Tile

- out必须配置有效的pad属性，例如pypto_pro.language.TilePad.zero。
- src的有效行数不得大于out的有效行数，out的有效行数不得超过其物理行数；src的有效列数不得超过out的物理列数。
- 在src的有效行内，填充范围为src有效列之后至out物理列末尾；在src有效行之后，填充至out有效行数为止的整行。out有效行之后的物理行不保证被填充。需要补齐整个目的Tile时，应将out的有效Shape设为其物理Shape。
- NORMAL模式下，源、目的物理Shape相同，地址不同。
- EXPAND模式下，out的各维物理Shape不得小于src，源、目的地址不同。
- INPLACE模式下，源、目的物理Shape相同，必须共享同一地址；有效数据在原地址保留。
- 使用4位类型时，Shape按逻辑元素计数，物理Shape最后一维须满足两个元素共用一个字节的打包要求。

### L1 Buffer Tile

- 使用NORMAL模式和NZ布局，out与src须为相同Tile类型，包括数据类型、物理Shape、有效Shape和pad配置。
- 当前L1 Buffer路径用于零填充。使用pypto_pro.language.TilePad.null创建源、目的Tile，不应将UB的其他pad策略直接用于该路径。
- 调用前须在out中装载需要保留的有效数据。该路径只填充out的padding，不从独立的src复制有效数据；补零边界由out的有效Shape决定，不使用src的运行时有效Shape。
- 可对已初始化的L1 Buffer Tile调用`pypto_pro.language.fillpad(tile, tile)`，在原有有效数据之外补零。
- EXPAND和INPLACE模式用于UB Tile，不用于L1 Buffer Tile。

## 调用示例

### NORMAL模式示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def fillpad_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 8], pl.DT_INT32],
):
    src_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    dst_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
    src = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    dst = pl.make_tile_group(type=dst_type, addrs=0x0100, mutex_ids=[1])
    with pl.section_vector():
        cur_src = src.current()
        cur_dst = dst.current()
        pl.set_validshape(cur_src, [5, 7])
        pl.load(cur_src, x, [0, 0])
        pl.fillpad(cur_dst, cur_src)
        pl.store(z, cur_dst, [0, 0])
```

### EXPAND模式示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def fillpad_expand_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 16], pl.DT_INT32],
):
    src_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    dst_type = pl.TileType(shape=[8, 16], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
    src = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    dst = pl.make_tile_group(type=dst_type, addrs=0x0100, mutex_ids=[1])
    with pl.section_vector():
        cur_src = src.current()
        cur_dst = dst.current()
        pl.set_validshape(cur_src, [5, 7])
        pl.load(cur_src, x, [0, 0])
        pl.fillpad(cur_dst, cur_src, mode=pl.FillPadMode.EXPAND)
        pl.store(z, cur_dst, [0, 0])
```

### INPLACE模式示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def fillpad_inplace_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 8], pl.DT_INT32],
):
    src_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec,
                           pad=pl.TilePad.zero, valid_shape=[-1, -1])
    dst_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
    src = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    dst = pl.make_tile_group(type=dst_type, addrs=0x0000, mutex_ids=[1])
    with pl.section_vector():
        cur_src = src.current()
        cur_dst = dst.current()
        pl.set_validshape(cur_src, [5, 7])
        pl.load(cur_src, x, [0, 0])
        pl.fillpad(cur_dst, cur_src, mode=pl.FillPadMode.INPLACE)
        pl.store(z, cur_dst, [0, 0])
```
