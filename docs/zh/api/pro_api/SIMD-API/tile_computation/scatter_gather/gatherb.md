# pypto_pro.language.gatherb

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

从源Tile中按offsets指定的字节偏移，每次取32字节的数据块（如16个DT_FP16元素），拼入目标Tile。

## 函数原型

```python
pypto_pro.language.gatherb(out: Tile, src: Tile, offsets: Tile) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存放按字节偏移聚合的结果。dtype与src一致，支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32。shape与valid_shape须与期望输出匹配。 |
| src | 输入 | 源操作数，Tile类型，dtype与out一致，shape与out一致。 |
| offsets | 输入 | 字节偏移Tile，支持DT_INT32、DT_UINT32。每个有效偏移指定一个32字节数据块的起始位置；偏移必须非负，且该块必须完全位于src可读取的存储范围内。设输出每行有效元素数为C、元素字节数为B，则每行至少需要ceil(C * B / 32)个有效偏移。物理Shape还须满足偏移Tile自身的布局对齐要求。 |

## 返回值说明

无。

## 约束说明

- 一个完整32字节块必须满足offset + 32不超过src可读取的字节范围；只检查offset小于源总字节数不足以避免越界。
- out、src和offsets须位于UB，输出与源的元素位宽一致。调用者须按实际读取范围提供全部块，尾块也须有合法的源存储空间。

## 调用示例

### 基本用法

```python
import pypto_pro.language as pl

DTYPE_SIZE = 2          # FP16
BLOCK_BYTES = 32
BLOCK_ELEMS = BLOCK_BYTES // DTYPE_SIZE   # 16
ROWS, COLS = 64, 128
OFFSETS_PER_ROW = COLS // BLOCK_ELEMS     # 8


@pl.jit(auto_mutex=True)
def gatherb_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    offsets: pl.Tensor[[64, 8], pl.DT_UINT32],
    dst: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_src_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_offsets_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 8], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    tile_dst_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4800, mutex_ids=[2])
    with pl.section_vector():
        tile_src = tile_src_group.current()
        tile_offsets = tile_offsets_group.current()
        tile_dst = tile_dst_group.current()
        pl.load(tile_src, src, [0, 0])
        pl.load(tile_offsets, offsets, [0, 0])
        pl.gatherb(tile_dst, tile_src, tile_offsets)
        pl.store(dst, tile_dst, [0, 0])
```

结果示例如下：

<!-- pypto-doc-output:gatherb:start -->
```bash
输入数据src：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 ...], ...]
输入数据offsets：[[0 32 64 96 128 160 192 224], [256 288 320 352 384 416 448 480], [512 544 576 608 640 672 704 736], [768 800 832 864 896 928 960 992], ...]
输出数据dst：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 ...], ...]
```
<!-- pypto-doc-output:gatherb:end -->
