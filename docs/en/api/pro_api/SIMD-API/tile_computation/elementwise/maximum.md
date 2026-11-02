# pypto_pro.language.maximum

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:41:52.954Z pushedAt=2026-10-28T11:57:29.246Z nl=2 -->

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->

## Description

maximum supports both element-wise maximum and maximum reduction along a dimension. Whether dim is passed determines the calling mode.

- **Tile-Tile mode**: Compares the elements at corresponding positions of lhs and rhs, and writes the larger value to out.
- **Tile-Scalar mode**: Compares each element of lhs with the rhs scalar, and writes the larger value to out.
- **Reduction mode**: Computes the maximum of lhs along the dimension specified by dim, and writes the result to out.

## Prototype

```python
pypto_pro.language.maximum(
    out: Tile,
    lhs: Tile,
    rhs: Union[Tile, Scalar],
    *,
    dim: Optional[int] = None,
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| out | Output | Destination operand of Tile type, which stores the element-wise computation result or the reduction result. In element-wise mode, its data type is the same as that of lhs, supporting DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_BF16, and DT_FP32. In reduction mode, its data type is the same as that of lhs; when dim=0, its shape is [number of rows, 1], and when dim=1, its shape is [1, number of columns]. |
| lhs | Input | Tile type. In element-wise mode, it is the left operand; in reduction mode, it is the source operand, which must be a two-dimensional Tile. When dim=0, it supports DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_FP16, DT_FP32, DT_INT64, and DT_UINT64; when dim=1, it additionally supports DT_UINT16, DT_UINT32, and DT_BF16. |
| rhs | Input | Tile or Scalar type. In element-wise mode, it is the right operand; in Tile-Tile mode, its data type is the same as that of out and its shape is the same as those of out and lhs. In reduction mode, it is a temporary Tile. |
| dim | Input | Optional, the reduction dimension. If not passed, element-wise computation is performed; if 0 is passed, reduction is performed along the last dimension; if 1 is passed, reduction is performed along the first dimension. |

## Constraints

None

## Return Value

None

## Examples

### Tile-Tile Mode

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def maximum_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32],
                   out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.maximum(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

The measured results are as follows.

<!-- pypto-doc-output:maximum:start -->
```bash
Input data a: [[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
Input data b: [[10 10.5 11 11.5 12 12.5 13 13.5 ...], [42 42.5 43 43.5 44 44.5 45 45.5 ...], [74 74.5 75 75.5 76 76.5 77 77.5 ...], [106 106.5 107 107.5 108 108.5 109 109.5 ...], ...]
Output data out: [[10 10.5 11 11.5 12 12.5 13 13.5 ...], [42 42.5 43 43.5 44 44.5 45 45.5 ...], [74 74.5 75 75.5 76 76.5 77 77.5 ...], [106 106.5 107 107.5 108 108.5 109 109.5 ...], ...]
```
<!-- pypto-doc-output:maximum:end -->

### Tile-Scalar Mode

```python
# Take the larger value between each element of Tile and the Scalar value.
pl.maximum(out, lhs, 0.0)
```

### reduction mode

```python
pl.maximum(row_out, src, tmp, dim=0)  # row_out shape: [rows, 1]
pl.maximum(col_out, src, tmp, dim=1)  # col_out shape: [1, columns]
```

#### dim = 0 Measured Results

```python
@pl.jit(auto_mutex=True)
def row_max_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
                   out: pl.Tensor[[64, 1], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                         layout=pl.DN)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.maximum(cur_out, cur_a, cur_tmp, dim=0)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:row_max:start -->
```bash
Input data a: [[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
Output data out: [[23.75], [55.75], [87.75], [119.75], ...]
```
<!-- pypto-doc-output:row_max:end -->

#### Measured Results for dim = 1

```python
@pl.jit(auto_mutex=True)
def col_max_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
                   out: pl.Tensor[[1, 128], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.maximum(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:col_max:start -->
```bash
Input data a: [[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
Output data out: [[2.008000e+03 2.008250e+03 2.008500e+03 2.008750e+03 2.009000e+03 2.009250e+03 2.009500e+03 2.009750e+03 ...]]
```
<!-- pypto-doc-output:col_max:end -->
