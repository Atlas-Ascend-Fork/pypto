# pypto_pro.language.sum

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T14:47:48.858Z pushedAt=2026-10-28T11:57:29.288Z nl=2 -->

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

Sums the source Tile along a specified dimension. When dim=0, each row is summed along the row direction; when dim=1, each column is summed along the column direction.

## Prototype

```python
pypto_pro.language.sum(
    out: Tile,
    src: Tile,
    tmp: Tile,
    *,
    dim: int = 0,
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| out | Output | Destination operand of the Tile type, which stores the reduction result. For supported data types, see [Constraints](#constraints).<br>If the shape of src is [M, N], the shape is [M, 1] when dim=0 and [1, N] when dim=1. |
| src | Input | Source operand of the Tile type. For supported data types, see [Constraints](#constraints). |
| tmp | Input | Temporary storage of the Tile type, a compatibility parameter.<br>When dim=0 is reduced to TROWSUM, this parameter is neither read nor written; when dim=1 is reduced to TCOLSUM, this parameter is not used on the default non-binary path. |
| dim | Input | Reduction dimension. 0 indicates reduction along the row direction, and 1 indicates reduction along the column direction. |

## Constraints

- Data types: When dim takes different values, the supported data type ranges for src and out differ, as follows.

  | Parameter | dim=0 | dim=1 |
  |---|---|---|
  | src | DT_INT16, DT_INT32, DT_INT64, DT_UINT64, DT_FP16, DT_FP32 | DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_BF16, DT_FP32 |
  | out | Keep consistent with src | Keep consistent with src |

- FP16/BF16 precision: When the data type of src (and out, which keeps consistent with src) is DT_FP16 or DT_BF16, the reduction is affected by input quantization, finite-precision accumulation, and output rounding. When the input scale is large or the values are large, the device computation result may differ from the high-precision reference result. The accumulation order used by the reduction instruction may also affect the final result.

## Return Value

None

## Examples

### dim = 0

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def row_sum_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
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
        pl.sum(cur_out, cur_a, cur_tmp)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:row_sum:start -->
```bash
Input data a: [[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
Output data out: [[1.008000e+03], [5.104000e+03], [9.200000e+03], [1.329600e+04], ...]
```
<!-- pypto-doc-output:row_sum:end -->

### dim = 1

```python
@pl.jit(auto_mutex=True)
def col_sum_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
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
        pl.sum(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:col_sum:start -->
```bash
Input data a: [[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
Output data out: [[6.400000e+04 6.401600e+04 6.403200e+04 6.404800e+04 6.406400e+04 6.408000e+04 6.409600e+04 6.411200e+04 ...]]
```
<!-- pypto-doc-output:col_sum:end -->

### Precision Differences with FP16 Input

FP16 reduction is affected by input quantization, finite-precision accumulation, and output rounding. When the input scale is large or the values are large, the device computation result may differ from the high-precision reference result. The accumulation order used by the reduction instruction may also affect the final result.

The following shows the device-measured result of sum(..., dim=1) with FP16 input:

<!-- pypto-doc-output:col_reduce_sum:start -->
```bash
Input data a: [[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
Output data z: [[6.390400e+04 6.390400e+04 6.390400e+04 6.416000e+04 6.416000e+04 6.416000e+04 6.419200e+04 6.419200e+04 ...]]
```
<!-- pypto-doc-output:col_reduce_sum:end -->
