# pypto_pro.language.sum

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

Performs a sum reduction on the source Tile along the specified dimension. When dim=0, each row is summed along the row direction; when dim=1, each column is summed along the column direction.

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

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand of the Tile type, which stores the reduction result. For details about the supported data types, see [Constraints](#constraints).<br>If the shape of src is [M, N], the shape is [M, 1] when dim=0 and [1, N] when dim=1.|
| src | Input| Source operand of the Tile type. For details about the supported data types, see [Constraints](#constraints).|
| tmp | Input| Temporary storage of type Tile, which is a compatibility parameter.<br>When dim=0 is lowered to TROWSUM, this parameter is not read or written. When dim=1 is lowered to TCOLSUM, the non-binary path does not use this parameter by default.|
| dim | Input| Reduction dimension. The value 0 indicates reduction along the row direction, and the value 1 indicates reduction along the column direction.|

## Constraints

- Data type: The data types supported by src and out vary depending on the value of dim. The details are as follows.

  | Parameter| dim=0 | dim=1 |
  |---|---|---|
  | src | DT_INT16, DT_INT32, DT_INT64, DT_UINT64, DT_FP16, and DT_FP32| DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_BF16, and DT_FP32|
  | out | Same as src| Same as src|

- FP16/BF16 precision: When src (and out, which matches src) is of type DT_FP16 or DT_BF16, the reduction can be affected by input quantization, finite-precision accumulation, and output rounding. For large input sizes or large values, device-computed results may differ from high-precision reference results. The accumulation order used by the reduction instruction can also affect the final result.

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
Input a: [[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
Output out: [[1.008000e+03], [5.104000e+03], [9.200000e+03], [1.329600e+04], ...]
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
Input a: [[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
Output out: [[6.400000e+04 6.401600e+04 6.403200e+04 6.404800e+04 6.406400e+04 6.408000e+04 6.409600e+04 6.411200e+04 ...]]
```
<!-- pypto-doc-output:col_sum:end -->

### Precision Difference with FP16 Input

FP16 reduction can be affected by input quantization, finite-precision accumulation, and output rounding. For large input sizes or large values, device-computed results may differ from high-precision reference results. The accumulation order used by the reduction instruction can also affect the final result.

The following are the measured on-device results for sum(..., dim=1) with FP16 input.

<!-- pypto-doc-output:col_reduce_sum:start -->
```bash
Input a: [[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
Output z: [[6.390400e+04 6.390400e+04 6.390400e+04 6.416000e+04 6.416000e+04 6.416000e+04 6.419200e+04 6.419200e+04 ...]]
```
<!-- pypto-doc-output:col_reduce_sum:end -->
