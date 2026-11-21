# pypto_pro.language.minimum

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

minimum supports both element-wise minimum computation and minimum reduction along a dimension. The call mode is determined by whether dim is passed.

- **Tile-Tile mode**: The elements at corresponding positions of lhs and rhs are compared, and the smaller value is written to out.
- **Tile-Scalar mode**: Each element in lhs is compared with the rhs scalar, and the smaller value is written to out.
- **Reduction mode**: The minimum value of lhs along the dimension specified by dim is computed, and the result is written to out.

## Prototype

```python
pypto_pro.language.minimum(
    out: Tile,
    lhs: Tile,
    rhs: Union[Tile, int, float],
    *,
    dim: Optional[int]=None,
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand of type Tile, which stores the element-wise computation result or reduction result. In element-wise mode, its data type is the same as that of lhs and it supports DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_BF16, and DT_FP32. In reduction mode, its data type is the same as that of lhs; when dim=0, its shape is [number of rows, 1], and when dim=1, its shape is [1, number of columns].|
| lhs | Input| Source tile. In element-wise mode, it is the left operand and supports DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_FP32 and DT_BF16. In reduction mode, it must be a 2D tile. For details about the supported types, see [Constraints] (#constraints).|
| rhs | Input| Tile, int, or float type. The corresponding runtime numeric scalar expression is also supported. In element-wise mode, it is the right operand. In Tile-Tile mode, its data type, shape, and valid shape are the same as those of lhs. In reduction mode, a temporary tile must be provided, and its data type must be the same as that of lhs.|
| dim | Input| (Optional) Reduction dimension. If this parameter is not passed, element-wise computation is performed. If this parameter is set to 0, reduction is performed along the last dimension. If this parameter is set to 1, reduction is performed along the first dimension.|

## Return Value

None

## Constraints

- In element-wise mode, out, lhs, and a rhs of type Tile must be located in UB, and their data types, shapes, and valid shapes must be consistent.
- In reduction mode, dim can only be 0 or 1, and the data type of out must be the same as that of lhs; when dim=0, the output shape is [M, 1]; when dim=1, the output shape is [1, N].

| Reduction Dimension| Supported Data Types of lhs|
|---|---|
| dim=0 | DT_FP16, DT_FP32, DT_INT32, DT_INT64, DT_UINT64, DT_INT16, DT_INT8, and DT_UINT8|
| dim=1 | DT_FP16, DT_FP32, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, and DT_BF16|

## Examples

### Tile-Tile Mode

Below is a complete Kernel. It loads two DT_FP32 inputs from GM to UB, uses pypto_pro.language.minimum to compute the minimum value element-wise, and writes the result back to GM. The Vector Kernel has auto_mutex enabled, and synchronization is managed automatically by pypto_pro.language.make_tile_group.

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def minimum_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32],
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
        pl.minimum(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

An example of the result is as follows.

<!-- pypto-doc-output:minimum:start -->
```bash
Input a: [[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
Input b: [[10 10.5 11 11.5 12 12.5 13 13.5 ...], [42 42.5 43 43.5 44 44.5 45 45.5 ...], [74 74.5 75 75.5 76 76.5 77 77.5 ...], [106 106.5 107 107.5 108 108.5 109 109.5 ...], ...]
Output data out: [[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
```
<!-- pypto-doc-output:minimum:end -->

### Tile-Scalar Mode

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def minimum_scalar_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.minimum(cur_out, cur_a, 100.0)
        pl.store(out, cur_out, [0, 0])
```

### Reduction Mode

#### Result example when dim = 0

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def row_min_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
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
        pl.minimum(cur_out, cur_a, cur_tmp, dim=0)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:row_min:start -->
```bash
Input data a: [[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
Output data out: [[-8], [24], [56], [88], ...]
```
<!-- pypto-doc-output:row_min:end -->

#### Result example when dim = 1

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def col_min_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
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
        pl.minimum(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:col_min:start -->
```bash
Input data a: [[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
Output data out: [[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...]]
```
<!-- pypto-doc-output:col_min:end -->
