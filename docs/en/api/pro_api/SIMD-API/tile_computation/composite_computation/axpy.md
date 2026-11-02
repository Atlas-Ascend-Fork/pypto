# pypto_pro.language.axpy

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T14:38:58.565Z pushedAt=2026-10-28T11:57:29.226Z nl=2 -->

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

Vector-scalar multiply-add: out[i] = alpha * src[i] + out[i]. Multiplies each element of the source Tile by the scalar alpha and accumulates the result into the destination Tile.

## Prototype

```python
pypto_pro.language.axpy(out: Tile, src: Tile, alpha: Scalar, /) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| out | Input/Output | Destination operand of Tile type, stored in UB, with a data type of DT_FP16 or DT_FP32. Its data type, shape, and valid shape must be consistent with those of src. It holds the initial accumulation value before the operation and the computation result after the operation. |
| src | Input | Source operand of Tile type, stored in UB, with a data type of DT_FP16 or DT_FP32. Its shape and valid shape must be consistent with those of out. |
| alpha | Input | Scaling factor of float or Scalar type, which can be a compile-time constant or a runtime floating-point Scalar. Its data type must be consistent with that of src. |

## Constraints

- out and src must be arranged in row-major order.

## Return Value

None

## Examples

### Compute y = 2.0 × x + y

```python
import pypto_pro.language as pl

ALPHA = 2.0


@pl.jit(auto_mutex=True)
def axpy_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_y_group = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        tile_x = tile_x_group.current()
        tile_y = tile_y_group.current()
        pl.load(tile_x, x, [0, 0])
        pl.load(tile_y, y, [0, 0])
        pl.axpy(tile_y, tile_x, ALPHA)
        pl.store(y, tile_y, [0, 0])
```

### Result

<!-- pypto-doc-output:axpy:start -->
```bash
Input data x: [[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
Original input data y: [[4 3.875 3.75 3.625 3.5 3.375 3.25 3.125 ...], [-4 -4.125 -4.25 -4.375 -4.5 -4.625 -4.75 -4.875 ...], [-12 -12.125 -12.25 -12.375 -12.5 -12.625 -12.75 -12.875 ...], [-20 -20.125 -20.25 -20.375 -20.5 -20.625 -20.75 -20.875 ...], ...]
Output data y: [[6 6.375 6.75 7.125 7.5 7.875 8.25 8.625 ...], [30 30.375 30.75 31.125 31.5 31.875 32.25 32.625 ...], [54 54.375 54.75 55.125 55.5 55.875 56.25 56.625 ...], [78 78.375 78.75 79.125 79.5 79.875 80.25 80.625 ...], ...]
```
<!-- pypto-doc-output:axpy:end -->
