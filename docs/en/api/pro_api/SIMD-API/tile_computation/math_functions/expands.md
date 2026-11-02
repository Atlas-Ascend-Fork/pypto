# pypto_pro.language.expands

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:46:29.572Z pushedAt=2026-10-28T11:57:29.275Z nl=2 -->

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

Fills a Tile with a specified scalar value. It is commonly used to initialize a Tile to negative infinity or zero.

## Prototype

```python
pypto_pro.language.expands(
    out: Tile,
    scalar: Scalar,
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| out | Output | Destination operand of the Tile type. All elements are filled with the scalar value.<br>Supported data types: DT_UINT8, DT_INT8, DT_UINT16, DT_INT16, DT_UINT32, DT_INT32, DT_INT64, DT_UINT64, DT_FP16, DT_BF16, DT_FP32.<br>Located in UB or L1 Buffer. |
| scalar | Input | Fill value. An integer or floating-point constant, or a runtime integer or floating-point scalar expression whose type must be compatible with the element type of out. |

## Constraints

None

## Return Value

None

## Examples

### Basic Usage

```python
import pypto_pro.language as pl

K_VALUE = 2.0


@pl.jit(auto_mutex=True)
def expands_kernel(dummy: pl.Tensor[[64, 64], pl.DT_FP32],
                   out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_out = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        cur_out = tile_out.current()
        pl.expands(cur_out, K_VALUE)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:expands:start -->
```bash
Input data K_VALUE: 2
Output data out: [[2 2 2 2 2 2 2 2 ...], [2 2 2 2 2 2 2 2 ...], [2 2 2 2 2 2 2 2 ...], [2 2 2 2 2 2 2 2 ...], ...]
```
<!-- pypto-doc-output:expands:end -->

### Initializing a Negative-Infinity Tile

```python
# Initialize a negative-infinity tile (causal mask).
pl.expands(neg_inf_vec, NEG_INF)
```

### Initializing a Zero Tile

```python
# Initialize a zero tile.
pl.expands(score_u16_row, 0)
```
