# pypto_pro.language.neg

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:43:21.685Z pushedAt=2026-10-28T11:57:29.255Z nl=2 -->

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

Computes the opposite number of each element in the source operand src and writes the result to the destination operand out.

## Prototype

```python
pypto_pro.language.neg(
    out: Tile,
    src: Tile,
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| out | Output | Destination operand of the Tile type, which stores the result of element-wise negation. The data type is the same as that of src, supporting DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FP16, DT_BF16, or DT_FP32. The shape is the same as that of src. It can be the same Tile as src for in-place computation. |
| src | Input | Source operand of the Tile type. The data type is the same as that of out. |

## Constraints

None

## Return Value

None

## Examples

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def neg_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32],
               out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.neg(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```

The following shows an example of the actual test result.

<!-- pypto-doc-output:neg:start -->
```bash
Input data a: [[-4 -3.875 -3.75 -3.625 -3.5 -3.375 -3.25 -3.125 ...], [4 4.125 4.25 4.375 4.5 4.625 4.75 4.875 ...], [12 12.125 12.25 12.375 12.5 12.625 12.75 12.875 ...], [20 20.125 20.25 20.375 20.5 20.625 20.75 20.875 ...], ...]
Output data out: [[4 3.875 3.75 3.625 3.5 3.375 3.25 3.125 ...], [-4 -4.125 -4.25 -4.375 -4.5 -4.625 -4.75 -4.875 ...], [-12 -12.125 -12.25 -12.375 -12.5 -12.625 -12.75 -12.875 ...], [-20 -20.125 -20.25 -20.375 -20.5 -20.625 -20.75 -20.875 ...], ...]
```
<!-- pypto-doc-output:neg:end -->
