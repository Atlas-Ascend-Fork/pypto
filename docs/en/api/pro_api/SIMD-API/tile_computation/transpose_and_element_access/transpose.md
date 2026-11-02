# pypto_pro.language.transpose

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:49:56.841Z pushedAt=2026-10-28T11:57:29.301Z nl=2 -->

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

Swaps the two axes of a 2D Tile to implement matrix transposition.

## Prototype

```python
pypto_pro.language.transpose(out: Tile, src: Tile) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| out | Output | Destination operand of the Tile type, which stores the transposition result. Its dtype is the same as that of src; its shape is the transposed shape of src. For example, when src is [64, 128], out is [128, 64]. It cannot be the same Tile as src. |
| src | Input | Source operand of the Tile type, a 2D UB. The dtype supports DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP16, DT_BF16, DT_INT32, DT_UINT32, and DT_FP32, and does not support 64-bit elements. The length of the major dimension of the source and destination multiplied by the element byte size must be 32-byte aligned. |

## Constraints

None

## Return Value

None

## Examples

### Basic Usage

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def transpose_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.transpose(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```

The measured results are as follows:

<!-- pypto-doc-output:transpose:start -->
```bash
Input data a: [[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
Output data out: [[1 17 33 49 65 81 97 113 ...], [1.25 17.25 33.25 49.25 65.25 81.25 97.25 113.25 ...], [1.5 17.5 33.5 49.5 65.5 81.5 97.5 113.5 ...], [1.75 17.75 33.75 49.75 65.75 81.75 97.75 113.75 ...], ...]
```
<!-- pypto-doc-output:transpose:end -->
