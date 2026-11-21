# pypto_pro.language.recip

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

Computes the element-wise reciprocal of src and writes the result to out. For floating-point types, the computation is 1/src. For DT_INT16 and DT_INT32, integer division is used with truncation toward zero; for example, when src is 2, the result is 0.

## Prototype

```python
pypto_pro.language.recip(out: Tile, src: Tile) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand of the Tile type, which stores the element-wise reciprocal result.<br>The supported data types are DT_INT16, DT_INT32, DT_FP16, and DT_FP32.<br>The shape must be the same as that of src. If valid_shape is set for src, valid_shape must also be set for out and must be the same as that of src.<br>It can be the same Tile as src to implement in-place computation.|
| src | Input| Source operand, which is of type Tile.<br>The supported data types are DT_INT16, DT_INT32, DT_FP16, and DT_FP32. They are the same as those of out.<br>Element values must not be zero. Otherwise, the hardware behavior is undefined, and the CPU simulator will trigger an assertion in debug mode.|

## Return Value

None

## Constraints

Valid elements of src must not be 0. Integer types do not retain fractional parts; when a floating-point reciprocal is required, use DT_FP16 or DT_FP32.

## Examples

### Basic Usage

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def recip_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32],
                 out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.recip(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:recip:start -->
```bash
Input a: [[1 1.125 1.25 1.375 1.5 1.625 1.75 1.875 ...], [9 9.125 9.25 9.375 9.5 9.625 9.75 9.875 ...], [17 17.125 17.25 17.375 17.5 17.625 17.75 17.875 ...], [25 25.125 25.25 25.375 25.5 25.625 25.75 25.875 ...], ...]
Output out: [[1 0.888889 0.8 0.727273 0.666667 0.615385 0.571429 0.533333 ...], [0.111111 0.109589 0.108108 0.106667 0.105263 0.103896 0.102564 0.101266 ...], [0.058824 0.058394 0.057971 0.057554 0.057143 0.056738 0.056338 0.055944 ...], [0.04 0.039801 0.039604 0.039409 0.039216 0.039024 0.038835 0.038647 ...], ...]
```
<!-- pypto-doc-output:recip:end -->
