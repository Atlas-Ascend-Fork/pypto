# pypto_pro.language.fused_mul_add

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

Fused multiply-add: out = lhs * out + rhs. Multiply lhs and out element by element, then add rhs, and write the result back to out. out serves as both a multiplication input and the accumulation output.

## Prototype

```python
pypto_pro.language.fused_mul_add(out: Tile, lhs: Tile, rhs: Tile) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand, a UB Tile, with data type DT_FP16 or DT_FP32, consistent with lhs and rhs. Its value before the call participates in the multiplication; after the call, it stores the result of lhs * out original value + rhs. Its valid Shape must be consistent with that of both source operands.|
| lhs | Input| Source operand (left operand), which is of type Tile and stored in UB; its data type, shape, and valid shape must be consistent with those of out.|
| rhs | Input| Source operand (right operand), which is of type Tile and stored in UB; its data type, shape, and valid shape must be consistent with those of out.|

## Return Value

None.

## Constraints

- out, lhs, and rhs must use a row-major layout.

## Examples

### Compute c = a × c + b.

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def fused_mul_add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32],
    c: pl.Tensor[[64, 64], pl.DT_FP32],
):
    # fused_mul_add is an in-place fused multiply-add: c = c * a + b.
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_c = tile_c.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.load(cur_c, c, [0, 0])
        pl.fused_mul_add(cur_c, cur_a, cur_b)
        pl.store(c, cur_c, [0, 0])
```

### Output

<!-- pypto-doc-output:fused_mul_add:start -->
```bash
Input a: [[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
Input b: [[-3 -2.5 -2 -1.5 -1 -0.5 0 0.5 ...], [29 29.5 30 30.5 31 31.5 32 32.5 ...], [61 61.5 62 62.5 63 63.5 64 64.5 ...], [93 93.5 94 94.5 95 95.5 96 96.5 ...], ...]
Original value of input c: [[2 1.875 1.75 1.625 1.5 1.375 1.25 1.125 ...], [-6 -6.125 -6.25 -6.375 -6.5 -6.625 -6.75 -6.875 ...], [-14 -14.125 -14.25 -14.375 -14.5 -14.625 -14.75 -14.875 ...], [-22 -22.125 -22.25 -22.375 -22.5 -22.625 -22.75 -22.875 ...], ...]
Output data c: [[-1 -0.15625 0.625 1.34375 2 2.59375 3.125 3.59375 ...], [-73 -76.15625 -79.375 -82.65625 -86 -89.40625 -92.875 -96.40625 ...], [-401 -408.15625 -415.375 -422.65625 -430 -437.40625 -444.875 -452.40625 ...], [-985 -996.15625 -1.007375e+03 -1.018656e+03 -1.030000e+03 -1.041406e+03 -1.052875e+03 -1.064406e+03 ...], ...]
```
<!-- pypto-doc-output:fused_mul_add:end -->
