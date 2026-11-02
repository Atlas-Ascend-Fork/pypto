# pypto_pro.language.getval

<!-- md-trans-meta sourceCommit=a2b017e80ae1f790a0eec175e776160841b83c50 translatedAt=2026-10-26T14:51:25.303Z pushedAt=2026-10-28T11:57:29.304Z nl=2 -->

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

Reads the value of a single element at a specified linear offset from a Tile or Tensor and returns a scalar. The corresponding implementation is selected based on the type of the first parameter (Tile or Tensor).

## Prototype

```python
pypto_pro.language.getval(container: Tile | Tensor, offset: int) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| container | Input | Source operand, of Tile or Tensor type, from which a single element is read. A Tile must reside in UB; a Tensor is a GM Tensor. Integer or floating-point types that can participate in scalar expressions are supported. Low-precision types used only for storage, such as DT_FP4, DT_FP4E2M1, DT_FP4E1M2, DT_FP8E4M3FN, DT_FP8E5M2, DT_FP8E8M0, DT_INT4, DT_UINT4, DT_HF4, and DT_HF8, are not supported. |
| offset | Input | Read position, expressed as a linear offset in elements. Supports integer constants or runtime integer scalar expressions (including loop variables). The value range is 0 ≤ offset < total number of elements; behavior is undefined when out of range. You can also use `container[i]` or `container[i, j, ...]` to read by coordinates, in which case the compiler converts the subscripts to a linear offset. The number of subscripts must match the rank of the container. |

## Constraints

None

## Return Value

When the element type of container is an integer type other than DT_UINT64, a scalar of type DT_INT64 is returned uniformly; in other cases, a scalar of the same type as the container element type is returned.
The `container[i, j, ...]` read in the SIMD context also follows this rule.

In SIMT entry functions and their SIMT helper functions, when reading an element using `container[i, j, ...]`, the returned scalar preserves the element type of container without integer promotion, so that it can be used directly in SIMT operations that require exact type matching.
Explicitly calling pypto_pro.language.getval is not supported in SIMT functions; use the subscript syntax described above instead.

## Examples

### Tile Scenario

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def getval_setval_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_a_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        tile_a = tile_a_group.current()
        pl.load(tile_a, a, [0, 0])
        value = pl.getval(tile_a, 0)
        pl.setval(tile_a, 1, value)
        pl.store(a, tile_a, [0, 0])
```

### Tensor Scenario

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def tensor_getval_setval_kernel(
    scale_tensor: pl.Tensor[[2], pl.DT_FP32],
):
    scale = pl.getval(scale_tensor, 0)
    pl.setval(scale_tensor, 1, scale)
```
