# pypto_pro.language.setval

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

Writes a scalar value to the specified linear offset position in a Tile or Tensor, used together with [pypto_pro.language.getval](getval.md). The corresponding implementation is selected based on the type of the first parameter (Tile or Tensor).

## Prototype

```python
pypto_pro.language.setval(container: Tile | Tensor, offset: int, value: Union[int, float]) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| container | Output| Destination operand, of the Tile or Tensor type, to which a single element is written. A Tile must reside in UB and a Tensor is a GM Tensor. Integer or floating-point types that can participate in scalar expressions are supported. Low-precision types used only for storage, such as DT_FP4, DT_FP4E2M1, DT_FP4E1M2, DT_FP8E4M3FN, DT_FP8E5M2, DT_FP8E8M0, DT_INT4, DT_UINT4, DT_HF4, and DT_HF8, are not supported. The type of the value to be written must match the element type or be constructible by the frontend according to the element type.|
| offset | Input| Write position, expressed as a linear offset in elements. Integer constants or runtime integer scalar expressions (including loop variables) are supported. The value range is 0 ≤ offset < total number of elements. Behavior is undefined when out of range. You can also write by coordinates using `container[i] = value` or `container[i, j, ...] = value`, in which case the compiler converts the subscripts into a linear offset. The number of subscripts must match the rank of the container.|
| value | Input| Scalar value to be written. An integer or floating-point constant, or a runtime integer or floating-point scalar expression, whose type must be compatible with the element type of container.|

## Return Value

None

## Constraints

None

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
