# pypto_pro.language.set_vec_mask

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

Explicitly sets the 128-bit value (higher 64 bits + lower 64 bits) of the vector mask register, specifying bit by bit which elements participate in subsequent vector computations.

The meaning of the mask register depends on the current mode:

- **norm mode** (default): 128-bit bitwise mask, where each bit indicates whether the corresponding element is active. A bit value of 1 means the element participates in the computation; 0 means it does not.
- **count mode**: mask_low is interpreted as the number of valid elements, and mask_high is ignored.

Mode switching is controlled by [pypto_pro.language.set_mask_norm](set_mask_norm.md) or [pypto_pro.language.set_mask_count](set_mask_count.md).

## Prototype

```python
pypto_pro.language.set_vec_mask(mask_high: int, mask_low: int) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| mask_high | Input| Integer constant or a runtime integer scalar expression representing a 64-bit mask bit pattern. In norm mode, it controls the active state of elements 64–127. In count mode, it is ignored.|
| mask_low | Input| Integer constant or a runtime integer scalar expression representing a 64-bit mask bit pattern. In norm mode, it controls the active state of elements 0–63. In count mode, it indicates the number of valid elements.|

## Return Value

None

## Constraints

- In norm mode, mask_high and mask_low are the raw mask values of the higher and lower 64 bits, respectively. The number of valid bits and the correspondence between bits and data elements are determined by the data type and instruction semantics of the subsequent vector instructions.
- In count mode, mask_low represents the total number of elements participating in the computation and is not interpreted bit-wise. This mode is used to process the tail block (where the number of valid elements is not aligned); the actual iteration count is inferred automatically by the Vector compute unit.
- In norm mode, setting both the higher and lower 64 bits to 0 indicates that the subsequent vector instructions using this mask have no active elements; the pypto_pro.language.set_vec_mask call itself still writes the mask register.
- Tile-level operations (such as [add](../elementwise/add.md) and [mul](../elementwise/mul.md)) manage the mask internally; a mask set by the user via pypto_pro.language.set_vec_mask will be overwritten. pypto_pro.language.set_vec_mask is mainly intended for custom operations or low-level control scenarios.
- After setting the mask with pypto_pro.language.set_vec_mask, [pypto_pro.language.reset_mask](reset_mask.md) must be called after the subsequent computation completes to restore the default all-ones mask; otherwise, subsequent vector computations may be affected.

## Examples

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def set_vec_mask_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        # Clear the higher 64 bits to zero, and set the lower 32 mask bits of the lower 64 bits to 1.
        pl.set_vec_mask(0, 0xFFFFFFFF)
        # Perform custom vector computation that reads the current mask state.
        pl.reset_mask()
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```
