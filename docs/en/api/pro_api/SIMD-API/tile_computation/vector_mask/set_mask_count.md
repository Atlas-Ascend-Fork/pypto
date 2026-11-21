# pypto_pro.language.set_mask_count

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

Switches the vector mask register to **count mode**. In this mode, the lower 64 bits of the mask register are interpreted as the number of elements participating in the computation, rather than as a per-bit mask. The computation data size can therefore be set directly, and the actual iteration count is inferred by the Vector compute unit.

When this API is used in conjunction with [pypto_pro.language.set_vec_mask](set_vec_mask.md), switch to count mode first, and then set the number of elements to precisely control the number of valid elements in the tail block computation.

## Prototype

```python
pypto_pro.language.set_mask_count() -> None
```

## Parameters

None

## Return Value

None

## Constraints

- After switching to the count mode, use [pypto_pro.language.set_vec_mask](set_vec_mask.md) to set the number of valid elements. Otherwise, the mask value is undefined.
- In count mode, the mask_low parameter of [pypto_pro.language.set_vec_mask](set_vec_mask.md) is interpreted as the element count (rather than a bit mask), and mask_high is ignored.
- After the tail block has been processed in count mode, call [pypto_pro.language.set_mask_norm](set_mask_norm.md) first to switch back to norm mode, and then call [pypto_pro.language.reset_mask](reset_mask.md) to restore the all-ones mask. pypto_pro.language.set_mask_norm itself does not modify the current mask value. Otherwise, the mask behavior of subsequent vector computations is undefined.
- Tile-level operations (such as [add](../elementwise/add.md) and [mul](../elementwise/mul.md)) manage the mask internally; a count mode set manually by the user will be overwritten. pypto_pro.language.set_mask_count is mainly intended for custom operations or low-level control scenarios.

## Examples

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def mask_count_norm_kernel(
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
        pl.set_mask_count()
        pl.set_vec_mask(0, 64 * 64)
        # Perform custom vector computation that reads the count mode mask.
        pl.set_mask_norm()
        pl.reset_mask()
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```
