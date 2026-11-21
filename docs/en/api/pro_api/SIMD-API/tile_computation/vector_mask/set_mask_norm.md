# pypto_pro.language.set_mask_norm

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

Switches the vector mask register back to **norm (plain per-bit mask) mode**. In this mode, the 128 bits of the mask register are interpreted as a per-bit mask, where each bit indicates whether the corresponding element participates in the computation.

This is the inverse operation of [pypto_pro.language.set_mask_count](set_mask_count.md): count mode is used for tail-block counting, while norm mode restores the default per-bit mask semantics.

## Prototype

```python
pypto_pro.language.set_mask_norm() -> None
```

## Parameters

None

## Return Value

None

## Constraints

- pypto_pro.language.set_mask_norm only switches the mask interpretation mode and does not overwrite the current mask value.
- After the tail block has been processed in count mode, pypto_pro.language.set_mask_norm must be called to switch back to norm mode; otherwise, the mask behavior of subsequent vector computations is undefined.
- After switching back to norm mode, the mask register still holds the value set in count mode. It must be reset either by calling [pypto_pro.language.set_vec_mask](set_vec_mask.md) again or by calling [pypto_pro.language.reset_mask](reset_mask.md) to restore the all-ones mask.
- Tile-level operations (such as [add](../elementwise/add.md) and [mul](../elementwise/mul.md)) manage the mask internally; the mode set manually by the user will be overwritten. pypto_pro.language.set_mask_norm is mainly intended for custom operations or low-level control scenarios.

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
