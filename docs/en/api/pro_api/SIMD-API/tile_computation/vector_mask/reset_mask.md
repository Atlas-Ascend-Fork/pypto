# pypto_pro.language.reset_mask

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

Resets the vector mask register to the default state (all ones), enabling all elements to participate in subsequent vector computations.

This is equivalent to pypto_pro.language.set_vec_mask(-1, -1), that is, all 128 bits of the mask are set to 1.

## Prototype

```python
pypto_pro.language.reset_mask() -> None
```

## Parameters

None

## Return Value

None

## Constraints

- After setting the mask via [pypto_pro.language.set_vec_mask](set_vec_mask.md), call this API to restore the default all-ones mask when needed.
- If currently in count mode, pypto_pro.language.reset_mask restores the all-ones mask but does not automatically switch back to norm mode. The semantics of an all-ones mask in count mode differ from those in norm mode. It is recommended to call [pypto_pro.language.set_mask_norm](set_mask_norm.md) first to switch back to norm mode, and then call pypto_pro.language.reset_mask.
- Tile-level operations (such as [add](../elementwise/add.md) and [mul](../elementwise/mul.md)) manage the mask internally; any mask set by the user via pypto_pro.language.set_vec_mask will be overwritten. pypto_pro.language.reset_mask is mainly intended for custom operations or low-level control scenarios.

## Examples

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def reset_mask_kernel(
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
        pl.set_vec_mask(0, 0xFFFFFFFF)
        # Perform custom vector computation that reads the current mask state.
        pl.reset_mask()
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```
