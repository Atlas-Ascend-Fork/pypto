# pypto_pro.language.fill_index

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

Generates a sequence of consecutive integers starting from start and fills the destination Tile: out[j] = start + j.

Typical scenario: Generates incremental indices required for position encoding, or initializes indices used for sorting and gathering.

## Prototype

```python
pypto_pro.language.fill_index(
    out: Tile,
    start: Union[int, float],
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination tile, which stores the generated index sequence. Its shape is [1, N] (the number of rows is 1 and the number of columns is the number of indices), and it must be in the UB.|
| start | Input| Start value. It can be an integer or a runtime integer scalar expression, which is converted to the element type of out during code generation.|

## Constraints

- out supports DT_INT16, DT_UINT16, DT_INT32, and DT_UINT32.
- out must be a single-row Tile (the 0th dimension of the shape is 1).
- out must be in the UB.
- The generated index sequence is [start, start+1, ..., start+N-1], where N is the number of valid columns of out (valid_shape[1]).
- This API must be called within section_vector.

## Return Value

None

## Examples

```python
import pypto_pro.language as pl

START = 0


@pl.jit(auto_mutex=True)
def fill_index_kernel(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
):
    tt = pl.TileType(shape=[1, 128], valid_shape=[-1, -1],
                     dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_out = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        m_dim = out.shape[0]
        n_dim = out.shape[1]
        for i in pl.range(0, m_dim, 1):
            cur_out = tile_out.current()
            pl.set_validshape(cur_out, [1, n_dim])
            pl.fill_index(cur_out, START)
            pl.store(out, cur_out, [i, 0])
```
