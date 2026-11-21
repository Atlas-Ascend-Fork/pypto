# pypto_pro.language.xor

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

Performs a bitwise XOR on the elements at corresponding positions of the source operands lhs and rhs, and writes the result to the destination operand out.

## Prototype

```python
pypto_pro.language.xor(
    out: Tile,
    lhs: Tile,
    rhs: Union[Tile, int],
    tmp: Tile,
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand of type Tile, which stores the result of element-wise bitwise XOR. The data type is consistent with that of lhs; when rhs is a Tile, it must also be consistent with that of rhs. DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, and DT_UINT32 are supported. The shape is the same as that of lhs; when rhs is a Tile, it must also be the same as that of rhs.|
| lhs | Input| Left operand, which is of type Tile. The data types are the same as those of out.|
| rhs | Input| Right operand, which is of type Tile or an integer scalar. When it is a Tile, it uses a row-major layout; its valid_shape must be consistent with that of out, its shape must be consistent with those of out and lhs, and its data type must be consistent with that of out.|
| tmp | Input| Compatibility parameter, which is of type Tile.|

## Constraints

None

## Return Value

None

## Examples

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def xor_kernel(
    a: pl.Tensor[[64, 64], pl.DT_INT32],
    b: pl.Tensor[[64, 64], pl.DT_INT32],
    out: pl.Tensor[[64, 64], pl.DT_INT32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    tile_out = pl.make_tile_group(type=tt, addrs=0xC000, mutex_ids=[3])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.xor(cur_out, cur_a, cur_b, cur_tmp)
        pl.store(out, cur_out, [0, 0])
```

The following is an example of the test result:

<!-- pypto-doc-output:xor:start -->
```bash
Input a: [[2 3 4 5 6 7 8 9 ...], [66 67 68 69 70 71 72 73 ...], [130 131 132 133 134 135 136 137 ...], [194 195 196 197 198 199 200 201 ...], ...]
Input b: [[1 2 3 4 5 6 7 8 ...], [1 2 3 4 5 6 7 8 ...], [1 2 3 4 5 6 7 8 ...], [1 2 3 4 5 6 7 8 ...], ...]
Output data out: [[3 1 7 1 3 1 15 1 ...], [67 65 71 65 67 65 79 65 ...], [131 129 135 129 131 129 143 129 ...], [195 193 199 193 195 193 207 193 ...], ...]
```
<!-- pypto-doc-output:xor:end -->
