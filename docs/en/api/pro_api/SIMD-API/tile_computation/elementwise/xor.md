# pypto_pro.language.xor

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:44:30.023Z pushedAt=2026-10-28T11:57:29.258Z nl=2 -->

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
    rhs: Tile,
    tmp: Tile,
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| out | Output | Destination operand of the Tile type, which stores the result of the element-wise bitwise XOR. Its data type is the same as that of lhs and rhs, and supports DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, or DT_UINT32. Its shape is the same as that of lhs and rhs. |
| lhs | Input | Left operand of the Tile type. Its data type is the same as that of out. |
| rhs | Input | Right operand of the Tile type. It uses the row-major layout, its valid_shape is the same as that of out, and its shape is the same as that of out and lhs. Its data type is the same as that of out. |
| tmp | Input | Compatibility parameter of the Tile type. |

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

The measured results are as follows.

<!-- pypto-doc-output:xor:start -->
```bash
Input data a: [[2 3 4 5 6 7 8 9 ...], [66 67 68 69 70 71 72 73 ...], [130 131 132 133 134 135 136 137 ...], [194 195 196 197 198 199 200 201 ...], ...]
Input data b: [[1 2 3 4 5 6 7 8 ...], [1 2 3 4 5 6 7 8 ...], [1 2 3 4 5 6 7 8 ...], [1 2 3 4 5 6 7 8 ...], ...]
Output data out: [[3 1 7 1 3 1 15 1 ...], [67 65 71 65 67 65 79 65 ...], [131 129 135 129 131 129 143 129 ...], [195 193 199 193 195 193 207 193 ...], ...]
```
<!-- pypto-doc-output:xor:end -->
