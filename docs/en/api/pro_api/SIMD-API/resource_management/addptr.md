# pypto_pro.language.addptr

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:16:53.919Z pushedAt=2026-10-28T11:57:29.125Z nl=2 -->

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

Performs an offset operation on a raw pointer (pypto_pro.language.Ptr[dtype]) to obtain a new pointer that points to the same GM with a shifted starting address. The offset is measured in **elements** (not bytes), and the element size is determined by the pointer's dtype.

It is commonly used to split a workspace (GM scratch area) into multiple segments as needed, and to wrap each segment into a loadable/storable Tensor view with [pypto_pro.language.make_tensor](make_tensor.md).

## Prototype

```python
pypto_pro.language.addptr(ptr: Ptr, offset: Union[int, Scalar]) -> Ptr
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| ptr | Input | Input pointer of the Ptr type. The returned new pointer has the same data type as ptr. The element bit width must be no less than 8 bits. Pointer arithmetic is not supported for sub-byte data types; you can first reinterpret them as a byte-addressable type such as DT_UINT8 through make_ptr. |
| offset | Input | Pointer offset, of the int or Scalar type, in elements. The compiler converts the actual byte offset based on the pointer's data type. The resulting address must still be within the original workspace range. |

## Constraints

None

## Return Value

Returns a Ptr with the same data type as ptr and an offset address.

## Examples

### Offset the Workspace Pointer and Create a Tensor View

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def workspace_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    workspace: pl.Ptr[pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    ws_buf_ptr = pl.addptr(workspace, 64 * 128)
    ws_buf = pl.make_tensor(ws_buf_ptr, [64, 128], [128, 1])

    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        t = tile.current()
        pl.load(t, a, [0, 0])
        pl.add(t, t, t)
        pl.store(ws_buf, t, [0, 0])
        pl.load(t, ws_buf, [0, 0])
        pl.store(out, t, [0, 0])
```
