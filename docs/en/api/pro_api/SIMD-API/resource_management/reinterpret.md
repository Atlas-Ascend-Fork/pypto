# pypto_pro.language.reinterpret

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T14:21:57.994Z pushedAt=2026-10-28T11:57:29.147Z nl=2 -->

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

Redeclares an allocated Tile with a new dtype/shape/layout and returns a new alias of the Tile that points to the same address. The original Tile is not affected.

## Prototype

```python
pypto_pro.language.reinterpret(
    tile: Union[Tile, TileGroup],
    /,
    *,
    dtype: Optional[DType] = None,
    shape: Optional[List[int]] = None,
    layout: Optional[TensorLayout] = None,
) -> Union[Tile, TileGroup]
```

## Parameters

| Parameter | Input/Output | Description                                                                                                                                                    |
| --------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| tile      | Input        | Object to be redeclared, of the Tile or TileGroup type. The source Tile must be bound to a Buffer address that is determinable at compile time. The new alias reuses the same address and size as the source object, and no data movement or type conversion is performed. |
| dtype     | Input        | Target data type, of the [pypto_pro.language.DataType](../basic_data_structures/DataType.md) type. Optional. When omitted, the original dtype is inherited. When dtype is specified, shape must also be specified, and the Tile base address must be aligned to the element byte size of the new dtype. |
| shape     | Input        | Target shape, of the List[int] type. Optional. Must be a non-empty compile-time integer list. When omitted, the original shape is inherited. The storage size determined by the new shape and dtype must not exceed the Buffer size of the original Tile. The runtime valid shape should be set using pypto_pro.language.set_validshape. |
| layout    | Input        | Target data layout, of the [pypto_pro.language.TensorLayout](../basic_data_structures/TensorLayout.md) type. Optional. When omitted, the original layout is inherited. The caller must ensure that the physical data in the Buffer actually conforms to the new layout. |

## Constraints

- At least one of the three optional parameters dtype, shape, and layout must be specified.
- The pypto_pro.language.set_validshape of the original Tile is not inherited.
- After a TileGroup is redeclared, it uses the same buffer management as the original TileGroup. The next() behavior of either one affects the buffer rotation. To rotate the buffer independently, use group[i] [pypto_pro.language.make_tile_group](make_tile_group.md).

## Return Value

Returns the alias of the new Tile or TileGroup redeclared at the original address and size.

## Examples

The following example combines three typical uses of reinterpret into a single runnable script:

1. **dtype redeclaration**: The same block of data is read and written as a new type, without numeric conversion.
2. **dtype bit-width change**: After the data type changes, the number of elements changes accordingly, so shape must be redeclared together with dtype.
3. **layout transformation**: After NZ is redeclared as ZN, it can be used directly as the right matrix of matmul to achieve transposition, saving one explicit transpose movement.

```python
import os
import pypto_pro.language as pl
import torch

TILE = 64


# ============ Use case 1: dtype redeclaration (same bit width, FP32 -> INT32) ============
# Redeclare the FP32 Tile as INT32. The address and size remain unchanged, and the data stays bit-identical.
@pl.jit(auto_mutex=True)
def dtype_reinterpret_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_INT32],
):
    tt = pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_in = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])

    with pl.section_vector():
        t = tile_in.current()
        pl.load(t, x, [0, 0])
        t2 = pl.reinterpret(t, shape=[TILE, TILE], dtype=pl.DT_INT32)
        pl.store(out, t2, [0, 0])


# ============ Use case 2: dtype bit-width change (FP32 -> FP16, element count doubles) ============
# FP32 is 32-bit and FP16 is 16-bit. After redeclaration, the element count doubles, so shape must be redeclared together with dtype.
@pl.jit(auto_mutex=True)
def width_change_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[128, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_in = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])

    with pl.section_vector():
        t = tile_in.current()
        pl.load(t, x, [0, 0])
        widened = pl.reinterpret(t, shape=[TILE * 2, TILE], dtype=pl.DT_FP16)
        pl.store(out, widened, [0, 0])


# ============ Use case 3: layout transformation (NZ -> ZN, matmul right matrix transpose) ============
# After the same block of data is declared as ZN, it can be used directly as the right matrix of matmul, which is equivalent to using its transpose.
@pl.jit(auto_mutex=True)
def layout_reinterpret_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    m0, k0 = TILE, TILE
    mat_type = pl.TileType(shape=[m0, k0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
    t1 = pl.make_tile_group(type=mat_type, addrs=0x0000, mutex_ids=[0])
    t2 = pl.make_tile_group(type=mat_type, addrs=0x10000, mutex_ids=[1])

    left_type = pl.TileType(shape=[m0, k0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    right_type = pl.TileType(shape=[m0, k0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    acc_type = pl.TileType(
        shape=[m0, m0], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
    )
    l0a = pl.make_tile_group(type=left_type, addrs=0x0000, mutex_ids=[2])
    l0b = pl.make_tile_group(type=right_type, addrs=0x0000, mutex_ids=[3])
    acc = pl.make_tile_group(type=acc_type, addrs=0x0000, mutex_ids=[4])

    with pl.section_cube():
        pl.load_tile(t1.current(), a, [0, 0])
        pl.load_tile(t2.current(), a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        t2r = pl.reinterpret(t2.current(), shape=[k0, m0], layout=pl.TensorLayout.ZN)
        pl.move(l0a.current(), t1.current())
        pl.move(l0b.current(), t2r)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc.current(), l0a.current(), l0b.current())
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        pl.store(out, acc.current(), [0, 0])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    # Usage 1: Reinterpret with the same bit-width dtype.
    x1 = torch.randn([64, 64], device=device, dtype=torch.float32)
    o1 = torch.zeros([64, 64], device=device, dtype=torch.int32)
    dtype_reinterpret_kernel[None, 1](x1, o1)
    torch.npu.synchronize()
    torch.testing.assert_close(o1.cpu(), x1.cpu().view(torch.int32), rtol=0, atol=0)
    print("Usage 1 dtype reinterpret PASSED")

    # Usage 2: Bit-width change.
    x2 = torch.randn([64, 64], device=device, dtype=torch.float32)
    o2 = torch.zeros([128, 64], device=device, dtype=torch.float16)
    width_change_kernel[None, 1](x2, o2)
    torch.npu.synchronize()
    torch.testing.assert_close(o2.cpu().view(torch.int32).flatten(), x2.cpu().view(torch.int32).flatten(), rtol=0, atol=0)
    print("Usage 2 dtype bit-width change PASSED")

    # Usage 3: Layout change.
    a = torch.randint(-8, 9, [TILE, TILE], device=device, dtype=torch.float16)
    o3 = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    layout_reinterpret_kernel[None, 1](a, o3)
    torch.npu.synchronize()
    got = o3.cpu().float()
    golden = a.cpu().float() @ a.cpu().float().T
    torch.testing.assert_close(got, golden, rtol=1e-2, atol=1e-1)
    print("Usage 3 layout change PASSED")
```
