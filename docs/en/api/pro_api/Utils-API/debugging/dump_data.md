# pypto_pro.language.dump_data

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

Prints the content of a GM tensor or tile. Full printing and window printing are supported.

- If the input is a tensor (GM tensor), the tensor data on the GM is printed.
- If the input is a tile (allocated by make_tile/make_tile_group), the tile data is printed.

Tiles are created through [pypto_pro.language.make_tile](../../SIMD-API/resource_management/make_tile.md) or [pypto_pro.language.make_tile_group](../../SIMD-API/resource_management/make_tile_group.md).

The printing result is directly output to the terminal. Only the data within the requested range is output in full and window modes, and the physical storage order of the corresponding layout is retained.

## Prototype

```python
pypto_pro.language.dump_data(
    data: Union[Tensor, Tile],
    offsets: Optional[Sequence[int]] = None,
    shapes: Optional[Sequence[int]] = None,
    *,
    workspace: Optional[Tensor] = None,
    loc: bool = False,
    flag: Optional[str] = None,
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| data | Input| Tensor or tile to be printed. Tensors support the ND and NZ layouts, but do not support pypto_pro.language.DT_INT4, pypto_pro.language.DT_UINT4, pypto_pro.language.DT_FP4, and pypto_pro.language.DT_HF4. Tiles do not support the L0A Buffer, L0B Buffer, L0A_MX Buffer, and L0B_MX Buffer.|
| offsets | Input| (Optional) Window start offset (per dimension). The value is a sequence (list or tuple) consisting of integer constants or runtime integer scalar expressions. The length must be equal to the number of dimensions of the data. This parameter must be provided together with shapes. The N-direction offset of the NZ Tensor window must be aligned based on C0 corresponding to dtype.|
| shapes | Input| (Optional) Window size (per dimension). The value is a sequence (list or tuple) consisting of integer constants or runtime integer scalar expressions. The length must be equal to the number of dimensions of the data. This parameter must be provided together with offsets.|
| workspace | Input| Reserved parameter. You do not need to pass it.|
| loc | Input| (Optional) Whether to print the source file/line number before the output. The value can be True or False (default).|
| flag | Input| (Optional) String label, which must be a compile-time constant and cannot be a runtime variable. If this parameter is provided, a line of [flag] label content is printed before the dump output to distinguish the output of multiple dump points.|

## Constraints

By default, the runtime allocates 32 KB of printing FIFO to each AI Cube/AI Vector core, and dump_data and printf on the device share this space. You can set the capacity of each core to a value ranging from 1 KB to 64 MB (an integer multiple of 8 bytes) before using the NPU for the first time. The configuration method for the JIT is the same as that for binary kernels.

```python
import pypto

pypto.set_debug_options(simd_printf_fifo_size_per_core=256 * 1024)

import pypto_pro.language as pl
import torch
import torch_npu
```

This configuration does not change the kernel parameters or binary content.

## Return Value

None

## Examples

### Tensor Input

```python
import pypto_pro.language as pl


@pl.jit()
def dump_data_tensor_full_kernel(
    out: pl.Tensor[[16], pl.DT_INT32],
):
    with pl.section_vector():
        for i in pl.range(0, 16):
            out[i] = i * 10
        pl.dump_data(out)
```

Example of full dump output:

```text
[AIV Block 0/1] [dump_tensor] Data Type: int32, Layout: ND, Source Shape: [16], Offsets: [0], Shape: [16]
[AIV Block 0] DumpTensor: desc=<dump_id>, addr=0x<address>, data_type=int32, position=GM, dump_size=16
[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
```

Dynamic offset example:

```python
# Loop variable as the offset (dynamic)
for i in pl.range(0, 16, 4):
    pl.dump_data(out, offsets=[i], shapes=[4])

# get_block_idx() as the offset (dynamic)
vidx = pl.get_block_idx()
pl.dump_data(out, offsets=[vidx * 4], shapes=[4])
```

Output with labels (used to distinguish multiple dump points):

```python
pl.dump_data(out, flag="checkpoint_A")
```

Label output example:

```text
[AIV Block 0/1] [flag] checkpoint_A
[AIV Block 0/1] [dump_tensor] Data Type: int32, Layout: ND, Source Shape: [16], Offsets: [0], Shape: [16]
[AIV Block 0] DumpTensor: desc=<dump_id>, addr=0x<address>, data_type=int32, position=GM, dump_size=16
[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
```

### UB Tile Input

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dump_data_tile_full_kernel(
    a: pl.Tensor[[8, 8], pl.DT_INT32],
    b: pl.Tensor[[8, 8], pl.DT_INT32],
    out: pl.Tensor[[8, 8], pl.DT_INT32],
):
    tt = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tb_group = pl.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[1])
    tc_group = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[2])
    with pl.section_vector():
        ta = ta_group.current()
        tb = tb_group.current()
        tc = tc_group.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.add(tc, ta, tb)
        pl.dump_data(tc)
        pl.store(out, tc, [0, 0])
```

Example of full dump output:

```text
[AIV Block 0/1] [dump_tile] Data Type: int32, Layout: ND, Memory Space: UB, Source Shape: [8, 8], Offsets: [0, 0], Shape: [8, 8], Valid Shape: [8, 8]
[AIV Block 0] DumpTensor: desc=<dump_id>, addr=0x<address>, data_type=int32, position=UB, dump_size=64
[[0,2,4,6,8,10,12,14],
[16,18,20,22,24,26,28,30],
...,
[112,114,116,118,120,122,124,126]]
```

### Tile Input on the Cube Side

The following example covers the printing of tiles in the L1 and L0C buffers.

Full dump:

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dump_data_cube_tiles_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[3])
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[4])

    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)

        pl.dump_data(cur_a, flag="l1")
        pl.dump_data(ac, flag="l0c")

        # Window printing is also applicable to L1 and L0C tiles.
        pl.dump_data(ac, offsets=[16, 16], shapes=[8, 8])

        pl.store(out, ac, [0, 0])
```

Example of output in window mode (offsets=[16, 16], shapes=[8, 8]):

```text
[AIC Block 0/1] [dump_tile] Data Type: float32, Layout: NZ, Memory Space: L0C Buffer, Source Shape: [64, 64], Offsets: [16, 16], Shape: [8, 8], Valid Shape: [8, 8]
[AIC Block 0] DumpTensor: desc=<dump_id>, addr=0x<address>, data_type=float32, position=L0C, dump_size=64
[[<value_0>,<value_1>,...,<value_7>],
...,
[<value_56>,<value_57>,...,<value_63>]]
```

Dump in window mode inside a tiled loop:

```python
for i in pl.range(0, 256, 64):
    for j in pl.range(0, 256, 64):
        # ... matmul ...
        pl.dump_data(ac, offsets=[8, 8], shapes=[32, 32])
        pl.store(out, ac, [i, j])
```
