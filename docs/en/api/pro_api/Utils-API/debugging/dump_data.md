# pypto_pro.language.dump_data

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:14:51.755Z pushedAt=2026-10-28T11:57:29.498Z nl=2 -->

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

Debug printing API used to print the content of a GM Tensor or Tile, supporting full printing and window printing.

- When the input is a Tensor (GM global memory tensor), prints the Tensor data on GM.
- When the input is a Tile (allocated through make_tile/make_tile_group), prints the Tile data.

A Tile is created through [pypto_pro.language.make_tile](../../SIMD-API/resource_management/make_tile.md) or [pypto_pro.language.make_tile_group](../../SIMD-API/resource_management/make_tile_group.md). Tiles in different memory spaces have different printing capabilities:

- UB Tile: Can be printed directly without a workspace.
- Tile in L0C Buffer: Must be printed through the workspace parameter (first write the Tile data back to GM, then print).
- Tiles in L1 Buffer, L0A Buffer, and L0B Buffer: cannot be printed directly.

The print result is output directly to the terminal.

## Prototype

```python
pypto_pro.language.dump_data(
    data: Union[Tensor, Tile],
    offsets: Optional[List[int]] = None,
    shapes: Optional[List[int]] = None,
    *,
    workspace: Optional[Tensor] = None,
    loc: bool = False,
    flag: Optional[str] = None,
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| data | Input | Data to be printed. It must be a Tensor (TensorType) or a Tile (TileType); other types raise TypeError. Tensor does not currently support the pypto_pro.language.DT_FP4, pypto_pro.language.DT_FP4E2M1, and pypto_pro.language.DT_FP4E1M2 types. |
| offsets | Input | Optional. Window start offsets (per dimension). The value is a sequence of integer constants or runtime integer scalar expressions, whose length must equal the number of dimensions of the data. It must be provided together with shapes or be None together with shapes. Tile windows currently support only two-dimensional tiles. When both offsets and shapes are None, the full amount of data is printed. |
| shapes | Input | Optional. Window sizes (per dimension). The value is a sequence of integer constants or runtime integer scalar expressions, whose length must equal the number of dimensions of the data. It must be provided together with offsets or be None together with offsets. Compile-time constants must be greater than 0. Tensor window mode also requires the innermost-dimension stride to be the compile-time constant 1. An NZ Tensor window must preserve complete fractals: the M shape is aligned to 16, and the N shape and offset are aligned to C0. Leading dimensions are printed batch by batch. Tile windows require two dimensions and require the physical Tile shape to be a compile-time constant. |
| workspace | Input | Optional. A temporary Tensor in GM, used only for relay printing of tiles in L0C Buffer. It is valid only when data is a Tile in L0C Buffer. Passing a Tensor raises ValueError, and tiles in other memory spaces raise an error. It must be a GM TensorType with the same dtype as the Tile, with a capacity sufficient to hold the complete physical Tile. |
| loc | Input | Optional. Whether to print the source file/line number before the output. The value is True or False (default). |
| flag | Input | Optional. A string label. It must be a compile-time constant and cannot be a runtime variable. When provided, a line `=== [flag] <flag> ===` is printed separately before the dump output to distinguish the output of multiple dump points. |

## Constraints

### Tile Scenarios in the L0C Buffer

Tiles in the L0C Buffer cannot be printed directly like UB Tiles. dump_data first writes the Tile data back to the workspace tensor in GM and then prints that tensor, subject to the following constraints:

- The workspace must be of TensorType. It can be a pypto_pro.language.Tensor in the kernel function parameters, or it can be constructed from pypto_pro.language.Ptr using pypto_pro.language.make_tensor. It cannot be a Tile.
- The dtype of the workspace must be the same as the dtype of the Tile to be dumped.
- Regardless of whether a full-amount dump or a window dump is performed, the complete Tile in the L0C Buffer is first written to the workspace. Therefore, its capacity must cover at least the complete physical Tile.
- The Tile in the L0C Buffer must be two-dimensional, and its physical shape must be a compile-time constant.

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

Full dump output example:

```text
=== [dump_tensor] dtype: int32, Layout: ND, shape=[1,1,1,1,16] ===
  Batch [0, 0, 0]:
0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150
```

Dynamic offset example:

```python
# Loop variable as offset (dynamic).
for i in pl.range(0, 16, 4):
    pl.dump_data(out, offsets=[i], shapes=[4])

# get_block_idx() as offset (dynamic).
vidx = pl.get_block_idx()
pl.dump_data(out, offsets=[vidx * 4], shapes=[4])
```

Labeled output (to distinguish multiple dump points):

```python
pl.dump_data(out, flag="checkpoint_A")
```

Label output example:

```text
=== [flag] checkpoint_A ===
=== [dump_tensor] dtype: int32, Layout: ND, shape=[1,1,1,1,16] ===
  Batch [0, 0, 0]:
0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150
```

### UB Tile Input

UB Tile can be dumped directly without a workspace. Tiles in L1 Buffer, L0A Buffer, and L0B Buffer cannot be dumped directly.

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dump_data_tile_full_kernel(
    a: pl.Tensor[[8, 8], pl.DT_INT32]
    b: pl.Tensor[[8, 8], pl.DT_INT32]
    out: pl.Tensor[[8, 8], pl.DT_INT32]
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

Full dump output example:

```text
=== [dump_tile] dtype: int32, shape=[8,8], valid=[8,8], Layout: ND ===
0 2 4 6 8 10 12 14
16 18 20 22 24 26 28 30
32 34 36 38 40 42 44 46
48 50 52 54 56 58 60 62
64 66 68 70 72 74 76 78
80 82 84 86 88 90 92 94
96 98 100 102 104 106 108 110
112 114 116 118 120 122 124 126
```

### Tile Input in L0C Buffer (Requires workspace)

Tiles in the L0C Buffer must be transferred through a temporary tensor in GM provided by the workspace parameter.

Full dump:

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dump_data_tile_acc_fp16_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
    workspace: pl.Tensor[[64, 64], pl.DT_FP32],
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

        # Print the entire Acc Tile (requires workspace for transfer).
        pl.dump_data(ac, workspace=workspace)

        # Print a window (with offsets/shapes and workspace).
        pl.dump_data(ac, offsets=[16, 16], shapes=[8, 8], workspace=workspace)

        pl.store(out, ac, [0, 0])
```

Window mode output example (offsets=[16, 16], shapes=[8, 8]):

```text
=== [TPRINT Acc Tile Window] Data Type: float32, Layout: NZ, TileType: Acc ===
  Source Shape: [64, 64], Window Offsets: [16, 16], Requested Shape: [8, 8], Valid Shape: [8, 8]
   2.435661  21.525450  -2.534927  -3.354072  -1.637453  -7.538389  -5.316622   7.327333
   3.319994   4.183826   9.192725 -15.309023   5.075872  15.763545  -1.755892  -7.553324
  -6.676492  -1.058733  -2.251584  -8.538083  -0.172604   9.005786  -1.326701   7.341187
   5.795816 -12.892869   3.342661   3.139680  10.270340  -0.026452  -2.230551   3.213134
   3.064780  -5.402464  -0.289040  -4.588926  -0.931392 -12.228477 -20.040319  10.303446
  -5.076264   0.564521  11.335535  -0.019537  -1.963741   4.344845  -0.789701   7.402071
 -13.048984  -7.837986 -16.793615   5.720566  -6.111812 -27.283802   1.088718  -7.852593
 -10.569035   7.459199   5.887267   7.939989   1.122919   4.743242 -10.458792  -0.729014
```

Use window dump in the tiled loop:

```python
for i in pl.range(0, 256, 64):
    for j in pl.range(0, 256, 64):
        # ... matmul ...
        pl.dump_data(ac, offsets=[8, 8], shapes=[32, 32], workspace=workspace)
        pl.store(out, ac, [i, j])
```
