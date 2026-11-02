# pypto_pro.language.Tensor

<!-- md-trans-meta sourceCommit=d15a7740ea679c63c4ade0c9eaf473865b3b634e translatedAt=2026-10-26T12:49:29.550Z pushedAt=2026-10-28T11:57:28.650Z nl=2 -->

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

Type annotation for multi-dimensional tensors in GM, mainly used for:

- Type declaration of GM tensor parameters in kernel functions.
- Data movement between GM and L1 Buffer/UB/L0C Buffer in conjunction with [pypto_pro.language.load](../memory_data_movement/load.md)/[pypto_pro.language.store](../memory_data_movement/store.md).
- Create an alias through assignment with the equal sign, and share the same GM memory segment with the original tensor without data copy. Chained aliases are supported. Once created, an alias points to a fixed target. Rebinding the original variable name to another tensor later does not change the target of an existing alias.

## Prototype

```python
pypto_pro.language.Tensor.__init__(
    self,
    shape: Sequence[int | _ShapePolicy | EllipsisType] | None = None,
    dtype: DataType | None = None,
    expr: Expr | None = None,
    layout: TensorLayout | None = None,
    memref: MemRef | None = None,
    _annotation_only: bool = False,
    direction: str | None = None,
)
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| shape | Input | List of sizes of each dimension.<br>- The forms are as follows:<br>&nbsp;&nbsp;- Fixed dimension: represented by a positive integer, for example, [64, 128]. The corresponding dimension must equal this integer when called.<br>&nbsp;&nbsp;- Dynamic dimension: represented by pypto_pro.language.DYNAMIC. The actual dimension is read when called. The dimension value does not participate in the compilation cache key, and different values reuse the same compiled variant.<br>&nbsp;&nbsp;- Compile-time specialized dimension: represented by pypto_pro.language.STATIC. The actual dimension is read when called, and the dimension value is fixed into the current compiled variant. A new compiled variant is generated when the value changes.<br>&nbsp;&nbsp;- Trailing ...: expands the remaining dimensions, each processed as pypto_pro.language.STATIC.<br>&nbsp;&nbsp;- Different strategies can be mixed, for example, [64, pl.DYNAMIC, pl.STATIC].<br>- In the kernel, the corresponding dimension can be read through tensor.shape[i], and the return value type is DT_INT64. |
| dtype | Input | Element data type, an enumeration value of [pypto_pro.language.DataType](DataType.md). |
| expr | Input | Optional. IR expression used internally by the framework to construct a runtime Tensor object associated with the underlying Expr. It cannot be configured together with shape or dtype, and you do not need to set it. |
| layout | Input | Optional. Memory layout, an enumeration value of [pypto_pro.language.TensorLayout](TensorLayout.md).<br>- Supports ND and NZ. ND is used when not specified.<br>- When configured as NZ, only the layout is declared, and no ND→NZ conversion is performed. |
| memref | Input | Optional. Explicit memory reference, a pypto_pro.language.MemRef instance. In the three-parameter form, when the third item is a MemRef instance, it is parsed as memref. Use the four-parameter form when both layout and memref need to be specified. |
| _annotation_only | Input | Framework internal parameter used to identify whether the current Tensor is used only as a type annotation. You do not need to set it. |
| direction | Input | Used only for type annotation. Optional. Direction of the Tensor parameter in Profiling metadata, with a value of pypto_pro.language.Input or pypto_pro.language.Output. Defaults to pypto_pro.language.Input when omitted. In the bracket form, it can be placed at any optional position after dtype. In the call form, it is supported only as the third positional parameter. |

## Constraints

- Each Tensor type annotation can declare at most one direction, whose value can only be `pl.Input` or `pl.Output`.
- For data movement constraints, see [pypto_pro.language.load](../memory_data_movement/load.md) and [pypto_pro.language.store](../memory_data_movement/store.md).

## Return Value

None

## Examples

### Type Declaration

```python
import pypto_pro.language as pl

# Fixed integer dimension.
x: pl.Tensor[[64, 128], pl.DT_FP16]

# Tensor with layout.
y: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ]

# High-dimensional NZ: the last two axes 64/128 are M/N, and the first two axes are batch.
y_4d: pl.Tensor[[2, 4, 64, 128], pl.DT_FP16, pl.NZ]

# Grouped scaling factor of E8M0 for matrix A: logical shape [M,G]=[64,4], GM physical shape [M,G/2,2]=[64,2,2].
scale_a: pl.Tensor[[64, 2, 2], pl.DT_FP8E8M0]

# Dynamic dimension declaration (for type annotation only).
dynamic_tensor: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]
```

### Tensor Parameter Direction Annotation

The Tensor type annotation of kernel formal parameters supports using the two enum values `pypto_pro.language.Input` and `pypto_pro.language.Output` to declare the direction of the parameter in the Profiling metadata:

```python
import pypto_pro.language as pl

# Bracket form.
input_tensor: pl.Tensor[[64, 128], pl.DT_FP16, pl.Input]
output_tensor: pl.Tensor[[64, 128], pl.DT_FP16, pl.Output]

# direction can be combined with layout. It is recommended to place it at the end of the annotation.
nz_output: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ, pl.Output]

# The third parameter of the call form is direction.
call_style_output: pl.Tensor([64, 128], pl.DT_FP16, pl.Output)
```

`pl.Input` indicates input, and `pl.Output` indicates output. The direction can be omitted, in which case `pl.Input` is used by default. This annotation is used to set the tensor input and output information in the Profiling result and does not change the read/write semantics of the kernel.

### Tensor Alias

```python
import pypto_pro.language as pl

# Both the first-level alias and the chained alias point to the input_tensor passed in the first time.
original_input_alias = input_tensor
original_input_alias_chain = original_input_alias

# The alias can be used as a Tensor operand.
pl.load(input_tile, original_input_alias_chain, [0, 0])

# Rebinding the original variable does not change the target of the existing alias.
input_tensor = replacement_tensor
pl.load(input_tile, original_input_alias, [0, 0])   # Still reads from the input_tensor passed in the first time.
pl.load(replacement_tile, input_tensor, [0, 0])     # Reads from replacement_tensor.
```

### DYNAMIC Dynamic Dimension

```python
import os

import pypto_pro.language as pl
import torch


@pl.jit(auto_mutex=True)
def dynamic_tensor_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b_group = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out_group = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        tile_a = tile_a_group.current()
        tile_b = tile_b_group.current()
        tile_out = tile_out_group.current()
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.add(tile_out, tile_a, tile_b)
        pl.store(out, tile_out, [0, 0])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    a = torch.randn([64, 64], device=device, dtype=torch.float32)
    b = torch.randn([64, 64], device=device, dtype=torch.float32)
    out = torch.zeros([64, 64], device=device, dtype=torch.float32)

    dynamic_tensor_kernel(a, b, out)
    torch.npu.synchronize()

    ref = a + b
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    print(f"max diff = {(out - ref).abs().max().item()}")
```

### STATIC Compile-Time Specialized Dimension

```python
import os

import pypto_pro.language as pl
import torch

TILE_M = 128
TILE_N = 128


@pl.jit(auto_mutex=True)
def add_static(
    x: pl.Tensor[[pl.STATIC, pl.STATIC], pl.DT_FP16],
    y: pl.Tensor[[pl.STATIC, pl.STATIC], pl.DT_FP16],
    z: pl.Tensor[[pl.STATIC, pl.STATIC], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[4, 5])

    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        m_tile_num = x.shape[0] // TILE_M
        n_tile_num = x.shape[1] // TILE_N

        for i in pl.range(core_id, m_tile_num, num_cores):
            for j in pl.range(0, n_tile_num, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    # shape is STATIC. The first call with [256, 256] generates a compiled variant.
    x = torch.randn([256, 256], device=device, dtype=torch.float16)
    y = torch.randn([256, 256], device=device, dtype=torch.float16)
    z = torch.zeros([256, 256], device=device, dtype=torch.float16)

    add_static[None, 8](x, y, z)
    torch.npu.synchronize()

    torch.testing.assert_close(z, x + y, rtol=1e-3, atol=1e-3)
    print(f"[256, 256] max diff = {(z - (x + y)).abs().max().item()}")
```
