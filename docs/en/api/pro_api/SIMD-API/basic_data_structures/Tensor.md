# pypto_pro.language.Tensor

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

Annotates the type of multi-dimensional tensors in GM. It is mainly used to:

- Declare the type of the GM tensor parameters in the kernel function.
- Work with [pypto_pro.language.load](../memory_data_movement/load.md) and [pypto_pro.language.store](../memory_data_movement/store.md) to move data between the GM and L1 Buffer/UB/L0C Buffer.
- Create an alias by assigning a value using the equal sign (=). The alias shares the same GM memory segment with the original tensor, and no data copy is generated. Chained aliases are supported. Once an alias is created, it always points to the same tensor. If the original variable name is rebound to another tensor, the target of the alias will not change.

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

| Parameter| Input/Output| Description|
|---|---|---|
| shape | Input| Sequence of sizes for each dimension (list or tuple).<br>- The formats are as follows:<br>&nbsp;&nbsp;- Fixed dimension: represented by a positive integer, for example, [64, 128]. The dimension to be called must be equal to the integer.<br>&nbsp;&nbsp;- Dynamic dimension: represented by pypto_pro.language.DYNAMIC. The actual dimension is read during calls. The dimension value is not involved in the compilation cache key. Different values reuse the same compiled variant.<br>&nbsp;&nbsp;- Specialized dimension at compile time: represented by pypto_pro.language.STATIC. The actual dimension is read during calls. The dimension value is fixed to the current compiled variant. When the value changes, a new compiled variant is generated.<br>&nbsp;&nbsp;- Trailing...: The remaining dimensions are expanded. Each dimension is processed according to pypto_pro.language.STATIC.<br>&nbsp;&nbsp;- Different strategies can be used together, for example, [64, pl.DYNAMIC, pl.STATIC].<br>- In the kernel, the corresponding dimension can be read through tensor.shape[i]. The return value type is DT_INT64.|
| dtype | Input| Data type of the elements. It is an enumerated value of [pypto_pro.language.DataType](DataType.md).|
| expr | Input| (Optional) IR expression used internally by the framework to construct the runtime tensor object associated with the underlying expression. It cannot be configured together with shape or dtype. You do not need to set it.|
| layout | Input| (Optional) Memory layout, which is an enumerated value of [pypto_pro.language.TensorLayout](TensorLayout.md).<br>- Both ND and NZ are supported. If this parameter is not specified, ND is used.<br>- If this parameter is set to NZ, only the layout is declared, and the ND-to-NZ conversion is not performed.|
| memref | Input| (Optional) Explicit memory reference, which is a pypto_pro.language.MemRef instance. In the three-parameter form, if the third parameter is a MemRef instance, it is parsed as memref. The four-parameter form is used when both layout and memref need to be specified.|
| _annotation_only | Input| Internal parameter of the framework. It is used to identify whether the current tensor is only used for type annotation. You do not need to set it.|
| direction | Input| Used only for type annotation. It is optional and indicates the direction of the tensor parameter in the profiling metadata. The value can be pypto_pro.language.Input or pypto_pro.language.Output. If this parameter is not specified, the default value pypto_pro.language.Input is used. In the square bracket form, it can be placed in any optional position after dtype. In the call form, it can only be passed as the third positional parameter.|

## Constraints

- Only one direction can be specified for each tensor type. The value can only be `pl.Input` or `pl.Output`.
- For details about the movement constraints, see [pypto_pro.language.load](../memory_data_movement/load.md) and [pypto_pro.language.store](../memory_data_movement/store.md).

## Return Value

None

## Examples

### Type Declaration

```python
import pypto_pro.language as pl

# Fixed integer dimension
x: pl.Tensor[[64, 128], pl.DT_FP16]

# Tensor with layout
y: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ]

# High-dimensional NZ: The last two axes 64/128 are M/N, and the first two axes are batch.
y_4d: pl.Tensor[[2, 4, 64, 128], pl.DT_FP16, pl.NZ]

# Grouped scaling factor of E8M0 for matrix A: The logical shape is [M,G]=[64,4], and the physical shape of GM is [M,G/2,2]=[64,2,2].
scale_a: pl.Tensor[[64, 2, 2], pl.DT_FP8E8M0]

# Dynamic dimension declaration (used only for type annotation)
dynamic_tensor: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]
```

### Tensor Parameter Direction Annotation

The tensor type annotation of kernel formal parameters can use two enumerated values `pypto_pro.language.Input` and `pypto_pro.language.Output` to declare the direction of the parameter in the profiling metadata:

```python
import pypto_pro.language as pl

# Bracket form
input_tensor: pl.Tensor[[64, 128], pl.DT_FP16, pl.Input]
output_tensor: pl.Tensor[[64, 128], pl.DT_FP16, pl.Output]

# direction can be combined with layout. It is recommended that you place it at the end of the annotation.
nz_output: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ, pl.Output]

# The third parameter of the call is direction.
call_style_output: pl.Tensor([64, 128], pl.DT_FP16, pl.Output)
```

`pl.Input` indicates the input, and `pl.Output` indicates the output. direction can be omitted. If it is not specified, `pl.Input` is used by default. This annotation is used to set the input and output information of tensors in the profiling result, without changing the read and write semantics of the kernel.

### Tensor Aliases

```python
import pypto_pro.language as pl

# Both the level-1 alias and chained alias point to input_tensor passed for the first time.
original_input_alias = input_tensor
original_input_alias_chain = original_input_alias

# Aliases can be used as tensor operands.
pl.load(input_tile, original_input_alias_chain, [0, 0])

# Rebinding the original variable does not change the direction of the existing alias.
input_tensor = replacement_tensor
pl.load(input_tile, original_input_alias, [0, 0])   # Still reads from the first input_tensor.
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

    # If shape is STATIC, a compiled variant is generated when [256, 256] is called for the first time.
    x = torch.randn([256, 256], device=device, dtype=torch.float16)
    y = torch.randn([256, 256], device=device, dtype=torch.float16)
    z = torch.zeros([256, 256], device=device, dtype=torch.float16)

    add_static[None, 8](x, y, z)
    torch.npu.synchronize()

    torch.testing.assert_close(z, x + y, rtol=1e-3, atol=1e-3)
    print(f"[256, 256] max diff = {(z - (x + y)).abs().max().item()}")
```
