# Kernel Function

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-27T02:06:34.977Z pushedAt=2026-10-27T10:42:15.887Z nl=2 -->

A Kernel is a function executed on the AI Core. Its inputs, outputs, and runtime parameters are defined in the function signature, and data movement and computation are organized in the function body.

## Define Kernel

### Using the JIT Decorator

A Kernel function must be decorated with `@pypto_pro.language.jit()`:

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def add_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP16],
    y: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    ...
```

jit parses the function body and triggers compilation at the first startup of the Kernel. For details about the JIT process, compilation signature, and compilation options, see [JIT Compilation](compilation_and_execution/JIT_compilation.md).

### Declaring Kernel Parameters

Kernel parameters must explicitly specify the data type and passing method through type annotations:

| Parameter Type | Purpose |
| --- | --- |
| [pypto_pro.language.Tensor](../../../../api/pro_api/SIMD-API/basic_data_structures/Tensor.md) | Receives multidimensional data in GM; declares shape, dtype, and optional layout in the type. |
| [pypto_pro.language.Ptr](../../../../api/pro_api/SIMD-API/basic_data_structures/Ptr.md) | Receives a raw pointer; typically uses [pypto_pro.language.make_tensor](../../../../api/pro_api/SIMD-API/resource_management/make_tensor.md) to construct a Tensor view. |
| [pypto_pro.language.DT_*](../../../../api/pro_api/SIMD-API/basic_data_structures/DataType.md) | Receives runtime scalars such as integers and floating-point numbers. |
| [TilingData class](tiling/tiling_parameter_definition.md#tilingdata) | Receives structured runtime parameters such as shape, stride, and loop bounds. |

Tensor parameters are suitable for directly using the shape of the caller-side Tensor; Ptr parameters are suitable for having shape and stride provided by TilingData:

```python
from dataclasses import dataclass

import pypto_pro.language as pl


@dataclass
class AddTiling:
    rows: int
    cols: int


@pl.jit(auto_mutex=True)
def dynamic_kernel(
    x: pl.Ptr[pl.DT_FP16],
    out: pl.Ptr[pl.DT_FP16],
    scale: pl.DT_FP32,
    tiling: AddTiling,
):
    tensor_x = pl.make_tensor(x, [tiling.rows, tiling.cols])
    tensor_out = pl.make_tensor(out, [tiling.rows, tiling.cols])
    ...
```

TilingData must be placed at the end of both the Kernel formal parameter list and the startup argument list. For complete fields and transfer rules, see [Tiling Parameter Definition and Passing](tiling/tiling_parameter_definition.md#tilingdata).

A Kernel does not return a Python value. Computation results are written back through the GM region corresponding to the Tensor or Ptr.

### Defining the Execution Domain

Use [pypto_pro.language.section_vector()](../../../../api/pro_api/SIMD-API/controlflow/section_vector.md) and [pypto_pro.language.section_cube()](../../../../api/pro_api/SIMD-API/controlflow/section_cube.md) to define the execution domain where the computation code resides:

| Kernel Composition | Execution Mode |
| --- | --- |
| Contains only the Vector execution domain | Starts a Vector Kernel. |
| Contains only the Cube execution domain | Starts a Cube Kernel. |
| Contains both the Cube and Vector execution domains | Starts a hybrid Kernel. |

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def mixed_kernel(
    x: pl.Ptr[pl.DT_FP16],
    out: pl.Ptr[pl.DT_FP16],
):
    with pl.section_cube():
        # Cube computation code
        ...

    with pl.section_vector():
        # Vector computation code
        ...
```

The execution domain determines the available instructions, on-chip buffers, and the meaning of block_dim. For Vector computation, see [Tile Computation](vector_computation/tile_computation.md) and [Reg Computation](vector_computation/reg_computation.md). For cube computation, see [Cube Computation](cube_computation.md).

### Organizing the Kernel Function Body

A kernel function body is typically organized in the following order:

1. Define the Tile type and bind the on-chip Buffer.
2. Enter the Vector or Cube execution domain.
3. Divide the tasks of the current core based on the logical Block index.
4. Move data from GM into the on-chip Buffer.
5. Perform vector or cube computation.
6. Write the results back to GM.

The following only shows the kernel structure. Tile creation and computation parameters are described in the corresponding sections:

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def add_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP16],
    y: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_x = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    tile_y = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2])

    with pl.section_vector():
        cur_x = tile_x.current()
        cur_y = tile_y.current()
        cur_out = tile_out.current()
        pl.load(cur_x, x, [0, 0])
        pl.load(cur_y, y, [0, 0])
        pl.add(cur_out, cur_x, cur_y)
        pl.store(out, cur_out, [0, 0])
```

For Tile declaration, addresses, and TileGroup operations, see [Tile Creation and Operations](tile_creation_and_operations.md). `auto_mutex=True` automatically inserts mutex synchronization into the Tile objects created through [pypto_pro.language.make_tile_group](../../../../api/pro_api/SIMD-API/resource_management/make_tile_group.md), ensuring the read/write timing of Tiles within the kernel. For scenarios that require explicit synchronization, see [Synchronization APIs](../../../../api/pro_api/SIMD-API/synchronization/index.md).

## Calling a Kernel

A kernel uses square brackets to specify the startup configuration and parentheses to pass function arguments:

| Call Form | Description |
| --- | --- |
| kernel(args...) | Uses the current stream, with block_dim=1. |
| kernel\[block_dim\](args...) | Uses the current stream and specifies the number of logical cores. |
| kernel\[stream, block_dim\](args...) | Specifies the stream and the number of logical cores. |
| kernel\[stream, block_dim, tiling_key\](args...) | Selects the compiled instance corresponding to the TilingKey. |
| kernel\[stream, block_dim, tiling_key, datatype\](args...) | Selects both the TilingKey and datatype specialization instances. |

When using TilingKey or datatype specialization, the corresponding dictionary must be passed through square brackets. When using only datatype specialization, the datatype dictionary is the third item; when using both TilingKey and datatype, they are the third and fourth items, respectively.

```python
import os

import torch
import torch_npu


device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)

x = torch.rand(64, 64, device=device, dtype=torch.float16)
y = torch.rand(64, 64, device=device, dtype=torch.float16)
out = torch.empty_like(x)

add_kernel[None, 1](x, y, out)
torch.npu.synchronize()
```

Kernel startup is asynchronous with respect to the host. Before the host reads the results, checks accuracy, or measures the complete execution time, the corresponding stream must be synchronized.

### Meaning and Setting of stream

stream specifies the NPU execution stream to which the Kernel is dispatched. It is used to maintain the execution order of asynchronous operations and ensure that they are executed on the device in the order in which they are called in the application code. When None is passed, the current Stream is used. When a Stream is explicitly passed, you can wait only for the tasks on that Stream:

```python
import torch
import torch_npu


stream = torch.npu.Stream()
add_kernel[stream, num_cores](x, y, out)
stream.synchronize()
```

Tasks in the same Stream are executed in the order in which they are dispatched. When data dependencies exist between different Streams, the dependencies must be explicitly established through the Stream synchronization mechanism.

### Meaning and Setting of blockDim

block_dim indicates the number of logical cores requested for startup. Its value must be a positive integer and must not exceed the actual available upper limit for the current device and Kernel type. The number of logical Blocks that actually take effect at runtime may be smaller than the requested value. Inside the Kernel, read the actual effective value through [pypto_pro.language.get_block_num()](../../../../api/pro_api/SIMD-API/system_variables/get_block_num.md).

The following examples assume that the stated numbers of cores are all within the currently available range.

| Kernel Type | Meaning of block_dim |
| --- | --- |
| Vector Kernel | Used to set the number of logical Vector (AIV) cores. For example, if an AI processor has 40 Vector cores, setting `block_dim=40` requests the startup of 40 logical Vector cores. |
| Cube Kernel | Used to set the number of logical Cube (AIC) cores. For example, if an AI processor has 20 Cube cores, setting `block_dim=20` requests the startup of 20 logical Cube cores. |
| Mixed Cube and Vector Kernel | Started as a combination of AIV and AIC, used to set the number of logical execution groups. For example, if an AI processor has 40 Vector cores and 20 Cube cores, and one execution group contains 2 Vector cores and 1 Cube core, setting `block_dim=20` requests the startup of 20 execution groups. If all 20 groups take effect, they correspond to 40 Vector cores and 20 Cube cores. The number of execution groups that actually take effect is obtained by `get_block_num()`. The number of logical cores on the Vector side must also be multiplied by [pypto_pro.language.get_subblock_num()](../../../../api/pro_api/SIMD-API/system_variables/get_subblock_num.md). |

## Using TilingKey and datatype

TilingKey is used to select a limited set of compile-time modes, and datatype is used to generate specialized instances based on the input or output data type:

```python
import pypto_pro.language as pl


key = {"UseScale": 1, "BlockM": 128}
datatype = {"x": pl.DT_FP16, "out": pl.DT_FP16}

kernel[None, block_dim, key, datatype](x, out, tiling)
```

For the declaration, encoding, and startup rules of TilingKey, see [Tiling Parameter Definition and Passing](tiling/tiling_parameter_definition.md#tilingkey). The datatype field is declared by `@pypto_pro.language.jit(datatype=...)`. For the specific compilation behavior, see [JIT Compilation](compilation_and_execution/JIT_compilation.md#compilation-signature-and-reuse).
