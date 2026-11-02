# Tiling Parameter Definition and Passing

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-27T01:55:39.007Z pushedAt=2026-10-27T07:22:46.345Z nl=2 -->

Tiling parameters are computed on the host side and then passed to the kernel. They are jointly handled by TilingKey and TilingData:

| Type | Effective Stage | Responsibility |
| --- | --- | --- |
| TilingKey | Kernel compilation | Selects which implementation is used this time, such as aligned versus unaligned, different tile templates, or data layouts. The number of candidates is limited, and each value combination generates a dedicated kernel instance. |
| TilingData | Kernel runtime | Carries the specific values required by this implementation, such as shape, stride, loop boundaries, and scaling coefficients. When the values change, the same compiled kernel is reused. |

That is, TilingKey determines which branch to take, and TilingData provides the data required for that branch to run. This document describes the declaration and passing methods of the two, as well as how to use them together. For the concept and splitting levels of tiling, see [Tiling Overview](tiling_overview.md).

## TilingData

TilingData is declared using a Python `dataclass`. The framework generates a device-side structure based on the field annotations and serializes the field values when the Kernel is launched. A Kernel can declare at most one TilingData parameter, and this parameter must be placed at the end of the parameter list.

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AddTiling:
    rows: int
    cols: int
    scale: float
    strides: int[2]
```

### Field Types

The fields of TilingData support the scalar types `int`, `float`, and `bool`, as well as the corresponding fixed-length arrays `int[N]`, `float[N]`, and `bool[N]`. Among them, `int` corresponds to DT_INT64, `float` corresponds to DT_FP32, and `bool` corresponds to DT_BOOL.

Field declarations must meet the following requirements:

- The class must contain at least one field, and each field can only use the types listed above.
- The length `N` of a fixed-length array must be written directly as an integer from 1 to 2048; the length of the list passed in at runtime must match the declaration.
- When using `int[N]`, `float[N]`, or `bool[N]`, you need to enable `from __future__ import annotations`.
- Fields can be assigned `dataclass` default values; array default values use `dataclasses.field(default_factory=...)`.
- The field order determines the field order, offset, and alignment of the device-side structure. The same data must not be interpreted in a different order on the Host side and the device side.

### Passing TilingData on the Host Side

On the host side, construct a TilingData instance and place it at the end of the Kernel launch argument list (the `add_kernel` below is a Kernel that receives `AddTiling`, defined in [Reading TilingData in the Kernel](#reading-tilingdata-in-the-kernel)):

```python
tiling = AddTiling(
    rows=513,
    cols=511,
    scale=1.0,
    strides=[511, 1],
)

block_dim = 8
add_kernel[None, block_dim](x, y, z, tiling)
```

When default values are used, fields that are not explicitly passed adopt the `dataclass` default values:

```python
from dataclasses import dataclass, field


@dataclass
class CopyTiling:
    length: int
    scale: float = 1.0
    offsets: int[4] = field(default_factory=lambda: [0, 0, 0, 0])
```

Fields with default values must be placed after fields without default values.

### Reading TilingData in the Kernel

After declaring the TilingData parameter in the Kernel function signature, you can read the runtime value by field name:

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def add_kernel(
    x: pl.Ptr[pl.DT_FP16],
    y: pl.Ptr[pl.DT_FP16],
    z: pl.Ptr[pl.DT_FP16],
    tiling: AddTiling,
):
    strides = [tiling.strides[0], tiling.strides[1]]
    tensor_x = pl.make_tensor(x, [tiling.rows, tiling.cols], strides)
    tensor_y = pl.make_tensor(y, [tiling.rows, tiling.cols], strides)
    tensor_z = pl.make_tensor(z, [tiling.rows, tiling.cols], strides)

    tile_rows = (tiling.rows + 127) // 128
    tile_cols = (tiling.cols + 127) // 128

    for tile_idx in pl.range(
        pl.get_block_idx(),
        tile_rows * tile_cols,
        pl.get_block_num(),
    ):
        ...
```

TilingData fields are runtime values and can be used for:

- The shape and stride of `pypto_pro.language.make_tensor`.
- The loop bounds of `pypto_pro.language.range`.
- Scalar computations such as the number of tiles, offsets, and effective shapes.
- Runtime `if` conditions and arithmetic expressions.

TilingData fields cannot be used for parameters that require compile-time Python constants, such as the static `shape` of `TileType`. Such finite candidates should be selected through TilingKey.

## TilingKey

TilingKey describes a finite set of compile-time configurations. The same Kernel source code can generate multiple specialized instances for different keys, and one of them is selected at launch time. Key fields are compile-time constants in the Kernel, and the corresponding conditional branches can be determined at the compilation stage.

### Declaring TilingKey

TilingKey is declared using a regular Python class, and class attributes use `TilingKeyField` to define candidate values:

```python
from pypto_pro.runtime.tilingkey import TilingKeyField


class AddKey:
    UseScale = TilingKeyField(bits=1, values=[0, 1])
    BlockM = TilingKeyField(bits=2, values=[64, 128])

    def is_valid(self, key):
        use_scale, block_m = key
        return use_scale == 0 or block_m == 128
```

Fields are collected in the order they are defined in the class. This order determines the order of the parameters in the `is_valid()` parameter tuple and the position of each field in the 64-bit Key. The indices of the candidate values of each field are packed into a 64-bit Key to work with the `uint64_t` TilingKey used in offline binary compilation scenarios.

`TilingKeyField` must satisfy the following constraints:

| Item | Constraint |
| --- | --- |
| `bits` | Must be greater than 0. |
| `values` | Candidate values can only be integers; strings, floating-point numbers, or `bool` are not allowed. The set must be non-empty and contain no duplicate values. |
| Encoding capacity | The number of candidate values cannot exceed `2**bits`. |
| Total bit width | The sum of `bits` across all fields cannot exceed 64. |
| `is_valid` | Optional; used to reject unsupported field combinations. |

Field encoding stores the index of the candidate value in `values`, rather than the candidate value itself. For example:

```python
class BlockKey:
    BlockM = TilingKeyField(bits=2, values=[32, 64, 128])
```

`BlockM=32`, `64`, and `128` are encoded as `0`, `1`, and `2`, respectively. When the Host side launches the Kernel, it still passes the actual values and does not need to compute the candidate indices.

### Binding TilingKey to a Kernel

Bind TilingKey through `@pypto_pro.language.jit(tiling_key=...)`. The field names can be used directly in the Kernel function body without being added to the parameter list:

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True, tiling_key=AddKey)
def add_kernel(
    x: pl.Ptr[pl.DT_FP16],
    y: pl.Ptr[pl.DT_FP16],
    z: pl.Ptr[pl.DT_FP16],
    tiling: AddTiling,
):
    tile_type = pl.TileType(
        shape=[BlockM, 128],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    if UseScale == 1:
        # This branch is determined at compile time
        ...
    else:
        ...
```

TilingKey field names must not duplicate Kernel parameters or module-level ordinary variables. Fields should only describe the limited patterns that affect code generation; do not define a separate Key for arbitrary runtime shapes.

### Selecting a TilingKey and Launching the Kernel

The Key dictionary is placed after `stream` and `block_dim` in the launch parameters:

```python
key = {"UseScale": 1, "BlockM": 128}
add_kernel[None, block_dim, key](x, y, z, tiling)
```

The Key dictionary must meet the following requirements:

- It contains all fields in the Schema and no extra fields.
- Each value belongs to the `values` of the corresponding field.
- The field combination passes the `is_valid()` check.
- The values passed in the dictionary are the actual candidate values, not the encoded indices.

When datatype specialization is also used, the datatype dictionary is placed after the TilingKey:

```python
add_kernel[None, block_dim, key, datatype](x, y, z, tiling)
```

For binary delivery, the TilingKey Schema is used to generate field declarations and valid Key combinations. For details, see [Offline Binary Compilation](../compilation_and_execution/offline_binary_compilation.md).

## Using TilingData with TilingKey

The same Kernel can use TilingKey to select the compile-time implementation, and then use TilingData to receive the runtime parameters of the current invocation:

```python
key = {"UseScale": 1, "BlockM": 128}
tiling = AddTiling(rows=m, cols=n, scale=0.5, strides=[n, 1])

add_kernel[None, block_dim, key](x, y, z, tiling)
```

When selecting a parameter transmission method, you can follow these rules:

| Parameter Characteristics | Transmission Method |
| --- | --- |
| May change on every invocation and does not require code regeneration | TilingData |
| Has a limited number of candidates and affects code branches, Tile templates, or layouts | TilingKey |
| Only determines the number of cores to launch | `block_dim`. For details, see [Meaning and Setting of blockDim](../kernel_function.md#meaning-and-setting-of-blockdim).|

## Usage Restrictions and Recommendations

- Do not put arbitrary shape values into TilingKey; otherwise, candidate combinations will increase rapidly and produce a large number of specialized Kernels.
- Do not put parameters that affect the static Tile type into TilingData, because such parameters are only available at Kernel runtime.
- Changing the field order or type of TilingData changes the device-side structure layout. The Host side and the Kernel must use the same definition.
- The length of an array field must exactly match its declaration. Runtime variable-length data should be split into a length field and a fixed-capacity array, or passed through a Tensor.
- When using TilingKey, datatype specialization, and TilingData together, check the order of the launch bracket parameters and the function call parameters separately.
