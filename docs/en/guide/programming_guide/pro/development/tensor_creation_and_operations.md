# Tensor Creation and Operation

<!-- md-trans-meta sourceCommit=2c2882d5239bbe8c5ef0f1357cad2493795d80dd translatedAt=2026-10-27T02:07:51.626Z pushedAt=2026-10-28T08:40:33.718Z nl=2 -->

A Tensor describes multidimensional data in GM and serves as the input or output of a Kernel, or as a data view constructed from a raw pointer. A Tensor records information such as the data type, shape, stride, and layout, but it is not responsible for allocating GM memory. The on-chip buffer used for Kernel computation is represented by a Tile. For details, see [Tile Creation and Operation](tile_creation_and_operations.md).

## Declaring a Tensor in a Kernel Signature

On the host side, PyTorch is used to create input and output data on the NPU, and the kernel receives this data through the pypto_pro.language.Tensor type annotation. The basic form of the Tensor type annotation is as follows:

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def kernel(
    x: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16],
):
    ...
```

In Tensor[[shape...], dtype], the first parameter is the shape and the second parameter is the element data type. A third parameter can also be used to declare the layout of the GM data:

```python
import pypto_pro.language as pl


nz_tensor: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ]
```

The layout annotation only describes the arrangement of existing data in GM; it does not perform data conversion between different layouts. Data conversion is performed by the data movement APIs that support the corresponding formats.

### Input and Output Direction Annotation

When using Profiling to collect Kernel information, you can declare the parameter direction through the pypto_pro.language.Input or pypto_pro.language.Output enumeration values in the Tensor type annotation:

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def kernel(
    x: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16, pl.Input],
    out: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16, pl.Output],
):
    ...
```

pypto_pro.language.Input indicates input, and pypto_pro.language.Output indicates output. The direction can be omitted; when not annotated, pypto_pro.language.Input is used by default. When you need to report a Tensor as output in the Profiling result, use pypto_pro.language.Output.

In the bracket form, direction can be declared together with layout and memref. It is recommended to place direction at the end of the annotation, for example, `pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ, pl.Output]`. The call form also supports passing direction as the third parameter, for example, `pl.Tensor([64, 128], pl.DT_FP16, pl.Output)`.

The direction annotation only affects the Tensor input and output information in the Profiling result and does not change the read/write semantics of the Kernel. For the complete syntax and constraints, see [pypto_pro.language.Tensor](../../../../api/pro_api/SIMD-API/basic_data_structures/Tensor.md).

### Shape Declaration Method

Each dimension of a Tensor can be declared in the following ways:

| Declaration Method | Description | Impact on Compilation Variants |
|:---|:---|:---|
| Positive integer | Fixed dimension; the actual size at launch must match the declaration | Fixed size |
| pypto_pro.language.DYNAMIC | Runtime dynamic dimension | Reuses the same compilation variant when the dimension value changes |
| pypto_pro.language.STATIC | Compile-time specialized dimension | Generates a new compilation variant when the dimension value changes |
| Trailing ... | Expands the remaining dimensions, each treated as STATIC | Generates a new compilation variant when the rank or dimension value changes |

Different methods can be mixed:

```python
import pypto_pro.language as pl


# Declare a mix of fixed, dynamic, and compile-time specialized dimensions.
x: pl.Tensor[[64, pl.DYNAMIC, pl.STATIC], pl.DT_FP16]

# The rank is determined at call time; the first dimension is DYNAMIC, and the remaining dimensions are treated as STATIC.
y: pl.Tensor[[pl.DYNAMIC, ...], pl.DT_FP16]
```

The ellipsis can appear at most once and must be at the end of the shape. For the complete type constraints of Tensor, see [pypto_pro.language.Tensor](../../../../api/pro_api/SIMD-API/basic_data_structures/Tensor.md).

## Creating a Tensor View from a Ptr or Tensor

When a kernel receives a raw pointer or needs to reinterpret an existing tensor with a new shape and stride, you can call pypto_pro.language.make_tensor to create a tensor view:

```python
pypto_pro.language.make_tensor(
    ptr,
    shape,
    stride=None,
    dtype=None,
)
```

ptr can be a pypto_pro.language.Ptr or an existing tensor. The new tensor shares the same GM address range as the source object; make_tensor does not allocate memory, nor does it copy or rearrange data.

### Creating a Contiguous Tensor View

When stride is omitted, the framework generates a contiguous row-major stride based on the shape:

```python
import pypto_pro.language as pl


tensor = pl.make_tensor(ptr, [8, 16])
# Equivalent to: tensor = pl.make_tensor(ptr, [8, 16], [16, 1])
```

The dimensions in the shape can be either compile-time integers or runtime integer Scalar expressions in the Kernel. Therefore, a dynamic shape can also be passed in through TilingData and then used to create a Tensor view with a fixed rank:

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dynamic_kernel(
    x: pl.Ptr[pl.DT_FP16],
    out: pl.Ptr[pl.DT_FP16],
    tiling: OpTiling,
):
    tensor_x = pl.make_tensor(x, [tiling.m, tiling.n])
    tensor_out = pl.make_tensor(out, [tiling.m, tiling.n])
    ...
```

For details on declaring and passing TilingData, see [Tiling Parameter Definition and Passing](tiling/tiling_parameter_definition.md#tilingdata).

### Creating a Tensor View with Explicit Stride

Explicitly passing a stride can describe a GM view with non-contiguous rows or swapped axes. The number of elements contained in the stride must be the same as the number of dimensions of the shape, and the unit is elements, not bytes. For data types smaller than 8 bits, the stride of the last dimension must be the compile-time constant 1:

```python
import pypto_pro.language as pl


# Each row contains 16 contiguous elements, and the starting positions of two adjacent rows are 32 elements apart.
pitched = pl.make_tensor(ptr, [8, 16], [32, 1])

# normal[i, j] and transposed[j, i] point to the same address.
normal = pl.make_tensor(ptr, [8, 16], [16, 1])
transposed = pl.make_tensor(ptr, [16, 8], [1, 16])
```

The operations above only change the mapping from logical indices to GM addresses, and do not transpose or move the original data. The caller must ensure that the access range described by shape, stride, and dtype does not exceed the source memory. For detailed parameter descriptions, see [pypto_pro.language.make_tensor](../../../../api/pro_api/SIMD-API/resource_management/make_tensor.md).

## Reading the Shape of a Tensor

Within a kernel, read the dimension of a tensor through `tensor.shape[axis]`. `axis` must be a compile-time integer and supports negative indexing. The dimension read can participate in address computation, loop bounds, and multi-core partitioning:

```python
m = x.shape[0]
n = x.shape[-1]
tile_rows = (m + TILE_M - 1) // TILE_M
tile_cols = (n + TILE_N - 1) // TILE_N
```

For a DYNAMIC dimension, the shape value is obtained at runtime; for a STATIC dimension, the shape value is fixed into the corresponding compilation variant.

## Tensor Alias and Pointer Conversion

### Creating a Tensor Alias

You can create an alias for a Tensor through Python assignment. The alias points to the same GM memory as the original Tensor, without any data copy:

```python
original_input = input_tensor
original_input_alias = original_input

# Rebinding the original variable does not change the target of an existing alias.
input_tensor = replacement_tensor
```

### Obtaining a Ptr from a Tensor

pypto_pro.language.make_ptr can extract the underlying pointer from a Tensor, and can also create a new element type view for an existing Ptr:

```python
import pypto_pro.language as pl


ptr = pl.make_ptr(tensor)
fp16_ptr = pl.make_ptr(byte_ptr, dtype=pl.DT_FP16)
```

The returned Ptr shares the address with the source object. Specifying dtype only changes the element type interpretation of the address and does not convert the original data; the caller must ensure that the address alignment and accessible range are correct. For details, see [pypto_pro.language.make_ptr](../../../../api/pro_api/SIMD-API/resource_management/make_ptr.md).

When you need to offset a Ptr element by element, you can use [pypto_pro.language.addptr](../../../../api/pro_api/SIMD-API/resource_management/addptr.md), and then wrap the offset address as a Tensor view through make_tensor. addptr does not support data types smaller than 8 bits, so you need to first reinterpret it as DT_UINT8 through make_ptr and offset it by bytes. This approach is often used to divide a block of GM Workspace into multiple regions.

## Data Movement Between Tensor and Tile

A Tensor resides in GM and cannot be directly used as an on-chip operand of the Tile computation APIs. A Kernel typically first moves data from a Tensor into a Tile, performs the computation, and then moves the result back to the Tensor:

```python
import pypto_pro.language as pl


with pl.section_vector():
    pl.load(input_tile, input_tensor, [row_offset, col_offset])
    # Omit the Tile computation process.
    pl.store(output_tensor, output_tile, [row_offset, col_offset])
```

- load and store use the element coordinates in the Tensor to locate the movement start point.
- load_tile and store_tile locate data blocks by Tile grid coordinates.
- The movement range, stride, layout, and tail block are subject to the constraints of the specific movement API. Invalid parameters that can be determined at compile time are checked by the framework, while dynamic access ranges are guaranteed by the caller.

A Tensor only describes a GM data view. For Tile creation, on-chip addresses, and buffer management, see [Tile Creation and Operations](tile_creation_and_operations.md). For vector computation on a Tile, see [Tile Computation](vector_computation/tile_computation.md). For cube computation, see [Cube Computation](cube_computation.md).
