# pypto_pro.language.TensorLayout

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

Enumerates the data layouts. It is used to describe the storage format of the GM tensor and the data arrangement format of the tile.

- The **GM tensor** supports ND and NZ. The default value is ND.
- The **tile** supports ND, DN, NZ, ZN, NN, and ZZ. The default value is determined by the memory space.

## Prototype

```python
PYPTO_DECLARE_ENUM(
    TensorLayout,
    ND,
    DN,
    NZ,
    ZN,
    NN,
    ZZ
)
```

## Parameters

| Value| Description|
|---|---|
| ND | Non-fractal row-major order, with the last dimension being contiguous. It is supported for tensors and tiles. The GM tensors use ND by default, which is typically used for common GM Tensor, UB Tile, and Fixpipe Buffer.|
| DN | Non-fractal column-major order. It is only supported for tiles and is typically used for the [ROWS, 1] column vector on the UB, such as the reduction results and histogram indices.|
| NZ | NZ fractal layout. It is supported for tensors and tiles and is typically used for the GM Tensor, L1 Buffer, L0A Buffer, and L0C Buffer.|
| ZN | ZN fractal layout. It is only supported for tiles and is typically used for the L0B Buffer and the L1 Buffer during transposed loading.|
| ZZ | ZZ fractal layout. It is only supported for tiles and is typically used for the layout of the left quantization coefficient matrix in the L1 Buffer and L0A_MX Buffer in the MX matrix multiplication.|
| NN | NN fractal layout. It is only supported for tiles and is typically used for the layout of the right quantization coefficient matrix in the L1 Buffer and L0B_MX Buffer in the MX matrix multiplication.|

## Constraints

- The GM Tensor only supports ND (row-major order, default) and NZ (fractal layout).

- NZ only declares the layout of the GM memory, and does not automatically convert an ordinary ND buffer into NZ. Before the kernel is called, the input buffer must be packed by NZ physical order. The NZ output must also use a buffer allocated in NZ format. The last two axes of the high-dimensional NZ tensor are fixed as [M, N]. All leading axes are used as batch axes. Specifying arbitrary fractal axes in the layout is not supported.

- NZ stores the logical [..., M, N] as [..., ceil(N/C0), ceil(M/16), 16, C0]. M/N does not require fractal alignment, but the actual storage space must be allocated and used based on the capacity of align(M, 16) × align(N, C0) and use the preceding NZ physical layout. A compact buffer allocated only for M × N elements is not supported. The padding region is not part of the logical tensor. The framework does not automatically expand or pack the passed-in buffer. For DT_FP4E2M1 and DT_FP4E1M2, C0 is 64. For DT_INT8, DT_UINT8, DT_FP8E4M3FN, DT_FP8E5M2, DT_FP8E8M0, and DT_HF8, C0 is 32. For DT_FP16, DT_BF16, DT_INT16, and DT_UINT16, C0 is 16. For DT_FP32, DT_INT32, and DT_UINT32, C0 is 8. For DT_INT64 and DT_UINT64, C0 is 4. The M/N of 4-bit data types is also counted in logical elements, and the number of packed bytes is not used as the tensor shape.

- The E8M0 group scaling factors used in MX matrix computation are still declared as ordinary ND tensors in the GM. For details about the physical shape and movement constraints, see [matmul_mx](../cube_computation/matmul_mx.md) and [load](../memory_data_movement/load.md).

- For details about the tile layout constraints, see [pypto_pro.language.TileType](TileType.md).

## Examples

### GM Tensor Layout Declaration

```python
import pypto_pro.language as pl

x: pl.Tensor[[64, 128], pl.DT_FP16]                   # ND by default
x_nz: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ]         # Explicitly specifies NZ.
```

### ND and DN of the UB Tile

```python
import pypto_pro.language as pl

# Use ND for regular data tiles in the UB.
tile_src = pl.TileType(shape=[32, 128], dtype=pl.DT_UINT16,
                       target_memory=pl.MemorySpace.Vec, layout=pl.ND)

# Use DN for single-column tiles in the UB, such as reduction results.
tile_red = pl.TileType(shape=[32, 1], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Vec, layout=pl.DN)
```

### Cube Fractal Layout

The L1 Buffer, L0A Buffer, L0B Buffer, and L0C Buffer involved in Cube computations use different default fractal layouts. The following example explicitly describes the default layout of each memory space:

```python
import pypto_pro.language as pl

# L1 Buffer and L0A Buffer use NZ by default.
mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
left_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                        target_memory=pl.MemorySpace.Left, layout=pl.NZ)

# L0B Buffer uses ZN by default.
right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN)

# L0C Buffer uses NZ by default.
acc_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
```

### ZZ and NN of the Quantization Coefficient for MX Matrix Multiplication

ZZ and NN are used for the left and right quantization coefficient matrices of the MX matrix multiplication, respectively. The data type of the quantization coefficients is DT_FP8E8M0, and the logical shapes are [M, K/32] and [K/32, N], respectively. The L1 Buffer uses NZ by default. Therefore, ZZ and NN need to be explicitly specified for the left and right quantization coefficient tiles. The L0A_MX Buffer and L0B_MX Buffer use the corresponding layouts by default.

```python
import pypto_pro.language as pl

M, K, N = 64, 128, 64
G = K // 32

# Tiles of left and right quantization coefficients in the L1 Buffer
scale_a_l1_type = pl.TileType(
    shape=[M, G], dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.Mat, layout=pl.ZZ,
)
scale_b_l1_type = pl.TileType(
    shape=[G, N], dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.Mat, layout=pl.NN,
)

# If the layout is not specified for the L0A_MX Buffer and L0B_MX Buffer, ZZ and NN are used by default.
scale_a_type = pl.TileType(
    shape=[M, G], dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.ScaleLeft,
)
scale_b_type = pl.TileType(
    shape=[G, N], dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.ScaleRight,
)
```
