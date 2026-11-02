# pypto_pro.language.TensorLayout

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:49:46.452Z pushedAt=2026-10-28T11:57:28.653Z nl=2 -->

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

Data layout enumeration, used to describe the storage form of GM Tensor and the data arrangement form of Tile.

- **GM Tensor** supports ND and NZ, defaulting to ND.
- **Tile** supports ND, DN, NZ, ZN, NN, and ZZ, with the default value determined by the memory space.

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

| Parameter Value | Description |
|---|---|
| ND | Non-fractal row-major layout, with the last dimension contiguous. Supported for Tensor and Tile. GM Tensor uses ND by default, typically used for ordinary GM Tensor, UB Tile, and Fixpipe Buffer. |
| DN | Non-fractal column-major layout. Only supported for Tile, typically used for [ROWS, 1] column vectors on UB, such as reduction results and histogram indices. |
| NZ | NZ fractal arrangement. Supported for Tensor and Tile, typically used for GM Tensor, L1 Buffer, L0A Buffer, and L0C Buffer. |
| ZN | ZN fractal arrangement. Only supported for Tile, typically used for L0B Buffer and L1 Buffer during transposed loading. |
| ZZ | ZZ fractal arrangement. Only supported for Tile, used for the layout of the left quantization coefficient matrix in L1 Buffer and L0A_MX Buffer in MX matrix multiplication. |
| NN | NN fractal arrangement. Only supported for Tile, used for the layout of the right quantization coefficient matrix in L1 Buffer and L0B_MX Buffer in MX matrix multiplication. |

## Constraints

- GM Tensor only supports ND (row-major, default) and NZ (fractal layout).

- NZ only declares the layout of GM memory and does not automatically convert an ordinary ND buffer to NZ. Before calling the Kernel, the input buffer must already be packed in NZ physical order; NZ output must also use a buffer allocated in NZ format. The last two axes of a high-dimensional NZ Tensor are fixedly interpreted as [M, N], and all leading axes are treated as batch axes. Specifying arbitrary fractal axes in the layout is not supported.

- NZ stores the logical [..., M, N] as [..., ceil(N/C0), ceil(M/16), 16, C0]. M/N do not require fractal alignment, but the actual storage space must be allocated with a capacity of align(M, 16) × align(N, C0) and use the NZ physical layout described above; a compact buffer that allocates only M × N elements is not supported. The padding region is not part of the logical tensor content, and the framework does not automatically expand or pack the passed-in buffer. For DT_FP4E2M1 and DT_FP4E1M2, C0 is 64; for DT_INT8, DT_UINT8, DT_FP8E4M3FN, DT_FP8E5M2, DT_FP8E8M0, and DT_HF8, C0 is 32; for DT_FP16, DT_BF16, DT_INT16, and DT_UINT16, C0 is 16; for DT_FP32, DT_INT32, and DT_UINT32, C0 is 8; and for DT_INT64 and DT_UINT64, C0 is 4. For 4-bit data types, M/N are also counted in logical elements, and the packed byte count is not used as the tensor shape.

- The E8M0 group scaling factors used in MX matrix computation are still declared as ordinary ND tensors in GM; for the physical shape and data movement constraints, see [matmul_mx](../cube_computation/matmul_mx.md) and [load](../memory_data_movement/load.md).

- For Tile layout constraints, see [pypto_pro.language.TileType](TileType.md).

## Examples

### GM Tensor Layout Declaration

```python
import pypto_pro.language as pl

x: pl.Tensor[[64, 128], pl.DT_FP16]                   # Default ND.
x_nz: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ]         # Explicitly specify NZ.
```

### ND and DN of UB Tile

```python
import pypto_pro.language as pl

# Use ND for regular data tiles in UB.
tile_src = pl.TileType(shape=[32, 128], dtype=pl.DT_UINT16,
                       target_memory=pl.MemorySpace.Vec, layout=pl.ND)

# Use DN for single-column tiles in UB, such as reduction results.
tile_red = pl.TileType(shape=[32, 1], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Vec, layout=pl.DN)
```

### Cube Fractal Layout

The L1 Buffer, L0A Buffer, L0B Buffer, and L0C Buffer involved in Cube computation use different default fractal layouts. The following example explicitly writes out the default layout of each memory space:

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

### ZZ and NN for MX Matrix Multiplication Quantization Coefficients

ZZ and NN are used for the left and right quantization coefficient matrices of MX matrix multiplication, respectively. The data type of the quantization coefficients is DT_FP8E8M0, and the logical shapes are [M, K/32] and [K/32, N]. The L1 Buffer uses NZ by default, so you must explicitly specify ZZ and NN for the left and right quantization coefficient tiles. The L0A_MX Buffer and L0B_MX Buffer use the corresponding layouts by default.

```python
import pypto_pro.language as pl

M, K, N = 64, 128, 64
G = K // 32

# Left and right quantization coefficient tiles in the L1 Buffer.
scale_a_l1_type = pl.TileType(
    shape=[M, G], dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.Mat, layout=pl.ZZ,
)
scale_b_l1_type = pl.TileType(
    shape=[G, N], dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.Mat, layout=pl.NN,
)

# When no layout is specified, the L0A_MX Buffer and L0B_MX Buffer use ZZ and NN by default, respectively.
scale_a_type = pl.TileType(
    shape=[M, G], dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.ScaleLeft,
)
scale_b_type = pl.TileType(
    shape=[G, N], dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.ScaleRight,
)
```
