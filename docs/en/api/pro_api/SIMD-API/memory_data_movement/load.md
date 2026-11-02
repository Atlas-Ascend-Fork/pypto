# pypto_pro.language.load

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:03:02.929Z pushedAt=2026-10-28T11:57:28.722Z nl=2 -->

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

Loads data from a GM Tensor into a Tile in the L1 Buffer or UB. offsets specifies the load start position by the absolute element coordinates of each Tensor dimension. When the Tensor rank is greater than the Tile rank, order specifies the correspondence between the two Tile dimensions and the Tensor dimensions, and controls whether to transpose.

To specify the load position by Tile block number, use [pypto_pro.language.load_tile](load_tile.md).

## Prototype

```python
pypto_pro.language.load(
    dst_tile: Tile,
    src_tensor: Tensor,
    /,
    offsets: Offset,
    *,
    order: Optional[List[int]] = None,
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| dst_tile | Output | Destination operand, of Tile type, with storage space in the L1 Buffer or UB. The first address must be 32-byte aligned. The API loads data according to the valid_shape of this Tile. For supported data types and fractal combinations, see [Constraints](#constraints). |
| src_tensor | Input | Source operand, of Tensor type, with storage space in GM. For supported data types and fractal combinations, see [Constraints](#constraints). |
| offsets | Input | Absolute element coordinates of each dimension of the source Tensor, of List[int or Scalar] type. The length must be the same as the rank of the source Tensor.<br>- Negative values are not supported.<br>- Dimensions not selected by order use the corresponding values as fixed subscripts.<br>- The load start position must be within the shape range of the source operand. You can use pypto_pro.language.set_validshape to ensure that the valid load range does not exceed the shape range of the source operand. |
| order | Input | Optional. Dimension mapping, a compile-time integer list of length 2, specifying the Tensor dimension indices of the source operand corresponding to each dimension of the destination operand Tile.<br>- The two dimension indices must be unique and within the Tensor dimension range of the source operand.<br>- Ascending order means no transpose, for example, order=[0, 1].<br>- Descending order means transpose, for example, order=[1, 0], where the Tensor fractal description of the source operand changes from ND to DN.<br>- When not configured, the last two dimensions of the source operand Tensor are specified, without transpose.<br>- When loading MX matrix multiplication quantization coefficients, order cannot select the last dimension of the source operand Tensor. When order is not set, the default is the last two dimensions excluding the last dimension of the source operand Tensor.<br>- When the Tensor is 1-dimensional, the order parameter is not supported. |

## Constraints

- Data type and fractal constraints:

  | Source → Destination | Fractal Requirement | Data Type Requirement |
  |---|---|---|
  | GM → UB | The source and destination fractals must be the same. ND, DN, and NZ are supported. | The source and destination data type bit widths must be the same. DT_INT8, DT_UINT8, DT_FP16, DT_BF16, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP8E8M0, DT_FP8E4M3FN, DT_FP8E5M2, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2 are supported. |
  | GM → L1 Buffer | ND → NZ. | The source and destination data type bit widths must be the same. DT_INT8, DT_UINT8, DT_FP16, DT_BF16, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, DT_UINT32, DT_FP8E8M0, DT_FP8E4M3FN, DT_FP8E5M2, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2 are supported. |
  | GM → L1 Buffer | DN → NZ. | The source and destination data type bit widths must be the same. DT_INT8, DT_UINT8, DT_FP16, DT_BF16, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, DT_UINT32, DT_FP8E8M0, DT_FP8E4M3FN, DT_FP8E5M2, and DT_HF8 are supported. |
  | GM → L1 Buffer | DN → ZN, NZ → NZ. | The source and destination data type bit widths must be the same. DT_INT8, DT_UINT8, DT_FP16, DT_BF16, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP8E8M0, DT_FP8E4M3FN, DT_FP8E5M2, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2 are supported. |
  | GM → L1 Buffer | ND → ND. | The source and destination data types must be the same. Only DT_INT64 and DT_UINT64 are supported. |
  | GM → L1 Buffer | ND/DN → ZZ/NN, used only for loading quantization coefficients of pypto_pro.language.matmul_mx or pypto_pro.language.matmul_mx_acc. | The source and destination data types must be DT_FP8E8M0. |

- NZ load constraints:

  - When src_tensor is declared as NZ, dst_tile must be NZ, and only forward-order load from the last two dimensions of src_tensor is supported. Transpose through a descending order is not supported. For details about the physical layout of NZ and Tensor shape constraints, see [TensorLayout](../basic_data_structures/TensorLayout.md).
  - In the shape and valid_shape of dst_tile, M must be aligned to 16, and N must be aligned to the C0 corresponding to the data type of src_tensor. The offset in the N direction must also be aligned to C0.

- MX matrix multiplication quantization coefficient load constraints:

  - Only src_tensor of DT_FP8E8M0 can be loaded into an L1 Buffer Tile with a fractal of 32 and a layout of ZZ or NN, and used as the quantization coefficient of pypto_pro.language.matmul_mx or pypto_pro.language.matmul_mx_acc.
  - The number of dimensions of src_tensor must be greater than or equal to 3. The last dimension is a physical phase axis with a length of 2, and the offset corresponding to the phase axis in offsets must be 0. When order is explicitly set, the phase axis cannot be selected.

- Tile address reuse constraints:

  When auto_mutex is enabled, if two consecutive pypto_pro.language.load operations write to the same UB or L1 Buffer address and no operation reads the data loaded by the previous load between the two loads, call [pypto_pro.language.system.bar_mte2](../synchronization/bar_mte2.md) between the two loads. pypto_pro.language.system.bar_mte2 only guarantees the order of the two write operations. If the data loaded by the previous load is still needed later, read or copy the data before reusing the address.

## Return Value

None

## Examples

### Load by Absolute Element Coordinates

Load the data at row indices 64 to 127 of the input Tensor into the UB Tile, perform the computation, and then write the result to row indices 0 to 63 of the output Tensor.

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_kernel(
    x: pl.Tensor[[128, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    x_tile = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    out_tile = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[1])

    with pl.section_vector():
        current_x = x_tile.current()
        current_out = out_tile.current()
        pl.load(current_x, x, [64, 0])
        pl.add(current_out, current_x, current_x)
        pl.store(out, current_out, [0, 0])
```

### Loading by Selecting Dimensions from a High-Dimensional Tensor

The dimension mapping of a high-dimensional tensor is as follows.

| Tensor shape | Tensor dimensions corresponding to the Tile | order | offsets example |
|---|---|---|---|
| [B, N, S, D] | S, D | Not set, default value is [2, 3] | [b, n, s_offset, 0] |
| [B, S, N, D] | S, D | [1, 3] | [b, s_offset, n, 0] |

The following example explicitly sets order=[1, 3], where the 0th and 2nd items in offsets fix the subscripts of B and N, respectively.

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_high_dim_kernel(
    x: pl.Tensor[[2, 64, 8, 128], pl.DT_FP16],
    out: pl.Tensor[[2, 64, 8, 128], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[64, 128],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    x_tile = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    out_tile = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[1])

    with pl.section_vector():
        current_x = x_tile.current()
        current_out = out_tile.current()
        # Fix B=1 and N=3, and start loading from S=0 and D=0.
        pl.load(current_x, x, [1, 0, 3, 0], order=[1, 3])
        pl.add(current_out, current_x, current_x)
        pl.store(out, current_out, [1, 0, 3, 0], order=[1, 3])
```

### Cube-Side Transpose Load

In Cube computation, the left matrix is loaded into the L0A Buffer using shape=[M, K] and NZ, and the right matrix is loaded into the L0B Buffer using shape=[K, N] and ZN. The shape of the L1 Buffer Tile must be the same as that of the corresponding L0A Buffer or L0B Buffer Tile. When loading from a regular ND GM Tensor into the L1 Buffer, whether a transpose is performed is jointly determined by order and the fractal of the L1 Buffer Tile.

| Matrix | Transpose | GM Tensor shape | L1 Buffer Tile | order |
|---|---|---|---|---|
| left matrix A[M, K] | No | [M, K] | shape=[M, K], NZ | not set, default value is [0, 1] |
| left matrix A[M, K] | Yes | [K, M] | shape=[M, K], ZN | [1, 0] |
| right matrix B[K, N] | No | [K, N] | shape=[K, N], NZ | not set, default value is [0, 1] |
| right matrix B[K, N] | Yes | [N, K] | shape=[K, N], ZN | [1, 0] |

order=[1, 0] means that dimension 0 of the Tile corresponds to dimension 1 of the Tensor and dimension 1 of the Tile corresponds to dimension 0 of the Tensor, and a transpose is performed during the load.

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_left_transpose_kernel(a_transposed: pl.Tensor[[128, 64], pl.DT_FP16]):
    tile_type = pl.TileType(
        shape=[64, 128],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.ZN,
    )
    a_l1 = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])

    with pl.section_cube():
        pl.load(a_l1.current(), a_transposed, [0, 0], order=[1, 0])
```

### Loading Tail Blocks

When the remaining data of the input tensor is smaller than the shape of the tile, handle it separately depending on whether the subsequent operation reads the invalid region:

- pypto_pro.language.set_validshape sets the valid shape of the current tile so that pypto_pro.language.load loads only the data within the valid_shape range, avoiding out-of-bounds reads from GM.
- If the subsequent operation reads the region outside valid_shape, such as reduction or matrix multiplication, set pad in TileType and call pypto_pro.language.fillpad to fill the invalid region with a safe value. Fill with zero for summation, min for maximum or softmax, and max for minimum.

#### Subsequent operation does not read the invalid region

In the following example, only the tail block is written back to GM, and the subsequent operation does not read the invalid region, so there is no need to call pypto_pro.language.fillpad.

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_tail_kernel(
    x: pl.Tensor[[70, 128], pl.DT_FP16],
    out: pl.Tensor[[70, 128], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[64, 128],
        valid_shape=[-1, -1],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    x_tile = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        current_x = x_tile.current()
        pl.set_validshape(current_x, [6, 128])
        pl.load(current_x, x, [64, 0])
        pl.store(out, current_x, [64, 0])
```

#### Reading Invalid Regions in Subsequent Operations

When a subsequent operation reads the entire Tile, load the valid data into a source Tile with a dynamic valid_shape, and then use pypto_pro.language.fillpad to write it into a destination Tile configured with pad. The following example loads the last 6 rows of the input Tensor into a Tile and fills the remaining regions with 0. The zero-padded Tile can be used for operations that read the entire Tile, such as summation.

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def load_tail_with_padding_kernel(
    x: pl.Tensor[[70, 128], pl.DT_FP16],
    padded_out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    src_type = pl.TileType(
        shape=[64, 128],
        valid_shape=[-1, -1],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    padded_type = pl.TileType(
        shape=[64, 128],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
        pad=pl.TilePad.zero,
    )
    src_tile = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    padded_tile = pl.make_tile_group(type=padded_type, addrs=0x4000, mutex_ids=[1])

    with pl.section_vector():
        current_src = src_tile.current()
        current_padded = padded_tile.current()
        pl.set_validshape(current_src, [6, 128])
        pl.load(current_src, x, [64, 0])
        pl.fillpad(current_padded, current_src)
        pl.store(padded_out, current_padded, [0, 0])
```

For the method of calculating the number of tail blocks, valid shapes, and task indices in multi-core tasks, see [Tail Block Processing](../../../../guide/programming_guide/pro/development/tiling/multi_core_tiling.md#tail-block-processing).

### Reusing Tile Addresses

When auto_mutex is enabled and the same UB or L1 Buffer address is reused consecutively, whether manual synchronization is required depends on whether the data loaded previously has been read.

```python
import pypto_pro.language as pl


# The data loaded previously has not been read: wait for the MTE2 transfer to complete before reusing the address.
pl.load(input_tile, x, [0, 0])
pl.system.bar_mte2()
pl.load(input_tile, x, [64, 0])

# The data loaded previously has been read by Vector computation: data dependency has already established synchronization,
# so the next transfer can directly reuse the address of input_tile.
pl.load(input_tile, x, [0, 0])
pl.add(output_tile, input_tile, input_tile)
pl.load(input_tile, x, [64, 0])
```
