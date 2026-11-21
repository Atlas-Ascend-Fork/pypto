# pypto_pro.language.load

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

Loads data from a GM Tensor into a Tile in the L1 Buffer or UB. offsets specifies the load start position by the absolute element coordinates of each Tensor dimension. When the number of Tensor dimensions is greater than the Tile dimensions, order specifies the correspondence between the two Tile dimensions and the Tensor dimensions, and controls whether to transpose.

To specify the load position by Tile block ID, use [pypto_pro.language.load_tile](load_tile.md).

## Prototype

```python
pypto_pro.language.load(
    dst_tile: Tile,
    src_tensor: Tensor,
    /,
    offsets: Sequence[int],
    *,
    order: Optional[List[int]] = None,
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| dst_tile | Output| Destination operand, of the Tile type, stored in the L1 Buffer or UB. The start address must be 32-byte aligned. The API loads data according to the valid_shape of this Tile. For supported data types and fractal combinations, see [Constraints](#constraints).|
| src_tensor | Input| Source operand, of the Tensor type, stored in GM. For supported data types and fractal combinations, see [Constraints](#constraints).|
| offsets | Input| Absolute element coordinates of each dimension of the source Tensor, an integer sequence (list or tuple). Each item can be an integer constant or a runtime integer scalar expression. The length must be the same as the number of dimensions of the source Tensor.<br>- Negative numbers are not supported.<br>- Dimensions not selected by order use the corresponding values as fixed subscripts.<br>- The load start position must be within the shape range of the source Tensor. For boundary Tiles, the valid shape can be set via pypto_pro.language.set_validshape to ensure that the valid load range does not exceed the source Tensor boundaries.<br>- For example, offsets=[64, 0] indicates that the load starts from row 64 and column 0 of a 2D Tensor.|
| order | Input| Optional, dimension mapping, which is a list of compile-time integers with a length of 2 or 3. All dimension indices must be unique and within the dimension range of src_tensor.<br>- In standard two-axis load, the i-th item in the list indicates the dimension of src_tensor corresponding to the i-th dimension of dst_tile. Ascending order indicates no transpose, for example, order=[0, 1]. Descending order indicates a transpose, for example, order=[1, 0].<br>- If order is not set for a standard Tensor, dst_tile corresponds to the last two dimensions of src_tensor by default, that is, [ndim - 2, ndim - 1], and no transpose is performed.<br>- When the MX matrix multiplication quantization coefficient is transferred, the last dimension of src_tensor is the physical phase axis and cannot be selected in order. If order is not set, it defaults to the two dimensions preceding the phase axis, that is, [ndim - 3, ndim - 2].<br>- When moving a GM ND Tensor into the L1 Buffer, you can also explicitly specify three axis indices to merge two source axes into one Tile axis. For details, see the constraints on three-axis merged load into L1 Buffer in [Constraints](#constraints).<br>- When the Tensor is one-dimensional, the order parameter cannot be passed.|

## Constraints

- Data type and fractal constraints:

  | Source → Destination| Fractal Requirements| Data Type Requirements|
  |---|---|---|
  | GM → UB | The source and destination fractals must be the same. ND, DN, and NZ are supported.| The bit widths of the source and destination data types must be the same. DT_INT8, DT_UINT8, DT_FP16, DT_BF16, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP8E8M0, DT_FP8E4M3FN, DT_FP8E5M2, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2 are supported.|
  | GM → L1 Buffer | ND → NZ.| The bit widths of the source and destination data types must be the same. DT_INT8, DT_UINT8, DT_FP16, DT_BF16, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, DT_UINT32, DT_FP8E8M0, DT_FP8E4M3FN, DT_FP8E5M2, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2 are supported.|
  | GM → L1 Buffer | DN → NZ.| The bit widths of the source and destination data types must be the same. DT_INT8, DT_UINT8, DT_FP16, DT_BF16, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, DT_UINT32, DT_FP8E8M0, DT_FP8E4M3FN, DT_FP8E5M2, and DT_HF8 are supported.|
  | GM → L1 Buffer | DN → ZN, and NZ → NZ.| The bit widths of the source and destination data types must be the same. DT_INT8, DT_UINT8, DT_FP16, DT_BF16, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP8E8M0, DT_FP8E4M3FN, DT_FP8E5M2, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2 are supported.|
  | GM → L1 Buffer | ND → ND.| The source and destination data types must be the same. Only DT_INT64 and DT_UINT64 are supported.|
  | GM → L1 Buffer | ND/DN → ZZ/NN, used only for loading quantization coefficients of pypto_pro.language.matmul_mx or pypto_pro.language.matmul_mx_acc.| The source and destination data types must be DT_FP8E8M0.|

- Constraints on NZ load:

  - When src_tensor is declared as NZ, dst_tile must also be NZ. Data can only be loaded from the last two dimensions of src_tensor in ascending order, and transpose using descending order is not supported. For details about the physical layout and Tensor shape constraints of NZ, see [TensorLayout](../basic_data_structures/TensorLayout.md).
  - In the shape and valid_shape of dst_tile, M must be aligned to 16, and N must be aligned to the C0 corresponding to the data type of src_tensor. The offset in the N direction must also be aligned to C0.

- Constraints on MX matrix multiplication quantization coefficient load:

  - Only src_tensor of DT_FP8E8M0 can be loaded into an L1 Buffer Tile with a fractal of 32 and a layout of ZZ or NN, and used as the quantization coefficient of pypto_pro.language.matmul_mx or pypto_pro.language.matmul_mx_acc.
  - The number of dimensions of src_tensor must be greater than or equal to 3. The last dimension is a physical phase axis with a length of 2. The offset corresponding to the phase axis in offsets must be 0. When the order is explicitly set, the phase axis cannot be selected.

- Constraints on Tile address reuse:

  When auto_mutex is enabled, if two consecutive pypto_pro.language.load operations write to the same UB or L1 Buffer address and no operation reads the data loaded by the previous load between the two loads, call [pypto_pro.language.system.bar_mte2](../synchronization/bar_mte2.md) between the two loads. pypto_pro.language.system.bar_mte2 only guarantees the order of the two write operations. If the data loaded by the previous load is still needed later, read or copy the data before reusing the address.

- Constraints on three-axis merged load into L1 Buffer:

    - For an ND Tensor `x[B, S, N, G, D]` in GM, if the inner axes are [S, D], standard loading requires N × G separate transfers of [S, D] due to the stride of the G axis. If S is small, the transfer efficiency is low. In scenarios where the G axis is not a batch axis:
    The three-axis `order` can merge the selected S and G axes as `r = s * G + g` while loading data into the L1 Buffer.
    The source Tensor view remains five-dimensional, and the data transfer uses the respective strides of the two source axes.

    - When axis merging is performed through [pypto_pro.language.make_tensor](../resource_management/make_tensor.md):
    Only the shape and stride are modified without data transfer. The adjacent dimensions to be merged must be contiguous with no gaps, and the number of dimensions of the resulting Tensor view is reduced.
    For example, when merging an outer axis with an inner axis, the following condition must be met: `stride (outer axis) = length of the inner axis × stride (inner axis)`.
    The three-axis `order` collects data from two source axes during the transfer, allowing for gaps between the axes, and the number of dimensions of the source Tensor view remains unchanged.

    | order | Destination Tile| Element Mapping|
    |---|---|---|
    | `[1, 3, 4]` | NZ, shape `[M, K]`| `dst[r, c] = x[b, s0 + r // G, h, r % G, k0 + c]` |
    | `[4, 3, 1]` | ZN, shape `[K, M]`| `dst[c, r] = x[b, s0 + r // G, h, r % G, k0 + c]` |

    - The reverse form maintains the S and G sequence within the merged axes and only swaps the two axes of the destination matrix.
    NZ `[M, K]` and ZN `[K, M]` correspond to the same byte layout. Therefore, both forms use the same data transfer, and subsequent loads into the L0 read according to the target layout.
    - The three axis indices must be unique and within the range of the source Tensor. Only a strict ascending order or its complete reverse order is accepted.
    The last axis in ascending order must be the innermost axis of the source Tensor, and the column stride must be the compile-time constant 1.
    The ascending order requires an NZ target, and the reverse order requires a ZN target.
    - The length G of the middle axis in ascending order must be a positive number, and the offset of this axis must be the compile-time constant 0.
    The source and destination dtype must be identical, with an element bit-width of 8, 16, or 32 bits, and must satisfy the data type requirements mentioned above.
    - The declared merged axis length M must be a compile-time constant within the range of `[1, 65535]`.
    M does not need to be exactly divisible by G. Both static and dynamic G support partial matrices, including cases where G is greater than M.
    - `pl.set_validshape` restricts the valid range of the current load. For details about how to set it, see [Loading Tail Blocks](#loading-tail-blocks).
    After a complete matrix is loaded, if there are remaining rows, an additional partial matrix will be loaded. The merged axis in reverse order corresponds to the number of valid columns in the Tile.
    The caller must ensure that the valid range is positive and does not exceed the declared Tile shape, and that the source Tensor covers all requested elements.
    - The three-axis form does not support Vec (UB), `load_tile`, or `store`.
    In Vector scenarios, the selected GM data needs to be moved to the correct UB position before the view is adjusted using `reinterpret`.
    `reinterpret` itself does not collect scattered GM data.

    - If `S_step = M_TILE // G` is used to advance the S axis in a loop, ensure that `M_TILE % G == 0` is still true. Otherwise, adjacent windows will overlap.
    The fact that a single load operation supports partial matrices does not mean that the preceding loop can correctly cover the entire Tensor.

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

### Load by Selecting Dimensions from a High-Dimensional Tensor

The dimension mapping of a high-dimensional Tensor is as follows:

| Tensor shape | Tensor Dimension Corresponding to the Tile| order | Example of offsets|
|---|---|---|---|
| [B, N, S, D] | S, D| Not set. The default value is [2, 3].| [b, n, s_offset, 0] |
| [B, S, N, D] | S, D| [1, 3] | [b, s_offset, n, 0] |

In the following example, order is explicitly set to [1, 3], and the 0th and 2nd items in offsets are used to fix the subscripts of B and N, respectively.

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

| Matrix| Transpose| GM Tensor shape | L1 Buffer Tile | order |
|---|---|---|---|---|
| Left matrix A[M, K]| No| [M, K] | shape=[M, K], NZ| Not set. The default value is [0, 1].|
| Left matrix A[M, K]| Yes| [K, M] | shape=[M, K], ZN| [1, 0] |
| Right matrix B[K, N]| No| [K, N] | shape=[K, N], NZ| Not set. The default value is [0, 1].|
| Right matrix B[K, N]| Yes| [N, K] | shape=[K, N], ZN| [1, 0] |

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

When the remaining data of the input Tensor is smaller than the shape of the Tile, handle it separately depending on whether the subsequent operation reads the invalid region:

- pypto_pro.language.set_validshape is used to set the valid shape of the current Tile so that pypto_pro.language.load loads only the data within the valid_shape range, preventing out-of-bounds reads from GM.
- If subsequent operations, such as reduction or matrix multiplication, read regions outside the valid_shape, you need to set pad in TileType and call pypto_pro.language.fillpad to fill the invalid regions with safe values. Fill with zero for summation, min for maximum or softmax, and max for minimum.

#### Subsequent operations do not read invalid regions

In the following example, only the tail block is written back to GM, and subsequent operations do not read invalid regions. Therefore, pypto_pro.language.fillpad does not need to be called.

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

#### Subsequent operations read invalid regions

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


# The data loaded previously has not been read: Wait for the MTE2 transfer to complete before reusing the address.
pl.load(input_tile, x, [0, 0])
pl.system.bar_mte2()
pl.load(input_tile, x, [64, 0])

# The data loaded previously has been read by Vector computation: Data dependencies are already synchronized,
# so the next transfer can directly reuse the address of input_tile.
pl.load(input_tile, x, [0, 0])
pl.add(output_tile, input_tile, input_tile)
pl.load(input_tile, x, [64, 0])
```

### Three-Axis Merged Load

For a tensor with shape [1, 4, 2, 12, 128], where 4 and 128 correspond to the M and K axes of matrix computation, respectively:
Typically, we directly use `pl.load(nz.current(), x, [0, 0, 0, 0, 0], order=[1, 4])` to load 4 × 128 data for computation.
However, this approach is inefficient. If it is confirmed that axis 12 is not a batch axis, axes 12 and 4 can be merged for loading to improve load efficiency.

```python
import pypto_pro.language as pl

@pl.jit(auto_mutex=True)
def load_merged_rows(x: pl.Tensor[[1, 4, 2, 12, 128], pl.DT_FP16]):
    nz = pl.make_tile_group(
        type=pl.TileType(shape=[48, 128], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x0000, mutex_ids=[0])
    # You can also enable transposed loading by reversing the order.
    zn = pl.make_tile_group(
        type=pl.TileType(shape=[128, 48], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x4000, mutex_ids=[1])
    with pl.section_cube():
        pl.load(nz.current(), x, [0, 0, 0, 0, 0], order=[1, 3, 4])
        pl.load(zn.current(), x, [0, 0, 0, 0, 0], order=[4, 3, 1])
```
