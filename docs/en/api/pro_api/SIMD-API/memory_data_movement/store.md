# pypto_pro.language.store

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-26T13:04:53.423Z pushedAt=2026-10-28T11:57:28.719Z nl=2 -->

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

Writes the Tile in UB or L0C Buffer back to GM by absolute element coordinates, corresponding to [pypto_pro.language.load](load.md). During data movement, it supports operations such as in-line ReLU, quantization, or atomic accumulation.

## Prototype

```python
pypto_pro.language.store(
    dst_tensor: Tensor,
    src_tile: Tile,
    /,
    offsets: Offset,
    *,
    relu_pre_mode: Optional[ReluPreMode] = None,
    scale: Optional[Union[float, Scalar, Tile]] = None,
    order: Optional[List[int]] = None,
    atomic: AtomicType = AtomicType.AtomicNone,
    phase: Optional[STPhase] = None,
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| dst_tensor | Output | Destination operand of the Tensor type, with GM as the storage space. For supported data type and fractal combinations, see [Constraints](#constraints). |
| src_tile | Input | Source operand of the Tile type, with UB or L0C Buffer as the storage space.<br>- The start address of a UB Tile must be 32-byte aligned.<br>- The start address of an L0C Buffer Tile must be 64-byte aligned. |
| offsets | Input | Optional. Absolute element coordinates of each dimension of the destination tensor, of the List[int or Scalar] type. The length must be the same as the number of dimensions of the destination tensor.<br>- Negative values are not supported.<br>- For a high-dimensional NZ tensor, the last two items correspond to the M and N directions. |
| relu_pre_mode | Input | Optional. Whether to enable the in-line ReLU operation during L0C Buffer→GM movement, of the [pypto_pro.language.ReluPreMode](../basic_data_structures/ReluPreMode.md) type. |
| scale | Input | Optional. Whether to enable quantization and set the quantization parameters in quantization mode. When data is moved out of L0C, Fixpipe multiplies it by this scale and converts it to the destination data type. Different input forms affect the quantization granularity. The following types are supported:<br>- **float type**: Pass a fixed value directly (for example, scale = 2.0), suitable for using the same scale for the entire tile.<br>- **Scalar type**: The quantization scale is determined at runtime and must be passed by data type.<br>&nbsp;&nbsp;- DT_FP32: Pass the original scale value directly (for example, 0.5).<br>&nbsp;&nbsp;- DT_INT32 and DT_INT64: Pass the integer converted from the pre-encoded float32 bit pattern (for example, `struct.pack("!f", 0.5)`).<br>- **Tile type**: Each column uses an independent scale and must meet the following requirements:<br>&nbsp;&nbsp;- target_memory must be pl.MemorySpace.Scaling.<br>&nbsp;&nbsp;- shape must be [1, N] (column-wise quantization), where N must be a multiple of 16 and N ≤ 512.<br>&nbsp;&nbsp;- dtype must be DT_INT64.<br>&nbsp;&nbsp;- When the Tile data type of the destination operand is DT_INT8, bit46 of each DT_INT64 element in the Scaling tile must be set to 1 to select signed quantization. If it is not set, negative values in the L0C Buffer are interpreted as unsigned.<br>&nbsp;&nbsp;- You must first move the scale data from GM to the L1 Buffer, then to Scaling, and complete the MTE1→FIX synchronization.<br>&nbsp;&nbsp;- It cannot be used together with atomic. |
| order | Input | Optional. Dimension mapping of the List[int] type, specifying the destination tensor dimension corresponding to each dimension of the source Tile.<br>- Each dimension number must be within the dimension range of the destination tensor and must not be repeated.<br>- Only ascending order is supported.<br>- When omitted, it corresponds to the last two dimensions of the destination tensor. When the GM fractal is NZ, it can only be specified as the last two dimensions of the destination tensor.<br>- When the tensor is 1-dimensional, the order parameter is not supported. |
| atomic | Input | Optional. Atomic write mode, of the [pypto_pro.language.AtomicType](../basic_data_structures/AtomicType.md) type. |
| phase | Input | Optional. unitFlag mechanism, of the [pypto_pro.language.STPhase](../basic_data_structures/STPhase.md) type. |

## Constraints

- Data Type and Fractal Requirements

  | Source → Destination | Fractal Requirements | Data Type Requirements |
  |---|---|---|
  | UB → GM | The source and destination fractals must be the same. ND, DN, and NZ are supported. | The source and destination data types must have the same bit width. DT_INT8, DT_UINT8, DT_FP16, DT_BF16, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP8E8M0, DT_FP8E4M3FN, DT_FP8E5M2, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2 are supported. |
  | L0C Buffer → GM (without scale) | NZ → ND, NZ → NZ. | DT_FP32 → DT_FP32/DT_FP16/DT_BF16 and DT_INT32 → DT_INT32/DT_FP16/DT_BF16 are supported. |
  | L0C Buffer → GM (with scale) | NZ → ND, NZ → NZ. | DT_FP32 → DT_INT8/DT_UINT8/DT_HF8/DT_FP8E4M3FN/DT_FP16/DT_BF16/DT_FP32 and DT_INT32 → DT_INT8/DT_UINT8/DT_FP16/DT_BF16 are supported. |

- GM NZ layout: For its physical arrangement, fractal axes, and shape constraints of the complete tensor, see [TensorLayout](../basic_data_structures/TensorLayout.md). store must also meet the following NZ movement constraints:
    - The Tile shape and valid M/N must satisfy that M is aligned to 16 and N is aligned to the C0 corresponding to the destination tensor dtype. The offset in the N direction must also be aligned to C0.
    - When a Tile in the L0C Buffer is written back to GM directly, if multiple N fractals are written at a time (valid N is greater than C0), the write-back range must cover the complete NZ physical M axis of the target tensor. If part of M spans multiple N fractals, move it to UB first and then write it back to GM from UB.

- The API does not automatically clear the target tensor. Before the first accumulation, you must initialize the target region to zero or to the expected initial accumulation value.

- When multiple cores accumulate to the same target address simultaneously, each update is atomic. Because floating-point addition does not satisfy the associative law, the result may differ slightly when the update order varies.

## Return Value

None

## Examples

### UB Write Back to GM

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])

    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```
