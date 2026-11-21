# pypto_pro.language.store

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

Writes the Tile in UB or L0C Buffer back to GM by absolute element coordinates, corresponding to [pypto_pro.language.load](load.md). During data movement, it supports operations such as on-the-fly ReLU, quantization, or atomic accumulation.

## Prototype

```python
pypto_pro.language.store(
    dst_tensor: Tensor,
    src_tile: Tile,
    /,
    offsets: Sequence[int],
    *,
    relu_pre_mode: Optional[ReluPreMode] = None,
    scale: Optional[Union[int, float, Tile]] = None,
    order: Optional[List[int]] = None,
    atomic: AtomicType = AtomicType.AtomicNone,
    phase: Optional[STPhase] = None,
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| dst_tensor | Output| Destination operand, of the Tensor type, stored in GM. For supported data types and fractal combinations, see [Constraints](#constraints).|
| src_tile | Input| Source operand, of the Tile type, stored in the UB or L0C Buffer.<br>- The start address of a UB Tile must be 32-byte aligned.<br>- The start address of an L0C Buffer Tile must be 64-byte aligned.|
| offsets | Input| Absolute element coordinates of each dimension of the destination Tensor, an integer sequence (list or tuple). Each item can be an integer constant or a runtime integer scalar expression. The length must be the same as the number of dimensions of the destination Tensor.<br>- Negative numbers are not supported.<br>- For a high-dimensional NZ Tensor, the last two items correspond to the M and N directions.|
| relu_pre_mode | Input| Optional. Whether to enable on-the-fly ReLU operation during data movement from the L0C Buffer to GM. The value is of the [pypto_pro.language.ReluPreMode](../basic_data_structures/ReluPreMode.md) type.|
| scale | Input| Optional. Whether to enable quantization and set quantization parameters in quantization mode. When data is moved out of the L0C Buffer, Fixpipe multiplies it by this scale factor and converts it to the destination data type. Different input formats affect the quantization granularity. The following types are supported:<br>- **Integer or floating-point constant**: The value is first converted to a floating-point value and then encoded in FP32 mode (for example, the encoding results of scale = 2 and scale = 2.0 are the same). This mode is applicable when the entire tile uses the same scale.<br>- **Runtime scalar expression**: The quantization scale is determined at runtime, and values need to be passed based on the data type.<br>&nbsp;&nbsp;- DT_FP32: Pass the original scale value (for example, 0.5).<br>&nbsp;&nbsp;- DT_INT32 and DT_INT64: Pass the integer converted from the pre-encoded float32 bit pattern (for example, the bit pattern integer 0x3F000000 corresponding to 0.5).<br>- **Tile type**: Each column uses an independent scale, which must meet the following requirements:<br>&nbsp;&nbsp;- target_memory must be pl.MemorySpace.Scaling.<br>&nbsp;&nbsp;- The shape is [1, N] (column-wise quantization). N must be a multiple of 16 and N ≤ 512.<br>&nbsp;&nbsp;- dtype is DT_INT64.<br>&nbsp;&nbsp;- When the Tile data type of the destination operand is DT_INT8, bit46 of each DT_INT64 element in the Scaling tile must be set to 1 to select signed quantization. If it is not set, negative values in the L0C Buffer will be interpreted as unsigned values.<br>&nbsp;&nbsp;- You need to move the scale data from GM to the L1 Buffer, then to Scaling, and complete the MTE1→FIX synchronization.<br>&nbsp;&nbsp;- It cannot be used together with atomic.|
| order | Input| Optional. Dimension mapping, of the List[int] type, specifying the destination tensor dimension corresponding to each dimension of the source Tile.<br>- The dimension IDs must be within the dimension range of the destination Tensor and must be unique.<br>- Only ascending order is supported.<br>- When omitted, it corresponds to the last two dimensions of the destination Tensor. When the GM fractal is NZ, it can only be specified as the last two dimensions of the destination Tensor.<br>- When the Tensor is one-dimensional, the order parameter cannot be passed.|
| atomic | Input| Optional. Atomic write mode, of the [pypto_pro.language.AtomicType](../basic_data_structures/AtomicType.md) type.|
| phase | Input| Optional. unitFlag mechanism, of the [pypto_pro.language.STPhase](../basic_data_structures/STPhase.md) type.|

## Constraints

- Data Type and Fractal Requirements

  | Source → Destination| Fractal Requirements| Data Type Requirements|
  |---|---|---|
  | UB → GM | The source and destination fractals must be the same. ND, DN, and NZ are supported.| The bit widths of the source and destination data types must be the same. DT_INT8, DT_UINT8, DT_FP16, DT_BF16, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP8E8M0, DT_FP8E4M3FN, DT_FP8E5M2, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2 are supported.|
  | L0C Buffer → GM (with scale not configured)| NZ → ND and NZ → NZ.| DT_FP32 → DT_FP32/DT_FP16/DT_BF16 and DT_INT32 → DT_INT32/DT_FP16/DT_BF16 are supported.|
  | L0C Buffer → GM (with scale configured)| NZ → ND and NZ → NZ.| DT_FP32 → DT_INT8/DT_UINT8/DT_HF8/DT_FP8E4M3FN/DT_FP16/DT_BF16/DT_FP32 and DT_INT32 → DT_INT8/DT_UINT8/DT_FP16/DT_BF16 are supported.|

- GM NZ layout: For its physical arrangement, fractal axes, and shape constraints of the complete tensor, see [TensorLayout](../basic_data_structures/TensorLayout.md). store must also meet the following NZ movement constraints:
    - The Tile shape and valid M/N must satisfy that M is aligned to 16 and N is aligned to the C0 corresponding to the destination Tensor dtype. The offset in the N direction must also be aligned to C0.
    - When a Tile in the L0C Buffer is written back to GM directly, if multiple N fractals are written at a time (valid N is greater than C0), the write-back range must cover the complete NZ physical M axis of the destination Tensor. If part of M spans multiple N fractals, move it to UB first and then write it back to GM from UB.

- The API does not automatically clear the destination Tensor. Before the first accumulation, you must initialize the target region to zero or to the expected initial accumulation value.

- When multiple cores accumulate to the same destination address simultaneously, each update is atomic. Because floating-point addition does not satisfy the associative law, the result may differ slightly when the update order varies.

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
