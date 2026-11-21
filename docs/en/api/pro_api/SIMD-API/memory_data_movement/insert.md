# pypto_pro.language.insert

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

Moves a smaller source Tile to the position specified by offset=[row, col] in a larger destination Tile. During the movement, operations such as on-the-fly format conversion, quantization, and activation can be performed. You are advised to use the [pypto_pro.language.move](move.md) API instead of the insert API.

## Prototype

```python
pypto_pro.language.insert(
    dst_tile: Tile,
    src_tile: Tile,
    /,
    offset: Sequence[int],
    *,
    relu_pre_mode: Optional[ReluPreMode] = None,
    scale: Optional[Union[int, float, Tile]] = None,
    phase: Optional[STPhase] = None,
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| dst_tile | Output| Destination operand, of the Tile type. The L0C buffer → L1 Buffer path must use the NZ layout, and the start address must be 32-byte aligned. For supported data types and fractal combinations, see [Constraints](#constraints).|
| src_tile | Input| Source operand, of the Tile type. For supported data types and fractal combinations, see [Constraints](#constraints).|
| offset | Input| Position offset, an integer sequence (list or tuple), with a mandatory length of 2, formatted as [row, col], in units of elements. The upper left corner of the source Tile is aligned with the offset position of the destination Tile. The values of row and col must be non-negative integers or runtime integer expressions, and must meet the following conditions: row + number of source rows ≤ number of destination rows, and col + number of source columns ≤ number of destination columns.|
| relu_pre_mode | Input| Optional. Whether to enable on-the-fly ReLU operation during data movement from the L0C Buffer to the L1 Buffer. The value is of the [pypto_pro.language.ReluPreMode](../basic_data_structures/ReluPreMode.md) type. Not supported on other paths.|
| scale | Input| Optional, used only for data movement from the L0C Buffer to the L1 Buffer to set on-the-fly quantization parameters. When data is moved out of the L0C Buffer, Fixpipe multiplies it by this scale factor and converts it to the destination data type. Different input formats affect the quantization granularity. The following types are supported:<br>- **Integer or floating-point constant**: The value is first converted to a floating-point value and then encoded in FP32 mode (for example, the encoding results of scale = 2 and scale = 2.0 are the same). This mode is applicable when the entire Tile uses the same scale.<br>- **Runtime scalar expression**: The quantization scale is determined at runtime, and values need to be passed based on the data type.<br>&nbsp;&nbsp;- DT_FP32: Pass the original scale value (for example, 0.5).<br>&nbsp;&nbsp;- DT_INT32 and DT_INT64: Pass the integer converted from the pre-encoded float32 bit pattern (for example, the bit pattern integer 0x3F000000 corresponding to 0.5).<br>- **Tile type**: Each column uses an independent scale, which must meet the following requirements:<br>&nbsp;&nbsp;- The destination storage area must be the Fixpipe Buffer.<br>&nbsp;&nbsp;- The shape is [1, N] (column-wise quantization). N must be a multiple of 16 and N ≤ 512.<br>&nbsp;&nbsp;- dtype is DT_INT64.<br>&nbsp;&nbsp;- When the Tile data type of the destination operand is DT_INT8, bit46 of each DT_INT64 element in the Fixpipe Buffer Tile must be set to 1 to select signed quantization. If it is not set, negative values in the L0C Buffer will be interpreted as unsigned values.<br>&nbsp;&nbsp;- You need to move the scale data from GM to the L1 Buffer, then to the Fixpipe Buffer, and complete the MTE1→FIX synchronization.|
| phase | Input| Optional. Whether to enable unit_flag synchronization during data movement from the L0C Buffer to the L1 Buffer. For details, see [Advanced Cube Computation](../../../../guide/programming_guide/pro/advanced_programming/cube_computation_advanced_usage.md).|

## Constraints

- Data type and fractal constraints:

  | Source → Destination| Fractal Requirements| Data Type Requirements|
  |---|---|---|
  | UB → UB | ND → ND and NZ → NZ.| The source and destination must be the same. DT_INT8, DT_INT32, DT_FP16, DT_BF16, DT_FP32, DT_FP8E4M3FN, DT_FP8E5M2, DT_FP8E8M0, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2 are supported.|
  | UB → L1 Buffer | The source supports ND and NZ. The fractal of the destination is not checked.| The source and destination must be the same. DT_INT8, DT_INT32, DT_FP16, DT_BF16, DT_FP32, DT_FP8E4M3FN, DT_FP8E5M2, DT_FP8E8M0, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2 are supported.|
  | L0C Buffer → UB | NZ → ND, NZ → DN, and NZ → NZ.| DT_FP32 → DT_FP32/DT_FP16/DT_BF16 and DT_INT32 → DT_INT32 are supported.|
  | L0C Buffer → L1 Buffer (scale not configured)| NZ → NZ.| DT_FP32 → DT_FP32/DT_FP16/DT_BF16 and DT_INT32 → DT_INT32 are supported.|
  | L0C Buffer → L1 Buffer (scale configured)| NZ → NZ.| DT_FP32 → DT_INT8/DT_UINT8/DT_FP16/DT_BF16/DT_HF8/DT_FP8E4M3FN and DT_INT32 → DT_INT8/DT_UINT8/DT_FP16/DT_BF16 are supported.|

  When the source operand is in the UB and is in ND format and the destination operand is in the L1 Buffer and is in NZ or ZN format, do not transfer data directly. The source operand must be transferred to another UB in NZ format by using pypto_pro.language.move, and then transferred to the destination operand. For the correct usage, see [Concatenating the Computation Results in the UB to the L1 Buffer](#concatenating-the-computation-results-in-the-ub-to-the-l1-buffer).

## Return Value

None

## Examples

### Concatenating the Computation Results in the UB to the L1 Buffer

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def insert_matmul_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
    rhs: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    v1_mat_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                         layout=pl.NZ),
        addrs=0x10000, mutex_ids=[0])

    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        off = sub_index * 32

        tile_x_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x0000, mutex_ids=[1])
        tile_y_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x2000, mutex_ids=[2])
        tile_sum_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x4000, mutex_ids=[3])
        tile_nz_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                             layout=pl.NZ),
            addrs=0x6000, mutex_ids=[4])
        v1_mat = v1_mat_group.current()
        tile_x = tile_x_group.current()
        tile_y = tile_y_group.current()
        tile_sum = tile_sum_group.current()
        tile_nz = tile_nz_group.current()

        pl.load(tile_x, x, [off, 0])
        pl.load(tile_y, y, [off, 0])

        pl.add(tile_sum, tile_x, tile_y)
        pl.move(tile_nz, tile_sum)   # ND -> NZ

        pl.insert(v1_mat, tile_nz, [off, 0])   # UB -> L1 NZ2NZ
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

    with pl.section_cube():
        rhs_mat_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                             layout=pl.NZ),
            addrs=0x0000, mutex_ids=[5])
        v1_left_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left,
                             layout=pl.NZ),
            addrs=0x0000, mutex_ids=[6])
        rhs_right_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right,
                             layout=pl.ZN),
            addrs=0x0000, mutex_ids=[7])
        c_l0c_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                             layout=pl.NZ, fractal=1024),
            addrs=0x0000, mutex_ids=[8])
        v1_mat = v1_mat_group.current()
        rhs_mat = rhs_mat_group.current()
        v1_left = v1_left_group.current()
        rhs_right = rhs_right_group.current()
        c_l0c = c_l0c_group.current()

        pl.load(rhs_mat, rhs, [0, 0])
        pl.move(rhs_right, rhs_mat)

        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.move(v1_left, v1_mat)

        pl.matmul(c_l0c, v1_left, rhs_right)

        pl.store(out, c_l0c, [0, 0])
```

### Other Two-Dimensional Offset Scenarios

```python
# Offsets exist in both dimensions.
pl.insert(p_mat_slot, p_f16_back_slot, [TKV // 2, TS_HALF * sub_id])

# Offset only along dimension 0
pl.insert(v1_mat, tile_nz, [off, 0])

# Write the L0C FP32 NZ sub-block to the L1 INT8 NZ destination window after on-the-fly quantization.
pl.insert(mat_int8, acc_fp32, [16, 32], scale=2.0, relu_pre_mode=pl.ReluPreMode.NormalRelu)
```
