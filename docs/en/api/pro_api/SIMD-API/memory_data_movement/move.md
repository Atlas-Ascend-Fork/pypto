# pypto_pro.language.move

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-26T13:02:38.992Z pushedAt=2026-10-28T11:57:28.712Z nl=2 -->

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

Provides data movement between L1 Buffer, L0A Buffer/L0B Buffer, L0C Buffer, UB, and other memory levels, and supports on-the-fly format conversion, quantization, and activation operations during the movement.

## Prototype

```python
pypto_pro.language.move(
    dst_tile: Tile,
    src_tile: Tile,
    /,
    offset: Optional[Offset] = None,
    *,
    acc_to_vec_mode: Optional[AccToVecMode] = None,
    relu_pre_mode: Optional[ReluPreMode] = None,
    scale: Optional[Union[float, Scalar, Tile]] = None,
    phase: Optional[STPhase] = None,
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| dst_tile | Output | Destination operand, of Tile type. For supported data types and fractals, see [Constraints](#constraints). |
| src_tile | Input | Source operand, of Tile type. For supported data types and fractals, see [Constraints](#constraints). |
| offset | Input | Optional. Indicates the relative position of a small Tile within a large Tile, in the format [offset_m, offset_n], in units of elements.<br>- When the shape of the source operand >= the shape of the destination operand, data is read starting from row offset_m and column offset_n of the source operand, and the amount of data moved is taken from the valid_shape of the destination operand.<br>- When the shape of the source operand < the shape of the destination operand, data is written starting from row offset_m and column offset_n of the destination operand, and the amount of data moved is taken from the valid_shape of the source operand. |
| acc_to_vec_mode | Input | Optional. Data distribution mode for L0C Buffer → UB movement, of [pypto_pro.language.AccToVecMode](../basic_data_structures/AccToVecMode.md) type, defaulting to `None`.<br>- When not configured or configured as `None`, data is moved only to Vec0, not to Vec1.<br>- When DualModeSplitN is used together with phase, if N is not aligned to 32, a hang may occur. You can set the valid shape of N on the L0B Buffer before the matrix multiplication operation to align it upward to a multiple of 32. |
| relu_pre_mode | Input | Optional. Whether to enable the on-the-fly ReLU operation during L0C Buffer → UB movement, of [pypto_pro.language.ReluPreMode](../basic_data_structures/ReluPreMode.md) type. |
| scale | Input | Optional. Whether to enable quantization and set the quantization parameters in quantization mode. When data is moved out of the L0C Buffer, Fixpipe multiplies it by this ratio and converts it to the destination data type. It cannot be used together with dual-destination movement (AccToVecMode.DualModeSplitM / AccToVecMode.DualModeSplitN). Different input forms affect the quantization granularity. The following types are supported:<br>- **float type**: Pass a fixed value directly (for example, scale = 2.0), suitable for using the same ratio for the entire Tile.<br>- **Scalar type**: The quantization ratio is determined at runtime and must be passed by data type.<br>&nbsp;&nbsp;- DT_FP32: Pass the original ratio value directly (for example, 0.5).<br>&nbsp;&nbsp;- DT_INT32, DT_INT64: Pass the integer converted from the pre-encoded float32 bit pattern (for example, struct.pack("!f", 0.5)).<br>- **Tile type**: Each column uses an independent ratio, and the following requirements must be met:<br>&nbsp;&nbsp;- The destination storage area must be a Fixpipe Buffer.<br>&nbsp;&nbsp;- The shape is [1, N] (column quantization), where N must be a multiple of 16 and N ≤ 512.<br>&nbsp;&nbsp;- The dtype is DT_INT64.<br>&nbsp;&nbsp;- When the Tile data type of the destination operand is DT_INT8, bit46 of each DT_INT64 element in the Tile in the Fixpipe Buffer must be set to 1 to select signed quantization; if it is not set, negative values in the L0C Buffer are interpreted as unsigned.<br>&nbsp;&nbsp;- You need to first move the ratio data from GM to L1 Buffer, then to the Fixpipe Buffer, and complete the MTE1→FIX synchronization.|
| phase | Input | Optional. Whether to enable unit_flag synchronization during L0C Buffer movement. For details, see [Advanced Cube Computation](../../../../guide/programming_guide/pro/advanced_programming/cube_computation_advanced_usage.md). |

## Constraints

- Data type and fractal constraints:

  | Source to destination | Fractal requirement | Data type requirement |
  |---|---|---|
  | L1 Buffer → L0A Buffer | The source supports ND, NZ, ZN, and ZZ, and the destination is fixed to NZ. | The source and destination must be the same, supporting DT_INT8, DT_FP8E4M3FN, DT_FP8E5M2, DT_HF8, DT_FP16, DT_BF16, DT_FP32, DT_FP4E2M1, DT_FP4E1M2, and DT_FP8E8M0. |
  | L1 Buffer → L0B Buffer | The source supports ND, NZ, ZN, and ZZ, and the destination is fixed to ZN. | The source and destination must be the same, supporting DT_INT8, DT_FP8E4M3FN, DT_FP8E5M2, DT_HF8, DT_FP16, DT_BF16, DT_FP32, DT_FP4E2M1, DT_FP4E1M2, and DT_FP8E8M0. |
  | UB → UB (the shape of the destination Tile is not greater than the source Tile) | The fractal is not checked. | The source and destination must be the same.<br>- Non-ND → NZ: Supports DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_BF16, DT_FP32, DT_FP8E4M3FN, DT_FP8E5M2, DT_FP8E8M0, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2.<br>- ND → NZ: Supports DT_INT8, DT_UINT8, DT_INT32, DT_FP16, DT_BF16, DT_FP32, DT_FP8E4M3FN, DT_FP8E5M2, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2. |
  | UB → UB (the shape of the destination Tile is greater than the source Tile) | ND → ND, NZ → NZ. | The source and destination must be the same, supporting DT_INT8, DT_INT32, DT_FP16, DT_BF16, DT_FP32, DT_FP8E4M3FN, DT_FP8E5M2, DT_FP8E8M0, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2. |
  | UB → L1 Buffer (the shape of the destination Tile is not greater than the source Tile) | The source supports ND and NZ, and the fractal of the destination is not checked. | The source and destination must be the same, supporting DT_INT8, DT_FP8E4M3FN, DT_FP8E5M2, DT_HF8, DT_FP16, DT_BF16, DT_FP32, DT_FP4E2M1, DT_FP4E1M2, and DT_FP8E8M0. |
  | UB → L1 Buffer (the shape of the destination Tile is greater than the source Tile) | The source supports ND and NZ, and the fractal of the destination is not checked. | The source and destination must be the same, supporting DT_INT8, DT_INT32, DT_FP16, DT_BF16, DT_FP32, DT_FP8E4M3FN, DT_FP8E5M2, DT_FP8E8M0, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2. |
  | L1 Buffer → BiasTable Buffer | The destination layout is fixed to ND. | Supports DT_INT32 → DT_INT32, DT_FP32 → DT_FP32, DT_FP16 → DT_FP32, and DT_BF16 → DT_FP32. |
  | L1 Buffer → Fixpipe Buffer | The fractal is not checked. | The destination must be DT_INT64 or DT_UINT64, and the source data type is not restricted. |
  | L1 Buffer → L0A_MX Buffer | The fractals of both the source and destination are ZZ. | The source and destination must be the same, supporting only DT_FP8E8M0. |
  | L1 Buffer → L0B_MX Buffer | The fractals of both the source and destination are NN. | The source and destination must be the same, supporting only DT_FP8E8M0. |
  | L0C Buffer → UB (scale not configured) | NZ → ND, NZ → DN, NZ → NZ. | Supports DT_FP32 → DT_FP32/DT_FP16/DT_BF16, and DT_INT32 → DT_INT32. |
  | L0C Buffer → UB (scale configured) | NZ → ND, NZ → DN, NZ → NZ. | Supports DT_FP32 → DT_INT8/DT_UINT8/DT_HF8/DT_FP16/DT_BF16/DT_FP8E4M3FN/DT_FP32, and DT_INT32 → DT_INT8/DT_UINT8/DT_FP16/DT_BF16. |
  | L0C Buffer → L1 Buffer (only the shape of the destination Tile greater than the source Tile is supported) | NZ → NZ. | Supports DT_FP32 → DT_FP32/DT_FP16/DT_BF16, and DT_INT32 → DT_INT32. |

  UB → L1 Buffer performs only a byte-wise physical copy without layout conversion. When the source UB Tile is ND, it is forbidden to directly receive it with an L1 Tile of layout=NZ/ZN, as this produces incorrect fractal data. You must first convert ND to NZ within UB (UB → UB) using pypto_pro.language.move, and then write it to the NZ/ZN Tile of L1 Buffer. For correct usage, see [Transposing UB Data and Writing It to the L1 Buffer](#transposing-ub-data-and-writing-it-to-the-l1-buffer).<br><br>

  In the tail block scenario, you need to use pypto_pro.language.set_validshape together with the compact parameter in [pypto_pro.language.TileType](../basic_data_structures/TileType.md); otherwise, precision failure or a hang may occur.<br><br>

  For L1 Buffer → L0A_MX Buffer/L0B_MX Buffer, the destination Tile must satisfy L0A_MX Buffer address = L0A Buffer address >> 4 or L0B_MX Buffer address = L0B Buffer address >> 4; otherwise, incorrect quantization coefficients are read during MX matrix multiplication.

## Return Value

None

## Examples

### L1->L0A/L0B

```python
import os
import pypto_pro.language as pl
import torch

@pl.jit(auto_mutex=True)
def kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[128, 32], pl.DT_FP16],
    out: pl.Tensor[[64, 32], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[128, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0, mutex_ids=[2])
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[128, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0, mutex_ids=[3])
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ),
        addrs=0x0, mutex_ids=[4])
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)      # L1 -> L0A
        pl.move(br, cur_b)      # L1 -> L0B
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    a = torch.randn([64, 128], device=device, dtype=torch.float16)
    b = torch.randn([128, 32], device=device, dtype=torch.float16)
    out = torch.zeros([64, 32], device=device, dtype=torch.float32)

    kernel(a, b, out)
    torch.npu.synchronize()

    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    print(f"max diff = {(out - ref).abs().max().item()}")
```

### Transposing UB Data and Writing It to the L1 Buffer

When the left matrix or right matrix is produced in the UB by Vector computation, you can first use pypto_pro.language.move to convert ND to NZ, and then write the result to the L1 Buffer:

- Without transposition, the L1 Buffer Tile uses NZ, and its shape is the same as that of the NZ Tile in the UB.
- With transposition, the L1 Buffer Tile uses ZN, and its two shape dimensions are swapped with those of the NZ Tile in the UB. The source Tile NZ [R, C] and the destination Tile ZN [C, R] represent the same physical fractal.

| Matrix | Transposed | NZ Tile in UB | L1 Buffer Tile |
|---|---|---|---|
| Left matrix A[M, K] | No | shape=[M, K] | shape=[M, K], NZ |
| Left matrix A[M, K] | Yes | shape=[K, M] | shape=[M, K], ZN |
| Right matrix B[K, N] | No | shape=[K, N] | shape=[K, N], NZ |
| Right matrix B[K, N] | Yes | shape=[N, K] | shape=[K, N], ZN |

When writing to the L1 Buffer in blocks, you can specify the write position through offset. The valid region of the source Tile must fall completely within the destination Tile range.

In the following example, vector_result represents the [K, M] data produced by Vector computation. The first move converts it to NZ, and the second move transposes it and writes it to the L1 Buffer, producing [M, K], ZN data for the left matrix.

```python
import pypto_pro.language as pl

M, K = 64, 128

# Output of the preceding Vector computation: ND [K, M] in the UB.
vector_result = pl.make_tile(
    pl.TileType(
        shape=[K, M], dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec, layout=pl.ND,
    ),
    addr=0x0000,
)

# UB Tile used when converting ND to NZ.
vector_nz = pl.make_tile(
    pl.TileType(
        shape=[K, M], dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
    ),
    addr=0x4000,
)

# Write the transposed data to the L1 Buffer with a logical shape of [M, K].
lhs_l1 = pl.make_tile(
    pl.TileType(
        shape=[M, K], dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
    ),
    addr=0x20000,
)

with pl.section_vector():
    pl.move(vector_nz, vector_result)  # ND [K, M] → NZ [K, M]
    pl.move(lhs_l1, vector_nz)         # NZ [K, M] → ZN [M, K]
```

### Using Quantization Parameters

```python
# Method 1: scale is a compile-time constant.
# acc:      Tile in L0C Buffer,   DT_FP32
# vec_tile: UB Tile,    DT_INT8
pl.matmul(acc, q_left, k_right)
pl.move(vec_tile, acc, scale=0.5)        # L0C Buffer -> UB, quantized to INT8 with scale 0.5 on-the-fly.


# Method 2: scale is a runtime scalar.
# When the input parameter is declared as DT_INT32 or DT_INT64, encode the scale into a float32 bit pattern first, as in this example.
# When the input parameter is declared as DT_FP32, pass the scale value directly without encoding.
import struct

@pl.jit()
def kernel(..., scale_bits: pl.DT_INT32):
    # ...
    pl.move(vec_tile, acc, scale=scale_bits)

scale_bits = struct.unpack("!I", struct.pack("!f", 0.5))[0]
kernel(..., scale_bits=scale_bits)


# Method 3: scale is of Tile type.
# fp_mat:   Tile in L1 Buffer,      shape [1, 64], DT_INT64
# fp_tile:  Tile in Fixpipe Buffer, shape [1, 64], DT_INT64
# acc:      Tile in the L0C Buffer,                    DT_INT32
# vec_tile: UB Tile,                     DT_FP16
@pl.jit()
def kernel(..., fp_params: pl.Tensor[[1, 64], pl.DT_INT64]):
    # ...
    pl.load(fp_mat, fp_params, [0, 0])       # GM -> L1
    pl.move(fp_tile, fp_mat)                 # L1 Buffer -> Fixpipe Buffer
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)

    pl.matmul(acc, q_left, k_right)
    pl.move(vec_tile, acc, scale=fp_tile)    # L0C Buffer -> UB, dequantized to DT_FP16 by independent per-column scale

# Case 1: The kernel requires DT_INT64. If a float32 scale tensor is used, convert it first.
import torch_npu

scale_value = 2.0
scale_fp32 = torch.ones(1, 64, dtype=torch.float32, device=device) * scale_value
fp_params = torch_npu.npu_trans_quant_param(scale_fp32)

# Case 2: When the destination Tile data type is DT_INT8, set bit46 of each INT64 element in the Tile in the Fixpipe Buffer to 1.
def _make_scale_tensor(device: str, scale_values: list) -> torch.Tensor:
    scale_bits_list = []
    for scale_value in scale_values:
        scale_bits = struct.unpack("!I", struct.pack("!f", scale_value))[0]
        scale_bits |= 1 << 46  # signed INT8 flag
        scale_bits_list.append(scale_bits)
    return torch.tensor(scale_bits_list, dtype=torch.int64, device=device).reshape(1, 64)

scale_values = [2.0] * 64
fp_params = _make_scale_tensor(device, scale_values)
```
