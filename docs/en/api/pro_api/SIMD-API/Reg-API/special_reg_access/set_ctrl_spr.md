# pypto_pro.language.set_ctrl_spr

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T14:13:31.974Z pushedAt=2026-10-28T11:57:29.111Z nl=2 -->

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

Sets the value of a specified bit range in the CTRL special register. The CTRL register controls multiple global modes of vector computation, including saturation mode and atomic operation configuration.

## Prototype

```python
set_ctrl_spr(start_bit: int, end_bit: int, value: int, /) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| start_bit | Input | Start bit (0-63) of the special register to be set, a compile-time constant. For configurable bits, see [Constraints](#constraints). |
| end_bit | Input | End bit (0-63) of the special register to be set, a compile-time constant. |
| value | Input | Value to be written to the specified bit range of the special register. |

## Constraints

- Writable CTRL bits: 6-10 (range), 45, 48, 50, 53, 59, 60 (single bits). Other bits are not writable. Description of each bit:

  | Bit | Name | Description |
  |---|---|---|
  | 6-10 | Atomic configuration | Atomic operation type and mode, used to control the atomic operation when data is moved from the L0C Buffer/Unified Buffer (UB)/L1 Buffer to Global Memory.<br>- CTRL[8:6], controls the data type of the atomic operation:<br>&nbsp;&nbsp;- 3'b000: Atomic operation disabled (default).<br>&nbsp;&nbsp;- 3'b001: Data type DT_FP32.<br>&nbsp;&nbsp;- 3'b010: Data type DT_FP16.<br>&nbsp;&nbsp;- 3'b011: Data type DT_INT16.<br>&nbsp;&nbsp;- 3'b100: Data type DT_INT32.<br>&nbsp;&nbsp;- 3'b101: Data type DT_INT8.<br>&nbsp;&nbsp;- 3'b110: Data type DT_BF16.<br>- CTRL[10:9], controls the atomic operation type, effective only when CTRL[8:6] configures an atomic operation:<br>&nbsp;&nbsp;- 2'b00: ADD, accumulation atomic operation (default).<br>&nbsp;&nbsp;- 2'b01: MAX, maximum-value atomic operation.<br>&nbsp;&nbsp;- 2'b10: MIN, minimum-value atomic operation. |
  | 45 | matmul configuration | Controls the register format during matmul operations.<br>- 1'b0: Convert according to the atomic operand type (default).<br>- 1'b1: When the complex data is DT_FP8E4M3FN, convert it to DT_HF8 before multiplication; other data is converted according to the atomic operand type. |
  | 48 | Floating-point saturation | Controls the saturation mode during floating-point data type conversion (effective only when CTRL[60] is configured as the global bit mode).<br>- 1'b0: Saturation mode. inf is converted to the maximum MAX, and NaN is converted to 0 (default).<br>- 1'b1: Non-saturation mode. inf/NaN are processed as-is.<br>This control bit supports only the following data types: DT_FP16 for atomic operations; DT_HF8, DT_FP8E8M0, DT_FP8E5M2, DT_FP8E4M3FN, DT_FP16, and DT_BF16 for floating-point data type conversion. |
  | 50 | Floating-point NaN handling | Controls the NaN handling mode during floating-point data type conversion (effective only when CTRL[48] is configured as saturation mode).<br>- 1'b0: NaN is converted to 0.0 (default).<br>- 1'b1: NaN remains NaN.<br>This control bit supports only the following data types: DT_FP8E8M0, DT_FP8E5M2, and DT_FP8E4M3FN. |
  | 53 | Integer saturation | Controls the saturation mode of integer-related instructions.<br>- 1'b0: Truncation mode. The computed value is truncated to the destination operand bit width, with high bits truncated and low bits retained (default).<br>- 1'b1: Saturation mode. The computed value saturates when it is greater than MAX. |
  | 59 | astype saturation | Controls the precision conversion saturation mode of floating-point conversion instructions and integer conversion instructions (effective only when CTRL[60] is configured as the global bit mode).<br>- 1'b0: Saturation mode. The computed value saturates when it is greater than the maximum MAX (default).<br>- 1'b1: Truncation mode. The computed value is truncated to the destination operand bit width, with high bits truncated and low bits retained. |
  | 60 | Saturation mode global configuration | Controls how the saturation mode takes effect globally.<br>- 1'b0: Single-instruction configuration takes effect.<br>- 1'b1: Global configuration takes effect (default).<br>This control bit can be used together with precision conversion APIs such as vf.astype, and with CTRL[48] and CTRL[59]. |

- After being set, it takes effect on all subsequent VF computation instructions until this API is called again or pypto_pro.language.reset_ctrl_spr is used to modify it.

## Return Value

None

## Examples

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    reg_i16 = vf.astype(reg, preg, dtype=pl.DT_INT16, layout=pl.CastLayout.ZERO)
    reg_f32 = vf.astype(reg_i16, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_f32, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    pl.set_ctrl_spr(60, 60, 1)
    pl.set_ctrl_spr(59, 59, 0)
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=256, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf(in_a, t_out)
        pl.store(out, t_out, [0, 0])
    pl.reset_ctrl_spr(59, 59)
    pl.reset_ctrl_spr(60, 60)

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32) * 50000
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, 1](a, out)
    torch.npu.synchronize()
    expected = a.clamp(-32768, 32767).to(torch.int16).to(torch.float32)
    torch.testing.assert_close(out, expected, rtol=0, atol=1.0)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
