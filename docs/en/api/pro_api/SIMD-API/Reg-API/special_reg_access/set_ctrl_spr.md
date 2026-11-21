# pypto_pro.language.set_ctrl_spr

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

Sets the value of a specified bit range in the CTRL special register. The CTRL register controls multiple global modes for vector computation, including the saturation mode and atomic operation configuration.

## Prototype

```python
set_ctrl_spr(start_bit: int, end_bit: int, value: int, /) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| start_bit | Input| Start bit position of the special register to be set (0–63); a compile-time constant. For configurable bit positions, see [Constraints](#constraints).|
| end_bit | Input| End bit position of the special register to be set (0–63); a compile-time constant.|
| value | Input| Value to be written into the specified bit range of the special register.|

## Constraints

- Writable CTRL bits: 6–10 (range), 45, 48, 50, 53, 59, and 60 (single bits) Other bit positions are not writable. Description of each bit position:

  | Bit Position| Name| Description|
  |---|---|---|
  | 6–10| Atomic configuration| Atomic operation type and mode, which are used to control atomic operations when data is moved from the L0C Buffer, Unified Buffer (UB), or L1 Buffer to Global Memory.<br>- CTRL[8:6]: controls the data type of atomic operations.<br>&nbsp;&nbsp;- 3'b000: Atomic operations are disabled (default).<br>&nbsp;&nbsp;- 3'b001: The data type is DT_FP32.<br>&nbsp;&nbsp;- 3'b010: The data type is DT_FP16.<br>&nbsp;&nbsp;- 3'b011: The data type is DT_INT16.<br>&nbsp;&nbsp;- 3'b100: The data type is DT_INT32.<br>&nbsp;&nbsp;- 3'b101: The data type is DT_INT8.<br>&nbsp;&nbsp;- 3'b110: The data type is DT_BF16.<br>- CTRL[10:9]: controls the atomic operation type. This is valid only when an atomic operation is configured in CTRL[8:6]:<br>&nbsp;&nbsp;- 2'b00: ADD, atomic accumulation operation (default)<br>&nbsp;&nbsp;- 2'b01: MAX, atomic max operation<br>&nbsp;&nbsp;- 2'b10: MIN, atomic min operation.|
  | 45 | matmul configuration| Used to control the register format during matmul operations.<br>- 1'b0: Conversion is performed based on the atomic operand type (default).<br>- 1'b1: If the complex data is entirely of type DT_FP8E4M3FN, it is converted to DT_HF8 before multiplication. Other data is converted based on the atomic operand type.|
  | 48 | Floating-point saturation| Used to control the saturation mode during floating-point data type conversion (valid only when CTRL[60] is configured as global bit mode).<br>- 1'b0: Saturation mode. inf is converted to the maximum value MAX, and NaN is converted to 0 (default).<br>- 1'b1: Non-saturation mode. inf/NaN are handled as-is.<br>This control bit supports only the following data types: DT_FP16 for atomic operations; DT_HF8, DT_FP8E8M0, DT_FP8E5M2, DT_FP8E4M3FN, DT_FP16, and DT_BF16 for floating-point data type conversion.|
  | 50 | Floating-point NaN processing| Used to control the NaN processing mode during floating-point data type conversion (valid only when CTRL[48] is configured as saturation mode).<br>- 1'b0: NaN is converted to 0.0 (default).<br>- 1'b1: NaN remains NaN.<br>This control bit supports only the following data types: DT_FP8E8M0, DT_FP8E5M2, and DT_FP8E4M3FN.|
  | 53 | Integer saturation| Used to control the saturation mode of integer-related instructions.<br>- 1'b0: Truncation mode. The computed value is truncated to the bit width of the destination operand, with higher bits truncated and lower bits retained (default).<br>- 1'b1: Saturation mode. Saturation occurs when the computed value is greater than MAX.|
  | 59 | astype saturation| Used to control the precision conversion saturation mode of floating-point conversion instructions and integer conversion instructions (valid only when CTRL[60] is configured as global bit mode).<br>- 1'b0: Saturation mode. Saturation occurs when the computed value is greater than MAX (default).<br>- 1'b1: Truncation mode. The computed value is truncated to the bit width of the destination operand. The higher bits are truncated and the lower bits are retained.|
  | 60 | Global configuration of the saturation mode| Used to control the global effective mode of the saturation mode.<br>- 1'b0: The single-instruction configuration takes effect.<br>- 1'b1: The global configuration takes effect (default).<br>This control bit can be used together with the precision conversion API such as vf.astype, or with CTRL[48] and CTRL[59].|

- The setting takes effect for all subsequent VF computation instructions until this API is called again or the setting is modified via pypto_pro.language.reset_ctrl_spr.

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
