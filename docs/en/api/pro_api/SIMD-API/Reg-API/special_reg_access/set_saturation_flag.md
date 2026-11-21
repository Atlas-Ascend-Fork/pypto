# pypto_pro.language.set_saturation_flag

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

Sets the saturation mode flag in the CTRL special register. The saturation mode controls the behavior of type conversion instructions, such as vf.astype, when the data exceeds the range of the destination type.

- **Saturation mode (enable=True)**: Data that exceeds the range of the destination type is clamped to the maximum or minimum value of the destination type.
- **Non-saturation mode (enable=False)**: Data that exceeds the range of the destination type is truncated to the lower bits of the destination data width.

The mode configuration is used together with the SaturateMode parameter of vf.astype. For details about the effective rules, see [Cast Saturation Mode Configuration Table](../type_conversion/astype.md#constraints).

## Prototype

```python
set_saturation_flag(*, mode: SaturationFlagMode, enable: bool) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| mode | Input| Saturation mode category, corresponding to the [SaturationFlagMode](../basic_data_structures/SaturationFlagMode.md) enumeration.<br>- pypto_pro.language.SaturationFlagMode.FLOAT: floating-point computation and floating-point precision conversion (CTRL bit 48)<br>- pypto_pro.language.SaturationFlagMode.FLOAT8: floating-point 8 computation (CTRL bit 50)<br>- pypto_pro.language.SaturationFlagMode.INT: integer computation (CTRL bit 53)<br>- pypto_pro.language.SaturationFlagMode.CAST: precision conversion from floating-point to integer or from integer to integer (CTRL bit 59)<br> The setting takes effect for all subsequent VF computation instructions until this API is called again.|
| enable | Input| Saturation mode enable bit. True indicates that the saturation mode is enabled, and False indicates that the saturation mode is disabled.|

## Constraints

- The polarity is inverted for FLOAT/FLOAT8/CAST modes: bit=0 indicates saturation on, and bit=1 indicates saturation off. INT mode uses normal polarity: bit=1 indicates saturation on, and bit=0 indicates saturation off.

- When the saturate parameter of vf.astype is set to pypto_pro.language.SaturateMode.ON or pypto_pro.language.SaturateMode.OFF, per-instruction mode is used (CTRL[60]=0), and the global flag set by this API does not take effect. To make the global saturation mode take effect, ensure that CTRL[60]=1.

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
    pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=True)
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=256, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf(in_a, t_out)
        pl.store(out, t_out, [0, 0])
    pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=False)

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32) * 50000
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    expected = a.clamp(-32768, 32767).to(torch.int16).to(torch.float32)
    torch.testing.assert_close(out, expected, rtol=0, atol=1.0)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
