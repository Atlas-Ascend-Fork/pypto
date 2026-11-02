# vf.abs

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:08:14.042Z pushedAt=2026-10-28T11:57:28.743Z nl=2 -->

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

- For real number types (DT_INT8, DT_INT16, DT_INT32, DT_FP16, DT_FP32)

    Computes the absolute value of each valid element in src and writes the result to the corresponding position in dst. The formula is as follows:

    $$dst_i = |src_i|$$

- For complex number types (DT_FP16 dual register mode, DT_FP32 dual register mode)

    Complex number types are implemented through the dual register mode: the DT_FP16 dual register mode corresponds to a complex number composed of two DT_FP16 values (the real part and the imaginary part each occupy 16 bits, totaling 32 bits), and the DT_FP32 dual register mode corresponds to a complex number composed of two DT_FP32 values (the real part and the imaginary part each occupy 32 bits, totaling 64 bits). reg[0] stores the real part, and reg[1] stores the imaginary part.

    Computes the modulus of each valid element in src and writes the result to the corresponding position in dst. The formula is as follows:

    $$dst_i = |src_i| = (\alpha^2 + \beta^2)^{1/2}$$

    where $src_i = \alpha + \beta i$, α is the real part of the complex number, and β is the imaginary part.

## Prototype

```python
abs(src, preg, mode: Optional[MergeMode] = None) -> dst
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| src | Input | Source operand, [reg_tensor](../basic_data_structures/reg_tensor.md). The source operand src and the destination operand dst must have the same data type. Supported data types: DT_INT8, DT_INT16, DT_INT32, DT_FP16, and DT_FP32. DT_FP16 and DT_FP32 support dual register mode for complex number modulus computation. |
| preg | Input | [mask_reg](../basic_data_structures/mask_reg.md). |
| mode | Input | Optional, of the [MergeMode](../basic_data_structures/MergeMode.md) type.<br>- pypto_pro.language.MergeMode.ZEROING (default): Elements not selected by preg are set to 0 in dst.<br>- pypto_pro.language.MergeMode.MERGING is not supported. |

## Constraints

None

## Return Value

Returns the destination operand dst, [reg_tensor](../basic_data_structures/reg_tensor.md), with the same supported data types as described for src. If the computation result of integer data exceeds the representable range of the data type, non-saturating truncation is applied. For example, for DT_INT8, if src is -128, its absolute value 128 is truncated to -128.

## Examples

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    reg_out = vf.abs(reg_a, preg)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.abs(a), rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
