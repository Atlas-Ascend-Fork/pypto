# vf.muls_cast

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:42:47.585Z pushedAt=2026-10-28T11:57:28.954Z nl=2 -->

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

This API multiplies the source operand src by the scalar scalar, converts the result to the DT_FP16 type according to layout, and writes the computation result to the destination operand dst based on preg. The computation formula is as follows:

$$
dst_i = cast\_round\_to\_f16(src_i \times scalar)
$$

## Prototype

```python
muls_cast(src, scalar, preg, dtype: DType, layout: Optional[CastLayout] = None) -> dst
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| src | Input | Source operand, [reg_tensor](../basic_data_structures/reg_tensor.md). For supported data types, see [Constraints](#constraints). |
| scalar | Input | Scalar source operand. |
| preg | Input | [mask_reg](../basic_data_structures/mask_reg.md). |
| dtype | Input | Data type of the target reg_tensor. For supported data types, see [Constraints](#constraints). Because a type conversion (DT_FP32→DT_FP16) is performed after the multiplication, the target type differs from the source type and must be explicitly specified, typically pypto_pro.language.DT_FP16. |
| layout | Input | Optional. Half in which the result is placed: pypto_pro.language.CastLayout.ZERO (even half, default, PART_EVEN) or pypto_pro.language.CastLayout.ONE (odd half, PART_ODD), corresponding to the [CastLayout](../basic_data_structures/CastLayout.md) type. The computation is rounded in CAST_ROUND mode. |

## Constraints

- Data type constraints:

  | src | scalar | dst |
  |---|---|---|
  | DT_FP32 | DT_FP32 | DT_FP16 |

- Overlap between the source operand register and the destination operand register is not supported.

## Return Value

Returns the destination operand dst, a [reg_tensor](../basic_data_structures/reg_tensor.md). For supported data types, see [Constraints](#constraints).

## Examples

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src_tile, 0)
    reg_f16 = vf.muls_cast(reg_src, 2.0, preg, dtype=pl.DT_FP16)
    reg_out = vf.astype(reg_f16, preg, dtype=pl.DT_FP32)
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
    expected = (a * 2.0).to(torch.float16).to(torch.float32)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
