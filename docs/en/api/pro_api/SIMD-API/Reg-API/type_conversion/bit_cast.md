# vf.bit_cast

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T14:14:21.679Z pushedAt=2026-10-28T11:57:29.118Z nl=2 -->

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

Performs a bitwise type cast on a vector register without any numeric conversion.

This API supports two usage modes:

- **Assignment form**: dst = vf.bit_cast(src, dtype=xxx) — Reinterprets src as the dtype type bitwise and assigns it to a newly declared dst register.
- **Nested argument form**: vf.xor(vf.bit_cast(reg_a, dtype=pypto_pro.language.DT_UINT32), vf.bit_cast(reg_b, dtype=pypto_pro.language.DT_UINT32), preg) — Used as an argument to other vf.xxx calls to satisfy the instruction's operand type requirements.

Difference from vf.astype:

- vf.astype performs an actual numeric type conversion (for example, DT_FP32→DT_FP16), in which the element values change.
- vf.bit_cast only changes the type tag, leaving the bit pattern unchanged.

## Prototype

```python
bit_cast(src, dtype: DType) -> dst
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| src | Input | Source operand, [reg_tensor](../basic_data_structures/reg_tensor.md). |
| dtype | Input | Target data type, specifying the data type into which src is reinterpreted. |

## Constraints

- The source operand and the destination operand must have the same bit width (that is, the return values of GetBit() for src and dtype must be identical); otherwise, the behavior is undefined.

## Return Value

Returns the dst destination operand, [reg_tensor](../basic_data_structures/reg_tensor.md). The data type is determined by the dtype parameter.

## Examples

### Assignment Form: FP32 to UINT32

Use the assignment form to reinterpret the FP32 register as UINT32 bitwise, and then perform a bitwise XOR operation.

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_bit_cast_assign(src_tile_a, src_tile_b, dst_tile):
    preg_u32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_a = vf.load_align(src_tile_a, 0)
    reg_b = vf.load_align(src_tile_b, 0)
    reg_a_u32 = vf.bit_cast(reg_a, dtype=pl.DT_UINT32)
    reg_b_u32 = vf.bit_cast(reg_b, dtype=pl.DT_UINT32)
    reg_c = vf.xor(reg_a_u32, reg_b_u32, preg_u32)
    vf.store_align(dst_tile, reg_c, preg_u32)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_b_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    in_b = in_b_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        example_vf_bit_cast_assign(in_a, in_b, t_out)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    b = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, b, out)
    torch.npu.synchronize()
    assert torch.equal(out.view(torch.int32), a.view(torch.int32) ^ b.view(torch.int32))

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### Nested Parameter Form: Reinterpreting HF8 as UINT8 for Bitwise Operations

Reinterpret the DT_HF8 register bitwise as DT_UINT8, and then perform the vf.or_ bitwise OR operation. HF8 is an 8-bit storage type, and after loading, the RegTensor contains 256 elements. vf.bit_cast reinterprets it bitwise as DT_UINT8 (also 256 elements), and vf.or_ performs a bitwise OR on the UINT8 register.

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_hf8_to_uint8(src_tile_a, src_tile_b, dst_tile):
    preg_b8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    reg_a = vf.load_align(src_tile_a, 0, dtype=pl.DT_HF8)
    reg_b = vf.load_align(src_tile_b, 0, dtype=pl.DT_HF8)
    reg_c = vf.or_(vf.bit_cast(reg_a, dtype=pl.DT_UINT8),
                   vf.bit_cast(reg_b, dtype=pl.DT_UINT8), preg_b8)
    vf.store_align(dst_tile, reg_c, preg_b8)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_HF8],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_HF8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_HF8],
):
    tf_in = pl.TileType(shape=[1, 256], dtype=pl.DT_HF8, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_in, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_b_grp = pl.make_tile_group(type=tf_in, addrs=0x100, mutex_ids=[1])
    in_b = in_b_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_in, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        example_vf_hf8_to_uint8(in_a, in_b, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_hf8():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 256], device=device, dtype=torch.float32)
    a = torch_npu.npu_dtype_cast(a, torch_npu.hifloat8)
    b = torch.randn([1, 256], device=device, dtype=torch.float32)
    b = torch_npu.npu_dtype_cast(b, torch_npu.hifloat8)
    out = torch.empty([1, 256], device=device, dtype=torch.uint8)
    example_kernel[None, core_nums](a, b, out)
    torch.npu.synchronize()
    expected = a.view(torch.uint8) | b.view(torch.uint8)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_hf8()
    print("PASSED")
```
