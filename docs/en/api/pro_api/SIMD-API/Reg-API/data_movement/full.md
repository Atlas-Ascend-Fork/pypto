# vf.full

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:44:16.370Z pushedAt=2026-10-28T11:57:28.965Z nl=2 -->

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

Broadcasts a scalar value or the lowest/highest element of the source reg_tensor to each element of the destination reg_tensor dst. Two modes are supported:

- **Scalar mode**: Broadcasts a scalar value to each element of the register.
- **Tensor mode**: Broadcasts the lowest or highest element of the source reg_tensor to each element of the destination reg_tensor. Tensor mode must carry a mask.

## Prototype

```python
full(src, preg, dtype: Optional[DType] = None, mode: Optional[MergeMode] = None, pos: Optional[DuplicatePos] = None) -> dst
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| src | Input | Source operand, which is a scalar value or a [reg_tensor](../basic_data_structures/reg_tensor.md). The source operand src and the destination operand dst must have the same data type.<br>- **Scalar mode**: Scalar value broadcast to each element of the register. Supported data types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP16, DT_BF16, DT_INT32, DT_UINT32, DT_FP32.<br>- **Tensor mode**: [reg_tensor](../basic_data_structures/reg_tensor.md), whose lowest or highest element is broadcast. Supported data types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP16, DT_BF16, DT_INT32, DT_UINT32, DT_FP32, DT_FP8E4M3FN, DT_FP8E5M2, DT_FP8E8M0, DT_HF8, DT_FP4E2M1, DT_FP4E1M2. |
| preg | Input | [mask_reg](../basic_data_structures/mask_reg.md). Required in Tensor mode; optional in Scalar mode. |
| dtype | Input | Optional. Specifies the data type. Required in Scalar mode; in Tensor mode, it can be automatically inferred from the source register. |
| mode | Input | Optional. Corresponds to the [MergeMode](../basic_data_structures/MergeMode.md) type.<br>- pypto_pro.language.MergeMode.ZEROING (default): Elements not selected by preg are set to 0 in dst.<br>- pypto_pro.language.MergeMode.MERGING is not currently supported. |
| pos | Input | Optional. In Tensor mode, selects which element of the source reg_tensor to broadcast, corresponding to the [DuplicatePos](../basic_data_structures/DuplicatePos.md) type:<br>- pypto_pro.language.DuplicatePos.LOWEST: Default. Broadcasts the lowest element.<br>- pypto_pro.language.DuplicatePos.HIGHEST: Broadcasts the highest element. |

## Constraints

None

## Return Value

Returns the destination operand dst, a [reg_tensor](../basic_data_structures/reg_tensor.md). The supported data types are the same as those described for src.

## Examples

### Scalar Mode

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    max0 = vf.full(3.0, dtype=pl.DT_FP32)
    sum0 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, max0, preg)

@pl.jit()
def example_kernel(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    t_out = t_out_grp.current()
    with pl.section_vector():
        example_vf(t_out)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.full([1, 64], 3.0, device=device, dtype=torch.float32), rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### Tensor Mode — FP8E4M3FN Register Broadcast

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp8(src_tile, dst_tile):
    preg_f8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP8E4M3FN)
    preg_f32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_f8 = vf.load_align(src_tile, 0, dtype=pl.DT_FP8E4M3FN)
    reg_dup = vf.full(reg_f8, preg_f8)
    reg_f32 = vf.astype(reg_dup, preg_f32, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_f32, preg_f32)

@pl.jit()
def example_kernel_fp8(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP8E4M3FN],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf_in = pl.TileType(shape=[1, 256], dtype=pl.DT_FP8E4M3FN, target_memory=pl.MemorySpace.Vec)
    tf_out = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_in, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_out, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_fp8(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_fp8():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 256], device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel_fp8[None, core_nums](a, out)
    torch.npu.synchronize()
    expected = a[:, :1].to(torch.float32).expand([1, 64])
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    test_example_fp8()
    print("PASSED")
```

### Tensor Mode - FP4E1M2 Register Broadcast

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp4(src_tile, dst_tile):
    preg_f4 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP4E1M2)
    preg_bf16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    reg_f4 = vf.load_align(src_tile, 0, dtype=pl.DT_FP4E1M2)
    reg_dup = vf.full(reg_f4, preg_f4)
    reg_bf16 = vf.astype(reg_dup, preg_bf16, dtype=pl.DT_BF16)
    vf.store_align(dst_tile, reg_bf16, preg_bf16)

@pl.jit()
def example_kernel_fp4(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
):
    tf_in = pl.TileType(shape=[1, 256], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    tf_out = pl.TileType(shape=[1, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_in, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_out, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_fp4(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_fp4():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(0, 256, [1, 256], device=device, dtype=torch.uint8)
    a[:, 0] = 0x44
    out = torch.empty([1, 128], device=device, dtype=torch.bfloat16)
    example_kernel_fp4[None, core_nums](a, out)
    torch.npu.synchronize()
    expected = torch.ones([1, 128], device=device, dtype=torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_fp4()
    print("PASSED")
```

### Tensor Mode — HF8 Register Broadcast

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_hf8(src_tile, dst_tile):
    preg_hf8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_HF8)
    preg_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    reg_hf8 = vf.load_align(src_tile, 0, dtype=pl.DT_HF8)
    reg_dup = vf.full(reg_hf8, preg_hf8)
    reg_f16 = vf.astype(reg_dup, preg_f16, dtype=pl.DT_FP16)
    vf.store_align(dst_tile, reg_f16, preg_f16)

@pl.jit()
def example_kernel_hf8(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_HF8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tf_in = pl.TileType(shape=[1, 256], dtype=pl.DT_HF8, target_memory=pl.MemorySpace.Vec)
    tf_out = pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_in, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_out, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_hf8(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_hf8():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.ones([1, 256], device=device, dtype=torch.float32)
    a = torch_npu.npu_dtype_cast(a, torch_npu.hifloat8)
    out = torch.empty([1, 128], device=device, dtype=torch.float16)
    example_kernel_hf8[None, core_nums](a, out)
    torch.npu.synchronize()
    expected = torch.ones([1, 128], device=device, dtype=torch.float16)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    test_example_hf8()
    print("PASSED")
```

### Tensor Mode — FP8E8M0 Register Broadcast

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp8e8m0(src_tile, dst_tile):
    preg_f8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP8E8M0)
    reg_f8 = vf.load_align(src_tile, 0, dtype=pl.DT_FP8E8M0)
    reg_dup = vf.full(reg_f8, preg_f8)
    vf.store_align(dst_tile, reg_dup, preg_f8)

@pl.jit()
def example_kernel_fp8e8m0(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
):
    tf_in = pl.TileType(shape=[1, 256], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    tf_out = pl.TileType(shape=[1, 256], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_in, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_out, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_fp8e8m0(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_fp8e8m0():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.zeros([1, 256], device=device, dtype=torch.uint8)
    a[:, 0] = 0x7E
    out = torch.empty([1, 256], device=device, dtype=torch.uint8)
    example_kernel_fp8e8m0[None, core_nums](a, out)
    torch.npu.synchronize()
    expected = torch.full([1, 256], 0x7E, device=device, dtype=torch.uint8)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_fp8e8m0()
    print("PASSED")
```
