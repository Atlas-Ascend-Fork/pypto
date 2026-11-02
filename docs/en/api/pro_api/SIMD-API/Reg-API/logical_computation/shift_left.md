# vf.shift_left

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T14:01:53.548Z pushedAt=2026-10-28T11:57:29.041Z nl=2 -->

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

The vf.shift_left API shifts the source operand src left based on preg and writes the result to the destination operand dst. The shift amount shift can be either a **scalar** (all elements shift by the same number of bits) or a **[reg_tensor](../basic_data_structures/reg_tensor.md)** (each element shifts by the number of bits of the corresponding lane). The API automatically selects the mode based on the type of the shift parameter:

- **Scalar Mode** (integer value or scalar variable): All elements shift left uniformly.

- **reg_tensor mode**: Element-wise left shift within the reg_tensor. Depending on the data type of the source operand, the left shift operation falls into the following two cases:

- **When the data type is unsigned: A logical left shift is performed.**

  A logical left shift moves the binary number left by the specified number of bits, discards the most significant bit, and fills the least significant bit with 0. For example, after a logical left shift of 1 bit on the binary number 1010101010101010 (DT_UINT16 type), the result is 0101010101010100.
- **When the data type is signed: An arithmetic left shift is performed.**

  Arithmetic left shift is the same as logical left shift in that the highest bit exceeding the data type bit width is discarded and 0 is padded into the lowest bit; the difference lies only in how the result is interpreted as a signed type. For example, after the binary number 1010101010101010 (of the DT_INT16 type) is shifted left by 1 bit, the bit pattern is 0101010101010100; after it is shifted left by 3 bits, the bit pattern is 0101010101010000.

$$
dst_i = src_i \ll shift_i
$$

## Prototype

```python
shift_left(src, shift, preg, mode: Optional[MergeMode] = None) -> dst
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| src | Input | Source operand, [reg_tensor](../basic_data_structures/reg_tensor.md). The source operand src and the destination operand dst must have the same data type. For supported data types, see [Constraints](#constraints). |
| shift | Input | Number of bits to shift left. A scalar (integer, all elements shifted uniformly) or [reg_tensor](../basic_data_structures/reg_tensor.md) (element-wise shift). For supported data types, see [Constraints](#constraints).<br>- For logical shift (unsigned data type) in **reg_tensor mode**, if the shift amount is greater than the data type bit width, the output is 0.<br>- For arithmetic shift (signed data type) in **reg_tensor mode**, if the shift amount is greater than the data type bit width, the output is 0.<br>- Negative values are not supported in either mode; the behavior for negative values is undefined. |
| preg | Input | [mask_reg](../basic_data_structures/mask_reg.md). |
| mode | Input | Optional, corresponding to the [MergeMode](../basic_data_structures/MergeMode.md) type.<br>- pypto_pro.language.MergeMode.ZEROING (default): Elements not selected by preg are set to 0 in dst.<br>- pypto_pro.language.MergeMode.MERGING is not currently supported. |

## Constraints

- Data type constraints:

  | dst | src | shift |
  | :-- | :-- | :-- |
  | DT_INT8 | DT_INT8 | Scalar Mode: Integer Scalar; reg_tensor mode: DT_INT8 |
  | DT_UINT8 | DT_UINT8 | Scalar Mode: Integer Scalar; reg_tensor mode: DT_INT8 |
  | DT_INT16 | DT_INT16 | Scalar Mode: Integer Scalar; reg_tensor mode: DT_INT16 |
  | DT_UINT16 | DT_UINT16 | Scalar Mode: Integer Scalar; reg_tensor mode: DT_INT16 |
  | DT_INT32 | DT_INT32 | Scalar Mode: Integer Scalar; reg_tensor mode: DT_INT32 |
  | DT_UINT32 | DT_UINT32 | Scalar Mode: Integer Scalar; reg_tensor mode: DT_INT32 |
  | DT_INT64 | DT_INT64 | Scalar Mode: Integer Scalar; reg_tensor mode: DT_INT64 |
  | DT_UINT64 | DT_UINT64 | Scalar Mode: Integer Scalar; reg_tensor mode: DT_INT64 |

## Return Value

Returns the destination operand dst, [reg_tensor](../basic_data_structures/reg_tensor.md). For supported data types, see [Constraints](#constraints).

## Examples

### Scalar Mode

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_scalar(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src = vf.load_align(src_tile, 0)
    reg_out = vf.shift_left(reg_src, 4, preg)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_scalar(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(0, 100, [1, 64], device=device, dtype=torch.int32)
    out = torch.empty([1, 64], device=device, dtype=torch.int32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a << 4, rtol=0, atol=0)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### reg_tensor Mode

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_vector(src_tile, shift_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src = vf.load_align(src_tile, 0)
    reg_shift = vf.load_align(shift_tile, 0)
    reg_out = vf.shift_left(reg_src, reg_shift, preg)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel_vector(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    shift: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tf_u32 = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    tf_i32 = pl.TileType(shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_u32, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_shift_grp = pl.make_tile_group(type=tf_i32, addrs=0x100, mutex_ids=[1])
    in_shift = in_shift_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_u32, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_shift, shift, [0, 0])
        example_vf_vector(in_a, in_shift, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_2():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(0, 100, [1, 64], device=device, dtype=torch.int32)
    shift = torch.full([1, 64], 4, device=device, dtype=torch.int32)
    out = torch.empty([1, 64], device=device, dtype=torch.int32)
    example_kernel_vector[None, core_nums](a, shift, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a << 4, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_2()
    print("PASSED")
```
