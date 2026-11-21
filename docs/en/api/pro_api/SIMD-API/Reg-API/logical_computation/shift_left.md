# vf.shift_left

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

vf.shift_left performs a left shift on the source operand src according to preg and writes the result into the destination operand dst. The shift amount (shift) can be either a **scalar** (all elements are shifted by the same number of bits) or a **[reg_tensor](../basic_data_structures/reg_tensor.md)** (each element is shifted by the number of bits in its corresponding lane). The API automatically selects the mode based on the type of the shift parameter:

- **Scalar mode** (integer value or scalar variable): All elements are shifted left in a unified manner.

- **reg_tensor mode**: Elements in reg_tensor are shifted left element by element. A left shift operation is classified into the following types based on the data type of the source operand:

- **If the data type is unsigned, a logical left shift is performed.**

  In this case, a left shift moves a binary number to the left by the specified number of bits, discarding the highest bits and filling the lowest bits with zeros. For example, if the binary number 1010101010101010 (DT_UINT16 type) is logically shifted left by one bit, the result is 0101010101010100.
- **If the data type is signed, an arithmetic left shift is performed.**

  Similar to a logical left shift, an arithmetic left shift discards the most significant bits that exceed the bit width of the data type and fills the least significant bits with 0. The difference lies in that the result is interpreted as a signed type. For example, if the binary number 1010101010101010 (DT_INT16 type) is shifted left by 1 bit, the bit pattern becomes 0101010101010100. If it is shifted left by 3 bits, the bit pattern becomes 0101010101010000.

$$
dst_i = src_i \ll shift_i
$$

## Prototype

```python
shift_left(src, shift, preg, /, *, mode: Optional[MergeMode] = None) -> dst
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| src | Input| Source operand, which is of type [reg_tensor](../basic_data_structures/reg_tensor.md). The source operand src and the destination operand dst must have the same data type. For details about the supported data types, see [Constraints](#constraints).|
| shift | Input| Number of bits to shift left. It can be a scalar (integer; all elements are shifted uniformly) or a [reg_tensor](../basic_data_structures/reg_tensor.md) (element-wise shift). For details about the supported data types, see [Constraints](#constraints).<br>- For logical shifts (unsigned data type) in **reg_tensor** mode, if the shift amount is greater than the bit width of the data type, the output is 0.<br>- For arithmetic shifts (signed data type) in **reg_tensor** mode, if the shift amount is greater than the bit width of the data type, the output is 0.<br>- Negative values are not supported in either mode. The behavior of negative values is undefined.|
| preg | Input| [mask_reg](../basic_data_structures/mask_reg.md).|
| mode | Input| (Optional) This parameter corresponds to the [MergeMode](../basic_data_structures/MergeMode.md) type.<br>- pypto_pro.language.MergeMode.ZEROING (default): Elements not selected by preg are set to 0 in dst.<br>- pypto_pro.language.MergeMode.MERGING: Not supported currently.|

## Constraints

- Data type constraints:

  | dst | src | shift |
  | :-- | :-- | :-- |
  | DT_INT8 | DT_INT8 | Scalar mode: integer scalar; reg_tensor mode: DT_INT8|
  | DT_UINT8 | DT_UINT8 | Scalar mode: integer scalar; reg_tensor mode: DT_INT8|
  | DT_INT16 | DT_INT16 | Scalar mode: integer scalar; reg_tensor mode: DT_INT16|
  | DT_UINT16 | DT_UINT16 | Scalar mode: integer scalar; reg_tensor mode: DT_INT16|
  | DT_INT32 | DT_INT32 | Scalar mode: integer scalar; reg_tensor mode: DT_INT32|
  | DT_UINT32 | DT_UINT32 | Scalar mode: integer scalar; reg_tensor mode: DT_INT32|
  | DT_INT64 | DT_INT64 | Scalar mode: integer scalar; reg_tensor mode: DT_INT64|
  | DT_UINT64 | DT_UINT64 | Scalar mode: integer scalar; reg_tensor mode: DT_INT64|

## Return Value

Returns the destination operand dst, which is of type [reg_tensor](../basic_data_structures/reg_tensor.md). For details about the supported data types, see [Constraints](#constraints).

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

### INT64 Data Type Example

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_int64(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT64)
    reg_a = vf.load_align(src_tile, 0)
    reg_out = vf.shift_left(reg_a, 2, preg)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel_int64(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
):
    tf = pl.TileType(shape=[1, 32], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=256, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_int64(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_int64():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(-100, 100, [1, 32], device=device, dtype=torch.int64)
    out = torch.empty([1, 32], device=device, dtype=torch.int64)
    example_kernel_int64[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a << 2, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_int64()
    print("PASSED")
```
