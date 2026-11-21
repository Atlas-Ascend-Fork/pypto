# vf.gather

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

This instruction gathers the source operand into the destination operand dst according to the index value (index). On the backend, it automatically dispatches into two forms based on the src parameter type:

**Tile→reg** (when src is a Tile): Data is gathered from the Tile into a reg_tensor according to the index. The gather granularity is selected via data_copy_mode:

- pypto_pro.language.DataCopyMode.NORM (default): NORM mode, where data is gathered by element, with index measured in elements. The gather process is shown in the figure below.

  **Figure 1** gather NORM mode

  ![gather NORM mode](../../../figures/gather_function.svg)

- pypto_pro.language.DataCopyMode.DATA_BLOCK_LOAD: DATA_BLOCK_LOAD mode, where data is gathered by DataBlock (32 bytes). The index is expressed in bytes and must be 32-byte aligned. The gather process is shown in the figure below.

  **Figure 2** gather DATA_BLOCK_LOAD mode

  ![gather DATA_BLOCK_LOAD mode](../../../figures/block_mode_gather.svg)

**reg→reg** (when src is a reg_tensor): Data is gathered element by element from one reg_tensor to another, and no mask is required.

> **Differences between the two forms**: In the Tile→reg form, data is read from the Tile, and the 8-bit data types (DT_INT8 and DT_UINT8) are zero-extended to 16 bits. In the reg→reg form, data is read from the reg_tensor, and the source data type remains unchanged.

## Prototype

```python
gather(src, index, preg=None, /, *, data_copy_mode: Optional[DataCopyMode] = None) -> dst
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| src | Input| Source operand, which can be a Tile (Tile→reg form; base address must be 32-byte aligned) or a [reg_tensor](../basic_data_structures/reg_tensor.md) (reg→reg form). For details about the supported data types, see [Constraints](#constraints).<br>- In NORM mode, when src is a 16-bit data type and index is DT_UINT32, each gathered 16-bit element occupies 32 bits (the lower 16 bits hold the data, and the higher 16 bits are zero-filled). The number of valid elements in the register is the index count (VL/4), rather than VL/2 in the case where index is DT_UINT16. When storing, use pypto_pro.language.StoreDist.NORM_B16 to write at 16-bit granularity; even positions in the output contain valid data, and odd positions contain zeros. This feature is suitable for scenarios where the index data is naturally 32 bits wide.<br>- In DATA_BLOCK_LOAD mode and reg→reg form, the data types of the source and destination operands must be the same.|
| index | Input| Index value, which is of type [reg_tensor](../basic_data_structures/reg_tensor.md). For details about the supported data types, see [Constraints](#constraints).<br>- In Tile→reg NORM mode, this is the position of each element in dst relative to src, measured in elements. For 8-bit data types (DT_INT8 and DT_UINT8), source data is zero-extended to 16 bits. For 16-bit source data types (DT_INT16, DT_UINT16, DT_FP16, and DT_BF16), the index supports DT_UINT16 and DT_UINT32.<br>- In Tile→reg DATA_BLOCK_LOAD mode, this is the position of each DataBlock relative to src, and must be 32-byte aligned, that is, one index value corresponds to one DataBlock.<br>- In the reg→reg form, this is the position of each element in src, measured in elements; the data type width must be consistent with src.<br>The data corresponding to the index value must be within the valid address range of the Tile (Tile→reg form). If the index value exceeds the maximum number of data elements that can be stored in the current reg_tensor, the index value is updated to i % (VL/sizeof(T)), where VL is 256 bytes (reg→reg form). Values in index may be repeated.|
| preg | Input| (Optional) [mask_reg](../basic_data_structures/mask_reg.md). The mask function is **supported only in the Tile→reg form**; the reg→reg form does not support this parameter.|
| data_copy_mode | Input| Optional keyword argument. Gather granularity: pypto_pro.language.DataCopyMode.NORM (default, element-wise) or pypto_pro.language.DataCopyMode.DATA_BLOCK_LOAD (by 32-byte DataBlock). This parameter is **supported only in the Tile→reg form**; the reg→reg form does not support this parameter.|

## Constraints

- Data type constraints:

  - **Tile→reg form, NORM mode (element-wise)**

    | dst | src | index |
    |---|---|---|
    | DT_INT16 | DT_INT8 | DT_UINT16 |
    | DT_INT16 | DT_INT16 | DT_UINT16 |
    | DT_UINT16 | DT_UINT8 | DT_UINT16 |
    | DT_UINT16 | DT_UINT16 | DT_UINT16 |
    | DT_FP16 | DT_FP16 | DT_UINT16 |
    | DT_BF16 | DT_BF16 | DT_UINT16 |
    | DT_INT32 | DT_INT32 | DT_UINT32 |
    | DT_UINT32 | DT_UINT32 | DT_UINT32 |
    | DT_FP32 | DT_FP32 | DT_UINT32 |
    | DT_INT64 | DT_INT64 | DT_UINT32 |
    | DT_INT64 | DT_INT64 | DT_UINT64 |
    | DT_UINT64 | DT_UINT64 | DT_UINT32 |
    | DT_UINT64 | DT_UINT64 | DT_UINT64 |
    | DT_INT16 | DT_INT16 | DT_UINT32 |
    | DT_UINT16 | DT_UINT16 | DT_UINT32 |
    | DT_FP16 | DT_FP16 | DT_UINT32 |
    | DT_BF16 | DT_BF16 | DT_UINT32 |

  - **Tile→reg form, DATA_BLOCK_LOAD mode (by 32-byte DataBlock)**

    The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP16, DT_BF16, DT_INT32, DT_UINT32, DT_FP32, DT_INT64, and DT_UINT64.
    The supported data type of the index value is DT_UINT32.

  - **reg→reg form**

    Supported data types: 8-bit (DT_INT8 and DT_UINT8), 16-bit (DT_INT16, DT_UINT16, DT_FP16, and DT_BF16), and 32-bit (DT_INT32, DT_UINT32, and DT_FP32).
    The supported data types of the index value are DT_UINT8, DT_UINT16, and DT_UINT32.

## Return Value

Returns the destination operand dst, which is of type [reg_tensor](../basic_data_structures/reg_tensor.md). For details about the supported data types, see [Constraints](#constraints). In NORM mode, when dst is a 16-bit data type (DT_INT16, DT_UINT16, DT_FP16, or DT_BF16) and src is an 8-bit data type (DT_INT8 or DT_UINT8), the lower 8 bits of the destination operand are the same as those of the source operand, and the higher 8 bits are automatically padded with 0s.

## Examples

### Tile→reg Form (DT_FP32 Data, NORM Mode)

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, index_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    index_reg = vf.load_align(index_tile, 0)
    dst_reg = vf.gather(src_tile, index_reg, preg)
    vf.store_align(dst_tile, dst_reg, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tf_idx = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_idx_grp = pl.make_tile_group(type=tf_idx, addrs=0x100, mutex_ids=[1])
    in_idx = in_idx_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_idx, idx, [0, 0])
        example_vf(in_a, in_idx, t_out)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    idx = torch.arange(64, device=device, dtype=torch.int32).reshape([1, 64])
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, idx, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### Tile→reg Form (DT_FP32 Data, DATA_BLOCK_LOAD Mode)

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_datablock(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_idx = vf.arange(0, dtype=pl.DT_UINT32)
    preg_u = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_idx_b = vf.shift_left(reg_idx, 5, preg_u)
    dst_reg = vf.gather(src_tile, reg_idx_b, preg,
                        data_copy_mode=pl.DataCopyMode.DATA_BLOCK_LOAD)
    vf.store_align(dst_tile, dst_reg, preg)

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
        example_vf_datablock(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_2():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example_2()
    print("PASSED")
```

### Tile→reg Form (DT_INT8 Data, NORM Mode)

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_int8(src_tile, index_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT16)
    index_reg = vf.load_align(index_tile, 0, dtype=pl.DT_UINT16)
    dst_reg = vf.gather(src_tile, index_reg, preg)
    vf.store_align(dst_tile, dst_reg, preg, dist=pl.StoreDist.NORM_B16)

@pl.jit()
def example_kernel_int8(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT16],
):
    tf = pl.TileType(shape=[1, 256], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    tf_idx = pl.TileType(shape=[1, 128], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec)
    tf_out = pl.TileType(shape=[1, 128], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_idx_grp = pl.make_tile_group(type=tf_idx, addrs=0x100, mutex_ids=[1])
    in_idx = in_idx_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_out, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_idx, idx, [0, 0])
        example_vf_int8(in_a, in_idx, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_int8():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(-128, 127, [1, 256], device=device, dtype=torch.int8)
    idx = torch.arange(128, device=device, dtype=torch.int32).reshape([1, 128]).to(torch.uint16)
    out = torch.empty([1, 128], device=device, dtype=torch.int16)
    example_kernel_int8[None, core_nums](a, idx, out)
    torch.npu.synchronize()
    expected = a[:, :128].to(torch.uint8).to(torch.int16)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_int8()
    print("PASSED")
```

### Tile→reg Form (DT_FP16 Data, NORM Mode)

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp16(src_tile, index_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    index_reg = vf.load_align(index_tile, 0, dtype=pl.DT_UINT16)
    dst_reg = vf.gather(src_tile, index_reg, preg)
    vf.store_align(dst_tile, dst_reg, preg)

@pl.jit()
def example_kernel_fp16(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tf = pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tf_idx = pl.TileType(shape=[1, 128], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_idx_grp = pl.make_tile_group(type=tf_idx, addrs=0x100, mutex_ids=[1])
    in_idx = in_idx_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_idx, idx, [0, 0])
        example_vf_fp16(in_a, in_idx, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_fp16():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 128], device=device, dtype=torch.float16)
    idx = torch.arange(128, device=device, dtype=torch.int32).reshape([1, 128]).to(torch.uint16)
    out = torch.empty([1, 128], device=device, dtype=torch.float16)
    example_kernel_fp16[None, core_nums](a, idx, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    test_example_fp16()
    print("PASSED")
```

### Tile→reg Form (DT_INT8 Data, DATA_BLOCK_LOAD Mode)

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_int8_datablock(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT8)
    preg_u32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_idx = vf.arange(0, dtype=pl.DT_UINT32)
    reg_idx_b = vf.shift_left(reg_idx, 5, preg_u32)
    dst_reg = vf.gather(src_tile, reg_idx_b, preg,
                        data_copy_mode=pl.DataCopyMode.DATA_BLOCK_LOAD)
    vf.store_align(dst_tile, dst_reg, preg)

@pl.jit()
def example_kernel_int8_db(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    tf = pl.TileType(shape=[1, 256], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_int8_datablock(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_int8_db():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(-128, 127, [1, 256], device=device, dtype=torch.int8)
    out = torch.empty([1, 256], device=device, dtype=torch.int8)
    example_kernel_int8_db[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_int8_db()
    print("PASSED")
```

### reg→reg Form

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_gather_reg(src_tile, index_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT8)
    src_reg = vf.load_align(src_tile, 0)
    index_reg = vf.load_align(index_tile, 0, dtype=pl.DT_UINT8)
    dst_reg = vf.gather(src_reg, index_reg)
    vf.store_align(dst_tile, dst_reg, preg)

@pl.jit()
def example_kernel_gather_reg(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
):
    tf = pl.TileType(shape=[1, 256], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_idx_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    in_idx = in_idx_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_idx, idx, [0, 0])
        example_vf_gather_reg(in_a, in_idx, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_gather_reg():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(-128, 127, [1, 256], device=device, dtype=torch.int8)
    idx = torch.arange(128, device=device, dtype=torch.int32).to(torch.uint8).reshape([1, 128])
    out = torch.empty([1, 128], device=device, dtype=torch.int8)
    example_kernel_gather_reg[None, core_nums](a, idx, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a[:, :128], rtol=0, atol=0)

if __name__ == "__main__":
    test_example_gather_reg()
    print("PASSED")
```

### Tile→reg Form (DT_FP16 Data, NORM Mode, and DT_UINT32 Index)

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp16_bc(src_tile, index_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    index_reg = vf.load_align(index_tile, 0, dtype=pl.DT_UINT32)
    dst_reg = vf.gather(src_tile, index_reg, preg)
    vf.store_align(dst_tile, dst_reg, preg, dist=pl.StoreDist.NORM_B16)

@pl.jit()
def example_kernel_fp16_bc(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tf = pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tf_idx = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_idx_grp = pl.make_tile_group(type=tf_idx, addrs=0x100, mutex_ids=[1])
    in_idx = in_idx_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_idx, idx, [0, 0])
        example_vf_fp16_bc(in_a, in_idx, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_fp16_bc():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 128], device=device, dtype=torch.float16)
    idx = torch.arange(64, device=device, dtype=torch.int32).reshape([1, 64])
    out = torch.empty([1, 128], device=device, dtype=torch.float16)
    example_kernel_fp16_bc[None, core_nums](a, idx, out)
    torch.npu.synchronize()
    expected = torch.zeros([1, 128], device=device, dtype=torch.float16)
    expected[:, ::2] = a[:, :64]
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    test_example_fp16_bc()
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
    reg_idx = vf.arange(0, dtype=pl.DT_UINT32)
    reg_out = vf.gather(src_tile, reg_idx, preg)
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
    torch.testing.assert_close(out, a, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_int64()
    print("PASSED")
```
