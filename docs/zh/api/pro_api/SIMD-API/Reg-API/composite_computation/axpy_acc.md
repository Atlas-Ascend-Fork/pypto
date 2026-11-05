# vf.axpy_acc

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->

## 功能说明

该接口根据preg对dst、src和标量scalar按元素执行乘加操作，将结果写入dst。计算公式如下：

$$
dst_i = scalar \times src_i + dst_i
$$

dst作为显式加数参数传入，提供本次累加的旧值；后端将其拷贝到新赋值的dst寄存器后执行vaxpy累加。

## 函数原型

```python
axpy_acc(dst, src, scalar, preg, /, *, mode: Optional[MergeMode] = None) -> dst
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| dst | 输入 | 加数操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)，本次赋值的dst累加的旧值。支持的数据类型和src中的说明一致。 |
| src | 输入 | 源操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)，源操作数src、scalar与目的操作数dst的数据类型保持一致。支持的数据类型为：DT_FP16、DT_FP32、DT_INT64、DT_UINT64。 |
| scalar | 输入 | 标量源操作数。 |
| preg | 输入 | [mask_reg](../basic_data_structures/mask_reg.md)。 - 本接口支持寄存器重叠。 |
| mode | 输入 | 可选，对应[MergeMode](../basic_data_structures/MergeMode.md)类型。<br>- pypto_pro.language.MergeMode.ZEROING（默认），preg未筛选的元素在dst中置0。<br>- pypto_pro.language.MergeMode.MERGING当前不支持。 |

## 约束说明

无。

## 返回值说明

返回dst目标/累加操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)，支持的数据类型和src中的说明一致。计算前作为被加数参与运算，计算后存储结果。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile, out_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    reg_out = vf.load_align(dst_tile, 0)
    reg_out = vf.axpy_acc(reg_out, reg_a, 2.0, preg)
    vf.store_align(out_tile, reg_out, preg)

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
        example_vf(in_a, in_b, t_out)
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
    torch.testing.assert_close(out, 2.0 * a + b, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
