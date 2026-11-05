# vf.move_acc

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

该接口按掩码preg将源操作数src中的元素移动到目标操作数dst的对应位置。

支持reg_tensor和mask_reg两种寄存器类型：

- **reg_tensor模式**：dst为[reg_tensor](../basic_data_structures/reg_tensor.md)，preg选中的位置将src中对应元素拷贝到dst，preg未选中的位置保留dst的旧值。
- **mask_reg模式**：dst为[mask_reg](../basic_data_structures/mask_reg.md)，preg选中的bit将src中对应bit拷贝到dst，preg未选中的bit保留dst的旧值。

## 函数原型

```python
move_acc(dst, src, preg, /, *, mode: Optional[MergeMode] = None) -> dst
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| dst | 输入 | 旧值来源操作数（本次赋值前的dst值），[reg_tensor](../basic_data_structures/reg_tensor.md)或[mask_reg](../basic_data_structures/mask_reg.md)类型，须与src的类型一致。支持的数据类型和src中的说明一致。 |
| src | 输入 | 源操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)或[mask_reg](../basic_data_structures/mask_reg.md)类型，须与dst的类型一致。支持的数据类型为：DT_BOOL、DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64。 |
| preg | 输入 | [mask_reg](../basic_data_structures/mask_reg.md)，指定哪些元素/bit参与拷贝。<br>- **reg_tensor模式**：preg选中的位置，将src中对应元素拷贝到dst；preg未选中的位置，dst保留旧值。<br>- **mask_reg模式**：preg选中的bit，将src中对应bit拷贝到dst；preg未选中的bit，dst保留旧值。 |
| mode | 输入 | 可选，对应[MergeMode](../basic_data_structures/MergeMode.md)类型。<br>- pypto_pro.language.MergeMode.MERGING（默认）：preg未选中的元素在dst中保留旧值。<br>- pypto_pro.language.MergeMode.ZEROING：当前**不支持**。 |

## 约束说明

- dst为本次移动的旧值来源寄存器，通常取赋值目标变量在上一次绑定时的值；接口先用该旧值初始化新dst，再按preg用src覆盖选中位置。

## 返回值说明

返回dst目的操作数，[reg_tensor](../basic_data_structures/reg_tensor.md)或[mask_reg](../basic_data_structures/mask_reg.md)类型，支持的数据类型和src中的说明一致。

## 调用示例

```python
import os
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_half = vf.update_mask(32, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src_tile, 0)
    reg_old = vf.full(1.0, preg_all)
    reg_dst = vf.move_acc(reg_old, reg_src, preg_half)
    vf.store_align(dst_tile, reg_dst, preg_all)

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
    expected = torch.ones_like(a)
    expected[:, :32] = a[:, :32]
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
