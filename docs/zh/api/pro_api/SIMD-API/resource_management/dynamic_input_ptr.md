# pypto_pro.language.dynamic_input_ptr

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

当输入有多个Tensor，个数、形状均不固定时，可使用[约束说明](#约束说明)中的打包方式将其按一定要求组合成动态输入，通过指针记录这多个输入Tensor的起始地址。本接口实现从该动态输入中，读取指定Tensor的起始地址，返回指针的元素类型由声明的dtype决定。

接口[pypto_pro.language.make_list_tensor_desc](make_list_tensor_desc.md)实现了打包封装。常用于按顺序遍历全部张量，配合[pypto_pro.language.make_tensor](make_tensor.md)把每次取出的地址包装成Tensor视图，读取每个张量的形状使用[pypto_pro.language.dynamic_input_dim](dynamic_input_dim.md)。

## 函数原型

```python
pypto_pro.language.dynamic_input_ptr(x: Ptr, i: int, /) -> Ptr
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                                                              |
| ------ | --------- | ----------------------------------------------------------------------------------------------------------------- |
| x      | 输入      | 输入指针，类型为[pypto_pro.language.Ptr](../basic_data_structures/Ptr.md)[dtype]，对多Tensor的打包方式见约束说明。 |
| i      | 输入      | 张量索引，整型常量或运行时整型标量，取值范围[0, N)，N为本次调用传入的Tensor个数。                                 |

## 约束说明

- 多个Tensor不包含形状信息的打包方式示意图见[pypto_pro.language.make_list_tensor_desc](make_list_tensor_desc.md)约束说明，代码实现可见[调用示例](#调用示例)。

## 返回值说明

返回第i个张量的地址，类型为pypto_pro.language.Ptr，元素类型与各张量一致。

## 调用示例

```python
import os

import pypto_pro.language as pl
import torch

P = 256


@pl.jit(auto_mutex=True)
def dynamic_input_ptr_sum(
    desc: pl.Ptr[pl.DT_FP32],
    out: pl.Ptr[pl.DT_FP32],
):
    out_t = pl.make_tensor(out, [1, P], [P, 1], dtype=pl.DT_FP32)
    tt = pl.TileType(shape=[1, P], dtype=pl.DT_FP32,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    src_db = pl.make_tile_group(type=tt, addrs=[0x0000, 0x1000], mutex_ids=[0, 1])
    acc_db = pl.make_tile_group(type=tt, addrs=[0x2000], mutex_ids=[2])
    with pl.section_vector():
        acc = acc_db.next()
        pl.set_validshape(acc, [1, P])
        # 取第 0 个张量的地址，包装成 Tensor 视图后搬入 UB
        p0 = pl.dynamic_input_ptr(desc, 0)
        pl.load(acc, pl.make_tensor(p0, [1, P], [P, 1], dtype=pl.DT_FP32), [0, 0])
        # 取第 1 个张量的地址，搬入 UB 后累加到第 0 个张量上
        t = src_db.next()
        pl.set_validshape(t, [1, P])
        p1 = pl.dynamic_input_ptr(desc, 1)
        pl.load(t, pl.make_tensor(p1, [1, P], [P, 1], dtype=pl.DT_FP32), [0, 0])
        pl.add(acc, acc, t)
        pl.store(out_t, acc, [0, 0])


def run():
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(0)
    # 两个张量独立分配（地址不连续），全部由该参数索引
    tensors = [torch.randn(P, device=device, dtype=torch.float32) for _ in range(2)]
    desc = pl.make_list_tensor_desc([t.data_ptr() for t in tensors]).to(device)
    out = torch.full((1, P), -1.0, device=device, dtype=torch.float32)
    dynamic_input_ptr_sum[None, 1](desc, out)
    torch.npu.synchronize()
    ref = torch.stack(tensors).sum(0, keepdim=True)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)
    print("PASSED")


if __name__ == "__main__":
    run()
```
