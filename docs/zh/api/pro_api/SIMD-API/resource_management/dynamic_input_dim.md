# pypto_pro.language.dynamic_input_dim

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

当输入有多个Tensor，各个Tensor维度相同，但个数、形状均不固定时，可使用[约束说明](#约束说明)中的打包方式将其按一定要求组合成动态输入，通过指针记录这多个输入Tensor的起始地址。本接口实现从该动态输入中，读取指定Tensor、指定维度的大小。

接口[pypto_pro.language.make_list_tensor_desc](make_list_tensor_desc.md)实现了打包封装。常与[pypto_pro.language.dynamic_input_ptr](dynamic_input_ptr.md)配合使用，在获取指定Tensor指定维度的大小后，可通过地址访问Tensor数据。

## 函数原型

```python
pypto_pro.language.dynamic_input_dim(x: Ptr, i: int, axis: int, /) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                                                              |
| ------ | --------- | ----------------------------------------------------------------------------------------------------------------- |
| x      | 输入      | 输入指针，类型为[pypto_pro.language.Ptr](../basic_data_structures/Ptr.md)，对多Tensor的打包方式见约束说明。 |
| i      | 输入      | 张量索引，整型常量或运行时整型标量，取值范围[0, N)，N为本次调用传入的Tensor个数。                                 |
| axis   | 输入      | 维度索引，整型常量或运行时整型标量，取值范围[0, dim)，dim为目标Tensor的维度。                                     |

## 约束说明

- 多个Tensor包含形状信息的打包方式示意图见[pypto_pro.language.make_list_tensor_desc](make_list_tensor_desc.md)约束说明，代码实现可见[调用示例](#调用示例)。

## 返回值说明

返回第i个张量第axis维的长度，类型为INT64标量。

## 调用示例

```python
import os

import pypto_pro.language as pl
import torch

MAX_AXIS = 4


@pl.jit(auto_mutex=True)
def dynamic_input_dim_read(
    desc: pl.Ptr[pl.DT_FP32],
    out: pl.Ptr[pl.DT_INT64],
    dim: pl.DT_INT32,
):
    with pl.section_vector():
        out_t = pl.make_tensor(out, [1, 4], [4, 1], dtype=pl.DT_INT64)
        # axis 在 [0, dim) 内取值，越界会在设备侧终止执行
        for axis in pl.range(0, dim):
            pl.setval(out_t, axis, pl.dynamic_input_dim(desc, 1, axis))


def run():
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(0)
    dim, shape0 = 1, 11
    # 两个独立分配的张量，该参数中每个张量只记录1个轴长度
    tensors = [torch.zeros(16, device=device, dtype=torch.float32) for _ in range(2)]
    desc = pl.make_list_tensor_desc([t.data_ptr() for t in tensors], shape=[shape0]).to(device)
    out = torch.full((1, MAX_AXIS), -1, device=device, dtype=torch.int64)
    dynamic_input_dim_read[None, 1](desc, out, dim)
    torch.npu.synchronize()
    got = out.cpu().flatten().tolist()[:dim]
    torch.testing.assert_close(got, [shape0], rtol=0, atol=0)
    print(f"dim={dim} shape0={shape0} PASSED")


if __name__ == "__main__":
    run()
```
