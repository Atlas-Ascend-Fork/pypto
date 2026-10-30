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

常与[pypto_pro.language.dynamic_input_ptr](dynamic_input_ptr.md)配合使用，在获取指定Tensor指定维度的大小后，可通过地址访问Tensor数据。

## 函数原型

```python
pypto_pro.language.dynamic_input_dim(x: Ptr, i: int, axis: int, /) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                                                              |
| ------ | --------- | ----------------------------------------------------------------------------------------------------------------- |
| x      | 输入      | 输入指针，类型为[pypto_pro.language.Ptr](../basic_data_structures/Ptr.md)[dtype]，对多Tensor的打包方式见约束说明。 |
| i      | 输入      | 张量索引，整型常量或运行时整型标量，取值范围[0, N)，N为本次调用传入的Tensor个数。                                 |
| axis   | 输入      | 维度索引，整型常量或运行时整型标量，取值范围[0, dim)，dim为目标Tensor的维度。                                     |

## 约束说明

- 多个Tensor包含形状信息的打包方式示意图如下，代码实现可见[调用示例](#调用示例)：

  ```text
           ┌──────────────┐
      w[0] │    offset    │  地址数组的字节偏移 = (1 + 2N) * 8
           ├──────────────┤
      w[1] │ dim | count  │  第0个结构的元信息字：低32 bit为dim(=1)，高32 bit为N
      w[2] │    shape     │  第0个Tensor的第0轴长度
      w[3] │      0       │  第1个结构的元信息字（填0）
      w[4] │    shape     │  第1个Tensor的第0轴长度
       ... │     ...      │  ...
           ├──────────────┤
   w[1+2N] │    addr_0    │  第0个Tensor的起始地址
   w[2+2N] │    addr_1    │  第1个Tensor的起始地址
       ... │     ...      │  ...
     w[3N] │  addr_{N-1}  │  第N-1个Tensor的起始地址
           └──────────────┘

  以N=2、dim=1、shape=11为例，该数组为[40, (2 << 32) | 1, 11, 0, 11, addr_0, addr_1]。
  ```

## 返回值说明

返回第i个张量第axis维的长度，类型为INT64标量。

## 调用示例

```python
import os

import pypto
import pypto_pro.language as pl
import torch

MAX_AXIS = 4


def _build_list_tensor_desc_dim1(addrs, shape0):
    """dim为1的动态输入：[dataPtrOffset, N个(1+dim)=2字的形状结构, addr_0..addr_{N-1}]。

    取址仍读连续的地址数组；形状块位于该参数起始处和地址数组之间，因此地址数组的偏移随N变化。
    """
    n = len(addrs)
    struct_sz = 2  # 1（元信息字）+ dim（=1）
    data_ptr_offset = 8 + n * struct_sz * 8
    w = [0] * (1 + n * struct_sz + n)
    w[0] = data_ptr_offset
    w[1] = (1 & 0xFFFFFFFF) | (n << 32)  # 第0个结构的元信息字：低32bit dim=1，高32bit count
    w[2] = shape0  # 第0个结构的shape[0]
    for i in range(1, n):
        w[1 + i * struct_sz] = 0  # 其余结构的元信息字，填0即可
        w[1 + i * struct_sz + 1] = shape0
    addr_off = data_ptr_offset // 8
    for i, a in enumerate(addrs):
        w[addr_off + i] = a
    return torch.tensor(w, dtype=torch.int64)


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


@pypto.options(pass_options={"enable_slice": False})
def run():
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(0)
    dim, shape0 = 1, 11
    # 两个独立分配的张量，该参数中每个张量只记录1个轴长度
    tensors = [torch.zeros(16, device=device, dtype=torch.float32) for _ in range(2)]
    desc = _build_list_tensor_desc_dim1([t.data_ptr() for t in tensors], shape0).to(device)
    out = torch.full((1, MAX_AXIS), -1, device=device, dtype=torch.int64)
    dynamic_input_dim_read[None, 1](desc, out, dim)
    torch.npu.synchronize()
    got = out.cpu().flatten().tolist()[:dim]
    torch.testing.assert_close(got, [shape0], rtol=0, atol=0)
    print(f"dim={dim} shape0={shape0} PASSED")


if __name__ == "__main__":
    run()
```
