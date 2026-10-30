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

获取运行时变长（DYNAMIC）输入端口描述符中第i个张量的地址。DYNAMIC输入端口在Host侧挂接N个张量，N在运行时确定；框架将这N个张量的列表描述符（list-tensor descriptor）传给Kernel，本接口按张量下标从描述符中取出对应张量的GM地址。

返回值是一个裸指针pypto_pro.language.Ptr，其元素类型取描述符声明的dtype。通常先配合[pypto_pro.language.make_tensor](make_tensor.md)构造Tensor视图，再像普通GM Tensor一样被[load](../memory_data_movement/load.md)/[store](../memory_data_movement/store.md)使用。若需要按其它元素类型查看同一段地址（例如当作原始UINT8字节），在[pypto_pro.language.make_tensor](make_tensor.md)的dtype参数中指定即可，本接口返回的指针元素类型默认即为描述符的元素类型。

## 函数原型

```python
pypto_pro.language.dynamic_input_ptr(x: Ptr, i: int, /) -> Ptr
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                                                                                                                                                       |
| ---- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| x    | 输入      | DYNAMIC输入端口的描述符指针，在Kernel签名中声明为[pypto_pro.language.Ptr](../basic_data_structures/Ptr.md)[dtype]，元素类型即描述符声明的dtype。 |
| i    | 输入      | 张量下标，取值为[0, N)，N为Host侧挂接的张量个数；可以是编译期常量或运行时整型标量表达式。                                                                                                                  |

## 约束说明

- x是DYNAMIC输入端口的描述符（pl.Ptr[dtype]）；其指向的内存必须确实是框架传递的列表描述符，否则会按描述符布局解读，属未定义行为。
- i必须落在[0, N)内，接口不对i做范围检查；越界会读取描述符地址表之外的内存，属未定义行为。
- 各张量可以独立分配，地址不必连续；张量个数与形状由描述符及Host侧共同提供。
- 元素类型的C类型映射：8/16/32/64位元素使用对应的C类型；4位元素按两个值一个字节打包（DT_FP4E2M1对应float4_e2m1x2_t、DT_FP4E1M2对应float4_e1m2x2_t），DT_INT4/DT_UINT4/DT_FP4/DT_HF4没有对应的打包C类型，不可在此使用。

## 返回值说明

返回描述符中第i个张量的地址，类型为pypto_pro.language.Ptr，元素类型为描述符声明的dtype。

## 调用示例

下面的Kernel消费一个DYNAMIC输入端口，按顺序读取其中的count个张量并逐元素求和。

```python
import pypto_pro.language as pl

P = 256


@pl.jit(auto_mutex=True)
def dynamic_input_ptr_sum(
    desc: pl.Ptr[pl.DT_FP32],
    out: pl.Ptr[pl.DT_FP32],
    count: pl.DT_INT32,
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
        for i in pl.range(1, count):
            t = src_db.next()
            pl.set_validshape(t, [1, P])
            pi = pl.dynamic_input_ptr(desc, i)
            pl.load(t, pl.make_tensor(pi, [1, P], [P, 1], dtype=pl.DT_FP32), [0, 0])
            pl.add(acc, acc, t)
        pl.store(out_t, acc, [0, 0])
```

如需把某个张量按原始字节查看，可在make_tensor中指定dtype：

```python
p0 = pl.dynamic_input_ptr(desc, 0)
# 描述符声明为 pl.Ptr[pl.DT_FP32]，一个元素占4字节，这里按 UINT8 重新解释同一段地址
raw = pl.make_tensor(p0, [1, P * 4], [P * 4, 1], dtype=pl.DT_UINT8)
```
