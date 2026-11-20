# pypto_pro.language.img2col

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

将L1 Buffer中的一个卷积特征图窗口按卷积核窗口几何展开成L0A矩阵（im2col展开），供后续[pypto_pro.language.matmul](../cube_computation/matmul.md)或[pypto_pro.language.matmul_acc](../cube_computation/matmul_acc.md)使用。

展开语义与数学卷积一致：目的矩阵的第r行为一个输出位置，第c列对应卷积核窗口内的一个输入元素。行由（Ho, Wo）枚举（Ho×Wo = 输出高×宽），列由（C1, Kh, Kw, C0）枚举。带padding时，越界的窗口位置填充``pad_value``。

目的矩阵形状由源窗口与窗口几何推导：

```text
M = align16(Ho × Wo)        Ho = (H + padTop + padBottom - dilatedKh) // strideH + 1
K = C1 × Kh × Kw × C0       dilatedKh = (Kh - 1) × dilationH + 1
```

其中``align16``表示M按16行向上对齐（L0A分形要求），对齐补齐的行不参与后续计算。展开写入的目的范围跟随目的Tile的运行时有效形状（valid shape），可用[pypto_pro.language.set_validshape](../transpose_and_element_access/set_validshape.md)在尾块场景下调整。

通道深度不由config配置：硬件读取的``channelSize``由源Tile的``C1 × C0``推导得出。

3D卷积的D维（深度）循环由调用方实现：每次调用处理一个深度切片，源窗口使用NDC1HWC0布局（D=1）。

## 函数原型

```python
pypto_pro.language.img2col(
    dst_tile: Tile,
    src_tile: Tile,
    /,
    *,
    config: Img2ColConfig,
    pos_m: int,
    pos_k: int,
) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| dst_tile | 输出 | 目的操作数，Tile类型，存储空间为L0A（MemorySpace.Left），布局NZ，数据类型与源一致（DT_FP16、DT_BF16）。shape须为推导结果``[M, K]``，行数（M）须为16的倍数。 |
| src_tile | 输入 | 源操作数，Tile类型，存储空间为L1 Buffer（MemorySpace.Mat）的卷积Tile，保存一个特征图窗口：NC1HWC0布局shape为``[1, C1, H, W, C0]``，NDC1HWC0布局shape为``[1, 1, C1, H, W, C0]``（D循环由调用方实现，每次一个深度切片）。 |
| config | 输入 | 窗口几何，[pypto_pro.language.Img2ColConfig](../basic_data_structures/Img2ColConfig.md)类型，编译期常量。 |
| pos_m | 输入 | 行枚举偏移：本次调用生成的第一行在（Ho, Wo）枚举中的元素序号，非负整数。窗口几何不变、分次生成多个目的块时使用。 |
| pos_k | 输入 | 列枚举偏移：本次调用生成的第一列在（C1, Kh, Kw, C0）枚举中的元素序号，非负整数。 |

## 约束说明

- 源窗口必须恰好保存一个特征图窗口：NC1HWC0的N维必须为1；NDC1HWC0的N维和D维必须均为1（3D卷积的D循环由调用方实现，一次一个深度切片）。
- 源Tile的C0必须等于``32B / sizeof(dtype)``（DT_FP16、DT_BF16时为16）。
- 源Tile的通道深度``C1 × C0``须在[1, 65535]范围内（img2col硬件寄存器位宽）。
- 目的Tile须为静态shape的二维Tile，行数须为16的倍数（L0A分形要求），shape须与推导结果``[align16(Ho × Wo), C1 × Kh × Kw × C0]``一致。
- 数据类型仅支持DT_FP16和DT_BF16，源与目的数据类型必须一致。
- ``pos_m``、``pos_k``必须为非负整数。
- 源窗口的H、W必须与调用时窗口几何推导的``fmap_h``、``fmap_w``一致（源Tile即窗口本身）。

## 返回值说明

无。

## 调用示例

以[pypto_pro.language.load](load.md)载入L1 Buffer的特征图窗口为源，展开后与L0B权重矩阵做matmul：

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def conv2d_slice_kernel(
    fmap: pl.Tensor[[1, 16, 8, 8], pl.DT_FP16],
):
    # 一个特征图窗口：NC1HWC0 [N=1, C1=1, H=8, W=8, C0=16]
    fmap_l1_type = pl.TileType(
        shape=[1, 1, 8, 8, 16], dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NC1HWC0)
    # 展开结果：Ho=Wo=8，M=align16(64)=64，K=C1*Kh*Kw*C0=1*3*3*16=144
    a_l0_type = pl.TileType(
        shape=[64, 144], dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Left, layout=pl.NZ)

    fmap_l1 = pl.make_tile_group(type=fmap_l1_type, addrs=0x00000, depth=2, mutex_ids="auto")
    a_l0 = pl.make_tile_group(type=a_l0_type, addrs=0x00000, depth=2, mutex_ids="auto")

    cfg = pl.Img2ColConfig(
        kernel=[3, 3], stride=[1, 1], dilation=[1, 1],
        padding=[1, 1, 1, 1],
    )

    with pl.section_cube():
        window = fmap_l1.next()
        a = a_l0.next()
        pl.load(window, fmap, [0, 0, 0, 0])
        pl.img2col(a, window, config=cfg, pos_m=0, pos_k=0)
```

3D卷积的D循环由调用方实现，每次载入一个深度切片：

```python
# NDC1HWC0窗口：[1, 1, C1, H, W, C0]，D维固定为1
fmap_l1_type = pl.TileType(
    shape=[1, 1, 1, 8, 8, 16], dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NDC1HWC0)
# 按深度切片循环载入并展开，配合matmul_acc跨切片累加
```
