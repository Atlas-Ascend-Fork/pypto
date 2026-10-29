# pypto.conv

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->

## 功能说明

实现输入input_conv、weight完成卷积运算，支持bias参数，计算公式为：out = input_conv @ weight + bias (@表示为卷积处理)。

- input_conv、weight、bias为源操作数；input_conv为输入矩阵，weight为权重矩阵，bias为输入的偏置。
- out为目的操作数，存放卷积处理结果的矩阵。
- 当前暂不支持fixpipe量化场景。

## 函数原型

```python
conv(input_conv, weight, out_dtype, strides, paddings, dilations, *, groups=1, transposed=False, output_paddings=[], extend_params=None) -> Tensor
```

## 参数说明

| 参数名            | 输入/输出 | 说明                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| input_conv       | 输入      | 输入特征图Tensor。<br>不支持空Tensor。<br>支持维度：3D（1D conv）、4D（2D conv）、5D（3D conv）。<br>支持格式：NCL、NCHW、NCDHW。<br>支持数据类型：DT_FP16、DT_BF16、DT_FP32。<br>各维度取值范围：[1, 1000000]。<br>input_conv的Cin需满足：weight的Cin * groups = input_conv的Cin。 |
| weight            | 输入      | 卷积核Tensor。<br>维度必须与input_conv一致（3D/4D/5D）。<br>数据类型必须与input_conv一致。<br>Cout、Cin和Kd取值范围：[1, 1000000]。<br>Kh和Kw取值范围：[1, 255]，且需满足Kh × Kw × (32 / sizeof(dtype)) ≤ 65535，dtype为input_conv的数据类型所占字节数，如FP16是2，FP32是4等。 |
| out_dtype         | 输入      | 输出Tensor数据类型。<br>支持：DT_FP16、DT_BF16、DT_FP32。<br>必须与input_conv一致；fixpipe量化场景可单独指定。 |
| strides           | 输入      | 卷积步长，单向参数。<br>- 1维（1D conv）<br>- 2维（2D conv）<br>- 3维（3D conv）<br>取值范围：[1, 63]。 |
| paddings          | 输入      | 卷积填充，双向参数。<br>- 2维（1D conv）<br>- 4维（2D conv）<br>- 6维（3D conv）<br>取值范围：[0, 255]，且每维填充值需小于对应卷积核大小。 |
| dilations         | 输入      | 空洞卷积膨胀率，单向参数。<br>- 1维（1D conv）<br>- 2维（2D conv）<br>- 3维（3D conv）<br>取值范围：[1, 63]。 |
| groups            | 输入      | 分组卷积组数，默认1。<br>取值范围：[1, 65535]。<br>Cin、Cout必须可被groups整除。 |
| transposed        | 输入      | 是否为转置卷积（反卷积），默认False。<br>当前暂不支持True。 |
| output_paddings   | 输入      | 转置卷积输出端填充，仅transposed=True时使用。<br>当前暂不支持。 |
| extend_params     | 输入      | 扩展参数字典，支持bias_tensor、scale、relu_type、scale_tensor：<br>- bias_tensor：可选的偏置张量，形状为(Cout,)，仅支持ND格式，不同型号支持的数据类型有所差异，详细请参见[约束说明](#约束说明)。<br>- scale：浮点型，per-tensor缩放因子。<br>- scale_tensor：uint64类型，per-channel缩放Tensor，shape [1, Cout]，仅ND格式。<br>- relu_type：激活类型，支持RELU/NO_RELU等。 |

## 返回值说明

返回卷积运算后的输出Tensor：

- 1D卷积输出shape：(Batch, Cout, Wout)
- 2D卷积输出shape：(Batch, Cout, Hout, Wout)
- 3D卷积输出shape：(Batch, Cout, Dout, Hout, Wout)

输出shape各维度均支持动态轴切分，注意当groups大于1时不允许切分Cout；输出shape各维度范围：[1, 1000000]。

## 约束说明

1. 缓存空间约束：调用conv接口前，必须通过pypto.set_conv_tile_shapes接口设置L1/L0层级的卷积TileShape切分大小。

2. 动态轴切分约束：

    <!-- npu="950" id4 -->
    - Ascend 950PR&950DT系列产品：前端循环切分时，Batch、Cout、Dout、Hout、Wout维度需要小于或等于out shape对应维度大小；不支持前端循环切分Cin。
    <!-- end id4 -->
    <!-- npu="A3" id5 -->
    - Atlas A3系列产品：不支持前端循环切分Cin。
    <!-- end id5 -->
    <!-- npu="910b" id6 -->
    - Atlas A2系列产品：不支持前端循环切分Cin。
    <!-- end id6 -->

3. 数据类型约束：

    <!-- npu="950" id7 -->
    - Ascend 950PR&950DT系列产品：支持的数据类型为DT_FP16、DT_BF16、DT_FP32。input_conv、weight、bias和out的数据类型需要相同。
    <!-- end id7 -->
    <!-- npu="A3" id8 -->
    - Atlas A3系列产品：支持的数据类型为DT_FP16、DT_BF16、DT_FP32。对于DT_FP16和DT_FP32类型，input_conv、weight、bias和out的数据类型需要相同；对于DT_BF16类型，input_conv、weight和out为DT_BF16类型，bias需为DT_FP32类型。
    <!-- end id8 -->
    <!-- npu="910b" id9 -->
    - Atlas A2系列产品：支持的数据类型为DT_FP16、DT_BF16、DT_FP32。对于DT_FP16和DT_FP32类型，input_conv、weight、bias和out的数据类型需要相同；对于DT_BF16类型，input_conv、weight和out为DT_BF16类型，bias需为DT_FP32类型。
    <!-- end id9 -->

## 调用示例

```python
# 2D卷积基础示例
input_conv = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 32, 1, 1), pypto.DT_FP16, "weight")

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1],
                   paddings=[0, 0, 0, 0],
                   dilations=[1, 1])

# 2D卷积带bias和ReLu
input_conv = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 32, 1, 1), pypto.DT_FP16, "weight")
bias = pypto.tensor((32,), pypto.DT_FP16, "bias")
extend_params = {'bias_tensor': bias, 'relu_type': pypto.ConvReLuType.RELU}

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1],
                   paddings=[0, 0, 0, 0],
                   dilations=[1, 1],
                   extend_params=extend_params)

# 3D卷积示例
input_conv = pypto.tensor((1, 96, 2, 16, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 96, 1, 1, 1), pypto.DT_FP16, "weight")

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1, 1],
                   paddings=[0, 0, 0, 0, 0, 0],
                   dilations=[1, 1, 1])
```
