# pypto.argmin

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
<!-- npu="9030" id4 -->
- Kirin9030处理器系列产品：不支持
<!-- end id4 -->
<!-- npu="x90" id5 -->
- KirinX90处理器系列产品：不支持
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin9050处理器系列产品：不支持
<!-- end id6 -->

## 功能说明

对一个多维向量在指定的维度求最小值的索引。

定义指定计算的维度（Reduce轴）为R轴，非指定维度（Normal轴）为A轴。对Shape为\(2, 3\)的二维矩阵进行运算，指定在第一维求最小值索引，输出结果为\[0, 0, 0\]；指定在第二维求最小值索引，输出结果为\[0, 0\]。

## 函数原型

```python
argmin(input: Tensor, dim: int = -1, keepdim: bool = False, propagate_nan: bool = True) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP16，DT_BF16，DT_FP32。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 指定进行归约的维度。<br>支持任意单轴。<br>默认值为-1。                            |
| keepdim | 输入      | 归约后是否保留被归约的维度。<br>默认值为False。 |
| propagate_nan | 输入 | 编译期bool常量。<br>为True时沿用原有设备规约及其NaN索引行为。为False时排除NaN并返回首个极值索引，全NaN时返回0。输出索引类型为DT_INT32。<br>默认值为True。 |

## 返回值说明

返回输出Tensor，输出Tensor的数据类型为DT_INT32，Shape与keepdim参数相关。

若keepdim参数为True，则在执行归约操作后保留被归约的维度。输出Tensor在除dim指定的维度外，其他维度的Shape与输入Tensor的Shape一致，而在dim指定的维度上的大小为1。

若keepdim参数为False（默认），则被归约的维度会从输出Tensor中移除，而TileShape中对应的维度不变，所以建议在调其他operation前重设TileShape。

## 约束说明

- TileShape大小不超过64KB；

- propagate_nan为True时沿用原有尾轴32字节对齐要求。propagate_nan为False时支持非对齐逻辑长度，编译器对UB中的物理布局执行32字节对齐。

- Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。
- 输入Tile、输出Tile和临时空间的总占用须小于可用UB。
   - propagate_nan=True沿用原生路径。记计算元素字节数为E（FP16输入先转FP32，其余输入沿用原类型），当前Tile的归约轴长度为K，尾轴归约时倒数第二维长度为H。专用临时空间在一维输入时为 `K * E` 字节，尾轴归约时为 `H * K * E` 字节，非尾轴归约时为768字节，物理布局按32字节对齐。
   - propagate_nan=False使用FP32计算，索引为INT32。以padding和轴组合后的二维布局计算H/W，尾轴时H为行数，非尾轴时H为归约轴长度。W按当前Tile的逻辑形状计算，非尾轴时为归约轴之后各维长度的乘积（先将输入尾维按32字节对齐）。源输入的额外padding通过算子内部的紧凑副本处理，不扩大工作区行步长。令 `A(x)=ceil(x/32)*32`、`B(x)=ceil(x/256)*256`，将W对齐为 `A(4*W)/4`，再计算：
     - 紧凑输入副本和工作数据各为 `D=4*H*W`，比较保护区 `C=B(4*W)-4*W`，位掩码 `M=H*A(ceil(W/8))`。
     - 单个结果区P在尾轴时为 `H*32`，非尾轴时为 `4*W`；结果行字节数F分别为32和 `4*W`。尾轴的数值、候选标记和索引共占3P；非尾轴的数值和索引直接写输出，scratch仅保留候选标记P。
     - 尾轴指令区G按 `4*W<=256`、`256<4*W<=512`、`4*W>512` 三档分别为 `H*32`、`H*256`、`H*floor(4*W/512)*256` 字节；非尾轴G=768。
     - 尾轴归约TileOp申请 `max(2*D+C+M+3*P+G+32, 2*D+C+M+2*P+B(F)-F)` 字节；非尾轴申请 `max(2*D+C+M+P+G+32, 2*D+C+M+P+B(F)-F)` 字节。
     - 归约轴切分时，按每个输出二维切片计算。令 `R=4*Ws`、`S=Hs*A(ceil(Ws/8))+32`、`J=Hs*A(R)`。当 `R>256` 且 `B(R)!=R` 时，比较工作区T为 `2*(Hs*R+B(R)-R)`，其余情况T=0。该切片申请 `S+J+T` 字节。外层切片顺序复用此空间。
     FP16/BF16转FP32的中间张量，以及输入、输出和各中间结果各自计入UB，最终峰值按生命周期取最大，而非各项求和。
   - 总占用超出UB时，编译器报告容量错误。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输入input一致。

如输入input shape为[m, n]，输出为[m, 1]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

注意：如果keepdim设置为false，则被归约的维度会从输出Tensor中移除，而TileShape中对应的维度不变，所以建议在调其他operation前重设TileShape。

### 接口调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.argmin(x, -1, True)
```

结果示例如下：

```python
输入数据x: [[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
输出数据y: [[0],
             [0]]
```
