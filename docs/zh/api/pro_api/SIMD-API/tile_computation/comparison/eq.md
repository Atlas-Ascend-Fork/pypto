# pypto_pro.language.eq

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

将lhs与rhs逐元素比较是否相等，生成bit-packed掩码。rhs可以是Tile或标量。掩码Tile的数据类型不做限制，通常使用DT_UINT8类型，配合[pypto_pro.language.select](../selection/select.md)使用。

## 函数原型

```python
pypto_pro.language.eq(out: Tile, lhs: Tile, rhs: Union[Tile, int, float]) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，数据类型不做限制，通常使用DT_UINT8类型，配合按位压缩的掩码布局使用。 |
| lhs | 输入 | 源操作数（左操作数），Tile类型，存储空间为UB，采用行主序排布。支持8、16、32、64位整型、DT_FP16、DT_BF16和DT_FP32。 |
| rhs | 输入 | 源操作数（右操作数），Tile、整型或浮点型常量，或运行时标量表达式。传入Tile时，存储空间为UB，采用行主序排布，数据类型、形状和有效形状必须与lhs一致；传入常量时，取值须在lhs元素数据类型的可表示范围内；传入运行时标量时，其取值将按lhs的元素类型转换，须在可表示范围内，否则精度损失未定义。 |

## 约束说明

out保存按位压缩的掩码，不能按逐元素0/1数值Tile解释。需要数值结果时，可与[pypto_pro.language.select](../selection/select.md)配合使用。

- out不可与lhs或Tile类型的rhs使用相同的UB起始地址：掩码按位打包写入该地址后，会就地覆盖这两路输入的前若干字节，输入数据被破坏，此时结果未定义。

## 返回值说明

无。

## 调用示例

```python
pl.eq(mask, lhs, rhs)  # Tile-Tile比较：lhs == rhs
pl.eq(mask, lhs, 0)    # Tile-Scalar比较：lhs == 0
```
