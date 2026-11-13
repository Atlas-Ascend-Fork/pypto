# DequantScaleRoundingMode

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

DequantScaleRoundingMode定义了MX量化时共享指数的舍入模式，用于指定[pypto.quant_mx](../operation/pypto-quant_mx.md)的mode参数，选择计算缩放因子的算法。接口的产品支持情况请参见对应接口文档。

## 原型定义

```python
class DequantScaleRoundingMode(enum.Enum):
    ROUND_UP = ...    # 使用NV算法计算缩放因子
    ROUND_DOWN = ...  # 使用OCP算法计算缩放因子，quant_mx的默认模式
```
