# pypto_pro.language.QuantMode

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->

## Description

Enumerates quantization modes. It is used to specify whether to perform symmetric or asymmetric quantization in the quant API.

## Prototype

```python
PYPTO_DECLARE_ENUM(QuantMode,
    SYM,   # Symmetric quantization. The output is of the DT_INT8 type. The offset parameter is not used.
    ASYM   # Asymmetric quantization. The output is of the DT_UINT8 type. The offset tile of the DT_FP32 type must be provided.
)
```

For details about the computation formula, parameter shape, and data type constraints, see [quant](../quantization/quant.md).
