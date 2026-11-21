# pypto.fmod

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->
<!-- npu="9030" id7 -->
- Kirin 9030 processors: Not supported
<!-- end id7 -->
<!-- npu="x90" id8 -->
- Kirin X90 processors: Not supported
<!-- end id8 -->
<!-- npu="9050" id9 -->
- Kirin 9050 processors: Not supported
<!-- end id9 -->

## Description

Performs the modulo operation on each element of input and the element at the corresponding position in other. The computation formula is as follows:

$$
res_i = input_i \;\%\; other_i
$$

## Prototype

```python
fmod(input: Tensor, other: Union[Tensor, float], precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> Tensor
```

## Parameters

| Parameter| Input/Output| Description                                                                |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input     | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| other  | Input     | Source operand.<br>Supported types: float and Tensor.<br>Supported data types of a tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| precision_type | Input| Precision mode enumeration type, which is used to control the precision mode of modulo computation. For details, see [PrecisionType](../datatype/PrecisionType.md).<br>The default value is HIGH_PRECISION (high-precision mode).|

## Return Value

Returns the output tensor. Its data type is the same as that of input and other, and its shape is the broadcast shape of input and other.

## Constraints

1. The data types of input and other must be the same.
2. When other is a number, implicit conversion is not supported.
3. other does not support special values such as nan and inf.
4. Usage notes for precision_type:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: The high-precision mode is supported.
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: The high-precision mode is not supported. The instruction mode `INTRINSIC` is used by default.
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: The high-precision mode is not supported. The instruction mode `INTRINSIC` is used by default.
   <!-- end id6 -->
5. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

In a non-broadcast scenario, if the input shape is [m, n], other is [m, n], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], other is [m, 1], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.fmod(a, b)
```

Result example:

```python
Input data a:    [[7.0 8.0 9.0]]
Input data b:    [[3.0 3.0 3.0]]
Output data out:  [[1.0 2.0 0.0]]
```

### High-Precision Mode Example

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.fmod(a, b, pypto.PrecisionType.HIGH_PRECISION)
```

### Instruction Mode Example

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.fmod(a, b, pypto.PrecisionType.INTRINSIC)
```
