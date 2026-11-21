# pypto.div

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
<!-- npu="9030" id12 -->
- Kirin 9030 processors: Supported
<!-- end id12 -->
<!-- npu="x90" id13 -->
- Kirin X90 processors: Supported
<!-- end id13 -->
<!-- npu="9050" id18 -->
- Kirin 9050 processors: Supported
<!-- end id18 -->

## Description

Divides each element of input by the element at the corresponding position in other. The computation formula is as follows:

$$
res_i = input_i \div other_i
$$

## Prototype

```python
div(input: Tensor, other: Union[Tensor, float, int], precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> Tensor
```

## Parameters

| Parameter| Input/Output| Description                                                                |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input     | Source operand.<br>Supported type: Tensor. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. 1 to 4 dimensions are supported. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| other  | Input     | Source operand.<br>Supported types: Tensor, float, and int. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. 1 to 4 dimensions are supported. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| precision_type | Input| Precision mode enumeration type, which is used to control the precision mode of the division computation. For details, see [PrecisionType](../datatype/PrecisionType.md).<br>The default value is HIGH_PRECISION (high-precision mode).|

## Return Value

Returns the output tensor. Its shape is the broadcast shape of input and other.
If the input is of the floating-point type, the output tensor has the same data type as the input. If the input is of the integer type, the output data type is DT_FP32.

## Constraints

1. If both input and other are tensors, their data types must be the same. If input is of the integer type, other does not support floating-point scalars.
2. A tensor supports the following data types:
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT products: DT_FP16, DT_FP32, DT_BF16, DT_INT16, DT_INT32, and DT_INT64
   <!-- end id7 -->
   <!-- npu="A3" id8 -->
   - Atlas A3 products: DT_FP16, DT_FP32, DT_BF16, DT_INT16, and DT_INT32
   <!-- end id8 -->
   <!-- npu="910b" id9 -->
   - Atlas A2 products: DT_FP16, DT_FP32, DT_BF16, DT_INT16, and DT_INT32
   <!-- end id9 -->
   <!-- npu="9030" id14 -->
   - Kirin 9030 processors: DT_FP16 and DT_FP32
   <!-- end id14 -->
   <!-- npu="x90" id15 -->
   - Kirin X90 processors: DT_FP16 and DT_FP32
   <!-- end id15 -->
   <!-- npu="9050" id20 -->
   - Kirin 9050 processors: DT_FP16 and DT_FP32
   <!-- end id20 -->
3. **Precision mode description**:
    - **HIGH_PRECISION**: This is the default mode. It uses a higher-precision computation mode in the underlying implementation. The support for different models is as follows:
      <!-- npu="950" id4 -->
      - Ascend 950PR&950DT products: Supported
      <!-- end id4 -->
      <!-- npu="A3" id5 -->
      - Atlas A3 products: Not supported
      <!-- end id5 -->
      <!-- npu="910b" id6 -->
      - Atlas A2 products: Not supported
      <!-- end id6 -->
      <!-- npu="9030" id16 -->
      - Kirin 9030 processors: Not supported
      <!-- end id16 -->
      <!-- npu="x90" id17 -->
      - Kirin X90 processors: Not supported
      <!-- end id17 -->
      <!-- npu="9050" id19 -->
      - Kirin 9050 processors: Not supported
      <!-- end id19 -->
    - **INTRINSIC**: This mode directly uses chip instructions for computation.
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.
5. Integer input constraints:
When the input is DT_INT16, DT_INT32, or DT_INT64, it is internally converted to DT_FP32 for computation (float32 has a 24-bit mantissa). Integers within the range of $[-2^{24},\ 2^{24}]$ can be accurately converted, while those outside this range may lose low-order precision during conversion.

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

#### Basic Usage (High-Precision Mode by Default)

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.div(a, b) # HIGH_PRECISION mode is used by default.
```

Result example:

```python
Input data a:    [[2.0 4.0 6.0]]
Input data b:    [[2.0 2.0 2.0]]
Output data out:  [[1.0 2.0 3.0]]
```

#### Explicitly Specifying the High-Precision Mode

```python
a = pypto.tensor([1, 3], pypto.DT_FP16)
b = pypto.tensor([1, 3], pypto.DT_FP16)
out = pypto.div(a, b, pypto.PrecisionType.HIGH_PRECISION)
```

#### Using the Instruction Mode

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.div(a, b, pypto.PrecisionType.INTRINSIC)
```

#### Using Operators (Automatically Using the High-Precision Mode)

```python
a = pypto.tensor([1, 3], pypto.DT_FP16)
b = pypto.tensor([1, 3], pypto.DT_FP16)
out = a / b  # The HIGH_PRECISION mode is automatically used.
out = a.div(b)  # The HIGH_PRECISION mode is automatically used.
```
