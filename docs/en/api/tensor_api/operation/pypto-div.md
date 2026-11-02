# pypto.div

<!-- md-trans-meta sourceCommit=50483671e36b55446f83f1726cc2e05a8b7b04bd translatedAt=2026-10-26T16:24:42.805Z pushedAt=2026-10-28T11:57:29.853Z nl=2 -->

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

## Description

Divides each element of **input** by the element at the corresponding position in **other**. The computation formula is as follows:

$$
res_i = input_i \div other_i
$$

## Prototype

```python
div(input: Tensor, other: Union[Tensor, float, int], precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input      | Source operand.<br>Supported type: Tensor. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. Supported dimensions: 1 to 4. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (that is, INT32_MAX). |
| other  | Input      | Source operand.<br>Supported types: Tensor, float, int. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. Supported dimensions: 1 to 4. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (that is, INT32_MAX). |
| precision_type | Input | Precision mode enumeration, used to control the precision mode of the division computation. For details, see [PrecisionType](../datatype/PrecisionType.md).<br>Defaults to **HIGH_PRECISION** (high-precision mode). |

## Return Value

Returns the output tensor, whose shape is the broadcast size of input and other.
When input is a floating-point type, the output tensor has the same data type as input; when input is an integer type, the output data type is DT_FP32.

## Constraints

1. When both input and other are tensors, their data types must be the same; when input is an integer type, other does not support floating-point scalars.
2. Tensor data type description:
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT products: DT_FP16, DT_FP32, DT_BF16, DT_INT16, DT_INT32, DT_INT64.
   <!-- end id7 -->
   <!-- npu="A3" id8 -->
   - Atlas A3 products: DT_FP16, DT_FP32, DT_BF16, DT_INT16, DT_INT32.
   <!-- end id8 -->
   <!-- npu="910b" id9 -->
   - Atlas A2 products: DT_FP16, DT_FP32, DT_BF16, DT_INT16, DT_INT32.
   <!-- end id9 -->
3. **Precision mode description**:
    - **HIGH_PRECISION (High-Precision Mode)**: The default mode, which uses higher-precision computation in the underlying implementation. Support across different models:
      <!-- npu="950" id4 -->
      - Ascend 950PR&950DT products: Supported
      <!-- end id4 -->
      <!-- npu="A3" id5 -->
      - Atlas A3 products: Not supported
      <!-- end id5 -->
      <!-- npu="910b" id6 -->
      - Atlas A2 products: Not supported
      <!-- end id6 -->
    - **INTRINSIC (instruction mode)**: Directly uses chip instructions for computation.
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.
5. Integer input constraints:
When the input is DT_INT16, DT_INT32, or DT_INT64, the input is internally converted to DT_FP32 for computation (float32 has a 24-bit mantissa). Integers within the range $[-2^{24},\ 2^{24}]$ can be converted exactly, while integers outside this range may lose low-order precision during conversion.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], **other** is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

#### Basic Usage (High-Precision Mode by Default)

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.div(a, b)  # Uses HIGH_PRECISION mode by default.
```

The results are as follows:

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

#### Using Operators (Automatically Using the HIGH_PRECISION Mode)

```python
a = pypto.tensor([1, 3], pypto.DT_FP16)
b = pypto.tensor([1, 3], pypto.DT_FP16)
out = a / b  # Automatically use HIGH_PRECISION mode.
out = a.div(b)  # Automatically use HIGH_PRECISION mode.
```
