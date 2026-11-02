# pypto.where

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:48:22.714Z pushedAt=2026-10-28T11:57:30.165Z nl=2 -->

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

**condition** is a Boolean mask tensor. For an element at any position in the tensor, this operation performs element-wise selection based on the Boolean mask tensor **condition**. Its computation behavior can be formally expressed as follows.

$$
result_{i}=
\begin{cases}
input_{i} & \text{if } condition_{i}==True \\
other_{i} & \text{if } condition_{i}==False
\end{cases}
$$

**condition** must be a **Tensor**. **input** and **other** can be a **Tensor**, **float**, or **Element**. The broadcast rules are as follows (only single-axis broadcast is supported):

1. When **input**, **other**, and **condition** are all **Tensor**s, the **Shape** of **result** is obtained by broadcasting the three.

    Example: **input**: \[1,20,20\], **other**: \[20,1,20\], **condition**: \[20,20,1\], **result**: \[20,20,20\]

2. When only **input** and **condition** are **Tensor**s, the **Shape** of **result** is obtained by broadcasting the two.

    Example: **input**: \[1,20,20\], **condition**: \[20,20,1\], **result**: \[20,20,20\]

3. When only other and condition are Tensors, the shape of result is obtained by broadcasting the two.

    Example: other: \[20,1,20\], condition: \[20,20,1\], result: \[20,20,20\]

4. When only condition is a Tensor, the shape of result is consistent with condition.

## Prototype

```python
where(
    condition: Tensor,
    input: Union[Tensor, float, Element],
    other: Union[Tensor, float, Element]
) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| condition   | Input      | Supported type: Tensor.<br>Supported data type of Tensor: DT_BOOL.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**).<br>Used as the condition to select elements from input or other. |
| input       | Input      | Supported type: float, Element, and Tensor.<br>When the type is float, it is automatically converted to the Element type, where float corresponds to DT_FP32. When other data types are required, they can be constructed through Element.<br>The Tensor and Element data types supported vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (that is, INT32_MAX). |
| other       | Input      | Supported type: float, Element, and Tensor.<br>When the type is float, it is automatically converted to the Element type, where float corresponds to DT_FP32. When other data types are required, they can be constructed through Element.<br>The Tensor and Element data types supported vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (that is, INT32_MAX). |

## Return Value

**result**: Tensor. The shape is obtained by broadcasting the inputs. For detailed broadcasting scenarios, see the preceding description. The data type is consistent with **input** and **other**.

## Constraints

1. When input or other is a float scalar, float is automatically converted to an Element of DT_FP32. If the other parameter is a Tensor whose data type is not DT_FP32 (such as DT_FP16 or DT_BF16), a type mismatch interception error is triggered, and this usage is not supported. Use Element to construct a data type consistent with the Tensor, for example: `pypto.Element(pypto.DT_FP16, 1.0)`.
2. The dimensions of condition, input (if it is a Tensor), and other (if it is a Tensor) must be the same. For example, condition: [64], input: [2, 64], other: [2, 64] is invalid; it should be set to condition: [1, 64], input: [2, 64], other: [2, 64].
3. Data type description for Tensor and Element of input and other:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_INT32, DT_FP32, DT_INT16, DT_FP16, DT_BF16, DT_UINT8, DT_INT8, DT_INT64, DT_UINT64
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16
   <!-- end id6 -->
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

Example 1: In a non-broadcasting scenario, where the input condition is [m, n], input is [m, n], other is [m, n], and the output is [m, n], with TileShape set to [m1, n1], m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

Example 2: Broadcast scenario, where **condition** is [m, 1], **input** is [m, n], **other** is [m, n], and the output is [m, n]. When **TileShape** is set to [m1, n1], m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
cond1 = pypto.tensor([4], pypto.DT_BOOL)
a1 = pypto.tensor([4], pypto.DT_FP32)
b1 = pypto.tensor([4], pypto.DT_FP32)
out1 = pypto.where(cond1, a1, b1)

# Using scalar inputs
out2 = pypto.where(cond1, 1, 0)

# Broadcasting example
cond2 = pypto.tensor([2, 2], pypto.DT_BOOL)
a2 = pypto.tensor([2], pypto.DT_FP32)
b2 = 0.0
out3 = pypto.where(cond2, a2, b2)
```

The results are as follows:

```python
Input data cond1: [True, False, True, False]
Input data a1:    [1.0  2.0  3.0  4.0]
Input data b1:    [10.0 20.0 30.0 40.0]
Output data out1:  [1.0  20.0 3.0  40.0]

Output data out2:  [1.0 0.0 1.0 0.0]

Input data cond2 = [[True, False], [False, True]]
Input data a2:      [1.0 2.0]
Output data out3:   [[1.0 0.0],
                 [0.0 2.0]]
```
