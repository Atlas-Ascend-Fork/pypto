# pypto.expand\_exp\_dif

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:31:25.906Z pushedAt=2026-10-28T11:57:29.877Z nl=2 -->

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

Computes **e** raised to the power of (**input** - **other**), where **e** is the base of the natural logarithm, and returns a tensor with the same shape as **input** after broadcasting. The computation formula is as follows:

$$
res_i = e^{(input_i - other_i)}
$$

## Prototype

```python
expand_exp_dif(input: Tensor, other: Tensor) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP16, DT_FP32, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| other   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP16, DT_FP32, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns the output tensor. Its data type is the same as that of **input** and **other**, and its shape is the size after **input** and **other** are broadcast.

## Constraints

1. When both **input** and **other** are tensors, their data types must be the same.
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], **other** is [1, n] or [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

#### Example 1: Broadcasting on the Last Axis

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.tensor([2, 1], pypto.DT_FP32)
out = pypto.expand_exp_dif(x, y)
```

The results are as follows:

```python
Input data x:     [[1, 2, 3], [4, 5, 6]]
Input data y:     [[1], [2]]
Output data out:   [[ 1.       ,  2.718282 ,  7.3890557],
               [ 7.3890557, 20.085537 , 54.59815  ]]
```

#### Example 2: Multi-axis Broadcasting

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.tensor([1, 1], pypto.DT_FP32)
out = pypto.expand_exp_dif(x, y)
```

The results are as follows:

```python
Input data x:     [[1, 2, 3], [4, 5, 6]]
Input data y:     [[1]]
Output data out:   [[ 1.       ,  2.718282 ,  7.3890557],
               [20.085537 , 54.59815  , 148.41316 ]]
```
