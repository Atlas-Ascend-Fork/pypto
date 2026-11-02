# pypto.arange

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:57:09.968Z pushedAt=2026-10-28T11:57:29.738Z nl=2 -->

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

Creates a one-dimensional tensor of length $\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil$, containing an arithmetic sequence within the interval \[start, end\) with **step** as the step size.

## Prototype

```python
arange(start: Union[int, float] = 0, end: Union[int, float], step: Union[int, float] = 1) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| start  | Input      | Source operand.<br>Supported data type is int or float.<br>Default value is 0. |
| end    | Input      | Source operand.<br>Supported data type is int or float.<br>This parameter cannot be omitted. |
| step   | Input      | Source operand.<br>Supported data type is int or float, with a default value of 1.<br>It cannot be 0. When used as a floating-point number, abs\(step\)\>1e-8. <br>\(end-start\)/step must be greater than 0. |

## Return Value

Returns a one-dimensional output tensor.

- If any input value is of the float data type, the output tensor data type is DT_FP32.
<!-- npu="950" id7 -->
- For Ascend 950PR&950DT product series, if the input value is of the int data type and exceeds the DT_INT32 range, the output tensor data type is DT_INT64.
<!-- end id7 -->
- In other cases, the output tensor data type is DT_INT32.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape and the output have the same dimensions, both being one-dimensional.

If the input **start** is m, **end** is n, and **step** is p, the output shape is [q], and the TileShape is set to [q1], then q1 is used to split the q axis.

```python
pypto.set_vec_tile_shapes(16)
```

### API Call Example

```python
y1 = pypto.arange(1.0, 4.0, 0.5)
y2 = pypto.arange(1.0, 4.0)
y3 = pypto.arange(4)
```

The results are as follows:

```python
Output data y1: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
Output data y2: [1.0, 2.0, 3.0]
Output data y3: [0, 1, 2, 3]
```
