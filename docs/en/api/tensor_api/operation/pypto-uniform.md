# pypto.uniform

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:44:53.987Z pushedAt=2026-10-28T11:57:30.145Z nl=2 -->

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

Generates uniformly distributed random numbers of the specified shape, with element values in the range $[0, 1)$.
$
x_i \sim U(0, 1)
$

## Prototype

```python
uniform(shape: List[int], key: List[int], counter: List[int], alg: List[int], dtype: DataType) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|--------------------------------------------------------------------|
| shape   | Input      | Shape of the output tensor.<br>The length supports 1 to 4 dimensions.                                        |
| key     | Input      | Seed of the random number generator.<br>The length only supports 1.                                         |
| counter | Input      | Counter of the random number generator.<br>The length only supports 2.                                          |
| alg     | Input      | Random number generation algorithm. Currently only value 1 (Philox algorithm) and value 3 (auto_select, which selects the Philox algorithm) are supported.<br>The length only supports 1. |
| dtype   | Input      | Data type of the output tensor.<br>Supported data types: DT_FP32, DT_FP16, and DT_BF16.            |

## Constraints

- Splitting the shape into multiple view shapes is not supported. The view shape must be consistent with the input shape.
- The last axis of the tile shape must be a multiple of 4.
- `counter[0]` is hardcoded to 0 internally. Although the API accepts a counter list of length 2, the value of `counter[0]` is ignored, and the Philox counter actually used is `[0, counter[1]]`.

## Return Value

Returns a tensor of the specified shape and data type **dtype**, whose elements follow a uniform distribution with values in the range $[0, 1)$.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape dimensions must be consistent with the input.

If the input shape is [m, n] and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 4)
```

### API Call Example

```python
shape = [4, 4]
key = [1234]
counter = [0, 1]
alg = [1]
dtype = pypto.DT_FP32

y = pypto.uniform(shape, key, counter, alg, dtype)
```

The results are as follows:

```python
Output data y: [[0.1689806  0.9725481  0.90036285 0.16582811]
            [0.1454581  0.48029935 0.02495587 0.99239147]
            [0.02835405 0.10649502 0.45283175 0.87260246]
            [0.6877538  0.24809706 0.95886254 0.24039495]]
```
