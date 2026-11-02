# pypto_pro.language.add_relu

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T14:45:41.824Z pushedAt=2026-10-28T11:57:29.272Z nl=2 -->

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

Performs element-wise addition on two Tiles and then applies ReLU activation to the result, setting elements less than 0 to 0. Unlike add_relu_cast, this API does not perform data type conversion.

## Prototype

```python
pypto_pro.language.add_relu(
  out: Tile,
  lhs: Tile,
  rhs: Tile
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| out | Output | Destination UB Tile that stores the result of element-wise addition and ReLU activation. Supported data types: DT_FP16, DT_FP32, and DT_INT32. |
| lhs | Input/Output | Left operand UB Tile that stores the intermediate result of addition during computation. Supported data types: DT_FP16, DT_FP32, and DT_INT32. |
| rhs | Input | Right operand UB Tile. Supported data types: DT_FP16, DT_FP32, and DT_INT32. |

## Constraints

- The data types and valid shapes of out, lhs, and rhs must be consistent.
- out, lhs, and rhs support ND, ZN, and ZZ.
- lhs is modified during computation, and its original data is not retained after the call.

## Return Value

None

## Examples

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def add_relu_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32],
                    out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(
      shape=[64, 64],
      dtype=pl.DT_FP32,
      target_memory=pl.MemorySpace.Vec,
      layout=pl.TensorLayout.ND,
    )
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add_relu(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

The actual test results are as follows:

<!-- pypto-doc-output:add_relu:start -->
```bash
Input data a: [[-6 -5.75 -5.5 -5.25 -5 -4.75 -4.5 -4.25 ...], [10 10.25 10.5 10.75 11 11.25 11.5 11.75 ...], [26 26.25 26.5 26.75 27 27.25 27.5 27.75 ...], [42 42.25 42.5 42.75 43 43.25 43.5 43.75 ...], ...]
Input data b: [[1 1.5 2 2.5 3 3.5 4 4.5 ...], [33 33.5 34 34.5 35 35.5 36 36.5 ...], [65 65.5 66 66.5 67 67.5 68 68.5 ...], [97 97.5 98 98.5 99 99.5 100 100.5 ...], ...]
Output data out: [[0 0 0 0 0 0 0 0.25 ...], [43 43.75 44.5 45.25 46 46.75 47.5 48.25 ...], [91 91.75 92.5 93.25 94 94.75 95.5 96.25 ...], [139 139.75 140.5 141.25 142 142.75 143.5 144.25 ...], ...]
```
<!-- pypto-doc-output:add_relu:end -->
