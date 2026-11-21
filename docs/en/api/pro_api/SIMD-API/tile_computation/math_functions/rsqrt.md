# pypto_pro.language.rsqrt

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

Performs an element-wise reciprocal square root computation. In-place operation is supported.

## Prototype

```python
pypto_pro.language.rsqrt(out: Tile, src: Tile) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand of the Tile type, which stores the element-wise reciprocal square root result.<br>The supported data types are DT_FP16 and DT_FP32.<br>The shape must be the same as that of src. If valid_shape is set for src, valid_shape must also be set for out and must be the same as that of src.<br>It can be the same Tile as src to implement in-place computation.|
| src | Input| Source operand, which is of type Tile.<br>The supported data types are DT_FP16 and DT_FP32. They must be the same as those of out.<br>Element values should be greater than 0; if they are 0 or negative, the hardware result is undefined.|

## Return Value

None

## Constraints

None

## Examples

### Basic Usage

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def rsqrt_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32],
                 out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.rsqrt(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:rsqrt:start -->
```bash
Input a: [[1 1.125 1.25 1.375 1.5 1.625 1.75 1.875 ...], [9 9.125 9.25 9.375 9.5 9.625 9.75 9.875 ...], [17 17.125 17.25 17.375 17.5 17.625 17.75 17.875 ...], [25 25.125 25.25 25.375 25.5 25.625 25.75 25.875 ...], ...]
Output out: [[1 0.942809 0.894427 0.852803 0.816497 0.784465 0.755929 0.730297 ...], [0.333333 0.331042 0.328798 0.326599 0.324443 0.322329 0.320256 0.318223 ...], [0.242536 0.241649 0.240772 0.239904 0.239046 0.238197 0.237356 0.236525 ...], [0.2 0.199502 0.199007 0.198517 0.19803 0.197546 0.197066 0.196589 ...], ...]
```
<!-- pypto-doc-output:rsqrt:end -->
