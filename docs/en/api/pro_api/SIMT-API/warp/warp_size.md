# pypto_pro.language.simt.warp_size

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

Obtains the number of threads contained in a SIMT warp of the target hardware.

## Prototype

```python
pypto_pro.language.simt.warp_size() -> Scalar
```

## Parameters

None

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the number of threads contained in the warp of the target hardware. The data type is DT_INT32.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=1)
def write_warp_size(output: pl.Tensor[[1, 1], pl.DT_INT32]):
    output[0, 0] = pl.simt.warp_size()


@pl.jit()
def warp_size_kernel(output: pl.Tensor[[1, 1], pl.DT_INT32]):
    with pl.section_vector():
        write_warp_size[1](output)
```
