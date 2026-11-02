# pypto.DYNAMIC

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:50:11.743Z pushedAt=2026-10-28T11:57:30.175Z nl=2 -->

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

`pypto.DYNAMIC` is used to define a dynamic dimension, allowing certain dimensions of a tensor to vary at runtime. This is useful for handling variable batch size, sequence length, and similar scenarios. Dynamic dimensions are typically defined at the module level and then used in the type annotations of JIT-compiled kernel functions.

Main application scenarios:

- **Dynamic batch size**: The batch size may vary with the number of requests during inference.
- **Dynamic sequence length**: The text sequence length is not fixed in NLP tasks.
- **Dynamic graph structure**: The number of nodes in a graph neural network is variable.
- **Conditional computation**: The computation flow is determined based on the input shape.

## Shape Marking

| Marker | Meaning |
| --- | --- |
| `pypto.DYNAMIC` or `pypto.DYN` | Dynamic axis. When the corresponding dimension of the passed torch tensor changes, **no recompilation is required**. |
| `pypto.STATIC` | Static axis. When the corresponding dimension of the passed torch tensor changes, **recompilation is triggered**. |
| `64` | Fixed axis. Only the specified fixed size is allowed; passing any other size reports an error (when runtime_debug_mode is set to 3, which enables validation). |
| `...` | The remaining axes are all treated as static axes. |

## Constraints

1. The dynamic dimension must be used in the type annotation of the JIT function.

## Examples

### Example 1: Basic Usage - Dynamic Batch Size

```python
import pypto

# Fixed axis.
HIDDEN_SIZE = 128

@pypto.frontend.jit
def add_bias(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    bias: pypto.Tensor([HIDDEN_SIZE], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)
):
    # Implement the add logic.
    # [pypto.DYNAMIC, ...] The first dimension is dynamic, and the ellipsis indicates that the remaining dimensions are static.
    ...

# Can be called with different batch sizes.
x1 = torch.randn(2, 128, dtype=torch.float32, device='npu:0')
out1 = torch.randn(2, 128, dtype=torch.float32, device='npu:0')
result1 = add_bias(x1, bias, out1)  # batch=2

x2 = torch.randn(8, 128, dtype=torch.float32, device='npu:0')
out2 = torch.randn(8, 128, dtype=torch.float32, device='npu:0')
result2 = add_bias(x2, bias, out2)  # batch=8
```

### Example 2: Multiple Dynamic Dimensions

```python
HIDDEN = 768

@pypto.frontend.jit
def attention_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, HIDDEN], pypto.DT_FP32),
    k: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, HIDDEN], pypto.DT_FP32),
    v: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, HIDDEN], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, HIDDEN], pypto.DT_FP32),
):
    # Implement the attention logic.
    # The first two dimensions (batch and sequence length) are both dynamic.
    ...
    return out

# Can handle different batch and sequence lengths.
attention_kernel(q_4_128, k_4_128, v_4_128, out)  # B=4, SEQ=128
attention_kernel(q_2_256, k_2_256, v_2_256, out)  # B=2, SEQ=256, no recompilation required.
```

## Best Practices

1. **Documentation**: Explain in code comments which dimensions are dynamic and what they mean.
2. **Test coverage**: Test different dynamic dimension values to ensure code correctness.
