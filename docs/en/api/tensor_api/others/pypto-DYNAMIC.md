# pypto.DYNAMIC

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
<!-- npu="9030" id4 -->
- Kirin 9030 processors: Supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Supported
<!-- end id6 -->

## Description

`pypto.DYNAMIC` is used to define dynamic dimensions, allowing some dimensions of a tensor to vary at runtime. This is useful for handling variable batch size, sequence length, and similar scenarios. Dynamic dimensions are typically defined at the module level and then used in the type annotations of the JIT-compiled kernel functions.

Main application scenarios:

- **Dynamic batch size**: The batch size may vary with the number of requests during inference.
- **Dynamic sequence length**: The text sequence length in NLP tasks is not fixed.
- **Dynamic graph structure**: The number of nodes in a graph neural network is variable.
- **Conditional computation**: The computation process is determined based on the input shape.

## Shape Marking Methods

| Marker| Definition|
| --- | --- |
| `pypto.DYNAMIC` or `pypto.DYN`| Dynamic axis. When the corresponding dimension of the input torch tensor changes, **no recompilation is required**.|
| `pypto.STATIC` | Static axis. When the corresponding dimension of the input torch tensor changes, **recompilation is triggered**.|
| `64` | Fixed axis. Only the specified fixed size can be passed. If other sizes are passed, an error will be reported (runtime_debug_mode is set to 3, which enables validation).|
| `...` | The remaining axes are processed as static axes.|

## Constraints

1. The dynamic dimension must be used in the type annotation of the JIT function.

## Examples

### Example 1: Basic Usage - Dynamic Batch Size

```python
import torch
import pypto

# Fixed axis
HIDDEN_SIZE = 128

@pypto.frontend.jit
def add_bias(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    bias: pypto.Tensor([HIDDEN_SIZE], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)
):
    # Implement the add logic.
    # The first dimension of [pypto.DYNAMIC, ...] is dynamic. The ellipsis (...) indicates that the remaining dimensions are static.
    ...

# It can be called with different batch sizes.
bias = torch.randn(HIDDEN_SIZE, dtype=torch.float32, device='npu:0')
x1 = torch.randn(2, 128, dtype=torch.float32, device='npu:0')
out1 = torch.randn(2, 128, dtype=torch.float32, device='npu:0')
result1 = add_bias(x1, bias, out1)  # batch=2

x2 = torch.randn(8, 128, dtype=torch.float32, device='npu:0')
out2 = torch.randn(8, 128, dtype=torch.float32, device='npu:0')
result2 = add_bias(x2, bias, out2)  # batch=8
```

### Example 2: Multiple Dynamic Dimensions

```python
import torch
import pypto

HIDDEN = 768

@pypto.frontend.jit
def attention_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, HIDDEN], pypto.DT_FP32),
    k: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, HIDDEN], pypto.DT_FP32),
    v: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, HIDDEN], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, HIDDEN], pypto.DT_FP32),
):
    # Implement the attention logic.
    # The first two dimensions (batch and sequence length) are dynamic.
    ...
    return out

# Different batches and sequence lengths can be processed.
q_4_128 = torch.randn(4, 128, HIDDEN, dtype=torch.float32, device='npu:0')
k_4_128 = torch.randn(4, 128, HIDDEN, dtype=torch.float32, device='npu:0')
v_4_128 = torch.randn(4, 128, HIDDEN, dtype=torch.float32, device='npu:0')
out_4_128 = torch.randn(4, 128, HIDDEN, dtype=torch.float32, device='npu:0')
attention_kernel(q_4_128, k_4_128, v_4_128, out_4_128)  # B=4, SEQ=128

q_2_256 = torch.randn(2, 256, HIDDEN, dtype=torch.float32, device='npu:0')
k_2_256 = torch.randn(2, 256, HIDDEN, dtype=torch.float32, device='npu:0')
v_2_256 = torch.randn(2, 256, HIDDEN, dtype=torch.float32, device='npu:0')
out_2_256 = torch.randn(2, 256, HIDDEN, dtype=torch.float32, device='npu:0')
attention_kernel(q_2_256, k_2_256, v_2_256, out_2_256)  # B=2, SEQ=256. No recompilation required.
```

## Best Practices

1. **Document description**: Describe which dimensions are dynamic and their meanings in code comments.
2. **Test coverage**: Test different dynamic dimension values to ensure the correctness of the code.
