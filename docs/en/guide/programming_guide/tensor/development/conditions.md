# Conditions and Branches

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T02:25:48.051Z pushedAt=2026-10-27T09:04:34.433Z nl=2 -->

Conditions and branches are used to implement conditional judgment in a program, so that different code logic is executed based on different conditions. The programming framework supports two types of conditions and branches:

- Static conditional branch: Configure conditional branches at compile time and generate fixed instructions for execution. Different kernels can be generated through multiple JIT compilations.
- Dynamic conditional branch: Determine conditions and branches at runtime and execute the corresponding functions.

## Static Conditional Branch

```python
# Generate a kernel with the input parameter add1_flag=False
@pypto.frontend.jit
def add_kernel_false(
    input0: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    val: int
):
    add_core(input0, input1, output, val, False)

# Generate a kernel with the input parameter add1_flag=True
@pypto.frontend.jit
def add_kernel_true(
    input0: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    val: int
):
    add_core(input0, input1, output, val, True)
```

Example code:

```python
def add_core(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int, add1_flag: bool = False):
    # Tiling configuration and loop logic
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    #calculate the loop parameters
    b = input0.shape[0]
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        t0_sub = input0[b_offset:b_offset_end, ...]
        t1_sub = input1[b_offset:b_offset_end, ...]
        t3_sub = t0_sub + t1_sub
        if add1_flag:
            output[b_offset:b_offset_end, ...] = t3_sub + val
        else:
            output[b_offset:b_offset_end, ...] = t3_sub
```

This use case defines two functions, add\_kernel\_false and add\_kernel\_true, which take False and True respectively as the add1\_flag input parameter to generate different kernels. If add1\_flag is True, the parameter val is added to the output result; otherwise, the result of the previous processing is output directly.

For the complete example, see [condition.py](../../../../../../examples/02_intermediate/controlflow/condition/condition.py).

## Dynamic Conditional Branch

Evaluate conditions and branches at runtime to execute the corresponding functionality. Core APIs include:

- `pypto.cond`(condition): Evaluates a condition at runtime.
- `pypto.is_loop_begin`(idx): Determines whether the current iteration is the first iteration of a loop.
- `pypto.is_loop_end`(idx): Determines whether the current iteration is the last iteration of a loop.

```python
@pypto.frontend.jit
def add_kernel(
    input0: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    val: int
):
    ...
    for idx in pypto.loop(b_loop):
        t3_sub = t0_sub + t1_sub
        if idx < 2:  # Dynamic conditional judgment
            output[b_offset:b_offset_end, ...] = t3_sub + val
        else:
            output[b_offset:b_offset_end, ...] = t3_sub

        # Or a condition based on the loop position
        if pypto.is_loop_begin(idx):
            output[b_offset:b_offset_end, ...] = t3_sub + val
        elif pypto.is_loop_end(idx):
            output[b_offset:b_offset_end, ...] = t3_sub + val + 1
        else:
            output[b_offset:b_offset_end, ...] = t3_sub
```

For the complete example, see:

[condition.py](../../../../../../examples/02_intermediate/controlflow/condition/condition.py)
