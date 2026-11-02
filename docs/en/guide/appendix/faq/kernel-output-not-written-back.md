# Computation Fails to Take Effect As the Kernel Function's Output Parameter Is Not Written Back

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:15:42.647Z pushedAt=2026-10-27T02:42:22.141Z nl=2 -->

## Symptoms

In the current PyPTO framework, kernel functions decorated with `pypto.frontend.jit` do not support return values. Outputs must be passed in as parameters and written back using `[:]` or similar operators. If an equal sign is used for direct assignment, data cannot be written into the output tensor.

Sample code:

```python
@pypto.frontend.jit
def add_kernel(x, y):
    pypto.set_vec_tile_shapes(4, 4)
    y = x + 1  # This creates a new Tensor y.

torch.npu.set_device(0)
x = torch.ones(4, 4, dtype=torch.float32)
y = torch.empty(4, 4, dtype=torch.float32)
add_kernel(pypto.from_torch(x), pypto.from_torch(y))
print(y)  # Outputs the uninitialized random value created by torch.empty.
```

Output data:

```python
tensor([[2.0703e-19, 7.1833e+22, 1.8502e+28, 6.8608e+22],
        [4.8011e+30, 1.2123e+25, 4.7418e+30, 1.8465e+25],
        [1.2122e+25, 4.6114e+24, 1.7836e+31, 1.7591e+22],
        [1.1306e+24, 4.2245e-39, 6.8664e-44, 0.0000e+00]])
```

## Possible Causes

When `y = x + 1` is executed inside the `add_kernel` function, `y` here is a local variable of the function (equivalent to creating a new variable `y`), which overwrites the reference of the incoming parameter `y`. In other words, this line of code merely makes `y` inside the function point to the new tensor `x + 1`, without modifying the content of the externally passed tensor `y`.

## Solution

Use the full-slice operator `[:]` to write the computation result into the original memory space of the function parameter `y`.

Sample code:

```python
@pypto.frontend.jit
def add_kernel(x, y):
    pypto.set_vec_tile_shapes(4, 4)
    y[:] = x + 1  # Write the result of x+1 into the original memory space of the function parameter y.

torch.npu.set_device(0)
x = torch.ones(4, 4, dtype=torch.float32)
y = torch.empty(4, 4, dtype=torch.float32)
add_kernel(pypto.from_torch(x), pypto.from_torch(y))
print(y)  # Outputs the result of x + 1.
```

Output data:

```python
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])
```

Here, `y[:] = x + 1` can also be replaced with `y.move(x + 1)` or `y.assemble(x + 1, [0, 0])`.
