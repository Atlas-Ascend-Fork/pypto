# Compilation and Execution

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T02:23:17.977Z pushedAt=2026-10-27T09:04:37.579Z nl=2 -->

PyPTO builds a compilable compute graph structure on NPU hardware through function definitions, and uses the @pypto.frontend.jit decorator to implement just-in-time (JIT) compilation. This fully leverages the parallel computing capabilities of the NPU and improves operator execution efficiency.

## Kernel Function Definition

Before performing JIT compilation, you need to define a kernel function, obtain input and output tensors, configure tiling information, and implement the computation logic.

- Basic function definition:

    ```python
    def add_kernel(input: pypto.Tensor, out: pypto.Tensor):
        # Tiling setting
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out[:] = input + 1
    ```

- Multi-input/output function definition:

    ```python
    def add_kernel(input0: pypto.Tensor, input1: pypto.Tensor, out: pypto.Tensor):
         # Tiling setting
         pypto.set_vec_tile_shapes(1, 4, 1, 64)
         out[:] = input0 + input1
    ```

## JIT Compilation

After the computation flow and data flow of a kernel are written through PyPTO functions, you can add the `pypto.frontend.jit` decorator to mark the function as a JIT compilation target, triggering the PyPTO compilation process.

```python
@pypto.frontend.jit
def add_kernel(
    input0: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    input1: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    out: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
):
     # Tiling setting
     pypto.set_vec_tile_shapes(1, 4, 1, 64)
     out[:] = input0 + input1
```

The JIT compilation process is as follows:

- On the first call, the function is executed in "recording mode", where operations are recorded and optimized into a compute graph. The compiler then generates NPU-optimized code and caches the binary file.
- On subsequent calls, the function directly invokes the cached binary file for execution on the NPU, eliminating the need for recompilation.

For the complete example, see [hello_world](../../../../../../examples/00_hello_world/hello_world.py).

## Conditional Compilation

The JIT decorator supports parameter configuration, enabling different conditional compilation based on the configuration:

```python
@pypto.frontend.jit(
    host_options={},
    pass_options={},
    runtime_options={},
    verify_options={},
    debug_options={}
)
def advanced_function(input0, input1):
    # Implement the custom computation logic
    pass
```

The JIT configuration options are described as follows:

- codegen\_options: code generation settings.
- host\_options: host-side options.
- pass\_options: compiler pass options.
- runtime\_options: runtime execution options.
- verify\_options: precision verification tool options.
- debug\_options: performance data collection configuration option.

In addition to using the JIT decorator to enable different configurations, you can also directly call the pypto.set\_codegen\_options, pypto.set\_host\_options, pypto.set\_pass\_options, pypto.set\_runtime\_options, pypto.set\_verify\_options, and pypto.set\_debug\_options APIs in the code for configuration, for example:

```python
pypto.set_codegen_options(support_dynamic_aligned=True)
```

It is recommended to prioritize using JIT parameters to configure various options, because JIT configuration options provide configuration convenience while avoiding code unrelated to data flow and computation inside the compute function.

## Defining Multiple JIT Functions

You can define multiple JIT functions and use them together:

```python
def add_core(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int, add1_flag: bool = False):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    if add1_flag:
        t3 = input0 + input1
        output[:] = t3 + val
    else:
        output[:] = input0 + input1

@pypto.frontend.jit
def add_kernel_true(
    input0: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    val: int
):
    add_core(input0, input1, output, val, True)


@pypto.frontend.jit
def add_kernel_false(
    input0: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    val: int
):
    add_core(input0, input1, output, val, False)


# Use these two functions
def add_add1flag_false(input_data0, input_data1, val=0):
    output_data = torch.empty_like(input_data0)
    add_kernel_false(input_data0, input_data1, output_data, val)
    return output_data

def add_add1flag_true(input_data0, input_data1, val=0):
    output_data = torch.empty_like(input_data0)
    add_kernel_true(input_data0, input_data1, output_data, val)
    return output_data

add_add1flag_false(input_data0, input_data1, val)
add_add1flag_true(input_data0, input_data1, val)
```

For the complete example, see [multi_jit.py](../../../../../../examples/03_advanced/patterns/function/multi_jit.py)
