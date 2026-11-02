# Loop Principles and Description

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:16:42.349Z pushedAt=2026-10-27T02:43:22.185Z nl=2 -->

During compilation, a loop is compiled into control flow, while the loop body is transformed into compute flow, corresponding to `kernel_aicpu` and `kernel_aicore`, respectively. `kernel_aicpu` is responsible for executing the control flow, creating `kernel_aicore` tasks, including memory allocation and preparation of execution parameters. It also analyzes input/output dependencies to merge multiple `kernel_aicore` tasks into a larger scheduling unit, and then submits this unit to the scheduler for execution.

## Prototype Introduction

```python
def loop(start, end, step=1, name=None, idx_name=None,
         unroll_list=[1], submit_before_loop=False):

def loop_unroll(start, end, step=1, name=None, idx_name=None,
              unroll_list=[1], submit_before_loop=False):
```

## Parameters

1. `start`, `end`, `step`: These represent the start value, end value, and step size of the loop, respectively. The type can be `SymbolicScalar` or `int`, and they are basically consistent with the `range` syntax in Python.

2. `name`: The name of the loop. Default value: `loop_{id}`. It has no impact on the actual runtime behavior and is used only for debugging purposes and for generating comment information in the code.

3. `idx_name`: The name of the loop index. Default value: `loop_idx_{id}`. Using the same `idx` for nested loops causes overwriting behavior, which is currently checked and reported as an error at the frontend.

4. `unroll_list`: Primarily used for loop unrolling, which generates a larger loop body to reduce scheduling overhead.

   - For `unroll_list=2`, `loop` generates code similar to the following:

      ```python
      new_start = start
      for k in unroll_list:
          left = (end - start) % k
          for idx in loop(new_start, end - left, k):
              for i in range(k):
                  body(idx)  # The user needs to process the step size of step=1 at a time
          new_start = end - left
      ```

   - `loop_unroll` generates code similar to the following:

      ```python
      new_start = start
      for k in unroll_list:
          left = (end - start) % k
          for idx in loop(new_start, end - left, k):
              body(idx, k)  # The user needs to process the step size of k at a time
          new_start = end - left
      ```

   In principle, if multiple `i` values can be processed at a time, using `loop_unroll` is more efficient; if only one `i` can be processed at a time, `loop` should be used.

5. `submit_before_loop`: Whether to submit a task before the loop starts. Default value: `False`. If it is set to `True`, the task preceding the loop is submitted to the scheduling queue first and waits for subsequent tasks to complete before starting execution. **Excessive use of `submit_before_loop` increases scheduling overhead**. It is recommended to set it to `True` only when necessary.

6. Impact of `unroll_list` on `pypto.cond`: Typically, a loop contains one `pypto.cond`, which generates two branches. When the unroll count is 4, 2⁴ = 16 path branches are generated. Each branch usually requires separate compilation, which significantly increases compilation time and the amount of compiled code. To support compilation optimization of the key operator FA, two special functions, `pypto.is_loop_begin()` and `pypto.is_loop_end()`, are provided for optimizing conditional branches.

7. Considering that applying `loop_unroll` to an outer loop does not increase the loop body size, unrolling is currently supported only for the innermost loop.

8. Implicit loop example: During the frontend compilation phase, the framework implicitly inserts a `loop(1)` at the beginning of the function. The loop ends before the next loop starts:

   ```python
   @pypto.frontend.jit
   def foo(a, b, c):
       c[:] = a + b
   # Equivalent to
   @pypto.frontend.jit
   def foo(a, b, c):
       for i in pypto.loop(1):
           c[:] = a + b

   @pypto.frontend.jit
   def foo(a, b, c):
       t = a + 1
       for i in pypto.loop(1):
           c[:] = t + b
   # Equivalent to
   @pypto.frontend.jit
   def foo(a, b, c):
       for i in pypto.loop(1):
           t = a + 1
           c[:] = t + b
   ```

   The framework does not automatically merge `loop(1)` at present. Therefore, in actual use, you are advised to manually merge `loop(1)` to improve efficiency.
