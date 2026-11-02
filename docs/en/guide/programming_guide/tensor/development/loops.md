# Looping and Data Tiling

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T02:34:21.641Z pushedAt=2026-10-27T09:03:58.646Z nl=2 -->

In the previous introduction, after tiling configuration, the input and output of a tensor are tiled and processed in the core according to the tiling configuration scheme. For example, if the original tensor is (1, 32, 1, 256) and the tiling scheme is configured as (1, 4, 1, 64), the core will perform tiling and loop processing according to (1, 4, 1, 64).

If the data volume increases and the tensor configuration becomes \(32, 32, 1, 256\), you can add loop logic for the first axis (shape[0] or Batch axis) through pypto.loop, enabling the framework to expand multiple batches for parallel processing.

## Basic Loop Structure

```python
SHAPE = (32, 32, 1, 256)

@pypto.frontend.jit
def add_kernel(
    input0: pypto.Tensor(SHAPE, pypto.DT_FP32),
    input1: pypto.Tensor(SHAPE, pypto.DT_FP32),
    out: pypto.Tensor(SHAPE, pypto.DT_FP32),
    val: int
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    #calculate the loop parameters
    b, n, s, d = SHAPE
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        t0_sub = input0[b_offset:b_offset_end, ...]
        t1_sub = input1[b_offset:b_offset_end, ...]
        t3_sub = t0_sub + t1_sub
        out[b_offset:b_offset_end, ...] = t3_sub + val
```

The complete API parameters of pypto.loop used for looping are as follows:

```python
for idx in pypto.loop(start, end, step, name="label", idx_name="idx_label", submit_before_loop=False)
```

- Parameters:
    - start, end, step: Optional parameters that support flexible configuration of the loop range.
    - name, idx\_name: Loop identifier and index variable name, used for debugging.
    - submit\_before\_loop: Controls the execution order of the loop.

It can also be simplified as needed to:

```python
for idx in pypto.loop(start, end, step)   # Without a loop name label
for idx in pypto.loop(start, end)         # Default step=1
for idx in pypto.loop(end)                # Default start=0, step=1
```

For a complete example of slicing syntactic sugar, see [loop.py](../../../../../../examples/02_intermediate/controlflow/loop/loop.py).

## Using view/assemble APIs to Process Loop Structures

The preceding example shows how to use Python slicing \[:, :, :\] in a loop to extract small blocks of data for computation. After the computation is complete, the data is output. Alternatively, you can use the pypto.view API to extract small blocks of data for computation, and then use pypto.assemble to assemble and output the data after the computation is complete.

```python
SHAPE = (32, 32, 1, 256)

@pypto.frontend.jit
def add_kernel(
    input0: pypto.Tensor(SHAPE, pypto.DT_FP32),
    input1: pypto.Tensor(SHAPE, pypto.DT_FP32),
    out: pypto.Tensor(SHAPE, pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    #calculate the loop parameters
    b, n, s, d = SHAPE
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        t0_sub = pypto.view(input0, [tile_b, n, s, d], [b_offset, 0, 0, 0])
        t1_sub = pypto.view(input1, [tile_b, n, s, d], [b_offset, 0, 0, 0])
        t3_sub = t0_sub + t1_sub
        pypto.assemble(t3_sub, [b_offset, 0, 0, 0], out)
```

For a complete example of the view/assemble APIs, see [add_scalar_loop_view_assemble.py](../../../../../../examples/01_beginner/transform/add_scalar_loop_view_assemble.py).

## Data Dependency and Loop Order

By default, pypto.loop unrolls and distributes iterations to multiple cores for parallel processing, which is suitable for scenarios without data dependency. If there is data dependency between loops (for example, the output of the previous loop serves as the input of the next loop), set submit\_before\_loop = True to ensure that the result of each loop iteration is written back to the tensor before the next loop starts:

```python
for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx", submit_before_loop=True):
```
