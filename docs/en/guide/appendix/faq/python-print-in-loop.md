# Using the Python print Function in a Loop

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:17:37.013Z pushedAt=2026-10-27T02:43:51.462Z nl=2 -->

## Symptoms

```python
@pypto.frontend.jit
def add_kernel_0(a, b, c):
    for i in pypto.loop(20):
        print("i = ", i)
        c[:] = a + b
>>>
i = 0

@pypto.frontend.jit
def add_kernel_1(a, b, c):
    for i in pypto.loop(20):
        print("i = ", i)
        if pypto.cond(i == 0):
            c[:] = a + b
        else:
            c[:] = a - b
>>>
i = 0
i = 1
```

## Possible Causes

The user operator describes the graph construction process rather than the actual execution logic. During the graph construction phase, loop execution is used only to traverse all execution paths.

- The sample code `add_kernel_0` contains only one execution path, so the loop executes only once.
- In `add_kernel_1`, there are two paths (if/else), so the loop executes twice.

## Solution

N/A
