# SymbolicScalar Does Not Support Increment Inside a Loop

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:19:16.370Z pushedAt=2026-10-27T02:45:02.962Z nl=2 -->

## Symptoms

```python
@pypto.frontend.jit
def add_kernel_1(a, b, c):
    count = 0
    for i in pypto.loop(20):
        count = count + 1
```

When `i = 1` is actually executed, `count` does not increment from 0 to 20 sequentially as the user expects.

## Possible Causes

The current PyPTO framework captures only the user's tensor operations, not scalar operations. Therefore, `count` is not treated as a variable. Currently, only the loop variable supports increment.

## Solution

Use the loop variable to express the increment logic.
