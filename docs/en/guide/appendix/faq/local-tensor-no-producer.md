# Device-Side Assertion Triggered When a Local Tensor Is Read Without a Producer

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:16:11.912Z pushedAt=2026-10-27T02:42:28.946Z nl=2 -->

## Symptoms

During operator development, if a local tensor is created but not written to before being read within a loop, a device-side assertion or exception may be triggered during compilation or execution on the device. A typical error is as follows:

```text
ASSERTION FAILED: WORKSPACE_ITER_INVALID
Root[...] incast ... slotIndex ... read from empty address.
```

This symptom usually indicates that the slot being read has no producer, meaning the tensor slot has not been written to by any operation.

Minimal example (semantic illustration):

```python
def kernel(x):
    t = pypto.tensor((1,), pypto.DT_FP32, "t")
    for i in range(n):
        _ = t[0]
```

## Possible Causes

The local tensor (such as `t`) is only created but has no write operations (such as slice assignment, `move`, `assemble`, or other writes that can establish a producer) before being read, causing the read path to access a null address and trigger an assertion.

## Solution

You can use either of the following methods to avoid this issue:

1. Perform a valid write operation (such as slice assignment, `move`, or `assemble`) on the local tensor before reading it, ensuring that it has a producer.
2. Write the local tensor inside a loop and use it as an intra-loop tensor, so that it is within the current loop scope. The PyPTO memory management policy will then allocate a memory address for it.
