# pypto.loop

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:43:03.881Z pushedAt=2026-10-28T11:57:29.650Z nl=2 -->

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

Defines a loop operation that implements the functionality of a **for** loop in Python.

## Prototype

```python
loop(stop: SymInt, /, **kwargs) -> Iterator[SymInt]
loop(start: SymInt, stop: SymInt, step: Optional[SymInt] = 1, /, **kwargs) -> Iterator[SymInt]
```

## Parameters

| Parameter            | Input/Output | Description                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| start             | Input      | Start value of the loop. |
| stop              | Input      | End value of the loop. |
| step              | Input      | Step size of each iteration. |
| **kwargs          | Input      | - name (str): Loop identifier name, defaulting to f"loop_{loop_idx}".<br> - idx_name (str): Name of the loop index variable, defaulting to f"loop_idx_{loop_idx}".<br> - submit_before_loop (bool): Whether to submit computation before the loop starts, defaulting to False. When enabled, accumulated computation tasks are force-submitted to the AICore for execution before the loop starts.<br> - parallel (bool): Whether to mark the loop as parallel-schedulable, defaulting to False. When set to True, the iterations of the loop have no dependencies and can be scheduled for parallel execution. Use together with the device_sched_parallelism configuration item.|

## Return Value

Returns a generator that yields a symbolic integer representing the value of each iteration in sequence.

## Constraints

The loop index variable returned by **loop** (such as **b_idx**) is of the **SymInt** type and cannot be used as a list subscript index.

## Examples

```python
for _ in pypto.loop(0, 10, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
   ...
```
