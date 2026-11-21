# pypto.loop

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

Defines a loop operation to implement the functionality of a for loop in Python.

## Prototype

```python
loop(stop: SymInt, /, **kwargs) -> Iterator[SymInt]
loop(start: SymInt, stop: SymInt, step: Optional[SymInt] = 1, /, **kwargs) -> Iterator[SymInt]
```

## Parameters

| Parameter           | Input/Output| Description                                                                |
|-------------------|-----------|----------------------------------------------------------------------|
| start             | Input     | Start value of the loop.|
| stop              | Input     | End value of the loop.|
| step              | Input     | Step of each loop.|
| **kwargs          | Input     | - name(str): Name of the loop identifier. The default value is f"loop_{loop_idx}".<br> - idx_name(str): Name of the loop index variable. The default value is f"loop_idx_{loop_idx}".<br> - submit_before_loop(bool): Whether to submit the computation before the loop starts. The default value is False. If this parameter is enabled, the accumulated computation tasks are forcibly submitted to the AI Core for execution before the loop starts.<br> - parallel(bool): Whether to mark the loop as parallel-schedulable. The default value is False. If this parameter is set to True, it indicates that there are no dependencies between iterations of the loop, and the loop can be scheduled and executed in parallel. This parameter must be used together with the device_sched_parallelism configuration item.|

## Return Value

Returns a generator that generates a symbolic integer representing the value of each iteration in sequence.

## Constraints

The loop index variable (such as b_idx) returned by loop is of the SymInt type and cannot be used as a list subscript index.

## Examples

```python
for _ in pypto.loop(0, 10, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
   ...
```
