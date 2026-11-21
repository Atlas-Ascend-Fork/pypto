# pypto.loop\_unroll

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

pypto.loop\_unroll is a loop iterator function that supports loop unrolling. It is similar to pypto.loop, but adds the unroll\_list parameter to support multiple unrolling modes.

## Prototype

```python
loop_unroll(start: SymInt = 0, stop: SymInt, step: SymInt = 1, *, name: str = None, idx_name: str = None, unroll_list: List[int] = None, submit_before_loop: bool = False) -> Iterator[Tuple[SymInt, int]]
```

## Parameters

| Parameter           | Input/Output| Description                                                                |
|-------------------|-----------|----------------------------------------------------------------------|
| *args             | Input     | There are three optional parameters: start (start value of the loop), stop (end value of the loop), and step (loop step). The three parameters can be written in the following three forms:<br> - Single-argument format: stop(SymInt). The default start value is 0, and the default step value is 1. This is equivalent to loop_unroll(0, stop, 1).<br> - Two-argument format: start(SymInt), stop(SymInt). This is equivalent to loop_unroll(start, stop, 1).<br> - Three-argument form: start(SymInt), stop(SymInt), step(SymInt). This is equivalent to loop_unroll(start, stop, step).|
| **kwargs          | Input     | - name(str): Name of the loop identifier. The default value is f"loop_{loop_idx}".<br> - idx_name(str): Name of the loop index variable. The default value is f"loop_idx_{loop_idx}".<br> - unroll_list(List[int]): Set of loop factors to be unrolled. The default value is an empty set. The loop provides several unrolling modes defined in unroll_list. When the unrolling count is n, the loop step becomes step*n, and the loop body is executed n times per iteration. A different code path is generated for each unrolling count.<br> - submit_before_loop (bool): Whether to submit the computation before the loop starts. The default value is False. If this parameter is enabled, the accumulated computation tasks are forcibly submitted to the AI Core for execution before the loop starts.|

## Return Value

Returns an iterator. Each iteration generates a tuple (idx, unroll\_factor\), where idx indicates the index of the current loop, and unroll\_factor indicates the selected unrolling mode.

## Constraints

- The unrolling factor list is sorted and deduplicated, and always contains 1.
- The unrolling factors are sorted in descending order.
- Each unrolling factor generates a sub-loop.
- If loop_unroll is used in multi-layer loops and unroll_list is configured, the number of graphs generated after compilation will increase significantly, affecting the compilation performance.

## Examples

```python
for idx, unroll_factor in pypto.loop_unroll(0, 10, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx", unroll_list=[1, 2, 4]):
   ...
```
