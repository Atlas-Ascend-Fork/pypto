# pypto.pass\_verify\_print

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

Saves the computation result of a specified tensor to a data file when the Verify feature for precision debugging is enabled.

## Prototype

```python
pass_verify_print(*values, cond: Union[int, SymbolicScalar] = 1) -> None
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| *values | Input     | Definition: Data or information to be printed.<br> Note: pypto.Tensor: Prints the tensor data. int/pypto.SymbolicScalar: Prints the corresponding value. Other Python object: Prints the corresponding string representation.<br> Type: List[pypto.Tensor,int,pypto.SymbolicScalar,Object]<br> Value range: N/A<br> Default value: N/A|
| cond    | Input     | Definition: Condition that the printed data must meet.<br> Note: If the expression result is 1, the specified data is printed. If the expression result is 0, the data is not printed. This parameter can be omitted. If this parameter is omitted, the default value is used. None cannot be explicitly passed.<br> Type: Union[int,pypto.SymbolicScalar]<br> Value range: [0, 1]<br> Default value: 1|

## Return Value

None

## Constraints

This function only takes effect after pypto.set_verify_options(enable_pass_verify=True) is set.

## Examples

```python
verify_options = {
        "enable_pass_verify": True,
      }

@pypto.frontend.jit(verify_options=verify_options)
def user_kernel(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor):
    ...
    for idx in pypto.loop(10):
        t0 = pypto.tensor(...)
        t1 = pypto.tensor(...)
        t2 = pypto.SOME_OP1(t0, t1)
        pypto.pass_verify_print(t2)
        t3 = pypto.SOME_OP2(t0, t2)
        pypto.pass_verify_print(t3, cond=(idx == 5))
         ...
```
