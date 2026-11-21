# pypto.pass\_verify\_save

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

Saves the result of a specified tensor to be computed to a data file when the Verify feature for precision debugging is enabled.

## Prototype

```python
pass_verify_save(
    tensor: Tensor,
    fname: Union[str, SymbolicScalar, int],
    cond: Union[int, SymbolicScalar] = 1,
    **kwargs: Union[int, SymbolicScalar, pypto_impl.SymbolicScalar]
) -> None
```

## Parameters

| Parameter  | Input/Output| Description                                                                |
|----------|-----------|----------------------------------------------------------------------|
| tensor   | Input     | Definition: pypto.Tensor in the pypto kernel function.<br> Note: Tensor object to be saved.<br> Type: pypto.Tensor<br> Value range: N/A|
| fname    | Input     | Definition: File name template, which defines the prefix of the file name for saving the tensor. The tensor memory dump is saved to {fname}.data, and the tensor metadata (shape and dtype) is saved to {fname}.csv. The saving path is ${work_path}/output/output_*/tensor/.<br> Note: str: Simple file name prefix. For the string to be matched that contains "$NAME", replace $NAME with the value of NAME in kwargs, and then use the replaced string as the file name prefix.<br> Type: Union[str, SymbolicScalar, int]<br> Value range: N/A|
| cond     | Input     | Definition: Condition that the printed data must meet.<br> Note: If the expression result is 1, the specified data is printed. If the expression result is 0, the data is not printed. This parameter can be omitted. If this parameter is omitted, the default value is used. None cannot be explicitly passed.<br> Type: Union[int,pypto.SymbolicScalar]<br> Value range: [0, 1]<br> Default value: 1|
| **kwargs | Input     | Value of the string to be matched in the fname parameter.|

## Return Value

None

## Constraints

This function only takes effect after pypto.set_verify_options(enable_pass_verify=True) is set or @pypto.frontend.jit(verify_options={"enable_pass_verify": True}) is enabled.

## Examples

```python
verify_options = {
        "enable_pass_verify": True
      }

@pypto.frontend.jit(verify_options=verify_options)
def user_kernel(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor):
    ...
    for idx in pypto.loop(10):
        t0 = pypto.tensor(...)
        t1 = pypto.tensor(...)
        t2 = pypto.SOME_OP1(t0, t1)
        # Save t2 to the t2.* file.
        pypto.pass_verify_save(t2, 't2-fileprefix')
        t3 = pypto.SOME_OP2(t0, t2)
        # When the value of idx is 5 (idx==5), save t3 to the t3_debug_loop_5.* file.
        pypto.pass_verify_save(t3, "t3_debug_loop_$idx", cond=(idx == 5), idx=5)
         ...

```
