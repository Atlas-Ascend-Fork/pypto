# pypto.pass\_verify\_save

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-26T17:53:01.167Z pushedAt=2026-10-28T11:57:30.195Z nl=2 -->

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

When the Verify feature for precision debugging is enabled, use this API to save the result to be computed for a specified tensor to a data file.

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

| Parameter | Input/Output | Description |
|----------|-----------|----------------------------------------------------------------------|
| tensor   | Input      | Meaning: pypto.Tensor in a pypto kernel function. <br> Description: Tensor object to be saved. <br> Type: pypto.Tensor <br> Value Range: N/A |
| fname    | Input      | Meaning: File name template that defines the file name prefix for saving the tensor. The tensor memory dump is saved to {fname}.data, and the tensor metadata (shape, dtype) is saved to {fname}.csv. The save path is: ${work_path}/output/output_*/tensor/ <br> Description: str: simple file name prefix; a string to be matched that contains "$NAME": replaces $NAME with the value corresponding to NAME in kwargs, and then uses the replaced string as the file name prefix. <br> Type: Union[str, SymbolicScalar, int] <br> Value Range: N/A |
| cond     | Input      | Meaning: Specifies the condition for printing data. <br> Description: If the expression evaluates to 1, the specified data is printed; if the expression evaluates to 0, the data is not printed. This parameter can be omitted, in which case the default value is used. Explicitly passing None is not supported. <br> Type: Union[int,pypto.SymbolicScalar] <br> Value Range: 0,1 <br> Default value: 1 |
| **kwargs | Input      | Specifies the value of the string to be matched in the fname parameter. |

## Return Value

None

## Constraints

This function takes effect only after pypto.set_verify_options(enable_pass_verify=True) is set or it is enabled through @pypto.frontend.jit(verify_options={"enable_pass_verify": True}).

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
        # Save t2 to the file t2.*
        pypto.pass_verify_save(t2, 't2-fileprefix')
        t3 = pypto.SOME_OP2(t0, t2)
        # When idx==5, save t3 to the file t3_debug_loop_5.*
        pypto.pass_verify_save(t3, "t3_debug_loop_$idx", cond=(idx == 5), idx=5)
         ...

```
