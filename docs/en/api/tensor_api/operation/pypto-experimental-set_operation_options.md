# pypto.experimental.set\_operation\_options

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:35:56.244Z pushedAt=2026-10-28T11:57:29.889Z nl=2 -->

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

This API is the core part of the runtime dynamic configuration management feature provided by the compilation framework. It transforms parameters that were originally statically written in the configuration file **tile\_fwk\_config.json** into dynamic, programmable instructions.

## Prototype

```python
set_operation_options(*, combine_axis: Optional[bool] = None)
```

## Parameters

| Parameter            | Input/Output | Description                                                                 |
|----------------------|--------------|----------------------------------------------------------------------|
| combine_axis         | Input        | **Meaning**: Implements last-axis broadcast inline during the code generation phase.<br>**Description**: For a binary operation (32,1) + (32,128), instead of first broadcasting (32,1) to (32,128), (32,1) is expanded to (32,8) through the brcb instruction, and then (32,8) + (32,128) is performed. The prerequisite is that (32,1) must be contiguous.<br>**Type**: bool<br>**Value range**: {True, False} |

## Return Value

**void**: The **Set** method has no return value. The setting takes effect immediately upon success.

## Constraints

- Usage scenario: For last-axis broadcasting, the last axis of the input must be contiguous; otherwise, the feature does not take effect. If the preceding node is a last-axis reduce, the reduce API can guarantee this. If the preceding node is COPY_IN, contiguity in gm must be ensured at the frontend.
- Type safety: The type of the passed combine_axis must exactly match the type defined for the parameter; otherwise, undefined behavior or runtime errors may occur.
- Scope: The parameter setting is local and only affects the compilation process within the current jit/loop. If it is not set, the setting in the upper-level jit/loop scope is inherited.

## Examples

```python
pypto.experimental.set_operation_options(combine_axis=True)
```
