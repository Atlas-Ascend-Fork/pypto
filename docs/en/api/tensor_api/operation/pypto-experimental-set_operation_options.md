# pypto.experimental.set\_operation\_options

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
- Kirin 9030 processors: Not supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Not supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Not supported
<!-- end id6 -->

## Description

This API is the core part of the runtime dynamic configuration management feature provided by the compilation framework. It converts the parameters that were originally statically written in the configuration file tile\_fwk\_config.json into dynamic, programmable instructions.

## Prototype

```python
set_operation_options(*, combine_axis: Optional[bool] = None)
```

## Parameters

| Parameter              | Input/Output| Description                                                                |
|----------------------|-----------|----------------------------------------------------------------------|
| combine_axis         | Input     | **Meaning**: Implements broadcast inline for the last axis during the code generation phase.<br>**Description**: For the (32,1) + (32,128) operation, you do not need to broadcast (32,1) to (32,128) first. Instead, you can use the brcb instruction to extend (32,1) to (32,8) and then perform the (32,8) + (32,128) operation. The prerequisite is that (32,1) must be contiguous.<br>**Type**: bool<br>**Value range**: {True, False}|

## Return Value

void: The Set method does not return any value. The setting takes effect immediately after the operation is successful.

## Constraints

- Application scenario: For last-axis broadcasting, the last axis of the input must be contiguous; otherwise, the feature does not take effect. If the preceding node is a last-axis reduce, the reduce API can guarantee this. If the preceding node is COPY_IN, contiguity in gm must be ensured at the frontend.
- Type safety: The type of the passed combine_axis must exactly match the type defined for the parameter; otherwise, undefined behavior or runtime errors may occur.
- Scope: The parameter setting is local and only affects the compilation process within the current jit/loop. If it is not set, the setting in the upper-level jit/loop scope is inherited.

## Examples

```python
pypto.experimental.set_operation_options(combine_axis=True)
```
