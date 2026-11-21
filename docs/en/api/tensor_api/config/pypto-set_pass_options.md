# pypto.set\_pass\_options

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
<!-- npu="9030" id24 -->
- Kirin 9030 processors: Supported
<!-- end id24 -->
<!-- npu="x90" id25 -->
- Kirin X90 processors: Supported
<!-- end id25 -->
<!-- npu="9050" id26 -->
- Kirin 9050 processors: Supported
<!-- end id26 -->

## Description

Modifies optimization parameters of a Pass. It is mainly used to dynamically modify the runtime parameter configuration of a specific Pass based on the specified tuning strategy in the compilation process, thereby implementing refined control and debugging.

## Prototype

```python
set_pass_options(*,
                     vec_nbuffer_setting: Optional[Dict[str, int]] = None,
                     cube_l1_reuse_setting: Optional[Dict[str, int]] = None,
                     cube_nbuffer_setting: Optional[Dict[str, int]] = None,
                     sg_set_scope: Optional[Union[int, Tuple[int, bool, bool]]] = None,
                     sg_set_ooo_scope: Optional[int] = None,
                     experimental: Optional[Dict[str, int]] = None,
                     ooo_sched_mode: Optional[str] = None,
                     sg_set_tunevf_mode: Optional[int] = None,
                     )
```

## Parameters

| Parameter                 | Input/Output| Description                                                                |
|-------------------------|-----------|----------------------------------------------------------------------|
| vec_nbuffer_setting     | Input     | Meaning: Graph merging parameter, which is used to configure the number of vector subgraphs with the same structure to be merged.<br> Description: This parameter is used to merge vector subgraphs with the same structure.<br><br>Type: dict[str, int] The following two key formats are supported:<br> 1. **Function-level key**: The string `"func{magic}_{order}"` or `"DEFAULT"` is used for refined configuration between different root functions. `func{magic}_{order}` matches a specific homogeneous subgraph group (hashorder=order) of a specific function (funcMagic=magic). `DEFAULT` matches all subgraph groups that are not explicitly specified. With the configuration, the graph merging information is directly displayed in the hashOrder-hint field of the swimlane diagram (including subGraphCount, which indicates the number of subgraphs before merging. The merging degree can be matched based on the number of cores). For details, see **Function-Level Key Configuration** below.<br> 2. **Semantic label key**: Any string without the prefix of func is used to control the graph merging granularity based on the semantic label. (For details, see **Semantic Label Key Configuration** below.) It can coexist with the function-level key.<br> Options:<br> {"DEFAULT": 1}: Skip vector subgraph merging.<br> {} (empty dictionary): Automatic merging. The merging granularity is automatically computed based on the number of available vector cores.<br> {"DEFAULT": N, "func8_0": N2}: Manual merging. The default granularity is N, and the granularity of the subgraph group corresponding to func8_0 is N2.<br> Default value: {} (empty dictionary)<br> Affected Pass scope: NBufferMerge|
| cube_l1_reuse_setting | Input| Meaning: Graph merging parameter, which is used to configure the number of subgraphs that repeatedly move the same global memory (GM) data to be merged.<br> Description: This parameter is used to merge subgraphs that contain Cube computation. When this parameter and cube_nbuffer_setting are configured together, the subgraphs are merged first, and then the subgraphs related to cube_nbuffer_setting are merged.<br><br>Type: dict[str, int] The following two key formats are supported:<br> 1. **Function-level key**: The string `"func{magic}_{order}"` or `"DEFAULT"` is used for refined configuration between different root functions. The graph merging information is directly displayed in the hashOrder-hint field of the swimlane diagram (including subGraphCount, which indicates the number of subgraphs before merging. The merging degree can be matched based on the number of cores). For details, see **Function-Level Configuration** below.<br> 2. **Semantic label key**: Control by semantic label (For details, see **Semantic Label Key Configuration** below.)<br> Options:<br>{"DEFAULT": 1}: Skip L1Reuse merging.<br> {} (empty dictionary): Automatic merging<br> {"DEFAULT": N, "func8_0": N2}: Manual merging. The default granularity is N.<br> Default value: {} (empty dictionary)<br> Affected Pass scope: L1CopyInReuseMerge|
| cube_nbuffer_setting    | Input     | Meaning: Graph merging parameter, which is used to configure the number of AIC subgraphs with the same structure to be merged.<br> Description: This parameter is used to merge AIC subgraphs with the same structure. When this parameter and cube_l1_reuse_setting are configured together, the subgraphs related to cube_l1_reuse_setting are merged first, and then the subgraphs related to this parameter are merged.<br><br>Type: dict[str, int] The following two key formats are supported:<br> 1. **Function-level key**: The string `"func{magic}_{order}"` or `"DEFAULT"` is used for refined configuration between different root functions. The graph merging information is directly displayed in the hashOrder-hint field of the swimlane diagram (including subGraphCount, which indicates the number of subgraphs before merging. The merging degree can be matched based on the number of cores). For details, see **Function-Level Configuration** below.<br> 2. **Semantic label key**: Control by semantic label (For details, see **Semantic Label Key Configuration** below.)<br> Options:<br>{"DEFAULT": 1}: Skip AIC subgraph merging.<br> {} (empty dictionary): Explicitly enable automatic merging.<br> {"DEFAULT": N, "func8_0": N2}: Manual merging. The default granularity is N.<br>Default value: {"DEFAULT": 1}. AIC subgraph merging is skipped by default.<br> Affected Pass scope: L1CopyInReuseMerge|
| sg_set_scope            | Input     | Meaning: Manually controls the subgraph tiling parameters.<br> Description: By assigning a scope to an operation, adjacent operations with the same scope ID (not -1) are forcibly merged into the same subgraph, overriding the automatic partitioning result of the tiling algorithm.<br> Type: `Tuple[int, bool, bool]` or `int`<br> **Format of the tuple**: `(scope_id, allow_parallel_merge, allow_cross_scope_merge)`. The meaning of each field is as follows:<br> - `scope_id` (int): Scope ID. The value ranges from –1 to 2147483647. Adjacent operations with the same scope ID are merged into the same subgraph. The value -1 indicates that the operations are not involved in scope merging, and the subgraph partitioning is determined by the tiling algorithm.<br> - `allow_parallel_merge` (bool): Controls the merging mode of operations with the same scope_id. The value can be True or False.<br>&emsp;&emsp;False (default): Only operations that are connected upstream and downstream can be merged into the same subgraph. That means operations can be merged into the same subgraph only when the output of operation A is used as the input of operation B.<br>&emsp;&emsp;True: Operations with the same scope ID in parallel branches (without data dependency) can be merged into the same subgraph.<br> - `allow_cross_scope_merge` (bool): Controls whether a subgraph with a scope can be merged with a subgraph without a scope (scope_id=-1) to expand the scope subgraph. The value can be True or False.<br>&emsp;&emsp;False (default): A subgraph with a scope is independent and is not merged with other subgraphs.<br>&emsp;&emsp;True: A subgraph with a scope can be merged with a subgraph whose scope ID is -1 (scope_id=-1). Subgraphs with different scope IDs cannot be merged.<br> **int format**: When a single int is passed, it is equivalent to `(scope_id, False, False)`, which means that only scope_id is set. Parallel branch merging and cross-scope merging are not allowed.<br> Default value: (-1, False, False)<br> Affected Pass scope: GraphPartition<br> Suggestions: (1) The view operation and its corresponding compute operation must be configured with the same scope_id. (2) The Reshape operation is special. In some scenarios, it is independently formed as a subgraph. In this case, manually control of the merging behavior may fail.|
| sg_set_ooo_scope            | Input     | Meaning: Controls the OoO scheduling in the MIX subgraph.<br> Description: By assigning an OoO scope (ooo_scope) to an operation, adjacent operations with the same ooo_scope_id (not -1) are forcibly merged into the same ooo_task. This ensures that adjacent operations with the same ooo_scope_id (not -1) are as close as possible in the pipeline generated by OoO scheduling. Parallel branch merging and merging across OoO scopes are not allowed. Operations with different loop counts cannot be merged.<br> Type: `int`, that is, setting ooo_scope_id<br> Default value: **-1**<br> Value range: -1 or 1 to 10000<br> `ooo_scope_id`: ooo_scope identifier. Adjacent operations with the same ooo_scope_id are merged into the same ooo_task. The value -1 indicates that the operations are not involved in ooo_scope merging, and the ooo_task partitioning is determined by the tiling algorithm.<br> Affected Pass scope: OoOSchedule<br> Configuration requirement: This function takes effect only for the MIX subgraph. If this function is used together with loop unroll, the unroll value cannot be greater than 10000. This parameter will be discarded in later versions.|
| ooo_sched_mode            | Input     | Meaning: Controls the OoO scheduling in the MIX subgraph.<br> Description: Sets the pipeline scheduling mode of ooo_task in the MIX subgraph.<br> Type: `str`<br> Default value: ""<br> Value range: {"", "GAPMIN", "HLF"}<br> Affected Pass scope: OoOSchedule<br> Configuration description: If the value is "" (default), the scheduling based on topological order traversal (GapMin scheduling) and local search (local-search) is used. If the value is "GAPMIN", only the GapMin scheduling is performed, and the local search is skipped. If the value is "HLF", the Highest Level First (HLF) scheduling is used (EFT insertion scheduling is performed after tasks are sorted in descending order by the length of their longest path to the sink).|
| sg_set_tunevf_mode           | Input     | Meaning: Controls the behavior of the Vector Fusion (VF) tuning Pass.<br> Description: This parameter controls the execution behavior of the TuneTileOpSeqForVF and TuneSyncForVF Passes.<br> Type: `int`<br> Default value: 0<br> Value range: {0, 1, 2}<br> - 0: Balanced mode. The operator sequence is automatically adjusted based on the operator sequence output by the OoO Pass, and the overall performance gains of the pipeline and VF are automatically balanced.<br> - 1: Instruction pipeline first mode. The operation sequence determined by the OoO Pass is not changed.<br> - 2: VF first mode. The performance gains are not evaluated through performance modeling. The operation sequence is adjusted to ensure that VF can be performed in a larger scope. This mode affects only the NeedAdjustOpSeq judgment in TuneSyncForVF and does not affect TuneTileOpSeqForVF.<br> Affected Pass scope: TuneTileOpSeqForVF and TuneSyncForVF|

## Return Value

None

## Constraints

- Setting time: This function does not need to be called before graph build starts. It can be set at any time.
- Type safety: Ensure that the types of the passed parameter values are the same as those defined for the parameters. Otherwise, undefined behavior or runtime errors may occur.
- Scope: The parameter settings are local and affect only the compilation process of the current JIT or loop. If the parameters are not set, the upper-layer scope is inherited.
- sg_set_scope consistency constraint: All operations with the same scope ID must be set with the same `allow_parallel_merge` and `allow_cross_scope_merge`. Otherwise, a compilation error is reported.
- When scope_id is -1, `allow_parallel_merge` and `allow_cross_scope_merge` must be set to False.
- Subgraphs with different scope IDs cannot be merged. `allow_cross_scope_merge` only controls the merging of subgraphs with scopes and subgraphs without scopes (scope_id=-1).
- Product differences:
   <!-- npu="950" id101 -->
   - Ascend 950PR&950DT products: vec_nbuffer_setting, cube_l1_reuse_setting, cube_nbuffer_setting, sg_set_scope, sg_set_ooo_scope, ooo_sched_mode and sg_set_tunevf_mode are supported. sg_set_scope supports the scope configuration in the pure Vector, pure Cube, and hybrid CV scenarios.
   <!-- end id101 -->
   <!-- npu="A3" id102 -->
   - Atlas A3 products: vec_nbuffer_setting, cube_l1_reuse_setting, and cube_nbuffer_setting are supported. sg_set_scope supports the scope configuration of pure Vector or pure Cube, but does not support the scope configuration in the hybrid CV scenario. sg_set_ooo_scope, ooo_sched_mode, and sg_set_tunevf_mode are not supported (because CV Mix graph merging is not supported).
   <!-- end id102 -->
   <!-- npu="910b" id103 -->
   - Atlas A2 products: vec_nbuffer_setting, cube_l1_reuse_setting and cube_nbuffer_setting are supported. sg_set_scope supports the scope configuration of pure Vector or Cube, but does not support the scope configuration in the hybrid CV scenario. sg_set_ooo_scope, ooo_sched_mode and sg_set_tunevf_mode are not supported (because CV Mix graph merging is not supported).
   <!-- end id103 -->
   <!-- npu="9030" id104 -->
   - Kirin 9030 processors: sg_set_ooo_scope, ooo_sched_mode, and sg_set_tunevf_mode are supported. vec_nbuffer_setting, cube_l1_reuse_setting, cube_nbuffer_setting, and sg_set_scope are not supported (Kirin processors do not support graph sharding).
   <!-- end id104 -->
   <!-- npu="x90" id105 -->
   - Kirin X90 processors: sg_set_ooo_scope, ooo_sched_mode, and sg_set_tunevf_mode are supported. vec_nbuffer_setting, cube_l1_reuse_setting, cube_nbuffer_setting, and sg_set_scope are not supported (Kirin processors do not support graph sharding).
   <!-- end id105 -->
   <!-- npu="9050" id106 -->
   - Kirin 9050 processors: sg_set_ooo_scope, ooo_sched_mode, and sg_set_tunevf_mode are supported. vec_nbuffer_setting, cube_l1_reuse_setting, cube_nbuffer_setting, and sg_set_scope are not supported (Kirin processors do not support graph sharding).
   <!-- end id106 -->

## Examples

```python
   # Function-level configuration (in func{magic}_{order} format)
   pypto.set_pass_options(
       vec_nbuffer_setting={"DEFAULT": 4, "func8_0": 1, "func8_1": 1},
       cube_l1_reuse_setting={"DEFAULT": 4, "func8_0": 1, "func8_1": 1},
       cube_nbuffer_setting={"DEFAULT": 4, "func8_0": 1, "func8_1": 1})

   # DEFAULT only
   pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 2})

   # Semantic label key configuration (It can coexist with the function-level key.)
   pypto.set_semantic_label("V1")
   sij_scale = pypto.mul(sij, softmax_scale)
   pypto.set_semantic_label("")
   ...
   pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 2, "V1": 1})
```

### Configuration Description of the dict Type (Function-Level Key/Semantic Label Key)

### Function-Level Key configuration (func{magic}\_{order})

#### Function Overview

By using keys in the `"func{magic}_{order}"` format, you can set the merging granularity for a **specific isomorphic subgraph group** (hashorder) of a **specific function** to implement refined configuration between different root functions. The configured hashOrder and subGraphCount information is directly displayed in the hashOrder-hint field of the swimlane diagram in the format of `l1ReuseInfo hashOrder: func8_0, subGraphCount: 24`. The merging granularity can be matched based on the number of subgraphs and the number of cores.

#### Key-Value Pair Description

Key: String in the format of `"func{magic}_{order}"` or `"DEFAULT"`.<br>

- `"func{magic}_{order}"`: Matches the homogeneous subgraph group whose hashorder is order in the function whose functionMagic is magic.<br>
- `"DEFAULT"`: Matches all homogeneous subgraph groups that are not explicitly specified.<br>

Value (N): Indicates the merging granularity. Every N subgraphs in a homogeneous subgraph group are merged into a new subgraph for execution. N=1 indicates no merging.

#### Format Constraints

- The `func` prefix must be in lowercase.
- magic and order must be integers.
- magic and order are separated by an underscore (`_`).
- Example valid keys: `"func0_0"`, `"func123_5"` and `"func8_1"`
- Example invalid keys: `"Func0_0"` (uppercase F), `"func_0"` (missing magic), and `"func123"` (missing order)

#### Configuration Behavior

When processing the subgraph merging of the current function, Pass follows the logic of "func{magic}\_{order} exact match > default configuration (DEFAULT) > automatic processing".<br>

- Exact match: If both funcMagic and hashorder are hit, the corresponding values are merged based on Value N.<br>
- Default configuration: If the exact match fails but `DEFAULT` exists in the dictionary, the corresponding values are merged based on `DEFAULT`.<br>
- Automatic processing: If the exact match fails and `DEFAULT` does not exist, the merging granularity is automatically computed.<br>

#### Configuration Examples

| Configuration| Description|
|------|------|
|`{"DEFAULT": 1}`|All homogeneous subgraph groups are skipped from merging.|
|`{"DEFAULT": 4, "func8_0": 1, "func8_1": 1}`|By default, four homogeneous subgraphs are merged into a group. The subgraph groups with hashorder 0 and 1 in func8 are skipped from merging.|
|`{"DEFAULT": 2, "func8_1": 4}`|By default, two homogeneous subgraphs are merged into a group. In func8, four subgraphs with hashorder 1 are merged into a group.|
|`{"func8_0": 2}`|In the func8 function, two subgraphs with hashorder 0 are merged into a group. For other homogeneous subgraph groups, the merging granularity is automatically computed based on the number of hardware cores.|

### Semantic Label Key Configuration

#### Function Overview

In addition to function-level keys, `vec_nbuffer_setting`, `cube_l1_reuse_setting`, and `cube_nbuffer_setting` also support string keys, which are semantic label names set using `pypto.set_semantic_label`. String keys allow you to precisely control the merging granularity of the subgraph (multiple subgraphs are allowed) of a specific operation, without considering the hashorder number.

#### String Key-Value Pair Description

Key (label): Semantic label name, which must exactly match `semantic_label` of at least one operation. If no operation is matched, no error is reported during compilation. Only WARN logs are printed, and the key is ignored.<br>
Value (N): Indicates the merging granularity.<br>

#### Priority Mechanism

The priority of string keys is **higher than** the default configuration of function-level keys. The process is as follows:<br>

1. First, determine the basic merging granularity of each homogeneous subgraph group based on the function-level key (`func{magic}_{order}` / `DEFAULT`).<br>
2. Then, directly replace the value of the string key with the merging granularity of the corresponding subgraph group (rather than taking the maximum value).<br>
3. When multiple different string labels point to the same homogeneous subgraph group, the maximum value among these label values is used.<br>

#### Semantic Label Behavior of vec_nbuffer_setting / cube_nbuffer_setting

The string key overrides the merging granularity of the **entire homogeneous subgraph group** corresponding to the operation.

#### Semantic Label Behavior of cube_l1_reuse_setting

Different from `vec_nbuffer_setting` and `cube_nbuffer_setting`, the string key of `cube_l1_reuse_setting` **only takes effect on the subgraphs that contain the operation with the corresponding label**, and is not expanded to the entire homogeneous group. That means only some subgraphs in the homogeneous group may be covered by the string key, and the value of the function-level key is retained for other subgraphs.

#### Semantic Label Configuration Examples

| Configuration                               | Description                                                                |
|-------------------------------------|----------------------------------------------------------------------|
|{"DEFAULT": 2, "V1": 1}|By default, the default merging granularity of all homogeneous subgraph groups is 2. However, the merging granularity of the homogeneous subgraph group with the V1 label is replaced with 1.|
|{"V1": 3}|The merging granularity of the homogeneous subgraph group with the V1 label is 3. The merging granularity of other homogeneous subgraph groups is automatically computed.|
|{"DEFAULT": 2, "V1": 1, "V2": 3}|The default merging granularity is 2. The merging granularity of the group with the V1 label is replaced with 1, and that of the group with the V2 label is replaced with 3. If a group has both V1 and V2 operations, the value is max(1, 3) = 3.|

#### Configuration Examples

```python
   # Configuration of both function-level and semantic label keys
   pypto.set_semantic_label("V1")
   sij_scale = pypto.mul(sij, softmax_scale)
   pypto.set_semantic_label("") # By changing the semantic label, you can precisely control that only the semantic label of the mul operation is "V1".
   ...
   pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 2, "V1": 1})

   # Configuration of only the semantic label key
   pypto.set_pass_options(cube_l1_reuse_setting={"MM1": 4})
```

### sg_set_scope Configuration Description

#### Configuration Examples

```python
# int format: Equivalent to (10, False, False), only setting scope_id
pypto.set_pass_options(sg_set_scope=10)

# tuple format: scope_id=1. Parallel branch merging is allowed, but cross-scope merging is not allowed.
pypto.set_pass_options(sg_set_scope=(1, True, False))

# tuple format: scope_id=2. Merging with subgraphs without scopes is allowed.
pypto.set_pass_options(sg_set_scope=(2, False, True))

# Restore to the default setting (not involved in scope merging, automatically determined by the graph merging algorithm)
pypto.set_pass_options(sg_set_scope=-1)
```

#### Typical Scenarios

<!-- npu="A3" id22 -->
- Atlas A3 products: Scenario 2 is not supported.
<!-- end id22 -->
<!-- npu="910b" id23 -->
- Atlas A2 products: Scenario 2 is not supported.
<!-- end id23 -->

##### Scenario 1: The entire computational graph is not tiled

When the entire computational graph is not tiled, multiple parallel branches are generated due to data tiling. These branches have no direct data dependency and are split into independent subgraphs by the default tiling algorithm. You are advised to set `sg_set_scope=(scope_id, True, False)` and use `allow_parallel_merge=True` to merge the parallel branches with the same scope_id into one subgraph.

##### Scenario 2: In the hybrid CV scenario, construct a MIX subgraph to reduce data movement on the GM

When Vec operations are performed before and after Cube operations, the objective is to construct a MIX subgraph that contains both Cube and Vec operations to avoid repeated transfer of intermediate results on the GM. There are two cases based on whether the scope boundary is specified:

**Scenario 2.1: Scope boundary specified**

When the boundary between the Cube operations and the adjacent Vec operations can be clearly specified, use `(scope_id, False, False)` to mark the boundary, so that the Cube operations and the adjacent Vec operations are forcibly included in the same subgraph to form a MIX subgraph.

```python
# Clearly mark the Cube operations and the adjacent Vec operations as the same scope to form a MIX subgraph.
pypto.set_pass_options(sg_set_scope=(1, False, False))
# ... Cube operation...
# ... Adjacent Vec operation...
pypto.set_pass_options(sg_set_scope=-1)
```

**Scenario 2.2: Scope boundary not specified, only the CV merging boundary marked**

When the boundary cannot be clearly specified, but the Vec subgraphs before and after the Cube operations need to be merged with the Cube subgraph to form a MIX subgraph, use `(scope_id, False, True)` to mark the Cube operations and the adjacent Vec operation as the merging anchor. With `allow_cross_scope_merge=True`, the Vec subgraph in the scope can be merged with the Vec subgraphs before and after the scope (scope_id=-1) to form a larger subgraph, reducing GM data movement. In addition, the Vec subgraph can be merged with the Cube subgraph to form a MIX subgraph.

```python
# Pre-Vec operation (scope_id=-1, automatically determined by the tiling algorithm)
vec_out = some_vec_op(x)

# Mark the CV merging anchor, which allows merging with adjacent subgraphs without scope.
pypto.set_pass_options(sg_set_scope=(1, False, True))
# ... Cube operation...
matmul_result = pypto.matmul(vec_out, w)
# ... Adjacent Vec operation...
pypto.set_pass_options(sg_set_scope=-1)

# Subsequent Vec operations (scope_id=-1), which allow merging with the upper scope subgraph to form a larger subgraph.
result = other_vec_op(add_result)
```

### sg_set_ooo_scope Configuration Description

#### Configuration Examples

```python
# Setting ooo_scope_id
pypto.set_pass_options(sg_set_ooo_scope=10)

# Restore the default setting (not involved in ooo_scope merging, automatically determined by the tiling algorithm)
pypto.set_pass_options(sg_set_ooo_scope=-1)
```

#### Typical Scenario: Controlling the Execution Order of Operations

As shown in the example below, by placing mul and exp in the same ooo_scope, you can ensure that exp is executed before add in the OoO scheduling result.

```python
# Because ooo_scope needs to be enabled in the MIX subgraph, sg_set_scope is used to construct the MIX subgraph.
pypto.set_pass_options(sg_set_scope=1)
# Cube operation
matmul_result = pypto.matmul(a, b)
pypto.set_pass_options(sg_set_ooo_scope=1)
# Vec operation (ooo_scope_id=1)
c = pypto.mul(matmul_result, scale)
pypto.set_pass_options(sg_set_ooo_scope=-1)
d = pypto.add(c, bias)
pypto.set_pass_options(sg_set_ooo_scope=1)
# Vec operation (ooo_scope_id=1)
e = pypto.exp(c)
pypto.set_pass_options(sg_set_ooo_scope=-1)
pypto.set_pass_options(sg_set_scope=-1)
```

### Description of sg_set_tunevf_mode

#### Configuration Examples

```python
# Instruction pipeline first mode. The operation sequence determined by the OoO Pass is not changed.
pypto.set_pass_options(sg_set_tunevf_mode=1)

# VF first mode. The performance gains are not evaluated through performance modeling. The operation sequence is adjusted to ensure that VF can be performed in a larger scope.
pypto.set_pass_options(sg_set_tunevf_mode=2)

# Restore the default behavior, which is the balanced mode.
pypto.set_pass_options(sg_set_tunevf_mode=0)
```
