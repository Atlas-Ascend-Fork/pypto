# pypto.set\_pass\_options

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-26T15:36:25.400Z pushedAt=2026-10-28T11:57:29.634Z nl=2 -->

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

Modifies the **Pass** optimization parameter information. Its main function is to dynamically modify the runtime parameter settings of specific optimization strategies and specific **Pass**es during the compilation process, thereby enabling fine-grained control and debugging.

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

| Parameter | Input/Output | Description                                                                 |
|-------------------------|-----------|----------------------------------------------------------------------|
| vec_nbuffer_setting     | Input      | Meaning: graph merging parameter used to configure the merging count of Vector subgraphs with the same structure. <br> Description: This parameter applies to the merging of Vector subgraphs with the same structure.<br><br>Type: dict[str, int]. Two key formats are supported:<br> ① **Function granularity key**: the string `"func{magic}_{order}"` or `"DEFAULT"`, which enables fine-grained configuration across different root functions. `func{magic}_{order}` matches a specific isomorphic subgraph group (hashorder=order) of a specific function (funcMagic=magic); `DEFAULT` matches all subgraph groups that are not explicitly specified. After configuration, the graph merging information is directly displayed in the hashOrder-hint field of the swimlane diagram (including subGraphCount, which represents the number of subgraphs before merging, and can be used to match the merging intensity based on the number of cores). For details, see the function granularity key configuration description below.<br> ② **Semantic tag key**: any non-func-prefixed string, which controls the graph merge granularity by semantic label (see the semantic label key configuration description below for details). It can coexist with the function granularity key.<br> Value range:<br> {"DEFAULT": 1}: skip Vector subgraph merging <br> {} (empty dictionary): automatic merging, where the merge granularity is automatically calculated based on the number of available Vector cores<br> {"DEFAULT": N, "func8_0": N2}: manual merging, where the default granularity is N and the granularity of the subgraph group corresponding to func8_0 is N2<br> Default value: {} empty dictionary <br> Affected Pass scope: NBufferMerge |
| cube_l1_reuse_setting | Input | Meaning: graph merging parameter used to configure the merging count of subgraphs that repeatedly load the same GM (Global Memory) data.<br> Description: This parameter applies to the merging of subgraphs containing Cube computation. When configured together with cube_nbuffer_setting, this merging is performed first, followed by the merging related to cube_nbuffer_setting.<br><br>Type: dict[str, int]. Two key formats are supported:<br> ① **Function granularity key**: the string `"func{magic}_{order}"` or `"DEFAULT"`, which enables fine-grained configuration across different root functions. The graph merging information is directly displayed in the hashOrder-hint field of the swimlane diagram (including subGraphCount, which represents the number of subgraphs before merging, and can be used to match the merging intensity based on the number of cores). For details, see "Function granularity configuration description" below.<br> ② **Semantic tag key**: controlled by semantic label (see the semantic label key configuration description below for details).<br> Value range:<br>{"DEFAULT": 1}: skip L1Reuse merging <br> {} (empty dictionary): automatic merging<br> {"DEFAULT": N, "func8_0": N2}: manual merging, where the default granularity is N<br> Default value: {} empty dictionary <br> Affected Pass scope: L1CopyInReuseMerge |
| cube_nbuffer_setting    | Input      | Meaning: graph merging parameter used to configure the merging count of AIC subgraphs with the same structure. <br> Description: This parameter applies to the merging of AIC subgraphs with the same structure. When configured together with cube_l1_reuse_setting, the merging related to cube_l1_reuse_setting is performed first, followed by this merging.<br><br>Type: dict[str, int]. Two key formats are supported:<br> ① **Function granularity key**: the string `"func{magic}_{order}"` or `"DEFAULT"`, which enables fine-grained configuration across different root functions. The graph merging information is directly displayed in the hashOrder-hint field of the swimlane diagram (including subGraphCount, which represents the number of subgraphs before merging, and can be used to match the merging intensity based on the number of cores). For details, see "Function granularity configuration description" below.<br> ② **Semantic tag key**: controlled by semantic label (see the semantic label key configuration description below for details).<br> Value range:<br>{"DEFAULT": 1}: skip AIC subgraph merging <br> {} (empty dictionary): explicitly enable automatic merging<br> {"DEFAULT": N, "func8_0": N2}: manual merging, where the default granularity is N<br>Default value: {"DEFAULT": 1}, that is, AIC subgraph merging is skipped by default <br> Affected Pass scope: L1CopyInReuseMerge |
| sg_set_scope            | Input      | Meaning: parameter for manually controlling subgraph partitioning.<br> Description: By assigning a scope to an Operation, adjacent Operations with the same scope_id (not -1) are forcibly merged into the same subgraph, thereby overriding the automatic partitioning result of the tiling algorithm. <br> Type: `Tuple[int, bool, bool]` or `int` <br> **Tuple format**: `(scope_id, allow_parallel_merge, allow_cross_scope_merge)`, where each field is described as follows: <br> - `scope_id` (int): scope identifier, with a value range of -1 to 2147483647. Adjacent Operations with the same scope_id are merged into the same subgraph; -1 means the Operation does not participate in scope merging, and the subgraph partitioning is determined by the tiling algorithm. <br> - `allow_parallel_merge` (bool): controls the merging mode of Operations under the same scope_id. Value: True/False.<br>&emsp;&emsp;False (default): only Operations with an upstream-downstream connection path are allowed to merge, that is, Operation A and Operation B can be merged into the same subgraph only when the output of Operation A serves as the input of Operation B.<br>&emsp;&emsp;True: Operations with the same scope_id on parallel branches (without data dependency) are also allowed to merge into the same subgraph. <br> - `allow_cross_scope_merge` (bool): controls whether a subgraph with a scope can merge with a subgraph without a scope (scope_id=-1) to expand the scoped subgraph. Value: True/False.<br>&emsp;&emsp;False (default): a subgraph with a scope remains independent and does not merge with other subgraphs.<br>&emsp;&emsp;True: a subgraph with a scope is allowed to merge with a subgraph whose scope_id=-1. Subgraphs with different scope_id values cannot merge with each other. <br> **int format**: when a single int is passed, it is equivalent to `(scope_id, False, False)`, that is, only scope_id is set, and parallel branch merging and cross-scope merging are not allowed. <br> Default value: (-1, False, False) <br> Affected Pass scope: GraphPartition <br> Configuration suggestions: 1) View-type Operations and their corresponding compute-type Operations should be configured with the same scope_id. 2) The Reshape Operation is special; in some scenarios it forms a subgraph on its own, and manually controlling the graph merging behavior may be ineffective.|
| sg_set_ooo_scope            | Input      | Meaning: controls OoO scheduling within MIX subgraphs.<br> Description: By assigning ooo_scope to an Operation, adjacent Operations with the same ooo_scope_id (not -1) are forcibly merged into the same ooo_task, so that adjacent Operations with the same ooo_scope_id (not -1) are as adjacent as possible on the pipeline generated by OoO scheduling. Parallel branch merging and cross-ooo_scope merging are not allowed, and merging of Operations with different loop iteration counts is not allowed. <br> Type: `int`, that is, setting ooo_scope_id <br> Default value: -1 <br> Value range: -1 or 1 to 10000 <br> `ooo_scope_id`: ooo_scope identifier. Adjacent Operations with the same ooo_scope_id are merged into the same ooo_task; -1 means the Operation does not participate in ooo_scope merging, and the ooo_task partitioning is determined by the tiling algorithm. <br> Affected Pass scope: OoOSchedule <br> Configuration requirements: this feature takes effect only on MIX subgraphs. When this feature is used together with loop unroll, the unroll setting must not exceed 10000. This parameter is planned to be deprecated in a later version.|
| ooo_sched_mode            | Input      | Meaning: controls OoO scheduling within MIX subgraphs.<br> Description: sets the pipeline scheduling mode of ooo_task within MIX subgraphs. <br> Type: `str` <br> Default value: "" <br> Value range: {"", "GAPMIN", "HLF"} <br> Affected Pass scope: OoOSchedule <br> Configuration description: when the value is "" (default), scheduling based on topological-order traversal and local search (GapMin scheduling + local-search) is used; when the value is "GAPMIN", only GapMin scheduling is executed and local-search is skipped; when the value is "HLF", the Highest Level First scheduling is used (tasks are sorted in descending order by the longest path to the sink, and then EFT insertion scheduling is performed).|
| sg_set_tunevf_mode           | Input      | Meaning: controls the behavior mode of the VF (Vector Fusion) tuning Pass.<br> Description: used to control the execution behavior of the two Passes TuneTileOpSeqForVF and TuneSyncForVF.<br> Type: `int` <br> Default value: 0 <br> Value range: {0, 1, 2} <br> - 0: balanced mode, which automatically adjusts the op order based on the op sequence output by the OoO Pass, automatically balancing the overall performance gain of pipeline pipelining and VF merge.<br> - 1: instruction pipeline priority mode, which does not change the op execution order arranged by OoO.<br> - 2: VF merge priority mode, which does not consider the gain evaluation of performance modeling and adjusts the op order as much as possible to ensure a larger range of VF merge. This mode affects only the NeedAdjustOpSeq judgment in TuneSyncForVF, and does not affect TuneTileOpSeqForVF.<br> Affected Pass scope: TuneTileOpSeqForVF, TuneSyncForVF |

## Return Value

None

## Constraints

- Setting time: It is not required to call before graph compilation starts; the setting can be performed at any time.
- Type safety: The type of the passed **value** must exactly match the type defined for the parameter; otherwise, undefined behavior or runtime errors may occur.
- Scope: Parameter settings are local and only affect the compilation process within the current jit or loop. If not set, the settings are inherited from the upper-level scope.
- `sg_set_scope` consistency constraint: All operations with the same `scope_id` must have the same `allow_parallel_merge` and `allow_cross_scope_merge` settings; otherwise, a compilation error is reported.
- When `scope_id` is -1, `allow_parallel_merge` and `allow_cross_scope_merge` must be **False**.
- Subgraphs with different `scope_id` values cannot be merged. `allow_cross_scope_merge` only controls the merging of subgraphs with a scope and subgraphs without a scope (`scope_id` = -1).
- **sg_set_scope** usage instructions:
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT products: scope configuration for pure Vector, pure Cube, and CV mixed scenarios is supported.
   <!-- end id7 -->
   <!-- npu="A3" id8 -->
   - Atlas A3 products: scope configuration for pure Vector or pure Cube is supported, but scope configuration for CV mixed scenarios is not supported.
   <!-- end id8 -->
   <!-- npu="910b" id9 -->
   - Atlas A2 products: scope configuration for pure Vector or pure Cube is supported, but scope configuration for CV mixed scenarios is not supported.
   <!-- end id9 -->
- **sg_set_ooo_scope** usage instructions:
   <!-- npu="950" id10 -->
   - Ascend 950PR&950DT products: Supported.
   <!-- end id10 -->
   <!-- npu="A3" id11 -->
   - Atlas A3 products: Not supported, because CV mix graph merging is not supported.
   <!-- end id11 -->
   <!-- npu="910b" id12 -->
   - Atlas A2 products: Not supported, because cv mix graph merging is not supported.
   <!-- end id12 -->
- **ooo_sched_mode** usage instructions:
   <!-- npu="950" id16 -->
   - Ascend 950PR&950DT products: Supported.
   <!-- end id16 -->
   <!-- npu="A3" id17 -->
   - Atlas A3 products: Not supported, because cv mix graph merging is not supported.
   <!-- end id17 -->
   <!-- npu="910b" id18 -->
   - Atlas A2 products: Not supported, because cv mix graph merging is not supported.
   <!-- end id18 -->
- **sg_set_tunevf_mode** usage instructions:
   <!-- npu="950" id19 -->
   - Ascend 950PR&950DT products: Supported.
   <!-- end id19 -->
   <!-- npu="A3" id20 -->
   - Atlas A3 products: Not supported, because cv mix graph merging is not supported.
   <!-- end id20 -->
   <!-- npu="910b" id21 -->
   - Atlas A2 products: Not supported, because cv mix graph merging is not supported.
   <!-- end id21 -->

## Examples

```python
   # Function granularity configuration (func{magic}_{order} format).
   pypto.set_pass_options(
       vec_nbuffer_setting={"DEFAULT": 4, "func8_0": 1, "func8_1": 1},
       cube_l1_reuse_setting={"DEFAULT": 4, "func8_0": 1, "func8_1": 1},
       cube_nbuffer_setting={"DEFAULT": 4, "func8_0": 1, "func8_1": 1})

   # Pure DEFAULT.
   pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 2})

   # Semantic tag key configuration (can coexist with function granularity keys).
   pypto.set_semantic_label("V1")
   sij_scale = pypto.mul(sij, softmax_scale)
   pypto.set_semantic_label("")
   ...
   pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 2, "V1": 1})
```

### dict Type Configuration Description (Function Granularity Key / Semantic Label Key)

### Function Granularity Key Configuration Description (func{magic}_{order})

#### Overview

Using a key in the `"func{magic}_{order}"` format, you can set the merging intensity for a **specific isomorphic subgraph group** (hashorder) of a **specific function**, enabling fine-grained configuration across different root functions. The configured **hashOrder** and **subGraphCount** information are directly displayed in the **hashOrder-hint** field of the swimlane diagram in the format `l1ReuseInfo hashOrder: func8_0, subGraphCount: 24`, allowing the merging intensity to be matched based on the number of subgraphs and the number of cores.

#### Key-Value Pair Meaning

Key: A string in the format `"func{magic}_{order}"` or `"DEFAULT"`.<br>

- `"func{magic}_{order}"`: Matches the isomorphic subgraph group whose **hashorder** is **order** in the function whose **functionMagic** is **magic**.<br>
- `"DEFAULT"`: Matches all isomorphic subgraph groups that are not explicitly specified.<br>

Value (N): Indicates the merge granularity, that is, every N subgraphs in an isomorphic subgraph group are fused into a new subgraph for execution. N=1 indicates no merge.

#### Format Constraints

- The `func` prefix must be lowercase.
- **magic** and **order** must be integers.
- **magic** and **order** are separated by an underscore `_`.
- Examples of valid keys: `"func0_0"`, `"func123_5"`, `"func8_1"`.
- Example of an invalid key: `"Func0_0"` (uppercase F), `"func_0"` (missing magic), `"func123"` (missing order).

#### Configuration Behavior

When processing subgraph merge for the current function, the Pass follows the logic of "exact match of `func{magic}_{order}` > `DEFAULT` configuration > automatic processing":<br>

- Exact match: If both **funcMagic** and **hashorder** are matched, merge is performed according to the corresponding **Value N**.<br>
- `DEFAULT` configuration: If no exact match is found but `DEFAULT` exists in the dictionary, merge is performed according to the **Value** corresponding to `DEFAULT`.<br>
- Automatic processing: If neither an exact match nor `DEFAULT` exists, the merge granularity is calculated automatically.<br>

#### Configuration Example

| Configuration | Description |
|------|------|
|`{"DEFAULT": 1}`|All isomorphic subgraph groups skip merging.|
|`{"DEFAULT": 4, "func8_0": 1, "func8_1": 1}`|By default, four isomorphic subgraphs are merged into one group. The subgraph groups with hashorder 0 and 1 in func8 skip merging (not merged).|
|`{"DEFAULT": 2, "func8_1": 4}`|By default, two isomorphic subgraphs are merged into one group. The subgraphs with hashorder 1 in func8 are merged four per group.|
|`{"func8_0": 2}`|In the func8 function, the subgraphs with hashorder 0 are merged two per group. For other isomorphic subgraph groups, the merging intensity is automatically calculated based on the number of hardware cores and then merged.|

### Semantic Label Key Configuration Description

#### Overview

In addition to the function granularity key, `vec_nbuffer_setting`, `cube_l1_reuse_setting`, and `cube_nbuffer_setting` also support string keys, that is, the semantic label names set through `pypto.set_semantic_label`. String keys allow users to precisely control the merge granularity of the subgraph (multiple subgraphs allowed) where a specific Operation resides, without the need to know its hashorder number.

#### Meaning of String Key-Value Pairs

Key (label): Semantic tag name, which must exactly match the `semantic_label` of at least one operation. If no operation is matched, no error is reported during compilation; only a WARN-level log is printed and the key is ignored.<br>
Value (N): Indicates the merge granularity.<br>

#### Priority Mechanism

The priority of a string key is **higher than** the default configuration of a function granularity key. The processing flow is as follows:<br>

1. First, determine the base merge granularity of each isomorphic subgraph group based on the function granularity key (`func{magic}_{order}` / `DEFAULT`).<br>
2. Then, the value of the string key **directly replaces** (rather than takes the max of) the merge granularity of the corresponding subgraph group.<br>
3. When multiple different string labels point to the same isomorphic subgraph group, the maximum value among these labels is used.<br>

#### Semantic Label Behavior of vec_nbuffer_setting / cube_nbuffer_setting

A string key overrides the merge granularity of the **entire isomorphic subgraph group** to which its corresponding Operation belongs.

#### Semantic Label Behavior of cube_l1_reuse_setting

Unlike `vec_nbuffer_setting` and `cube_nbuffer_setting`, the string key of `cube_l1_reuse_setting` **applies only to the subgraph containing the corresponding label Operation** and is not expanded to the entire isomorphic group. That is, within an isomorphic group, only some subgraphs may be covered by the string key, while other subgraphs retain the value of the function granularity key.

#### Semantic Label Configuration Example

| Configuration                                | Description                                                                 |
|-------------------------------------|----------------------------------------------------------------------|
|{"DEFAULT": 2, "V1": 1}|By default, the merging intensity of all isomorphic subgraph groups is 2. However, the merging intensity of the isomorphic subgraph group where the V1 tag resides is replaced with 1.|
|{"V1": 3}|The merging intensity of the isomorphic subgraph group where the V1 tag resides is 3. For other isomorphic subgraph groups, the merging intensity is automatically calculated.|
|{"DEFAULT": 2, "V1": 1, "V2": 3}|The default merging intensity is 2. The group where V1 resides is replaced with 1, and the group where V2 resides is replaced with 3. If a group contains both V1 and V2 OPs, max(1, 3) = 3 is used.|

#### Configuration Example

```python
   # Mixed function-granularity key and semantic label key configuration.
   pypto.set_semantic_label("V1")
   sij_scale = pypto.mul(sij, softmax_scale)
   pypto.set_semantic_label("") # Change the semantic label to precisely control that only this mul OP has the semantic label "V1".
   ...
   pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 2, "V1": 1})

   # Pure semantic label key configuration.
   pypto.set_pass_options(cube_l1_reuse_setting={"MM1": 4})
```

### sg_set_scope Configuration Description

#### Configuration Example

```python
# int format: equivalent to (10, False, False), setting only scope_id.
pypto.set_pass_options(sg_set_scope=10)

# tuple format: scope_id=1, allowing parallel branch merging but not cross-scope merging.
pypto.set_pass_options(sg_set_scope=(1, True, False))

# tuple format: scope_id=2, allowing merging with subgraphs that have no scope.
pypto.set_pass_options(sg_set_scope=(2, False, True))

# Restore the default (not participating in scope merging, automatically determined by the graph merging algorithm).
pypto.set_pass_options(sg_set_scope=-1)
```

#### Typical Scenarios

<!-- npu="A3" id22 -->
- Atlas A3 products: Scenario 2 is not supported.
<!-- end id22 -->
<!-- npu="910b" id23 -->
- Atlas A2 products: Scenario 2 is not supported.
<!-- end id23 -->

##### Scenario 1: Entire Computation Graph Not Partitioned

When the entire computation graph needs to remain unpartitioned, data tiling produces multiple parallel branches that have no direct data dependency between them, and these branches are split into independent subgraphs by the tiling algorithm by default. It is recommended to set `sg_set_scope=(scope_id, True, False)` so that, through `allow_parallel_merge=True`, the parallel branch Operations with the same **scope_id** are fused into the same subgraph.

##### Scenario 2: CV mixed scenario, constructing a Mix subgraph to reduce GM transfer

When both the preceding and following operations of a Cube operation are Vec operations, the goal is to construct a Mix subgraph containing both Cube and Vec, avoiding repeated data movement of intermediate results on GM. Depending on whether the scope boundary is explicit, the following two cases are distinguished:

**Scenario 2.1: Explicit Scope Boundary**

When the boundary between a Cube operation and its adjacent Vec operations can be explicitly defined, use `(scope_id, False, False)` to mark the boundary, forcing the Cube and the adjacent Vec into the same subgraph to form a Mix subgraph.

```python
# Explicitly mark the Cube and the adjacent Vec as the same scope to form a Mix subgraph.
pypto.set_pass_options(sg_set_scope=(1, False, False))
# ... Cube operation ...
# ... Adjacent Vec operation ...
pypto.set_pass_options(sg_set_scope=-1)
```

**Scenario 2.2: Unclear Scope Boundary; Only the CV Fusion Boundary Marked**

When the boundary cannot be clearly defined, but the Vec subgraphs before and after the Cube need to be fused with the Cube subgraph to form a Mix subgraph, use `(scope_id, False, True)` to mark the Cube and its adjacent Vec as merge anchors. With `allow_cross_scope_merge=True`, the Vec in this scope subgraph can be fused with the preceding and following Vec subgraphs that have no scope (scope_id=-1), forming a larger subgraph to reduce GM data movement, and forming a Mix subgraph together with the Cube subgraph.

```python
# Preceding Vec operation (scope_id=-1, automatically determined by the splitting algorithm).
vec_out = some_vec_op(x)

# Mark the CV merging anchor to allow merging with adjacent subgraphs without a scope.
pypto.set_pass_options(sg_set_scope=(1, False, True))
# ... Cube operation ...
matmul_result = pypto.matmul(vec_out, w)
# ... Adjacent Vec operation ...
pypto.set_pass_options(sg_set_scope=-1)

# Subsequent Vec operation (scope_id=-1), which can be merged with the scope subgraph above into a larger subgraph.
result = other_vec_op(add_result)
```

### **sg_set_ooo_scope** configuration description

#### Configuration Example

```python
# Set ooo_scope_id.
pypto.set_pass_options(sg_set_ooo_scope=10)

# Restore the default (do not participate in ooo_scope merging; the splitting algorithm decides automatically).
pypto.set_pass_options(sg_set_ooo_scope=-1)
```

#### Typical Scenario: Controlling the Execution Order of Operations

As shown in the following example, by wrapping **mul** and **exp** in the same **ooo_scope**, you can make **exp** execute before **add** in the OoO scheduling result.

```python
# Because ooo_scope must be enabled within the MIX subgraph, use sg_set_scope to construct the MIX subgraph.
pypto.set_pass_options(sg_set_scope=1)
# Cube operation.
matmul_result = pypto.matmul(a, b)
pypto.set_pass_options(sg_set_ooo_scope=1)
# Vec operation (ooo_scope_id=1).
c = pypto.mul(matmul_result, scale)
pypto.set_pass_options(sg_set_ooo_scope=-1)
d = pypto.add(c, bias)
pypto.set_pass_options(sg_set_ooo_scope=1)
# Vec operation (ooo_scope_id=1).
e = pypto.exp(c)
pypto.set_pass_options(sg_set_ooo_scope=-1)
pypto.set_pass_options(sg_set_scope=-1)
```

### sg_set_tunevf_mode Configuration Description

#### Configuration Example

```python
# Instruction pipelining priority mode, which does not change the op execution order scheduled by OoO.
pypto.set_pass_options(sg_set_tunevf_mode=1)

# VF merge priority mode, which does not consider the benefit evaluation of performance modeling and adjusts the op order as much as possible to ensure a wider range of VF merge.
pypto.set_pass_options(sg_set_tunevf_mode=2)

# Restore the default behavior, that is, the balanced mode.
pypto.set_pass_options(sg_set_tunevf_mode=0)
```
