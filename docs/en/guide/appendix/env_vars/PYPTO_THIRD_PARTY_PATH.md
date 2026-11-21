# PYPTO_THIRD_PARTY_PATH

<!-- md-trans-meta sourceCommit=50972549b8df8d831de411545a92c8acac5eac27 translatedAt=2026-11-15T01:29:58.667Z pushedAt=2026-11-18T06:03:01.285Z nl=2 -->

## Description

Specifies the path of the third-party open-source software source package required for PyPTO compilation. When the compilation environment cannot access the `cann-src-third-party` repository for automatic download, you need to manually prepare the source package and specify its path using this variable.

- Type: string (absolute path)

## Configuration Example

```bash
# Set after manually preparing the third-party source package.
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>
```

After the setting is complete, perform compilation and installation.

## Constraints

- The path must contain the source packages of `json-3.11.3` and `libboundscheck-v1.1.16`.
- Can take effect only during compilation and installation based on source code. It is not required for PyPI installation.

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
- Kirin 9030 processor products: Supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processor products: Supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processor products: Supported
<!-- end id6 -->
