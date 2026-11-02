# PYPTO_THIRD_PARTY_PATH

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-27T01:12:52.281Z pushedAt=2026-10-27T02:34:04.698Z nl=2 -->

## Description

Specifies the path of the third-party open-source software source package required for PyPTO compilation. When the compilation environment cannot access the `cann-src-third-party` repository for automatic download, you need to manually prepare the source package and specify its path using this variable.

- Type: string (absolute path)

## Configuration Example

```bash
# Set after manually preparing the third-party source packages.
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
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Supported
<!-- end id3 -->
