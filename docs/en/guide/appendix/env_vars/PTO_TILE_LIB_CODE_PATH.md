# PTO_TILE_LIB_CODE_PATH

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-27T01:11:16.570Z pushedAt=2026-10-27T02:32:45.204Z nl=2 -->

## Description

Specifies the pto-isa source code path for compiling and running PyPTO operators. CANN toolkit includes pto-isa by default after being installed, so manual setting is not required. When the built-in version does not meet requirements, you can use this environment variable to point to a standalone pto-isa source code directory.

- Type: string (absolute path)

## Configuration Example

```bash
# Use the built-in pto-isa of CANN (recommended)
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/$(uname -m)-linux

# Use standalone pto-isa source code
git clone https://gitcode.com/cann/pto-isa.git
export PTO_TILE_LIB_CODE_PATH="$PWD/pto-isa"
```

## Constraints

- The path must contain the `include/pto/` directory.
- If the built-in PyPTO in the CANN package is used, pto-isa is installed with the CANN package and does not need to be set separately.

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
