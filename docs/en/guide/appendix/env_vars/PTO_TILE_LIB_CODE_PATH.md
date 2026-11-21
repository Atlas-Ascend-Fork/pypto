# PTO_TILE_LIB_CODE_PATH

<!-- md-trans-meta sourceCommit=50972549b8df8d831de411545a92c8acac5eac27 translatedAt=2026-11-15T01:28:31.883Z pushedAt=2026-11-18T06:03:01.278Z nl=2 -->

## Description

Specifies the pto-isa source code path for compiling and running PyPTO operators. CANN toolkit includes pto-isa by default after being installed, so manual setting is not required. When the built-in version does not meet requirements, you can use this environment variable to point to a standalone pto-isa source code directory.

- Type: string (absolute path)

## Configuration Example

```bash
# Use the CANN built-in pto-isa (recommended)
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/$(uname -m)-linux

# Use a standalone pto-isa source
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
