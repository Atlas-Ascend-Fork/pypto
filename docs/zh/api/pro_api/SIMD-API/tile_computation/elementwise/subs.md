# pypto_pro.language.subs

## 产品支持情况

- Ascend 950PR&950DT 系列产品：支持。
- Atlas A2/A3 系列产品：不支持。

## 功能说明

逐元素标量减法，结果为 `out[i, j] = lhs[i, j] - rhs`。这是 [pypto_pro.language.sub](sub.md) 的 Tile-Scalar 调用形式；允许 out 与 lhs 指向同一 Tile。

## 函数原型

```python
pypto_pro.language.subs(out: Tile, lhs: Tile, rhs: Scalar) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| out | 输出 | 结果 Tile，位于 Vec（UB）。 |
| lhs | 输入 | 输入 Tile，位于 Vec（UB），shape 和 dtype 与 out 相同。 |
| rhs | 输入 | 标量操作数；不是 Tile，不参与 Tile-Tile 广播。 |

## 调用示例

```python
with pl.section_vector():
    pl.subs(out_tile, input_tile, 0.5)
```

接口按原生 block.subs 解析，auto_mutex 根据 out/lhs 的写/读角色工作。具体 dtype 支持遵循对应 Tile-Scalar 操作与当前工具链约束；接口声明本身不扩大硬件支持范围。
