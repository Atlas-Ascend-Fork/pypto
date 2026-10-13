# 术语及表达约定

适用范围：`docs/` 下所有文档。

规则编号范围：`TRM-01~12`。

本文件分两部分：**规则**——术语约定（TRM-01~06：术语来源与写法、常见变体对照、环境变量来源）与表达约定（TRM-07~12：全文通用的标识符引用、锚点与链接规范、pro 接口命名、数据类型写法、功能开关用词）；**检查方法**——文末统一给出对应排查手段。

## 规则

### 术语来源与写法（TRM-01~05）

术语事实来源分两级：

| 级别 | 覆盖范围 | 事实来源 |
|------|----------|----------|
| 第一级：PyPTO 术语表 | PyPTO 专有术语：PyPTO、PTO、Operation、计算图（Tensor Graph / Tile Graph / Block Graph / Execute Graph）、TileShape、Pass、合图、同构子图、stitch 等 | `docs/zh/guide/appendix/glossary.md`（仓库根相对路径，检查时读取） |
| 第二级：昇腾术语表 | 昇腾/CANN 通用术语：AI Core、AI CPU、Cube/Vector、UB、GM、L1 Buffer、NZ/ND、Host/Device 等 | [昇腾术语表](https://www.hiascend.com/document/detail/zh/Glossary/gls/gls_0001.html)（CANN 官方，检查时按需查阅） |

| 编号 | 规则 | 严重级别 |
|------|------|---------|
| TRM-01 | 文档中术语的写法（大小写、分合写、中英文选择）必须与对应级别术语表完全一致：PyPTO 专有术语以第一级为准，昇腾通用术语以第二级为准 | HIGH |
| TRM-02 | 检查新文档时，若引入了两级术语表均未覆盖且面向用户的新领域术语，PR 须同步扩展 `glossary.md`（新增一行术语+含义）；属昇腾通用术语的，确认官方术语表已有定义即可 | MEDIUM |
| TRM-03 | 术语/缩写首次出现时必须有解释或全称说明，不得使用未定义的缩写 | MEDIUM |
| TRM-04 | 昇腾/CANN 通用术语的写法以[昇腾术语表](https://www.hiascend.com/document/detail/zh/Glossary/gls/gls_0001.html)为准；两级来源均未覆盖的术语，与仓内既有文档主流写法保持一致，同一文档内不得混用多种写法 | MEDIUM |
| TRM-05 | 首次出现的英文缩写须给出全称，如"PTO（Parallel Tensor/Tile Operation）"；后续可直接使用缩写 | LOW |

### 常见变体对照

| 标准写法 | 常见错误变体 | 来源 |
|----------------|--------------|------|
| PyPTO | pypto（叙述文本中）、PyPto、PYPTO、Py-PTO | 第一级 |
| TileShape | tile shape、Tile Shape、tileshape、tile_shape（叙述文本中） | 第一级 |
| Tensor Graph | tensor graph、TensorGraph | 第一级 |
| Operation | operation（句首）、算子（PyPTO 语境用"Operation"） | 第一级 |
| Pass | pass（句首除外）、编译pass | 第一级 |
| 泳道图 | 泳道图（SWIMLANE） | 第一级 |
| AI Core | AICore、AI core、ai core | 第二级 |
| AI CPU | AICPU、ai cpu | 第二级 |
| DataType / 数据类型 | datatype、dtype（概念表达时） | 仓内约定 |
| Format / 数据排布格式 | format（概念表达时） | 仓内约定 |
| 形状 / Shape | shape（概念表达时） | 仓内约定 |
| Tensor | tensor（概念表达时） | 仓内约定 |
| Tile | tile（概念表达时） | 仓内约定 |
| Global Memory / GM | global memory、GM Buffer、gm | 第二级＋仓内禁变体 |
| Unified Buffer / UB | unified buffer、UB Buffer、UnifiedBuffer | 第二级＋仓内禁变体 |
| L1 Buffer | L1 buffer、L1Buffer、L1BUFFER、**L1（单独使用，须写完整 L1 Buffer）** | 第二级＋仓内禁变体 |
| L0A Buffer | L0A buffer、L0ABuffer、**L0A（单独使用）** | 第二级＋仓内禁变体 |
| L0B Buffer | L0B buffer、L0BBuffer、**L0B（单独使用）** | 第二级＋仓内禁变体 |
| L0C Buffer | L0C buffer、L0CBuffer、**L0C（单独使用）** | 第二级＋仓内禁变体 |
| Fixpipe Buffer | Fixpipe buffer、FixpipeBuffer | 仓内约定 |
| BiasTable Buffer | BiasTable buffer、BiasTableBuffer | 仓内约定 |
| L0A_MX Buffer | L0A_MX buffer、L0A MX Buffer、L0AMX Buffer、**L0A_MX（单独使用）** | 仓内约定 |
| L0B_MX Buffer | L0B_MX buffer、L0B MX Buffer、L0BAMX Buffer、**L0B_MX（单独使用）** | 仓内约定 |
| Byte / 字节 | bytes、Bytes、BYTES、byte（概念表达时）、**B 单独用作字节单位（如 512B、512 B）**（KB/MB/GB 等复合单位不受此限） | 仓内约定 |

（标“第二级”的行，变体仅为示例，以昇腾术语表当前内容为准；标“第二级＋仓内禁变体”的行，仅允许标准写法及上表缩写，禁止任何变体；其余硬件存储单元行同样禁止任何变体）

**「概念表达」的界定**：指叙述性文本中将上述词语作为概念使用（如"Shape仅支持1-4维"、"Tensor的数据类型"）。以下情况**不受此约束**，保持实际写法：
- 参数名、属性名、枚举成员等代码标识符（如参数表第一列的参数名 shape、tile，属性 Tensor.dtype，原型中的 dtype 参数）
- 文件路径、目录名、链接目标（如 `tensor_api/`、`pypto-Tensor-dtype.md`）
- 代码块与命令行内容
- 特性名中的硬件词（如"fixpipe量化"中的 fixpipe 指量化特性而非 Fixpipe Buffer 存储单元）

### 环境变量来源（TRM-06）

环境变量事实来源分两个范围：

| 范围 | 事实来源 |
|------|----------|
| 范围一：PyPTO 领域环境变量 | `docs/zh/guide/appendix/env_vars/`（仓库根相对路径，检查时读取目录清单） |
| 范围二：CANN 环境变量 | [CANN 环境变量参考](https://gitcode.com/cann/docs/blob/master/docs/zh/env-vars/README.md)（检查时按需查阅） |

| 编号 | 规则 | 严重级别 |
|------|------|---------|
| TRM-06 | 文档中使用的环境变量须来源于上述两个范围之一，**禁止使用范围外的环境变量**——引用不存在的环境变量会误导用户配置无效参数；确需引入新环境变量时，须在同一 PR 中于 `docs/zh/guide/appendix/env_vars/` 下新增对应说明文档并登记 index.md 的 toctree（见 STR-01） | HIGH |

### 表达约定（TRM-07~12，全文通用）

以下规则约束文档全文的叙述正文（功能说明、参数说明、约束说明等所有章节），不限于特定章节。

| 编号 | 规则 | 严重级别 |
|------|------|---------|
| TRM-07 | 叙述正文中提及接口名、参数名、类型名、枚举值、取值/默认值时**裸写**，不加反引号（如"input和other都为Tensor时"、"应通过set_vec_tile_shapes设置"、"默认为False"、"默认为HIGH_PRECISION"）——存量文档主体风格为裸写，agent 生成文档高频误用反引号或引号包裹，造成风格不一致；反引号仅用于命令、文件路径及完整调用表达式（如 `pypto.tensor([2], pypto.DT_INT16)`）；环境变量文档的取值反引号为其既定风格，不在此列；引用日志或报错原文的情况除外；同一文档内不得混用两种风格 | MEDIUM |
| TRM-08 | 文内引用本文档其他章节使用锚点链接（如 `[约束说明](#约束说明)`），避免"参见下文"等模糊表述 | LOW |
| TRM-09 | 正文描述中引用 pro 接口时，须使用完整模块路径名（如 pypto_pro.language.trap()、pypto_pro.language.CastLayout.ZERO），**禁止使用导入别名缩写**（如 pl.trap()、pl.Input）——别名依赖示例中的 import 语句，脱离上下文后用户无法定位接口；别名仅可出现在调用示例代码中，且示例须包含对应的 `import pypto_pro.language as pl` 语句 | HIGH |
| TRM-10 | 数据类型写法须与对应接口体系的 DataType 枚举值保持一致，不得混用两个体系的写法：Tensor 接口参考 `docs/zh/api/tensor_api/datatype/DataType.md`（如 DT_INT8、DT_FP16、DT_BF16）；pro 接口参考 `docs/zh/api/pro_api/SIMD-API/basic_data_structures/DataType.md`（如 DT_FP4、DT_FP8E4M3FN） | HIGH |
| TRM-11 | 链接引用规范：① 参见**本仓库**文档内容时，使用相对路径链接（如 `[术语表](./guide/appendix/glossary.md)`、`[约束说明](#约束说明)`），**禁止使用指向本仓库的绝对链接**（如 `https://gitcode.com/cann/pypto/blob/master/docs/...`）——绝对链接会跳出文档中心且路径易失效（指向 issues 等平台功能的链接除外）；② 参见 **CANN 其他产品文档**时，使用 [comterms.md](https://gitcode.com/cann/docs/blob/master/specification/comterms.md)（CANN 组织文档路径汇总）中登记的官方入口链接 | HIGH |
| TRM-12 | 描述功能、特性、选项的启动/开启时，**禁止使用“使能”**，应使用“启用”或“开启”（如“启用tensor dump功能”），以免引起歧义 | HIGH |

## 检查方法

### 术语检查（TRM-01~05）

检查流程：

1. 读取 `docs/zh/guide/appendix/glossary.md`，提取 PyPTO 专有术语及标准写法
2. 对目标文档提取疑似术语变体，按「常见变体对照」表比对
3. 注意区分**代码语境**与**叙述语境**：
   - 命令、文件路径及完整调用表达式用行内代码格式：`` `pip install -r requirements.txt` ``、`` `pypto_pro.language.trap()` ``
   - 叙述性文本指框架/产品时用 PyPTO（非代码格式）
   - `` `pypto` `` 出现在叙述句中（如"pypto采用PTO编程范式"）应改为"PyPTO"
   - 叙述正文中标识符与取值的引用风格按 TRM-07 执行
4. 涉及昇腾/CANN 通用术语时，查证[昇腾术语表](https://www.hiascend.com/document/detail/zh/Glossary/gls/gls_0001.html)的标准写法后比对；无法在线查证时，与仓内既有文档的主流写法保持一致，同一文档内不得混用多种写法
5. 按「环境变量检查」核对目标文档中使用的环境变量（TRM-06）

```bash
# 大小写变体排查（示例）
grep -rnoE 'PyPto|PYPTO|Py-PTO' docs/zh --include='*.md'
# 分合写变体排查（示例）
grep -rnoiE 'tile[ _-]?shape' docs/zh --include='*.md' | grep -v 'TileShape'
# CANN 通用术语常见变体（示例，标准写法以昇腾术语表为准）
grep -rnoE 'AICore|AI core|ai core|AICPU|ai cpu' docs/zh --include='*.md'
# 叙述句中裸写 pypto（无反引号包裹）
grep -rnE '[^`_]pypto[^`_]' docs/zh --include='*.md' | grep -vE '`pypto|\.pypto|pypto\.'
# 概念表达小写变体（命中后按「概念表达的界定」人工排除参数名/属性名/路径等标识符）
grep -noE '\b(datatype|dtype|shape|format|tensor|tile)\b' <目标文件>
# 硬件存储单元变体（-i 全量命中后人工核对大小写与拼接，非标准写法即违规）
grep -noiE 'global memory|unified buffer|GM buffer|UB buffer|L1 ?buffer|L0A ?buffer|L0B ?buffer|L0C ?buffer|fixpipe buffer|biastable buffer|L0A_MX ?buffer|L0B_MX ?buffer' <目标文件>
# 单独使用 L1/L0A/L0B/L0C/L0A_MX/L0B_MX（不带 Buffer 后缀；命中后人工排除代码标识符，如 AiCorePrintL1Tensor 为函数名）
grep -noP '(?<![A-Za-z0-9_])(L1|L0A|L0B|L0C|L0A_MX|L0B_MX)(?!\s?Buffer)(?![A-Za-z0-9_])' <目标文件>
# 字节单位变体（标准写法为 Byte 或 字节；\b 已排除 nextDepthTotalBytes 等标识符，命中后人工排除代码块与日志原文引用）
grep -noE '\b[Bb]ytes\b|BYTES|\bbyte\b' <目标文件>
# 数字后单独用 B 作字节单位（如 512B、512 B；KB/MB/GB 等复合单位、F7100B 等错误码不会命中）
grep -noP '(?<![A-Za-z0-9_])[0-9]+ ?B\b' <目标文件>
```

输出结果需人工判断是否为代码语境，代码语境（import 语句、调用示例、路径）不算违规。

### 环境变量检查（TRM-06）

```bash
# PyPTO 领域环境变量清单（范围一）
ls docs/zh/guide/appendix/env_vars/*.md | xargs -n1 basename | sed 's/\.md$//'
# 提取目标文档中的疑似环境变量（全大写下划线词；人工排除 DT_ 数据类型、INT32_MAX 等常量；CANN 范围二按需查证 README）
grep -noE '\b[A-Z][A-Z0-9]*(_[A-Z0-9]+)+\b' <目标文件> | grep -v 'DT_'
```

### 表达约定检查（TRM-07~12）

```bash
# 叙述句中反引号包裹的标识符（TRM-07，命中后人工确认：命令/路径/完整调用表达式为合法用法）
grep -noE '`[a-zA-Z_][a-zA-Z0-9_.]*`' <目标文件>
# 引号包裹标识符（TRM-07，需人工确认非日志/报错原文引用）
grep -noE '“[a-zA-Z_][a-zA-Z0-9_.]*”' <目标文件>
# pro 接口别名引用（TRM-09，命中后人工排除代码块内及 import 别名声明行）
grep -n 'pl\.[a-zA-Z_]' <目标文件>
# 数据类型写法提取（TRM-10，与对应体系 DataType.md 的枚举值清单比对）
grep -oE 'DT_[A-Z0-9_]+' <目标文件> | sort -u
# 指向本仓库文档的绝对链接（TRM-11，issues 等平台链接不在此列）
grep -noE 'https://gitcode\.com/cann/pypto/(blob|tree)/[^) ]*' <目标文件>
# 功能开关语境误用“使能”（TRM-12，全量命中后人工确认是否为硬件信号/寄存器位/lane 掩码语境）
grep -n '使能' <目标文件>
```
