# 目录文件维护要求

适用范围：涉及文档文件新增、删除、移动、重命名，或修改 `index.md` / `index_hiascend.md` / toctree / 图片资源的 PR。

规则编号范围：`STR-01~04`。纯内容修改（不涉及文件增删移）可跳过本清单大部分条目。

本文件分两部分：**规则**（文件增删移时必须同步执行的要求）与**检查方法**（排查是否漏登记、漏同步的手段）。

> 文件编码格式、文件命名（FN-01~06）与图片存放/命名（IMG-01~02）属于开源文档写作规范，在检查 `.md` 文件时执行，不在本文件重复。检视时须注意两处与本仓惯例的差异，以仓内惯例为准：① IMG-02 要求图片存放于文档**同级** `figures/`，本仓惯例为分类级（上级）`figures/`（如 `tensor_api/figures/`、`guide/figures/`）；② 社区版规范要求文件名多单词用下划线连接，本仓 API 文档惯用中划线（如 `pypto-abs.md`，md-review-skill 版 FN-02 两者皆可）。

## 目录结构

`docs/` 关键结构：

```
docs/
├── zh/
│   ├── install/          # 环境部署
│   ├── invocation/       # 样例运行
│   ├── guide/            # 算子开发指南（含 appendix 术语表/FAQ/环境变量）
│   └── api/
│       ├── tensor_api/   # Tensor API（operation/element/datatype/controlflow/...）
│       └── pro_api/      # Pro API（SIMD-API/SIMT-API/Utils-API）
├── CONTRIBUTION_DOC.md / CONTRIBUTION_DOC_en.md
└── README.md / README_en.md
```

## 规则

### 目录索引登记（STR-01、STR-02、STR-04）

`.md` 文件有**两套索引**，服务两个发布渠道：各级目录 `index.md` 的 toctree（构建 PyPTO 文档中心，STR-01）与 `api/index_hiascend.md`、`guide/index_hiascend.md` 聚合清单（发布到昇腾社区，STR-02）。文件发生新增/删除/移动/重命名时两套均须同步，任一遗漏会导致文档在对应渠道缺失。

| 编号 | 规则 | 严重级别 |
|------|------|---------|
| STR-01 | toctree 须与文件实际状态保持一致：新增文档须登记到所在目录 `index.md` 的 `{toctree}`（条目为去掉 `.md` 后缀的文件名）；新增子目录时须同时新建该目录的 `index.md`，并将其登记到上级 `index.md` 的 toctree；条目须与实际文件名完全一致（拼写错误会导致 Sphinx 构建告警且链接失效）；删除/移动/重命名文件时，须同步更新引用它的所有 toctree 及全仓交叉链接。**漏登记或漏同步会导致文档无法出现在文档中心或产生死链** | CRITICAL |
| STR-02 | `docs/zh/api/`、`docs/zh/guide/` 下任何层级的 `.md` 文件新增/删除/移动/重命名时，须同步更新 `api/index_hiascend.md`、`guide/index_hiascend.md` 聚合清单。`index_hiascend.md` 为 Markdown 链接清单，条目为带相对路径的 `.md` 文件 | CRITICAL |
| STR-04 | 各级目录 `index.md`（父节点索引，如 `docs/zh/api/tensor_api/operation/index.md`）只允许包含：标题、`{toctree}` 围栏块（含 `:maxdepth:` 等指令参数与条目）、空行，**不得添加"本章介绍xxx"等描述性正文或其他代码块**——昇腾社区发布渠道不渲染 toctree 索引页的正文内容，写了也无法展示；引导性说明应写入具体子文档或 guide 正文页。文档根 `index.md`（docs/zh/index.md，文档中心首页）豁免 | HIGH |

### 成对文件同步（STR-03）

| 编号 | 规则 | 严重级别 |
|------|------|---------|
| STR-03 | `docs/` 根下 README 与 CONTRIBUTION_DOC 为中英文成对文件（`xxx.md` + `xxx_en.md`），修改其一须评估另一份是否需同步 | MEDIUM |

## 检查方法

### toctree 登记检查（STR-01）

```bash
# 找出目录内未登记进 toctree 的文档（以 operation 为例）
cd docs/zh/api/tensor_api/operation
ls *.md | sed 's/\.md$//' | grep -vx index | sort > /tmp/files.txt
grep -oE '^[a-z0-9_-]+' index.md | sort -u > /tmp/toctree.txt
comm -23 <(sort /tmp/files.txt) /tmp/toctree.txt   # 输出即漏登记文件

# 反向：toctree 中不存在对应文件的条目
comm -13 <(sort /tmp/files.txt) /tmp/toctree.txt
```

注意：条目匹配的正则须包含连字符（`[a-z0-9_-]`），否则 `pypto-experimental-xxx` 这类多段连字符文件名会被截断产生误报。

### index_hiascend.md 覆盖检查（STR-02，api / guide 通用）

```bash
cd docs/zh/api   # 或 docs/zh/guide
find . -name '*.md' ! -path './index.md' ! -name 'index_hiascend.md' | sed 's|^\./||;s/\.md$//' | sort > /tmp/files.txt
grep -oE '\]\([^)]+\.md\)' index_hiascend.md | sed 's/^\](//;s/)$//;s/\.md$//' | sort -u > /tmp/hiascend.txt
comm -23 /tmp/files.txt /tmp/hiascend.txt   # 输出即漏登记文件
comm -13 /tmp/files.txt /tmp/hiascend.txt   # 反向：清单中不存在对应文件的条目
```

注意：清单可能含跨目录引用（如 guide 清单中的 `../install/prepare_environment_hiascend.md`），反向比对出现此类条目时，确认目标文件存在即可。

### index.md 描述性内容检查（STR-04）

```bash
# 提取 index.md 中 toctree 围栏外、非标题、非空行的内容，并识别非 toctree 代码块（输出即违规）
awk 'FNR==1{infence=0} /^```/{ if (!infence && $0 !~ /toctree/) print FNR": 非toctree代码块"; infence=!infence; next }
     infence{next} /^#/{next} /^[[:space:]]*$/{next} {print FNR": "$0}' <目标index.md>

# 批量扫描 docs/zh 下所有目录 index.md（文档根 index.md 首页豁免）；FNR==1 重置围栏状态，避免跨文件残留
find docs/zh -name index.md ! -path 'docs/zh/index.md' \
  -exec awk 'FNR==1{infence=0} /^```/{ if (!infence && $0 !~ /toctree/) print FILENAME": "FNR": 非toctree代码块"; infence=!infence; next }
             infence{next} /^#/{next} /^[[:space:]]*$/{next} {print FILENAME": "FNR": "$0}' {} +
```

输出非空即存在描述性内容；检视模式仅报 PR 变更引入的问题，存量命中（如 `guide/programming_guide/pro/debug/index.md` 的存量正文）不重复报告。
