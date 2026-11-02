# 状态脚本协议

主 agent 负责检视、修复及证据核实，是状态的唯一写入者；脚本记录状态并控制轮次和转移。
输入和状态均为 JSON；证据字段存可追溯的说明或产物路径。先核实用户授权、源码版本和检视结果，再记录状态。
命令按项目指令的 shell 执行；路径均替换为实际绝对路径，JSON 用文件传递，避免 shell 拼接正文。

```bash
python3 <skill>/scripts/review_state.py --state <run>/state.json init --input <run>/init.json
python3 <skill>/scripts/review_state.py --state <run>/state.json begin --input <run>/begin.json
python3 <skill>/scripts/review_state.py --state <run>/state.json finish --input <run>/result.json
python3 <skill>/scripts/review_state.py --state <run>/state.json show
```

每次变更成功输出简短状态；`show` 返回完整记录。退出码 0 表示状态记录成功，**不代表审核通过**，
必须读取 `status`。输入错误退出 1，状态不变；不能靠重新初始化绕过错误或轮次上限。

## 输入

`init`：`base` 为原始比较基点/本地基线产物，`plan` 为已确认方案及来源，
`review_kind` 为 `all`（默认）或 `rules`。需求未提供方案时在 `plan` 如实写明，不自行编造。
已有状态文件拒绝覆盖。人工批准的新尝试在新的任务目录初始化，旧记录保持不变，输入产物中保留关联和授权依据。

`begin`：

```json
{
  "snapshot": "当前源码 SHA 或本地快照内容标识",
  "input": "/absolute/run/round-1/input.md",
  "claimed_fixed": [],
  "repair_evidence": null
}
```

本地标识须绑定 diff、相关新增文件及实际读取内容，不能仅改标签制造新版本。
`claimed_fixed` 列出本轮前主 agent 声称已修复的旧问题 ID，用于发现严重问题修复失败。
`repair_evidence` 可记录修改和验证产物。下一轮必须有实际源码变化；同版本补齐可取得的证据后再 `finish`，
不能在没有变化时反复发起检视。`begin` 增加一次累计次数，整组角色共用同一轮。
上轮报告/版本无法可靠取得或原覆盖失效时，额外提供 `full_review_reason` 说明原因，强制本轮完整审核；
此操作不重置计数，不能强制将完整审核降为增量。

`finish`：

```json
{
  "snapshot": "与 begin 相同的内容标识",
  "roles": {"functional": "complete", "rules": "complete", "verifier": "complete"},
  "coverage_complete": true,
  "coverage_evidence": "/absolute/run/round-1/coverage.md",
  "validation": {
    "status": "passed",
    "snapshot": "与 begin 相同的内容标识",
    "evidence": "实际验证命令、结果及日志路径；无需运行时注明依据"
  },
  "report": "/absolute/run/round-1/report.md",
  "findings": [
    {
      "id": "F1",
      "root": "稳定的根因标识，不能用会移动的行号代替",
      "severity": "Major",
      "status": "open",
      "evidence": "verifier 的源码证据、实际影响及排除的反例",
      "introduced_by_fix": false
    }
  ]
}
```

- 所需角色必须各自记录 `complete / incomplete / failed`；仅规范模式不要求 `functional`。
  角色不完整、覆盖不足或未决候选不能通过。`finish` 应在可独立完成的核实结束后调用，仍有缺口则返回人工交接。
- 验证为 `passed / failed / blocked`；已定位、可修复的代码验证失败用 `failed`，
  无法在范围内解决的环境或证据阻塞用 `blocked`。验证必须对应受审版本，不可使用旧构建的成功记录。
- `findings` 是完整台账，包括旧问题。状态为 `open / resolved / rejected / unresolved`：
  已修改但尚未被独立核实的问题仍为 `open`；关闭、否定或证据不足都要说明依据。
  稳定 ID 与根因不能被替换来清空历史，所有旧 ID 必须出现；同根因跨角色发现先合并再记录。
  重复 ID/根因会拒绝写入，不能按两条统计。允许附加分类、位置及来源候选等字段，脚本原样保留。
- `introduced_by_fix` 仅在源码比较证明确由修复引入时置 true，不把此次首次发现的旧问题算成修复回归。

## 反思、批准与重置

同样以 `--input <JSON文件>` 调用：

| 事件 | 输入字段 | 前置状态与效果 |
|---|---|---|
| `assess` | `outcome: sound/flawed/unknown`、`evidence`；sound 给出 `next_action`，首次 flawed 给出 `proposal` | `assess` 或 `fix`；每轮只允许一次；sound 清零连续方案问题次数，flawed 累加，第二次立即结束 |
| `approve` | `evidence`：真实用户确认原文或已有对应授权的位置 | `awaiting_approval`；记录授权，进入 `awaiting_revision`，尚不清零阶段计数 |
| `revise` | `plan`：新方案及来源、`snapshot`：落实后的新源码版本、`evidence`：修订实施证据 | `awaiting_revision`；增加方案版本并清零阶段次数，保留累计次数及连续方案问题次数 |
| `stop` | `kind: user/human`、`reason` | 未终结状态；停止并保留记录 |

`assess` 没有可执行修复方向或结论为 unknown 时返回 `needs_human`。方案批准必须来自用户授权，
不能把主 agent 认为方案合理当成批准。重复调用同一轮反思不能人为打断连续计数。

终结状态为 `passed / needs_human / stopped`，拒绝继续变更；等待方案确认则保存记录并等待真实回复。
`show` 始终可读取。终结或第五轮后的剩余问题交给人工，不能新建目录自动重启同一任务。
