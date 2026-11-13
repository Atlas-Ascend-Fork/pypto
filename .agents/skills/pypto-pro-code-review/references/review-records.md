# 一份证据与分批复核记录

规则 reviewer 使用 `coverage.json` 维护事实和逐规则判断；候选问题正文可单独保存并引用证据 ID。
不再另写一份逐函数反查论证。功能 reviewer 使用同样的证据编号原则记录行为路径，无需填写规则表。
同一事实可以复用，各规则的适用性与结论仍须分别判断。独立角色之间首次发现不共享事实总结。

## 输入与记录

主 agent 从固定 diff 和完整规则文件准备 `manifest.json`，包含：

- `snapshot`：源码、比较基点、实际目标和规则快照标识组成的对象。
- `rules`：全部通用和已加载模块的规则编号列表；新增模块影响时追加，不静默删减。
- `files`：完整受审变更文件列表。程序性 skill 变更等不属于框架检视的文件仍列入清单，另注明排除理由。

`coverage.json` 与 manifest 的 `snapshot` 相同，其余字段如下：

```json
{
  "snapshot": {"source": "fixed-source", "base": "fixed-base", "target": "fixed-target", "rules": "fixed-rules"},
  "subjects": [{"id": "S1", "path": "include/example.h", "symbol": "Example"}],
  "excluded_files": {},
  "evidence": [{"id": "E1", "locations": ["include/example.h:Example", "binding/example.cpp:Example"], "fact": "记录实际读取后确认的事实及必要反例，不预填结论。"}],
  "checks": [{"id": "C1", "rule": "PRO-API-001", "subjects": ["S1"], "evidence": ["E1"], "status": "checked", "basis": "针对本规则的判断及候选编号；事实引用 E1。"}]
}
```

- `subjects` 列全修改的函数、状态、类型、注册点及测试等检查对象；同一文件不能只列一个代表函数。
  由 reviewer 对照完整 diff 核实，verifier 独立确认。脚本只能检查文件/ID 覆盖，不能证明符号清单完整。
- `excluded_files` 为文件路径到排除理由的映射；主 agent 与 verifier 核对排除是否符合用户范围。
- `evidence` 保存一次事实与完整定位。需要补查时更新事实，不能只增加引用数量。
- 每条 `check` 对应一个规则，可覆盖有同一判断依据的多个修改点。`status` 为 `checked`、
  `not_applicable` 或 `pending`；`basis` 写该规则的判断或缺口，不能只有“通过”。
  `checked` 必须引用源码证据及修改点；`not_applicable` 必须说明条件为何不成立。
  所有适用规则和实际位置均要检查，不因已有某条 check 而省略其余位置。
- 增量沿用的 check 在 `basis` 中引用旧记录及复核结果，并注明源码、规则、目标接口、依赖和契约
  未受影响的证据；这些条件不成立时重新检查。脚本不判断语义失效，也不自动授权沿用。

## 执行与交接

1. 列出全部对象、规则及调用链。按链批量读取完整函数，记录事实，再逐条完成相关判断。
2. 每完成一批，将有效 JSON 保存为一个不再覆盖的批次快照，交主 agent 通知 verifier。
   reviewer 继续补充自己的工作记录，所有批次使用同一 source/rules snapshot。
3. verifier 逐项回源检查候选和覆盖，记录已复核的批次/证据/check ID。返回结果只交主 agent；
   reviewer 首次发现全部冻结后，才回传反例、分歧和新候选。结果修订显式标出受影响的旧 ID。
4. 最终记录引用既有事实，补齐缺口。verifier 核对最终版本和全部批次；每一条适用检查都要覆盖，
   不能用早期批次通过代替整轮完成。共同事实只需回源确认一次，不代表其他规则判断自动成立。

使用小工具生成两种视图并检查结构：

```bash
python <skill>/scripts/review_coverage.py manifest.json coverage.json --partial --output coverage-progress.md
python <skill>/scripts/review_coverage.py manifest.json coverage.json --output coverage.md
```

`--partial` 允许尚未覆盖的项存在，并在输出中明确显示缺口；最终检查不允许缺口或 pending。
脚本只检查版本一致、ID/引用、清单覆盖和记录完整性，不评价事实真伪、排除理由、规则适用性或
符号清单是否完整。退出成功不代表检视通过，仍需独立 verifier 和 SKILL.md 的全部完成条件。
输入记录无需再转写成另一套长表；生成的视图是同一份记录的索引。遇到工具失败保留原记录并
修复具体结构问题，不重写已确认的事实或整轮重查。
