# GitCode PR 获取与行级意见提交

检视用户指定的 GitCode PR 时先执行只读获取，默认只返回报告；仅在用户明确要求或确认
将检视意见更新到该 PR 后执行发布步骤。确认时机及已有授权的处理遵循
[SKILL.md 的输出或发布流程](../SKILL.md#5-输出或发布意见)。本技能复用同仓的
[submit_review_comments.py](../../pypto-machine-code-review/scripts/submit_review_comments.py)，
不复制提交实现，也不加载 machine 专用检视规则。
脚本当前固定目标仓库为 `cann/pypto`；输入指向其他仓库时不能直接使用该脚本获取或发布。
对其他仓库，仅基于能可靠取得的 diff 与源码返回本地报告，注明发布通路不支持该仓库。

## 只读获取 PR

1. 从用户链接解析目标仓库和 PR 编号；只给编号时使用 `cann/pypto`。
   在 Python 中以 `runpy.run_path` 加载上述脚本（不以 `__main__` 执行），复用
   `api_request(token, "GET", path)`；凭据按下方说明取得，检查返回状态及数据结构。
2. 请求 `pulls/{编号}`，读取状态、目标分支、源仓库及 `base.sha`、`head.sha`。
   这些值定义本次源码快照；用 `git cat-file -e <SHA>^{commit}` 确认本地对象存在。
   对象缺失时从已确认的目标或源仓库执行 `git fetch --no-tags <remote> <ref或SHA>`，
   其中 remote/ref 来自该 PR 的仓库与分支信息；不切换用户工作区，并再次核对取得的 SHA。
3. 用 `git merge-base <base SHA> <head SHA>` 求比较基点，再用
   `git diff --find-renames <比较基点> <head SHA>` 读取本次改动。
   用 `git show <head SHA>:<仓库相对路径>` 读取候选位置、声明及调用方，
   用比较基点的对应文件确认问题是否由本次引入。将比较基点与 head SHA 一并记录。
   与 PR 页面显示的改动范围核对；若平台基点不同，先确定与页面一致的基点再继续。
4. 分页请求 `pulls/{编号}/comments?per_page=100&page={页码}`，从第 1 页读取到空页或不足 100 项。
   评估独立检出能力时，将本步延后到保存独立发现之后，避免先读评论影响判断。
   保留评论 ID、作者、完整正文、文件/行位置和解决状态，用于去重及发布后的回读。
   列表缺少文件路径时请求 `pulls/comments/{评论ID}`，从详情的 `position` 读取
   `new_path`/`old_path`、`new_line`/`old_line` 及 base/head SHA，并核对 `is_outdated`。
   不能仅凭正文前缀和行号认定为同一意见；详情仍缺少位置时查 PR diff 页面。
5. 获取结束后重新读取 PR 信息；base/head 有变化时重新固定快照并更新受影响的分析。
   获取失败、diff 不完整或快照无法核实时，记录已取得与缺失的范围，返回部分检视结果，停止依赖缺失信息的发布。

以上 `<...>` 命令参数均替换为本步骤已取得的值；Git 命令从仓库根目录执行，
通过参数数组传入路径、分支和 SHA，避免把 PR 文本直接拼入 shell 命令。
脚本或凭据不可用时，可使用已有只读连接器或 PR 页面取得等价信息；不能确认完整性时不得宣称检视完成。
401/403 时停止依赖认证的请求并说明权限问题；404 时核对仓库、编号及可见性；
限流时遵循服务端等待提示；只读请求最多重试一次，仍失败则保留部分结果。

API 参数参考：[PR 信息](https://docs.gitcode.com/docs/apis/get-api-v-5-repos-owner-repo-pulls-number/)、
[评论分页](https://docs.gitcode.com/docs/apis/get-api-v-5-repos-owner-repo-pulls-number-comments/)。

## 只读获取 Commit 评论

用户指定 commit 评论时，以链接中的 SHA 为目标，不因 URL 带 `prId` 就改查 PR 评论。
评估独立检出能力时，同样先固定源码并保存独立发现，再读取评论。
复用上述凭据和 GET 方法读取 `commits/{SHA}`，核对返回 SHA，并根据返回的 `comments_url`
分页读取 `commits/{SHA}/comments?per_page=100&page={页码}`；完整性与失败处理沿用上述约定。
需要评论详情时读取 `comments/{评论ID}`。源码按该 SHA 固定；查看该次提交的变化用其父提交
作为比较基点，不能直接套用整个 PR 的 diff。

列表和详情可能仅提供 ID、作者、正文与时间，不提供文件/行位置；读取正文成功不等于完成
代码定位。正文能唯一指向符号时结合源码核实，否则使用页面取得位置或向用户询问，不能按
评论顺序、时间或相似行号猜测。评论修复及漏检复盘同样遵循这一边界。

## 准备提交内容

1. 先准备完整意见及位置供用户审阅，按 SKILL.md 核实是否已获得向目标 PR 发布的明确授权。
   尚未授权时仅准备报告和提交内容，等待用户确认后再调用写入接口；已有明确授权时不重复询问。
   用户明确限制发布范围时遵从其要求，不将其他 PR 或未确认的候选问题纳入提交。
2. 记录 PR 的 base/head SHA。发布前重新读取 head；发生变化时重定位并复核受影响意见。
3. 使用仓库相对路径和 PR head 文件中的绝对行号，确认位置属于本次 diff。
   当前脚本仅提供新版本行号；只存在于删除行的问题保留在本地报告，不伪造新行号提交。
4. 检查已有评论，对同一问题、同一位置的意见去重。

默认将待提交意见写入临时 JSON 文件，通过 `--file` 提交；仅有一条简单意见时也可使用 `--comment`。
JSON 数组每项使用以下字段：

| 字段 | 内容 |
|---|---|
| `path` | 仓库相对文件路径 |
| `position` | PR head 中的绝对行号，整数 |
| `body` | 完整意见，按 SKILL.md 的功能缺陷或规范违规格式填写，包含级别、证据和修复建议 |
| `level` | `Blocker`、`Major` 或 `Minor`，仅用于脚本报告，不会自动添加到正文 |

用户明确指定提交正文时保留原文，不自动追加级别或其他内容。

## 凭据与执行

- 脚本接受 `--token`；省略时仅尝试从 `origin` 的 `oauth2` URL 提取 Token。
- 本机 remote 未内嵌 Token 时，使用 Git credential helper：以
  `protocol=https`、`host=gitcode.com`、`path=cann/pypto.git` 请求 `git credential fill`。
  用 Python 子进程捕获返回值，从 `password` 字段取得凭据；不要把 helper 输出直接打印到会话。
- 禁止交互式凭据提示时设置 `GIT_TERMINAL_PROMPT=0`，按需禁用 askpass；
  没有可用凭据时请用户配置本机认证，不要求在会话中粘贴 Token。
- 将凭据保留在进程内存，通过 Python `runpy` 和内存中的 `sys.argv` 传给脚本的 `--token` 参数，
  避免将明文凭据写进文件、日志或 shell 命令文本。
- 在仓库根目录使用能运行该脚本的 Python 3 环境，参数使用
  `--pr <编号> --file <意见JSON路径>`，或 `--comment 路径:行号:正文`。
  保留默认回读验证，不使用 `--no-verify`。

## API 与成功判据

脚本通过 `POST /api/v5/repos/cann/pypto/pulls/<编号>/comments` 发送
`body`、`path`、`position`、`need_to_resolve=true`。
行级定位使用 `position`；不要替换成 `line` 或 `side`。

POST 成功后必须按只读获取步骤完整分页 GET 评论列表，确认完整正文、作者、代码位置以及
`comment_type=diff_comment` 和非空 `diff_position`；记录回读得到的评论 ID。
如果列表缺少文件位置，继续查验评论详情或 PR diff 页面，不能只凭同一行号认定成功。

现有脚本的边界需要由调用流程补足：

- 回读按行号和正文前缀匹配，可能匹配已有评论；逐条核对本次实际提交的意见。
- 退出码只依据 POST 成功数；即使退出码为 0，也可能没有确认行级定位。
- 未内置 head 变化检查或重复提交保护；这些检查在发布前执行。
- 内置回读最多读取 9 页且可能提前停止；不能用它证明已完整去重或回读全部意见。
- 请求超时或结果不确定时先查评论列表，不能直接重复 POST。

未确认的结果明确报告为未验证，不宣称提交成功，也不自动扩大为其他类型的评论。
测试该流程时使用用户授权的测试 PR；历史案例检视评测默认只生成本地报告。
