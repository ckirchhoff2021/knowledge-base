# Codex CLI 完全使用教程：从安装、登录到自动化与自定义模型接入

> **教程日期**：2026-11-02　**实测版本**：codex-cli **0.142.2**（官方开源仓库 `github.com/openai/codex`）
>
> **TL;DR**：Codex CLI 是 OpenAI 开源（Apache-2.0）的终端 AI 编程助手。它不是简单的「终端版 ChatGPT」，而是一个能**读代码库、写文件、执行命令、自我验证**的 Agent，并由「沙箱 + 审批 + 规则文件」三重机制约束安全。本教程覆盖：① 四种安装方式；② 两种登录方式（ChatGPT 套餐 / API key）；③ 三种运行模式（交互 TUI / 非交互 exec / 代码评审 review）；④ 沙箱三档与审批策略；⑤ `~/.codex/config.toml` 配置全集与 `-c` 临时覆盖；⑥ AGENTS.md 规则文件与 `/init`；⑦ 会话管理（resume/fork/apply）；⑧ MCP 工具与插件；⑨ CI/CD 自动化；⑩ 接入本地开源模型（Ollama/LM Studio）和第三方兼容端点；⑪ 常见问题与 `codex doctor`。所有命令均在本机实测，不同小版本以 `codex --help` 和 `/help` 实际输出为准。

## 一、Codex CLI 是什么

### 1.1 定位

Codex CLI 是一个运行在终端里的**自主编程 Agent**。与普通问答式 AI 不同，它在接到任务后会自主执行一个循环：

```text
理解任务 → 浏览代码库 → 制定计划 → 编写代码 → 运行命令/测试 → 根据结果修正 → 交付
```

它可以独立完成「修复这个 bug」「给这个模块加单测」「重构这个函数」「评审这次改动」这类**多步骤、需要读写仓库、需要运行验证**的任务，而不只是输出一段代码让你自己粘贴。

### 1.2 安全模型：为什么敢让它执行命令

让 AI 在你的机器上执行 shell 命令天然有风险，Codex 用三重机制控制：

| 机制 | 作用 |
|---|---|
| **沙箱 Sandbox** | 限制命令的读写范围和网络访问（三档可选，见第五章） |
| **审批 Approval** | 超出沙箱或高风险操作前请求确认，可手动批准也可配置自动评审 |
| **规则文件 AGENTS.md** | 你写的项目约定，模型必须遵守（构建命令、代码风格、禁区，见第八章） |

理解这三个概念是用好 Codex 的前提——**你对「沙箱档位」和「审批策略」的选择，决定了它的自主性与安全性的平衡点。**

## 二、安装：四种方式任选

以下命令来自官方仓库 README（2026-11-02 核实）。

### 2.1 一键脚本（macOS / Linux）

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

该脚本默认从 `https://releases.openai.com/codex` 下载，不可用时回退 GitHub Releases。强制使用 GitHub Releases：

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | CODEX_INSTALLER_USE_RELEASES_OPENAI_COM=false sh
```

### 2.2 Windows（PowerShell）

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://chatgpt.com/codex/install.ps1 | iex"
```

### 2.3 包管理器

```bash
# npm（全平台，需要 Node.js）
npm install -g @openai/codex

# Homebrew（macOS）
brew install --cask codex
```

### 2.4 手动下载二进制

到 GitHub Releases 页面选对应平台压缩包：<https://github.com/openai/codex/releases/latest>

常用包：

| 平台 | 文件 |
|---|---|
| macOS Apple Silicon | `codex-aarch64-apple-darwin.tar.gz` |
| macOS Intel | `codex-x86_64-apple-darwin.tar.gz` |
| Linux x86_64 | `codex-x86_64-unknown-linux-musl.tar.gz` |
| Linux arm64 | `codex-aarch64-unknown-linux-musl.tar.gz` |

解压后里面的文件名带平台后缀，重命名为 `codex` 放进 `PATH` 即可。安装完验证：

```bash
codex --version
# 输出示例：codex-cli 0.142.2
```

## 三、登录认证：两种路线

首次运行 `codex` 会引导登录。官方推荐用 **ChatGPT 账号**，也可以用 **API key**。

### 3.1 用 ChatGPT 套餐登录（推荐）

直接运行 `codex`，选择 **Sign in with ChatGPT**，浏览器完成授权。这样 Codex 用量包含在你的 ChatGPT Plus / Pro / Business / Edu / Enterprise 套餐中，**不单独按 token 计费**，通常是性价比最高的方式。

### 3.2 用 API key 登录

```bash
# 方式一：交互输入（不留在 shell 历史里更安全）
codex login

# 方式二：管道传入
printenv OPENAI_API_KEY | codex login --with-api-key
```

用 API key 时按实际 token 消耗计费。查看当前登录状态：

```bash
codex login status
```

退出登录：

```bash
codex logout
```

> 凭据保存在 `$CODEX_HOME`（默认 `~/.codex/`）下的 `auth.json`，该文件不应提交到任何仓库。

## 四、三种运行模式

### 4.1 交互模式（TUI，默认）

在项目目录直接运行：

```bash
codex
# 或带初始任务
codex "帮我看看这个项目的结构，然后修复 utils.py 里日期解析的 bug"
```

进入全屏终端界面，底部输入框对话，主区域展示它的思考、文件改动和命令执行过程。

**常用快捷键**（具体以 `/help` 和实际版本为准）：

| 操作 | 按键 |
|---|---|
| 发送消息 | `Enter` |
| 换行 | `Shift` + `Enter`（部分终端） |
| 循环切换「审批/沙箱」模式 | `Shift` + `Tab` |
| 中断当前生成/执行 | `Ctrl` + `C` |
| 斜杠命令 | 输入 `/` 会弹出命令列表 |

**常用斜杠命令**（0.142 版命令很多，下表为最常用的稳定子集，输入 `/` 可看完整列表）：

| 命令 | 作用 |
|---|---|
| `/init` | 扫描仓库并生成 `AGENTS.md`（首次进入一个项目建议先跑） |
| `/model` | 查看/切换当前使用的模型 |
| `/diff` | 查看当前会话产生的所有改动 diff |
| `/review` | 对当前改动做一次代码评审 |
| `/clear` | 清空当前对话上下文（不删文件） |
| `/compact` | 手动压缩上下文（长会话省 token） |
| `/new` | 开始一个新会话 |
| `/config` | 查看当前生效的配置 |
| `/cwd` | 查看/切换工作目录 |
| `/agents` | 查看多 Agent / 子任务状态 |
| `/quit` | 退出 |

### 4.2 非交互模式（exec，适合脚本与自动化）

`codex exec` 把任务跑完后退出，不进入交互界面，是 shell 脚本和 CI 的入口：

```bash
# 直接传任务
codex exec "给 src/parser.py 补充单元测试，覆盖空输入和非法格式"

# 从 stdin 读任务
cat requirements.txt | codex exec "根据这份需求检查代码是否全部满足，列出差距"

# 指定工作目录、沙箱、模型
codex exec -C ./backend -s workspace-write -m "gpt-5.5" "修复所有 lint 错误"
```

`exec` 的常用专有参数（均经本机 `codex exec --help` 核实）：

| 参数 | 作用 |
|---|---|
| `-s, --sandbox <MODE>` | 沙箱：`read-only` / `workspace-write` / `danger-full-access` |
| `-C, --cd <DIR>` | 指定工作根目录 |
| `--add-dir <DIR>` | 追加可写目录（可重复） |
| `-m, --model <MODEL>` | 指定模型 |
| `-i, --image <FILE>` | 给初始 prompt 附加图片（可多个） |
| `--json` | 以 JSONL 输出事件流（方便程序解析） |
| `-o, --output-last-message <FILE>` | 把最终回复写入指定文件 |
| `--output-schema <FILE>` | 用 JSON Schema 约束最终输出结构 |
| `--skip-git-repo-check` | 允许在非 Git 目录运行（默认要求在仓库里） |
| `--ephemeral` | 不向磁盘保存会话文件 |
| `--ignore-user-config` | 不加载 config.toml（auth 仍生效） |
| `-p, --profile <NAME>` | 叠加使用指定配置 profile |
| `--dangerously-bypass-approvals-and-sandbox` | 跳过所有审批与沙箱（仅限外部已隔离环境，见第十章） |

### 4.3 代码评审模式（review）

`codex review` 专门针对改动做非交互评审：

```bash
# 评审相对 main 分支的改动
codex review --base main

# 评审未提交（含暂存/未暂存/未跟踪）的改动
codex review --uncommitted

# 评审某个具体 commit
codex review --commit <SHA>

# 自定义评审关注点
codex review --base main "重点检查并发安全和错误处理，风格问题不用提"
```

适合接在提交前或 PR 流程里，做一道自动化的第二双眼睛。

## 五、沙箱与审批：安全配置详解

### 5.1 沙箱三档（最重要）

`codex exec -s` 或 `--sandbox` 可选：

| 档位 | 文件权限 | 网络 | 适用 |
|---|---|---|---|
| **read-only** | 只读，不能改任何文件 | 默认关 | 让它分析、讲解、出方案，零风险 |
| **workspace-write**（最常用） | 可写当前工作区，但不能越界 | 可配置开关 | 绝大多数改代码任务 |
| **danger-full-access** | 整个文件系统可写 | 开 | 明确信任、需要系统级操作时 |

`workspace-write` 的网络可单独配置（config 里 `[sandbox_workspace_write]` 的 `network_access`，见第六章）。**默认从保守档位开始，需要时再放开——这是推荐的使用习惯。**

### 5.2 审批策略

当模型想做沙箱不允许的事（如写入工作区外、联网），会触发审批。`approval_policy` 常用值：

| 值 | 行为 |
|---|---|
| `untrusted` | 频繁确认，最保守 |
| `on-request` | 按请求审批（常用平衡项） |
| `on-failure` | 命令失败时才确认 |
| `never` | 不询问（配合宽松沙箱，风险自担） |

交互模式中最快捷的切换方式就是 `Shift+Tab`：在「自动执行（当前权限内）→ 只读 → 完全访问」几个组合间循环。

## 六、配置文件：~/.codex/config.toml

### 6.1 常用配置项（通用化示例）

`$CODEX_HOME/config.toml`（默认 `~/.codex/config.toml`）持久化所有偏好。一份带注释的典型配置：

```toml
# ── 模型 ──
model = "gpt-5.5"                          # 默认模型
model_reasoning_effort = "high"            # 推理投入：low/medium/high
web_search = "live"                        # 联网搜索：live 开启

# ── 安全 ──
approval_policy = "on-request"             # 审批策略
sandbox_mode = "workspace-write"           # 沙箱档位

# workspace-write 沙箱的细项
[sandbox_workspace_write]
network_access = true                      # 允许联网（改 false 完全离线）

# ── 项目信任（对应路径上的信任级别）──
[projects."/path/to/your/project"]
trust_level = "trusted"

# ── TUI 偏好 ──
[tui]
# 可在其中配置 keymap、显示等
```

### 6.2 自定义模型供应商（接第三方兼容端点）

Codex 支持自定义 provider，这对使用兼容 OpenAI 协议的网关 / 其他云 / 自建端点非常有用：

```toml
model = "your-model-name"
model_provider = "custom_provider"

[model_providers.custom_provider]
name = "Custom Provider"
base_url = "https://your-endpoint.example.com/v1"
wire_api = "responses"                     # "responses" 或 "chat"（按端点支持情况）
env_key = "CUSTOM_API_KEY"                 # 从该环境变量取 key（不写死）
model_context_window = 262144
max_output_tokens = 8192
supports_streaming = true
supports_function_calling = true
supports_vision = false
```

要点：**密钥一律走 `env_key` 指定的环境变量，绝不写进配置文件或命令行**；`wire_api` 要与端点实际支持的 API 形态匹配（新版用 responses，传统兼容端用 chat）。

### 6.3 临时覆盖：`-c` 与 profile

不必为一次性需求改配置文件：

```bash
# -c 覆盖任意键，值按 TOML 解析
codex -c 'model="o3"'
codex -c 'model_reasoning_effort="low"' "快速回答这个报错什么意思"
codex exec -c 'sandbox_workspace_write.network_access=false' "离线检查代码"

# profile：把不同场景的配置存成独立文件 $CODEX_HOME/<name>.config.toml，按需叠加
codex exec -p ci "运行完整测试并修复失败项"
```

用 `--strict-config` 可让无法识别的配置字段直接报错，排查拼写问题很好用。

## 七、会话管理：恢复、分叉、应用改动

Codex 会持久化会话（除非 `--ephemeral`），可以事后操作：

```bash
codex resume            # 弹出选择器，恢复之前的交互会话
codex resume --last     # 直接接上最近一次会话
codex fork --last       # 从最近会话分叉出一条新线（保留原会话不动）
codex archive           # 归档会话（选择器）
codex unarchive         # 取消归档
codex delete            # 永久删除会话

codex apply             # 把最近一次会话产生的 diff 用 git apply 应用到当前工作树
```

典型场景：昨天的任务没做完，`codex resume --last` 直接继续；想保留原版试个新思路，`fork` 一条出来。

## 八、AGENTS.md：给 Agent 立规矩

### 8.1 是什么

`AGENTS.md` 是放在仓库里的**项目指令文件**，Codex 每次启动自动加载，相当于整个团队对 Agent 的约定。建议用 `/init` 自动生成初稿，再手工完善。

### 8.2 推荐内容

```markdown
# AGENTS.md

## 项目概述
一句话说明这是个什么项目、技术栈。

## 常用命令
- 安装依赖：pip install -e .
- 运行测试：pytest -q
- 代码检查：ruff check .
- 启动服务：uvicorn app.main:app --reload

## 代码风格
- 类型注解必填；函数需 docstring；行宽 100。

## 注意事项 / 禁区
- 不要修改 generated/ 目录（自动生成）。
- 不要提交 .env；数据库迁移需人工确认。

## 测试要求
- 改完代码必须跑 pytest；新增逻辑需带单测。
```

层级规则：根目录的 `AGENTS.md` 始终生效，子目录可放各自的指令，进入该子树时自动叠加。**它比口头在 prompt 里重复要求更可靠，且可以 Git 管理、团队共享。**

## 九、MCP 工具与插件

### 9.1 MCP（接外部能力）

Codex 内建 MCP 客户端，可接入数据库、浏览器、内部 API 等外部工具：

```bash
codex mcp list                          # 查看已配置的 server
codex mcp add mydb -- npx -y some-mcp-server   # 添加一个 stdio 类型 server
codex mcp get mydb                      # 查看详情
codex mcp remove mydb                   # 移除
codex mcp login mydb                    # 处理该 server 的认证
```

### 9.2 插件

```bash
codex plugin --help       # 插件的安装/列出/管理
```

适合把常用 MCP 组合、主题、额外命令打包复用。

## 十、自动化与 CI/CD

`exec` 模式 + JSON 事件流是自动化的标准组合。

### 10.1 本地脚本示例

```bash
codex exec \
  --json \
  -s workspace-write \
  --skip-git-repo-check \
  -o /tmp/last_msg.md \
  "修复编译错误并运行全部测试，全部通过后总结改动"
```

- `--json` 让外层程序逐行解析 Agent 的动作（思考、命令、文件改动、结果）；
- `-o` 拿到最终结论用于后续通知或门禁判断。

### 10.2 在 CI 中运行

CI Runner / 容器本身已经是隔离环境，此时可以使用：

```bash
codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  --ephemeral \
  --skip-git-repo-check \
  "确保 CI 失败的测试被修复；不要推送，不要修改 CI 配置"
```

⚠️ **`--dangerously-bypass-approvals-and-sandbox` 名字里的危险是认真的**——它完全放开权限，**只能在容器/CI 这种外部已隔离、可丢弃的环境使用**，绝不能在日常开发机图省事加这个参数。`--ephemeral` 让会话不留痕，适合无状态 CI。

### 10.3 结构化输出对接流水线

需要 Agent 产出给程序消费时：

```bash
codex exec --output-schema result.schema.json -o result.json \
  "分析本次构建日志，按 schema 返回失败原因分类与建议"
```

## 十一、接入本地开源模型（离线/隐私）

不想把代码发到云端，可以用本地 provider：

```bash
# 走开源 provider，交互式选择或指定本地后端
codex exec --oss --local-provider ollama "解释这个函数在做什么"
codex exec --oss --local-provider lmstudio "给这个文件写注释"

# 纯只读 + 本地模型，实现完全离线分析
codex exec --oss --local-provider ollama -s read-only \
  -c 'sandbox_workspace_write.network_access=false' "审查这段代码的安全问题"
```

前提是本机已启动 **Ollama** 或 **LM Studio** 并加载了支持工具调用的模型。注意：本地小模型在复杂多步 Agent 任务上的能力与云端旗舰差距明显，适合分析、注释、简单修改，复杂任务仍建议云端模型。

## 十二、常用命令速查表

| 需求 | 命令 |
|---|---|
| 开始交互 | `codex` |
| 直接下任务 | `codex "任务描述"` |
| 非交互执行 | `codex exec "任务描述"` |
| 只读分析（零风险） | `codex exec -s read-only "讲讲这个模块"` |
| 指定模型 | `codex -m "gpt-5.5"` |
| 临时改配置 | `codex -c 'approval_policy="never"'` |
| 生成 AGENTS.md | TUI 内 `/init` |
| 评审改动 | `codex review --base main` |
| 评审未提交改动 | `codex review --uncommitted` |
| 继续昨天的会话 | `codex resume --last` |
| 应用最近改动 | `codex apply` |
| 添加 MCP 工具 | `codex mcp add NAME -- CMD` |
| 本地开源模型 | `codex exec --oss --local-provider ollama` |
| 诊断问题 | `codex doctor` |
| 查看登录状态 | `codex login status` |
| Shell 补全 | `codex completion bash`（支持 zsh/fish 等） |
| 更新 CLI | `codex update` |

## 十三、常见问题与排查

**Q：执行命令被拦 / 权限不足？**
用 `Shift+Tab`（交互）或 `-s`（exec）切换更合适的沙箱档位；或检查 `[projects.路径]` 的 `trust_level`。不要直接跳到 `danger-full-access`，优先用 `workspace-write` + `--add-dir` 精确放权。

**Q：提示不在 Git 仓库无法运行？**
Codex 默认要求在 Git 仓库中工作（便于 diff 和回滚）。确认目录无误后加 `--skip-git-repo-check`；更推荐先 `git init`，这样改动可审查、可撤销。

**Q：模型调用失败 / 认证异常？**
先跑 `codex doctor`，它会检查安装、配置、auth 和运行时健康；再 `codex login status` 确认登录方式与套餐/额度。

**Q：长会话变慢、token 爆？**
用 `/compact` 手动压缩，或 `/clear` 重新开始；自动化场景可拆成多个 `codex exec` 子任务。

**Q：自定义端点连不上？**
核对 `base_url` 是否正确、`wire_api` 是否与端点匹配（responses vs chat）、`env_key` 指向的环境变量是否已设置。

**Q：担心它乱改东西？**
从 `read-only` 开始，让它先出方案；确认后再用 `workspace-write`；改动统一用 `/diff` 检查；由于基于 Git，不满意可 `git checkout` 回滚。让 Agent「先计划后执行」是最稳的协作方式。

## 十四、参考资料

### 官方一手资料

1. Codex 开源仓库：https://github.com/openai/codex （Apache-2.0）
2. Codex 官方文档入口：https://developers.openai.com/codex
3. 交互模式特性：https://developers.openai.com/codex/cli/features
4. 斜杠命令：https://developers.openai.com/codex/cli/slash-commands
5. 认证方式：https://github.com/openai/codex —— 见 docs/authentication
6. GitHub Releases：https://github.com/openai/codex/releases/latest
7. ChatGPT 套餐包含 Codex 的说明：https://help.openai.com/en/articles/11369540-codex-in-chatgpt

### 知识库内相关教程

8. 《Hermes Agent API 指南》，`agents/hermes/API.md`——另一款本地 Agent 的 API 调用方式，可对照
9. 《Agent Memory 系统设计报告》，`models/agent-memory-systems/`——理解 Codex 两阶段记忆流水线
10. 《Agent Harness 报告》，`models/agent-harness/`——理解「Agent = Model + Harness」，AGENTS.md/沙箱都属于 Harness 层
11. 《Context Engineering 报告》，`models/context-engineering/`——长会话上下文管理与 `/compact` 的原理

---

*本教程命令基于 codex-cli 0.142.2 实测（2026-11-02），CLI 迭代较快，参数与斜杠命令可能随版本增减；以你本机 `codex --help`、`codex exec --help` 和 TUI 内 `/help` 为最终依据。*
