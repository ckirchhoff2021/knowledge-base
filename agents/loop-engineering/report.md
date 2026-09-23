# Loop Engineering（循环工程）技术调研报告

> 主题：系统梳理 2026 年爆火的 Loop Engineering（循环工程）——从「人给 Agent 写提示词」到「人设计给 Agent 写提示词的循环系统」的范式转移。覆盖概念定义、形式化拆解、四类循环、5+1 工程组件、循环契约与验证者模式、工程反模式与落地 checklist。
>
> 调研日期：2026-09-23 ｜ 主要信源：Addy Osmani 原博、Anthropic/Claude 官方博客、Andrew Ng《The Batch》、arXiv:2607.00038（均已抓取原文核实）+ 知乎社区实践文章 20+ 篇

---

## 目录

1. [背景与问题定义](#1-背景与问题定义)
   - [1.1 一句话定义与引爆事件](#11-一句话定义与引爆事件)
   - [1.2 旧模式的三个痛点：验收、记忆、推进全靠人](#12-旧模式的三个痛点验收记忆推进全靠人)
   - [1.3 范式定位：Prompt → Context → Harness → Loop](#13-范式定位prompt--context--harness--loop)
   - [1.4 直觉类比：human in the loop → human on the loop](#14-直觉类比human-in-the-loop--human-on-the-loop)
2. [原理详解：一个 Loop 由什么构成](#2-原理详解一个-loop-由什么构成)
   - [2.1 形式化：Loop Specification 五要素](#21-形式化loop-specification-五要素)
   - [2.2 关键消歧：Agent Loop ≠ Loop Engineering（内环 vs 外环）](#22-关键消歧agent-loop--loop-engineering内环-vs-外环)
   - [2.3 运转机制：一个循环周期的六个动作](#23-运转机制一个循环周期的六个动作)
   - [2.4 停止条件：必须命名终态](#24-停止条件必须命名终态)
   - [2.5 理论脉络：ReAct → Reflexion → Ralph → Loop Engineering](#25-理论脉络react--reflexion--ralph--loop-engineering)
3. [方案设计：四类循环、5+1 组件与三层嵌套](#3-方案设计四类循环51-组件与三层嵌套)
   - [3.1 Anthropic 四类循环：触发、停止与适用场景](#31-anthropic-四类循环触发停止与适用场景)
   - [3.2 Addy Osmani 的 5+1 工程组件](#32-addy-osmani-的-51-工程组件)
   - [3.3 吴恩达三层嵌套循环](#33-吴恩达三层嵌套循环)
   - [3.4 验证者模式与五级验证阶梯](#34-验证者模式与五级验证阶梯)
   - [3.5 方法演进谱系（2025-07 → 2026-07）](#35-方法演进谱系2025-07--2026-07)
4. [核心实现：循环契约与 builder/checker 模式](#4-核心实现循环契约与-builderchecker-模式)
   - [4.1 循环契约模板](#41-循环契约模板)
   - [4.2 builder/checker 伪代码](#42-builderchecker-伪代码)
   - [4.3 状态文件与 SKILL.md 示例](#43-状态文件与-skillmd-示例)
   - [4.4 工程反模式与典型坑](#44-工程反模式与典型坑)
5. [结论与选型建议](#5-结论与选型建议)
6. [参考资料](#6-参考资料)

---

## 1. 背景与问题定义

### 1.1 一句话定义与引爆事件

**Loop Engineering（循环工程）是：把「人」从逐轮提示 Agent 的位置上替换下来，改为设计一套自动驱动 Agent 的循环系统。**

Addy Osmani（曾任 Google Chrome、Google Cloud AI 开发者体验负责人）2026 年 6 月 7 日在个人博客发表《Loop Engineering》，给出的原始定义是：

> "Loop engineering is replacing yourself as the person who prompts the agent. You design the system that does it instead."（循环工程就是把"提示 Agent 的那个人"替换成你自己设计的系统。）—— addyosmani.com，2026-06-07

概念被两句话点燃，Osmani 在原文中均有引述：

- **Peter Steinberger**（开源 Agent 项目 OpenClaw 作者）："You shouldn't be prompting coding agents anymore. You should be designing loops that prompt your agents."（你不该再给编码 Agent 写提示词了，你该去设计给 Agent 写提示词的循环。）
- **Boris Cherny**（Anthropic Claude Code 负责人）："I don't prompt Claude anymore. I have loops running that prompt Claude and figuring out what to do. My job is to write loops."（我已经不直接提示 Claude 了，我有一堆循环在运行，由它们提示 Claude 并决定下一步，我的工作是写循环。）

两周多后，两个权威来源把社区口号变成了正式框架：

- **2026-06-26**，吴恩达（Andrew Ng）在 DeepLearning.AI《The Batch》发表《Three Key Loops for Building Great Software》，把循环抽象为三层嵌套反馈环；
- **2026-06-30**，Anthropic 官方博客发布《Loop engineering: Getting started with loops》（作者 Delba de Oliveira、Michael Segner），给出 Agent 循环的官方定义与四类分类，标志 Loop 从社区实践变成**平台级功能**。

### 1.2 旧模式的三个痛点：验收、记忆、推进全靠人

过去两年使用编码 Agent 的标准姿势是：写一个好 prompt + 塞足上下文 → Agent 回一版 → 人读结果、跑测试、贴报错 → 再写下一条 prompt。这个「人肉循环」里，Agent 只是活塞，人才是引擎：

| 环节 | 手动 Prompt 模式下谁在做 | 痛点 |
|---|---|---|
| **验收（verification）** | 人：手动跑、肉眼看、判断"看起来能登录了" | 主观、易漏；人成为吞吐瓶颈 |
| **记忆（memory）** | 人：记得上次踩过什么坑、进度到哪 | 上下文一关全忘；同一个坑反复踩 |
| **推进（progression）** | 人：决定下一步问什么、何时停 | 人不说话 Agent 就不动，无法过夜运行 |

知乎社区的一个典型吐槽（锦恢《小白教程（五）》）：Agent 说网站做完了，打开一看注册登录有 bug、几个路由只实现了五成；人只能重新编织 prompt、等它再写一轮、再检查——**这里面没有真正的思考，全是重复劳动**。

### 1.3 范式定位：Prompt → Context → Harness → Loop

Loop Engineering 不是凭空出现的第四个营销词，而是 AI 编程技术栈逐层上移的最新一层（Osmani 原文明确说 "Loop engineering sits one floor above the harness"）：

| 层级 | 解决的问题 | 单位 | 人的位置 |
|---|---|---|---|
| **Prompt Engineering**（提示工程） | 一次交互怎么说清楚 | 单轮请求 | 每一轮都在 |
| **Context Engineering**（上下文工程） | 正确的信息在正确时机进入模型 | 单次任务 | 每轮筛选材料 |
| **Harness Engineering**（脚手架/驾驭工程，2026 年初流行） | 单个 Agent 运行的环境：工具注册、权限、上下文、状态、重试 | 单个 Agent 运行时 | 设计运行环境 |
| **Loop Engineering**（循环工程，2026-06） | 谁来触发、带什么材料、用什么证据验收、何时停 | 跨轮/跨会话的流程 | 站在循环之上设计它 |

关键关系：**Harness 是 Loop 得以运行的底座**。Harness 提供 Agent 内部的"感知—行动—观察"机制，而 Loop 是按时启动、在 Harness 上派生多个 helper agent、并自我喂养（feeds itself）的外部系统。arXiv:2607.00038 特别强调要把外部的 loop specification 与两样容易混淆的东西区分开：①普通编程里的 while 循环；②Harness 已经内置的 perceive-act-observe 管道循环。

> **注意**：Loop Engineering 并不淘汰 Prompt Engineering。循环内的每一次模型调用仍要靠 prompt、上下文和工具描述工作；变化的是工程注意力从"这一句怎么写"移到了"谁触发、带什么材料、用什么证据判停"。

### 1.4 直觉类比：human in the loop → human on the loop

- **手动挡 → 自动驾驶**：以前像开手动挡，踩油门、换挡、转弯每步自己来；Loop Engineering 像设定好目的地和安全规则的自动驾驶，遇红灯自己停、路况变了自己绕、到点自己停，人只需偶尔看仪表盘（喻体来自程序员鱼皮、Simon Wong 等多篇知乎文章）。
- 更精确的表述是位置的变化：**从 human in the loop（人是循环里亲手转齿轮的人）变成 human on the loop（人站在循环之上，设计循环、观察循环、必要时介入）**。

![图 1 占位](diagrams/)

---

## 2. 原理详解：一个 Loop 由什么构成

### 2.1 形式化：Loop Specification 五要素

arXiv:2607.00038《Stop Hand-Holding Your Coding Agent: Engineering the Loops that Replace Step-by-Step Prompting》（作者 Sandeco Macedo，2026-06-28 提交）把实践对象形式化为 **loop specification（循环规约）**：一份有界、可复用的工件，由人交给 Agent Harness（如 Claude Code、Codex），让 Agent 自主追求目标，替代逐步提示。其五个组成部分为：

```text
LoopSpec := (Trigger, Goal, Verification, StoppingRule, Memory)

  Trigger       触发方式：手动 / 定时 / 事件（决定"谁发起下一轮"）
  Goal          目标：可验证的完成标准（最好是确定性、可机检的）
  Verification  验证步骤：由谁、用什么证据判定一轮结果（独立于执行者）
  StoppingRule  停止规则：终态集合 + 轮次/时间/预算上限 + 升级条件
  Memory        记忆：存活在单次对话之外的状态（progress.md / issue board / git 历史）
```

Osmani 的工程化版本是「五个组件 + 一个记东西的地方」（five building blocks plus state，见 §3.2），与论文的五元组在工程实现上一一对应：Automations 对应 Trigger，Skills/Sub-agents 服务于 Verification，State 即 Memory。

一个最小 loop 的递归目标语义（Osmani 原文）：

```text
loop(purpose):
    while not done:
        result = agent.act( purpose, memory.read(), tools )
        verdict = verifier.check( result, goal )      # 注意：不是执行者自己判
        memory.write( result, verdict )
        done = stopping_rule( verdict, budget, state )
    escalate_to_human() if verdict == "cannot_resolve"
```

### 2.2 关键消歧：Agent Loop ≠ Loop Engineering（内环 vs 外环）

这是中文社区最容易混淆的一点（lufeikevin、CRUD 等多篇知乎文章专门做了区分）：

| 维度 | Agent Loop（内环 / inner loop） | Loop Engineering（外环 / outer loop） |
|---|---|---|
| 本质 | 模型与工具之间的内部循环 | 面向需求与验收标准的外部自执行流程 |
| 位置 | Harness 内部的管道 | 构建在 Harness 之上 |
| 典型过程 | 读上下文 → LLM 决策 → 调工具 → 结果写回 → 直到 stop_reason 不再是 tool_use | 触发 → 取状态 → 派活 → 独立验证 → 判停/升级 → 落盘 → 下一轮 |
| 解决的问题 | 单次对话/单个任务能跑完 | 人工验收、跨任务调度、跨会话记忆 |
| 状态存活处 | 上下文窗口内（消息历史） | 磁盘上的持久文件（SKILL.md、progress.md、Linear） |

知乎《打造 Claude Code 可持续推进的工作流》给出的总结很精炼：**Inner Loop 决定这一轮跑不跑得通，Outer Loop 决定下一轮还踩不踩同一个坑。**

内环的最小代码形态（袋鼠云数栈 UED《Agent Loop 核心机制》总结）：`messages` 是短期记忆；`stop_reason === "tool_use"` 表示模型要继续；工具结果重新写回消息历史。这是"为什么能跑"的最小心智模型，真实源码还要处理事件流、上下文压缩、并行工具、用户中断等稳定性问题。

一个生产级内环实例是 Claude Code 插件 Pi 的两级循环（知乎《从 Pi 窥见 Loop Engineering 的递归美学》，基于源码）：

- **run / turn 两级**：ReAct 是单层 loop，Pi 拆成 run（一次完整任务）与 turn（一轮模型交互），子 Agent 再嵌套完整 loop；
- **两个队列**：steering（转向，用户中途插话，消息排队到当前 turn 结束、下一次请求模型前注入）与 follow-up（追问，Agent 准备退出前检查队列，有货就再跑一轮）——中途插话靠"两个 turn 的缝隙注入"实现，无需打断；
- **稳定事件流**：`agent_start / turn_start / message_* / tool_execution_* / turn_end / agent_end`，UI、会话保存、扩展、自动压缩全部挂在事件上，不伸手进循环；
- **边界处理**：输出被 token 上限截断时，该消息内所有工具调用**整批标记失败、一个都不执行**（流式解析器会"抢救"出残缺 JSON 参数，执行残缺的 write 可能毁文件）；工具异常不中断循环，包装成结果喂回模型。

### 2.3 运转机制：一个循环周期的六个动作

综合官方四类循环与知乎实践，一个外环周期可以拆成六个动作：

1. **触发（trigger）**：手动 prompt、定时器（`/loop 5m`、cron）、或外部事件（新 issue、CI 失败、PR 评论）。
2. **装载状态（load state）**：从磁盘读取 progress 文件/看板，知道昨天试过什么、什么还开着——模型跨 run 会忘，仓库不会忘（"The agent forgets, the repo doesn't."——Osmani）。
3. **执行（act）**：在隔离 worktree 中由主 Agent 或子 Agent 干活，工具调用发生在真实环境里。
4. **观察（observe）**：收集测试、lint、浏览器截图、监控 Trace 等客观证据。
5. **验证（verify）**：由**独立验证者**（另一个 Agent / 评估模型 / 确定性脚本）对照 Goal 判定，执行者不给自己打分。
6. **判停与落盘（decide & persist）**：对照停止规则决定成功退出、继续、预算耗尽失败、还是升级给人；把结果写回 Memory，下一轮从断点继续。

### 2.4 停止条件：必须命名终态

Anthropic 官方对 loop 的定义只有一句话：**"agents repeating cycles of work until a stop condition is met"（Agent 重复工作周期，直到满足停止条件）**。工程上最容易被轻视、也最容易出事的就是停止条件。

沐风《把 /goal 和 /loop 变成可验证的 AI 工程循环》给出的实战教训："很多人第一次用循环踩坑，不是模型太弱，而是没有写 Stop Rules。**没有刹车的 Agent，不是自动化，是成本风险。**"

一个完整停止规则至少命名四种出口（综合 §4.1 循环契约实践）：

```text
SUCCESS          所有确定性检查通过（如 tests/auth 12 个用例全绿 + lint 零警告）
BUDGET_EXHAUSTED 轮次/时间/Token 预算用尽（如最多 5 轮、单轮 15 分钟超时）
REPEAT_FAILURE   同一检查连续 N 轮以同样方式失败 → 停，整理 diff 和日志升级给人
BLOCKED          触碰硬性约束（要改迁移文件、要加依赖、缺权限）→ 停，等人决策
```

论文 arXiv:2607.00038 对 50 个真实公开 loop 的手工编码统计（**该数字来自论文摘要转述，引用前建议查原文核对**）：74% 的 loop 显式命名了终态（named terminal states），而自动化触发与持久记忆仍是相对薄弱的环节。

### 2.5 理论脉络：ReAct → Reflexion → Ralph → Loop Engineering

- **ReAct（2022，Princeton × Google，arXiv:2210.03629）**：推理与行动交替，`thought → action → observation → …`，这是内环 Agent Loop 的学术原型。
- **Reflexion（2023，arXiv:2303.11366）**：任务失败后用自然语言反思原因并存入记忆，下一次尝试调用记忆改进表现。Loop 工程里"失败 → 反思 → 写进 SKILL.md/progress → 下一轮受益"的外环学习，本质就是工程化的 Reflexion。
- **Ralph（2025-07 → 2026-01）**：Geoffrey Huntley 的 Claude Code 插件，一个 bash 脚本驱动的循环：开发功能 → 检查预期 → 继续开发，直到全部完成；记忆不靠上下文窗口，靠 git commit 和文本文件。2025-07-14《Ralph Wiggum as a 'Software Engineer'》、2026-01-17《Everything Is a Ralph Loop》（ghuntley.com，本报告已核实可访问）。当时还没有 Loop Engineering 这个词。
- **平台原语化（2026-05 → 06）**：2026 年 5 月 Codex 上线 `/goal`；6 月 7 日 Osmani 定名；6 月底 Claude Code 与 Codex 都内置了 Automations、worktree、skills、subagents、state 全套能力。Osmani 感慨：一年前跑 loop 要写一堆 bash 并永久维护，现在零件直接随产品发布，"一旦发现形状相同，你就不再争论用哪个工具"。

---

## 3. 方案设计：四类循环、5+1 组件与三层嵌套

### 3.1 Anthropic 四类循环：触发、停止与适用场景

Claude Code 团队按**四件事**分类循环：怎么触发、怎么停止、使用什么原语、适合什么任务（来源：claude.com 官方博客 2026-06-30，本报告已逐字核实）：

| 类型 | 触发 | 停止条件 | 使用原语 | 最适合的任务 | 成本管理手段 |
|---|---|---|---|---|---|
| **① Turn-based 回合制** | 用户发一条 prompt | Claude 自己判断已完成，或需要更多上下文 | 普通对话 + SKILL.md 强化自验 | 不属于例行流程的短任务 | 把人工验证步骤写进 skill，减少回合数 |
| **② Goal-based 目标循环** | 实时手动 prompt 启动 | **目标达成 或 达到最大轮次**；每次 Claude 想停，由一个**独立评估模型**检查条件 | `/goal` | 有可验证退出标准的任务 | 确定性完成标准 + 显式轮次上限（"stop after 5 tries"） |
| **③ Time-based 定时循环** | 指定时间间隔 | 人取消，或工作自然完结（PR 合并、队列清空） | `/loop`（本机，关机即停）；`/schedule`（上云常驻） | 周期性工作；对接外部系统（盯 PR/CI、每日摘要） | 拉长间隔；尽量事件驱动而非时间驱动 |
| **④ Proactive 主动循环** | 事件或计划，全程无人实时参与 | 每个子任务达标即退出；例程本身一直运行直到被关闭 | `/schedule` + `/goal` + Dynamic Workflows + Auto mode 组合 | 源源不断的定义清晰工作：bug 报告、issue 分诊、迁移、依赖升级 | 例行步骤路由给小/快模型，关键判断才用最强模型 |

官方 goal 循环原例：

```text
/goal get the homepage Lighthouse score to 90 or above, stop after 5 tries.
```

官方 proactive 组合原例：

```text
/schedule every hour: check #project-feedback for bug reports.
/goal: don't stop until every report found this run is triaged, actioned, and responded to.
When fixing a bug, use a workflow to explore three solutions in parallel
worktrees and have a judge adversarially review them.
```

> **选型原则（官方明确提醒）**：不是所有任务都需要复杂循环，**从最简单的方案开始，选择性使用这些模式**。四类循环不是互斥功能，而是自治度阶梯（邦比快跑在知乎的总结："区别只在于谁来发起下一轮"）。

### 3.2 Addy Osmani 的 5+1 工程组件

一个可运转的 loop 需要五个组件加一个记忆位置。下表的工具映射直接来自 Osmani 原博：

| 组件 | 在循环里的职责 | Codex app | Claude Code |
|---|---|---|---|
| **Automations 自动化调度** | 按计划自动发现任务、初步分诊（循环的"心跳"） | Automations 面板：选项目/prompt/频率/环境，结果进 Triage 收件箱，无发现自动归档 | Scheduled tasks、cron、`/loop`、hooks（生命周期触发 shell）、推到 GitHub Actions |
| **Worktrees 隔离工作树** | 并行 Agent 各占独立工作目录+分支，文件层面不可能互相踩 | 内置 worktree 支持 | `git worktree`、`--worktree` 标志、子 Agent 配 `isolation: worktree`（用完自动清理） |
| **Skills 技能** | 把项目知识（约定、构建步骤、历史教训）写下来，避免 Agent 靠猜 | Agent Skills（SKILL.md），`$name` 调用或按描述隐式触发 | Agent Skills（SKILL.md） |
| **Plugins & connectors 插件连接器** | 把 Agent 接入既有工具链（能在真实环境里动手而非只动嘴） | Connectors（MCP）+ 分发插件 | MCP servers + plugins |
| **Sub-agents 子代理** | **提议者与检查者分离**：一个出主意，另一个（常换不同指令甚至不同模型）检查 | `.codex/agents/` 下用 TOML 定义，按需并行派生后汇总 | `.claude/agents/` 子代理 + agent teams |
| **State 状态（第 6 件）** | 活在单次对话之外的记忆：做过什么、下一步什么 | Markdown 或经 connector 连 Linear | Markdown（AGENTS.md、progress 文件）或 MCP 连 Linear |

几个来自原文的工程细节：

- **自动化是循环成为循环的原因**（"the heartbeat"）。OpenAI 内部用它做每日 issue 分诊、CI 失败摘要、commit 简报、追查上周引入的 bug；automation 可以调用 skill，避免把一大墙指令贴进没人会更新的定时器。
- **Skill 是创作格式，plugin 是分发方式**；skill 的描述（description）写得"无聊而精确"比写得花哨更有效，因为隐式触发靠它匹配。
- **子代理是循环里最有价值的结构性设计**：写代码的模型给自己打分太宽容（"way too nice grading its own homework"）；loop 在没人盯着时运行，**一个你真正信任的验证者，是你敢于走开的唯一理由**。代价是子代理各自消耗 token，只在第二意见值钱的地方用。
- `/goal` 底层就是 maker/checker 分离应用在停止条件上：由一个新的模型实例判定是否完成。
- **Worktree 只解决机械冲突，不解决编排税（orchestration tax）**：你的 review 带宽才是并行度的真实天花板。

### 3.3 吴恩达三层嵌套循环

吴恩达 2026-06-26《The Batch》提出，用 AI 做 0→1 产品本质是三个不同速度、彼此嵌套驱动的反馈环（本报告已抓取原文核实）：

| 循环 | 周期 | 谁在转 | 人的角色变化 |
|---|---|---|---|
| **① Agentic coding loop 智能体编码循环** | 分钟级 | Agent：按规格写码→测试→修 bug→再测，直到无 bug 且满足规格 | 可以长时间不介入（吴恩达的 Agent 曾连续自主工作约 1 小时，用浏览器多次自检） |
| **② Developer feedback loop 开发者反馈循环** | 几十分钟～小时级 | 开发者审查当前产品，引导改进 | 从"给 Agent 当 QA 手动找 bug"解放出来，转向高层产品决策（功能取舍、UI、用户流） |
| **③ External feedback loop 外部反馈循环** | 小时～天、周级 | 朋友反馈、alpha 测试、A/B 测试 | 数据反哺产品愿景 → 规格 → Agent；工程师部分承担产品经理角色 |

三层是同心齿轮：最内层转得最快，外层数据逐层向内驱动。**Agent 越能干，开发者反而越忙——但忙的是"决定让 AI 写什么、写得对不对"。**

### 3.4 验证者模式与五级验证阶梯

「谁来喊停」是循环工程方法论最密集的部分（北方的郎《Loop Engineering 第 4 章》等）。按目标的可形式化程度选择验证模式：

| 任务特征 | 推荐验证模式 | 典型场景 |
|---|---|---|
| 目标可写成确定性检查代码 | **确定性验证** | 测试通过数、格式合规、数值阈值 |
| 目标依赖主观判断、难以形式化 | **独立评判者**（fresh context 的模型） | 文案是否得体、方案是否合理、报告论证是否充分 |
| 兼具硬性约束与质量维度 | **混合验证** | 报告既要格式合规又要内容准确 |
| 依赖特定领域规则 | 确定性验证 + **领域规则引擎/知识库** | 工程规范、行业标准核查；规则越结构化越不要依赖模型"常识判断" |

验证者设计的第一原则：**验证逻辑运行时看不到执行者的推理过程**，并尽量拿到执行者接触不到的独立证据。实战案例（千问云《从日志扫描到预发部署的全自主闭环》）：单测能放过 `logger.error → logger.warning` 的假修复，但第 3 层独立诊断复查 Langfuse Trace 发现 ERROR observation 仍在——**修复 Agent 能骗自己，骗不了独立验证 Agent**。

arXiv:2607.00038 进一步提出 **five-level verification ladder（五级验证阶梯）**，并在 50 个真实 loop 语料中观察到 70% 的循环在阶梯的"自主区"完成验证（具体五级划分本报告未取得论文全文，**引用前请查原文核对**）。论文同时指出设计原则植根于自我纠正（self-correction）、奖励黑客（reward hacking）与 model-as-judge 脆弱性的科学文献。

### 3.5 方法演进谱系（2025-07 → 2026-07）

| 时间 | 事件 | 性质 |
|---|---|---|
| 2025-07-14 | Geoffrey Huntley《Ralph Wiggum as a "Software Engineer"》 | 社区原型：bash 驱动循环（ghuntley.com/ralph） |
| 2026-01-17 | Huntley《Everything Is a Ralph Loop》 | 循环理念成型（ghuntley.com/loop，已核实可访问） |
| 2026-02 | Harness Engineering 概念在中文社区流行 | 底座层成熟 |
| 2026-05 | Codex 发布 `/goal`，可跨轮跑到条件成立 | 平台原语化开始（时间为知乎社区转述） |
| 2026-06-07 | Peter Steinberger 发声 + Addy Osmani《Loop Engineering》定名 | 概念引爆（原文已核实） |
| 2026-06-10 / 06-12 / 06-22 | The New Stack 跟进报道；Ponytail 上线；Osmani 在 O'Reilly Radar 发修订扩展版 | 媒体扩散（**知乎转述，未独立核实**） |
| 2026-06-26 | 吴恩达《The Batch》三层循环框架 | 权威背书（原文已核实） |
| 2026-06-28 | arXiv:2607.00038 提交（Sandeco Macedo） | 首个系统性学术形式化（摘要已核实） |
| 2026-06-30 | Anthropic《Getting Started with Loops》四类循环 | 平台级官方框架（原文已核实） |
| 2026-07-15 / 07-16 | Osmani《Own the Outer Loop》；Boris Cherny 放出五级 AI 采用成熟度模型 | 讨论推进到"人的责任边界"（**知乎转述，未独立核实**） |
| 2026-08 | Osmani《Practical Loop Engineering》回顾早期 bash/cron 拼装到产品化 | 实践复盘（**知乎转述，未独立核实**） |

---

## 4. 核心实现：循环契约与 builder/checker 模式

### 4.1 循环契约模板

综合知乎多篇实战文章（沐风、北方的郎、陶刚），一个可直接复用的**循环契约（loop contract）**七项一项不能少。反例是一句"登录坏了，帮我修一下"——那是人肉循环；工程化版本如下：

```yaml
# loop-contract.yaml
goal: |
  tests/auth/ 下 12 个用例全部通过，且 npm run lint 零警告
scope:
  allow: [lib/auth, tests/auth]
  forbid:
    - 不得修改 tests/ 目录          # 防止"改测试让它变绿"
    - 不得改动公开 API 签名
verifier:
  deterministic: [npm test -- auth, npm run lint]   # 在 agent 之外运行
  independent_judge: reviewer-subagent   # 全新上下文，看不到执行者推理
state: progress.md            # 跨轮持久：试过什么、结果、下一步
stop_rules:
  success: 上述检查全部通过
  budget: 最多 5 轮；单轮 >15 分钟判超时
  repeat_failure: 同一测试连续 2 轮以同样方式失败 → 停止升级
  blocked: 需要改迁移文件 / 新增依赖 / 缺权限 → 停止升级
escalation: 停止时把两轮 diff、日志、Trace 链接整理成报告交给人
```

陶刚《AI 编程从写提示词到造循环》强调的细节：测试和 lint 要**在 agent 之外运行**、结果写回状态文件；还要有一个不参与写代码的验证步骤检查约束是否被绕过（比如测试文件有没有被动过）。

### 4.2 builder/checker 伪代码

JEECG 低代码平台《三个文件让 Claude Code 循环验证代码直到全绿》给出的最小可运行模式（builder 与 checker 两个子代理循环，已按本报告语境改写并加注释）：

```python
# loop.py —— builder/checker 循环的最小实现（Python 风格伪代码）
def run_loop(task: str, contract: LoopContract, max_turns: int = 5):
    state = State.load("progress.md")          # 记忆在磁盘，不在上下文

    for turn in range(1, max_turns + 1):
        # 1) builder：实现任务，或修复上一轮失败（给它失败报告，不帮它解读）
        patch = builder_subagent.act(
            task=task,
            history=state.failed_attempts,
            worktree=f"wt-{turn}",             # 每轮隔离工作树
        )

        # 2) checker：独立子代理运行所有检查；与 builder 上下文隔离
        verdict = checker_subagent.verify(
            patch=patch,
            goal=contract.goal,
            checks=contract.verifier.deterministic,   # 测试/lint 在 agent 外跑
            guardrails=contract.scope,                # tests/ 没被改过？
        )

        state.record(turn, patch, verdict)
        state.save("progress.md")               # 每轮落盘，可断点续跑

        # 3) 判停：成功 / 预算 / 重复失败 / 硬阻塞，终态必须显式
        if verdict.status == "ALL_GREEN":
            return show_diff_and_report(patch, verdict)       # SUCCESS
        if verdict.repeat_failure(prev=state.last_verdict):
            return escalate_to_human(verdict, state)          # REPEAT_FAILURE
        if verdict.blocked_by_contract:
            return escalate_to_human(verdict, state)          # BLOCKED
        # FAILED：把 checker 的完整失败报告原样转给 builder，不解读不过滤
        state.failed_attempts.append(verdict.full_report)

    return escalate_to_human("BUDGET_EXHAUSTED after 5 turns", state)
```

对应到 Claude Code，这套机制已有平台原语：`/goal` 在每轮后调用独立评估模型；dynamic workflows 在并行 worktree 里探索多个方案，再由一个 judge 子代理对抗式评审。

### 4.3 状态文件与 SKILL.md 示例

State（记忆）的最小形态就是一个 Markdown 文件——"听起来蠢到不像有用，但这是每个长跑 Agent 都依赖的同一招"（Osmani）：

```markdown
<!-- progress.md：外环记忆，模型跨 run 全靠它续上 -->
## Goal
Lighthouse 首页分数 ≥ 90

## Done
- [x] turn1: 图片懒加载，分数 78 → 84（证据：lighthouse-2026-09-23-a.html）
- [x] turn2: 移除未用 JS，84 → 88

## Open / Blocked
- turn3: LCP 仍 2.3s，疑似 hero 图 CDN 配置问题，需要运维权限 → 已升级 @人

## Lessons（Reflexion：失败反思写下来，喂给后续每一轮）
- 不要用 CSS 隐藏未用模块骗 Lighthouse，checker 会查 DOM
```

SKILL.md 则把"什么叫高质量"编码成可检查步骤，让 Agent 能端到端自检（Anthropic 官方 turn-based 循环的建议）：

```markdown
---
name: verify-frontend-change
description: Verify any UI change end-to-end before declaring it done.
---
# Verifying frontend changes
Never report a UI change as complete based on a successful edit alone:
1. 启动 dev server，浏览器打开改动页面。
2. 新控件（按钮/输入/开关）必须实际点击，确认状态变化，截图 before/after。
3. 浏览器 console 零新增 error/warning。
```

### 4.4 工程反模式与典型坑

1. **没有刹车的循环（no stop rules）**：模糊目标（"做得更好一点"）会让 Agent 永远觉得没做完，或过早自我宣布完成。退出标准必须确定性（测试数、分数阈值），并配轮次/预算上限。
2. **执行者即打分者（self-grading）**：写代码的模型判自己完成，等于没有验证。maker/checker 必须分离，且 checker 用全新上下文、最好能接触执行者看不到的独立证据。
3. **对同一错误用同样方法重试（空转而非学习）**：这不是学习是烧钱。可恢复错误（语法、缺 import）与硬性阻塞（缺权限、未定义行为）要分流；重复失败要能升级。Reflexion 式反思必须落盘进记忆，否则每轮从零再犯。
4. **奖励黑客 / 自我欺骗（reward hacking）**：改测试让它变绿、用 CSS 隐藏元素骗过截图检查。靠 scope 约束（禁止改 tests/）+ 独立证据（Trace、真实浏览器交互）兜住；这也是 arXiv 论文引用 reward hacking 文献的原因。
5. **Worktree 幻觉并行**：机械隔离解决了文件冲突，但每个 PR 仍需 review，人的审查带宽才是并行度天花板（Osmani 称之 orchestration tax）。
6. **Token 失控**：Osmani 原博开头即警告 token 成本差异巨大；proactive 循环要用小模型跑例行步骤、最强模型只做关键判断；定时器尽量改事件驱动。
7. **理解债（comprehension debt）**：循环越快交付你没亲手写的代码，"存在的东西"与"你真正理解的东西"之间的裂缝越大，且理解债增长得更快。
8. **认知缴械（cognitive surrender）**：循环自运转后最舒服也最危险的姿势是放弃判断、来什么收什么。"同样一个 loop，两个人用出相反结果：一个在自己深刻理解的工作上加速，另一个用它来逃避理解工作。循环分不清区别，你能。"（Osmani）
9. **可恢复性缺口**：进程退出前的工具调用处于未开始/执行中/副作用已完成但结果未落盘哪种状态？目前多数会话层没有 durable run cursor，恢复未完成的 write/bash/API 调用需要工具幂等、执行 receipt、验收与补偿（知乎《从 Pi 窥见》高赞评论提出的工程边界）。

---

## 5. 结论与选型建议

### 5.1 核心结论

1. **Loop Engineering 的杠杆点从"写好一句 prompt"移到了"设计触发、验证、停止与权限组成的可靠系统"**。它不是让 Agent 一直跑，而是把刹车和验收做对。
2. **内环（Agent Loop）与外环（Loop Engineering）是两层问题**：Harness 让单轮跑得通，Loop 让跨轮/跨会话的验收与记忆跑得通；Loop 建立在 Harness 之上，不取代 prompt 与 harness。
3. **可验证的目标 + 独立验证者 + 显式终态，是敢于无人值守的三个前提**；`/goal` 底层就是 maker/checker 分离。
4. **从 turn-based 起步，按自治度阶梯升级**：短任务用回合制，可验证任务上 `/goal`，周期/对接外部系统用 `/loop`、`/schedule`，定义清晰的流工作才组合 proactive；不是越自动越好。
5. **循环放大工程判断，不替代它**：保持 codebase 整洁、持续读循环产出的代码、守住 review 与升级路径，否则平滑的循环只会更快累积理解债。

### 5.2 落地 checklist

- ☐ 目标写成确定性、可机检的完成标准（测试数、阈值、格式），不用模糊形容词
- ☐ 写明四类终态：成功、预算耗尽、重复失败升级、硬阻塞升级
- ☐ 设置轮次/时间/Token 三重预算；单轮超时阈值
- ☐ 验证者独立于执行者：全新上下文的 checker 子代理或 agent 外脚本
- ☐ 验证证据独立且多维：测试 + lint + 真实浏览器/截图 + 监控 Trace
- ☐ scope 护栏：禁止改测试/公开 API；每轮检查护栏是否被绕过
- ☐ 状态外置到 progress.md/看板，每轮落盘，支持断点续跑
- ☐ 并行 Agent 一律 worktree 隔离；估算自己的 review 带宽再定并行度
- ☐ 重复性知识沉淀进 SKILL.md；失败反思（Reflexion）写入持久记忆
- ☐ 模型分级：例行步骤小模型，判断与评审强模型
- ☐ 定时器优先改为事件驱动；本机 `/loop` 与云端 `/schedule` 按关机容忍度选择
- ☐ 保留人工升级通道：重复失败、要加依赖/改迁移/缺权限时停下报告
- ☐ 定期阅读循环产出的代码，主动偿还理解债；上线前先在自己深度理解的仓库试点

### 5.3 注意事项与诚实性声明

- 本报告的**一手事实**（四类循环定义与示例、5+1 组件与工具映射、三层循环、Ralph 文章、三篇论文编号与标题）均通过抓取原始网页/arXiv 核实。
- 标注"**知乎转述，未独立核实**"的时间线条目（The New Stack/Ponytail/O'Reilly 日期、Codex `/goal` 5 月上线、Osmani 7-8 月后续文章标题、Boris 五级成熟度模型）来自知乎文章引用，引用前请二次核对。
- arXiv:2607.00038 的语料统计数字（70%/74%）与"五级验证阶梯"细节来自论文摘要，未获取全文逐字核对。
- 工具原语（`/goal`、`/loop`、`/schedule`、dynamic workflows、auto mode）迭代很快，具体语法以 Claude Code / Codex 官方文档当前版本为准。

---

## 6. 参考资料

### 原始出处（官方博客/一手来源）

1. Addy Osmani. 《Loop Engineering》. 2026-06-07. https://addyosmani.com/blog/loop-engineering/
2. Anthropic / Claude Code 团队（Delba de Oliveira, Michael Segner）. 《Loop engineering: Getting started with loops》. 2026-06-30. https://claude.com/blog/getting-started-with-loops
3. Andrew Ng. 《Three Key Loops for Building Great Software》. The Batch, DeepLearning.AI, 2026-06-26. https://www.deeplearning.ai/the-batch/three-key-loops-for-building-great-software/
4. Geoffrey Huntley. 《Everything Is a Ralph Loop》. 2026-01-17. https://ghuntley.com/loop/
5. Geoffrey Huntley. 《Ralph Wiggum as a "Software Engineer"》. 2025-07-14. https://ghuntley.com/ralph/

### 研究论文

6. Sandeco Macedo. 《Stop Hand-Holding Your Coding Agent: Engineering the Loops that Replace Step-by-Step Prompting》. arXiv:2607.00038, 2026-06-28. https://arxiv.org/abs/2607.00038
7. Shunyu Yao 等. 《ReAct: Synergizing Reasoning and Acting in Language Models》. arXiv:2210.03629, 2022. https://arxiv.org/abs/2210.03629
8. Noah Shinn 等. 《Reflexion: Language Agents with Verbal Reinforcement Learning》. arXiv:2303.11366, 2023. https://arxiv.org/abs/2303.11366

### 知乎文章（社区实践与中文阐释）

9. 代码里程碑.《从 Pi 窥见 Loop Engineering 的递归美学》. https://zhuanlan.zhihu.com/p/2061271933296154175
10. PaperAgent.《提示词工程已死，Loop Engineering 来了！》. https://zhuanlan.zhihu.com/p/2060404811510571249
11. 程序员鱼皮.《提示词工程已死，Loop Engineering 称王！保姆级教程 + 项目实战》. https://zhuanlan.zhihu.com/p/2050246303012148996
12. 溯流而上.《Blog 速读：Loop Engineering》（Addy Osmani 译文）. https://zhuanlan.zhihu.com/p/2047832303997563703
13. 搬砖的小明.《Loop Engineering，炒概念还是真革命？》. https://zhuanlan.zhihu.com/p/2054994809631290487
14. Simon Wong.《理解 Loop Engineering：写代码从手动挡到全自动》. https://zhuanlan.zhihu.com/p/2058650346986141248
15. 锦恢.《锦恢的 AI Agent 小白教程（五）Loop Engineering 与 goal》. https://zhuanlan.zhihu.com/p/2064738034575226689
16. 锦恢.《小白教程（四）智能体基本原理：Agent Loop，规划模式与子智能体》. https://zhuanlan.zhihu.com/p/2057385707082093544
17. 袋鼠云数栈 UED.《Agent Loop：让 Agent 真正跑起来的核心机制》. https://zhuanlan.zhihu.com/p/2061058101885850658
18. 晓码的创造栈.《Loop engineering：四类 loop 的触发、停止与适用场景》. https://zhuanlan.zhihu.com/p/2061029890938364857
19. lufeikevin.《Harness 还没弄明白 Loop Engineering 又来了》. https://zhuanlan.zhihu.com/p/2051245811766391870
20. 技术极简主义.《打造 Claude Code 可持续推进的工作流：Loop Engineering 完整上手攻略》. https://zhuanlan.zhihu.com/p/2058220227637392413
21. 沐风.《Loop Engineering 实战：把 /goal 和 /loop 变成可验证的 AI 工程循环》. https://zhuanlan.zhihu.com/p/2057427259162727944
22. 北方的郎.《Loop Engineering，第 4 章：谁来喊停，终止条件与验证者模式》. https://zhuanlan.zhihu.com/p/2064632428308721863
23. 北方的郎.《Loop Engineering：大模型智能体的自主运行之道》（全书目录）. https://zhuanlan.zhihu.com/p/2063947506501809711
24. 千问云.《Loop Engineering 实战：实现从日志扫描到预发部署的全自主闭环》. https://zhuanlan.zhihu.com/p/2060371303559868657
25. JEECG 低代码平台.《爆火的 Loop Engineering，三个文件让 Claude Code 循环验证代码直到全绿》. https://zhuanlan.zhihu.com/p/2057405303231189116
26. 陶刚.《AI 编程从写提示词到造循环：Loop Engineering 工程化了什么》. https://zhuanlan.zhihu.com/p/2083837763582011161
27. 神经 CAE.《从 Harness 工程到 Loop 工程？——把人从内环引擎位，移到外环方向盘》. https://zhuanlan.zhihu.com/p/2064796369110541793
28. 非著名程序员.《吴恩达最新长文：AI 时代，真正重要的不是写代码，而是设计循环》. https://zhuanlan.zhihu.com/p/2056347456036664299
29. 夜晚星空下的你.《吴恩达说人的核心价值不是「品味」——「循环工程」三重回路到底在说什么》. https://zhuanlan.zhihu.com/p/2054334162258094078
30. 七牛云行业应用.《Loop Engineering 实践教程：从写提示词到设计循环》. https://zhuanlan.zhihu.com/p/2056419614930876230
31. 玉鸯.《Loop Engineering 会成为 AI Agent 时代的新基础能力吗？》（时间线梳理）. https://www.zhihu.com/question/2050706169056977955/answer/2062138878719432164
32. 邦比快跑.《如何看待由 OpenClaw 作者引发的 "Loop 工程" 讨论？》（四类循环实例）. https://www.zhihu.com/question/2048003050531558553/answer/2056061192188204497
33. CRUD.《Loop Engineering，循环工程是什么？》（概念借了哪些老概念）. https://www.zhihu.com/question/2049888183597512610/answer/2057204243904345250
34. gwave.《Loop Engineering（循环工程）》. https://zhuanlan.zhihu.com/p/2051768479712384475
35. 人工智能研究所.《Loop Engineering：不要再写 prompt 了？而是设计一套 Loop 系统！》. https://zhuanlan.zhihu.com/p/2052002838067393271
36. JavaGuide.《万字详解 AI Agent 核心概念：Agent Loop、Plan-and-Execute、A2A、Agentic Workflows、Tools 注册》. https://zhuanlan.zhihu.com/p/2044328943587743048
37. 思林广不记.《超级 Loop：让 AI 干长活的完整架构》. https://zhuanlan.zhihu.com/p/2063723265848243487
38. Arron.《如何构建一个超越 99% 人的 Claude Loop Engineering（循环工程）》. https://zhuanlan.zhihu.com/p/2050553784628146887
