# Claude Tag 技术调研报告

> **调研主题**：Claude Tag —— Anthropic 推出的团队协作型 Agent（"AI 同事"）
> **调研来源**：知乎开放平台（20+ 篇文章/回答）+ Anthropic 官方公告页（已核实）
> **报告日期**：2026-08-18
> **配套材料**：架构图 3 张（`../assets/claude-tag-diagrams/` 目录，浏览器直接打开）

---

## 一、背景：从"工具"到"同事"的范式迁移

### 1.1 Claude Tag 是什么

Claude Tag 是 Anthropic 于 **2026 年 6 月 23 日** 发布的团队协作产品。官方定位（原文，已在 anthropic.com/news/introducing-claude-tag 核实）：

> "Claude Tag is a new way for teams to work with Claude. We're starting on Slack, which Claude can join as a team member. Grant Claude access to selected channels, and connect it to whichever tools, data—and even codebases—you choose. Then, anyone in the channel can tag @Claude in, and delegate tasks to it."
>
> "We see Claude Tag as the beginning of an evolution of Claude Code: it makes the model even more proactive, and it works better with a full team."

一句话概括：**Claude 不再是被你打开的工具，而是被拉进 Slack 工作群、拥有组织上下文、能被 @ 派活、会主动跟进的"数字同事"**。

### 1.2 关键事实速览（均来自官方公告，已核实）

| 维度 | 事实 |
|---|---|
| 发布时间 | 2026-06-23 |
| 首发平台 | Slack（后续计划扩展更多工作场景） |
| 可用范围 | Claude Enterprise 和 Team 客户（beta） |
| 底层模型 | Opus 4.8（固定，暂不可切换） |
| 内部采用 | Anthropic 产品团队 **65% 的代码**由内部版 Claude Tag 生成 |
| 替代关系 | 取代现有 Claude in Slack 应用，管理员 30 天内可迁移 |
| 计费 | 管理员可设组织级 + 频道级 token 消费上限；符合条件者发放启动额度 |

### 1.3 Karpathy 的"第三次范式"论断

刚加入 Anthropic 的 Andrej Karpathy 第一时间点评，认为 Claude Tag 代表 **LLM 用户界面的第三次重大重构**：

| 范式 | 形态 | 代表 |
|---|---|---|
| 第一代 | LLM 是一个**网站**，你去访问它 | ChatGPT |
| 第二代 | LLM 是下载到电脑的**应用**，你进入它的环境 | Claude Code、Codex |
| 第三代 | LLM 是**独立、持久、异步的实体**，拥有组织范围内的工具和上下文，与团队协同 | **Claude Tag** |

这个论断是理解 Claude Tag 的钥匙：**前两代都是"人去找 AI"，第三代是"AI 进驻人的协作场"**。前两代解决个人执行链路，第三代解决组织协作链路。

### 1.4 与既有形态的本质区别

很多评论第一反应是"不就是把 Claude 接进 Slack 吗"。关键区别在于它从 **chatbot 变成了 agent**：

| 维度 | 旧版 Slack Claude | Claude Tag |
|---|---|---|
| 定位 | 聊天助手（chatbot） | AI 团队成员（agent） |
| 使用方式 | 一对一私聊 | 频道里 @ 召唤，多人共享 |
| 上下文 | 单轮对话 | 完整频道历史 + 组织记忆 |
| 工具能力 | 基本无 | GitHub / Jira / 代码库 / 数据源 |
| 任务持续 | 几分钟 | 数小时甚至数天 |
| 主动性 | 被动响应 | 可主动介入提醒（ambient 模式） |

更精准的类比（知乎作者"多元宇宙喵"）：**它是一个被正式拉进项目群、但只拿到指定门禁卡的 AI 同事**——不是全公司的上帝视角，而是在特定频道、特定权限、特定上下文里工作的参与者。

---

## 二、原理：四大核心能力与身份模型

Claude Tag 的差异化能力可归纳为四个，官方公告逐一阐述，知乎多篇深度解析印证：

### 2.1 Multiplayer（多人共享）—— 一个频道一个 Claude

> "Within a given Slack channel, there's one Claude that interacts with everyone. Anyone can see what it's working on, and can pick up the conversation from where the last person left off."

同一个 Slack 频道里，**只有一个 Claude 在服务所有人**。这带来两个根本变化：

- **任务可接力**：前一个人布置的任务、积累的输出，同事可以直接从中断处继续推进，不需要重复解释上下文。
- **群聊即 Prompt**：交互通过在频道中 @Claude 触发，无需切换窗口，频道里已经发生的讨论本身就是上下文。

这是它区别于"每个人私聊一个 Claude"的核心——Claude 成了**公共的、持续进化的团队知识库**，而非私人对话框。

### 2.2 Learns over time（持续学习）—— 频道级持久记忆

> "As Claude follows along with its channel, it builds more context about the work. And Claude can even automatically learn from other Slack channels and data sources, if it's granted permission. (It doesn't report from private channels.)"

Claude 长期驻留频道，自动积累对团队工作的理解：业务词汇、技术决策、在推进的项目、协作习惯。**时间越长，对团队隐性知识掌握越深**。关键约束：

- 记忆**按频道/身份严格隔离**：工程频道的 Claude 不知道销售频道在聊什么，私密频道记忆严格限定在该频道内。
- 不会从私密频道汇报（隐私硬边界）。
- 管理员可在组织设置中查看、编辑、删除频道/工作区记忆。

### 2.3 Takes initiative（主动行动）—— Ambient 模式

> "If 'ambient' behavior is enabled, Claude will proactively keep you updated about whatever it thinks you might need to know. It'll flag relevant information from across the channels it's in and the tools it's connected to, and follow up on threads or tasks that have gone quiet without being resolved."

开启 ambient 行为后，Claude 会：
- 默默关注频道对话，主动发现需要跟进的线程和遗忘的任务；
- 主动推送它认为你可能需要了解的信息；
- 跟进长期无人响应、未解决的讨论串——像一个靠谱的项目经理。

这是从"你问它才答"到"它自己判断介入点"的跃迁，但也对 AI 治理提出新挑战（见第五章风险）。

### 2.4 Works asynchronously（异步执行）—— 长程任务

> "Set Claude a task, and you can focus on your other priorities while it works. It can also schedule tasks for itself, pursuing a project autonomously over hours or days."

把任务 @Claude 后你可以去做别的，它把任务拆成多阶段，调用已授权工具逐步完成，在 Slack 线程异步汇报结果，**支持数小时甚至数天自主推进**。Anthropic 内部观察：大家花在"委派任务给多个 Claude"上的时间，已远多于亲手执行。

### 2.5 Agent Identity：安全层面最重要的创新

这是 Anthropic 在官方博客专门阐述的设计理念，也是知乎技术评论者公认的 Claude Tag 最关键设计决策。

**传统做法的矛盾**：AI 以用户身份执行、借用某个人的凭证干活。但在多人协作场景里这有根本性矛盾——三个工程师加一个 PM 共享一个频道，任务该以谁的名义执行？选谁都不对。

**Agent Identity 的解法**：让 Claude **以自己的身份行动**：

- 在 Slack 里以 **Claude App** 身份发帖；
- 在 GitHub 里以 **Claude GitHub App** 身份创建 PR；
- 在数据仓库里以**管理员配置的服务账号**身份查询。

Claude 拥有自己的公司账户，**不是"用你的 token"**。这带来三个好处：

1. **责任边界清晰**：每一次操作都能追溯到"是 Claude 干的"，而非模糊地挂在某个人头上。
2. **审计可还原**：消息来源、可用工具、审批人、执行回执可落进同一条 trace，出事时能还原它依据了什么。
3. **权限不串扰**：不会因借用了某个高权限用户的凭证而越权。

---

## 三、技术方案：运行时架构拆解

Claude Tag 最难的部分不是"能聊天"，而是它背后的**运行时系统**。知乎作者 xingzhe 的判断很到位：*"这已经不是聊天插件的词汇表，而是运行时系统的词汇表"*——session、sandbox、thread、channel access、Agent Proxy。

### 3.1 Channel Session：三个系统的协作

当 Claude 处理频道任务时，涉及三个系统（源自官方文档 How Claude Tag works，知乎林夕一文详细转述）：

1. **请求发生在 Slack 工作区**：用户 @Claude 执行操作，或计划任务开始；
2. **Claude 的工作在 Anthropic 基础设施的沙盒环境运行**：不会在你的网络里安装任何东西；
3. **代理凭据用于连接外部系统**：GitHub、数据仓库等。

**完整执行流程**：

| 步骤 | 动作 | 说明 |
|---|---|---|
| ① | 频道中 @Claude | 请求启动一个新会话（session） |
| ② | 会话沙箱启动 | Claude 在隔离环境执行：读文件、写文档、跑代码。**凭据不放入沙箱**——它们保留在凭据存储（credential store）中 |
| ③ | 请求经过 Agent Proxy | 当需要访问沙箱外资源（调 GitHub API、查数仓），请求经过 **Agent Proxy**——它是沙箱与其他所有网络环境之间的**边界** |
| ④ | Proxy 附加凭据 | 从凭据存储库匹配对应凭据（管理员的连接信息存在这里），注入请求 |
| ⑤ | 返回结果 | 经过身份验证的请求到达目标系统（GitHub/数仓），结果返回线程 |

这个设计的精髓：**凭据隔离 + 边界代理**。沙箱里跑的是 Claude 的代码执行，但凭据从不进入沙箱，只在出站的 Agent Proxy 处注入——即使沙箱被提示注入攻破，攻击者也拿不到凭据。

### 3.2 DM 的差异化处理

与 Claude 的私信（DM）工作方式与频道不同：

- 私信**没有身份绑定机制**，使用**你自己的 claude.ai 账户**运行；
- 就像在网页上使用 Claude Code 会话，用你自己的连接器和凭据，结果归于你名下；
- 仅限一对一交流，不支持群组私信。

即：**频道 = 组织身份（服务账号），DM = 个人身份（个人账户）**。官方建议敏感信息走 DM。

### 3.3 分层隔离的持久化记忆

Claude Tag 的记忆与上下文是"分层、隔离的持久化记忆"，按层级继承：

```
组织（Org）
  └── 工作区（Workspace）
        └── 私有频道（Private Channel）
```

继承规则：**向下继承，不向上**。频道内的工作记忆绑定到组织的 Claude 身份；DM 记忆绑定到个人账户。不同频道的 Claude 实例相互独立，记忆、权限、数据完全隔离。

管理员可创建**独立 Claude 身份（scoped identity）**做精细控制：频道访问、工具权限、数据范围、token 消费上限（组织级 + 频道级），并配有行动日志和审计追踪。官方示例：*销售场景的 Claude 不会把记忆传给工程场景的 Claude，工程师也拿不到任何销售数据或工具*。

### 3.4 Managed Agents：Session / Harness / Sandbox 三接口

Anthropic 的 Managed Agents 提供了更底层的架构蓝图，把 Agent 虚拟化成三个稳定接口（灵感来自操作系统虚拟化硬件的思路，知乎 riba2534、DeepToken 两文解析）：

| 接口 | 职责 | 方法 |
|---|---|---|
| **Session** | 追加式事件日志，记录所有发生的事 | `getSession(id)`、`emitEvent(id, event)` |
| **Harness** | 调用 Claude 并路由工具请求的循环（"大脑"） | `wake(sessionId)` |
| **Sandbox** | 代码执行和文件编辑环境（"双手"） | `provision({resources})`、`execute(name, input) → string` |

**核心解耦思想**：把 brain 从 container 里拿出来，把 hands 变成按需调用的工具接口，把 session 单独抽成可持久化的日志对象。收益：

- container 死了 ≠ session 死了；
- harness 挂了可重启后从 session log 继续接上；
- Claude 只在需要时调用 sandbox，避免为每个会话预付容器启动成本；
- 工具、MCP、外部系统、VPC 资源都能通过统一接口接入；
- 凭据不必暴露在代码执行沙箱里，安全边界更清晰。

**工程指标**：采用该架构后，p50 的 time-to-first-token 下降约 **60%**，p95 超过 **90%**。

**Managed Agents 公布的平台能力**：secure sandboxing（安全沙箱）、authentication & tool execution（认证与工具执行托管）、long-running sessions（数小时长会话）、progress persistence（断线后进度保留）、trusted governance（权限/身份/执行追踪治理）、multi-agent coordination（多 Agent 协同，research preview）、session tracing / analytics（每个工具调用、决策、失败模式可观测）。

这说明 Anthropic 不只是做"云上的 Claude Code"，而是把**多 Agent 编排 + 会话级可观测性 + 组织级治理**一起做成平台能力。

### 3.5 一个更深的技术命题：状态 vs 记忆

知乎作者 xingzhe 提出了一个值得重视的洞察：

> "Claude Tag 最难解决的可能不是'记忆'，而是'状态'。记忆回答的是：过去发生过什么？状态回答的是：现在到底以哪一个版本为准？"

在企业协作里，真正危险的往往不是找不到旧资料，而是**找到了旧资料，却把已经被替代的方案当成当前事实**——旧 owner、旧流程、旧 PR、旧 workaround、旧架构讨论，都可能在语义上"很相关"，但在状态上已经失效。

**这正是 RAG 的边界**：它很擅长找到"像不像"（语义相似度），但不天然知道"现在对不对"（状态有效性）。这揭示了企业级 Agent 记忆系统不能只是 "Slack + Embedding + VectorDB + Prompt"，还需要状态管理与事实失效机制。

---

## 四、落地实践：从 Pilot 到规模化

综合知乎多篇实操文章（林夕《万字深度解析》、高国生成式《深度实操》等），落地实践可归纳为部署、安全、质量、知识、成本五个维度。

### 4.1 部署与初始化

官方给出的四步上手流程（已核实）：

1. 将 Claude Tag 与你的 Slack workspace 配对；
2. 给 Claude 访问你的工具的权限；
3. 设置组织月度消费上限；
4. **在私有频道测试 Claude，确认它工作正常**。

一线实践建议：

- **先拿一个私有频道做 Pilot**，验证所有连接正常后再推广；
- 每个服务创建**专用账号**（如 `claude@yourcompany.com`），而不是绑定任何人的个人账号；
- **IP 白名单**要求提前和网络团队沟通——Anthropic 发布了固定的出口 IP 段，很多企业需要走流程授权。

### 4.2 安全与权限：最小权限 + 审计可追溯

核心原则是"让 AI 在安全的边界内发挥最大能力"，而非"限制 AI 的能力"：

- **按频道授权**：每个频道的 Claude 只能访问该频道需要的资源；
- **代码库按仓库授权**：核心仓库需要单独审批；
- **数据库只读**：只能查不能改（或只能改指定的表）；
- **所有操作有日志**：谁让 Claude 干了什么一清二楚；
- **敏感操作需人工确认**（如线上部署）。

推荐的分级策略（来自实操经验）：

| 操作类型 | 策略 |
|---|---|
| 生产环境操作 | 必须人工二次确认 |
| 测试环境操作 | Claude 可自动执行 |
| 代码读取 | 开放 |
| 代码写入 | 需要 review |

跨频道访问在架构层面不可能——不是开关设置，是**硬边界**。权限是**按任务分配的，不是按角色分配的**：给它"读取文档"的任务，它不会获得"删除文档"的权限。

### 4.3 代码质量：人类 review 不可省

实操共识：Claude 写的代码**比很多新人靠谱，但不能完全放心**。最佳实践流水线：

```
AI 写代码 → AI 自测 → 静态检查 → 人类 review → 合并
```

具体做法：
- Claude 写的代码必须经过人类 review；
- 但可让 Claude 先做第一轮 review，筛掉低级问题（review 时间可从 1 小时降到 15 分钟）；
- 强制要求 Claude 写单元测试；
- 接入静态代码检查工具，不通过不让合并。

人类 reviewer 只需关注：业务逻辑对不对、架构设计合不合理、Claude 没覆盖到的深层问题。

### 4.4 知识沉淀：主动喂，而非只靠记忆

Claude Tag 有记忆，但记忆有限。真正的团队知识需要**主动喂给它**：

- 团队的代码规范、设计规范、最佳实践文档；
- 历史的技术方案、架构设计文档；
- 常见问题的解决方案、踩坑记录；
- 定期更新，保持知识新鲜。

效果对比：新人要培训 3 个月才能上手；Claude Tag 把文档喂给它，第二天就能按团队规范干活。

### 4.5 成本控制：Opus 4.8 不便宜

Claude Tag 目前只支持 Opus 4.8，成本是真实痛点（多位知乎用户称其为"账单杀手"）。优化建议：

- **简单任务**（整理讨论、查数据）用 Sonnet；
- **复杂任务**（写代码、架构设计）才用 Opus；
- 配置**路由规则**，自动根据任务复杂度选模型；
- 设置**每日/每月预算上限**，超了就降级；
- 别让 AI 提的效还不够付 API 费用。

### 4.6 典型落地场景（9 大场景覆盖）

| 团队 | 场景 |
|---|---|
| 产品团队 | 写 PRD、拆解用户故事、生成设计稿、直接写前端代码 |
| 工程团队 | 定位复杂 Bug、Review 代码、写测试、追踪技术债务 |
| 支持/运营 | 处理工单、总结客户反馈、生成周报 |
| 销售/增长 | 追踪产品指标、分析数据、准备客户提案 |
| 跨团队协作 | 一个项目涉及多部门时，@Claude 成为信息中转和进度追踪的"超级助手" |
| 通用 | 长讨论快速补课：@Claude 总结"刚决定了什么、还有哪些问题没解决、谁要做什么" |

### 4.7 开源复刻路线：Open Tag

值得关注的是，Zilliz 基于其内部研发近半年的 **MFS（记忆与工具引擎）** 项目，两天手搓了开源版 Claude Tag，名为 **Open Tag**（github.com/zilliztech/...）。它证明了这套模式可以用开源组件复刻，让普通 Claude 用户和 Codex 用户免费使用：在 Slack 频道 @OpenClaude（或 @OpenCodex），它读懂当前线程上下文，结合授权的代码、文档、工单、聊天记录、数据库，输出结果并贴回 Slack。这说明 Claude Tag 的产品形态正在被快速追赶，**核心壁垒不在形态，而在模型能力与组织记忆的积累**。

---

## 五、结论、风险与选型建议

### 5.1 结论

1. **Claude Tag 是范式迁移，不是功能升级**。它把 AI 从"人去找的工具"变成"进驻协作场的同事"，对应 Karpathy 所说的 LLM 交互第三次重构。其战略意义在于抢占企业 AI 的"工作流入口"。

2. **四大能力（multiplayer / learns / initiative / async）+ Agent Identity 是产品骨架**，其中 **Agent Identity（Claude 以自己身份行动）是安全层面最关键的设计决策**，解决了多人协作场景下"以谁名义执行"的根本矛盾。

3. **运行时架构是真正的技术护城河**：Session（事件日志）/ Harness（大脑循环）/ Sandbox（双手执行）三接口解耦 + Agent Proxy 凭据隔离 + 分层隔离记忆，实现了长会话、可恢复、可治理、可观测。凭据不进沙箱、只在出站 Proxy 注入，是企业级安全的核心设计。

4. **"状态 vs 记忆"是更深的未解难题**。RAG 擅长找"像不像"，但不知道"现在对不对"。企业级 Agent 记忆系统需要超越 Embedding+VectorDB，引入状态管理与事实失效机制。

5. **65% 的内部代码占比是最强的背书，也是最需要审慎看待的数字**——它证明模式在 Anthropic 这种顶级工程组织里成立，但能否迁移到流程、工具、文化不同的普通企业，仍需更多第三方验证。

### 5.2 风险与局限（务必正视）

| 风险 | 说明 | 缓解 |
|---|---|---|
| **成本失控** | 长程任务极费 token，被称"账单杀手"；长线程可能比人还贵 | 模型路由 + 预算上限 + 命中缓存机制 |
| **长程任务漂移** | 提供多文档后，AI 后期可能进入 inline hardcode，需再费 token 重构；容易漏 review | 任务拆小 + 强制 review + 阶段性校验 |
| **Ambient 治理挑战** | 主动介入意味着 AI 自己判断介入点，可能变成"数字监工"，员工反感 | 明确 ambient 边界 + 团队共识 |
| **责任边界** | 65% 很醒目，但若消息来源/工具/审批/回执不落同一条 trace，出事无法还原依据 | 全链路审计 trace |
| **平台锁定** | "公司记忆"是核心壁垒——模型可换，Agent 积累的工作流/习惯/上下文换不了，迁移即归零 | 评估记忆可移植性 / 多平台策略 |
| **缺乏第三方 benchmark** | 截至 2026-06-25，没有公开、严肃、可复现的 Claude Tag 第三方评测 | 以自有 Pilot 实测为准 |

### 5.3 选型建议

- **适合**：协作密集、上下文重、有 Slack 工作流、愿意投入治理的工程/产品/支持团队；任务可拆解、有明确验收标准（代码、数据、工单）。
- **谨慎**：涉及强合规/高敏感操作且无法人工兜底的场景；token 预算紧张的小团队；期望"全自动无人值守"替代关键决策的场景。
- **落地路径**：私有频道 Pilot → 专用服务账号 + IP 白名单 → 分级权限（生产人工确认）→ 强制 review 流水线 → 模型路由控成本 → 逐步扩频道。

### 5.4 注意事项（可信度声明）

- 本报告产品定位、四大能力、65%、Opus 4.8、发布时间等**已对照 Anthropic 官方公告页（anthropic.com/news/introducing-claude-tag）逐条核实**。
- 技术架构细节（Channel Session 流程、Agent Proxy、Managed Agents 三接口、p50/p95 性能数字）来自**知乎文章对官方文档的转述**（主要参考 xingzhe、林夕、riba2534、DeepToken 等文），其中 Managed Agents 部分为 Anthropic 平台能力的公开介绍，与 Claude Tag 产品本身的对应关系需以官方文档为准，**引用前建议二次核对**。
- 开源项目 Open Tag / MFS 的 GitHub 地址在知乎原文中被截断（github.com/zilliztech/m...），未补全，使用前需自行检索确认。
- "65% 代码占比"为 Anthropic 自述，无第三方审计。
- Karpathy"第三次范式"、"特洛伊木马（公司记忆壁垒）"等为评论者观点，非官方表述。

---

## 六、参考资料

### 官方出处（已核实）
1. *Introducing Claude Tag*, Anthropic, 2026-06-23 — https://www.anthropic.com/news/introducing-claude-tag （产品定位、Slack 起点、四大能力、65%、scoped identity、Opus 4.8）
2. Claude Tag 官方支持文档（知乎转述篇目）：Claude Tag overview / How Claude Tag works / Good habits / What Claude Tag remembers / Security and data handling / What is Claude Tag?

### 知乎主要参考文章与回答（检索时间 2026-08-18）
3. 程墨Morgan《如何评价 Anthropic 最新发布的 Claude Tag?》（Karpathy 三范式）— https://www.zhihu.com/question/2053047749281690000/answer/2053204648211788801
4. 量子位《刚刚，Claude Code 大升级！卡帕西：LLM 第三次变革》— https://zhuanlan.zhihu.com/p/2053083217167751142
5. 机器之心《一夜之间，Claude 成我同事了》— https://zhuanlan.zhihu.com/p/2053066165874906385
6. 恋猫《如何评价 Anthropic 最新发布的 Claude Tag?》（分层记忆、scoped identity）— https://www.zhihu.com/question/2053047749281690000/answer/2053056568460056154
7. 新智元《如何评价 Anthropic 最新发布的 Claude Tag?》（shared identity）— https://www.zhihu.com/question/2053047749281690000/answer/2053056125101139043
8. 字母榜《agent 进驻工作群，我们给豆包支的招，Claude 听进去了》— https://zhuanlan.zhihu.com/p/2053424542501037405
9. Zilliz《官宣｜我们推出了开源版 Claude Tag，以及它背后记忆与工具引擎 MFS》— https://zhuanlan.zhihu.com/p/2055615685284328451
10. 智东西《刚刚，Claude 进入美国版飞书，成了我的 AI 新同事》— https://zhuanlan.zhihu.com/p/2053122612826583192
11. xingzhe《尝试分析 Claude Tag：如果企业级 AI 记忆系统不只是 RAG，那么它会长什么样?》（状态 vs 记忆）— https://zhuanlan.zhihu.com/p/2053514302267602588
12. 多元宇宙喵《Claude Tag 到底是什么？Anthropic 的新野心》— https://zhuanlan.zhihu.com/p/2053945438433292710
13. Kitt在进化《如何评价 Anthropic 最新发布的 Claude Tag?》（Agent Identity、Ambient Mode）— https://www.zhihu.com/question/2053047749281690000/answer/2060787262057625110
14. 观察者-女《Claude AI 推出新功能，该功能有哪些亮点?》（公司记忆壁垒）— https://www.zhihu.com/question/1962534853443555497/answer/2062483704979666227
15. 智见AGI《拆解 Agent Workspace 三条技术路线：WorkBuddy、Claude Tag 与 EragonAI》— https://zhuanlan.zhihu.com/p/2062817050687541399
16. 高国生成式《「@Claude 帮我把这个 bug 修了」——Claude Tag 深度实操：AI 同事真的来了》— https://zhuanlan.zhihu.com/p/2054527342681257407
17. 林夕《万字深度解析 Claude Tag！9 大落地场景全覆盖》（Channel Session、Agent Proxy）— https://zhuanlan.zhihu.com/p/2058507714796451830
18. riba2534《把 Agent 关进盒子里》（Session/Harness/Sandbox 三接口）— https://zhuanlan.zhihu.com/p/2028230645747393820
19. DeepToken《Claude Managed Agents 正式进入公测》（解耦收益、p50/p95 指标）— https://zhuanlan.zhihu.com/p/2027360594957715287
20. 健康人猿《Anthropic 重磅更新 Claude Tag》（5 大核心能力）— https://zhuanlan.zhihu.com/p/2053073974448411655
21. 数据与AI爱好者《如何评价 Anthropic 最新发布的 Claude Tag?》（团队记忆）— https://www.zhihu.com/question/2053047749281690000/answer/2053211613709260733
