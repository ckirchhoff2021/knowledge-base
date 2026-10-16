# 智能体互联协议栈前沿技术报告：MCP、A2A 与 AGENTS.md —— Agent 时代的「TCP/IP」

> 关键词：模型上下文协议（MCP）、智能体间协议（A2A / Agent2Agent）、AGENTS.md、Function Calling、智能体互操作（Agent Interoperability）、工具投毒（Tool Poisoning）、Rug Pull、提示注入（Prompt Injection）、OAuth 2.1、Agent Card、多智能体系统（MAS）、Agentic AI Foundation（AAIF）

> 信源声明：本报告基于 **19 篇 arXiv 原始论文**（编号已通过 arxiv.org 摘要页逐一核实，标题、作者、日期一一对应；作者另对其中 4 篇关键论文二次抽查无误）、MCP / A2A / AGENTS.md 官方规范与文档，以及 Linux 基金会、Anthropic、Google、OpenAI 官方公告交叉验证。凡厂商或基金会自报数据，均标注「官方口径」；第三方扫描与估算数据均注明出处与采集日期。报告日期：2026-10-16。

## 摘要

2024 年 11 月之前，把一个大模型接入外部工具是一场「点对点」的苦役：每新增一个模型、一个工具，就要写一套定制的函数调用与鉴权胶水，N 个模型 × M 个工具意味着 N·M 份集成。2024 年 11 月 25 日，Anthropic 开源 **模型上下文协议（Model Context Protocol，MCP）**，用一套基于 JSON-RPC 2.0 的客户端—服务器标准，把 N·M 降为 N+M，被官方称为「AI 应用连接外部世界的 USB-C 接口」。短短一年内，MCP 月 SDK 下载量突破 **9700 万**（官方口径，2025-12），公开 MCP 服务器超过 **1 万个**，并被 ChatGPT、Claude、Gemini、Microsoft Copilot、VS Code、Cursor 等主流平台一等公民式地支持。

如果说 MCP 解决的是「智能体**纵向**连接工具与数据」，那么 2025 年 4 月 Google 发布、6 月捐赠给 Linux 基金会的 **智能体间协议（Agent2Agent，A2A）** 解决的就是「智能体**横向**发现并协作」：通过机器可读的 **Agent Card** 做能力发现，以有状态的 **Task**（submitted / working / input-required / completed / failed / canceled / unknown 七态）协调跨框架、跨组织的任务委派，并刻意保持「不透明智能体」以保护各方内部实现。而 2025 年 8 月 OpenAI 发布的 **AGENTS.md**，则把标准下沉到项目层——用一个与工具无关的 Markdown 文件，告诉 AI 编码智能体该如何与某个代码库协作，已被 6 万多个开源项目采用（官方口径）。

2025 年 12 月 9 日是这一方向的里程碑：Linux 基金会宣布成立 **Agentic AI Foundation（AAIF）**，Anthropic、Block、OpenAI 分别把 MCP、goose、AGENTS.md 作为创始项目捐赠；AWS、Anthropic、Block、Bloomberg、Cloudflare、Google、Microsoft、OpenAI 八家互为竞争对手的公司同为白金成员。这标志着智能体互联标准从「单一厂商发起」进入「全行业中立共治」，智源研究院《2026 十大 AI 技术趋势》更将其列为第 3 位，直言「Agent 时代的 TCP/IP 初具雏形」。

与此同时，安全成为这套协议栈最大的短板：独立扫描反复发现 30%–82% 的公开 MCP 服务器存在可利用缺陷，仅约 8.5% 使用 OAuth；**工具投毒（Tool Poisoning）**、**Rug Pull（静默重定义）**、工具影子（Tool Shadowing）、跨服务器级联与「致命三要素（Lethal Trifecta）」等攻击在 2025 年集中曝光，CVE-2025-6514、CVE-2025-49596、CVE-2025-54136 等高危漏洞已波及数十万开发者环境；学术测量显示 MCP 架构会把攻击成功率相对非 MCP 集成放大约 23–41 个百分点。本报告系统拆解三大协议的原理与协同，梳理协议全景（IBM ACP、ANP、AGNTCY、IoA），深潜安全攻击面与防御，并给出工程落地清单与开放问题。

## 目录

0. 为什么是现在：从孤立模型到互联智能体
1. 问题本质：N×M 集成困境与三层协议栈
2. MCP 深度拆解：智能体纵向连接工具与数据
3. A2A 深度拆解：智能体横向发现与协作
4. AGENTS.md：项目层的人机—机机约定
5. 三协议协同与协议全景对比
6. 安全纵深：攻击面、CVE 与防御体系
7. 生态、治理与产业格局
8. 工程落地 Checklist
9. 开放问题与未来展望
10. 结论
11. 参考文献

---

## 0. 为什么是现在：从孤立模型到互联智能体

### 0.1 大模型价值的瓶颈正在从「模型」转向「连接」

过去数年，AI 竞争的主线是模型参数与基准分数。但当模型被放进真实工作流，人们很快发现一个更强的模型并不等于一个更有用的系统：模型需要读取实时数据库、操作日历与邮箱、调用内部 ERP/CRM、执行代码、在多个专业系统之间跑腿。**智能体（Agent）的能力上限，越来越取决于它能多可靠、多低成本地与外部世界连接**，而非其参数规模。

这带来一个工程现实：在标准缺位时，连接是「一对一」定制的。每一家模型厂商有自己的函数调用（Function Calling）格式，每一个 SaaS 暴露自己的 REST API 与鉴权方式，二者之间需要人工编写并长期维护适配层。集成数量随模型与工具的数量**乘法增长**，且高度脆弱——工具接口一改、模型一升级，胶水代码就可能失效。

### 0.2 2024–2026：三层标准在一年内快速成形

智能体互联的标准化在极短时间内沿三个层次同时落地：

- **工具层（纵向）**：MCP 让任意智能体以统一方式发现并调用任意外部工具、读取数据、复用提示模板；
- **通信层（横向）**：A2A 让由不同框架构建、不同组织运营的智能体互相发现能力、协商交互模式并委派任务；
- **项目层（就地）**：AGENTS.md 让代码库以机器可读方式声明「智能体应如何与我协作」。

三者并非竞争关系，而是像网络协议栈一样分层互补：智能体先读项目约定（AGENTS.md），通过标准协议使用工具（MCP），并在需要时把任务委派给其他智能体（A2A）。2025 年 12 月，三大项目汇聚于 Linux 基金会旗下新成立的 AAIF，互为对手的主要厂商全部加入治理——这是判断一个标准「是否已成事实标准」的最强信号。

### 0.3 为什么本报告值得写：知识库中的空白与方向的时效性

经与既有知识库逐一比对：MCP / A2A 此前仅在《Loop Engineering》《智能体记忆》《Claude Tag》等报告中被零星提及（合计个位数次），**从无一篇专题报告**；AGENTS.md 则几乎未被覆盖。而该方向恰是 2025 年末至 2026 年产业落地与安全风险最集中的基础设施层。本报告即填补这一空白，并与知识库中聚焦「单体智能体内部机制」的报告（智能体记忆、Agentic RL、后训练等）形成互补——那些回答「一个智能体如何学习与记忆」，本篇回答「众多智能体与工具如何说同一种语言」。
---

## 1. 问题本质：N×M 集成困境与三层协议栈

### 1.1 从 N×M 到 N+M：标准化的第一性价值

在没有通用协议时，集成关系是一张**完全二分图**：N 个 AI 应用/模型，M 个外部服务，最坏需要 N×M 套定制适配器。每一套都要处理：如何发现对方有哪些能力、参数 schema 是什么、如何鉴权、如何传输、如何处理流式与错误、如何管理状态。MCP 的核心价值，是在模型与服务之间插入一个中立的协议层，使每个 AI 应用只需实现一次 MCP 客户端、每个服务只需实现一次 MCP 服务器，集成总数降为 **N+M**。

> diagrams/1-stack.html

这与互联网史上的标准化同构：USB-C 统一了外设的物理与电气接口，ODBC/JDBC 统一了数据库访问，TCP/IP 统一了主机互联。MCP 官方将其比喻为「AI 应用的 USB-C」，而智源报告直接称之为「Agent 时代的 TCP/IP」。需要强调的是，MCP **并不取代** Function Calling 或 OpenAPI：Function Calling 是模型层「按 schema 输出结构化调用」的能力，OpenAPI 描述传统 REST 接口，MCP 则是架在二者之上、面向智能体的**连接与发现层**——业界普遍做法正是用 MCP 服务器去包装既有的 OpenAPI 服务。

### 1.2 三层协议栈：项目层、工具层、通信层

| 层次 | 代表标准 | 发起方 / 治理 | 解决的问题 | 类比 |
|---|---|---|---|---|
| 项目层 | **AGENTS.md**（及 CLAUDE.md / .cursorrules 等前身） | OpenAI 发起，2025-12 捐赠 AAIF | 告诉智能体如何与某个代码库/项目协作 | 给机器读的 README |
| 工具层（纵向） | **MCP** | Anthropic 发起，2025-12 捐赠 AAIF | 智能体 ↔ 工具 / 数据 / 应用的统一连接 | AI 的 USB-C |
| 通信层（横向） | **A2A** | Google 发起，2025-06 捐赠 Linux 基金会 | 跨框架、跨组织的智能体 ↔ 智能体协作 | 大使馆之间的外交协议 |

- **项目层是「就地约定」**：随代码库分发，描述编码规范、架构决策、测试要求、安全边界；
- **工具层是「纵向加深」**：单个智能体通过它触达更多工具与数据，是一种**客户端—服务器**的紧耦合集成，服务器向客户端显式暴露 tools/resources/prompts；
- **通信层是「横向拓展」**：智能体之间以**点对点**方式协作，核心原则是「不透明（opaque）」——彼此不暴露内部状态、记忆与工具，只通过 Agent Card 声明能力、通过 Task 交换消息与产物。

一个功能完整的智能体网络通常三者并用：编码智能体读 `AGENTS.md` 了解项目，用 MCP 连接 GitHub / 数据库 / 错误追踪，再通过 A2A 把测试任务委派给另一个团队、另一套框架构建的 QA 智能体，双方都无需了解对方内部实现。

### 1.3 与单体智能体技术的关系：正交而非重叠

需要厘清本报告与知识库既有方向的边界。智能体记忆（Memory）研究智能体内部如何存取经验；Agentic RL 研究如何在多轮环境中训练策略；后训练研究如何塑造模型权重；Loop Engineering 研究如何为单个智能体设计循环。这些都发生在**智能体内部**。而互联协议栈研究的是**智能体与智能体、智能体与工具之间的接口契约**：能力如何被发现、消息如何被格式化、身份如何被认证、信任边界划在哪里。它是让「单体智能」可组合为「群体智能」的社会—技术基础设施，与模型能力正交，却决定了应用上限。

---

## 2. MCP 深度拆解：智能体纵向连接工具与数据

### 2.1 起源与定位

MCP 由 Anthropic 于 **2024 年 11 月 25 日**开源，最初是为解决其内部团队把 Claude 接入各类数据源时的重复集成问题。它是一个**开放、有状态、基于 JSON-RPC 2.0** 的协议，统一了智能体与外部系统之间的能力发现、消息格式、传输与授权。其设计目标有三：**即插即用**（服务器可被任意兼容客户端发现与复用）、**安全可控**（主机可对工具调用做人类确认与权限隔离）、**传输无关**（本地进程与远程服务共用一套语义）。

### 2.2 三大角色：Host、Client、Server

MCP 采用客户端—服务器架构，包含三个关键角色：

- **Host（宿主）**：面向用户的 AI 应用，如 Claude Desktop、VS Code、Cursor、自研 Agent 运行时。它负责解析用户意图、聚合多个服务器的能力、执行安全策略（如调用前弹窗确认）；
- **Client（客户端）**：宿主内部的连接器，与**一个** MCP 服务器保持一条专用的 1:1 会话。宿主要连多个服务器，就实例化多个 Client；
- **Server（服务器）**：封装某一类外部系统（数据库、GitHub、Slack、文件系统……），向客户端暴露标准化能力。

这种「一个 Client 对一个 Server」的设计把会话状态与权限边界隔离在单条连接内：一个服务器被攻破或行为异常，不必然污染其他连接（尽管后文会看到，跨服务器级联仍是高危场景）。

> diagrams/2-mcp.html

### 2.3 三类服务器原语：Tools、Resources、Prompts

服务器通过三类原语对外提供价值，可粗略记为「动词、名词、句式」：

| 原语 | 语义 | 由谁控制 | 典型方法 | 例子 |
|---|---|---|---|---|
| **Tools**（工具） | 可执行的函数，产生副作用 | **模型**决定何时调用 | `tools/list`、`tools/call` | 发送 Slack 消息、执行 SQL、下单 |
| **Resources**（资源） | 只读的上下文数据，由 URI 标识 | **应用/宿主**决定何时提供 | `resources/list`、`resources/read` | 一个文件、一封邮件、一条数据库记录 |
| **Prompts**（提示模板） | 可复用的交互模板/斜杠命令 | **用户**选择触发 | `prompts/list`、`prompts/get` | 「生成 {区间} 销售总结」模板 |

- **Tools 是「动词」**：参数用 JSON Schema 严格描述，模型据此判断「何时用、传什么」。工具是唯一被设计为可改变外部世界状态的原语，因此也是安全管控的焦点；
- **Resources 是「名词」**：以 `file://`、`db://` 等 URI 定位，内容可为 UTF-8 文本或 base64 二进制（PDF、图片、音频）。资源的关键风险在于其内容可能携带**间接提示注入**——一封恶意邮件的正文也是「资源」；
- **Prompts 是「句式」**：把最佳实践固化为带占位符的模板，降低使用门槛、保证团队内调用一致性。

此外协议还定义了 **Sampling**（服务器反向请求宿主中的模型做一次补全）与 **Roots**（客户端告知服务器可操作的文件系统根范围）等能力。值得注意的是，Sampling 允许服务器把内容「向上」注入模型上下文，安全研究指出该路径缺乏来源认证，是数据外泄与持久化的高危通道（见第 6 章）。

### 2.4 传输层：stdio 与 Streamable HTTP

MCP 在两种传输上承载同一套 JSON-RPC 语义：

- **stdio（标准输入输出）**：服务器作为本地子进程运行，零网络开销，适合本地工具（文件系统、Git、本地脚本）。这也是当前最常见的部署形态——Clutch Security 的扫描（第三方，2026）称约 **86%** 的 MCP 服务器运行在开发者本机；
- **Streamable HTTP（可流式 HTTP）**：单个 HTTP 端点，用 HTTP POST 传 JSON-RPC，可选通过 Server-Sent Events 流式返回，支持 OAuth 授权，适合远程、多租户的服务端部署。

协议演进上，早期的「HTTP+SSE」双端点传输已被**弃用**，由 2025-06-25 版引入的 Streamable HTTP 取代；2025-11-25 版继续完善授权；后续版本进一步放松了强制会话（session）要求以支持无状态水平扩展。

### 2.5 授权：从静态 API Key 到 OAuth 2.1

远程 MCP 的授权在 2025 年中被显著加强：规范把**资源服务器**与**授权服务器**分离，推荐采用 **OAuth 2.1 + PKCE**，支持细粒度 scope 与令牌刷新。但现实 adoption 严重滞后——Astrix 对 5,200 多个服务器的审计（第三方，2025-10）发现：88% 需要凭据，其中 **53% 依赖静态 API Key / PAT，仅约 8.5% 使用 OAuth，79% 通过环境变量传递密钥**。这意味着大量部署把「长期有效、权限过大、易泄漏」的静态令牌直接交给远程服务器，是供应链与凭据失窃的主要温床。

### 2.6 一次完整调用的生命周期

以「帮我分析华东区 Q2 销售并发邮件」为例：① 宿主启动并为每个服务器建立 Client；② Client 发起 `initialize` 握手并 `tools/list` 拉取能力清单；③ 模型结合用户意图与工具描述，决定调用 `query_sales(region, quarter)`；④ 宿主依安全策略决定是否需要人类确认，随后以 `tools/call` 发出 JSON-RPC 请求；⑤ 服务器执行（可能访问数据库）并把结构化结果返回；⑥ 模型读取结果继续推理，再调用 `send_email` 工具完成副作用；⑦ 全过程在宿主侧留下可审计日志。能力发现（②）与调用（④）的分离，正是 MCP 相对裸 Function Calling 的关键增量——工具是**自描述**的，无需把每个端点硬编码进应用。
---

## 3. A2A 深度拆解：智能体横向发现与协作

### 3.1 起源、定位与治理

**A2A（Agent2Agent，智能体间协议）** 由 Google 于 **2025 年 4 月 9 日**在 Google Cloud Next 大会发布，初始即有 50 多家合作伙伴（Atlassian、Salesforce、SAP、ServiceNow、PayPal、Workday 及多家咨询公司）。它是一个运行在 HTTP(S) 之上、基于 **JSON-RPC 2.0**、可选 SSE 流式传输的开放协议，采用 Apache 2.0 许可，提供 Python、Go、JavaScript、Java、.NET 五种 SDK。**2025 年 6 月 23 日，Google 将 A2A 捐赠给 Linux 基金会**，此后支持组织增至 150 家以上。

A2A 要解决的问题与 MCP 正交：当一个企业里 Salesforce 智能体管 CRM、ServiceNow 智能体管 IT 运维、自研智能体做分析，这些由不同框架构建、不同组织运营、从未被设计为协同工作的智能体，如何**发现彼此能力、协商交互格式、在长任务上协作**？

### 3.2 核心设计原则：不透明智能体（Opaque Agents）

A2A 最重要的架构原则是**不透明**：协作的智能体之间**不暴露**各自的内部状态、记忆、工具与系统提示。一个差旅智能体与一个支付智能体协作时，双方都无需、也无法看到对方「怎么想、用了哪些工具」。这一原则有双重价值：保护各方知识产权与商业机密；把互操作建立在**最小必要的外部契约**（能力声明 + 消息交换）之上，从而天然支持多供应商、跨组织协作。这与 MCP「服务器向客户端显式暴露 tools/resources/prompts」形成鲜明对照——MCP 是纵向信任域内的深度集成，A2A 是横向信任边界上的最小耦合。

### 3.3 Agent Card：机器可读的能力名片

互操作的前提是发现。每个 A2A 智能体在一个众所周知的路径 `/.well-known/agent.json` 发布一份 **Agent Card（智能体卡片）**，用 JSON 描述：

- **身份与端点**：智能体名称、描述、开发者信息、服务 URL；
- **能力（capabilities）**：是否支持流式传输（streaming）、推送通知（pushNotifications）、长任务等；
- **技能（skills）**：一组带 `id`、名称、描述、标签（tags）、示例的能力声明，供其他智能体判断「它擅长什么」；
- **认证要求**：需要何种鉴权方案；
- **支持的交互模态**：文本、表单、富媒体（文件、图片、结构化数据）。

其他智能体先抓取 Agent Card，在发起通信前就了解对方能做什么、需要什么凭据、如何与之对话，无需人工预先撮合。这与微服务世界的服务发现/能力清单（如 OpenAPI + 注册中心）思路一致，但语义是为「智能体消费智能体」量身定制的。安全研究（arXiv:2504.16902）也指出，Agent Card 的真实性、防篡改与生命周期管理是 A2A 信任链的第一道关口。

> diagrams/3-a2a.html

### 3.4 Task：有状态协作的核心

A2A 以 **Task（任务）** 作为协作的有状态实体。**Task 总是由客户端创建，但其状态由远端智能体（服务器）判定与维护**；多个 Task 可通过可选的 `sessionId` 归入同一会话，以支持多轮协作（例如先说「画一只兔子」，再说「把它改成红色」）。一个 Task 有七个明确状态：

| 状态 | 含义 |
|---|---|
| `submitted` | 任务已提交，等待/开始处理 |
| `working` | 远端智能体正在执行 |
| `input-required` | 智能体需要客户端补充信息才能继续 |
| `completed` | 任务成功完成 |
| `failed` | 任务执行失败 |
| `canceled` | 任务被取消 |
| `unknown` | 状态未知（容错态） |

收到任务后，远端智能体可采取多种行为：立即满足、延后调度、拒绝、协商另一种执行方式、请求更多信息、或进一步**委派给其他智能体/系统**。即使初始目标已达成，客户端仍可在同一 Task 上下文中追加修改——这对「对话式、可迭代」的协作至关重要。

### 3.5 Message、Part 与 Artifact：消息与产物的结构

A2A 的数据交换围绕三个结构：

- **Message（消息）**：一次对话往返，携带 `role`（user/agent）与若干 Part；
- **Part（分片）**：消息内容的最小单元，可以是纯文本，也可以是文件、数据等富媒体（用统一的内容结构表示），从而支持多模态协作；
- **Artifact（产物）**：智能体在任务中生成的**工作成果**（如生成的文档、图片、代码、报表），可增量更新，并与交互消息（想法、指令）区分开。

在接口层面，客户端通过 `message/send`（同步）或 `message/stream`（流式，经 SSE 持续返回 `TaskStatusUpdateEvent` 与 `TaskArtifactUpdateEvent`）与服务器交互；对于可能耗时数分钟到数小时的长任务，还支持异步通知（`TaskStatusUpdateEvent` 中的 `final` 标志标志事件流结束）与断线后的推送通知/重新订阅。

### 3.6 三种交互模式：同步、流式、异步

- **同步请求—响应**：适合简单、快速的任务，一次 `message/send` 返回结果；
- **流式（SSE）**：服务器在执行过程中持续推送状态更新与逐步生成的 Artifact，客户端可实时看到进展；
- **异步长任务 + 推送通知**：任务运行时间很长时，客户端可断开，服务器在完成或需要输入（回到 `input-required`）时通过推送通知唤回客户端。

### 3.7 A2A 与 MCP：纵向与横向的互补

二者是**互补而非竞争**，业界已形成清晰的组合范式：

```
用户
 │
 ▼
编排智能体（Orchestrator）
 │  用 A2A 把任务横向委派给专家智能体
 ▼
专家智能体（Specialist）
 │  用 MCP 纵向调用工具 / 读取数据
 ▼
工具 · 数据库 · API
```

编排智能体用 A2A 发现专家智能体、以 Task 协调；专家智能体则在自己的信任域内用 MCP 连接具体工具完成执行。学术工作（arXiv:2506.01804）专门论证了这一 MCP × A2A 框架如何提升自主智能体的互操作性；安全研究（arXiv:2504.16902）则进一步讨论了二者协同时的威胁建模——横向委派会把信任与攻击面沿任务链传递，因此身份认证、最小权限与任务完整性必须跨协议贯通设计。
---

## 4. AGENTS.md：项目层的人机—机机约定

### 4.1 从给人读的 README 到给智能体读的 AGENTS.md

`AGENTS.md` 是一种**约定优于配置（convention over configuration）**：在代码仓库根目录（及子目录）放置一个 Markdown 文件，为在该仓库上工作的 AI 编码智能体提供指令。README 告诉人类开发者如何搭建与参与项目，`AGENTS.md` 则告诉 AI 智能体应遵守哪些编码规范、已做过哪些架构决策、测试与部署规则是什么、哪些文件不可触碰。它通常包含：

- **编码规范**：语言约定、格式化规则、首选库；
- **架构决策**：目录组织、不同类型代码的存放位置；
- **测试要求**：使用什么测试框架、最低覆盖率、提交前必须跑的命令；
- **构建与部署**：构建流程、CI/CD 的预期行为；
- **安全边界**：哪些文件（如 `.env`）不可修改、用户输入如何校验、密钥如何处理。

一个最小示例：

```markdown
# Project Guidelines

## Architecture
- Next.js 15 + App Router，TypeScript strict 必须开启
- 所有 API 路由放在 app/api/ 下

## Coding Standards
- 函数式组件 + Hooks；优先 Server Component
- 所有数据库访问走 Prisma

## Testing
- 单元测试 Jest，端到端 Playwright；新代码覆盖率 ≥ 80%
- 提交前必须运行 npm test

## Security
- 禁止修改 .env；所有用户输入用 Zod 校验
```

### 4.2 起源谱系：从各家私有约定到中立标准

「给 AI 智能体的项目指令」并非凭空出现，而是 2025 年从 AI 辅助编程社区自下而上涌现，各工具先有私有变体：Anthropic 的 Claude Code 用 `CLAUDE.md`，Cursor 用 `.cursorrules`，Windsurf 用 `.windsurfrules`。**OpenAI 于 2025 年 8 月发布与工具无关的通用标准 `AGENTS.md`**，意图统一这些碎片化约定；**2025 年 12 月 9 日，OpenAI 将其作为三大创始项目之一捐赠给 AAIF**。据 Linux 基金会官方新闻稿（2025-12-09，官方口径），`AGENTS.md` 当时已被 **6 万多个**开源项目与智能体框架采用，包括 Amp、Codex、Cursor、Devin、Factory、Gemini CLI、GitHub Copilot、Jules、VS Code 等。

### 4.3 深层意义：代码库第一次以「非人类智能体」为主要受众

`AGENTS.md` 标志着计算机历史上一个微妙但重要的转折：**代码库第一次被设计为由非人类智能体作为主要受众来消费**。它不是「给可能使用 AI 工具的人」看的文档，而是直接写给 AI 工具的契约。这与自然语言编程、智能体原生软件开发的大趋势一脉相承，也让项目层成为协议栈中分发最广、摩擦最小的一层——它不需要运行任何服务，只随仓库版本化分发。

需要注意，`AGENTS.md` 与 MCP / A2A 处在不同抽象级：它不负责传输、鉴权或能力调用，而是**就地（in-repo）提供策略上下文**。智能体运行时往往先加载该文件，再决定如何调用 MCP 工具、是否发起 A2A 协作。它也是提示注入的潜在入口——仓库中被投毒的指令文件可能诱导智能体执行越权操作，因此其内容应与其他外部输入一样纳入信任分级。

---

## 5. 三协议协同与协议全景对比

### 5.1 一个端到端协同场景

设想一个开发团队用 AI 智能体交付功能：① 智能体读取仓库根目录的 `AGENTS.md`，得知项目使用 Next.js 15 + TypeScript + Prisma、所有 API 路由放在 `app/api/`、提交前必须通过测试；② 它通过 **MCP** 连接 GitHub（读写 PR）、Sentry（查错误日志）、PostgreSQL（查表结构），这些连接经过统一的发现与鉴权；③ 写完代码后，它通过 **A2A** 把变更发布给另一个团队、另一套框架构建的独立 QA 智能体——对方依据其 Agent Card 被发现，运行自动化测试并回传结果；两个智能体全程不了解对方内部实现。这条链路恰好对应人类协作：读项目文档（AGENTS.md）、使用自己的工具（MCP）、与同事沟通（A2A）。

### 5.2 三协议横向对比

| 维度 | AGENTS.md | MCP | A2A |
|---|---|---|---|
| 方向 | 就地策略 | 智能体 → 工具（纵向） | 智能体 → 智能体（横向） |
| 发起 / 治理 | OpenAI → AAIF | Anthropic → AAIF | Google → Linux 基金会 |
| 架构形态 | 仓库内 Markdown | 客户端—服务器（Host/Client/Server） | 点对点 + Agent Card 发现 |
| 核心对象 | 项目指令 | Tools / Resources / Prompts | Agent Card / Task / Message / Artifact |
| 传输 | 无（随仓库分发） | stdio、Streamable HTTP | HTTP(S) + JSON-RPC 2.0 + SSE |
| 状态 | 无 | 有状态会话（逐步放松） | 有状态 Task（七态）+ sessionId |
| 内部可见性 | — | 服务器向客户端暴露能力 | **不透明**，互不暴露内部 |
| 授权 | — | OAuth 2.1 + PKCE（推荐） | 多种鉴权方案，Agent Card 声明 |
| 首次发布 | 2025-08 | 2024-11-25 | 2025-04-09 |
| 类比 | 给机器读的 README | AI 的 USB-C | 智能体间的外交协议 |

### 5.3 更广阔的协议全景：不止 MCP 与 A2A

MCP 与 A2A 并非孤例，2025 年涌现出一批互操作协议，并在 Linux 基金会等中立治理下快速收敛：

| 协议 | 发起方 | 定位 | 治理归宿 / 现状 |
|---|---|---|---|
| **MCP** | Anthropic | 智能体—工具/数据连接 | 2025-12 捐赠 AAIF |
| **A2A** | Google | 智能体—智能体通信 | 2025-06 捐赠 Linux 基金会，150+ 组织支持 |
| **AGENTS.md** | OpenAI | 项目级智能体指令 | 2025-12 捐赠 AAIF |
| **IBM ACP**（Agent Communication Protocol） | IBM Research（2025-03） | 轻量、REST 原生的智能体间通信，驱动 BeeAI | **2025-09 与 A2A 正式合并**，理念被吸收，IBM 成员进入 A2A 技术指导委员会 |
| **AGNTCY** | Cisco Outshift（2025-03，联合 LangChain、Galileo） | 「智能体互联网」全栈：目录、身份、消息（SLIM）、可观测性、OASF schema | 2025-07 捐赠 Linux 基金会，组件融入 AAIF / A2A / ARD |
| **ANP**（Agent Network Protocol） | 社区开源项目 | **去中心化、身份优先**，基于 W3C DID / JSON-LD，面向跨信任域 P2P 网络 | 已向 AAIF 提交项目提案（arXiv:2508.00007 白皮书） |
| **IoA**（Internet of Agents） | 清华等（arXiv:2407.07061） | 异构智能体的类即时通讯集成协议与动态组队 | 学术框架 |

> **缩写警示**：生态中有两个不同的「ACP」。其一是 **IBM 的 Agent Communication Protocol**（智能体通信协议，已并入 A2A）；其二是 **OpenAI 的 Agentic Commerce Protocol**（智能体商务/支付协议，用于 ChatGPT 应用内交易），二者完全无关，引用时务必区分。

整体格局可概括为：**MCP 统一了「接工具」，A2A（吸收 IBM ACP）正在统一「连智能体」，AGNTCY 补齐目录/身份/可观测性等基础设施，ANP 探索去信任的点对点未来**，而所有主要玩家在 2025 年底汇聚到 Linux 基金会的中立治理之下。这种「先各自跑通、再捐给中立基金会收敛」的路径，与 Kubernetes、Node.js 等关键基础设施的成长轨迹高度相似。

### 5.4 与 Function Calling / OpenAPI 的边界（再强调）

- **Function Calling** 是**模型层能力**：模型按 JSON Schema 输出一次结构化调用。它是点对点、每工具自定义的，不含标准的发现、连接、状态与授权；
- **OpenAPI** 描述传统 REST 接口的契约，面向人类与通用 HTTP 客户端；
- **MCP** 在其上提供面向智能体的**自描述发现 + 标准传输 + 安全控制 + 状态管理**，通常以 MCP 服务器包装既有 OpenAPI 服务。

因此「MCP 会不会取代 Function Calling / REST」是个伪命题：MCP 复用模型的函数调用能力作为执行手段，复用 HTTP 作为传输，它新增的是**互操作契约与生态网络效应**。
---

## 6. 安全纵深：攻击面、CVE 与防御体系

互联协议让智能体强大，也让它第一次拥有了可被规模化利用的「攻击面」。2025 年是 MCP 安全问题集中曝光的一年：学术研究、工业审计与真实 CVE 三条线索相互印证，结论高度一致——**采用速度远远跑在了安全成熟度前面**。

> diagrams/4-security.html

### 6.1 威胁全景：为什么智能体协议比传统 API 更危险

传统 API 的调用参数由程序员写定，而 MCP/A2A 把「调用哪个工具、传什么参数」的决策权交给了模型，而模型会读取并信任**外部内容**（邮件、网页、工具返回、资源文件）。这就引入了传统软件没有的新攻击链：攻击者不必直接攻破程序，只需把恶意指令藏进模型会读到的内容，诱导模型自己发起越权调用。Simon Willison 于 2025 年 4 月概括的「**致命三要素（Lethal Trifecta）**」精准描述了风险成立的三个条件：① 智能体可访问**私有数据**；② 智能体可访问**不可信数据**（含提示注入）；③ 智能体拥有**外部通信/副作用能力**。三者同时具备时，提示注入即可升级为数据外泄与系统接管。

### 6.2 主要攻击类型

| 攻击 | 原理 | 典型后果 |
|---|---|---|
| **间接提示注入（Indirect Prompt Injection）** | 把恶意指令藏进工具返回、资源（邮件/文档/网页）中，模型当作指令执行 | 越权操作、数据外泄 |
| **工具投毒（Tool Poisoning）** | 在工具的名称/描述/参数 schema 等**元数据**中嵌入恶意指令，无需工具被正常执行即触发 | 诱导泄露、篡改调用、植入后门 |
| **Rug Pull（抽地毯/静默重定义）** | 已被信任并安装的服务器，在事后通过更新**悄悄改变**工具描述或行为（版本固定也难防动态发现） | 供应链式持久化后门 |
| **工具影子（Tool Shadowing）** | 恶意工具伪装成可信工具的同名/近似能力，劫持本应发往正当工具的调用 | 凭据/数据截获 |
| **跨服务器攻击 / 级联** | 一个被攻陷的服务器借 Sampling、共享上下文或委派，把攻击传播到其他服务器 | 权限横向移动 |
| **命令注入 / 路径穿越 / SSRF** | 服务器实现层未校验输入（与传统 Web 漏洞同源，但暴露面被协议放大） | RCE、任意文件读写、内网探测 |
| **混淆代理（Confused Deputy）/ OAuth 缺陷** | 智能体持高权限令牌，被诱导以自身权限替攻击者行事 | 越权访问第三方资源 |
| **恶意技能 / 供应链投毒** | 注册表中的恶意服务器或智能体技能包（npm/PyPI 投毒） | 安装即中招 |

### 6.3 工业审计与学术测量：量化的风险

多份独立测量给出了相互印证的高风险比例（均为第三方口径，方法与样本不同，数字有差异，应看区间而非单点）：

- **服务器实现缺陷普遍**：Endor Labs（2025）分析 2,614 个实现，发现 **82%** 使用文件操作的服务器易受路径穿越；Equixly（2025–2026）测得约 **43%** 存在命令注入；BlueRock Security（2026）对 7,000 多个服务器测得约 **36.7%** 存在 SSRF；Enkrypt AI（2025-10）扫描的 1,000 个服务器中 **33%** 有严重漏洞。
- **鉴权薄弱、裸奔普遍**：Astrix（2025-10）审计 5,200 多个服务器，**仅约 8.5% 使用 OAuth，53% 用静态 API Key**；Trend Micro（2025-07）发现 **492 个 MCP 服务器直接暴露在公网且无任何认证/加密**，可触达 1,402 个工具（后续追踪称该数量增至约 1,467）。
- **工具投毒极易得手，且越强的模型越易中招**：MCPTox（arXiv:2508.14925）基于 45 个真实在线服务器、353 个真实工具构造 1,348 个恶意用例、测试 20 个主流智能体，最高攻击成功率达 **72.8%（o1-mini）**；智能体极少拒绝攻击，最高拒绝率（Claude-3.7-Sonnet）也**不到 3%**。一个反直觉但关键的发现是：**模型指令遵循能力越强，越容易执行藏在工具元数据里的恶意指令**。
- **协议本身放大攻击**：《Breaking the Protocol》（arXiv:2601.17549）在 847 个攻击场景、5 个服务器实现上测得，MCP 架构使总体攻击成功率达 **52.8%**（非 MCP 基线 26.4%，相对放大约 23–41 个百分点）；跨服务器传播成功率从 19.7% 升至 **61.3%**，Sampling 注入高达 **67.2%**；在「5 个服务器中 1 个被攻陷」的配置下，总体成功率 **78.3%、级联率 72.4%**。该研究同时提出缓解方案 **ATTESTMCP**（来源认证），可把总体成功率从 52.8% 降到 **12.4%**（降 76.5%），每条消息中位延迟仅约 8.3ms。
- **生态注水与维护断层**：首个大规模生态测量（arXiv:2509.25292）用爬虫采集六大市场 17,630 条记录，仅 **49.1%** 有效（过半为占位、分叉或废弃项目）；分析的 8,060 个有效服务器中 **21.9% 已超过一年不活跃**，11.2% 暴露敏感 API；客户端侧 **80.9% 仅支持单服务器连接**。这意味着大量未打补丁的长尾服务器与缺乏多连接隔离的客户端并存。
- **现有防护基本失效**：MCPSecBench（arXiv:2508.13220）归纳出 4 大攻击面、17 种攻击类型，在 Claude、OpenAI、Cursor 上**所有攻击面均可得手**，现有防护机制平均成功率不足 30%。

> 方法论提示：不同扫描器误报率差异很大（有审计称基于 YARA 的扫描器误报率约 78%），上述百分比应理解为「风险量级」而非精确测量；但「实现缺陷普遍、鉴权薄弱、投毒类攻击对强模型尤其有效」这一定性结论被多方一致支持。

### 6.4 真实 CVE 与供应链事件

2025 年起，MCP 相关高危漏洞开始进入官方 CVE 通道，且多个已在真实场景实现远程代码执行：

| CVE / 事件 | 组件 | 等级 | 要点 |
|---|---|---|---|
| **CVE-2025-6514** | mcp-remote | CVSS 9.6 | OS 命令注入，影响 v0.0.5–0.1.15（0.1.16 修复）；下载量超 43.7 万，JFrog 称其首次在真实场景实现「连接不可信远程服务器即 RCE」 |
| **CVE-2025-49596** | Anthropic MCP Inspector | CVSS 9.4 | 浏览器/DNS rebinding + 绑定 0.0.0.0 导致 RCE，v0.14.1 修复（Oligo Security） |
| **CVE-2025-54136（MCPoison）** | Cursor | CVSS 7.2 | 借「被信任但被调包」的 MCP 配置实现持久化 RCE，Cursor 1.3 修复（Check Point） |
| **CVE-2025-54135（CurXecute）** | Cursor | — | MCP 自启动提示注入导致 RCE（Aim Labs） |
| **CVE-2025-53110 / 53109** | 官方 Filesystem Server | 7.3 / 8.4 | 目录包含绕过、符号链接绕过（Trend Micro） |
| **CVE-2025-68143/68144/68145** | Anthropic Git Server | — | 路径穿越、参数注入等，2026-01 披露 |
| **CVE-2026-33032（MCPwn）** | nginx-ui MCP | CVSS 9.8 | 认证绕过，已被在野利用，2026-03-15 修复 |
| **postmark-mcp（npm）** | 第三方服务器 | 供应链 | 恶意版本把处理的邮件密送外部域名，Snyk 2025-09 披露，是首个被追踪的恶意 MCP 服务器事件 |

此外 GitGuardian《State of Secrets Sprawl 2026》在公开 GitHub 的 MCP 相关配置中发现 24,008 个密钥，其中 2,117 个仍有效；针对智能体技能生态，Antiy CERT 确认了 1,184 个恶意技能，Snyk 的 ToxicSkills 审计（3,984 个技能）发现 13.4% 至少含一个严重安全问题。HackerOne《2025 报告》则记录提示注入类漏洞报告量同比激增 **540%**，称其为「AI 安全中增长最快的威胁」。

### 6.5 防御体系：从协议到工程的分层加固

防御不能只靠模型「更对齐」——MCPTox 已证明对齐对「借合法工具越权」几乎无效。需要分层纵深：

1. **协议级来源认证与能力证明**：为工具/资源/Sampling 请求附加可验证来源与完整性证明（ATTESTMCP 思路），让模型能区分「指令来自用户」还是「内容来自外部数据」，从根上切断间接注入的权威伪装；
2. **授权最小化**：用 OAuth 2.1 + PKCE 取代静态 API Key，按 scope 下发短期、最小权限令牌；资源服务器与授权服务器分离；严格校验 redirect URI 防 rebinding；
3. **人类在环（Human-in-the-loop）**：对有副作用的工具（发信、下单、删改、转账）默认要求显式确认，宿主侧统一弹窗并展示「将以你的身份执行什么」；
4. **输入输出双向校验与沙箱**：服务器侧对参数做 schema 校验、命令参数化、路径规范化防穿越、SSRF 出站白名单；工具在最小权限容器/沙箱中执行；
5. **信任分级与上下文隔离**：把外部返回内容一律标记为「不可信数据」而非指令；多服务器之间做连接隔离，防跨服务器级联；高敏数据与不可信内容不要同时进入同一智能体（拆解致命三要素）；
6. **供应链治理**：锁定服务器版本与校验和、审计工具描述变更（防 Rug Pull）、只从受信注册表安装、扫描配置中的泄漏密钥、对技能/服务器做发布前安全评审；
7. **检测与红队**：部署静态扫描 + 神经语义检测 + LLM 仲裁的多级过滤（如 MCP-Guard，arXiv:2508.10991，其微调检测器对抗性提示识别准确率报告为 96.01%），并用 MCPXKIT（31 种攻击方法，arXiv:2508.12538）、MCPSecBench 等做持续红队；
8. **组织治理**：建立智能体身份战略（CSA/Strata 2026 调查称仅 23% 企业有正式的 agent 身份策略）、统一审计日志与事件响应。Gartner 预测到 2028 年 25% 的企业泄露可追溯到智能体滥用，治理水位正成为落地生死线。

学术与产业已在协同补位：OWASP 发布了 MCP Top 10，CoSAI（由 Google 于 2024-07 发起、托管于 OASIS，成员含 Amazon、Anthropic、IBM、Microsoft、NVIDIA、OpenAI 等）推动安全的智能体组件与供应链标准，协议级加固（来源认证、无状态扩展下的安全模型）也在持续进入新版规范。
---

## 7. 生态、治理与产业格局

### 7.1 采用规模：一年内从 0 到事实标准

MCP 的网络效应在 2025 年快速形成。采用数据（注意区分官方口径与第三方快照）：

| 指标 | 数值 | 口径 / 日期 |
|---|---|---|
| 月 SDK 下载量（Python + TS） | **9700 万+** | 官方口径，Anthropic / LF，2025-12-09 |
| 公开 MCP 服务器 | **1 万+** | 官方口径，2025-12-09 |
| 官方 MCP Registry 记录 | 9,652 条 latest（28,959 条含历史版本） | 第三方拉取 Registry API，2026-05-24 |
| GitHub `mcp-server` topic 仓库 | 15,926 | 第三方 GitHub API，2026-05-24 |
| `modelcontextprotocol/servers` star | 86,148 | 第三方快照，2026-05-24 |
| AGENTS.md 采用项目 | 6 万+ | 官方口径，2025-12-09 |
| A2A 支持组织 | 150+ | 第三方（IBM Think 等），2025 下半年 |

主流客户端已全面支持：Claude（创始客户端）、OpenAI/ChatGPT（Responses API 支持远程 MCP，Apps SDK 构建其上，2025-05 加入 MCP 指导委员会）、Google Gemini（SDK 支持 MCP 与自动工具循环）、Microsoft Copilot Studio / VS Code、Cursor、GitHub MCP Server（2025-04 公开预览）。早期采用者还包括 Block、Apollo、Zed、Replit、Codeium、Sourcegraph。

需要澄清一个流传甚广的说法：「78% 的企业 AI 团队已在生产用 MCP」并无可靠信源，已被相关研究方撤回。较可信的 Stacklok《State of MCP in Software 2026》（n=300 资深技术负责人，2025-12 调查）显示约 **41%**（29% 有限生产 + 12% 广泛生产）进入某种生产阶段；而 Clutch Security（第三方，2026）称约 86% 的 MCP 活动仍发生在开发者本机、仅约 5% 在生产环境。这说明 MCP 已在**开发者侧高度普及**，但企业级、受治理的规模化部署才刚起步——这也正是 2026 年的主要机会与风险窗口。

### 7.2 治理中立化：AAIF 的标志性意义

**2025 年 12 月 9 日**，Linux 基金会宣布成立 **Agentic AI Foundation（AAIF）**，定位为其旗下的定向基金（directed fund），三大创始项目为 Anthropic 的 MCP、Block 的 goose、OpenAI 的 AGENTS.md。成员结构极具标志性：

- **白金成员（8 家）**：AWS、Anthropic、Block、Bloomberg、Cloudflare、Google、Microsoft、OpenAI；
- **黄金成员（18 家）**：Adyen、Arcade.dev、Cisco、Datadog、Docker、Ericsson、IBM、JetBrains、Okta、Oracle、Runlayer、Salesforce、SAP、Shopify、Snowflake、Temporal、Tetrate、Twilio；
- **白银成员（24 家）**：Apify、Chronosphere、Hugging Face、LanceDB、Pydantic、Stacklok、SUSE、Uber、WorkOS、Zapier、ZED 等。

八家白金成员囊括了所有主要竞争对手，这是判断 MCP「已从 Anthropic 标准变为行业公共品」的决定性证据。在此之前，Google 已于 2025-06-23 把 A2A、Cisco 已于 2025-07 把 AGNTCY、IBM 已于 2025-09 把 ACP（并入 A2A）先后交由 Linux 基金会体系治理。加上 MCPCon / AGNTCon 等社区会议（2026 年已排定北美与欧洲场次），一个中立、开放、多厂商共治的智能体基础设施生态已具雏形。

### 7.3 产业判断：智源趋势与「连接层」的战略位置

智源研究院《2026 十大 AI 技术趋势》将「多智能体系统决定应用上限，Agent 时代的 TCP/IP 初具雏形」列为**第 3 位**，指出 MCP、A2A 等协议的标准化让智能体拥有通用「语言」，多智能体系统将在科研、工业等复杂工作流中成为关键基础设施。这一判断的深层含义是：当模型能力趋于收敛、单体智能遇到天花板，**价值与护城河将向「连接、编排、治理」层迁移**——谁掌握标准互操作、身份信任与跨组织协作，谁就掌握规模化落地的枢纽。对企业而言，这意味着今天围绕 MCP/A2A 做的架构投资，具有类似早期投资 REST/云原生的长期网络效应。

---

## 8. 工程落地 Checklist

### 8.1 如果你要「接入工具」（MCP）

- [ ] 优先复用成熟的官方/受信 MCP 服务器，只在必要时自研；自研时用官方 Python/TS SDK；
- [ ] 工具描述清晰、参数用严格 JSON Schema、提供使用示例；对所有输入做长度/类型/范围校验；
- [ ] 本地工具用 stdio，远程服务用 Streamable HTTP；弃用旧 HTTP+SSE；
- [ ] 远程授权用 OAuth 2.1 + PKCE、短期最小权限令牌，杜绝静态 API Key 硬编码与环境变量裸传；
- [ ] 对有副作用的工具默认开启人类确认；宿主侧统一记录可审计日志；
- [ ] 工具在最小权限沙箱中执行；命令参数化、路径规范化、出站 SSRF 白名单；
- [ ] 多服务器分连接隔离；外部返回内容标记为不可信数据；锁定服务器版本/校验和防 Rug Pull。

### 8.2 如果你要「连接智能体」（A2A）

- [ ] 在 `/.well-known/agent.json` 发布真实、完整、防篡改的 Agent Card，明确技能、模态与鉴权；
- [ ] 正确实现 Task 七态与 `sessionId`，长任务用流式/异步 + 推送通知；
- [ ] 用 Message/Part 表达多模态内容，用 Artifact 区分工作成果与交互消息；
- [ ] 坚持不透明原则：只暴露最小必要契约，不泄露内部状态、记忆与系统提示；
- [ ] 编排方对委派任务做身份认证、最小权限与完整性校验，防信任沿任务链被滥用；
- [ ] 与 MCP 协同时，横向委派（A2A）与纵向执行（MCP）的权限/审计要贯通。

### 8.3 组织与安全治理

- [ ] 建立智能体身份与授权战略，统一注册表准入、版本锁定与发布前安全评审；
- [ ] 部署分层检测（静态扫描 + 语义检测 + LLM 仲裁）并用红队工具包持续演练；
- [ ] 扫描并清除配置中的泄漏密钥；只从受信来源安装服务器/技能；
- [ ] 拆解致命三要素：避免同一智能体同时持有高敏数据、不可信内容与外发副作用；
- [ ] 制定智能体事件响应预案，纳入 Gartner 所警示的智能体滥用风险。

---

## 9. 开放问题与未来展望

1. **协议级信任与来源认证尚未普及**：ATTESTMCP 等方案在实验中有效（攻击成功率降约 76%），但尚未成为默认内置于主流客户端的能力；如何让模型在架构上稳定区分「用户指令」与「外部内容」，仍是根本难题。
2. **强模型更易被投毒的悖论**：指令遵循与工具投毒易感性正相关，意味着单纯扩模/对齐无法解决，需要在训练目标中显式引入「对外部元数据的批判性信任」与对抗性工具数据。
3. **无状态扩展与安全的张力**：规范放松强制会话以支持水平扩展，但有状态协作、审计与最小权限如何在无状态部署中保持，需要新的安全模型。
4. **身份、声誉与可问责性缺失**：当智能体跨组织甚至在开放 P2P 网络（ANP 方向）协作，「谁对一次越权调用负责」尚无成熟的身份、声誉与法律框架；两个「ACP」式的标准歧义也提示术语与边界仍待收敛。
5. **生态注水与长尾维护**：过半注册表项目无效、逾两成服务器年久失修，如何建立可信注册表、安全评分与自动淘汰机制，是公共卫生问题。
6. **评测的真实性**：合成/刷分友好的基准失真，需要更多生成器从未见过的真实世界 holdout 任务，以及对「工具调用是否真正达成意图」的端到端评测。
7. **多智能体系统的宏观安全**：智能体间自主协商、交易与连锁反应可能产生设计者未预期的涌现行为（系统性欺骗、合谋、级联失效），这正与智源趋势 10（从幻觉到欺骗、机制可解释与自演化攻防）交汇，是下一阶段的研究前沿。

展望未来，智能体互联协议栈很可能重演互联网的历史：先统一「传输与互操作」（MCP/A2A 已基本完成），再在其上长出身份、信任、商务支付、治理与声誉等「应用层」，最终让 AI 智能体成为数字经济中一等的交易与协作者。安全与可问责性将决定这一进程的速度与边界。

---

## 10. 结论

2024 年 11 月以来的一年里，智能体互联完成了从「N×M 定制苦役」到「N+M 标准互联」的关键跃迁。**MCP** 用一套 JSON-RPC 客户端—服务器标准统一了智能体与工具、数据的纵向连接；**A2A** 以 Agent Card 与有状态 Task 打通了跨框架、跨组织的横向协作，并吸收 IBM ACP、联合 AGNTCY 形成更完整的基础设施；**AGENTS.md** 则把标准下沉到项目层，让代码库第一次以非人类智能体为主要受众。2025 年 12 月三大项目汇聚于 Linux 基金会 AAIF、所有主要厂商共治，确立了其事实标准地位，也印证了智源「Agent 时代 TCP/IP 初具雏形」的判断。

但同一组网络效应也放大了风险：工具投毒、Rug Pull、跨服务器级联与提示注入在 2025 年集中爆发，强模型反而更易被操纵，多个 CVSS 9+ 漏洞已实现真实 RCE。互操作的历史反复证明，**安全与治理不能事后补课**。对研究者与工程团队而言，理解并主动塑造这套协议栈——在拥抱网络效应的同时，把来源认证、最小权限、人类在环与供应链治理内建为默认——正是 2026 年最值得投入的基础设施课题。
---

## 11. 参考文献

> 说明：以下 arXiv 论文编号均通过 arxiv.org 摘要页逐一核实（标题、作者、日期一一对应），作者另对 [2][11][4][14] 四篇关键论文二次抽查无误。日期以 v1 提交日期为主。官方规范、公告与第三方报告单列于后。

### 11.1 MCP 安全漏洞分析、工具投毒与审计

1. **arXiv:2504.03767** — Radosevic, B.; Halloran, J. *MCP Safety Audit: LLMs with the Model Context Protocol Allow Major Security Exploits.* 2025-04-02. https://arxiv.org/abs/2504.03767
2. **arXiv:2503.23278** — Hou, X.; Zhao, Y.; Wang, S.; Wang, H. et al. *Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions.* 2025-03-30. https://arxiv.org/abs/2503.23278
3. **arXiv:2508.13220** — Yang, Y.; Gao, C.; Wu, D.; Li, Y.; Wang, S. et al. *MCPSecBench: A Systematic Security Benchmark and Playground for Testing Model Context Protocols.* 2025-08-17. https://arxiv.org/abs/2508.13220
4. **arXiv:2508.14925** — Wang, Z.; Gao, Y.; Wang, Y.; Sun, H.; Cheng, H. et al. *MCPTox: A Benchmark for Tool Poisoning Attack on Real-World MCP Servers.* 2025-08-19. https://arxiv.org/abs/2508.14925
5. **arXiv:2508.12538** — Guo, Y.; Liu, P.; Ma, W.; Deng, Z.; Xiao, X.; Wen, S. et al. *MCPXKIT: The Unified Toolkit for Analyzing Model Context Protocol Security.* 2025-08-18. https://arxiv.org/abs/2508.12538
6. **arXiv:2508.10991** — Xing, W.; Qi, Z.; Qin, Y.; Han, M. et al. *MCP-Guard: A Multi-Stage Defense-in-Depth Framework for Securing Model Context Protocol in Agentic AI.* 2025-08-14. https://arxiv.org/abs/2508.10991
7. **arXiv:2509.21011** — He, P.; Li, C.; Zhao, B.; Du, T.; Ji, S. *Automatic Red Teaming LLM-based Agents with Model Context Protocol Tools.* 2025-09-25. https://arxiv.org/abs/2509.21011
8. **arXiv:2505.14590** — Jing, H.; Li, H.; Hu, W.; Song, Y. et al. *MCIP: Protecting MCP Safety via Model Contextual Integrity Protocol.* 2025-05-20. https://arxiv.org/abs/2505.14590
9. **arXiv:2505.16957** — Xiong, J.; Zhu, C.; Lin, S.; Zhang, Y.; Liu, Y. et al. *Invisible Prompts, Visible Threats: Malicious Font Injection in External Resources for Large Language Models.* 2025-05-22. https://arxiv.org/abs/2505.16957
10. **arXiv:2601.17549** — Maloyan, N.; Namiot, D. *Breaking the Protocol: Security Analysis of the Model Context Protocol Specification and Prompt Injection Vulnerabilities in Tool-Integrated LLM Agents.* 2026-01-24. https://arxiv.org/abs/2601.17549

### 11.2 MCP 生态测量

11. **arXiv:2509.25292** — Guo, H.; Hao, Y.; Zhang, Y.; Xu, M.; Lv, P.; Chen, J.; Cheng, X. *A Measurement Study of Model Context Protocol Ecosystem.* 2025-09-29. https://arxiv.org/abs/2509.25292

### 11.3 工具集成智能体的间接提示注入（基础工作）

12. **arXiv:2403.02691** — Zhan, Q.; Liang, Z.; Ying, Z.; Kang, D. et al. *InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents.* 2024-03-05. https://arxiv.org/abs/2403.02691
13. **arXiv:2312.14197** — Yi, J.; Xie, Y.; Zhu, B.; Kiciman, E.; Sun, G.; Xie, X.; Wu, F. *Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models.* 2023-12-21（v2 2025-01-27）. https://arxiv.org/abs/2312.14197

### 11.4 多智能体通信与互操作协议

14. **arXiv:2505.02279** — Ehtesham, A.; Singh, A.; Gupta, G. K.; Kumar, S. *A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, and ANP.* 2025-05-04. https://arxiv.org/abs/2505.02279
15. **arXiv:2504.16902** — Habler, I.; Huang, K.; Narajala, V. S.; Kulkarni, P. *Building a Secure Agentic AI Application Leveraging A2A Protocol.* 2025-04-23. https://arxiv.org/abs/2504.16902
16. **arXiv:2506.01804** — Jeong, C. *A Study on the MCP × A2A Framework for Enhancing Interoperability of LLM-based Autonomous Agents.* 2025-06-02. https://arxiv.org/abs/2506.01804
17. **arXiv:2407.07061** — Chen, W.; You, Z.; Li, R.; Guan, Y.; Qian, C.; Zhao, C.; Yang, C.; Xie, R.; Liu, Z.; Sun, M. *Internet of Agents: Weaving a Web of Heterogeneous Agents for Collaborative Intelligence.* 2024-07-09. https://arxiv.org/abs/2407.07061
18. **arXiv:2508.00007** — Chang, G.; Lin, E.; Yuan, C.; Cai, R.; Chen, B.; Xie, X.; Zhang, Y. *Agent Network Protocol Technical White Paper.* 2025-07-18. https://arxiv.org/abs/2508.00007

### 11.5 基于智能体的系统级攻击

19. **arXiv:2507.06850** — Lupinacci, M.; Pironti, F. A.; Blefari, F.; Romeo, F.; Arena, L.; Furfaro, A. *The Dark Side of LLMs: Agent-based Attack Vectors for System-level Compromise.* 2025-07-09. https://arxiv.org/abs/2507.06850

### 11.6 官方规范、公告与权威报告

20. **MCP 官方规范（2025-06-25 / 2025-11-25 版）** — Model Context Protocol. https://modelcontextprotocol.io/specification/2025-06-25 ；https://modelcontextprotocol.io/specification/2025-11-25
21. **MCP 介绍公告** — Anthropic, *Introducing the Model Context Protocol.* 2024-11-25. https://www.anthropic.com/news/model-context-protocol
22. **AAIF 成立官方新闻稿** — The Linux Foundation, *Linux Foundation Announces the Formation of the Agentic AI Foundation (AAIF)…* 2025-12-09. https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation
23. **Anthropic 捐赠 MCP 公告** — Anthropic, *Donating the Model Context Protocol and Establishing the Agentic AI Foundation.* 2025-12-09. https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation
24. **A2A 发布博客** — Google, *A2A — A New Era of Agent Interoperability.* 2025-04-09. https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/
25. **A2A 捐赠 Linux 基金会** — Google, *Google Cloud Donates A2A to Linux Foundation.* 2025-06-23. https://developers.googleblog.com/en/google-cloud-donates-a2a-to-linux-foundation
26. **A2A 官方仓库与 Task 概念** — https://github.com/a2aproject/A2A ；https://agent2agent.info/docs/concepts/task/
27. **AGENTS.md 官方站点** — https://agents.md/
28. **AGNTCY（Cisco Outshift）** — https://outshift.cisco.com/the-internet-of-agents/agntcy ；https://github.com/agntcy
29. **IBM ACP 与 BeeAI（ACP→A2A 迁移）** — https://agentcommunicationprotocol.dev/ ；https://beeai.dev/
30. **新华网：智源发布 2026 十大 AI 技术趋势** — 2026-01-08. http://www.news.cn/tech/20260108/b99127c88a4640bbab49e5c6294264d2/c.html
31. **MCP 安全统计汇编（第三方，含 CVE 与审计来源）** — Practical DevSecOps, *MCP Security Statistics 2026.* 2026-06-26. https://www.practical-devsecops.com/mcp-security-statistics-2026-report
32. **MCP 采用数据复核（第三方）** — Digital Applied, *MCP Adoption Statistics 2026.* 2026-04-20（05-24 更新）. https://www.digitalapplied.com/blog/mcp-adoption-statistics-2026-model-context-protocol
33. **Trend Micro：暴露在公网的 MCP 服务器** — *MCP Security: Network-Exposed Servers Are Backdoors to Your Private Data.* 2025-07.
34. **JFrog：CVE-2025-6514（mcp-remote RCE）** — 2025-07-09.
35. **OWASP MCP Top 10** 与 **CoSAI（OASIS）** — https://owasp.org/ ；https://coalesceai.org/

> 数据可信度说明：第 31–32 项为第三方汇编/估算，其内部数字随扫描方法与采集日期变化，引用时已在正文标注口径与日期；涉及 CVE 的条目以 NVD / 各厂商安全公告为准；所有「厂商口径」「官方口径」数据均未与独立复现混同。
