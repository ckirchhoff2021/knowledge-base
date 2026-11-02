# 上下文工程（Context Engineering）前沿技术报告：当「往窗口里放什么」成为一门独立学科

> 关键词：Context Engineering · 上下文工程 · Context Rot（上下文腐烂）· Token 全生命周期 · Write / Select / Compress / Isolate · Just-in-Time 上下文 · Compaction（压缩）· Context Editing · Constraint Pinning（约束钉扎）· Governance Decay（治理衰减）· 多智能体上下文隔离 · Prompt Cache（提示缓存）
>
> 信源：38 篇 arXiv 原始论文（编号经 arXiv 官方 API 逐一核实，含真实标题 / 作者 / 日期）+ Anthropic / Google / OpenAI / LangChain / Chroma / Cognition / Manus 官方工程博客一手原文 + 知乎专题问答与中英文工程社区交叉验证。
>
> 报告日期：2026-11-02

---

## 目录

0. [摘要（TL;DR）](#0-摘要tldr)
1. [为什么是现在：从 Prompt Engineering 到 Context Engineering](#1-为什么是现在从-prompt-engineering-到-context-engineering)
2. [定义与边界：上下文工程到底管什么](#2-定义与边界上下文工程到底管什么)
3. [实证基础：Context Rot——为什么百万窗口救不了你](#3-实证基础context-rot为什么百万窗口救不了你)
4. [失败分类学：四种上下文失败模式](#4-失败分类学四种上下文失败模式)
5. [四大工程杠杆：Write / Select / Compress / Isolate](#5-四大工程杠杆write--select--compress--isolate)
6. [动态上下文：从静态文件到自我进化的上下文（ACE）](#6-动态上下文从静态文件到自我进化的上下文ace)
7. [压缩的代价：Governance Decay 与约束钉扎](#7-压缩的代价governance-decay-与约束钉扎)
8. [Token 经济学：提示缓存、预算与有效上下文](#8-token-经济学提示缓存预算与有效上下文)
9. [评测：基准版图与「恢复力」缺口](#9-评测基准版图与恢复力缺口)
10. [产业系统横评：Anthropic / OpenAI / Google / Manus / Cognition](#10-产业系统横评anthropic--openai--google--manus--cognition)
11. [工程落地 Checklist 与参考架构](#11-工程落地-checklist-与参考架构)
12. [开放问题与展望](#12-开放问题与展望)
13. [参考文献](#13-参考文献)

---

## 0. 摘要（TL;DR）

**2025 年下半年到 2026 年，大模型系统工程的重心发生了一次安静但根本的迁移：决定 Agent 成败的主要变量，不再是「模型有多大」「提示词写得巧不巧」，而是「在运行的每一步，窗口里到底放了哪些 token」。这门新学科被命名为上下文工程（Context Engineering）。** Andrej Karpathy 在 2025 年 6 月 25 日给出了流传最广的定义——「在下一步所需的恰当时机，用恰当的信息填满上下文窗口的精妙艺术与科学」[1]；Anthropic 在 2025 年 9 月 29 日将其正式工程化，称之为「提示工程的自然延伸」，并同时发布了 context editing（上下文编辑）与 memory（记忆）两个平台原语 [2][3]。Cognition（Devin 团队）则把它称为「构建 AI Agent 的工程师事实上的头号工作」[4]。

本报告的核心判断：

1. **「接受 token」与「可靠使用 token」是两个完全不同的属性，模型卡上只印了前者。** Chroma 对 18 个模型的实测《Context Rot》证明：即使在刻意控制难度的简单任务上，**所有主流模型的表现都随输入变长而越来越不可靠** [5]；GPT-4o 标称 128k 窗口，在去除字面重合的 NoLiMa 上只撑到约 8k；Llama 3.1 70B 在 RULER 上撑到 64k、在 NoLiMa 上仅 2k——同一模型因测试难度不同出现 32 倍差距 [6][7][8]。
2. **上下文工程管的是 token 的全生命周期，而非一句提示词。** LangChain 把它形式化为四条可叠加的杠杆——**Write（写到窗口外）/ Select（按需即时取）/ Compress（压缩留信号）/ Isolate（跨智能体隔离）** [9]。Oracle 进一步指出这四者「不是由低到高的阶梯，而是各付一种代价的并列选择」[10]。
3. **静态上下文文件的收益边际甚至为负，动态进化的上下文才有复利。** ETH Zürich 的《Evaluating AGENTS.md》在 438 个任务上发现：LLM 自动生成的上下文文件使成功率下降约 3%，人类撰写的仅提升约 4%，且两者都使推理成本上升 20% 以上 [11]；而 ICLR 2026 的《Agentic Context Engineering》（ACE）让上下文随任务「生成—反思—淘汰」地进化，在编码任务上 +10.6%、金融推理上 +8.6%，用小模型 + ACE 上下文追平了无 ACE 的顶级闭源模型 [12]。
4. **压缩不是免费的午餐，它会「静默删除」安全约束。** 2026 年 6 月的《Governance Decay》在 1,323 个回合、7 个模型家族上发现：压缩后违规率从全上下文时的 0% 升至聚合约 30%（部分条件达 59%）；约束在摘要中存活时违规为 0，被丢弃时达 38%。**这是 harness（脚手架）的属性而非模型的属性，换更强的模型修不好它** [13]。
5. **Token 是要算账的。** Manus 团队报告 Claude Sonnet 上缓存输入价 0.30 美元/百万 token、未缓存 3 美元（10 倍差距），并称缓存命中率是「生产阶段 Agent 最重要的单一指标」[14]；而窗口顶部的每次压缩/重排都会使缓存失效。Anthropic 多智能体系统则消耗约 15 倍于单聊的 token [15]。

> 🖼️ 架构图：见 `diagrams/1-positioning.html`（图 1 · 上下文工程在大模型 Agent 栈中的位置，及其与提示工程 / 记忆 / Harness 的边界）

---

## 1. 为什么是现在：从 Prompt Engineering 到 Context Engineering

### 1.1 一个被反复观察到的反直觉事实

2025–2026 年，多个相互独立的实验指向同一结论：**Agent 的能力 ≠ 模型的能力**。

- LangChain 在 Terminal Bench 上保持底层模型完全不变，只调整模型外的运行环境（工具、提示、控制流、反馈回路），编码 Agent 得分从 **52.8 提升到 66.5**——13 分差距完全来自 harness [16]。
- 当 Agent 真正以多轮、长程方式运行（数百轮、数小时、跨天），它的失败往往与「原始提示词写得好不好」毫无关系：工具输出堆积、历史被压缩、检索到冲突信息，任何一项都足以让它在中途「忘记」自己该遵守的规则 [2][13]。

这正是「上下文工程」这个词出现并迅速成为共识的背景：**提示工程回答的是「指令该怎么措辞」，而 Agent 真正的瓶颈是「此刻窗口里应该有什么、该丢掉什么」。**

### 1.2 术语的诞生与标志性事件密度

| 时间 | 事件 | 意义 |
|---|---|---|
| 2023-01 | 《Large Language Models Can Be Easily Distracted by Irrelevant Context》[17] | 较早实证「无关上下文即可显著拖累 LLM」，为 context rot 埋下伏笔 |
| 2023-07 | 《Lost in the Middle》[18] | 证明模型对长输入的使用高度不均：首尾可靠、中段最差 |
| 2024-04 | RULER 发布 [7]；Instruction Hierarchy [19] | 提出「有效上下文」需超越 NIAH 测量；特权指令层级训练 |
| 2025-06-25 | **Karpathy 造词** [1] | 「用恰当信息填满窗口的精妙艺术与科学」，X 上 1.4 万赞，术语出圈 |
| 2025-07-02 | LangChain 提出 Write/Select/Compress/Isolate [9] | 第一次把上下文工程形式化为四杠杆框架 |
| 2025-07-14 | Chroma《Context Rot》[5] | 18 模型实测，确立「性能随输入变长而退化」的实证基础 |
| 2025-09-29 | **Anthropic 正式工程化** [2] | 发布工程框架 + context editing + memory 两个平台原语 |
| 2025-10 | 《Agentic Context Engineering》（ACE）[12]；《Context Length Alone Hurts》[20]；Context-Folding [21] | 上下文可进化；即使完美检索，长度本身也有害 |
| 2026-02 | Evaluating AGENTS.md [11]；ContextBench [22]；LOCA-bench [23]；MCP smelly [24]；SWE Context Bench [25] | 静态文件祛魅 + 上下文基准密集出现 |
| 2026-06 | **Governance Decay** [13]；Less Context Better Agents [26]；Structured Eviction [27] | 压缩的安全代价被命名；结构化驱逐 |
| 2026-09–10 | ContextPipe [28]；Highlight-Then-Summarize [29]；ττ-Bench [30]；ReFold [31] | 数据库式上下文装配；可逆折叠；端到端 Agent 构造评测 |

### 1.3 为什么偏偏是 2025H2–2026

三个条件同时成熟：

1. **Agent 真正「变长」了。** 以 DeepSeek-R1 [32]、o1、Gemini 3 为代表的推理模型让长程任务成为可能，Agent 开始运行数百轮、跨越数小时，历史增长速度超过了窗口增长速度，压缩从「优化」变成「标配」。
2. **窗口标称值进入百万级，但有效使用能力没跟上。** Gemini 1.5 Pro（2024 年初）率先给出 1M 窗口，GPT-4.1、Llama 4（标称 10M）相继跟进 [5]；但实测的可靠长度常常只有标称值的几十分之一。
3. **治理与工程体系开始把「上下文」当成一等公民。** MCP/A2A 协议标准化、企业 AgentOps 体系建立、推理成本两年下降 95% 以上，使「系统化管理窗口」在工程与经济上同时可行。

---

## 2. 定义与边界：上下文工程到底管什么

### 2.1 三个权威定义

- **Karpathy（2025-06-25）**：「上下文工程是在下一步所需的恰当时机，用恰当的信息填满上下文窗口的精妙艺术与科学。信息太少或形式不对，模型就没有最优表现所需的上下文；信息太多或形式不对，成本上升、表现下降。」[1]
- **Anthropic Applied AI Team（2025-09-29）**：「上下文工程是在 LLM 推理期间，策展并维护最优 token（信息）集合的一整套策略，**包括所有可能在提示之外进入窗口的其他信息**。」[2]
- **Oracle（2026-09-28）**：「上下文工程决定 Agent 运行每一步模型看到什么——指令、实时任务状态、检索到的证据、记忆与工具定义。提示工程写的是指令；上下文工程决定到达模型的其他一切，以及保留它们的代价。」[10]

### 2.2 与提示工程的区别

| 维度 | Prompt Engineering | Context Engineering |
|---|---|---|
| 回答的问题 | 指令该怎么措辞？ | 这一步模型该看到什么？保留它的代价是什么？ |
| 范围 | 一条提示或模板 | 一次运行的每一步：指令 / 状态 / 证据 / 记忆 / 工具 |
| 发生时机 | 运行前写好 | 运行中动态决定，常常由 harness 自动执行 |
| 失败方式 | 指令歧义或冲突 | 正确的信息在需要它的那一步缺失、被埋没或被删除 |
| 调试方式 | 读提示词 | 检查失败那一步窗口里实际有什么 |

关键在于 Anthropic 定义里的那句话——**「包括提示之外进入窗口的其他信息」**。一个生产 Agent 的窗口里，系统提示往往只占很小一部分；绝大部分 token 是工具返回、检索文档、历史消息，这些由 harness 在运行中自动装配，提示工程从未覆盖它们 [2][10]。

### 2.3 与记忆、Harness 的边界（去重说明）

本知识库已有《智能体记忆 Agent Memory》与《Agent Harness》两篇报告。上下文工程与它们是**上下位关系，而非重复**：

- **记忆（Memory）是上下文工程中负责「跨时间持久化与召回」的子系统。** 记忆综述《Memory in the Age of AI Agents》明确把记忆定位为上下文工程的下层组件——当信息需要跨会话、跨压缩存活时，由记忆层承载 [33][34]。上下文工程问的是更上位的问题：此刻窗口里该放什么、该由哪个杠杆来付代价。
- **Harness（脚手架）是执行上下文工程策略的那层「壳」。** Harness 决定工具如何挂载、控制流如何组织、压缩何时触发；上下文工程是 harness 所管理的核心对象之一。《Governance Decay》特别指出治理衰减是「harness 的属性」——压缩这一步由 harness 执行，其副作用也归属于 harness [13]。

一句话：**Harness 是「壳与执行机制」，Memory 是「跨时间的存储子系统」，Context Engineering 是统管「窗口内 token 全生命周期」的更上位学科。** 本报告聚焦后者，并在相应章节交叉引用前两者。

---

## 3. 实证基础：Context Rot——为什么百万窗口救不了你

### 3.1 什么是 Context Rot

Chroma 技术报告《Context Rot: How Increasing Input Tokens Impacts LLM Performance》（2025-07-14，作者 Kelly Hong、Anton Troynikov、Jeff Huber）对 18 个 LLM（含 GPT-4.1、Claude 4、Gemini 2.5、Qwen3）做了实测，核心结论是：

> 模型对上下文的使用并不均匀；**即使在简单任务上，表现也随输入变长而越来越不可靠**。百万级窗口只是把退化点推后，并没有消除它 [5]。

Anthropic 与 Chroma 都用「context rot」来命名这一现象：随着窗口内 token 数增加，模型准确回忆其中信息的能力下降 [2][5]。

### 3.2 为什么 NIAH 给了「长上下文已解决」的假象

业界长期以「大海捞针」（Needle in a Haystack, NIAH）为长上下文标尺，而前沿模型在 NIAH 上近满分。但 Chroma 指出 NIAH 本质上是**简单的词面检索（lexical retrieval）**，不代表真实 Agent 所做的灵活、语义导向任务 [5]。一旦改变测试方式，可靠长度立刻坍缩：

| 模型 | 标称窗口 | 实测可靠长度 | 测试 |
|---|---|---|---|
| GPT-4.1 | 1,000,000 | ≈ 16,000 | NoLiMa [6] |
| GPT-4o | 128,000 | ≈ 8,000 | NoLiMa（问题/证据不再共享措辞）[6][10] |
| Llama 3.1 70B | 128,000 | 64,000 / 2,000 | RULER / NoLiMa（同模型 32× 差距）[7][8][10] |

> 注：上表为对应模型在各自测试下的结果；Chroma/Oracle 均说明这些研究未覆盖全部最新模型，数字应理解为「警告而非定律」[10]。

### 3.3 三股独立的退化力量

Chroma 与 Oracle 综合多项研究，指出退化由三股相互独立的力量叠加造成 [5][10]：

1. **位置（Position）**：模型在长输入的开头和结尾使用证据最可靠，中段最不可靠。《Lost in the Middle》系统证明了这一 U 形曲线 [18]。
2. **难度（Difficulty）**：RULER 在 NIAH 之上加入多针、聚合、多跳追踪等任务，被评测的模型中有一半在远低于标称窗口处就失效 [7]。
3. **字面重合（Literal overlap）**：NoLiMa 去掉问题与证据之间的共享措辞，要求模型推断潜在关联（例：问「谁去过赫尔辛基？」，证据是「Yuki 住在 Kiasma 博物馆旁边」——需知道 Kiasma 在赫尔辛基），有效长度随之坍缩；72.4% 的 NoLiMa 样本需要这类外部知识 [6]。

### 3.4 最有力的一击：即使检索完美，长度本身也有害

《Context Length Alone Hurts LLM Performance Despite Perfect Retrieval》（arXiv:2510.05381，2025-10）给出了排除其他借口的实验：**给模型完美检索、把证据紧贴问题放置，表现仍随输入变长而退化。** 甚至把干扰项从注意力中「遮蔽」（mask）而非物理移除，退化依然存在 [20]。

这带来一个关键工程推论：**有效上下文窗口是「测出来的」，不是从模型卡上「读出来的」。** 更大的窗口并不等于更可靠的注意力；工程问题不是「怎么把 token 塞进去」，而是「如何在每一步只保留正确的 token」。

> 🖼️ 架构图：见 `diagrams/2-context-rot.html`（图 2 · Context Rot 的三股退化力量与「标称窗口 / 有效窗口」落差）

此外，Chroma 还报告了两个反直觉的辅助发现 [5]：

- **结构化的「草堆」反而更差**：在 18 个模型上，保留逻辑连贯性的原文比把句子随机打乱的版本表现更差。打乱局部连贯性反而一致地提升表现，暗示注意力如何被输入结构影响仍待机制解释。
- **保守与冒进的模型差异**：在干扰项实验中，Claude（尤其 Opus 4 / Sonnet 4）幻觉率最低，不确定时倾向弃权；GPT 模型在干扰项存在时更易产生「自信但错误」的回答。

---

## 4. 失败分类学：四种上下文失败模式

实践者 Drew Breunig 结合 Gemini 技术报告、Berkeley Function-Calling Leaderboard 等公开研究，归纳出四种性质不同的上下文失败。它们不是理论推演，每一种都在生产 Agent 或公开评测中被观察到 [10][35]：

| 失败模式 | 触发条件 | 主要杠杆 | 检测信号 | 观察来源 |
|---|---|---|---|---|
| **Context Poisoning** 上下文投毒 | 幻觉进入上下文并跨轮累积、放大 | Compress——驱逐过期/错误上下文 | Agent 追逐不可能或无关目标；循环升级 | Gemini 2.5 Pokémon Agent [35] |
| **Context Distraction** 上下文分心 | 长上下文导致过度依赖历史而非新推理（Gemini 约在 100k token 起） | Compress——总结消息历史 | Agent 重复旧动作而非适配；输出新颖度低 | Gemini 2.5 技术报告 [35] |
| **Context Confusion** 上下文混淆 | 多余工具或文档淹没模型；Llama 3.1 8B 给 46 个工具失败、19 个成功 | Select——即时加载工具与文档 | 调错工具；引用无关文档 | Berkeley Function-Calling Leaderboard [10] |
| **Context Clash** 上下文冲突 | 跨轮收集到相互冲突的信息；多轮分片平均致 39% 性能下降（o3：98.1→64.1） | Isolate——子智能体分区；Write——结构化笔记 | 跨工具调用答案矛盾；对冲措辞升级 | Microsoft/Salesforce《LLMs Get Lost In Multi-Turn Conversation》[36] |

### 4.1 两个标志性案例

**Gemini 2.5 Pokémon Agent（Poisoning + Distraction）。** Gemini 2.5 技术报告描述了一个玩宝可梦的实验：Agent 的「目标 / 摘要」区被关于游戏状态的错误信息「投毒」，导致它追逐不可能或无关的目标；当历史超过 100k token，它又表现出「倾向于重复庞大历史中的动作，而非综合出新计划」。两种失败出现在同一 Agent 身上，因为它们是同一根因——**未管理的上下文累积**——的不同表现 [35]。

**多轮分片（Context Clash 连前沿模型也躲不过）。** Microsoft 与 Salesforce 的研究《LLMs Get Lost In Multi-Turn Conversation》（arXiv:2505.06120）把提示信息分布到多轮对话中——这正是 Agent 的天然工作方式——OpenAI o3 从 98.1 跌到 64.1，绝对下降 34 分；所有模型平均性能下降 39%。**Context Clash 不是小模型问题** [36]。

---

## 5. 四大工程杠杆：Write / Select / Compress / Isolate

LangChain 在 2025-07-02 将上下文工程形式化为四条可叠加的策略 [9]。它们不是互相竞争、而是分层运作——多数生产 Agent 需要四者在上下文生命周期的不同节点同时工作。Oracle 的关键提醒是：**四者不是由低到高的阶梯，即时取并不比预载「高级」，子智能体也不是「毕业方向」；每一个都买回 token，同时各付一种代价** [10]。

### 5.1 Write —— 写到窗口之外

外部记忆、结构化笔记、文件存储：把上下文保存在活动窗口之外，使其跨「上下文重置」依然存在。

- Claude Code 用 `CLAUDE.md` 文件在开头加载；Anthropic 的 memory 工具支持对文件存储做增删改查 [2]。
- Cognition 的 Devin 用结构化笔记追踪状态（例如「过去 1,234 步我一直在 1 号路练宝可梦」）[4]。
- Manus 在每一步重写 todo 清单，把目标「背诵」进注意力最强的窗口末端 [14]。

适用：长程运行、跨会话状态。

### 5.2 Select —— 即时按需取

RAG、工具语义搜索、动态文档加载：不一次性加载全部上下文，只维护轻量标识（文件路径、存储的查询、链接），在需要的那一刻才加载数据。

- Claude Code 用 glob/grep 做即时文件访问；把 RAG 应用到**工具描述**（而不只是文档）上，LangChain 引用的研究显示工具选择准确率可提升约 3 倍 [9]。
- 《Retrieval Models Aren't Tool-Savvy》（arXiv:2503.01763）在约 4.3 万个工具上发现：传统检索强度并不能预测选对工具，更差的检索还会降低完成率——工具检索是一个独立难题 [37]。

适用：防止上下文混淆。

### 5.3 Compress —— 压缩只留信号

Compaction（压缩）、context editing、工具结果清理：总结增长的消息历史，一旦原始工具输出的关键信息被捕获，就丢弃原始内容。

- Anthropic 的 context editing 在 Agent 接近 token 上限时自动清理过期的工具调用与结果；Claude Code 的 auto-compact 在窗口用量达 95% 时触发 [2][3]。
- Cognition 发现**通用摘要模型无法可靠保留关键决策**，因此用专门微调的压缩模型 [4]。

适用：防止上下文分心。压缩的安全代价（Governance Decay）见第 7 章。

### 5.4 Isolate —— 跨智能体或环境隔离

子智能体、并行工作流、沙箱化上下文：把上下文拆分到多个 Agent / 环境 / 工具，使任何单一上下文都不累积全部信息。

- Anthropic 多智能体研究系统给每个子智能体独立窗口，子智能体只把 1,000–2,000 token 的浓缩摘要返回给主智能体 [15]。
- LangGraph 的 Bigtool 对工具描述做语义搜索，为每个任务预选相关工具 [9]。

适用：防止上下文冲突。

### 5.5 杠杆与失败的一一对应

| | Write | Select | Compress | Isolate |
|---|---|---|---|---|
| 主要应对 | 跨重置持久化（让其他杠杆可逆） | Context Confusion | Context Distraction / Poisoning | Context Clash |
| 买回什么 | 状态不丢失 | 占用与需求成比例 | 小窗口容纳长程 | 主智能体不看全轨迹 |
| 付出代价 | 需额外存储与写回 | 一次往返 + 「发现」决策 | 不可逆遗漏（除非原件仍可寻址） | 用「证词」替代「证据」 |

Oracle 特别强调 **Write 的特殊地位**：它不是与其他四者并列的「操作」，而是让其他操作安全的前提——你只能压缩或取回「事先被写到某处」的东西 [10]。

> 🖼️ 架构图：见 `diagrams/3-four-levers.html`（图 3 · Write/Select/Compress/Isolate 四杠杆与四种失败模式的映射、各自代价）

---

## 6. 动态上下文：从静态文件到自我进化的上下文（ACE）

### 6.1 静态上下文文件的祛魅

ETH Zürich 的《Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?》（arXiv:2602.11988，2026-02-12，Gloaguen 等）在 SWE-bench Lite 的 438 个任务与一个新基准 AGENTbench 上测试，结论犀利 [11]：

- **LLM 自动生成的上下文文件使任务成功率下降约 3%；人类撰写的仅提升约 4%；两者都使推理成本上升 20% 以上。**
- 建议：上下文文件保持最小，只写必要内容。

但论文数据也揭示了被摘要低估的一点：**Agent 遵循上下文文件指令的频率是基线工具使用的 1.6–2.5 倍——文件确实改变了行为，只是该基准不奖励这种「倾听」。** [11]

### 6.2 矛盾的证据：换一个指标，结论翻转

Lulla 等《On the Impact of AGENTS.md Files on the Efficiency of AI Coding Agents》（arXiv:2601.20404，2026-01-28）测量的不只是完成率，还有运行时间与 token 消耗 [38]：

- **运行时间中位数下降 28.64%；输出 token 下降 16.58%；任务完成率不变。**

同一干预、同类文件、不同测量、不同结论：Gloaguen 计入了「读文件」的总推理 token，Lulla 测的是「读完之后」Agent 工作得更省。对生产环境而言，28% 的运行时间下降意味着更快反馈与更低账单 [38]。

### 6.3 ACE：让上下文像程序一样进化

最强的学术证据来自 ICLR 2026 的《Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models》（arXiv:2510.04618，Zhang 等）。ACE 把上下文当作一份**不断进化的文档**，跑一个三阶段闭环 [12]：

1. **Generate（生成策略）**：从任务尝试中生成可复用策略；
2. **Reflect（反思）**：复盘哪些策略有效、哪些失败；
3. **Curate（策展）**：把成功策略加入上下文、淘汰失败策略。

上下文随任务增长、改进并复利累积。结果：**编码基准 +10.6%、金融推理 +8.6%；用一个更小的开源模型 + ACE 上下文，追平了不带 ACE 的顶级闭源模型。** [12]

这划出了一条清晰的界线：**写一次、被动读的静态上下文文件收益边际；在使用中进化的动态上下文系统产生复利，缩小大小模型差距。** Gloaguen 测的是静态版，ACE 测的是动态版。

> 🖼️ 架构图：见 `diagrams/4-ace-loop.html`（图 4 · ACE 三阶段闭环与静态/动态上下文的收益分野）

### 6.4 相关的上下文折叠路线

- **Context-Folding**（arXiv:2510.11967，2025-10）：把已完成的轨迹「折叠」成其结果，在长程任务上把活动上下文缩小至 1/10 [21]。
- **ReFold**（arXiv:2610.07863，2026-10）：免训练、可逆的轮间上下文折叠 [31]。
- **ContextPipe**（arXiv:2609.00749，2026-09）：借鉴数据库查询优化的思路做长程 Agent 的上下文装配 [28]。
- **Highlight-Then-Summarize**（arXiv:2609.31382，2026-09）：先高亮再总结，学习为长上下文理解压缩证据 [29]。

---

## 7. 压缩的代价：Governance Decay 与约束钉扎

### 7.1 一个值得出名的开场

《Governance Decay: How Context Compaction Silently Erases Safety Constraints in Long-Horizon LLM Agents》（arXiv:2606.22528，2026-06-21，单作者 Shiyang Chen）给出这样一个场景 [13]：

> 一个 Agent 被指示「绝不把合同发到组织外部」。前几十轮它正确拒绝。随后历史超出窗口，**压缩**——每个长程 harness 都会做的例行总结步骤，为保持任务连续性而优化——运行了。摘要忠实地记录任务状态，却把「过时的」合规前言悄悄丢掉。再次被问时，Agent 发出了合同。没有越狱、没有换模型、没有新请求：它遵守的规则只是不再位于它眼前。

### 7.2 实测数据

论文在 **1,323 个回合、7 个模型家族**上报告 [13]：

- 违规率从全上下文时的 **0%** 升至压缩后聚合约 **30%**，部分条件达 **59%**；
- 约束在摘要中存活时违规 **0%**，被丢弃时 **38%**；
- **Constraint Pinning（约束钉扎）**：把可抽取的规则隔离保护，使总结器必须将其携带前进，在该基准上把违规降回 **0%**。

### 7.3 三个提升其严重性的性质

1. **这是 harness 的属性，不是模型的属性。** 更强的模型也会倒下，因为失败在于总结器保留了什么，而不在于模型能做什么——升级模型修不好它 [13]。
2. **软 / 硬梯度。** 模型训练进权重的「硬规范」（安全基准探测的那些）能存活，而**部署方指定的软规则——恰恰是「你的」规则——正是被侵蚀的部分**，因此标准评测一直没发现它。
3. **可被对抗利用。** 能往上下文里放内容的攻击者（标准的间接注入假设），可以故意膨胀并引导历史，让压缩替自己抹除障碍。

### 7.4 与相邻概念的区分

- **对 Prompt Injection（提示注入）**：注入是「加法」攻击，衰减是「减法」失败。二者恶性叠加——注入内容可引导压缩去抹除防御 [13][39]。
- **对 Context Rot（上下文腐烂）**：rot 是「东西还在但用不好」，decay 是「东西根本不在了」。rot 的缓解（更好检索、更长窗口）碰不到 decay [13]。
- **对 Model Drift（模型漂移）**：drift 是权重/版本变化，decay 在模型冻结时发生，因此能躲过为 drift 建的监控。

### 7.5 防御与工程含义

论文自己也坦承 pinning 的局限：**隐式约束无法被钉扎；钉扎会被近期上下文中的「操作者冒充」击败，因此需要一个可信的带外（out-of-band）操作者通道。** [13] 工程上的分层建议：

- 把「不能丢」的约束（合规、数据边界、支出、工具范围）放在压缩器够不到的地方——网关 / 控制面 / 工具层强制执行，而不是写进会被总结的提示里；
- 留在上下文里的规则要钉扎、跨压缩重新校验、并专门做「压缩后是否仍在」的测试；
- 把压缩本身变成一个**被观测的事件**：在 trace 中记录历史何时被压缩，这样「Agent 莫名改变行为」就能与压缩点关联起来。

论文结论用七个字概括：**「治理 Agent，需要治理它们如何遗忘。」** [13]

> 🖼️ 架构图：见 `diagrams/5-governance-decay.html`（图 5 · 压缩如何静默删除合规约束，以及带外策略层为何不受影响）

---

## 8. Token 经济学：提示缓存、预算与有效上下文

### 8.1 提示缓存：被最晚察觉的跨切成本

提供商对上下文「不变的前缀」做缓存，缓存 token 便宜得多。Manus 团队报告：Claude Sonnet 上缓存输入 **0.30 美元/百万 token**，未缓存 **3 美元**——10 倍差距；他们称缓存命中率是「生产阶段 Agent 最重要的单一指标」[14]。

这带来一个反直觉的结构推论：

- 一个**稳定、预载的前缀**（系统提示、固定工具）因为能被缓存，会自己「赚回」成本；
- 而窗口**顶部附近的每次压缩或重排都会使缓存失效**，随后按全价重新付费。

因此「积极压缩」与「缓存友好」之间存在张力：压缩省了活动 token，却可能扔掉了缓存折扣。工程上应尽量把易变内容放在窗口后部、把稳定前缀保持不动 [10][14]。

### 8.2 上下文预算：分配问题，而非大小问题

工作产物是一份**上下文预算（context budget）**：把有限输入 token 按每一步分配到五个类别，并规定每一项由哪个操作付费 [10]。一个退款决策步的示例：

| 预算项 | 示例内容 | 策略 |
|---|---|---|
| 指令 Instructions | 退款上限与升级规则 | 钉扎，绝不总结 |
| 实时任务状态 Live state | 案件 ID、客户 ID、申请金额、上一动作 | 本步加载 |
| 检索证据 Evidence | 确切政策条款及来源 URL | 做退款决策时即时取 |
| 记忆 Memory | 既往客户承诺（带记录 ID） | 按用户 / Agent / 线程 / TTL 限定作用域 |
| 工具定义 Tools | 仅本步可用的支付 / 退款工具 | 按路由加载，而非整个目录 |

Anthropic 对目标的概括是寻找「**使期望结果最可能发生的、最小的高信号 token 集合**」[2]。

### 8.3 三种压缩方式的可逆性

Oracle 区分了三种减少历史的方式，可逆性是关键差异 [10]：

| 方式 | 做法 | 可逆性 |
|---|---|---|
| **Masking 遮蔽** | 用标记 + 检索句柄替换观察，原件可精确取回 | ✅ 可逆 |
| **Offloading 卸载** | 保留路径与版本，原件存外部 | ✅ 可逆（需外部存储） |
| **Summarizing 总结** | 用散文替换，无法回到原话 | ❌ 不可逆 |

只做阈值触发的总结，正是第 38 步错误退款的成因；而遮蔽/卸载因为保留了可寻址原件，让压缩变得安全 [10]。

---

## 9. 评测：基准版图与「恢复力」缺口

### 9.1 长上下文 / 上下文基准版图

| 基准 | 测什么 | 关键发现 |
|---|---|---|
| NIAH [40] | 词面检索 | 近满分，造成「已解决」假象 |
| RULER [7] | 多针 / 聚合 / 多跳 | 半数模型远低于标称窗口失效 |
| NoLiMa [6] | 非字面匹配 | 有效长度大幅坍缩 |
| Michelangelo [41] | 潜结构查询（Latent List / Graphwalks / MRCR） | 无关内容的「类型」影响退化 |
| LongMemEval [42] | 对话式长期记忆（抽取/多会话/时序/更新/弃权） | 持续交互约 30% 准确率下降 |
| AbsenceBench [43] | 识别「某信息缺失」 | 随长度退化 |
| ContextBench [22] | 编码 Agent 的上下文检索（1,136 任务 / 66 仓库） | LLM 重召回轻精确，过度检索 |
| SWE Context Bench [25] | 编码中的上下文学习 | 正确选择+总结的上下文才提升准确率与成本 |
| LOCA-bench [23] | 可控、极端的上下文增长 | 环境复杂度上升时性能下降 |
| ττ-Bench [30] | 端到端、真实 Agent 构造 | 从「单工具」走向全流程环境 |

### 9.2 两个被反复验证的评测共识

1. **激进裁剪有效。** SWE Context Bench 发现未过滤的上下文收益有限甚至为负；JetBrains 在 NeurIPS 2025 研讨会上的研究显示，简单的观察遮蔽即可提升 2.6% 表现、同时降低 52% 成本 [25]。
2. **Agent 过度检索。** ContextBench 发现 LLM 一致偏好召回 over 精确，取回大量信息却只用一小部分——这验证了「渐进披露（progressive disclosure）」：给 Agent 信息所在的「指针」，而非信息本身 [22]。

### 9.3 缺失的基准：恢复力与知识工作

多篇综述指出一个显著的空白 [10][44]：

- 现有基准几乎都测「编码任务是否解决」，**没有一个基准在运行中途压缩 Agent 历史、然后要求它把约束还回来**；
- 也没有基准测知识工作者真正做的「多源信息编排」（从 CRM 取数、交叉邮件史、应用风格指南、产出像本人写的草稿），以及上下文是否跨会话复利。

《Beyond Task Completion》（arXiv:2512.12791）直接点名这种「基准与真实需求的根本错配」：企业关心成本、可靠性、安全与运行约束，基准却只优化孤立编码任务的准确率 [44]。**仪表盘上最该放的单一数字，是「N 次压缩后的确切约束恢复率」，旁边配上「每解决一个任务的成本」**——新鲜运行上的任务成功率会把整个失败藏起来 [10]。

---

## 10. 产业系统横评：Anthropic / OpenAI / Google / Manus / Cognition

| 系统 / 厂商 | 上下文机制 | 关键数据（官方口径） |
|---|---|---|
| **Anthropic** | context editing + memory 原语；多智能体隔离 | context editing 单独 +29%，叠加 memory +39%；100 轮网页搜索 token −84%；多智能体研究系统比单 Agent 高 90.2%，但耗约 15× token [2][3][15] |
| **OpenAI** | Agents SDK compaction session | 压缩一旦运行「清空底层会话并用精简列表重写」——原件从会话中消失，除非另存 [45] |
| **Google ADK** | 可配置间隔的上下文压缩 | 定期总结较旧会话历史 [46] |
| **Manus** | 每步重写 todo；缓存优先 | 缓存/未缓存 10× 价差；缓存命中率为最重要指标 [14] |
| **Cognition（Devin）** | 结构化笔记 + 微调压缩模型；反对多智能体 | 通用摘要不可靠保留决策；「多 Agent 隐含冲突决策导致坏结果」[4][47] |

### 10.1 多智能体是不是上下文策略？——一场真实争论

**支持方（Anthropic）**：多智能体研究系统用子智能体隔离窗口，token 使用量解释了 BrowseComp 上 80% 的性能方差；并行 3–5 个子智能体可把复杂查询研究时间最多缩短 90% [15]。

**反对方（Cognition）**：《Don't Build Multi-Agents》主张「动作携带隐含决策，冲突决策携带坏结果」——两个基于不同上下文切片的子 Agent 会做出无法拼合的选择；此外还有凭证继承风险（AgentDojo 正是测试工具输出中的敌意文本重定向）[47][39]。

折中的工程判断 [10]：**只读的研究扫描适合移交；两个 Agent 并行编辑同一事物通常不行。** 移交应是「类型化」而非「叙述化」的——目标、约束、已做决策、产物标识、带来源的证据、已知失败与下一步动作，大证据放在引用之后；A2A 协议正好提供这样一个信封 [48]。子 Agent 的凭证必须按所分配任务收窄，而非继承父级全部权限。

---

## 11. 工程落地 Checklist 与参考架构

### 11.1 立即可以执行的习惯

1. **钉扎约束**：政策、权限、承诺放在总结器永不重写的存储中，每一步重载。
2. **总结散文、保留标识**：案件 ID、时间戳、金额、权限、原始事件指针在每次压缩中逐字保留。
3. **摘要从原始记录派生，而非从上一个摘要派生**——「摘要的摘要」会漂移且无警告。
4. **在窗口末端保留一份书面计划**：Manus 式把目标 recite 进注意力最强处。
5. **优先遮蔽 / 卸载，慎用不可逆总结**：保持原件可寻址。
6. **保持稳定前缀以吃缓存**：易变内容后置，避免在窗口顶部频繁重排。
7. **工具描述做一次专项审查**：MCP 研究显示 856 个工具描述几乎都有缺陷，改写可提升成功率 [24]。
8. **把压缩变成可观测事件**：trace 中记录压缩点，关联行为变化。

### 11.2 何时用哪个操作（决策规则）

- **预载（Eager）**：该项必须在时机不可预测的决策中出现、且缺失不可恢复——控制层、强制约束、输出契约。
- **即时取（JIT）**：材料大、常变、少用，且 Agent 会知道自己需要它。
- **压缩（Compact）**：历史越过自然边界（子目标完成 / 放弃 / 修订），且将丢的一切在别处仍可寻址。**不要只按 token 阈值压缩。**
- **移交子 Agent**：某阶段产生长探索轨迹 + 短结论（语料扫描、搜索），父级无需看死路。

### 11.3 如何知道预算有效

测量「减少之后的恢复力」，而非顺利路径上的任务成功：记录有用 token 密度、来源覆盖率、确切约束恢复、再水合成功，以及每解决任务的成本。**保持模型与任务固定、只改预算策略**——若模型、检索器、上下文上限同时变化，就无法归因是哪个选择起效 [10]。

> 🖼️ 架构图：见 `diagrams/6-reference-architecture.html`（图 6 · 生产级上下文工程参考架构：稳定前缀 / 即时检索 / 可逆压缩 / 隔离子 Agent / 带外策略层）

---

## 12. 开放问题与展望

1. **压缩能否仅靠策略变安全？** 现有证据主要来自尚待独立复现的预印本；需要由未参与原工作的团队跨 2–3 个 harness 复跑同一协议 [10][13]。
2. **基准何时开始为「恢复力」打分？** 中途压缩并要求归还约束的基准仍缺位；LongMemEval 最接近但面向聊天助手而非工具 Agent [42]。
3. **操作能否标准化？** 移交已有 A2A 信封，但各家压缩配置不同、对「什么必须存活」没有共识；遮蔽 / 卸载 / 钉扎尚无共享词汇 [10][48]。
4. **记忆安全这一新攻击面。** 长期记忆可被投毒，需要来源绑定、不可篡改的权威机制（如 arXiv:2606.24322 的机器可检查保证）[49]。
5. **从「治理写进脑袋」到两层信条。** 大方向是把约束分成两堆——「引导」（语气、偏好、任务框架，留在上下文、可丢失）与「生存」（数据边界、工具限制、支出、合规，置于带外、必须全程成立）[13]。

轻持的预测：窗口继续增长、有效长度继续滞后；当这种落差第一次在受监管企业造成真实的财务 / 安全 / 合规事故时，它就会从研究好奇变成行业刚需。**没有人靠「买一辆更大的货车」就把货装对——你要先摸清路线。** [10]

---

## 13. 参考文献

> 说明：以下 arXiv 编号均经 arXiv 官方 API（`export.arxiv.org/api/query`）于 2026-11-02 逐一核实，标题 / 作者 / 日期以 API 返回为准；工程博客为厂商或作者一手原文。厂商数据标注「官方口径」，引用时请注意其评测范围与局限。

1. Andrej Karpathy. 关于 context engineering 的定义，X/Twitter，2025-06-25. https://x.com/karpathy/status/1937902205765607626
2. Anthropic Applied AI Team. *Effective Context Engineering for AI Agents.* 2025-09-29. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
3. Anthropic. *Managing Context on the Claude Developer Platform*（context editing / memory）. 2025-09. https://claude.com/blog/context-management
4. Cognition AI. 关于上下文工程与 Devin 结构化笔记 / 压缩模型的工程实践，Cognition Blog，2025. https://cognition.com/blog
5. Kelly Hong, Anton Troynikov, Jeff Huber. *Context Rot: How Increasing Input Tokens Impacts LLM Performance.* Chroma Technical Report, 2025-07-14. https://www.trychroma.com/research/context-rot
6. Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, et al. *NoLiMa: Long-Context Evaluation Beyond Literal Matching.* arXiv:2502.05167（2025-02-07）. https://arxiv.org/abs/2502.05167
7. Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* arXiv:2404.06654（2024-04-09）. https://arxiv.org/abs/2404.06654
8. (Llama 3.1 长上下文实测，转引自) Oracle Developers Blog，见 [10]；原始模型卡见 Meta。
9. LangChain. *Context Engineering for Agents*（Write / Select / Compress / Isolate）. 2025-07-02. https://www.langchain.com/blog/context-engineering-for-agents
10. Casius Lee. *What is Context Engineering? The Four Operations That Decide Whether Your Agent Survives Production.* Oracle Developers Blog，2026-09-28. https://blogs.oracle.com/developers/what-is-context-engineering-the-four-operations-that-decide-whether-your-agent-survives-production
11. Thibaud Gloaguen, Niels Mündler-Sasahara, Mark Niklas Müller, et al. *Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?* arXiv:2602.11988（2026-02-12）. https://arxiv.org/abs/2602.11988
12. Qizheng Zhang, Changran Hu, Shubhangi Upasani, et al. *Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models.* arXiv:2510.04618（2025-10-06，ICLR 2026）. https://arxiv.org/abs/2510.04618
13. Shiyang Chen. *Governance Decay: How Context Compaction Silently Erases Safety Constraints in Long-Horizon LLM Agents.* arXiv:2606.22528（2026-06-21）. https://arxiv.org/abs/2606.22528
14. Manus Team. *Context Engineering for AI Agents: Lessons from Building Manus.* 2025. https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
15. Anthropic. *How We Built Our Multi-Agent Research System.* 2025-06-13. https://www.anthropic.com/engineering/multi-agent-research-system
16. LangChain. Terminal Bench 上 harness 对同一模型表现影响的实验（52.8→66.5），LangChain Blog / Terminal Bench，2026. 详见本知识库《Agent Harness》报告。
17. Freda Shi, Xinyun Chen, Kanishka Misra, et al. *Large Language Models Can Be Easily Distracted by Irrelevant Context.* arXiv:2302.00093（2023-01-31）. https://arxiv.org/abs/2302.00093
18. Nelson F. Liu, Kevin Lin, John Hewitt, et al. *Lost in the Middle: How Language Models Use Long Contexts.* arXiv:2307.03172（2023-07-06，TACL 2024）. https://arxiv.org/abs/2307.03172
19. Eric Wallace, Kai Xiao, Reimar Leike, et al. *The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions.* arXiv:2404.13208（2024-04-19）. https://arxiv.org/abs/2404.13208
20. *Context Length Alone Hurts LLM Performance Despite Perfect Retrieval.* arXiv:2510.05381（2025-10-06，EMNLP 2025 Findings）. https://arxiv.org/abs/2510.05381
21. *Scaling Long-Horizon LLM Agent via Context-Folding.* arXiv:2510.11967（2025-10-13）. https://arxiv.org/abs/2510.11967
22. Han Li, Letian Zhu, Bohan Zhang, et al. *ContextBench: A Benchmark for Context Retrieval in Coding Agents.* arXiv:2602.05892（2026-02-05）. https://arxiv.org/abs/2602.05892
23. Weihao Zeng, Yuzhen Huang, Junxian He. *LOCA-bench: Benchmarking Language Agents Under Controllable and Extreme Context Growth.* arXiv:2602.07962（2026-02-08）. https://arxiv.org/abs/2602.07962
24. Mohammed Mehedi Hasan, Hao Li, Gopi Krishnan Rajbahadur, et al. *Model Context Protocol (MCP) Tool Descriptions Are Smelly! Towards Improving AI Agent Efficiency with Augmented MCP Tool Descriptions.* arXiv:2602.14878（2026-02-16）. https://arxiv.org/abs/2602.14878
25. *SWE Context Bench: A Benchmark for Context Learning in Coding.* arXiv:2602.08316（2026-02-09）. https://arxiv.org/abs/2602.08316
26. Abhilasha Lodha, Mahsa Pahlavikhah Varnosfaderani, Abir Chakraborty, et al. *Less Context, Better Agents: Efficient Context Engineering for Long-Horizon Tool-Using LLM Agents.* arXiv:2606.10209（2026-06-08）. https://arxiv.org/abs/2606.10209
27. Andrew Semenov, Svyatoslav Dorofeev. *Beyond Compaction: Structured Context Eviction for Long-Horizon Agents.* arXiv:2606.11213（2026-05-01）. https://arxiv.org/abs/2606.11213
28. Peng Xu, Zuyu Zhang, Yuze Sun, et al. *ContextPipe: Database-Inspired Context Assembly for Long-Horizon Agents.* arXiv:2609.00749（2026-09-01）. https://arxiv.org/abs/2609.00749
29. Zhaoyuan Xia, Qinghongbing Xie, Yung Xiang Hue, et al. *Highlight-Then-Summarize: Learning to Compress Evidence for Long-Context Understanding.* arXiv:2609.31382（2026-09-25）. https://arxiv.org/abs/2609.31382
30. Quan Shi, Keshav Dhandhania, Karthik Narasimhan, et al. *ττ-Bench: An Environment for End-To-End, Realistic Agent Construction.* arXiv:2609.04611（2026-09-04）. https://arxiv.org/abs/2609.04611
31. *ReFold: Training-Free Reversible Inter-Turn Context Folding for Long-Horizon Agents.* arXiv:2610.07863（2026-10-06）. https://arxiv.org/abs/2610.07863
32. DeepSeek-AI, Daya Guo, Dejian Yang, et al. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948（2025-01-22）. https://arxiv.org/abs/2501.12948
33. Yuyang Hu, Shichun Liu, Yanwei Yue, et al. *Memory in the Age of AI Agents.* arXiv:2512.13564（2025-12-15）. https://arxiv.org/abs/2512.13564
34. Wujiang Xu, Zujie Liang, Kai Mei, et al. *A-MEM: Agentic Memory for LLM Agents.* arXiv:2502.12110（2025-02-17，NeurIPS 2025）. https://arxiv.org/abs/2502.12110
35. Google DeepMind. *Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, and the Agentic Era*（含 Pokémon Agent 的 context poisoning / distraction 观察）. arXiv:2507.06261（2025-07）. https://arxiv.org/abs/2507.06261
36. *LLMs Get Lost In Multi-Turn Conversation.* arXiv:2505.06120（2025-05-09，Microsoft/Salesforce）. https://arxiv.org/abs/2505.06120
37. Zhengliang Shi, Yuhan Wang, Lingyong Yan, et al. *Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models.* arXiv:2503.01763（2025-03-03）. https://arxiv.org/abs/2503.01763
38. Jai Lal Lulla, Seyedmoein Mohsenimofidi, Matthias Galster, et al. *On the Impact of AGENTS.md Files on the Efficiency of AI Coding Agents.* arXiv:2601.20404（2026-01-28）. https://arxiv.org/abs/2601.20404
39. Edoardo Debenedetti, Jie Zhang, Mislav Balunović, et al. *AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.* arXiv:2406.13352（2024-06-19）. https://arxiv.org/abs/2406.13352
40. Greg Kamradt. *Needle In A Haystack - Pressure Testing LLMs.* GitHub Repository, 2023. https://github.com/gkamradt/LLMTest_NeedleInAHaystack
41. Kiran Vodrahalli, Santiago Ontanon, Nilesh Tripuraneni, et al. *Michelangelo: Long Context Evaluations Beyond Haystacks via Latent Structure Queries.* arXiv:2409.12640（2024-09-19）. https://arxiv.org/abs/2409.12640
42. Di Wu, Hongwei Wang, Wenhao Yu, et al. *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.* arXiv:2410.10813（2024-10-14）. https://arxiv.org/abs/2410.10813
43. Harvey Yiyun Fu, Aryan Shrivastava, Jared Moore, et al. *AbsenceBench: Language Models Can't Tell What's Missing.* arXiv:2506.11440（2025-06-13）. https://arxiv.org/abs/2506.11440
44. Sreemaee Akshathala, Bassam Adnan, Mahisha Ramesh, et al. *Beyond Task Completion: An Assessment Framework for Evaluating Agentic AI Systems.* arXiv:2512.12791（2025-12-14）. https://arxiv.org/abs/2512.12791
45. OpenAI. *OpenAI Agents SDK: Sessions*（compaction session）. 官方文档，2025–2026. https://openai.github.io/openai-agents-js/guides/sessions/
46. Google. *Agent Development Kit (ADK): Context Compaction.* 官方文档，2025–2026. https://adk.dev/context/compaction/
47. Cognition. *Don't Build Multi-Agents.* Cognition Blog，2025. https://cognition.com/blog/dont-build-multi-agents
48. Linux Foundation / Agent2A. *A2A Protocol Specification.* 2025–2026. https://a2a-protocol.org/dev/specification/
49. Yedidel Louck. *Securing LLM-Agent Long-Term Memory Against Poisoning: Non-Malleable, Origin-Bound Authority with Machine-Checked Guarantees.* arXiv:2606.24322（2026-06-23）. https://arxiv.org/abs/2606.24322

### 补充信源（背景与交叉验证）

- 知乎专题问答：「大模型的上下文工程，都说很重要，但相关的研究那么少，对 Agent 开发有什么影响？」（知乎反爬，经搜索快照与多回答交叉验证）. https://www.zhihu.com/question/1975867034127599388
- Chenyu Zhou, Huacan Chai, Wenteng Chen, et al. *Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness Engineering.* arXiv:2604.08224（2026-04-09）. https://arxiv.org/abs/2604.08224
- Joel Becker, Nate Rush, Elizabeth Barnes, et al. *Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity.* arXiv:2507.09089（2025-07-12，METR RCT）. https://arxiv.org/abs/2507.09089
- Harang Ju, Sinan Aral. *Collaborating with AI Agents: Field Experiments on Teamwork, Productivity, and Performance.* arXiv:2503.18238（2025-03-23）. https://arxiv.org/abs/2503.18238
- Anthropic. *Code Execution with MCP*（observation shaping）. 工程博客，2025–2026. https://www.anthropic.com/engineering/code-execution-with-mcp
- Anthropic. *Effective Harnesses for Long-Running Agents*（progress files / 结构化记录）. 工程博客，2025–2026. https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
