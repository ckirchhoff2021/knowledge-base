# 多智能体 LLM 系统前沿技术报告：当「更多智能体」遇到祛魅——协调、信用分配、记忆一致性与组合安全

> **一句话**：多智能体（Multi-Agent LLM System, MAS）是 2023 年以来最流行也最被滥用的 Agent 架构；2026 年 11 月，一篇综合 **9 项实证、516 种配置** 的元分析给出了迄今最硬的反常识结论——**在样本数或算力预算对齐后，多智能体平均准确率差 −0.042，74% 的对比中等于或差于基线**。但祛魅不是死刑：同一批论文在「token 级信用分配、主子联合 RL、去中心化协作、harness 进化、组合安全」上完成了「再发现」。本报告讲清多智能体什么时候有用、为什么会崩、2026 年该怎么设计与评测。

![timeline](diagrams/1-timeline.html)

## 目录

0. TL;DR：十条核心结论
1. 缘起：从「多智能体狂欢」到「算力对齐祛魅」的三年
2. 形态学：拓扑、通信与编排模式——多智能体到底「多」在哪里
3. 祛魅：为什么多智能体常常不涨点——三组硬证据
4. 失败解剖：MAST 14 种失败模式与失败归因（CoLFA）
5. 记忆与证据流：冲突事实为什么会在委派中「消失」
6. 组合安全：只在「多智能体」时才出现的破防
7. 再发现（一）：训练——token 级信用分配与主子联合 RL
8. 再发现（二）：去中心化、harness 进化与并行自改进
9. 多智能体何时有用：资源对齐评测与选型决策框架
10. 工程落地 Checklist（14 条）
11. 开放问题与 2027 展望
12. 参考文献（54 篇 arXiv 编号逐一核实 + 2 篇一手工程博客）

---

## 0. TL;DR：十条核心结论

1. **祛魅有了定量结论**：2026-11 的元分析（arXiv:2611.17610）综合 9 项一手实证、516 种 MAS 配置，按「资源向量 R（六维）× baseline 阶梯（四档）」重新记账：对单次贪心解码（A 档）的优势总体为负、去掉最大单项研究后近零；对**样本数对齐（B 档）或算力预算对齐（C 档）**的基线，MAS 平均准确率差 **−0.042**，**74%** 的对比中等于或差于基线，多层随机效应模型在对数几率尺度上各敏感性检验均为负。
2. **「多智能体涨点」长期是记账错误**：以往论文拿「会通信的多智能体」对比「单次贪心单模型」，把**样本多重性（sample multiplicity）**和**推理算力**的收益误记成了**协调（coordination）**的收益；仅把基线从贪心换成采样，就平均吃掉 −0.038 的表面优势。
3. **正收益的三个必要条件**：当各智能体证据完全相同时，通信不增加任何信息；正收益只可能来自 **① 证据分区（partitioned evidence）② 外部验证（external verification）③ 跨模型误差多样性（error diversity）**——这是元分析给出的结构性判据，也是本报告全篇的选型第一原理。
4. **产业界早有两极共识**：Anthropic（2025-06）用 orchestrator-worker 多智能体做研究系统，实测 **agent 约 4× chat 的 token、多智能体约 15× chat 的 token**，明确「多智能体主要帮你花掉足够多的 token 去并行探索」，只适合高价值任务；Cognition（2025-06-12，Devin 团队）则发《Don't Build Multi-Agents》，主张**共享上下文**、动作隐含决策，单智能体+并行 subagent 优先。2026-11 的学术证据站在「谨慎」这一边。
5. **失败是有谱系的**：MAST（arXiv:2503.13657）标注 7 个主流框架 1600+ 条轨迹，归纳出 **3 类 14 种**失败模式（系统设计 / 智能体间失配 / 任务验证），标注一致性 κ=0.88；2026-11 的 CoLFA（2611.14980）进一步用协作记录做轻量失败归因，定位「决定性错误步与责任智能体」。
6. **委派会让冲突「消失」**：Harvey LAB 文档评审实验（2611.22245）中，委派把两条冲突主张的**检索率从 50.2% 提到 68.7%**，但**冲突检出率反而从 16.2% 降到 13.2%**——worker 只在 35.2% 的检索对里同时上报两条；同一 worker 读两条时检出高于单智能体，两条被拆给不同 worker 时检出更低。生成式搜索的独立研究《Lost in Delegation》（2611.16526）同样发现委派**没有一致的准确率提升**。
7. **多智能体有「独有攻击面」**：Claude Code、Codex 等未经修改的真实系统，会被「单模型时代早已失效」的攻击攻破（2611.17520）：强编排者自己搜文件几乎零违规，但把搜索委派给子智能体后会采信伪造记录；SwarmSafe-Delta（2611.16123）用配对实验隔离出**消息本身**的因果效应——N=4 时真实消息致 21.4% 违规，安慰剂 5.0%、隐藏投递 1.0%；对 LangChain/LangGraph/AutoGen/OpenClaw 的 12 项 OWASP 评测（2611.21259）则显示工具链与智能体间信任传播普遍存在越权。
8. **「再发现」的核心是训练而非堆人**：OPSD4MAS（2611.16206）把未来通信转成 **token 级信用**（外部模型粗滤 + leave-one-out 细滤 + 与 GRPO 线性退火联合），数学 pass@4 较 GRPO 最高 **+20.00 个百分点**；LEAD-RL（2611.17041）做**主-子智能体联合训练 + 贡献感知信用分配**，BrowseComp +2.49pp 且关键路径成本 −24.5%，验证期近等成本下 +6.78pp。
9. **前沿在向去中心化与自动化迁移**：DecArena（2611.15521）在图约束通信、100K+ 工具的设定下发现现有 LLM agent **难以发现非局部资源、标准 RL 也难以学会协作**；MULTILOOP（2611.15754）用编码 agent 自动进化**多智能体 harness**，仅 10 个训练样本即 +4.6%；ConvRSI（2611.20891）用「卷积」把多智能体自改进并行化（MATH-500 +5.04%）。
10. **人机决策侧的好消息与边界**：真人用户实验（2611.14901）显示多智能体审议（MAD）优于单轮 AI 辅助，主要驱动力是**多轮交互**而非「第二个 AI」，第二个 AI 仅在第一个尚有提升空间时增值；异构 debate 的收益可分解为 ΔA ≈ κ·Γ⁺·ρ（同伴同任务优势 × 接收度），本质是**能力转移**而非「集体推理涌现」（2611.20649）。

> **与知识库既有话题的边界**：本报告聚焦「**多个 LLM 智能体如何组织、通信、训练、评测与防御**」这一系统级主题，与 `agentic-rl`（单/多轮环境 RL 算法病理）、`long-horizon-agents`（时间维度的长程完成）、`agent-harness`（单智能体脚手架）、`context-engineering`（token 生命周期）、`agent-interop`（MCP/A2A 协议标准）、`multi-agent` 在 knowledge-base 中**此前无专文**——本文补齐这一环，并在第 9 节给出与上述话题的交叉索引。

---

## 1. 缘起：从「多智能体狂欢」到「算力对齐祛魅」的三年

### 1.1 2023：辩论、角色扮演与软件公司——范式的集体想象

多智能体 LLM 的叙事起点几乎与 ChatGPT 插件生态同步。2023 年 3–11 月，五条线索在 8 个月内密集出现：

- **角色扮演式任务求解**：CAMEL（arXiv:2303.17760）提出 role-playing 的 inception prompting，让两个 communicative agent 在最小人类干预下协作完成任务，并顺带产出了大规模指令数据；
- **多智能体辩论（MAD）**：Du 等的《Improving Factuality and Reasoning through Multiagent Debate》（2305.14325）让多个模型实例多轮辩论后汇总，显著提升事实性与数学推理；Liang 等的《Encouraging Divergent Thinking》（2305.19118）用「总裁/员工」式多角色辩论增强发散思维；ChatEval（2308.07201）把同一思想用于 LLM 评测；
- **软件公司隐喻**：MetaGPT（2308.00352）把 SOP（标准作业流程）编码进多智能体，用产品经理/架构师/工程师角色把一句话需求变成代码与文档；ChatDev（2307.07924）用瀑布流的 CEO/CTO/程序员/测试角色做软件开发；
- **可组合对话框架**：AutoGen（2308.08155）把「可对话、可执行代码、可换人/换 LLM」的智能体做成通用框架，极大降低了多智能体搭建门槛；AgentVerse（2308.10848）给出招募—协作的通用流程并观察涌现行为；DyLAN（2310.02170）引入动态团队组建与智能体选择；
- **社会模拟**：Generative Agents（2304.03442，斯坦福小镇）证明 25 个有记忆、有反思、有计划的 agent 能产生可信的社会行为；Agent Hospital（2405.02957）把可进化 agent 搬进医院模拟。

同期三篇高引综述确立了研究地图：Wang 等的自主 agent 综述（2308.11432）、Xi 等的 LLM agent 全景（2309.07864）、以及后来的 agent 记忆机制综述（2404.13501）。

这一阶段的集体想象是：**既然一个 LLM 很强，让多个专业化 LLM 对话、辩论、分工，就能像人类组织一样涌现出更高智能**。

### 1.2 2024：规模化、自动化与第一波质疑

2024 年，范式沿三条线膨胀，同时出现第一波严肃质疑：

- **「更多智能体」缩放假说**：《More Agents Is All You Need》（2402.05120）声称随智能体数量增加，性能可在多种模型/任务上持续提升；Mixture-of-Agents（MoA，2406.04692）用多层聚合器把多个模型的回答逐层融合，在 AlpacaEval 等榜单上逼近闭源前沿；
- **通用多智能体系统**：微软 Magentic-One（2411.04468）给出 Orchestrator + WebSurfer/FileSurfer/Coder/ComputerTerminal 的通用解题系统；
- **自动化设计**：AgentSquare（2410.06153）在模块化设计空间里自动搜索 agent 组合；ADAS《Automated Design of Agentic Systems》（2408.08435）让元 agent 自动发明 agent 系统；
- **第一波质疑**：几乎与辩论热潮同期，《Should we be going MAD?》（2311.17371）就发现多智能体辩论的收益对任务、模型和裁判高度敏感，并不稳健——这是「祛魅」的学术伏笔。

### 1.3 2025：失败分类、通信综述与产业两极

2025 年社区从「能不能搭」转向「为什么崩」：

- **MAST 失败分类学**（2503.13657，见第 4 节）成为被引最广的「冷水」文献；
- **通信中心的综述**：《Beyond Self-Talk》（2502.14321）批评大量工作只是「把一个模型的话换个角色再说一遍」（self-talk），并以通信为中心重构分类；《Multi-Agent Collaboration Mechanisms》（2501.06322）系统梳理协作机制；
- **产业两极公开化**：2025 年 6 月，Anthropic 发表多智能体研究系统工程博客（orchestrator-worker、约 15× chat token），Cognition 同周发表《Don't Build Multi-Agents》（共享上下文原则）。**「何时该用多智能体」从此成为一个工程问题而非信仰问题**；
- **博弈论视角**：2026-01 的《Game-Theoretic Lens on LLM-based MAS》（2601.15047）用机制设计与博弈均衡重审智能体协作；通信「5W」综述（2602.11583，Who/Whom/When/What/Why）把通信设计拆成五个可独立优化的维度。

### 1.4 2026-11：祛魅与再发现的合流

本报告抓取了 arXiv cs.CL/cs.AI/cs.MA 三个分类 **2026 年 11 月的 1824 篇新论文**（去重后），标题词频显示：泛 agent 论文 390 篇（约 21%），其中标题含 multi-agent 的 44 篇、编排/委派/辩论/swarm 等扩展命中 20+ 篇。与 2024 年同类论文「报喜不报忧」不同，**2026 年 11 月这一集群的主旋律是实证祛魅 + 机制化再发现**：

- 祛魅侧：516 配置元分析（2611.17610）、委派无效（2611.16526）、记忆不一致（2611.22245）、异构辩论=能力转移（2611.20649）、人机审议实测（2611.14901）；
- 安全侧：最弱智能体破防（2611.17520）、消息因果效应（2611.16123）、框架授权边界（2611.21259）、潜伏妥协（2611.18693）、拓扑偏见注入（2611.21804）、授权以人为代价（2611.19598）、隐私检索（2611.13696）；
- 再发现侧：token 级信用（2611.16206）、主子联合 RL（2611.17041）、去中心化基准（2611.15521）、harness 进化（2611.15754）、并行自改进（2611.20891）、失败归因（2611.14980）、消息路由（2611.13454）、边缘调度（2611.20915）、MAS 构造基准（2611.20997）、部署级文档维护（2611.23142）。

> 媒体侧的佐证：2026-11 量子位等渠道的热点集中在 Qwen4 发布（主打「智能体及编程能力」）、GPT-6.2 延期、RSI 原生 AgentOS 等，产业叙事已全面 agent 化；而学术社区正在为这股热潮补上「对照组」与「工程量账本」。本报告即以这 20+ 篇 11 月新论文为主证据，辅以 2023–2026 奠基文献与一手工程博客。

---

## 2. 形态学：拓扑、通信与编排模式——多智能体到底「多」在哪里

![taxonomy](diagrams/2-topology.html)

讨论「多智能体有没有用」之前，必须先消歧：同样叫 multi-agent，可能是三个同构实例投票，可能是主管带十个研究员，也可能是图上自由对话的群体。**不同形态的成本结构和失败模式完全不同**。综合三篇综述（2501.06322、2502.14321、2602.11583）与 MAST 的框架观察，可用三个正交轴刻画。

### 2.1 轴一：控制拓扑（谁指挥谁）

| 拓扑 | 结构 | 代表 | 优点 | 典型病灶 |
|---|---|---|---|---|
| 去中心化对等 | 所有 agent 同构、全互联或按图邻居通信 | Multiagent Debate（2305.14325）、ChatEval、DecArena（图约束） | 无单点、误差多样性高 | 不收敛、token 爆炸、谣言传播 |
| 层级编排 | orchestrator 拆解—委派—汇总，worker 独立上下文 | Anthropic Research、Magentic-One、LEAD-RL | 上下文隔离、可并行、职责清晰 | 委派质量瓶颈、证据在边界丢失（第 5 节） |
| 流水线/SOP | 角色按固定工序流转（产品→架构→编码→测试） | MetaGPT、ChatDev | 可复现、工程化 | 错误沿工序级联、角色僵化 |
| 动态组队 | 按任务选择/淘汰 agent，结构随轮次变化 | DyLAN、AgentSquare、ADAS | 自适应、省算力 | 选择器本身有偏、评估噪声 |
| 群体/社会模拟 | 大量带记忆与计划的持久 agent | Generative Agents、Agent Hospital | 涌现社会行为 | 评测困难、不可复现 |

2026 年前沿的一个明显走向是**从中心化层级走向去中心化**：DecArena（2611.15521）明确指岀个人 AI agent 时代，中心化编排者「能看到所有 agent、资源与通信」带来可扩展性与隐私双重问题，因而研究**图约束通信**（每个 agent 只见局部状态、只能与邻居交互），并构造了含 100K+ 工具的可扩展分布式工具调用基准。结论很不客气：现有 LLM agent **难以主动发现非局部的相关资源，也难以在去中心化设定下协作，标准 RL 同样难以学会协作**。

### 2.2 轴二：通信内容与粒度（5W）

通信综述（2602.11583）把通信设计拆成 Who（谁发言）/ Whom（对谁）/ When（何时）/ What（发什么）/ Why（为何发）五个维度。实践中的关键二分是：

- **自然语言消息 vs 结构化状态**：自由对话最灵活但最易丢信息；结构化（黑板、共享状态、token 级信用信号）更可控。多智能体状态管理（2605.20563）代表了后者方向。
- **全量转发 vs 压缩转述**：Anthropic 模式要求每个 subagent「把最重要的 token 压缩后再交给 lead」；而 2611.22245 恰恰证明压缩转述是冲突证据消失的地点。
- **广播团队奖励 vs 个体贡献信号**：把同一个团队优势广播给所有 agent，会「奖励误导性报告、惩罚有用报告」（LEAD-RL 的核心批评，2611.17041）；这正是信用分配问题（见第 7 节）。

### 2.3 轴三：同构/异构与「self-talk」之辨

- **同构 self-talk**：同一模型、同一提示的多个副本对话。《Beyond Self-Talk》（2502.14321）批评这类系统缺乏真正的信息异质性——这与祛魅元分析「证据相同时通信不增加信息」的判据完全一致。
- **异构模型混合**：MoA（2406.04692）式跨模型聚合。2026-11 的《Borrowed Strength, Imported Risk》（2611.20649）用匹配面板给出分解：异构辩论的总收益主要由**能力转移（capability transfer）**主导——某个同伴恰好具备该任务能力、接收者又能吸收一部分；诚实收益近似 ΔA_d^hon ≈ κ·Γ_d⁺·ρ_d（同伴同任务正优势 × 接收度）。而且**首轮就存在对抗性暴露（adversarial exposure）**：借能力的同时也借来了风险。
- **角色异构 vs 模型异构**：MetaGPT/ChatDev 的角色异构在同模型上仍可能是 self-talk；真正的误差多样性需要不同模型或不同证据源。

### 2.4 「多智能体」与三个近邻概念的边界

- **vs 集成（ensemble）/ Best-of-N**：多个独立采样再聚合是集成；只有当实例间**交换信息并据此调整**时才是多智能体。祛魅元分析的关键正在于把「协调」与「样本多重性」分开记账（第 3 节）。
- **vs 并行 subagent/工具**：主 agent 派出无状态并行子调用（搜索、跑测试）更接近「并行工具」；Cognition 主张这应是默认形态，而把「多个有独立决策权的持久 agent」当例外。
- **vs 测试时计算（TTS）**：多智能体是 TTS 的一种（昂贵）实现。Snell 等（2408.03314）证明测试时算力本身可换性能；问题在于多智能体是否比「同样算力的单模型多样本/长思考」更优——元分析给出的答案多数情况下是否定的。

---

## 3. 祛魅：为什么多智能体常常不涨点——三组硬证据

![debunk](diagrams/3-debunk.html)

### 3.1 核心证据：516 种配置的元分析（arXiv:2611.17610，2026-11-13）

**《More Agents, More Samples, or More Thinking?》** 是迄今对「多智能体增益」最严格的一次重新记账。其方法论贡献比结论更重要：

1. **资源向量 R（六维）**：把每个系统消耗的资源沿六个维度登记（智能体/样本数、通信轮次、每轮 token、思考预算等），使「多智能体」与「单智能体多采样/长思考」可在同一资源空间比较；
2. **baseline 阶梯（四档）**：
   - **A 档**：单次贪心解码（既往论文最常用的稻草人）；
   - **B 档**：样本数对齐的单模型采样/集成；
   - **C 档**：总算力预算对齐（含通信与思考 token）；
   - **D 档**：进一步对齐模型家族与提示等条件；
3. **数据**：9 项一手实证评估、**516 个不同的多智能体配置**；用多层随机效应模型在对数几率（log odds）尺度上做综合，并做多项敏感性检验。

**结论**：

- 对 A 档（单次贪心）：多智能体准确率差总体为负，去掉贡献最大的单项研究后接近零——即「表面优势」高度依赖个别研究；
- 对 B/C 档（样本或预算对齐）：多智能体平均准确率差 **−0.042**，多层模型在各敏感性检验下**持续为负**；
- **74%** 的可比对比中，多智能体等于或差于基线；几乎所有被纳入研究的研究均值都非正；
- 仅把基线从贪心换成采样（A→B），就平均使准确率差移动 **−0.038**——这定量证明了既往「多智能体胜出」里有约 3.8 个点纯属「没给基线采样」；
- **结构性判据**：当 agent 间证据相同，通信不引入信息；正收益只可能来自 (i) 证据分区、(ii) 外部验证、(iii) 跨模型误差多样性。论文同时给出审稿人 checklist 与记账工具。

> **读法提示**：−0.042 是跨任务平均，不等于「多智能体从不有用」。它的意义是**举证责任倒置**：在没有证据分区/外部验证/误差多样性的任务上，默认假设应为「多智能体不优于等算力单模型」，主张有用的一方需拿出资源对齐的对照实验。

### 3.2 第二组证据：委派在真实任务上不带来一致增益

- **《Lost in Delegation》（2611.16526，2026-11-12）**：在生成式搜索中专门构造「截止日期前/后」数据切分，比较多种委派模式。结论：**委派相对非委派设定没有一致的准确率提升**，效果随任务是否含编排 LLM 自身无法可靠解决的 post-cutoff 信息而变化；委派的价值更多在上下文管理（把长工具轨迹挡在主上下文之外）而非准确率。
- **人机决策侧（2611.14901）**：两项真人用户研究（回答核验、事实核查）发现 MAD 优于单轮 AI 辅助，但**主要驱动力是「多轮交互」而非「第二个 AI」**；第二个 AI 只在单 AI 尚有提升空间时增值。这把「多智能体有用」进一步收窄为「多轮反思有用 + 能力互补时有用」。
- **辩论的能力转移分解（2611.20649）**：如 2.3 节所述，异构辩论收益 ≈ 同伴能力优势 × 接收度，且首轮即有对抗暴露——「集体推理涌现」的浪漫解释被会计式分解取代。

### 3.3 第三组证据：成本结构——15× token 的经济学

Anthropic 工程博客给出的一手账本：普通 agent 约用 **4×** 于 chat 的 token，多智能体研究系统约 **15×**。其成功案例（如快速回答「标普 500 中 X 公司 Y」）的机理被明确表述为：**多智能体主要通过并行探索帮助系统「花掉足够多的 token」来覆盖解空间**，subagent 的独立上下文还能降低路径依赖（path dependency）。因此 Anthropic 自己给出的适用条件是「**任务价值高到足以覆盖增量性能的成本**」。

Cognition 的反面主张（2025-06-12）则是两条上下文工程原则：

1. **共享上下文（Share context）**：多智能体把决策所需的信息切散到各自上下文，任何一个 agent 都看不到全局，导致不一致决策；应尽量在一个共享上下文内工作；
2. **动作隐含决策（Actions carry implicit decisions）**：agent 的每个动作都在替用户做未明说的决策，多一层 agent 就多一层不可审计的隐含决策。

把三方证据合起来看，2026 年的共识不是「多智能体无用」，而是：**多智能体是一种用 4–15 倍 token 购买「并行探索 + 上下文隔离 + 职责分离」的昂贵架构；只有当任务同时满足证据可分区、需要外部验证、且单上下文确实装不下时，这笔交易才划算。**

---

## 4. 失败解剖：MAST 14 种失败模式与失败归因

![failures](diagrams/4-failures.html)

### 4.1 MAST：第一个多智能体失败分类学（arXiv:2503.13657）

MAST 团队观察到一个尴尬事实：多智能体在流行 benchmark 上的增益常常很小，却缺乏「为什么失败」的系统解释。他们做了三件事：

1. **MAST-Data**：跨 **7 个主流 MAS 框架**采集 **1600+ 条带标注轨迹**，是首个刻画多智能体失败动态的数据集；
2. **MAST 分类**：在专家标注者参与下分析 150 条轨迹，得到 **14 种失败模式、聚为 3 大类**，标注者间一致性 κ=**0.88**；
3. **可扩展标注管线**：训练 LLM-as-a-Judge 自动打标，与人类标注高一致；并跨 GPT-4、Claude 3、Qwen2.5、CodeLlama 与编码/数学/通用 agent 任务分析失败分布。

三大类（具体 14 模式的代表性内容）：

- **系统设计问题（System Design）**：规格模糊、任务拆解不当、信息共享缺失、错误的终止条件等——「架构本身把任务做错了」；
- **智能体间失配（Inter-Agent Misalignment）**：角色职责重叠或真空、消息歧义、不一致的输出格式、无效的交接（handoff）、对同伴输出的错误假设——「对话把事情做坏了」；
- **任务验证失败（Task Verification）**：缺少验证、验证标准错误、未检出同伴错误、过早接受结果——「没人核对」。

MAST 的关键立场是：这些失败**靠换更强的基座模型不能全部解决**，需要更好的 MAS 设计（明确契约、共享证据、验证关口）。这与本报告第 9 节的选型框架直接对应。

### 4.2 失败归因：从「系统崩了」到「哪一步、谁的责任」

多智能体失败后最难的是归因：分工与信息交换造成**交错步骤（Interleaved Steps）与混杂内容（Cluttered Content）**，要在冗长记录里把执行关联起来、逐条核对细节，难以兼顾定位精度与 token 成本。

**CoLFA（2611.14980，2026-11-11）** 的观察是：协作过程本身留下了「执行如何衔接、内容如何在 agent 间传递」的记录，这正是追溯错误形成的证据。其 Collaboration-Guided Lightweight Failure Attribution 利用协作结构做轻量归因，定位**决定性错误步（decisive error step）与责任 agent**，在控制 token 成本的同时提升定位准确率。这对生产系统的意义是：失败归因可以从「把整条轨迹喂给大模型」升级为「沿协作边做有向追溯」。

### 4.3 从失败模式到设计反模式

把 MAST 与 11 月新证据合并，可提炼出六个高频反模式：

1. **稻草人对比反模式**：用多智能体打单次贪心单模型（2611.17610 证明这是记账错误）；
2. **无证据分区反模式**：所有 agent 看到同样的输入却期待辩论增值（违反 3.1 的结构判据）；
3. **无关口委派反模式**：lead 不验证 worker 证据就汇总（2611.22245 的冲突消失、2611.17520 的伪造记录）；
4. **共享奖励反模式**：团队优势广播到每个 agent（2611.17041 证明会奖励误导报告）；
5. **隐式信任反模式**：子 agent 输出默认比外部内容更可信（2611.17520、2611.21259 证明这是主要攻击面）；
6. **无终止/无预算反模式**：辩论不收敛、token 无上限（Anthropic 的 15× 账本与 DecArena 的可扩展性观察）。

---

## 5. 记忆与证据流：冲突事实为什么会在委派中「消失」

多智能体相对单智能体最常被援引的好处是「分工收集证据」。但 2026-11 的两篇论文从相反方向揭示了代价：**分工制造了证据分布，而汇总点常常无法重建冲突。**

### 5.1 核心实验：Multi-Agent Memory Inconsistency（arXiv:2611.22245，2026-11-17）

- **设定**：在 Harvey LAB 文档评审任务中注入**受控的事实冲突**（同一问题存在两条相互矛盾的主张），比较单智能体执行与多智能体委派；配套一个证据流分析器（evidence-flow profiler）。
- **反直觉结果**（三个模型、生成预算可比的合并口径）：
  - 委派使两条冲突主张的**检索率从 50.2% 升到 68.7%**（分工确实找到了更多证据）；
  - 但**冲突检出率反而从 16.2% 降到 13.2%**（找到了却没意识到冲突）；
  - worker 仅在 **35.2%** 的已检索冲突对中同时上报两条；
  - 关键交互效应：**当同一个 worker 读到两条主张时，检出高于单智能体；当两条主张被拆给不同 worker 时，检出更低**。
- **机理**：worker 各自只有局部上下文，向 lead 汇报时经过压缩与选择；lead 拿到的是两份「各自合理」的摘要，冲突在跨上下文边界时被蒸发。这不是模型笨，而是**信息流拓扑决定了冲突在数学上可能不可见**——与 Cognition「共享上下文」原则的担忧精确吻合。

### 5.2 生成式搜索的独立印证：Lost in Delegation（2611.16526）

用 pre/post-cutoff 切分数据比较多种委派模式后发现：委派对准确率**没有一致提升**，其主要可复现收益是把长工具轨迹隔离在主编排者上下文之外（上下文卫生），而非更好的答案。换言之，业界从委派中实际获得的是**上下文管理价值**，不是「集体智慧」。

### 5.3 缓解路线：让分歧浮现，而不是假装解决

- **Provenance-weighted memory**：《Surfacing Disagreement Beats Resolving It》（2611.22461）用门控消融实验表明，**把分歧连同出处一起浮现，优于急于「解决」分歧**；带出处权重的记忆让下游能看到矛盾而非收到抹平后的结论。
- **消息路由**：SafeRoute-MAS（2611.13454）在多智能体科学推理中，仅用集成前信息决定「保留全部专家消息还是移除某条」：在 5328 个委员会的封存缓存上，源兼容工作点 **+1.80pp（95% CI [+1.13, +2.52]）并移除 12.76% 消息 token**；但在公开端点的 4580 次受控集成里，选择器优势未能在 token 对齐比较中稳定建立（+1.68pp 不显著）——作者对「智能路由必然更好」保持了克制。
- **设计含义**：把「冲突检出率」「证据出处完整率」列为与准确率并列的 MAS 指标；高冲突风险任务（法律、医疗、尽调）应让同一认知单元持有冲突双方，或强制 lead 做对抗式核对。

---

## 6. 组合安全：只在「多智能体」时才出现的破防

单模型安全做了三年，提示注入的成功率在前沿模型上已大幅下降。但 2026-11 的安全论文集群指出一个系统性盲区：**模型在隔离状态下防得住的攻击，在多智能体系统里照样成功——因为攻击面从「模型」变成了「组件间的信任关系」。**

### 6.1 最弱环节：未修改的 Claude Code / Codex 被攻破（arXiv:2611.17520）

《Only as Safe as the Weakest Agent》设计了一个极简攻击：在磁盘文件库中植入一份**伪造记录**，要求 agent 回答库中问题。

- **强编排者自己搜索**时几乎零有害采信；
- 但一旦通过**系统自带的子智能体机制**把搜索委派出去，编排者就会采信子智能体带回的伪造内容并产生有害行为；
- 攻击在 **Claude Code、Codex 和一个参考 harness 上全部成功，均为未修改版本、使用各自原生的 sub-agent 机制**。

论文的定性是：存在一类「**只在系统是多智能体时才出现**」的破防（a break that appears only when the system is multi-agent）。根因是编排者把「来自我的子智能体」当成了可信信号，而子智能体的上下文里并没有编排者的安全姿态。

### 6.2 消息效应的因果隔离：SwarmSafe-Delta（2611.16123）

如何证明「是上游消息改变了执行者动作」而非聚合指标的巧合？该工作提出配对可执行协议：**冻结上游计算**，对比真实投递 / 隐藏投递 / 格式匹配的安慰剂投递，对单一已提交动作按确定性的良性/风险策略状态打分。

- 四个模型、N=4 时：真实消息导致 **21.4%** 风险世界违规，安慰剂 **5.0%**，隐藏投递仅 **1.0%**；配对差分别为 **+16.4pp 与 +20.3pp**；
- 真实消息同时也提高良性任务成功率——说明消息是**中性的能力通道**，安全与能力在同一信道上耦合，不能简单「砍通信」。

### 6.3 框架级授权与信任边界评测（2611.21259）

对 **LangChain、LangGraph、AutoGen、OpenClaw** 用统一模型（gpt-4o-mini）做 12 项对照 **OWASP Agentic Top 10** 的实验，覆盖提示注入、未授权资源访问与命令执行、记忆投毒、智能体间信任与身份攻击。结果：对抗输入可诱导**未授权工具调用**，工具链接（tool chaining）与智能体间身份/信任传播是系统性薄弱点。这意味着风险不只在应用代码，也在编排框架的默认授权模型里。

### 6.4 持久化、拓扑与授权伦理

- **潜伏妥协（Agentic Sleeper）**：Swarm Shepherd（2611.18693）定义了一种系统级对手——先合法行为获取路由信任，再跨记忆、agent 逻辑与基础设施激活；逐轮防御会漏掉这种跨会话/跨 agent/跨层累积的妥协，需以「部分可观测下的推断」做系统级防御。
- **拓扑决定偏见传播**：Corrupting Ties（2611.21804）用受控偏见注入（把指定 agent 响应替换为预置偏见响应）做干预式分析，提出决策发散度度量，刻画网络拓扑与注入位置如何塑造系统偏见——审计单个组件或只看聚合结果都发现不了这类公平性风险。
- **隐私**：2611.13696 针对安全多智能体系统中的私有知识检索与私有推断，分析语义/情景/短期记忆各自被记忆投毒利用的方式。
- **授权的伦理盲区**：《Protecting the System, Stranding the User》（2611.19598）指出现有多智能体授权架构优先保护系统完整性、把失败成本外部化给操作者：意图没有形式化定义所以「已理解用户意图」的校验不可能确定；动态信任分在智能体交接中不成立。主张**公开报告概率化意图估计、按动作风险缩放谨慎程度**。
- **评测有效性警告**：《Safety, or Just Capability?》（2607.28685）对 agent 安全基准做效度审计，发现部分「安全分」实际测的是通用能力——引用安全数字前必须先问基准测的是什么。

### 6.5 与单智能体安全基准的衔接

AgentDojo（2406.13352）提供了动态评估提示注入攻防的环境，是单智能体侧的标尺；2026-11 集群的贡献是把它扩展到**组合态**：同样的注入，经过委派、记忆共享、工具链后成功率回升。工程上需要为「智能体边界」建立与「模型边界」同等的防御：出处标记、最小权限、子智能体输出视同外部不可信输入、跨 agent 的信任不传递。

---

## 7. 再发现（一）：训练——token 级信用分配与主子联合 RL

祛魅回答的是「现成模型拼起来会不会自动变聪明」（多数情况下不会）；2026 年真正的进展是**把多智能体当成可训练对象**，用 RL 解决协作本身。瓶颈集中在经典多智能体 RL 的老大难：**信用分配（credit assignment）**——团队结果对了，是哪条消息、哪个 token 的功劳？

### 7.1 OPSD4MAS：把通信变成 token 级信用（arXiv:2611.16206，2026-11-12）

现有多智能体 LLM 的 RL 要么忽略通信知识，要么把通信压缩成标量奖励，导致信用过粗，无法告诉每个 agent「哪一步该加强、哪一步该修正」。OPSD4MAS（On-Policy Self-Distillation for Multi-Agent Systems）的做法：

1. **粗滤**：外部模型从未来通信中抽取事实，评估事实与轨迹的关系；
2. **细滤**：对目标 token 做 **leave-one-out** 式影响力分析，对照「逐 agent 的结果优势（agent-wise outcome advantage）」筛出真正有贡献的 token；
3. **事实条件策略**提供 token 级信用，与 **GRPO** 通过线性退火的联合目标结合；
4. 消融支持「完整过滤」，冻结历史注入实验用于诊断。

**结果**：数学 pass@4 较 GRPO 最高 **+20.00 个百分点**。这是「协调可以被训练出来」的强证据，也反衬出零训练辩论的脆弱。

### 7.2 LEAD-RL：主-子智能体联合训练 + 贡献感知信用（arXiv:2611.17041，2026-11-12）

多智能体深度研究系统用委派并行搜索、把长工具轨迹挡在主上下文外。但委派只有在「任务分配」与「子任务执行」协调时才有用：主 agent 要发对的子任务，子 agent 要回可靠的证据。已有训练要么只优化主 agent，要么把同一团队优势广播给所有 agent——后者会**奖励误导性报告、惩罚有用报告**。

LEAD-RL（Learning Effective Agent Delegation with RL）：

- **主-子联合训练**；训练时用一套 rubric 评估每份子 agent 报告的「研究效用」，以此为门控决定它能继承多少团队优势（contribution-aware credit assignment）；
- 最终答案正确性是**唯一团队奖励**；明确不使用 token、步数或加速奖励（避免投机取巧）；
- **结果**：五个基准、两个模型家族上提升答案质量；BrowseComp 准确率 **+2.49pp**、关键路径成本 **−24.5%**（相对共享团队信用的联合训练）；验证期在近等关键路径成本下 **+6.78pp**。

意义：委派从「提示工程」变成「可训练的协调策略」，且同时改善质量—成本前沿。

### 7.3 训练基础设施与自改进

- **FlexMARL（2602.09578）**：图驱动的 rollout—训练协同设计，针对 LLM 多智能体 RL 中异构展开、消息依赖导致的工程复杂度，是 2026 年较早系统化的多智能体 RL 基础设施；
- **ConvRSI（2611.20891）**：多智能体自改进中，跨 agent 与跨精炼轮的顺序依赖会传播误差、增加成本、限制并行；逐个孤立精炼又忽略依赖。ConvRSI 用「卷积」并行化：**intra-agent 卷积**联合精炼相关 agent 特征，**inter-agent 图卷积**按协作拓扑聚合消息；两阶段都在潜空间完成，无需反复解码—重编码中间消息。MATH-500 上 **+5.04%**（作者口径），代表「递归式自改进 → 并行卷积式自改进」的工程方向；
- 更早的多智能体 RL 训练（如 2025 年的 MIRROR 等工作）多在工具调用/推理环境中验证群体 RL；本报告以 FlexMARL 与 11 月两篇为锚点，不逐一展开。

### 7.4 训练侧的共同方法论

| 维度 | 零训练拼装（2023 范式） | 可训练协作（2026 范式） |
|---|---|---|
| 信用信号 | 团队标量奖励/无奖励 | token/报告级贡献信用 |
| 通信 | 自由对话 | 事实抽取、影响力过滤、潜空间消息 |
| 主-子关系 | 提示约定 | 联合策略 + rubric 门控 |
| 成本目标 | 无 | 关键路径成本显式优化 |
| 评测基线 | 单次贪心（稻草人） | 样本/算力预算对齐 |

---

## 8. 再发现（二）：去中心化、harness 进化与部署级系统

### 8.1 去中心化协作基准 DecArena（arXiv:2611.15521，2026-11-11）

个人 AI agent 的普及使中心化编排（全局 orchestrator 可访问所有 agent、资源与通信）面临可扩展性与隐私问题。DecArena 研究**图约束通信下的去中心化多智能体协作**：每个 agent 只见局部状态、只能与邻居交互，需主动发现非局部资源（工具/技能）并协作。基准含合成的可扩展分布式工具调用（**100K+ 工具**）。两个负面发现同样重要：(1) 现有 LLM agent 难以发现非局部相关资源；(2) 标准 RL 难以在该设定下学会协作——指向「资源发现协议」与「局部可观测信用分配」是下一阶段的硬骨头。

### 8.2 MULTILOOP：自动进化多智能体 harness（arXiv:2611.15754，2026-11-11）

agent 性能不只取决于模型，也取决于 harness（提示、工具、编排逻辑）。已有用编码 agent 分析轨迹、诊断失败、改写 harness 代码的自动化工作，但都聚焦单智能体；多智能体 harness 更难，因为修复不仅要改每个 agent 的行为，还要改**agent 之间如何交互**。MULTILOOP 在复杂信息检索任务（DeepSearchQA）上做案例研究：编码 agent 迭代「分析—评估—编辑」harness，**仅用 10 个训练样本即使准确率较基线 harness +4.6%**。这与知识库既有话题 `agent-harness`（Harness Scaling）直接衔接：多智能体把 harness 优化的搜索空间从「单循环」扩展到「交互拓扑」。

### 8.3 构造、调度与部署

- **OpenMAS-GApp（2611.20997）**：面向跨领域 MAS 应用构造的综合基准，统一任务分解、agent/工具选择、执行组织四个既往各自评测的环节，解决「不同方法用不同任务/组件/设置无法比较」的问题；
- **边缘多 LLM 调度（2611.20915）**：把多智能体放到边缘，联合优化 agent 调度与资源分配，呼应 15× token 成本下的经济性压力；
- **部署级文档维护（2611.23142）**：一个真实部署的多智能体系统，利用显式信号（已解决的支持工单）与隐式信号（自动化检测到的代码—文档不一致）维护客户文档，防止陈旧信息被下游 agent 当权威——是「多智能体在生产中到底做什么」的少见落地样本；
- **企业流程组合（2611.13196）**：BPMN 流程模型 + agent + 企业知识组合为可编排 agentic workflow，代表传统 IT 对多智能体的吸收方式。

---

## 9. 多智能体何时有用：资源对齐评测与选型决策框架

![decision](diagrams/5-decision.html)

### 9.1 评测的正确姿势：资源向量 × baseline 阶梯

引用或报告多智能体结果时，按 2611.17610 的记账法核对：

1. 写出系统的**资源向量 R**（agent 数、样本数、通信轮次、每轮输入/输出 token、思考预算、模型单价）；
2. 至少与 **B 档（样本数对齐）和 C 档（总预算对齐）** 基线比较，而不是只打 A 档贪心单模型；
3. 报告**配对对比**与置信区间（参考 SafeRoute-MAS 的封存缓存 + 95% CI 做法），区分「源兼容工作点」与「token 对齐比较」；
4. 把**冲突检出率、证据出处完整率、单任务成本/关键路径延迟、违规率**与准确率并列；
5. 安全评测先过效度审计（2607.28685：别把能力分当安全分），并在**组合态**（委派+工具链+共享记忆）复测。

### 9.2 选型决策：什么时候真的需要多智能体

把三方证据（元分析结构判据、Anthropic 工程经验、Cognition 上下文原则）压缩成一棵决策树：

1. **单上下文 + 单模型能否装下并完成？** 能 → 不要多智能体（默认单 agent + 并行无状态子调用）。
2. **是否存在天然的证据分区**（不同数据源、不同子问题、可并行探索）且分区后会有**显式汇聚与冲突核对**？否 → 多智能体只增加成本。
3. **是否需要外部验证闭环**（运行代码、检索、交叉质询）？需要 → 用角色分工，但验证器必须独立于生产者。
4. **参与模型是否提供误差多样性**（异构模型/不同证据源），而非同模 self-talk？否 → 用 BoN/集成替代，更便宜。
5. **任务价值是否覆盖 4–15× token 成本**，且质量—成本前沿经 LEAD-RL 式优化后仍为正？否 → 不做。
6. **安全姿态能否传播到每个子 agent**（出处标记、最小权限、子输出视同不可信外部输入）？不能 → 先补安全再上线。

**适合**：高价值开放式深度研究（并行探索 + 证据隔离）、跨大量独立工具/数据源的长任务（DecArena 式）、需要对抗式验证的高风险推理、异构模型能力互补。
**不适合**：单一知识源问答、短链工具调用、预算敏感的高并发场景、证据高度重叠的「辩论」、安全边界无法治理的自动化动作。

### 9.3 与知识库既有话题的交叉索引

- 与 `agentic-rl`：多智能体 RL 是其在多主体设定的延伸，新增信用分配与消息依赖两大难题（第 7 节）；
- 与 `long-horizon-agents`：多智能体是长程任务的一种状态外部化手段，但长程完成率仍受 durable runtime 与 compaction 制约；
- 与 `agent-harness`：MULTILOOP 把 harness 进化扩展到交互拓扑层；
- 与 `context-engineering`：委派的主要可复现收益是上下文隔离；冲突消失是 compaction/转述的跨 agent 版本；
- 与 `agent-interop`：A2A/MCP 解决「能不能通信」，本报告解决「通信有没有用、安不安全」；
- 与 `test-time-scaling`：多智能体是 TTS 的昂贵实现，必须在同一资源向量下与长思考/BoN 比较。

---

## 10. 工程落地 Checklist（14 条）

**立项与选型**

1. 先写资源向量 R（agent 数、样本数、轮次、token、思考预算、单价），再谈架构；默认从单 agent + 并行无状态子调用开始。
2. 用三必要条件自检：证据可分区？需要外部验证？存在跨模型误差多样性？三者皆无则不做多智能体。
3. 任务价值须覆盖 4–15× token 成本（Anthropic 实测口径），并设定关键路径延迟与美元上限。

**评测纪律**

4. 必须对 B 档（样本对齐）与 C 档（预算对齐）基线做配对对比，报告置信区间；禁止只打单次贪心基线。
5. 主指标之外加测：冲突检出率、证据出处完整率、每任务成本、关键路径延迟、违规率。
6. 安全基准先做效度审计（区分安全分与能力分），并在组合态（委派 + 工具链 + 共享记忆）复测。

**架构与证据流**

7. 高冲突风险任务让同一认知单元持有冲突双方，或强制 lead 做对抗式核对；worker 上报保留出处与原文引用。
8. 分歧先浮现后解决：用 provenance-weighted memory，禁止在边界处把矛盾抹平成单一结论。
9. lead 对 worker 输出设独立验证关口（生产—验证分离），验证器不得复用生产者上下文。
10. 通信能结构化就不自由文本；考虑 token/报告级贡献信用，避免团队奖励广播。

**安全**

11. 把所有子智能体输出视同外部不可信输入；信任与权限不沿委派链自动传递（最小权限 + 显式授权）。
12. 对 agent 身份、工具链接、共享记忆做投毒防护；监控跨会话的潜伏妥协（Agentic Sleeper 模型）。
13. 高风险动作采用概率化意图确认，谨慎程度随动作风险缩放；审计日志要能沿协作边归因到责任 agent。

**演进**

14. 把多智能体 harness（含交互拓扑）纳入自动化优化循环（MULTILOOP 式），但每次进化必须过资源对齐评测与安全回归。

---

## 11. 开放问题与 2027 展望

1. **协调的可证明边界**：−0.042 是经验均值，尚缺「在何种信息论条件下多智能体严格优于等算力单模型」的定理；证据分区/外部验证/误差多样性三条件目前是经验判据，等待形式化（与博弈论视角 2601.15047 合流）。
2. **多智能体信用分配的规模化**：OPSD4MAS 的 token 级信用依赖外部事实抽取与 leave-one-out，成本随消息量二次增长；能否在去中心化、局部可观测设定下廉价近似（DecArena 的开放问题），决定该路线能否走出数学任务。
3. **证据拓扑的工程理论**：2611.22245 表明「谁和谁看到同一证据」是一阶变量，但目前没有类似网络容量定理的设计指南；冲突检出率应成为 MAS 的一等公民指标。
4. **组合安全的攻防竞赛**：从「模型安全」到「系统安全」的迁移刚起步；框架级默认授权（LangChain/AutoGen 等）如何提供安全默认值、OWASP Agentic Top 10 如何变成可执行测试，是 2027 的工程焦点。
5. **去中心化个人 agent 生态**：当数百万个人 agent 按图协作、跨组织发现工具（100K+ 工具尺度），资源发现、信任协商、隐私推断（2611.13696）与经济学定价会融合成新的中间件层，与 A2A/MCP 标准相互塑造。
6. **人机审议的设计科学**：2611.14901 证明「多轮」比「多 AI」更重要，persona 与人类介入程度如何设计才能真正改善人类决策（而非制造认同假象），仍是开放的人因问题。
7. **评测范式**：需要跨研究可复用的「资源对齐 + 组合安全 + 冲突检出」三维基准与记账工具（2611.17610 已提供起点），否则 MAS 论文的数字仍不可比。

> **总判断**：多智能体没有死，但「多召几个智能体商量一下」作为默认解法已经死了。2026 年 11 月这批论文完成了一次健康的范式收缩——**多智能体从「免费的智能涌现」回归为「有明确成本、条件与攻击面的分布式系统」**；真正的增量在可训练的协调（信用分配/主子联合 RL）、去中心化协议、harness 自动化与组合安全。对工程团队，正确姿态是 Cognition 式的克制做默认、Anthropic 式的重投入做例外，并用资源对齐评测为每一次例外记账。

---

## 12. 参考文献（54 篇 arXiv + 2 篇一手工程博客；arXiv 编号经 arxiv.org 摘要页逐一核实，0 错号）

**A. 2023 奠基：辩论、角色扮演与框架**

1. Du, Y. et al. *Improving Factuality and Reasoning in Language Models through Multiagent Debate.* arXiv:2305.14325, 2023.
2. Liang, T. et al. *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate.* arXiv:2305.19118, 2023.
3. Li, G. et al. *CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society.* arXiv:2303.17760, 2023.
4. Hong, S. et al. *MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework.* arXiv:2308.00352, 2023.
5. Wu, Q. et al. *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation.* arXiv:2308.08155, 2023.
6. Qian, C. et al. *ChatDev: Communicative Agents for Software Development.* arXiv:2307.07924, 2023.
7. Chen, W. et al. *AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors.* arXiv:2308.10848, 2023.
8. Chan, C.-M. et al. *ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate.* arXiv:2308.07201, 2023.
9. Liu, Z. et al. *A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration (DyLAN).* arXiv:2310.02170, 2023.
10. Park, J. S. et al. *Generative Agents: Interactive Simulacra of Human Behavior.* arXiv:2304.03442, 2023.
11. Du, Y. et al. *Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs.* arXiv:2311.17371, 2023.
12. Wang, L. et al. *A Survey on Large Language Model based Autonomous Agents.* arXiv:2308.11432, 2023.
13. Xi, Z. et al. *The Rise and Potential of Large Language Model Based Agents: A Survey.* arXiv:2309.07864, 2023.

**B. 2024 扩张：规模化、自动化与通用系统**

14. Li, J. et al. *More Agents Is All You Need.* arXiv:2402.05120, 2024.
15. Wang, J. et al. *Mixture-of-Agents Enhances Large Language Model Capabilities.* arXiv:2406.04692, 2024.
16. Fourney, A. et al. *Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks.* arXiv:2411.04468, 2024.
17. Quan, S. et al. *AgentSquare: Automatic LLM Agent Search in Modular Design Space.* arXiv:2410.06153, 2024.
18. Chen, X. et al. *Automated Design of Agentic Systems (ADAS).* arXiv:2408.08435, 2024.
19. Zhou, T. et al. *Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents.* arXiv:2405.02957, 2024.
20. Zeng, A. et al. *A Survey on the Memory Mechanism of Large Language Model based Agents.* arXiv:2404.13501, 2024.
21. Snell, C. et al. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* arXiv:2408.03314, 2024.

**C. 2025–2026 理性：失败学、通信综述、博弈与安全基准**

22. Cemri, M. et al. *Why Do Multi-Agent LLM Systems Fail? (MAST)* arXiv:2503.13657, 2025.
23. *Multi-Agent Collaboration Mechanisms: A Survey of LLMs.* arXiv:2501.06322, 2025.
24. *Beyond Self-Talk: A Communication-Centric Survey of LLM-Based Multi-Agent Systems.* arXiv:2502.14321, 2025.
25. *The Five Ws of Multi-Agent Communication: Who Talks to Whom, When, What, and Why — A Survey.* arXiv:2602.11583, 2026.
26. *A Game-Theoretic Lens on LLM-based Multi-Agent Systems.* arXiv:2601.15047, 2026.
27. *Multi-agent Collaboration with State Management.* arXiv:2605.20563, 2026.
28. *FlexMARL: Graph-Driven Rollout-Training Co-Design for LLM-Based Multi-Agent Reinforcement Learning.* arXiv:2602.09578, 2026.
29. Debenedetti, E. et al. *AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.* arXiv:2406.13352, 2024.
30. *Safety, or Just Capability? A Validity Audit of Agent-Safety Benchmarks.* arXiv:2607.28685, 2026.

**D. 2026-11 祛魅集群（本报告主证据）**

31. *More Agents, More Samples, or More Thinking? Untangling Coordination, Ensembling, and Test-Time Compute in Multi-Agent LLMs.* arXiv:2611.17610, 2026-11-13.
32. *Lost in Delegation: Studying the Effectiveness of Task Orchestration in Generative Search.* arXiv:2611.16526, 2026-11-12.
33. *Multi-Agent Memory Inconsistency: Where Factual Conflicts Disappear in LLM Agents.* arXiv:2611.22245, 2026-11-17.
34. *Borrowed Strength, Imported Risk: Capability Gaps and Reception Govern Heterogeneous LLM Debate.* arXiv:2611.20649, 2026-11-16.
35. *Understanding the Effects of Multi-Agent Deliberation for Human-AI Decision-Making.* arXiv:2611.14901, 2026-11-11.
36. *Surfacing Disagreement Beats Resolving It: A Gated Ablation Study of Provenance-Weighted Memory for Multi-Agent Debate.* arXiv:2611.22461, 2026-11.
37. *SafeRoute-MAS: Coverage-Aware and Risk-Calibrated Message Routing for Multi-Agent Scientific Reasoning.* arXiv:2611.13454, 2026-11-10.

**E. 2026-11 失败归因与安全集群**

38. *Only as Safe as the Weakest Agent: A Break That Appears Only When the System Is Multi-Agent.* arXiv:2611.17520, 2026-11-12.
39. *SwarmSafe-Delta: Isolating Message Effects on LLM Agent Safety.* arXiv:2611.16123, 2026-11-12.
40. *Evaluating Authorization and Trust Boundaries in Agentic Orchestration Frameworks.* arXiv:2611.21259, 2026-11-16.
41. *Swarm Shepherd: Securing Multi-Agent Ecosystems Against Persistent Latent Compromise.* arXiv:2611.18693, 2026-11-04.
42. *Corrupting Ties: Investigating Robustness to Bias Injection Across Multi-Agent Network Topologies.* arXiv:2611.21804, 2026-11-16.
43. *Protecting the System, Stranding the User: The Limits of Multi-Agent Authorization.* arXiv:2611.19598, 2026-11-14.
44. *On Private Knowledge Retrieval and Private Inference for Secure Multi-Agent Systems.* arXiv:2611.13696, 2026-11-10.
45. *CoLFA: Collaboration-Guided Lightweight Failure Attribution for LLM-Based Multi-Agent Systems.* arXiv:2611.14980, 2026-11-11.

**F. 2026-11 再发现：训练、去中心化、harness 与部署**

46. *Turning Multi-Agent Communication into Token-Level Credit with On-Policy Self-Distillation (OPSD4MAS).* arXiv:2611.16206, 2026-11-12.
47. *From Delegation to Coordination: Learning Efficient Multi-Agent Deep Research (LEAD-RL).* arXiv:2611.17041, 2026-11-12.
48. *DecArena: Evaluating Large-Scale Decentralized Multi-Agent Collaboration.* arXiv:2611.15521, 2026-11-11.
49. *MULTILOOP: Evolving Multi-Agent Harnesses for Complex Information-Seeking Tasks.* arXiv:2611.15754, 2026-11-11.
50. *From Recursion to Convolution: Parallel Self-Improvement for Multi-Agent Systems (ConvRSI).* arXiv:2611.20891, 2026-11-16.
51. *OpenMAS-GApp: A Comprehensive Benchmark for Domain-Adaptive Multi-Agent Application Construction.* arXiv:2611.20997, 2026-11-16.
52. *Multi-LLM Agentic AI at the Edge: Joint Agent Scheduling and Resource Allocation.* arXiv:2611.20915, 2026-11.
53. *Mind the Gap: A Deployed Multi-Agent System for Documentation Maintenance.* arXiv:2611.23142, 2026-11-17.
54. *From Process Models to Composable Agentic Workflows: Combining BPMN, Agents and Enterprise Knowledge.* arXiv:2611.13196, 2026-11-09.

**G. 一手工程博客（非 arXiv）**

55. Anthropic. *How we built our multi-agent research system.* 2025-06. https://www.anthropic.com/engineering/multi-agent-research-system （原文实测：agent ≈ 4× chat token，多智能体 ≈ 15× chat token；orchestrator-worker 架构）
56. Yan, W. (Cognition/Devin). *Don't Build Multi-Agents.* 2025-06-12. https://cognition.ai/blog/dont-build-multi-agents （上下文工程两原则：Share context / Actions carry implicit decisions）

---

## 附：信源与核实方法说明

- **热点调研**：2026-11-19 抓取 arXiv cs.CL/cs.AI/cs.MA 当月新提交 1824 篇做标题词频（multi-agent 44 篇、其中 LLM 相关 20+ 篇居 agent 子方向前列）；量子位 RSS（2026-11-13～19）、知乎热榜（tophub 与官方 API 受 JS/鉴权限制，改以可访问的腾讯云开发者社区等全网信源交叉）；产业信源直接读取 Anthropic / Cognition 官方博客原文核对关键数字。
- **文献核实**54 篇 arXiv 全部于 2026-11-19 经 `https://arxiv.org/abs/<id>` 摘要页逐一核实标题与日期，0 错号（含 CAMEL 常见误植 2303.17766 的勘误，正确为 2303.17760）；2 篇非 arXiv 给岀可访问的官方链接。
- **口径标注**：论文实验数字均标注「作者口径」；Anthropic 的 4×/15× 为其工程博客一手实测；媒体说法与论文证据冲突时以论文/官方原文为准。
- **去重声明**：本报告与 knowledge-base models 分类下既有 36 个话题无重复；与 `agentic-rl`、`long-horizon-agents`、`agent-harness`、`context-engineering`、`agent-interop`、`test-time-scaling` 六个相邻话题的边界已在第 9.3 节逐条划清。
