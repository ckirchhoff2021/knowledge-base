# 主动式智能体（Proactive Agents）前沿技术报告：当大模型不再等待指令——从「能不能做」到「该不该现在做」的干预时机问题

> **信源**：33 篇 arXiv 论文（编号已经 arXiv 官方 API 逐一核实标题与日期）+ 1 篇 TMLR 2026 同行评议综述（OpenReview / Zenodo）+ 8 篇经典 HCI/UbiComp 文献（ACM/Springer 正式 DOI）+ Anthropic / Microsoft / ByteDance 等官方工程文档与产品页
> **报告日期**：2026-11-01
> **关键词**：Proactive Agent · Intervention Timing · Always-On Assistant · Mixed-Initiative · 注意力管理 · 决策论门控 · 智能眼镜 / MR

---

## 摘要

过去三年，大模型的竞争主轴是**能力（capability）**：模型能不能写代码、能不能做多步推理、能不能操作电脑。而 2025 年末至 2026 年，一个更前置的问题随着「始终在线」（always-on）的智能眼镜、MR 头显和环境式副驾（ambient copilot）的普及而浮出水面：**当一个系统每小时要做数百次决定时，它该不该在「无人指令」的此刻主动开口或动手？**

这就是**主动式智能体（Proactive Agent）**与**干预时机（Intervention Timing）**问题。它与本知识库已覆盖的 Agent 记忆、Harness、GUI Agent、全双工语音等方向正交：那些工作回答的是「如何把一件被交代的事做好」，而本报告回答的是「在没有被交代时，如何判断一件事值不值得现在做」。

本报告以 TMLR 2026 综述 *When Should the Agent Speak?* 提出的单一决策规则为骨架——

> **intervene ⇔ E[行动收益] − E[打断成本] > θ**
> （当且仅当行动的期望净收益超过阈值 θ 时才干预）

系统梳理：(1) 从被动响应到主动干预的范式转移与能力分级；(2) 信号—决策—行动—记忆—评测五层技术架构；(3) 两条互不相通的思想谱系——1999–2022 年把「打断成本」形式化却没有能干的执行者，2024–2026 年有了能干的执行者却把成本项重新做成碎片；(4) Proactive Agent/ProactiveBench、PRISM、TGL 门控、ProAgent、Pro²Assist、EgoSocial、Plan-Watch-Recover 等十余个旗舰系统的逐一拆解；(5) 2026 年三篇独立工作收敛出的「廉价门控 / 昂贵执行器」架构；(6) 「reward ⊆ eval」的根本困境与评测基准矩阵；(7) 以 Anthropic Claude Duet 为代表的产业落地与安全边界；(8) 开放问题与工程落地 Checklist。

核心判断：**主动式智能体是 2026 年下半年 Agent 范式从「对话工具」走向「终身同事」的关键一跃，而决定这一跃成败的不是更强的执行者，而是被长期忽视、正在被重新发现的「打断成本」这一项。**

---

## 目录

1. 引言：为什么是「现在」——从挤牙膏式交互到始终在线的同事
2. 什么是主动式智能体：定义、边界与能力分级
3. 核心决策问题：干预是不确定性下的决策
4. 五层技术架构：信号 · 决策 · 行动 · 记忆 · 评测
5. 两条互不相通的思想谱系（1999–2026）
6. 旗舰系统与方法深度拆解
7. 关键架构收敛：廉价门控 / 昂贵执行器
8. 评测体系与「reward ⊆ eval」
9. 产业落地：以 Claude Duet 为镜
10. 开放问题
11. 工程落地 Checklist
12. 结语
- 参考文献

---

## 1. 引言：为什么是「现在」

### 1.1 「你说一句，我动一下」的天花板

今天绝大多数 AI 产品的交互逻辑仍是**被动响应（reactive）**：所有行为必须由用户的显性指令触发。用户想让模型写一份活动方案，缺预算模块就要补一句「加上预算」，缺裂变规则再补一句，来回五六次。这种模式在 AI 只做查询与创作时尚可接受，但当人们期望它成为真正的个人助理、融入办公、健康、家居等长期陪伴场景时，短板被无限放大：**交互效率低、指令成本高、完全无法处理需要预判和长期跟进的事务。**

### 1.2 触发拐点的三股力量

2025 年末至 2026 年，三股力量把「主动性」从学术议题推成了工程刚需：

- **硬件始终在线化**：以智能眼镜（Meta Ray-Ban 等）和 MR 头显为代表的设备具备连续的第一人称感知能力。一个「等被提问才工作」的眼镜助手，本质上只是「多了几步操作的录音机」。
- **执行者真正成熟**：多步任务智能体已能对着可验证目标把整件事做完（见本知识库 GUI Agent、Agent Harness 等报告）；第一人称感知也能解析用户正在看什么、伸手够什么。**唯独「该不该现在开口」这个每小时要做数百次的决定，没有人系统建模。**
- **产品范式集体转向**：2026 年 10 月，头部厂商在同一月内密集落地主动式产品——Anthropic 于 10 月 21 日开放 **Claude Duet** 研究预览（监控已连接应用、主动提示待办、用户离开时继续工作、重大动作前请求批准）；Google 让 Gemini 新增 Tasks 模式可跨服务后台工作；Meta 则把持续上下文与穿越障碍的主动聚焦做成模型级能力。

> ![](diagrams/1-paradigm-shift.html)

### 1.3 与既有方向的正交性

本知识库已覆盖的方向中，Agent 记忆解决「记得什么」，Harness 解决「如何被脚手架支撑」，GUI Agent 解决「如何操作界面」，全双工语音解决「如何边听边说」。它们共享一个隐含前提：**任务已经被发起**。主动式智能体研究的恰恰是这个前提之前的决定——**在无人发起时，是否、何时、以何种强度介入。** 这是一个尚未被覆盖的新方向。

---

## 2. 什么是主动式智能体

### 2.1 定义

- **被动响应式智能体（Reactive Agent）**：仅响应显性输入，所有行为的触发源都来自用户的直接请求（普通版 ChatGPT、传统语音助手）。
- **主动式智能体（Proactive Agent）**：具备多模态感知、长期记忆与意图预判能力，可在**无显性指令**时，基于上下文与历史行为主动识别潜在需求，发起交互甚至自动执行任务的智能体。

需要强调：**「主动」不等于「放任」**。一个设计良好的主动式系统，其标志恰恰是知道何时保持沉默、何时先询问、以及把重大动作的最终控制权留给用户。

### 2.2 主动性的两个正交轴

TMLR 综述与 IBM 的工作共同提示，主动性至少要在两个正交轴上刻画：

- **时机轴（timing）**：是否在无人指令时行动、何时行动——本报告主线。
- **内容轴（content）**：一旦决定行动，去追求什么信息。IBM 把它进一步分为**水平主动（horizontal）**——追求当前上下文已经暴露出的未言明信息，与**垂直主动（vertical）**——追求只有更早的证据才能揭示的深层需求。

### 2.3 能力分级：从被动响应到自主执行

综合产业实践，可把主动能力分为五级，落地时应逐级放开：

| 级别 | 名称 | 触发方式 | 典型表现 | 风险 |
|---|---|---|---|---|
| L0 | 被动响应 | 完全由用户触发 | 一问一答 | 最低 |
| L1 | 规则提醒 | 预设规则 | 日程/待办提醒 | 低 |
| L2 | 主动提议 | 预判 + 上下文 | 提前给出建议，**待用户确认** | 中（打扰） |
| L3 | 授权自动执行 | 高信任 + 边界内 | 自动整理纪要、草拟回复 | 高 |
| L4 | 跨场景自主协同 | 多设备/长程目标 | 主动监控、异常处置、长期跟进 | 最高 |

主流产品（如 Claude Duet）在 2026 年仍谨慎地停留在「可读可草拟可监控、发送/支付/删除必须审批」的 L2–L3 之间。

---

## 3. 核心决策问题：干预是不确定性下的决策

### 3.1 统一决策规则

整个领域可以由一条规则统领：

> **intervene ⇔ E[benefit of acting] − E[cost of interrupting] > θ**

即：**干预是一个在不确定性下的决策，当且仅当行动的期望收益减去打断的期望成本超过阈值 θ 时才介入。** 每一项研究都可以按它推进了不等式的哪一部分来归类。

与之等价的意图概率视角是贝叶斯更新：以用户历史行为序列 S 为先验、当前上下文 C 为观测，计算潜在需求 A 的后验

  P(A|S,C) = P(C|A,S) · P(A|S) / P(C|S)

再用效用函数 U(A) = V(A) − λ · D(A)（V 为提议价值、D 为打扰成本、λ 为用户对打扰的耐受权重）决定是否推送。

### 3.2 「沉默」与「询问」也是一等动作

决策几何的关键洞察（最早可追溯到 Horvitz 1999 年的混合发起界面原则）是：

- **等待/沉默是具有可计算价值的一等动作**，其价值随时间贴现；
- 动作阈值会**随用户专注深度而升高**——用户越专注，开口门槛越高；
- 把「追问澄清」作为第三个动作，就得到**三态策略：保持沉默 / 先询问 / 直接行动**。这比简单的「说或不说」二元决策精细得多，也是现代助手重新发现「澄清式提问」这一 UX 模式的理论源头。

### 3.3 干预不是二元开关： graduated action（分级动作）

语音社区的 turn-taking 研究提示，干预存在一个**成本递增的动作菜单**：一次轻提示音（chime）、一个附和式 backchannel、一句完整插话、一次接管式操作，成本逐级升高。「一声提示音不等于一次接管」，因此度量时要对分级动作给予折扣。这一点直接决定了第 8 节的指标设计。

> ![](diagrams/2-intervention-pipeline.html)

---

## 4. 五层技术架构

TMLR 综述把主动干预组织为一条五层管线（信号 → 决策 → 行动 → 记忆，外层由评测与自改进闭环包裹）：

### 4.1 信号层 Signals：什么能触发干预

可观测的触发信号包括：注视/眼手协同、场景变化、时间上下文、用户行为序列、生理与环境传感器。

- **眼手协同**：SIAgent 把注视与手部微动作翻译成自然语言，再由 LLM 推断意图，在 60+ VR 任务上达到 97.2% 的意图识别率（优于仅用注视+捏合的 93.1%）。
- **场景变化 + 任务对齐**：YETI 用相邻帧的结构相似性（SSIM）检测场景切换，并把用户动作与预期任务步骤对齐以标记偏差。
- **分层日常感知**：ProAgent 让廉价线索始终在线、富感知按需触发，并与存储的偏好融合，在 AR 眼镜上实现主动预测准确率 +27.7%、误检 −20.5%。
- **时间本身**：TicToc 测试模型的工具调用决策是否符合人类时间感知，在 76 个场景中，**即使把显式时间戳递到手里，也没有模型超过 65% 的对齐度**——「时间盲」是比「缺信号」更强的发现。
- **注视的前瞻性**：StreamGaze 研究注视能否预测「接下来会发生什么」（anticipation），而非仅当前意图。

### 4.2 决策层 Decision：何时行动（领域核心）

决策机制可粗分为三族：**规则式**（偏离脚本即干预，无成本项，如 YETI）、**决策论式**（显式权衡收益—成本，如 PRISM 与 1999–2017 的通知管理谱系）、**学习式**（用人类接受/拒绝标签训练奖励模型，如 Proactive Agent；或用小型图模型直接产出触发概率，如 TGL）。第 5、6 节将展开。

### 4.3 行动层 Action：决定之后做什么

行动层是相对成熟的「手」：GUI 操作（UI-TARS-2 多轮 RL，OSWorld 47.5 / AndroidWorld 73.3）、视觉理解（Seed1.5-VL 在 38/60 公开基准取得 SOTA）、几何空间定位（Depth Anything 3 从任意视角恢复视觉空间）、毫秒级何时开口（Full-Duplex-Bench 的停顿处理、话轮转换、backchannel、barge-in）。**瓶颈在决策层而非行动层。**

### 4.4 记忆层 Memory：它了解你什么

时机决策是高度个人化的：同一条干预对甲是帮助、对乙是噪音，因此成本模型必须由记忆条件化。记忆基础设施（MemGPT 的 OS 式分页、Mem0 的生产级长期记忆、MemOS 把记忆作为一等 OS 资源）在本知识库的记忆专题已有详述；在主动场景下，SpeechLess 实现了 AR 中个性化空间记忆助手，是「记忆条件化干预」最接近的先例。

### 4.5 评测与自改进层：闭合环路

这是最不标准化、却「卡住一切」的一层：**没有可靠评测，就无法定义干预质量的 RL 奖励。** 详见第 8 节。

---

## 5. 两条互不相通的思想谱系（1999–2026）

TMLR 综述最有价值的史学判断是：**主动干预的两半分别握在两个互不引用的社群手里。**

### 5.1 左线：1999–2022，成本项被形式化，但没有执行者

在任何语言模型能代你行动之前，一个 HCI/普适计算社群花了二十年研究另一半问题：**打断一个人的成本是多少、该如何权衡。**

- **1999 · Horvitz, Jacobs & Hovel《Attention-Sensitive Alerting》**：把告警建模为期望效用——在推断出的潜在注意力状态上求「告警期望成本」，把沉默当作一等动作并做时间贴现，仅当净期望价值为正才触发；推断的消息紧急度与专家打分相关性达 0.9。
- **1999 · Horvitz《混合发起界面原则》**：动作阈值随专注深度升高，加入对话得到沉默/询问/行动三态策略。
- **2002 · McFarlane**：时机是一种「协调」选择——立即/协商/中介/计划式，36 人双任务实验中「协商式」（先告知、让用户自选时机）在多数设计目标上胜出。
- **2003 · Horvitz & Apacible**：从桌面/日历/音频流中以「支付意愿」的美元口径学习打断成本，并预测下一个可低成本打断的窗口；跨用户迁移的模型甚至差于多数类猜测——这是对「人群通用时机策略」的长期警告。
- **2006 · Iqbal & Bailey**：仅凭任务结构（无需传感器）就能预测打断成本，三个计划特征可造成恢复滞后 3.7 倍的跨度（464 → 1,702 ms）。
- **2007 · Iqbal & Horvitz**：27 名知识工作者、2,267 记录小时的现场实测——**一次告警带来约 20–25 分钟的扰动弧，27% 被打断的窗口在两小时后仍未恢复。** 任何把打断成本定价为「模型回答时长」的模型，都偏差了两个数量级。
- **2014 · InterruptMe**：打断成本是路径依赖的——前两小时内已有 8 条通知后，对新通知的好感度下降过半；因此背靠背触发的上下文门控在「可打断性耗尽」时比随机更糟。
- **2017 · Pielot 等**：把问题从「能否打断」推进到「是否是投入的良机」，337 人现场研究中行为历史带来 5 倍提升。
- **2019 · Okoshi 等**：基于断点的延迟调度，382,518 人部署研究点击率最高 +60.7%，随后为 Yahoo! JAPAN App 全部 1,000 万+ Android 用户启用——这条谱系的人口规模级证据。
- **2022 · Chen 等《Opportune Moments in VR》**：可打断性预测在主动式 MR 浪潮的四年前就已进入头显，却被后者忽略。
- **移动健康 JITAI（2016, Nahum-Shani 等）**：给出时机的形式化——决策点、裁剪变量、显式的「什么都不提供」动作，以及把负担/疲劳列为一等成本；但其阈值是理论设定的自由参数，两个期望都未被估计。

**这条线为何沉寂？** 不是工作失败，而是无处复利：当一次完美时机干预的最大收益不过是「用户稍早一点读到一条消息」时，更好的成本模型没有回报。深刻的形式化是为一个**尚不存在的执行者**而建。

### 5.2 右线：2024–2026，有了执行者，成本项碎成四片

执行者现在存在了，但建造它的浪潮把决策层从零重做，且几乎不引用左线。TMLR 综述审计了其引用网络：**右线只有一篇论文引用了左线，而那一篇最终去定价的是算力。**

更诚实的现状判断不是「没人给打断定价」，而是更糟也更可修复的局面：**四个社群各自只定价成本的一个组件，互不引用，也不复用彼此正在独立重新发现的形式化。**

| 成本组件 | 由谁定价（年份） | 状态 |
|---|---|---|
| 误报（false alarm） | PRISM（2026） | 已定价（贝叶斯风险阈值） |
| 认知负荷（cognitive load） | Precision Proactivity（2025） | 已测量，未接入决策 |
| 社交冒犯（social violation） | EgoSocial（2025） | 已命名，未计费 |
| 算力（compute） | TGL / PRPF / ProAgent（2026） | 仅定价算力 |
| 注意力状态（attentional state） | —— | 自 1999 后无人接手 |
| 沉默成本（cost of silence） | —— | 自 1999 后无人接手 |
| 告警疲劳（alert fatigue） | InterruptMe（2014） | 自 2014 后无人接手 |
| 个人化成本 | Sensing What Surveys Miss（2026） | 单一实验室实现 |

**两个没有主人的条目，恰恰是 1999 年就已经拥有的那两个。** 这正是「两条线互不相通」造成的直接损失。

> ![](diagrams/3-two-lineages.html)

---

## 6. 旗舰系统与方法深度拆解

### 6.1 Proactive Agent 与 ProactiveBench（2024.10）：数据驱动的奖励模型范式

清华等团队的 *Proactive Agent: Shifting LLM Agents from Reactive Responses to Active Assistance* 是 LLM 时代主动性的奠基工作。其方法链条：

1. 采集真实世界人类活动，生成主动任务预测；
2. 由人工标注员把预测标注为「接受/拒绝」；
3. 用标注数据训练一个**模拟人类判断的奖励模型**，作为 LLM 主动性的自动评估器；
4. 构建数据生成管线，产出含 **6,790 个事件**的 **ProactiveBench**；
5. 用 ProactiveBench 微调模型，微调后主动提供帮助的 **F1 达 66.47%**，超过所有开源与闭源模型。

**关键教训（单边奖励导致「沉默崩溃」）**：当团队在推理期把单边的奖励模型反馈（只惩罚误报、不珍视及时帮助）回灌时，GPT-4o 的召回率从 **98.11 骤降到 56.76**。这从经验上复现了「决策为什么必须有两个项」——**只罚误报而不估价及时帮助，最优解就是缄默（mutism）。**

### 6.2 PRISM（2026.02）：决策论门控的现代重建

*PRISM: Festina Lente Proactivity —— Risk-Sensitive, Uncertainty-Aware Deliberation for Proactive Agents* 把问题形式化为**成本敏感的选择性干预**，由三部分组成：

- **决策论门控**：推理时仅当一个**经过校准的用户接受概率**超过某个阈值才干预；该阈值由「错过帮助」与「误报」的**非对称成本**导出。这正是 Horvitz 规则用「学习到的概率」替换「启发式概率」的现代版。
- **双过程推理**：取意「欲速则不达（festina lente）」，只在决策边界附近才调用昂贵的 Slow 模式做反事实检查，把算力集中在模糊与高风险情形。
- **门控对齐、模式锁定的蒸馏**：跑完整管线的教师在无标签交互轨迹上提供密集、可执行的监督；学生学习的响应策略与干预门控显式解耦，以便可调、可审计。

**实测（ProactiveBench，摘要原文）**：误报减少 **22.78%**、F1 提升 **20.14** 个点。TMLR 综述给出的绝对口径为：误报率从 **50.22% 降到 22.94%**、F1 从 **66.47 升到 86.61**。

**PRISM 仍看不见的东西**：成本比是一个「部署常量」，作者把它从 1:4 扫到 1.2:1，却从未从任何用户身上实测；因此门控对「打断一个正望向窗外的人」和「打断一个会议中话说到一半的人」收取同样的价格。**这种状态依赖性正是 1999 年的核心洞见，它没有随形式化一起回来。** PRISM 自己在 §6 也承认：LLM 判官代理「无法完全捕捉复杂心流状态中不合时宜打断的认知负荷」。

### 6.3 TGL 门控（2026.05）：决策层或许根本不需要 LLM

*Do Proactive Agents Need an LLM to Decide When to Act?* 的核心洞察是：**用户活动天然具有图结构**——事件涉及持续存在的实体，实体的复现把多次交互连接起来；触发与上下文选择可直接映射为对事件节点与实体节点的预测。

- 提出**时间图学习（TGL）控制器**，一次前向同时产出「触发概率」与「打分后的上下文实体」；
- GPU 服务器上每事件 **11.13 ms**，在九种触发架构中 AUC 最高，相对两种单次前向 LLM 触发器取得约 **4–7× 的触发阶段加速**；
- 共享的 TGL 模型为全部 14 个下游骨干平均提升 F1 **16.7** 个点；
- 在消费级笔记本上 **13.99 ms**、约 **220 MiB** BF16 常驻占用，使主动控制可端侧部署。

### 6.4 ProAgent（2025.12）：按需分层感知的真实世界系统

*ProAgent: Harnessing On-Demand Sensory Contexts* 是一套端到端、落地于 AR 眼镜的真实世界系统：

- **按需分层感知**：廉价上下文线索始终在线、富感知按需触发，持续感知周围环境；
- **面向主动的上下文抽取**：整合感官上下文与人类偏好，得到分层上下文；
- **上下文感知主动推理器**：推断需求并调用外部工具交付帮助。

实测：主动预测准确率最高 **+27.7%**、误检 **−20.5%**；20 人用户研究中 **85%** 满意并愿意日常使用。

### 6.5 Pro²Assist（2026.05）：长程程序任务的步骤感知

*Pro²Assist* 针对多步骤的长程程序任务：用 AR 眼镜的多模态数据做基于运动的感知，从多尺度时间动态与专家知识中抽取步骤化程序上下文，持续推理并在眼镜上显示及时帮助。程序化动作理解准确率超最强基线 **21%** 以上，**主动时机准确率最高达基线的 2.29 倍**；20 人研究中 **90%** 认为有用。

### 6.6 EgoSocial（2025.10）：把社交时机做成指标

*EgoSocial* 构建含 **13,500 个社交视频-问答对**的第一人称数据集，专测社交干预。前沿全模态 LLM 在干预时机上仍严重吃力——**Gemini 2.5 Pro 仅 14.4%**，而它们在「察觉社交互动正在发生」上已能达到 51.65–56.08 的 macro-F1：**看得见局面，却选不准时机。** 其提出的 EgoSoD 用社交思维图融合音视频线索，把 Phi-4 的干预时机提升 45.6%、Gemini 2.5 Pro 提升 9.9%。

### 6.7 Plan, Watch, Recover / EgoProactive 与 Pro²Bench（2026.06，Meta）

*Plan, Watch, Recover* 引入 **EgoProactive** 智能眼镜数据集（四个领域，用户在步骤级偏离计划并从中恢复），结合五个重标注数据集组成 **Pro²Bench**，对干预时机、程序完成度与响应质量打分。

- 零样本下，**没有任何前沿或开放权重模型能在首次出错前完成超过 26% 的程序干预**；
- 提出 **PWR 架构**：在上下文中保持显式计划状态，每次干预后修订计划，偏离后补入恢复步骤；
- 在三个开放权重骨干上，无计划地训练几乎不改变性能，加入动态计划后平均程序完成度从 **20% 升到 27%**，且在未见过的偏离上依然有效。

### 6.8 内容轴的主动：IBM Q&D（2026.09）

*Asking for What Was Never Requested* 研究「决定行动后去追求什么」。用从基准自身分解恢复出的**需求图（need graph）**记录需求间依赖，使水平/垂直主动及「是否在恰当时机停止」都能仅凭转录打分、无需模型判官。其 **Q&D（提问者+起草者）**训练提问者偏好「续写能检索到更多所需证据」的问题：在等检索开销下，训练后的提问者在三个多跳 QA 基准上全面超过同一模型的提示版，并在其中两个上胜过 **15 倍大**的提示模型；迁移到客服智能体后，能以更少提问完成更多任务。

### 6.9 生理信号个性化：Sensing What Surveys Miss（2026.02）

该工作用**皮肤电活动（EDA）与鼠标移动**预测受访者何时需要支持，个性化分类器 + 基于规则的阈值自适应来触发 LLM 澄清。32 人被试内实验：时机对齐的帮助把回答准确率提升 **21%**、假阴性率从 50.9% 降到 22.9%；这是罕见的、对「时机本身」做消融（对齐 vs 错位 vs 随机）的研究。

### 6.10 回避时机问题的最可部署系统：VisionClaw（2026.04）

*VisionClaw* 在 Meta Ray-Ban 眼镜上把实时第一人称感知与 OpenClaw 智能体执行耦合，支持把实物加入购物车、从纸质文档生成笔记、从海报建日程等。12 人实验室 + 5 人长期部署研究显示任务完成更快、交互开销更低。**但它的行动严格由语音发起、由用户充当闸门**——最可部署的 2026 系统以「不拥有时机问题」的方式绕开了时机问题。

---

## 7. 关键架构收敛：廉价门控 / 昂贵执行器

2026 年一个值得专门标记的现象是：**三篇相互独立的论文收敛到同一个架构——用一个廉价的门控器守在昂贵的执行器前面。**

- **PRISM 的审议门控**：校准接受概率 vs 成本导出阈值，只在边界附近调用昂贵 Slow 推理；
- **TGL 时间图触发器**：小图模型先决定是否触发，LLM 只在触发后才被调用（4–7× 加速、约 220 MiB 端侧可跑）；
- **Pre-Reasoning Perception Framework（PRPF，2026.06）**：在推理之前先做一层廉价感知，为高效可靠的移动智能体把关。

这一收敛的工程含义清晰：

1. **时机判断是高频、低延迟、需要常驻的决策**，让一个每事件毫秒级、几百 MiB 的小模型（甚至非神经网络的规则/生理阈值）来做，比让 LLM 反复解读活动历史更合理——后者在助手保持沉默时也在持续支付推理成本。
2. **昂贵的 LLM 只应在门控放行后处理「说什么、怎么做」**。
3. 这同时是一条成本治理路线：把「算力」这一可度量的成本项接到门控上，是目前唯一被右线真正定价的成本。

但要警惕：廉价门控解决的是「何时值得花算力」，**它并不自动等于「何时值得打扰用户」**——后者需要的注意力状态/认知负荷项仍是空的（见第 5.2 节）。把算力门误当成打扰门，是这个架构最容易犯的范畴错误。

> ![](diagrams/4-cheap-gate-architecture.html)

---

## 8. 评测体系与「reward ⊆ eval」

### 8.1 为什么评测是卡住一切的一层

强化学习在奖励有真值（ground truth）时才成立。GUI 智能体有「任务是否完成」作为奖励，而**干预质量没有等价物**：没有人能断言某一时刻的某次干预究竟好不好。因此：

> **reward ⊆ eval（奖励包含于评测）**：主动智能体可被训练的目标，上界于这个领域能构造出的评测。在干预质量可度量之前，主动性的 RL 可以被运行，却无法被验证。

移动健康是反例也是范本：它能在部署中在线学习干预时机，是因为有一个经过验证的近端结果可学（如提示后 30 分钟内的步数）。开放世界辅助没有这样的结果，于是今天写下的任何奖励，定价的都是一个尚未被证明能度量「真正重要之物」的代理。

### 8.2 反馈的四个深度层级

每个已部署的主动系统其实都在持续产生奖励流，多数却被丢弃。反馈随深度递增，信号越真、归因越难：

| 层级 | 信号 | 成本 | 归因 |
|---|---|---|---|
| L0 显式 | 接受 / 忽略 / 拒绝 /「三天后再提醒」 | 免费但稀疏、有偏 | 平凡 |
| L1 隐式行为 | 是否采纳建议、注视停留 vs 回避、打断后恢复时间、撤销 | 稠密、零用户负担 | 易—中 |
| L2 结果 | 任务完成度变化、节省时间、避免的错误 | 需任务埋点 | 延迟 → 信用分配难 |
| L3 长期 | 留存、信任、用户彻底关闭主动性（终极负奖励） | 仅人群级 A/B | 最难 |

实践中可见三种成熟度：**(1) 只记录**（多数产品）；**(2) 阈值自适应**——按用户/上下文以 bandit 方式更新 θ 这一个参数，是「最小可行的 RL」，也应是常驻助手的起点；**(3) 策略学习**——在干预上学习奖励模型（Proactive Agent 是其离线版）。

**核心不对称**：负信号廉价而丰富（每次忽略都是一个），正信号微弱而延迟（没被打扰的人什么都不会说，明天才省下的十分钟今天不产生任何信号）。这就是为什么自改进被画在评测层内部而非独立的第六阶段：**反馈边的质量，上限由定义它的度量决定。**

### 8.3 为什么「接受率」不能当奖励

时机选得好时接受率约 0.676，随机时机 0.444，反时机 0.388——接受率确实携带时机信号。但**只看接受率会把 PRISM 的误报率推高到 62.50%**，因为用户常常会接受「内容有用但时机糟糕」的建议。接受率度量的是「我是否想要这个内容」，而非「我是否想要它在此刻出现」。它混淆了 *I wanted this* 与 *I wanted this now*。

### 8.4 基准矩阵：没有一个基准答全三个问题

一个上线的时机策略必须能回答：**你选择行动的时刻里，有多少是受欢迎的？需要你的时刻里，你抓住了多少？你的错误代价几何？**

| 基准 | 受欢迎？ | 被抓住？ | 被定价？ | 说明 |
|---|---|---|---|---|
| HoloAssist（2023） | ○ | ○ | ○ | 166h 第一人称协作操作；窗口锚定在「已知会发生」的干预上，只分类干预类型、从不判断是否该开口 |
| ProactiveBench（2024） | ● | ◐ | ○ | 6,790 事件接受/拒绝标签 |
| ProactiveEval（2025） | ○ | ○ | ○ | 主动对话体系统一框架 |
| ProAgentBench（2026） | ● | ● | ○ | 真实世界数据的主动辅助评测 |
| EgoSocial（2025） | ◐ | ◐ | ○ | 社交时机即指标；前沿模型 12.50–17.67% |
| ProactiveVideoQA / PAUC（2025） | ◐ | ● | ○ | 首个经人验证的时机动态指标 PAUC，但开篇即无成本项 |
| Pro²Bench（2026） | ● | ● | ○ | 五基准统一；但其质量分对假阳性和假阴性都记 0 |
| TMLR §8 设计 | ● | ● | ● | must/may/must-not 三态窗口 + 成本加权曲线 |

第三列（错误是否被定价）整列为空，这正是领域内任意两个系统都无法在「本问题真正关心的量」上被比较的原因。最尖锐的例子：**Pro²Bench 的质量分对假阳性和假阴性都给 0，于是「在别人倒东西时插嘴说无关的话」和「眼睁睁看着菜烧糊」被当成相同结果**——再强的策略学习也修不好一个对问题本质差异无动于衷的奖励。

### 8.5 TMLR §8.4：三态窗口与四个一起报告的数

TMLR 综述提出的标注协议让标注员标记决策**窗口**（而非瞬间），每个窗口给三态标签：**must（必须干预）/ may（可以干预）/ must-not（禁止干预）**。三态是全部要点：二元标签会把 *may* 的质量强行压向某一侧，抹掉那块「靠成本—收益推理而非分类」发挥作用的区域。

由此产出四个**必须一起报告、不可压成一个分**的数：must 窗口上的时机召回、must-not 窗口上的冒犯率、成本加权分（其兑换率以**曲线扫描**而非隐藏常量给出）、以及分级动作折扣。在 *may* 窗口保持沉默永不被罚（避免 ProactiveBench 已暴露的单边惩罚错误）。

其机器打分试点（规则播种、尚非人工判定）在 1,758 个会话、131.5 小时、4,144 个 must / 21,707 个 may / 3,170 个 must-not 窗口上跑了五个参考策略：「每秒都说」的策略时机召回最高（0.962）却在每个兑换率下成本加权分最低（λ=1 时 −367,752）；一个抓住 70% 错误、并在用户说话时保持沉默的模拟检测器，以 0.09%（对比 0.82%）的冒犯率达到 0.703 召回、λ=1 时 +5,932；但从 λ=2 起「永不开口」反而更优——这正说明兑换率必须作为曲线报告。

---

## 9. 产业落地：以 Claude Duet 为镜

2026 年 10 月 21 日，Anthropic 开放 **Claude Duet** 研究预览，这是第一个围绕「主动回路」构建的 Anthropic 产品，也是观察产业如何处理时机问题的最佳样本（以下事实依据 Anthropic 官方产品页及公开资料，研究预览阶段，功能可能变化）。

**产品形态**：独立应用（iOS / Android / Web），运行在 Claude Opus 5.5 上，面向美国 Claude Pro/Max 用户候补；研究预览期不额外收费、共享订阅额度。

**三个相对普通 Claude 的新增能力**：一条持续的对话线程、一台云端私有计算机（带浏览器的云 VM，用户设备关机也能继续工作）、以及**主动发起对话**的能力。

**它如何处理「主动」与「可控」的张力**：

- **周期性而非连续监控**：每隔几小时检查一次已连接账户（Gmail、Slack、QuickBooks、Stripe、HubSpot、DocuSign、Shopify 等），用户可指定关注内容与频率；官方明确提示它不保证实时、可能误判紧急程度，频繁后台活动会更快消耗订阅额度。
- **分级授权**：读取邮件/日历/文件、草拟回复/报告可自主进行；**发送消息、支付、签署文件、删除数据必须先请求批准**；证券交易、账户间转账、访问他人账户则即便批准也禁止。
- **安全架构**：凭证隔离、外部审批层、敏感数据分类器——与构建可信企业级智能体所需的模式一致。

**对照本报告框架的解读**：Claude Duet 在行动层（云 VM + 连接器）与记忆层（复用既有 Claude 上下文）上相当完整，但在决策层刻意保守——用「低频周期检查 + 重大动作人工审批」这种**制度性约束**替代了尚未成熟的个性化时机模型。它把时机风险主要转化为「审批门」而非「智能门」，这与学术前沿「廉价门控 + 昂贵执行器」的收敛方向一致，只是把门控器换成了人。这也印证了第 10 节的判断：在注意力状态成本项被可靠度量之前，**人在环（human-in-the-loop）的审批是当前唯一可规模化的安全解。**

同月，Google（Gemini 的 Tasks 后台模式）、Meta（把持续记忆与主动聚焦做成模型级能力）等也朝同一方向演进；产业共识已经从「长时执行」进一步走向「无需提示的主动工作」，但持久上下文、主动性、后台执行与审批检查点正在成为消费级与专业级智能体的共同特征。

---

## 10. 开放问题

1. **无真值下的奖励。** GUI 智能体有任务完成作奖励，干预质量没有等价物。什么样的评测能让 RL 奖励变得可定义？（reward ⊆ eval）
2. **统一的成本项 θ。** 截至 2026 年，成本被碎成四块（误报/认知负荷/社交冒犯/算力）分别建模，无人把它们统一进一个 θ，也无人继承 1999–2017 的效用形式化。最近的可行修复是「替代」而非「综合」：把 PRISM 形状正确的阈值里那个被扫描的误报成本，替换成 Precision Proactivity 已证明可从交互轨迹算出的负荷估计。
3. **无脚本的时机。** YETI 式偏差检测需要任务脚本，开放世界日常场景没有脚本。
4. **延迟收益的信用分配。** 一次「明天才省下十分钟」的干预，今天收不到任何信号。
5. **个性化阈值。** 同一个 θ 对不同用户、情境、时段都是错的，时机策略必须由记忆条件化；而 2003 年的跨用户迁移失败警告仍在。
6. **基准碎片化。** 各基准测量不同切片，Pro²Bench 的五合一仍是脚本锚定的，没有任何基准端到端覆盖开放世界 MR 干预时机。
7. **模型的时间盲。** TicToc 表明即便递上时间戳，模型对人类时间感知的对齐也不超过 65%；「经过多久」本身作为信号尚未被模型真正掌握。
8. **主动性的安全与隐私。** 始终在线感知放大监控面；主动执行叠加凭证与网络后，越权与提示注入的后果比被动助手严重得多（与本知识库机制可解释性、Agent 安全报告呼应）。端侧优先、数据最小化、一键关闭与可解释的提议理由是基本要求。

---

## 11. 工程落地 Checklist

**范围与分级**

- [ ] 从单一窄场景切入（如周报提醒），不要一上来做全场景；按 L0→L4 逐级放开主动性。
- [ ] 明确三态策略：保持沉默 / 先询问 / 直接行动；默认偏保守。
- [ ] 为每个动作标注风险等级，发送/支付/签署/删除一律人工审批，红线动作（转账、交易）直接禁止。

**决策层**

- [ ] 显式实现效用门控 U = V − λ·D > θ，而非无条件触发或让 LLM 自由发挥。
- [ ] 采用「廉价门控 / 昂贵执行器」：毫秒级小模型/规则做高频触发，LLM 仅在放行后生成内容。
- [ ] 成本比不要写死成全局常量；至少接入可从交互轨迹估计的认知负荷/忙碌状态，并支持按用户、时段、场景个性化。
- [ ] 尊重「可打断性耗尽」：避免背靠背触发，纳入告警疲劳与路径依赖。

**反馈与评测**

- [ ] 同时采集 L0 显式与 L1 隐式反馈（采纳率、注视、恢复时间、撤销），不要只记不用。
- [ ] 从「单参数 bandit 调 θ」起步做在线自适应，而非直接上策略学习。
- [ ] 绝不能只用接受率当奖励（会混淆「想要内容」与「想要此刻」）；必须同时统计冒犯率与时机召回。
- [ ] 用 must/may/must-not 三态窗口评估，成本兑换率以曲线报告；在 may 窗口沉默不罚。

**系统与信任**

- [ ] 每条主动提议都给出可解释理由；提供一键关闭主动功能、自定义免打扰时段、查看/删除全部记忆。
- [ ] 端侧优先、数据最小化；凭证隔离、外部审批层、敏感数据分类器。
- [ ] 周期性检查 + 明确非实时承诺，管理用户对紧急性的预期；监控后台算力/额度消耗。

---

## 12. 结语

主动式智能体把 AI 交互的基本问题从「用户适应机器」翻转为「机器主动适应用户」。但 2026 年的证据给出一个反直觉的结论：**这一跃的瓶颈不在更强的执行者，而在一个被忽视了二十年的成本项。** 1999–2017 年的 HCI 社群把「打断值多少钱」研究得极为透彻，却没有能交付收益的执行者；2024–2026 年的 LLM 浪潮拥有了能完成任务的执行者，却把决策层推倒重做，让成本碎成四片、让两个最关键的成本（注意力状态、沉默成本）长期无人认领。

好消息是，桥梁已经出现：PRISM 用学习概率重建了 Horvitz 的阈值，TGL 证明时机判断可以不需要 LLM，三篇独立工作收敛到「廉价门控/昂贵执行器」，TMLR 的三态窗口第一次让「错误的代价」可被度量。当有人把可测量的认知负荷放进形状正确的阈值、把个性化记忆接进 θ、并用三态评测闭合奖励回路时，主动式智能体才会从「会插嘴的工具」真正变成「懂得分寸的同事」。

**在下一个十年，衡量一个智能体高下的，可能不再只是它能做多少事，而是它有多少次恰当地选择了什么都不做。**

---

## 参考文献

> 说明：arXiv 编号均已经 arXiv 官方 API（export.arxiv.org/api）于 2026-11-01 逐一核实标题与发布日期；HCI/普适计算经典文献给出 ACM/Springer 正式 DOI；TMLR 综述给出 OpenReview 与 Zenodo。厂商产品事实标注「官方口径」。

### 一、综述与领域地图

1. Tao An. *When Should the Agent Speak? A Survey of Intervention Timing for Always-On AI Assistants.* Transactions on Machine Learning Research (TMLR), 2026. OpenReview: https://openreview.net/forum?id=b0yKEdAXEr    Zenodo 存档（DOI 始终指向最新版）: https://doi.org/10.5281/zenodo.21438396
2. Tao An. *The Missing Cost Term*（综述交互伴读文章）, 2026-07-14（更新 2026-09-28）. https://tao-hpu.github.io/articles/intervention-timing
3. Tao An. *awesome-proactive-agents*（活的领域地图）. GitHub. https://github.com/tao-hpu/awesome-proactive-agents
4. Junlong Li, Huaiyuan Xu, Sijie Cheng, Kejun Wu, et al. *Building Egocentric Procedural AI Assistant: Methods, Benchmarks, and Challenges.* arXiv:2511.13261, 2025. https://arxiv.org/abs/2511.13261
5. Deng et al. *Proactive Conversational AI.* ACM Transactions on Information Systems (TOIS), 2025. https://doi.org/10.1145/3715097

### 二、决策论与注意力管理经典谱系（1999–2022）

6. Eric Horvitz, Andy Jacobs, David Hovel. *Attention-Sensitive Alerting.* UAI 1999（arXiv 存档:1301.6707）. https://arxiv.org/abs/1301.6707
7. Eric Horvitz. *Principles of Mixed-Initiative User Interfaces.* CHI 1999. https://doi.org/10.1145/302979.303030
8. Daniel McFarlane. *Comparison of Four Coordinating Methods for the-Way Human-Computer Interruption.* Human–Computer Interaction, 2002. https://doi.org/10.1207/S15327051HCI1701_2
9. Eric Horvitz, Johnson Apacible. *Learning and Reasoning about Interruption.* ICMI 2003. https://www.microsoft.com/en-us/research/publication/learning-and-reasoning-about-interruption/
10. Shamsi T. Iqbal, Brian P. Bailey. *Leveraging Task Structure to Improve the Cost of Interruption.* CHI 2006. https://doi.org/10.1145/1124772.1124882
11. Shamsi T. Iqbal, Eric Horvitz. *Disruption and Recovery of Computing Tasks.* CHI 2007. https://doi.org/10.1145/1240624.1240730
12. Veljko Pejović, Mirco Musolesi. *InterruptMe: Designing Intelligent Prompting Mechanisms for Pervasive Applications.* UbiComp 2014. https://doi.org/10.1145/2632048.2632062
13. Martin Pielot, et al. *Beyond Interruptibility: Predicting Opportune Moments to Engage.* IMWUT/UbiComp 2017. https://dl.acm.org/doi/10.1145/3130956
14. Tadashi Okoshi, et al. *Adaptive Notification Scheduling for Production.* KDD 2019. https://doi.org/10.1145/3292500.3330732
15. Chen, Chang, Chan. *Opportune Moments in VR.* CHI 2022. https://doi.org/10.1145/3491102.3517529
16. Inbal Nahum-Shani, et al. *Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health.* Annals of Behavioral Medicine, 2018. https://doi.org/10.1007/s12160-016-9830-8
17. Anderson, et al. *Attention Management Systems: A Survey.* IMWUT 2018（arXiv:1806.06771）. https://arxiv.org/abs/1806.06771

### 三、LLM 时代的主动智能体（2024–2026）

18. Yaxi Lu, Shenzhi Yang, Cheng Qian, Guirong Chen, et al. *Proactive Agent: Shifting LLM Agents from Reactive Responses to Active Assistance.* arXiv:2410.12361, 2024. https://arxiv.org/abs/2410.12361
19. Yuxuan Fu, Xiaoyu Tan, Teqi Hao, Chen Zhan, et al. *PRISM: Festina Lente Proactivity — Risk-Sensitive, Uncertainty-Aware Deliberation for Proactive Agents.* arXiv:2602.01532, 2026. 项目页: https://prism-festinalente.github.io/
20. Xiaoze Liu, Ruowang Zhang, Amir H. Abdi, Michel Galley, et al. *Do Proactive Agents Need an LLM to Decide When to Act?* arXiv:2605.30152, 2026. https://arxiv.org/abs/2605.30152
21. Bufang Yang, Lilin Xu, Liekang Zeng, Yunqi Guo, et al. *ProAgent: Harnessing On-Demand Sensory Contexts for Proactive LLM Agent Systems in the Wild.* arXiv:2512.06721, 2025. https://arxiv.org/abs/2512.06721
22. Lilin Xu, Bufang Yang, Siyang Jiang, Kaiwei Liu, et al. *Pro²Assist: Continuous Step-aware Proactive Assistance with Multi-modal Egocentric Perception for Long-horizon Procedural Tasks.* arXiv:2605.04227, 2026. https://arxiv.org/abs/2605.04227
23. Saptarashmi Bandyopadhyay, et al. *YETI (YET to Intervene): Proactive Interventions by Multimodal AI Agents in Augmented Reality Tasks.* arXiv:2501.09355, 2025. https://arxiv.org/abs/2501.09355
24. Xijun Wang, Tanay Sharma, Achin Kulshrestha, Abhimitra Meka, et al. *EgoSocial: Benchmarking Proactive Intervention Ability of Omnimodal LLMs via Egocentric Social Interaction Perception.* arXiv:2510.13105, 2025. https://arxiv.org/abs/2510.13105
25. Kaustav Kundu, Ritvik Shrivastava, Maxim Arap, Nanshu Wang, et al. *Plan, Watch, Recover: A Benchmark and Architectures for Proactive Procedural Assistance.* arXiv:2606.04970, 2026. https://arxiv.org/abs/2606.04970
26. Ido Levy, Asaf Yehudai, Segev Shlomov, Asaf Adi. *Asking for What Was Never Requested: Horizontal and Vertical Proactivity in Agents.* arXiv:2609.37236, 2026. https://arxiv.org/abs/2609.37236
27. Ailin Liu, Yesmine Karoui, Fiona Draxler, Frauke Kreuter, et al. *Sensing What Surveys Miss: Understanding and Personalizing Proactive LLM Support by User Modeling.* arXiv:2602.00880, 2026. https://arxiv.org/abs/2602.00880
28. Xiaoan Liu, DaeHo Lee, Eric J. Gonzalez, Mar Gonzalez-Franco, et al. *VisionClaw: Always-On AI Agents through Smart Glasses.* arXiv:2604.03486, 2026. https://arxiv.org/abs/2604.03486
29. Yichi Zhang, Xin Luna Dong, Zhaojiang Lin, Andrea Madotto, et al. *Proactive Assistant Dialogue Generation from Streaming Egocentric Videos.* arXiv:2506.05904, 2025. https://arxiv.org/abs/2506.05904
30. Zhimin Wang, Chenyu Gu, Feng Lu. *SIAgent: Spatial Interaction Agent via LLM-powered Eye-Hand Motion Intent Understanding in VR.* arXiv:2603.00522, 2026. https://arxiv.org/abs/2603.00522
31. Yize Cheng, Arshia Soltani Moakhar, Chenrui Fan, Parsa Hosseini, et al. *Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception.* arXiv:2510.23853, 2025. https://arxiv.org/abs/2510.23853
32. Daeun Lee, Subhojyoti Mukherjee, Branislav Kveton, Ryan A. Rossi, et al. *StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos.* arXiv:2512.01707, 2025. https://arxiv.org/abs/2512.01707
33. Sicheng Yang, Yukai Huang, Weitong Cai, Shitong Sun, et al. *Egocentric Co-Pilot: Web-Native Smart-Glasses Agents for Assistive Egocentric AI.* arXiv:2603.01104, 2026. https://arxiv.org/abs/2603.01104
34. Zhijie Ding, Weinan Hong, Zicheng Zhu, Lei Li, et al. *Perceive Before Reasoning: A Pre-Reasoning Perception Framework for Efficient and Reliable Proactive Mobile Agents.* arXiv:2606.03236, 2026. https://arxiv.org/abs/2606.03236
35. Tianjian Liu, Fanqi Wan, Jiajian Guo, Xiaojun Quan. *ProactiveEval: A Unified Evaluation Framework for Proactive Dialogue Agents.* arXiv:2508.20973, 2025. https://arxiv.org/abs/2508.20973
36. Yuanbo Tang, Huaze Tang, Tingyu Cao, Lam Nguyen, et al. *ProAgentBench: Evaluating LLM Agents for Proactive Assistance with Real-World Data.* arXiv:2602.04482, 2026. https://arxiv.org/abs/2602.04482
37. Yueqian Wang, Xiaojun Meng, Yifan Wang, Huishuai Zhang, et al. *ProactiveVideoQA: A Comprehensive Benchmark Evaluating Proactive Interactions in Video Large Language Models.* arXiv:2507.09313, 2025. https://arxiv.org/abs/2507.09313
38. Brandon Lepine, Juho Kim, Pamela Mishkin, Matthew Beane. *Precision Proactivity: Measuring Cognitive Load in Real-World AI-Assisted Work.* arXiv:2505.10742, 2025. https://arxiv.org/abs/2505.10742
39. Yoonsang Kim, Devshree Jadeja, Divyansh Pradhan, Yalong Yang, et al. *SpeechLess: Micro-utterance with Personalized Spatial Memory-aware Assistant in Everyday Augmented Reality.* arXiv:2602.00793, 2026. https://arxiv.org/abs/2602.00793

### 四、支撑性感知 / 行动 / 记忆基础设施

40. Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, et al. *UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning.* arXiv:2509.02544, 2025. https://arxiv.org/abs/2509.02544
41. Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, et al. *Seed1.5-VL Technical Report.* arXiv:2505.07062, 2025. https://arxiv.org/abs/2505.07062
42. Haotong Lin, Sili Chen, Junhao Liew, Donny Y. Chen, et al. *Depth Anything 3: Recovering the Visual Space from Any Views.* arXiv:2511.10647, 2025. https://arxiv.org/abs/2511.10647
43. Guan-Ting Lin, Jiachen Lian, Tingle Li, Qirui Wang, et al. *Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities.* arXiv:2503.04721, 2025. https://arxiv.org/abs/2503.04721
44. Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, et al. *MemGPT: Towards LLMs as Operating Systems.* arXiv:2310.08560, 2023. https://arxiv.org/abs/2310.08560
45. Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, et al. *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory.* arXiv:2504.19413, 2025. https://arxiv.org/abs/2504.19413
46. Zhiyu Li, Chenyang Xi, Chunyu Li, Ding Chen, et al. *MemOS: A Memory OS for AI System.* arXiv:2507.03724, 2025. https://arxiv.org/abs/2507.03724
47. Xin Wang, Taein Kwon, Mahdi Rad, Bowen Pan, et al. *HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World.* arXiv:2309.17024, 2023（ICCV）. https://arxiv.org/abs/2309.17024
48. Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, et al. *Ego4D: Around the World in 3,000 Hours of Egocentric Video.* arXiv:2110.07058, 2021（CVPR）. https://arxiv.org/abs/2110.07058

### 五、产业产品与趋势信源

49. Anthropic. *Claude Duet*（研究预览，2026-10-21 开放）. 官方产品页（官方口径）: https://claude.com/product/claude-duet
50. LLM Trends. *LLM Trends 2026: Monthly AI Model & Agent Updates*（2026-10 月条目：Agents stop waiting for prompts）. https://llmtrends.com

---

*本报告所有 arXiv 编号已经 arXiv 官方 API 逐一核实；经典 HCI 文献为 ACM/Springer 正式 DOI；产业事实以厂商官方页为准并标注「官方口径」，聚合趋势站标注为二手信源。报告为技术调研，不构成任何投资或采购建议。*
