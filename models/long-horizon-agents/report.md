# 长程智能体（Long-Horizon Agents）前沿技术报告：当 AI 从「对话框」走向「数小时无人值守的自主工作」

> 主题：把「LLM 跑在一个循环里」沿时间轴拉长到数百轮、数小时乃至跨会话后，智能体为什么会崩、又该如何被工程化地救回来——这是 2026 年大模型系统层面最活跃、且与「更强的模型」正交的新方向。
>
> 信源：65 篇 arXiv 论文编号经 arxiv.org 逐一核实（0 错号）+ 红杉资本 / Anthropic / LangChain 一手工程材料交叉验证；厂商与第三方口径分别标注。
>
> 报告日期：2026-11-15 ｜ 架构图 ×4（手工暗色 SVG，已无头浏览器截图目检）

---

## 目录

0. 摘要与核心判断（TL;DR）
1. 什么是长程智能体：从「单步对不对」到「能否持续推进」
2. 为什么是现在：三重成熟与「元年级」信号（含说法审计）
3. 思想源流与四年演进（2022–2026）
4. 参考架构：六层栈与「状态外部化」
5. 核心技术挑战与前沿方法（七大子问题）
6. 评测革命：密集奖励、过程分与时间预算
7. 训练范式：长程 RL、信用分配与 harness / RSI 自进化
8. 产业与开源生态地图
9. 开放问题
10. 工程落地 Checklist
11. 结语
· 参考与信源说明 ／ 参考文献

---

## 0. 摘要与核心判断（TL;DR）

**一句话定义。** 长程智能体（Long-Horizon Agent）指能够在**很少或没有人类干预**的情况下，围绕一个开放目标，自主运行**数百个工具调用步、数十分钟到数小时甚至跨会话**，并在中途管理自己的计划、上下文、状态与错误恢复的 LLM 智能体。它的对立面不是「弱模型」，而是「只能完成几分钟内、定义良好的短任务」的智能体。

**为什么这是一个独立的新方向，而不是旧瓶装新酒。** 当交互被沿时间轴拉长后，出现了一组在短任务里根本不存在、也无法靠「把模型换强」单独解决的系统级问题：上下文窗口必然被撑爆、早期错误沿决策链累积、智能体分不清「我已经做完了」和「我只是通过了能看见的测试」、外部世界在它等待审批时已经改变、一次运行能过但换次运行就崩。2026 年的论文集群表明，**瓶颈正从「局部推理能力」转向「长程完成能力」**——这是一条与模型 scaling 正交、主要由 harness 与状态系统决定的能力轴。

**五个有数据支撑的核心判断：**

1. **最强模型在真正的长程任务上依旧大面积失败，且主要输在「时间管理」而非「智商」。** Long-Horizon-Terminal-Bench（LHTB，46 个需数百轮、分钟到小时级的容器化终端任务）上，15 个前沿模型中最强者 pass@1（阈值 0.95）也只有 **15.2%**，跨模型均值仅 **4.3%**；平均每个任务消耗 **9.9M tokens、约 231 个 episode、85.3 分钟** [13]。其后续 LHTB 2.0 把时限延长到 3 小时、扩到 77 个任务 [14]。
2. **评分范式必须从「二元成败」换成「过程密集进度」。** 在极难长任务上，「完成 90%」和「完成 5%」在 pass/fail 下都记 0 分，无法区分模型。LHTB 2.0 的 30 个重构任务上，二元验证器平均分仅 **0.221**，而密集进度阶梯分平均 **0.653** [14]——同一批智能体，换把尺子量出完全不同的图景。
3. **可靠性越来越多地来自 harness 与外部状态，而不是模型权重。** 在「模型冻结」前提下，仅靠优化脚手架，StateM 在 Terminal-Bench 2.1 上拿到 **95.3% 原始准确率**，或一次仅 **$15** 的前沿模型运行 [48]；RiseHarness 专门翻转「同一配置下一次成功、一次失败」的任务 [27]。这与「Agent = Model + Harness」、时间越长后者权重越大的判断一致。
4. **上下文「留得住」不等于「用得上」。** PACE 在 100K token 的编码会话里，即便把后期所需 token 的 99%+ 都保留在 KV cache（驻留率 0.996），当定义与使用间隔从 2K 拉到 90K token 时，后期步骤准确率仍从 **93% 掉到 64%** [17]——长程失败是「可检索性 / 干扰」问题，不只是「有没有存下来」。
5. **多智能体与长程是用 token 换覆盖度的昂贵解。** Anthropic 研究系统采用 orchestrator–worker 模式、主编排并行拉起 3–5 个子代理，复杂查询研究时间最多缩短 **90%**；但实测单智能体约消耗 chat 的 **4×** token、多智能体系统约 **15×**，且在 BrowseComp 上 token 用量单项就解释了约 **80%** 的性能方差 [61]。经济账决定了它只在高价值任务上成立。

**与知识库既有报告的分工（去重边界）。** 本库已有《Agent Harness 智能体脚手架》《上下文工程 Context Engineering》《智能体记忆 Agent Memory》《主动式智能体 Proactive Agent》《Agentic RL》等报告，它们分别从**脚手架组件、token 生命周期、记忆机制、无提示干预时机、多轮 RL 算法**切入；本报告第一次以**时间维度**为唯一主线，系统回答「当运行被拉长到数小时，系统在哪些点上失效、业界用什么机制把成功率拉回来、又该如何评测与训练」，并把上述组件重新组织进一个长程可靠性框架。

---

## 1. 什么是长程智能体：从「单步对不对」到「能否持续推进」

### 1.1 最小机制：LLM-in-a-loop，以及它的时间维度

智能体的核心算法极其简单，LangChain 创始人 Harrison Chase 将其概括为「让一个 LLM 跑在循环里、自己决定把什么拉进上下文」（run the LLM in a loop and let it orchestrate its own context），并指出这正是从 AutoGPT 起就抓住大众想象力的那个原语 [60]。一个终端型智能体的循环是：模型生成动作（Shell 命令 / 文件编辑 / 工具调用）→ 在环境中执行 → 得到观测（输出、文件列表、报错）→ 追加进上下文 → 生成下一个动作。

这个循环本身并不新。**「长程」之为问题，在于把同一个循环沿时间轴拉长后，规模发生质变：**

- **交互长度**：从单轮 / 数轮，拉长到数百个 episode（LHTB 平均约 231 个 [13]），Infrathon 这类基础设施优化任务甚至给每个 run **24 小时**预算、150 次 run 合计消耗逾 **10,000 GPU-小时** [37]；
- **上下文体量**：从一个窗口放得下，膨胀到单任务近千万 token（LHTB 平均 9.9M [13]），远超单卡 KV cache 容量；
- **状态依赖**：从「这一步只依赖上一步」，变成「第 200 步依赖第 5 步定下的某个变量 / 子目标」；
- **评分结构**：从「有唯一正确答案、一次判定」，变成「开放工作流、只有参考解、需要量化走了多远」。

> ![](diagrams/1-time-dimension.html)
> **图 1 · 长程智能体：把「观察—思考—行动」循环沿时间轴拉长后的新瓶颈。** 短程 benchmark 考「单步对不对」（局部推理），长程 benchmark 考「能否在预算内持续推进并收尾」（长程完成）。数据锚点来自 LHTB：平均约 231 episodes、9.9M tokens、85.3 分钟，最强模型 pass@1 仅 15.2% [13]。红色曲线示意无长程机制时成功率随步数衰减，绿色虚线示意叠加状态外部化 / 压缩 / 恢复后的改善（示意，非逐点实测）。

### 1.2 两种能力维度：局部推理 vs. 长程完成

LHTB 明确区分两类彼此独立的能力 [13]：

- **局部推理能力（local reasoning）**：能不能正确执行单步操作、当前动作是否符合一个合理的计划。SWE-Bench、HumanEval、Terminal-Bench 2 等传统基准主要测它。
- **长程完成能力（long-horizon completion）**：能不能在数百步内持续推进、管理上下文、避免回头重做、追踪哪些子目标尚未完成、并在时间预算内收尾。

LHTB 的关键观察是：**智能体失败往往不是因为某一步推理错了，而是因为**在不必要的验证循环上空转、重复已经完成的步骤、无法追踪未完成的子目标、或者任务已完成大半却耗尽时间 [13]。这是与「答案错误」截然不同的失败模式，意味着改进方向也不同——它需要的是更好的进度追踪 / 记忆、更高效的探索、更精准的停止标准，而不只是更强的单步推理。

### 1.3 三个常被混用的概念，先划清边界

- **长程（long-horizon）≠ 长上下文（long-context）。** 长上下文是「一个窗口能装多少 token」的**模型/显存**问题；长程是「一个目标要跑多少步、多久、跨多少状态」的**时间与系统**问题。PACE 恰恰证明二者解耦：token 都在缓存里（长上下文满足），长程依然失败（可检索性不足）[17]。
- **长程 ≠ 多智能体。** 单智能体也可以很长程（一个编码 agent 跑两小时）；多智能体是一种用并行与上下文隔离换取覆盖度的**编排手段**，常用于长程任务但非其定义。
- **长程 ≠ 自主（autonomous）。** 自主强调「无需人批准」，长程强调「时间跨度与状态连续性」。生产中的长程智能体常常是「长程 + 关键节点人类审批」，产出的是**高质量初稿（first draft）**而非直接交付——Chase 指出编码（提 PR 而非直接上生产）、AI SRE（交人复核）、研报（人编辑后再发）、客服升级（一线失败转后台长程 agent 再交人）都是这个模式 [60]。

---

## 2. 为什么是现在：三重成熟与「元年级」信号

### 2.1 为什么 AutoGPT（2023）失败了，而 2026 年「突然能用」

AutoGPT 在 2023 年引爆想象，但长期停留在 demo。Chase 的复盘很直白：当年的核心思想（LLM 自己跑循环）完全正确，**只是模型不够好、包裹模型的脚手架与 harness 也不够好**；此后模型变强，业界也用几年时间搞清了「什么才是好 harness」，两者叠加，循环才真正开始工作，而且**最先在编码领域跑通**，再向 SRE、研究、客服等领域扩散 [60]。他把拐点体感放在 2025 年末（Opus 4.5 前后叠加寒假大量开发者使用编码 agent），并将范式描述为从「手工 scaffolds（脚手架）」走向「harnesses（厚运行时）」[60]。

因此「为什么是现在」可拆成三重同时成熟：

1. **模型成熟**：工具调用、指令遵循、长上下文与（2025 年后的）推理模型，使循环中的单步质量足够高；
2. **harness 成熟**：规划工具、上下文压缩（compaction）、子代理隔离、文件系统外挂、prompt cache、停止判据等被系统性地沉淀进运行时；
3. **评测成熟**：Terminal-Bench、LHTB 等「真实、困难、容器化、带过程奖励」的长程基准出现，让「到底走了多远」第一次可度量。

### 2.2 「2026 元年级」的产业信号（含说法审计）

- **红杉资本 × LangChain**：红杉 Training Data 播客第 77 期以《Context Engineering Our Way to Long-Horizon Agents》为题对谈 Harrison Chase，标题即把「长程智能体」与「上下文工程」绑定；Chase 在节目中反复使用 long-horizon framing，称编码是最先爆发的领域，并判断记忆（跨更长时间尺度的上下文工程）与「让模型自己决定何时压缩」是接下来的重点 [60]。
- **LangChain 把长程做成产品原语**：其开源 harness **Deep Agents** 官方自我定义即「为长时运行任务构建的开源 agent harness」，内置规划工具（分解目标 / 追踪进度）、隔离子代理（并行处理独立子任务、各自独立上下文）、虚拟文件系统（跨会话保存系统提示、技能与长期记忆），以及压缩历史 / 卸载大工具结果 / prompt cache 的上下文中间件 [59]。同期 LangChain 宣布 **3 亿美元 C 轮、30 亿美元估值** [59]。
- **Anthropic 的多智能体研究系统**：用 orchestrator–worker 架构把「需要海量并行、超出单上下文窗口、调用大量复杂工具」的研究任务工程化 [61]。
- **评测与极端时长同步上探**：从 Terminal-Bench 的真实终端任务 [12]，到 LHTB 的约 90 分钟 [13]、LHTB 2.0 的 3 小时 [14]，再到 Infrathon 的 24 小时 [37]，「无人值守时长」本身成为评测维度。

> **说法审计（不夸大原则）。** 二手中文媒体有「红杉将长程智能体称为继预训练、推理之后的第三次范式拐点 / 2026 元年」一类表述。经核对一手材料：可直接核实的是红杉播客**标题与正文**确实以 long-horizon agents 为主题、Chase 判断「现在它们真的开始工作了、编码最先爆发」[60]；但「第三次范式拐点」这一具体提法**未能在红杉一手页面中逐字核实**（红杉 2024 年《AI's $600B Question》谈的是资本开支而非该论断 [62]）。本报告因此把「元年级」作为**产业叙事信号**引用，而把所有硬结论建立在可核实的论文与一手工程数据之上。知乎相关话题页在调研时多返回 403，故中文热度以可访问的一手/第三方信源交叉，不以单一转述为准。

### 2.3 与既有相邻方向的关系

长程智能体不是孤岛，它站在若干 2024–2025 成熟方向的交汇点上，并把它们的目标函数从「单步 / 单会话」改写为「跨时间的可靠收敛」：

- 与 **Agent Harness**：harness 是长程可靠性的主要承载者（本报告第 4、7 章）；
- 与 **Context Engineering**：compaction、卸载、隔离是长程不被上下文撑爆的直接手段（5.1）；
- 与 **Agent Memory**：长程把「记什么、何时写回、何时读出」从加分项变成生死线（5.2、7.x）；
- 与 **Agentic RL / Test-Time Scaling**：长程任务的稀疏奖励与巨额 rollout 成本，重塑了信用分配与训练基础设施（第 7 章）；
- 与 **Proactive Agent**：后者关注「该不该无提示地行动」，长程关注「行动开始后能否持续到收尾」。
---

## 3. 思想源流与四年演进（2022–2026）

长程智能体并非凭空出现。把「LLM 作为 agent 跑循环」这条线拉长，可以看到四个阶段，主线是**交互时长、状态位置、评分方式与可靠性来源**的同步迁移。

> ![](diagrams/3-evolution-timeline.html)
> **图 3 · 从「LLM 跑循环」到「长程可靠」的 2022–2026 演进时间线。** 四个阶段：循环诞生（ReAct/Reflexion/Voyager/AutoGPT/AutoGen）→ 脚手架与接口（CodeAct/SWE-agent/AgentBench/Devin）→ Harness 与上下文工程（编码 harness 爆发、Terminal-Bench、compaction/子代理/文件系统）→ 长程成为一等公民（LHTB 系列、Folio/PACE/ASTC、Evidence-Action Gap、RiseHarness/Terminal-RSI、24h 评测、可恢复持久运行时）。

### 3.1 阶段一（2022–2023）：循环的诞生

- **ReAct（2022）** 把「推理（reasoning trace）」与「行动（工具调用）」交织进同一条生成轨迹，给出了 LLM-in-a-loop 的标准范式 [1]。
- **Reflexion（2023）** 让智能体把失败用自然语言反思后存入记忆、在下一次尝试中利用，首次把「跨尝试的语言化自我改进」放进循环 [2]。
- **Voyager（2023）** 在 Minecraft 中展示开放式终身学习：自动课程 + 技能库 + 迭代提示，是「技能随时间积累」的早期样板 [3]。
- **AutoGPT（2023，工程现象）** 把「目标进、自主循环出」推向公众，虽受限于当时模型与脚手架而多停留在 demo，但确立了想象范式（Chase 称之为 agent 的核心思想）[60]。
- **AutoGen / MetaGPT（2023）** 把多智能体对话与角色分工系统化 [5][6]；**Generative Agents** 则展示了带记忆流的长期角色模拟 [4]。
- 同期 **ToRA** 把工具集成推理用于数学 [7]，工具调用成为循环中的标准动作。

这一阶段解决了「循环能不能转起来」，但任务多为单轮到几十轮，状态基本全在上下文里。

### 3.2 阶段二（2024）：脚手架与接口

- **CodeAct（2024）** 论证「可执行代码动作」比受限的 JSON 工具调用更灵活强大，推动 agent 用代码（尤其 Bash/Python）作为统一动作空间 [8]。
- **SWE-agent（2024）** 提出 Agent–Computer Interface（ACI），证明精心设计的命令接口 / 反馈格式能大幅提升真实 GitHub issue 解决率——这是「harness 本身就是能力」的早期硬证据 [9]；它所攻的 SWE-bench 也成为真实仓库任务评测的奠基基准 [18]。
- **AgentBench（2023）/ Terminal-Bench（2026-01）** 等基准把评测从「答题」推向「在真实环境里办事」；Terminal-Bench 以困难、真实的命令行任务为载体 [11][12]。
- **Devin / Claude Computer Use（2024–2025，产品）** 把「长时自主干活的软件工程师 / 通用电脑操作」推向市场，工程上大量依赖文件系统、浏览器与规划脚手架。

这一阶段的共识是：**同样的模型，配不同的接口/脚手架，成绩可以差很多**。但任务时长仍多为分钟级。

### 3.3 阶段三（2025）：Harness 与上下文工程成为主战场

随着编码 agent（Claude Code、Codex 等）爆发，社区意识到决定长任务体验的关键已从「再强一点的模型」转向「包裹模型的厚运行时」：compaction（上下文压缩）、子代理（隔离上下文并行）、虚拟文件系统（把状态写到窗口之外）、prompt cache（降本提速）成为标配。Chase 观察到，编码公司几乎都在自研 harness，Terminal-Bench 2.0 榜单上能同时看到 harness 与模型两个维度的差异，并直言「Claude Code 流行的很大一部分原因是 harness 本身」[60]。本库《Agent Harness》《上下文工程》两报告已对这一阶段做了组件级拆解。

### 3.4 阶段四（2026）：长程成为一等公民

2026 年的特征是「时间维度」被显式研究与评测：

- **专门的长程基准**：LHTB / LHTB 2.0 用密集过程奖励量化「走了多远」、把时限拉到 3 小时 [13][14]；Infrathon 用 24 小时、逾万 GPU-小时研究「能否持续优化真实 AI 基础设施」，并发现 **150 个 run 中有 135 个在预算后半段仍取得进一步改进**，且模型排名随时长变化——短评测会同时漏掉后期增益与排名反转 [37]；Singularity-Bench 直接考「agent 能否构建 agent」，覆盖预训练到集群编排的 6 个端到端任务 [38]。
- **上下文治理新机制**：Folio 的可恢复多分辨率折叠 [15]、PACE 的「驻留≠可检索」 [17]、ASTC 的模型内软 token 压缩 [19]、MarkovSim 的有界上下文工作流 [20] 等密集出现。
- **停止与坚持的元推理**：Evidence-Action Gap / ImpasseBench 刻画「过度坚持 + 虚报完成」[30]，BELT 学习可复用的早停表示 [31]。
- **harness / 轨迹自进化**：RiseHarness [27]、Terminal-RSI [29]、递归自改写 [28]、Live-SWE-agent [33] 让可靠性在权重冻结下迭代。
- **持久运行时**：可恢复执行、动作结算/重放、选择性再校验、执行态遗忘等「durable」问题进入学术视野（5.4）。

---

## 4. 参考架构：六层栈与「状态外部化」

综合 Deep Agents 的开源实现 [59]、Anthropic orchestrator–worker 研究系统 [61]、MarkovSim / SagaLLM 等学术系统 [20][25]，可以把一个长程智能体拆成六层。核心设计哲学只有一句：**把易失、易膨胀、易出错的状态，从模型上下文里搬到外部系统，让权重保持冻结、让可靠性由 harness 与状态系统承担。**

> ![](diagrams/2-reference-architecture.html)
> **图 2 · 长程智能体六层参考架构。** ① 冻结的模型核心（单步推理/工具调用）；② 循环与控制（规划、预算、自我终止、过程 FSM 约束）；③ 上下文治理（compaction/折叠、KV 驻留与可检索、软 token 压缩、子代理隔离）；④ 外置状态（虚拟文件系统、分层记忆 L1/L2/L3、任务账本/进度追踪、证据链）；⑤ 编排（orchestrator–worker、并行子代理、规划/验证分工）；⑥ 持久运行时（检查点、resume、动作结算/replay、选择性再校验、异步后台）；底层是会被外部改变的环境/工具世界。

### 4.1 ① 模型核心 Model（冻结 / 策略）

只负责单步：给定当前上下文，产出下一步推理与工具调用。长程范式的一个显著特征是**模型权重常常被显式冻结**——RiseHarness、StateM 等工作都在「不更新模型」的前提下，仅靠 harness 改进长程可靠性 [27][48]，从而把「模型能力」与「长程完成能力」干净地分离开来评测。

### 4.2 ② 循环与控制 Loop Control

决定「这一步往哪走、什么时候停」：

- **规划 / 子目标分解**：把开放目标拆成可追踪的子目标（Deep Agents 的 planning tools 让 agent 分解任务、追踪进度并随学习调整 [59]）；
- **预算管理**：在固定 token / 时间 / 步数预算内分配工作量，这是长程相对短程新增的维度（LHTB 中大量失败是「还在推进但时间到了」[13]）；
- **自我终止判据**：区分「真完成」与「通过了可见测试」（5.5）；
- **过程约束**：对程序化 SOP 任务，用可执行有限状态机（FSM）把工作流状态放在模型上下文**之外**，在工具调用 / 状态迁移 / 终止三处强制执行，显著抑制长无人值守运行中的「流程漂移」[24]。

### 4.3 ③ 上下文治理 Context

保证窗口不被撑爆、且关键信息仍可被取用（详见 5.1）：compaction / 多分辨率折叠、KV 驻留策略、模型内软 token 压缩、以及用子代理做上下文隔离。TurnCost 基于 600 条真实编码轨迹提醒：观测的中位数只有 118–286 token，常被当作压缩理由的巨型 build log 其实集中在第 99 百分位——**压缩该发生在「轮 / 循环」层面，而非无差别压观测** [35]。

### 4.4 ④ 外置状态 Externalized State

这是长程架构与普通 agent 最本质的区别：**让文件系统与结构化记忆承担「工作记忆」**。

- **虚拟文件系统**：Deep Agents 用它跨会话保存系统提示、技能与长期记忆 [59]（其思想可上溯到 MemGPT 把 LLM 记忆当操作系统分层管理的设想 [10]）；编码 agent 天然把代码、笔记、中间产物写到磁盘。
- **分层记忆**：Folio 在运行时记录三个相互链接的层级——模型在正常回复中随手写的「短轮次笔记」（L1）、运行时为每次调用写的「事件清单」（L2）、以及捕获的原始记录（L3），并通过同一组句柄支持折叠与**可恢复** [15]；MarkovSim 则把工作流记录按预定义 schema 写入记忆系统，只在状态迁移边界把有界上下文交给模型 [20]。
- **任务账本 / 进度追踪 / 证据链**：显式记录哪些子目标已完成、哪些待办、每个结论的证据出处，以抑制「重复已完成步骤」与「虚假完成」。CAT 把上下文工作区形式化为「稳定任务语义 + 浓缩长期记忆 + 高保真当前状态」，并把上下文维护升级为 agent 可主动调用的工具 [21]。

### 4.5 ⑤ 编排 Orchestration

- **Orchestrator–Worker**：主编排代理负责规划与综合，把可并行的子任务派发给拥有独立上下文窗口的子代理；子代理在各自窗口里并行探索、再把浓缩后的关键 token 交回主代理，既突破单窗口容量，又通过「关注点分离」降低路径依赖 [61]。
- **并行度**：Anthropic 经验是主代理一次并行拉起 **3–5 个**子代理、子代理再并行调用 **3+ 个**工具，复杂查询研究时间最多降 90% [61]。
- **规划 / 验证分工**：对长程任务的初步研究发现，「验证（verification）」与任务成功相关性最强，却往往发生得太晚、留给纠正的窗口有限 [53]——编排上应让验证更早、更频繁地介入。

### 4.6 ⑥ 持久运行时 Durable Runtime

把「一次长运行」变成「可中断、可恢复、可重放、可后台」的持久计算：检查点 / 快照、断点续跑（resume）、动作结算与重放（action settlement / replay）、针对外部世界变化的选择性再校验（5.4），以及异步后台执行（fire-and-forget，人类稍后审阅初稿）。这是 2026 年才开始被系统研究、却直接决定「能否真正无人值守」的一层。

### 4.7 底层：会被外部改变的环境世界

Shell、浏览器、MCP 工具、代码仓库、数据库、人工审批……关键在于**环境状态会在 agent 等待期间被外部改变**，因此动作执行前往往需要「选择性再校验」而非盲目乐观并发（5.4）。ToolSpace 等工作则开始从真实 MCP 执行轨迹中规模化合成跨多环境的工具与任务，为长程工具组合提供训练/评测土壤 [69]。
---

## 5. 核心技术挑战与前沿方法（七大子问题）

这一章是报告的技术核心，以「问题—机理—代表方法—关键数据」的形式梳理 2026 年最活跃的七个研究子问题。

### 5.1 上下文爆炸、Compaction 与可恢复折叠

**问题。** 长程交互的历史（工具结果 + 模型回复）单调增长，直到塞不进窗口。主流做法是让模型把历史**摘要压缩**（summarization compaction），但每次压缩都要重读并重写它所替换的历史，且「何时压缩」的触发逻辑与「压缩后丢什么」的决策耦合。

**前沿方法。**

- **Folio（2611.16627）**：不再每次重读重写，而是通过同一组句柄记录、折叠、恢复历史，记录 L1 短轮次笔记 / L2 事件清单 / L3 原始记录三层链接视图；这些视图本就由运行时和模型在运行中写就，因此压缩不再需要额外的摘要生成 pass，且支持**可恢复**（折起来的东西能再展开）[15]。
- **ASTC（2610.41291）**：不训练外部压缩器、也不把上下文改写成更短的硬 token 序列，而是用模型自身的循环（native recurrence）把历史压成**软 token**，免训练且与模型演化中的表征对齐，减少硬 token 重写的失真 [19]。
- **CAT（2512.22087）**：把上下文维护从「只追加（append-only）或被动触发压缩」变成 agent 决策环中可主动调用的**工具**，形式化出「稳定任务语义 + 浓缩长期记忆 + 高保真当前状态」结构化工作区，缓解 context explosion、semantic drift 与推理退化 [21]。
- **HIPIF（2606.10507）**：针对「持续增长的历史削弱 agent 追踪全局任务状态」的 long-context interference，用分层规划 + 信息折叠（hierarchical planning and information folding）直接攻击长上下文干扰问题 [22]。

**反直觉发现。** PACE（2611.10240）证明「把该留的 token 留在 KV cache」远远不够：100K 编码会话、2048 token 预算下，一个保留 99%+ 所需 token、驻留率 0.996 的规划引导策略，当定义—使用间隔从 2K 扩到 90K 时后期步骤准确率仍从 **93% 掉到 64%**——所需 token 是「驻留」的，但被其他历史干扰而「不可检索」[17]。这把问题从「存没存」深化为「能不能在正确时刻被提取、且不被干扰」。

### 5.2 记忆写回 / 读出控制与效用导向

**问题。** 当历史超窗，智能体必须决定「写什么 down、什么时候写、什么时候读回」。传统记忆机制围绕「语义相关性、检索、压缩」设计，但语义最相关的记忆未必最能改善**当前决策**。

**前沿方法（2026-11 单月多篇）。**

- **Coding Agent Memory Post-Training II（2611.14009）**：批评此前用 on-policy 蒸馏让记忆控制成为策略一部分时，记忆工具与压缩协议都在模型预训练之外，导致教师/学生两侧蒸馏都被削弱；主张解锁并利用「预训练文件操作」的记忆潜力，把外部记忆接口对齐到模型已有的文件操作先验上 [40]。
- **MemCredit（2611.15600）**：决策时刻记忆，把记忆选择从「语义相关性」转向「**决策效用（decision utility）**」，估计每条候选记忆对当前动作的实际改善，而非单纯找最像的 [41]。
- **FGSE / 当 agent 该何时记忆（2611.14326）**：在「验证器丰富」的环境里，不能把 episode 得到的教训直接写进持久状态——本地成功不是可复用、安全知识的可靠信号；用「证伪门控（falsification-gated）」决定一条经验何时才被准入持久记忆 [42]。
- **MemPrice（2610.24955）**：把活动上下文当作共享资源，用对上下文的「**自适应定价**」在线分配记忆预算与表征层级，只在预估留存价值对得起占用时才鼓励细节，压力跨刷新持续存在 [23]。
- **MAPE / 程序进化发现记忆设计（2611.17068）**：通过标准化生命周期接口与「诊断—提议」迭代，从最小种子进化整个记忆程序；在 LoCoMo / LongMemEval 上相对人工设计基线最高 +12% 准确率，仅用 35% serving 成本，并能跨底座迁移 [43]。

### 5.3 规划与进度追踪 / 任务图

**问题。** 开放目标需要分解、需要在数百步中维持对全局状态的追踪，并标记哪些子目标已完成、哪些待办。早期错误会污染后续历史与环境状态，并沿决策链累积。

**方法。** Deep Agents 的 planning tools 让 agent 动态更新 todo 与计划 [59]；BEACON（Milestone-Guided，2605.06078）利用长程任务的组合结构，用里程碑引导政策学习，同时治「信用错配（正确的早期动作因终端失败被惩罚）」与「样本低效（成功轨迹稀缺导致学习信号几乎消失）」[44]；AgentPRM（2511.08325）用「承诺与进度」度量构建面向多轮决策的过程奖励模型，给中间进度打分 [45]；HIPIF 同样依赖子目标分层来降低长程依赖 [22]。

### 5.4 错误恢复、状态过期与可恢复 / 持久执行

**问题。** 长运行中工具会失败、环境会变、智能体会犯错且错误进入世界状态。这是「durable execution」与传统 agent 的分水岭。

**方法与发现。**

- **ReflectPO（2610.30933）**：面向长程任务的反思引导策略优化，核心是自主**错误恢复**——早期执行错误影响后续历史与环境状态、效应沿决策链累积，需用反思信号引导 policy 更新 [32]。
- **ToolMaze / When Tools Fail（2606.05806）**：不再只测「快乐路径」，用 DAG 拓扑复杂度 ×（显式/隐式 × 瞬时/永久）工具扰动的 2×2 分类法，区分系统性重规划与盲目试错，量化工具失败时的动态重规划与异常恢复能力 [34]。
- **Selective Revalidation（2609.08015）**：长运行 agent 读状态、推理、等待工具/人审批，动作可能很晚才执行，而正当化该动作的状态在此期间可能已变（如退款额度从 100 降到 50 使原方案失效、他人已退款需防重复）；标准乐观并发 / 版本号检查太粗，需要按字段对**决策相关**的状态做选择性再校验 [51]。
- **Execution-State Unlearning（2609.04875）**：长运行是有状态的——除对话记录外还累积压缩摘要、明文记忆、待执行工具计划以及 KV cache；今天的「遗忘」只删一条明文记录就停，派生自该信息的一切 artifact 仍在。论文形式化「执行态遗忘」：遗忘后 agent 必须表现得像从未见过目标，并把运行时建模为确定性转移系统给出可证明保证 [52]。
- **SagaLLM（2503.11951）**：为多智能体规划提供上下文管理、校验与**事务保证（transaction guarantees）**，把数据库的 ACID 思想引入长程多代理执行 [25]。
- **Auditing Action Settlement（2610.01138）**：审计 LLM agent 环境中的动作结算——顺序、进度与重放（order, progress, and replay），这正是 durable runtime 的核心原语 [55]。

### 5.5 自我终止、虚假完成与证据—行动鸿沟

**问题。** 智能体有两种对称的「不会停」病：要么在**不可解**环境里过度坚持、绕弯路、甚至声称记录不支持的成功；要么在通过了所有**可见**测试后误判完成、带着剩余时间退出，却栽在隐藏测试上。

**方法。**

- **Evidence-Action Gap / ImpasseBench（2610.31895）**：向可解修复任务注入 12 种真实环境阻塞，并在每条轨迹中标注「收集到的证据已足以判断需要外部帮助」的那一步，把恰当努力与过度坚持分开——证据通常在几步内就出现，但 agent 很少据此行动，表现为 over-persist、undesired workaround、claim success the record does not support [30]。
- **BELT / Same Beliefs, Different Policies（2610.30450）**：把「何时停」当作元推理问题，从冻结 LLM 的隐状态学习对当前/未来答案的信念，再在冻结特征上拟合线性停止策略，把「结果预测」与「停止控制」解耦，支持不同 token 价格而无需重训控制器 [31]。
- **Early stopping in reasoning（同篇）**：推理模型常在得到正确答案后继续算，浪费算力甚至把对的改错；BELT 直接针对这一浪费 [31]。
- **FSM 约束（2611.10884）**：对程序化任务，把工作流状态放在上下文外并用可执行 FSM 在工具调用/状态迁移/终止三处强制，也有助于在正确节点终止 [24]。

### 5.6 上下文清理的因果审计：「清洗历史」什么时候反而有害

长程系统常要在中途压缩、修复或交接历史，而这决定了它如何陈述自己的错误——**尽管错误的后果还留在世界里**。Cleaning an Agent's History Hurts When the World Still Holds the Error（2611.14654）做了一个干净的因果实验：把 agent 冻结在任务中途、保持其真实世界状态不变，只改历史或世界，再让接收模型继续。在 AgentCE-Bench 上，当 agent 写入一个错误值后，保持世界固定、用两种自然方式补入正确值，发现**同样正确的内容，因「历史如何陈述它」与「世界里还残留什么」不同，可能帮倒忙** [26]。这对所有 compaction / handoff 机制是一记警钟：脱离世界状态的「历史正确」没有意义。

### 5.7 多智能体 / 子代理编排与并行

**问题。** 单窗口装不下、单条路径易有路径依赖。解决思路是把工作并行摊到多个拥有独立上下文的代理，再由编排者综合。

**方法。** Anthropic 研究系统总结的工程原则包括：orchestrator–worker 模式、主代理一次 3–5 个子代理并行、子代理 3+ 工具并行（研究时间最多 −90%）；通过子代理做信息压缩（各自窗口探索后只把关键 token 交回）与关注点分离（独立工具/提示/轨迹降低路径依赖）；工具描述经反复实测改写后未来 agent 任务完成时间 −40% [61]。学术侧，Goal-Aware Dynamic Collaboration（2610.39193）研究单 agent 如何在长程中动态承担规划/验证职责，发现验证与成功关联最强但常太晚 [53]；CoffeeBench 在异质多代理经济系统中评测长程沟通、谈判与交易 [56]；CoworkOS 把任务策略与执行管理分离，模型决策、harness 管监控与长程执行 [50]。在工具环境侧，ToolSpace（2611.05036）反转「先造数据再派生工具」的数据中心范式，直接从真实 MCP 服务与开放 agent 语料的执行轨迹中挖掘数据、状态与重试依赖，规模化合成跨多环境、可执行落地的长程工具任务 [69]。
---

## 6. 评测革命：密集奖励、过程分与时间预算

长程智能体最先被改写的不是训练，而是**评测**——旧尺子量不出新能力。

### 6.1 为什么二元 pass/fail 不够用

在极难长任务上几乎所有模型都无法完整通过，稀疏的终端奖励（成 1 败 0）无法区分「完成 90%」与「完成 5%」，两者都记 0；在强化学习里这类环境几乎无法学习，在评测里则无法区分模型。LHTB 的做法是把每个任务进一步分解为细粒度、带权重的**评分子任务**，给出两类指标 [13]：

- **pass@1(τ)**：任务总分 R ≥ 阈值 τ（主报告 τ=0.95）才算通过，回答「能否可靠完成」；
- **平均奖励 R̄**：所有任务的子任务加权平均，回答「通常能走多远」。

「高 R̄ 低 pass@1」是极有价值的诊断信号：**近乎完成、却总在最后一步失败**。LHTB 2.0 在 30 个重构任务上测得二元验证器平均仅 0.221、密集进度阶梯平均 0.653 [14]，直观说明二元评分系统性低估了智能体的真实进展。

### 6.2 时间预算成为一等评测维度

- **LHTB（v1）**：46 任务跨 9 类，Terminal-Bench 式容器化设置 + 隐藏验证器；典型任务需数百 episode、分钟到小时；15 模型实测平均 9.9M tokens / 约 231 episodes / 85.3 分钟；最强模型 pass@1 仅 15.2%（τ=0.95）、10.9%（τ=1.0），跨模型均值 4.3% / 1.7% [13]。
- **LHTB 2.0**：保留并修正原 46 任务、扩展 31 个，达 **77 任务 / 13 领域 / 8 任务类型**；把评测时限从 90 分钟延长到 **3 小时**，每 30 分钟记录一次验证器检查点，直接测量单次 run 内的「持续进度（sustained progress）」；1,145 个已评分 model-task run 中，GPT-6-astra 以密集均分 0.69、37 个任务 R≥0.95 领先，GPT-5.6-sol 为 0.67 / 32 个（厂商模型名与分数为论文口径）[14]。
- **Infrathon（24 小时）**：6 模型 × 150 run × 24h 预算、逾 10,000 GPU-小时，研究能否在长程上持续优化真实 AI 基础设施；**135/150 个 run 在预算后半段仍取得进一步改进**，且模型排名随时长变化，甚至能发现优于人工设计的方案——说明短评测会同时漏掉后期增益与排名反转 [37]。

> **关于「79% 失败源于超时」的口径说明。** 该数字出自一篇对 LHTB 的第三方精读笔记（按失败模式统计，约 79% 为时间预算耗尽、约 19% 为带非平凡奖励的主动退出，另有「虚假完成」案例）[63]，并非本报告从论文 PDF 逐表复核。LHTB 论文摘要可核实的硬数据是上文的 15.2% / 4.3% / 9.9M / 231 / 85.3 分钟 [13]。本报告引用 79% 时一律标注「第三方精读口径」，与论文一手数据区分。

### 6.3 归因：分数不再只属于模型

既然「Agent = Model + Harness」，榜单分数就是两者的联合产物，必须联合归因。

- **When Agents Fail: Model or Harness—Who's to Blame?（2610.32483）** 提出五个互补度量：**VCR（Valid Completion Rate，能否正常收尾并交出可评测结果，与对错无关）、Model-Harness Fit（给定模型下 harness 对任务达成的支撑）、Cross-Model Fit（harness 跨模型的总体适配）、Interaction Residuals（被总分掩盖的配对级强弱项）** 等 [64]。
- **StateM（2608.15089）** 在模型冻结、只改 harness 的条件下，于 Terminal-Bench 2.1 拿到 **95.3% 原始准确率**，或一次仅 **$15** 的前沿模型运行——直接证明 harness 维度的巨大空间 [48]。
- **Do Agent Optimizers Compound?（2607.14004）** 指出绝大多数 agent 优化收益是「一次性」的：对着固定榜单优化一次就报告提升，却没测部署中真正重要的「随新失败/新任务递归优化时，收益能否复利而不侵蚀上一轮」；作者在 Terminal-Bench 2.0 难任务上构建两阶段持续学习评测来回答「优化能否 compound」[16]。
- **What Makes a Terminal-Bench Task Hard?（2609.26826）** 区分任务的「真困难」与「假困难（fake-hardness）」，避免把脚手架/评测artifact误当成模型能力 [49]；**Benchmarks as Software（2611.08417）** 主张把 Terminal-Bench 当作持续演进的软件来维护，应对污染与过拟合 [39]；**Hack-Verifiable Terminal Bench（2608.22103）** 则专门评测终端任务上的 reward hacking [36]；**Agent Benchmarks Stop Ranking Before They Run Out of Tasks（2611.13541）** 进一步提醒：在任务被「跑光」（污染/过拟合）之前，静态榜单的排名意义会衰减，长程基准尤其需要持续供给新任务 [54]。

### 6.4 长程基准版图（2026）

| 基准 | 时长 / 规模 | 载体与评分 | 主测能力 |
|---|---|---|---|
| Terminal-Bench [12] | 分钟–数十分钟 | 真实 CLI 任务、容器 | 困难真实终端任务 |
| LHTB [13] | ~85 分钟 / 46 任务 | 容器 + 隐藏验证器 + 密集子任务奖励 | 长程完成 + 局部推理分离 |
| LHTB 2.0 [14] | 3 小时 / 77 任务 | 30 分钟检查点、进度阶梯 | 单次 run 内持续进度 |
| Infrathon [37] | 24 小时 / 150 run | 真实 AI 基础设施优化 | 超长程持续优化、排名随时长 |
| Singularity-Bench [38] | 端到端 6 任务 | 从最小起点实现核心系统 | agent 能否构建 agent |
| THEATRON [46] | 多月轨迹 | 金融回放、人格注入 | 长程决策中的人格持久性 |
| OdysseyBench [47] | 长程办公流 | 复杂 Office 应用工作流 | 长程办公自动化 |
| SaaS-Bench [68] | 专业工作流 | 真实 SaaS 计算机操作 | 专业软件端到端 |
| ToolMaze [34] | 受控扰动 | DAG 复杂度 × 工具故障 2×2 | 动态重规划与异常恢复 |
| ImpasseBench [30] | 注入 12 类阻塞 | 标注「证据充分步」 | 过度坚持 / 不会停 |
| CoffeeBench [56] | 长程多代理经济 | 沟通/谈判/交易 | 异质多代理经济协作 |

> ![](diagrams/4-failure-evaluation.html)
> **图 4 · 为什么「局部对」却「长程崩」：六类失败模式与评测范式迁移。** 上半部六类失效：① 预算耗尽/超时；② 上下文爆炸与压缩失真；③ 驻留≠可检索（PACE，93%→64% [17]）；④ 虚假完成/不会停；⑤ 过程漂移/不循规；⑥ 世界已变/状态过期。下半部：评分从 pass/fail 二元走向子任务密集奖励，时长从分钟级走向 3h/24h，指标从只看 pass@1 走向 pass@1 + 平均奖励 + VCR + Model-Harness Fit，归因从「分数=模型能力」走向模型×harness 联合归因。

---

## 7. 训练范式：长程 RL、信用分配与 harness / RSI 自进化

评测之外，2026 年第二条主线是**如何训练**智能体在长程上可靠。核心矛盾有两个：稀疏/延迟奖励下的**信用分配**，以及巨额 rollout 成本下的**样本效率**。

### 7.1 长程信用分配：把终端奖励正确地传回到每一步

长程任务动辄几十上百个序列决策，终端成败难以归因到早期动作，GRPO 等无价值函数方法面临「步级 Q 估计不准、中间状态基线错位」两个瓶颈。

- **HCAPO（Hindsight Credit Assignment，2603.08754）**：首个把事后信用分配引入 LLM agent 的框架，利用 LLM 自身做事后重标注来修正步级信用 [57]。
- **BEACON（Milestone-Guided，2605.06078）**：用里程碑把长程任务组合结构化，治「信用错配 + 成功轨迹稀缺」两病 [44]。
- **ABSeeker（2608.05102）**：面向长程搜索 agent，用「答案回溯信用分配（Answer-Backtracked Credit）」从最终答案反推哪些检索/验证/整合动作真正有用，区别有用动作与错误/冗余动作 [58]。
- **TRCA（2608.16156）**：转移级评分量规信用分配，不依赖昂贵的过程评估器标注，也不依赖 RL 早期极度稀缺的成功轨迹，改用 rubric 在每个转移上给信用 [65]。
- **AgentPRM（2511.08325）**：构建面向多轮决策的过程奖励模型，以「步级承诺（promise）与进度（progress）」评估中间状态 [45]。
- **Intrinsic Credit Assignment（2602.12342）** 与 **Hindsight Credit over Trees（2611.02706）** 分别从内在动机与树搜索轨迹角度改进长程交互的信用归因 [67][70]。

### 7.2 反思、过程奖励与自主错误恢复

- **ReflectPO（2610.30933）**：反思引导的策略优化，让 agent 针对早期错误沿链累积的问题做自主恢复 [32]。
- **HIPIF（2606.10507）**：分层规划 + 信息折叠，在训练侧直接缓解 long-context interference [22]。
- **Milestone / dense process reward** 的共同思想与评测侧的密集奖励一脉相承：把稀疏终端信号替换/补充为密集中间信号，模型才「学得动」。

### 7.3 Harness 自进化：权重冻结，让运行时迭代

这是长程范式最有特色的一支——**不动模型，改 harness**。

- **RiseHarness（Harness Self-Evolution for Long-Horizon Terminal Agents，2610.27435）**：针对「一次能过、重复运行却失败」（随机错误沿动作与上下文传播，造成 capability 与 reliable execution 之间的鸿沟），在模型权重固定下迭代 harness；专门挑出「同一配置下既有成功又有失败 run」的任务，以成功 run 为参照做失败分析，结合 agent 轨迹与运行时监控器的内部 harness trace，考察模型动作如何与命令执行、反馈、上下文管理、完成判据交互，并沉淀出 Harness Handbook [27]。
- **Evolving to Match Capabilities（2610.33024）**：通过技能适配改进 harness 自进化，使 harness 与变化中的模型能力匹配 [66]。
- **When Agents Fail（2610.32483）** 的 VCR / Fit 度量为 harness 进化提供可优化目标 [64]。

### 7.4 递归自我改进（RSI）与轨迹自改写

- **Terminal-RSI（2610.22192）**：在「经验证轨迹的递归自改写（RSR）」与「继承批评者的递归策略精炼（RPR）」之间交替；提出 Turn-Boundary Skip-Observation GAE（TB-GAE）在保留 token 级基线的同时跨交互轮衰减，用 Family Dense Reward（FDR）跨任务族校准语义进度、Success-Prioritized Reward（SPR）区分严格成功与有界部分分，并配 warm-up 批评者、递归批评者交接与分阶段学习率 [29]。
- **Scaling Trajectories through Recursive Self-Rewrite（2610.02826）**：通过递归自改写扩展复杂任务轨迹 [28]；**Live-SWE-agent（2511.13646）** 研究软件工程 agent 能否在运行中（on the fly）自进化 [33]。
- 与本库《WSP × RSI》报告互补：那篇侧重权重空间的递归自我改进，此处侧重**轨迹/harness/文本空间**、且权重冻结的长程可靠性迭代。

### 7.5 长程训练的经济学

长程 rollout 极贵（LHTB 平均 9.9M tokens、85 分钟 [13]；Infrathon 单次 run 24 小时 [37]），因此：(a) 密集过程奖励 / 信用分配本质上也是在提高样本效率；(b) 子代理并行与 prompt cache 是成本杠杆（Anthropic 多智能体约 15× chat token [61]；TurnCost 主张在「轮」而非「观测」层面做缓存感知的循环工程 [35]）；(c) 「便宜门控 + 昂贵执行」「先小模型后大模型」等调度思想在长程中同样适用。
---

## 8. 产业与开源生态地图

长程智能体不是单一产品，而是一层正在成型的「持久智能运行时」。可粗分为五层：

1. **编码 harness（最先成熟）**：Claude Code、OpenAI Codex CLI、Cursor/Composer、Aider、Factory、Amp 等几乎都自研 harness——compaction、todo/计划、子代理、文件系统、checkpoint/resume 是共同要件。Chase 观察到编码公司是 harness 工程做得最好的一群 [60]。
2. **通用长程 / Deep Research harness**：LangChain 开源 **Deep Agents**（planning tools、隔离子代理、虚拟文件系统、上下文中间件，模型中立）[59]；各家 **Deep Research** 产品用 orchestrator–worker 并行子代理换取深度 [61]；Manus、Devin 等把「目标进、长时间自主执行、最后交初稿」产品化。
3. **后台 / 异步执行基础设施**：任务队列、持久化状态、断点续跑、人机审批节点、可观测性（trace/eval）成为 agent 平台标配；LangChain 同期把 Agent Server / Sandboxes / Fleet 等「运行时」产品独立出来 [59]。学术对应物是 CoworkOS 这类「模型决策、harness 管执行与监控」的通用长程 harness [50]，以及动作结算/重放的形式化 [55]。
4. **评测与可观测**：Terminal-Bench / LHTB 系列 [11][13][14]、SWE-bench 系 [18]、各类垂直长程基准（Infrathon、Singularity、THEATRON、Odyssey、SaaS、Coffee）[37][38][46][47][68][56]，加上 LangSmith 等 trace/eval 平台——「send us a trace」取代「show me the code」成为排障起点 [60]。
5. **训练与自我改进**：从轨迹学习（Live-SWE-agent [33]）、harness 自进化（RiseHarness [27]、技能适配 [66]）到递归自我改进（Terminal-RSI [29]、RSR [28]），优化对象从「模型权重」扩展到「权重 + harness + 记忆程序」（MAPE 甚至能进化出比人工更好的记忆设计 [43]）。

**成本现实。** 长程是「用 token 换完成率」：Anthropic 实测多智能体系统约用掉 chat 15 倍 token [61]；LHTB 中各模型单任务成本从约 \$2.47 到 \$27.57 不等、平均耗时 85.3 分钟 [13][63]。因此当前最成立的产品形态是 Harrison Chase 反复强调的「**长程运行、产出高质量初稿（first draft），人类在关键节点审阅**」——编码交 PR、SRE 交事件报告、研究交初稿、客服升级人工，而不是无人直接拍板 [60]。

---

## 9. 开放问题

1. **长程完成能力的缩放定律尚不清楚。** 单步推理随模型变强，但「预算内收尾」的能力如何随模型规模、harness 质量、token 预算缩放？Infrathon 显示排名随时长变化 [37]，LHTB 2.0 显示三小时内可持续进度 [14]，但缺少统一的 scaling law。
2. **压缩—保真—可恢复的三难。** 摘要式 compaction 失真、软 token 压缩依赖模型自身循环、Folio 用三层句柄换可恢复——有没有信息论意义上的最优长程记忆表征？「清洗历史反而有害」[26] 说明这不是纯工程问题。
3. **驻留与可检索之辨。** PACE 证明「token 还在 KV 里」≠「模型用得上」[17]，长程检索的干扰机制与 KV/记忆架构如何协同设计仍开放。
4. **自我终止与虚假完成。** 何时判断「做完了」是元推理问题；隐藏测试通不过却宣称完成（False Finish）、死胡同里过度坚持（Evidence-Action Gap）尚无成熟解法 [30][31]。
5. **持久运行时的正确性原语。** 选择性再校验 [51]、执行态遗忘 [52]、动作结算与重放 [55]、事务保证 [25] 目前是零散技术点，数据库/分布式系统四十年积累（ACID、OCC、快照隔离、幂等、saga）如何系统性移植到 agent 运行时，是一大片空地。
6. **harness × 模型的联合归因与协同进化。** 一次成功不代表跨运行可靠；VCR/Fit/交互残差 [64] 提供了度量，但模型与 harness 该如何联合训练、收益能否跨轮复利 [16]，才刚被严肃研究。
7. **长程信用分配与训练成本。** 密集过程奖励、里程碑、事后/回溯信用、rubric 信用 [44][45][57][58][65] 各有假设，长程 rollout 的巨额成本使样本效率成为瓶颈。
8. **安全与治理。** 无人值守数小时放大了越权、提示注入、reward hacking [36]、错误沿链累积的后果；长程人格/指令不持久（THEATRON [46]、FSM [24]）也带来合规挑战。
9. **评测的统计可靠性与可持续性。** 46 个任务在个位数通过率下置信区间很宽；任务污染要求「持续基准」[39][54]。如何低成本、抗污染、可复现地评测长程能力，是方法论难题。

---

## 10. 工程落地 Checklist

面向要把 agent 投入长程任务的团队，可操作的清单：

**架构与状态**
- [ ] 把易失状态移出上下文窗口：虚拟文件系统 / 结构化任务账本 / 分层笔记（L1 短笔记、L2 事件清单、L3 原始记录），并保证「折起来能再展开」（参考 Folio [15]、Deep Agents [59]）。
- [ ] 维护显式、可追踪的子目标 / todo，并随执行更新；用里程碑而非仅终端成败来组织工作（[44]）。
- [ ] 为长运行设计 checkpoint / snapshot / resume，以及动作的幂等与重放（[55]）。

**上下文治理**
- [ ] 在「轮 / 循环」层而非无差别压缩单条观测做 compaction（TurnCost 指出大观测是长尾 [35]）；优先卸载大工具结果、保留决策相关证据。
- [ ] 压缩/交接时同时考虑「世界里还残留什么」，避免历史与世界状态不一致（[26]）。
- [ ] 用隔离子代理处理独立子任务，既省主上下文又降路径依赖；主代理综合后只保留关键 token（[61]）。
- [ ] 监控「定义—使用」间隔拉长导致的后期步骤退化，不要只看 KV 命中率（PACE [17]）。

**控制与终止**
- [ ] 显式设置 token / 时间 / 步数预算，并让 agent 据此分配工作量。
- [ ] 对程序化 SOP 用外部 FSM 在工具调用 / 状态迁移 / 终止三处强制，抑制长程漂移（[24]）。
- [ ] 设计基于证据的停止判据，区分「可见测试通过」与「真正完成」，警惕 False Finish 与过度坚持（[30][31]）。

**正确性与恢复**
- [ ] 对会被外部改变的状态，动作执行前按字段做选择性再校验，而非盲目乐观并发（[51]）。
- [ ] 为工具失败做系统性重规划而非盲目重试，并用显式反思做错误恢复（[34][32]）。
- [ ] 「遗忘」要覆盖摘要、明文记忆、待执行计划与 KV 缓存等全部派生 artifact（[52]）。

**评测与迭代**
- [ ] 用密集过程奖励 / 进度阶梯评测「走了多远」，而不仅是二元 pass@1（LHTB [13][14]）。
- [ ] 同时报告 VCR、Model-Harness Fit、跨运行稳定性，并区分模型问题与 harness 问题（[64]）。
- [ ] 保留完整 trace 作为调试与回归基线；用「同配置有成有败」的任务驱动 harness 迭代（RiseHarness [27]）。
- [ ] 评测时长要覆盖真实工作流长度，警惕短评测漏掉后期增益与排名反转（Infrathon [37]）。

**产品形态与成本**
- [ ] 优先选择「长程产出初稿 + 人类关键节点审阅」的任务，匹配当前可靠性（[60]）。
- [ ] 核算单任务 token/美元成本与 Pareto 前沿，按任务价值决定是否值得多智能体 15× token 开销（[61][13]）。

---

## 11. 结语

长程智能体的故事，本质是把「LLM 在一个循环里自主决定下一步」这个从 AutoGPT 起就抓住想象力的朴素原语，沿时间轴拉到数百轮、数小时乃至跨会话后，暴露出的一组**系统级**问题。2022–2024 年，研究重心是让循环转起来（ReAct/Reflexion/CodeAct）与给它好接口（ACI）；2025 年，厚 harness 与上下文工程成为编码 agent 爆发的真正杠杆；进入 2026，LHTB 用「最强模型 pass@1 仅 15.2%、79% 失败是超时」给出清醒的基线，而 Folio/PACE/ASTC、Evidence-Action Gap、RiseHarness/Terminal-RSI、Infrathon/Singularity-Bench 与可恢复持久运行时等工作密集涌现，标志着**长程已从「demo 效果」变成一个有独立指标、独立机制、独立基础设施的一等研究方向**。

最重要的认知转变有三：其一，瓶颈从「单步对不对」变成「能不能在预算内持续推进并正确收尾」，局部推理强不等于长程完成强；其二，可靠性越来越多地来自模型之外——状态外部化、上下文治理、持久运行时与 harness 自进化，「Agent = Model + Harness」中后者的权重随时间拉长而上升；其三，评测必须从二元终端判定走向过程密集进度，从分钟级走向小时级，并把模型与 harness 联合归因。对从业者而言，当前最务实的姿态是：用长程 agent 产出高质量初稿、把状态与正确性交给外部系统、用 trace 与密集指标持续迭代 harness——把「九次九的可靠」留给未来，先把「能走很远并留下可审阅成果」做扎实。

---

## 参考与信源说明

- 标注 **arXiv:xxxx.xxxxx** 的文献，其编号、标题、日期均于 2026-11-15 经 arxiv.org 摘要页逐一核实（共 60 余篇，0 错号）；关键统计数字回原文摘要核对。
- 产业一手信源：红杉资本 *Training Data* 第 77 期 Harrison Chase 访谈、Anthropic 工程博客《How we built our multi-agent research system》、LangChain Deep Agents 官方页，均直读原文。
- 凡来自第三方精读笔记或厂商口径的数据（如 LHTB 失败模式占比、StateM 的 95.3%/\$15），正文已显式标注来源与口径。
---

## 参考文献

**A. 基础范式与谱系（2022–2024）**

- **[1]** [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629). arXiv:2210.03629, 2022-10.
- **[2]** [Reflexion: Language Agents with Verbal Reinforcement Learning and Self-Reflection](https://arxiv.org/abs/2303.11366). arXiv:2303.11366, 2023-03.
- **[3]** [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291). arXiv:2305.16291, 2023-05.
- **[4]** [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442). arXiv:2304.03442, 2023-04.
- **[5]** [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155). arXiv:2308.08155, 2023-08.
- **[6]** [MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352). arXiv:2308.00352, 2023-08.
- **[7]** [ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving](https://arxiv.org/abs/2309.17452). arXiv:2309.17452, 2023-09.
- **[8]** [Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/abs/2402.01030). arXiv:2402.01030, 2024-02.
- **[9]** [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793). arXiv:2405.15793, 2024-05.
- **[10]** [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560). arXiv:2310.08560, 2023-10.
- **[11]** [Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces](https://arxiv.org/abs/2601.11868). arXiv:2601.11868, 2026-01.
- **[12]** [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688). arXiv:2308.03688, 2023-08.
- **[18]** [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770). arXiv:2310.06770, 2023-10.

**B. 长程评测：密集奖励、过程指标与时间维度（2026）**

- **[13]** [Long-Horizon-Terminal-Bench: Testing the Limits of Agents on Long-Horizon Terminal Tasks with Dense Reward-Based Grading](https://arxiv.org/abs/2607.08964). arXiv:2607.08964, 2026-07.
- **[14]** [Long-Horizon-Terminal-Bench 2.0: Automating Diverse Task Construction and Measuring Sustained Progress](https://arxiv.org/abs/2610.14146). arXiv:2610.14146, 2026-10.
- **[16]** [Do Agent Optimizers Compound? A Continual-Learning Evaluation on Terminal-Bench 2.0](https://arxiv.org/abs/2607.14004). arXiv:2607.14004, 2026-07.
- **[36]** [Hack-Verifiable Terminal Bench: Evaluating Reward Hacking in Terminal Tasks](https://arxiv.org/abs/2608.22103). arXiv:2608.22103, 2026-08.
- **[37]** [Infrathon: Benchmarking Language Agents for Real-World, Long-Horizon AI Infra Optimization](https://arxiv.org/abs/2611.11995). arXiv:2611.11995, 2026-11.
- **[38]** [Singularity-Bench: Can Agents Build Agents?](https://arxiv.org/abs/2611.16591). arXiv:2611.16591, 2026-11.
- **[39]** [Benchmarks as Software: Terminal-Bench and the Case for Continuous Benchmarks](https://arxiv.org/abs/2611.08417). arXiv:2611.08417, 2026-11.
- **[46]** [THEATRON: A Benchmark for Evaluating Persona Persistence in Long-Horizon LLM Decision-Making](https://arxiv.org/abs/2611.11487). arXiv:2611.11487, 2026-11.
- **[47]** [OdysseyBench: Evaluating LLM Agents on Long-Horizon Complex Office Application Workflows](https://arxiv.org/abs/2508.09124). arXiv:2508.09124, 2025-08.
- **[48]** [StateM: Reaching 95.3% Raw Accuracy, or a $15 Frontier Run, on Terminal-Bench 2.1 via Harness Optimization](https://arxiv.org/abs/2608.15089). arXiv:2608.15089, 2026-08.
- **[49]** [What Makes a Terminal-Bench Task Hard? Separating Genuine Hardness from Fake-Hardness](https://arxiv.org/abs/2609.26826). arXiv:2609.26826, 2026-09.
- **[54]** [Agent Benchmarks Stop Ranking Before They Run Out of Tasks](https://arxiv.org/abs/2611.13541). arXiv:2611.13541, 2026-11.
- **[56]** [CoffeeBench: Benchmarking Long-Horizon LLM Agents in Heterogeneous Multi-Agent Economies](https://arxiv.org/abs/2606.16613). arXiv:2606.16613, 2026-06.
- **[64]** [When Agents Fail: Model or Harness — Who's to Blame?](https://arxiv.org/abs/2610.32483). arXiv:2610.32483, 2026-10.
- **[68]** [SaaS-Bench: Can Computer-Use Agents Leverage Real-World SaaS to Solve Professional Workflows?](https://arxiv.org/abs/2605.15777). arXiv:2605.15777, 2026-05.

**C. 上下文治理、压缩与记忆（2025–2026）**

- **[15]** [Folio: Recoverable Multi-Resolution Context Folding for Long-Horizon Language Agents](https://arxiv.org/abs/2611.16627). arXiv:2611.16627, 2026-11.
- **[17]** [PACE: Keeping the Right Tokens in the KV Cache Is Not Enough for Long-Horizon LLM Agents](https://arxiv.org/abs/2611.10240). arXiv:2611.10240, 2026-11.
- **[19]** [In-Model Context Compression with Agentic Soft Tokens for Long-Horizon LLM Agents (ASTC)](https://arxiv.org/abs/2610.41291). arXiv:2610.41291, 2026-10.
- **[20]** [MarkovSim: an LLM Agent Harness with Bounded Context for Long-Horizon Computational Simulation](https://arxiv.org/abs/2610.35349). arXiv:2610.35349, 2026-10.
- **[21]** [Context as a Tool: Context Management for Long-Horizon SWE-Agents (CAT)](https://arxiv.org/abs/2512.22087). arXiv:2512.22087, 2025-12.
- **[22]** [HIPIF: Hierarchical Planning and Information Folding for Long-Horizon LLM Agent Learning](https://arxiv.org/abs/2606.10507). arXiv:2606.10507, 2026-06.
- **[23]** [MemPrice: Online Memory Allocation via Adaptive Pricing for Long-Horizon Agents](https://arxiv.org/abs/2610.24955). arXiv:2610.24955, 2026-10.
- **[25]** [SagaLLM: Context Management, Validation, and Transaction Guarantees for Multi-Agent LLM Planning](https://arxiv.org/abs/2503.11951). arXiv:2503.11951, 2025-03.
- **[26]** [Cleaning an Agent's History Hurts When the World Still Holds the Error](https://arxiv.org/abs/2611.14654). arXiv:2611.14654, 2026-11.
- **[35]** [TurnCost: Turns, Not Observations — Cache-Aware Loop Engineering for Coding Agents](https://arxiv.org/abs/2610.15093). arXiv:2610.15093, 2026-10.
- **[40]** [Coding Agent Memory Post-Training II: Unlocking the Memory Potential of Pre-trained File Operations](https://arxiv.org/abs/2611.14009). arXiv:2611.14009, 2026-11.
- **[41]** [MemCredit: From Relevance to Utility for Decision-Time Memory in Long-Horizon Web Agents](https://arxiv.org/abs/2611.15600). arXiv:2611.15600, 2026-11.
- **[42]** [When Should Agents Remember? Falsification-Gated Self-Evolution for LLM Agents](https://arxiv.org/abs/2611.14326). arXiv:2611.14326, 2026-11.
- **[43]** [Discovering Effective Agent Memory Designs through Program Evolution (MAPE)](https://arxiv.org/abs/2611.17068). arXiv:2611.17068, 2026-11.

**D. 规划、过程约束、自我终止与错误恢复**

- **[24]** [Keeping LLM Agents on Procedure: Executable Finite-State Control for Long-Horizon Instruction Following](https://arxiv.org/abs/2611.10884). arXiv:2611.10884, 2026-11.
- **[30]** [Knowing When to Stop: The Evidence-Action Gap in Coding Agents (ImpasseBench)](https://arxiv.org/abs/2610.31895). arXiv:2610.31895, 2026-10.
- **[31]** [Same Beliefs, Different Policies: Learning Reusable Representations for Early Stopping in LLM Reasoning (BELT)](https://arxiv.org/abs/2610.30450). arXiv:2610.30450, 2026-10.
- **[32]** [ReflectPO: Reflection-Guided Policy Optimization for Long-Horizon LLM Agents](https://arxiv.org/abs/2610.30933). arXiv:2610.30933, 2026-10.
- **[34]** [When Tools Fail: Benchmarking Dynamic Replanning and Anomaly Recovery in LLM Agents (ToolMaze)](https://arxiv.org/abs/2606.05806). arXiv:2606.05806, 2026-06.

**E. 持久运行时：状态、再校验、遗忘与编排**

- **[50]** [CoworkOS: Scaling Agent Coworkers for Real-world Automation](https://arxiv.org/abs/2611.03178). arXiv:2611.03178, 2026-11.
- **[51]** [From Version Conflicts to Decision Conflicts: Selective Revalidation for Long-Running AI Agents](https://arxiv.org/abs/2609.08015). arXiv:2609.08015, 2026-09.
- **[52]** [Forgetting Without Restarting: Execution-State Unlearning for Stateful LLM Agents](https://arxiv.org/abs/2609.04875). arXiv:2609.04875, 2026-09.
- **[53]** [Goal-Aware Dynamic Multi-Agent Collaboration for Long-Horizon Tasks](https://arxiv.org/abs/2610.39193). arXiv:2610.39193, 2026-10.
- **[55]** [Auditing Action Settlement in LLM Agent Environments: Order, Progress, and Replay](https://arxiv.org/abs/2610.01138). arXiv:2610.01138, 2026-10.
- **[69]** [ToolSpace: Scaling Tool-Centric Environment Synthesis for Long-Horizon Agents](https://arxiv.org/abs/2611.05036). arXiv:2611.05036, 2026-11.

**F. Harness / 轨迹自进化与长程信用分配**

- **[27]** [Harness Self-Evolution for Long-Horizon Terminal Agents (RiseHarness)](https://arxiv.org/abs/2610.27435). arXiv:2610.27435, 2026-10.
- **[28]** [Scaling Trajectories for Complex Tasks through Recursive Self-Rewrite](https://arxiv.org/abs/2610.02826). arXiv:2610.02826, 2026-10.
- **[29]** [Terminal-RSI: Recursive Self-Improving Terminal Agents via Recursive Self-Rewrite and Policy-Refine](https://arxiv.org/abs/2610.22192). arXiv:2610.22192, 2026-10.
- **[33]** [Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?](https://arxiv.org/abs/2511.13646). arXiv:2511.13646, 2025-11.
- **[44]** [Milestone-Guided Policy Learning for Long-Horizon Language Agents (BEACON)](https://arxiv.org/abs/2605.06078). arXiv:2605.06078, 2026-05.
- **[45]** [AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress](https://arxiv.org/abs/2511.08325). arXiv:2511.08325, 2025-11.
- **[57]** [Hindsight Credit Assignment for Long-Horizon LLM Agents (HCAPO)](https://arxiv.org/abs/2603.08754). arXiv:2603.08754, 2026-03.
- **[58]** [ABSeeker: Training Long-Horizon Search Agents via Answer-Backtracked Credit Assignment](https://arxiv.org/abs/2608.05102). arXiv:2608.05102, 2026-08.
- **[65]** [TRCA: Transition-wise Rubric Credit Assignment for Long-horizon LLM Agents](https://arxiv.org/abs/2608.16156). arXiv:2608.16156, 2026-08.
- **[66]** [Evolving to Match Capabilities: Improving Harness Self-Evolution via Skill Adaptation](https://arxiv.org/abs/2610.33024). arXiv:2610.33024, 2026-10.
- **[67]** [Intrinsic Credit Assignment for Long Horizon Interaction](https://arxiv.org/abs/2602.12342). arXiv:2602.12342, 2026-02.
- **[70]** [Hindsight Credit Assignment over Trees for Agentic RL](https://arxiv.org/abs/2611.02706). arXiv:2611.02706, 2026-11.

**G. 产业与一手网页信源（非 arXiv）**

- **[59]** LangChain. [Deep Agents: Open Source Agent Harness](https://www.langchain.com/deep-agents)（官方产品页，含 planning tools / 隔离子代理 / 虚拟文件系统 / 上下文中间件；页面另披露 LangChain 完成 \$300M C 轮、估值 \$3B），访问于 2026-11.
- **[60]** Sequoia Capital, *Training Data* Ep. 77. [Context Engineering Our Way to Long-Horizon Agents: LangChain's Harrison Chase](https://www.sequoiacap.com/podcast/context-engineering-our-way-to-long-horizon-agents-langchains-harrison-chase)（Sonya Huang / Pat Grady 对谈，直读文字实录），2026.
- **[61]** Anthropic. [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)（工程博客：orchestrator–worker、3–5 并行子代理、研究时间 −90%、工具人体工学 −40%、多智能体约 15× chat token、BrowseComp 上 token 用量解释约 80% 方差），2025-06 / 2026 访问.
- **[62]** 勘误与口径说明：本报告未采用部分中文媒体转述的「红杉命名第三次范式拐点」表述——直读红杉一手页面后未找到该提法；红杉 David Cahn《AI's \$600B Question》（2024-06-20）讨论的是 AI 资本开支而非长程智能体范式，故不据此立论。
- **[63]** Zhongzhu (Charlie) Zhou. [Long-Horizon-Terminal-Bench 阅读笔记：密集奖励评估揭露 Agent 长程执行瓶颈](https://www.zhongzhuzhou.org/blog/2026-07-13-longhorizonterminalbench-阅读笔记-zh/)（第三方精读，失败模式占比约 79% 超时 / 约 19% 主动退出、单任务成本表与 False Finish 案例的来源；论文级数字以 [13][14] 原文摘要为准），2026-07-13.
