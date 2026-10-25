# Test-Time Scaling 前沿技术报告：当大模型学会「在回答前思考」

> **测试时计算扩展（Test-Time Scaling, TTS）**：在模型权重冻结之后，通过在推理阶段投入更多计算——更长的思维链、并行采样、树/图搜索、自我验证、测试时学习——来换取更高的任务表现。它是继预训练缩放（scaling model size）、后训练对齐之后，被工业界广泛采用的**第三条能力扩展轴**。
>
> 关键词：Test-Time Compute · Reasoning Model · Chain-of-Thought · Best-of-N · Process Reward Model · MCTS · Budget Forcing · Overthinking · Adaptive Compute · Latent Reasoning · Test-Time RL

---

## 摘要

2024 年 9 月 OpenAI 发布 o1-preview，把一个学术界已讨论两年的想法——「让模型在回答前想久一点」——变成了产品级的能力跃迁：在 ARC-AGI、竞赛数学、代码等高难任务上，推理模型以数量级的测试时算力换来了普通「快思考」模型无法企及的准确率。2024 年 8 月 Snell 等人的奠基性实证表明，**在总 FLOPs 固定时，把计算从「扩大模型」转移到「测试时搜索」，可以击败 14 倍大的模型**，且最优分配应随题目难度自适应 [2408.03314]。2025 年 1 月 DeepSeek-R1 用纯强化学习（无人工推理标注）复现并开源了这一能力，引爆了整个开源生态 [2501.12948]。

一年半之后（2026 年 10 月），TTS 已经从「一项技巧」长成一个完整的技术子领域：它有自己的缩放定律（覆盖率随采样数呈幂律增长 [2407.21787]）、自己的核心矛盾（**过度思考 overthinking**——简单题也长篇大论，造成延迟与成本浪费 [2412.21187]）、自己的控制接口（reasoning effort、thinking budget），并正在向**自适应计算分配**、**潜空间推理**、**测试时强化学习**、**多模态/语音/具身**等方向纵深发展。截至 2026-10，arXiv 标题中含 "test-time scaling" 的论文已近千篇，仅 2026 年 10 月单月就有数十篇，是当前大模型方向中论文密度最高、产业关联最紧的领域之一。

本报告基于 **48 篇逐一核实的 arXiv 论文**（编号、标题、日期均经 arxiv.org 摘要页 meta 标签核对，含多处撞号勘误）与 OpenAI / Anthropic / Google DeepMind / Qwen / ARC Prize 等一手信源，系统梳理 TTS 的思想源流、两大计算模式（顺序修订 vs 并行采样）、验证器与搜索、自适应计算、潜空间推理、测试时学习、产业落地与推理经济，并给出开放问题与工程选型清单。

---

## 1. 引言：第三条扩展轴

### 1.1 从「缩放模型」到「缩放思考」

过去十年大模型的主线叙事是**参数缩放**：更多数据、更大模型、更多训练 FLOPs，能力沿 Chinchilla 式缩放定律平滑增长。但到 2024 年，这条曲线出现两个现实约束：一是前沿模型训练成本已达数亿美元量级、单次训练以月计；二是在数学竞赛、ARC-AGI 这类需要**多步组合推理**的任务上，单纯扩大模型的收益明显放缓——GPT-4o 在 ARC-AGI-1 上仅约 5%。

Test-Time Scaling 提供了一个正交的杠杆：**模型权重不变，在单次推理中花更多 FLOPs**。其直觉来自人类认知——面对 `2+3=?` 我们脱口而出，面对一道 IMO 压轴题则需要打草稿、试错、回头检查。Kahneman 的「系统 1 / 系统 2」双过程理论是这一范式最常被引用的隐喻：普通前向采样是快速、直觉的系统 1，而带长思维链与搜索的推理是缓慢、审慎的系统 2。

需要强调的是，TTS 并非某一种算法，而是一族**推理时计算分配策略**的统称。Snell 等把测试时计算粗略分为两类 [2408.03314]：

- **顺序修订（sequential revision）**：从一个初稿出发，模型逐步生成、自我评估、回溯与改进，计算沿「时间」展开，代表是显式/隐式思维链、MCTS；
- **并行采样（parallel sampling）**：同时独立生成多个候选解，再用验证器或多数投票选出最好的，计算沿「宽度」展开，代表是 Best-of-N、self-consistency。

真实系统几乎总是二者的混合（例如先并行起多条推理树，每条内部再顺序展开）。

### 1.2 为什么现在成为主流

TTS 之所以在 2024–2026 爆发，是三类条件同时成熟：

1. **可验证任务与 RLVR 的成熟**：数学、代码等任务有标准答案或可执行验证器，使得强化学习可以用结果奖励（RLVR）直接训练模型「为了答对而多思考」，无需昂贵的过程标注；DeepSeek-R1 证明纯 RL 即可让自我反思、验证、策略切换等行为自发涌现 [2501.12948]。
2. **推理算力相对训练算力更廉价、更弹性**：测试时计算是按请求分摊的小额开销，无需一次性巨额训练投入；它还让「用小模型 + 多思考逼近大模型」成为可能（s1 用 1,000 条样本微调 32B 即超 o1-preview [2501.19393]；更有研究以 1B 模型 + 测试时计算叫板 405B [2502.06703]）。
3. **产品与 API 的工程化**：OpenAI、Anthropic、Google、阿里等把「思考预算」做成显式可控参数（reasoning effort / thinking budget），使开发者能在延迟、成本、质量之间连续权衡。

### 1.3 本报告的组织与信源纪律

第 2 章梳理 2022–2024 的思想源流；第 3 章讲奠基论文与第一性原理；第 4 章讲 o1 时刻与 budget forcing；第 5 章讲 R1 冲击波与开源生态；第 6、7 章分别讲验证器与搜索；第 8 章是当前最核心的自适应计算与过度思考治理；第 9、10 章讲潜空间推理与测试时学习；第 11 章讲产业落地与推理经济；第 12 章讨论开放问题。

**信源纪律**：所有 arXiv 编号均经 `arxiv.org/abs/<id>` 页面的 `citation_title / citation_author / citation_date` 元数据逐一核对；正文关键测量在句末标注编号；厂商自报数据标注「厂商口径」，第三方（如 Artificial Analysis、ARC Prize 复测）标注「第三方」。网传但无法核实的数字一律不采信。

---

## 2. 思想源流（2022–2024）：TTS 不是一夜之间出现的

o1 常被叙事为「横空出世」，但它的每一块拼图在 2022–2024 的学术工作中都已就位。理解这条谱系，才能看清今天的前沿在解决什么。

### 2.1 思维链：把中间推理显式化（2022）

一切的起点是 **Chain-of-Thought Prompting（CoT）**[2201.11903]：Wei 等人发现，只要在 few-shot 提示中给出带中间步骤的推理示例，大模型就能把复杂问题分解为多步、逐步求解，并在 GSM8K 等基准上获得显著增益。CoT 的关键意义在于：**它把「推理长度」变成了一个可操纵的变量**——模型不再被要求一步出答案，而是被允许生成一串推理 token。这为「多生成 token → 更高准确率」提供了最初的经验证据。

但 CoT 是线性的：一条路走到黑，中间错了无法回头。

### 2.2 自我改进与言语强化：让模型当自己的裁判（2023）

2023 年出现了一批「不更新权重、仅靠推理时迭代」的工作，构成顺序修订路线的源头：

- **Self-Refine** [2303.17651]：同一个模型先后扮演生成器、反馈者、改进者，对自己的输出提出批评再据此修改，无需任何训练；在 7 类任务上平均绝对提升约 20%。
- **Reflexion** [2303.11366]：把一次尝试失败后的经验以自然语言「反思」写入情节记忆，供下一次尝试参考，在决策、推理、编程等交互任务上显著提升。
- **Self-Rewarding Language Models** [2401.10020]：让模型既当候选生成器又当奖励模型，用自己的打分做 DPO 迭代训练，推动质量随轮次上升——首次展示「模型自己定义好坏」的闭环。

这一脉络确立了 TTS 的一个核心动作：**生成 → 自评 → 修订**。

### 2.3 树搜索与过程监督：从「一条线」到「一张图」（2023）

并行/搜索路线同样在 2023 年成形：

- **Tree of Thoughts（ToT）**[2305.10601]：把推理组织成一棵思维树，每个节点是一个「想法」，模型用启发式自评对节点打分，用 BFS/DFS 做前瞻、回溯与剪枝。在 24 点、创意写作等需要搜索的任务上，ToT 让 GPT-4 能系统性地探索多条路径。
- **Let's Verify Step by Step**（Lightman 等，OpenAI）[2305.20050]：系统比较**结果奖励模型（ORM）**与**过程奖励模型（PRM）**，证明对每一个中间步骤打分（过程监督）显著优于只看最终答案，并发布了 **80 万条人工步骤级标注 PRM800K**；过程监督模型在 MATH 代表性子集上达 78%。这篇论文奠定了「用验证器引导搜索」的方法论与数据基础。
- **Math-Shepherd** [2312.08935]：针对 PRM800K 标注昂贵的痛点，提出**自动构造过程奖励**——从每个中间步骤出发多次随机续跑，用「最终能否走到正确答案」的比例作为该步骤的奖励，无需人工标注；Mistral-7B 经逐步 PPO 后 MATH 从 28.6% 升到 33.0%，再加验证重排达 43.5%。

到 2023 年底，「树搜索 + 过程验证」的配方已经完整，后来的 o1 与众多开源推理模型本质上是把这套配方用强化学习**内化进权重**，而不再依赖外部显式搜索。

### 2.4 内化理性：Quiet-STaR、ReST-MCTS* 与重复采样（2024）

o1 发布前的几个月，几条线索进一步收敛：

- **Quiet-STaR** [2403.09629]：让模型在**每个 token 之后**都生成一段内部「理性（rationale）」，用后续 token 的可预测性来评估这段理性有没有用，并据此泛化地训练模型「逢事就想」，把 CoT 从提示技巧变成模型的通用习惯。
- **ReST-MCTS\*** [2406.03816]：把过程奖励引导的 MCTS\* 与自训练结合，自动推断步骤奖励、用树搜索收集高质量轨迹，再回头训练策略与奖励模型，形成「搜索 → 数据 → 训练」的闭环。
- **Large Language Monkeys** [2407.21787]：用最简单的**重复采样（repeated sampling）**作为扩展轴，发现**覆盖率（至少一个样本正确的概率）随样本数呈幂律、跨 4 个数量级增长**；在 SWE-bench Lite 上 DeepSeek-Coder-V2 从单样本 15.9% 提到 250 样本的 56%。这为并行 TTS 给出了清晰的「推理时缩放定律」。

至此，o1 所需的三块基石——可操纵的推理长度、过程验证、以及搜索/采样的缩放规律——全部就绪。
---

## 3. 奠基论文与第一性原理：Snell et al. 2024

2024 年 8 月（早于 o1-preview 一个月）Google DeepMind 的 Snell、Lee、Xu、Kumar 发表《Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters》[2408.03314]，这是 TTS 范式被正式命名并给出实证规律的奠基之作。

### 3.1 三个核心发现

**发现一：测试时计算可以替代模型参数。** 在总 FLOPs（训练 + 推理）固定的约束下，把更多算力分配给测试时搜索、用更小的模型，可以击败「大模型 + 少思考」的配置——论文报告 compute-optimal 策略在 FLOPs 匹配下可击败约 **14 倍大**的模型。

**发现二：存在 compute-optimal 的测试时计算量，且它是「题目难度」的函数。** 对简单题，多思考纯属浪费（模型一次就能答对）；对极难题，给再多计算也可能卡在能力天花板之外；**收益最大的是中等难度、模型「跳一跳够得着」的题目**。因此最优策略不是「一律多想」，而是**按题自适应分配**。

**发现三：顺序修订与并行采样各有适用区间，且需要好的验证机制。** 论文报告 compute-optimal 的动态分配比固定的 best-of-N 基线效率高 **4 倍以上**；但顺序修订的收益高度依赖「模型能否准确判断自己当前的解好不好」——验证/回溯机制失效时，额外 token 只是在错误道路上越走越远。

### 3.2 第一性原理：TTS 的收益从哪里来

把上述实证抽象成第一性原理，测试时计算的增益来自三个机制：

1. **摊销幸运（amortizing luck）**：自回归采样有随机性。一次采样答对的概率为 p，则 N 次独立采样至少一次答对（pass@N / 覆盖率）为 1−(1−p)^N。只要 p>0，增大 N 就能逼近 1——这就是 Monkeys 观察到的幂律 [2407.21787]。并行 TTS 的本质是「多抽几次奖，再选出中奖的那张」。
2. **分解与回溯（decomposition & backtracking）**：把难题显式拆成可验证的小步，一旦某步走错可以退回分叉点换一条路（ToT/MCTS），把指数大的解空间里的搜索做「深」做「广」。
3. **自我验证（self-verification）**：生成候选后，用 ORM/PRM/多数投票/可执行测试等机制判别优劣，把「采样到好解」转化为「选出好解」。验证器质量决定了并行宽度能否兑现为 pass@1。

### 3.3 两个被反复验证的推论

- **TTS 不是免费午餐**：增益受限于模型的「天花板能力」与验证器质量；在模型本就会的题上，多算只是浪费——这正是后来 overthinking 研究的伏笔（第 8 章）。
- **训练时与测试时应当协同缩放**：要让模型从额外思考中获益，需要先用 RLVR 等手段训练它「会用」这些计算（会反思、会验证、会在错的时候停）。一个没经过推理训练的普通 Instruct 模型，给它再多 token 也不会自动变聪明。这解释了为什么 o1/R1 的关键不只是推理时算法，更是**产生长 CoT 的训练方法**。

---

## 4. o1 时刻与 budget forcing：推理模型的产品化

### 4.1 OpenAI o1：把系统 2 做成产品

2024 年 9 月 12 日，OpenAI 发布 **o1-preview 与 o1-mini**，12 月 5 日 o1 转正。官方将其定位为「学会使用强化学习进行推理的新模型」：它在回答前先用一段**隐藏的内部思维链（chain of thought）**进行思考，再给出答案，在科学、数学、编程等需要深度思考的任务上相对 GPT-4o 取得阶跃式提升。o1 的产品意义有三：

- 首次把「测试时计算」从学术技巧变成**默认开启的产品能力**，并按更高的 token 单价计费（o1 API 定价输入 $15 / 输出 $60 每百万 token，显著高于当时 GPT-4o，第三方模型档案亦记录其 2024-12 转正）【官方口径/第三方】；
- 引入「reasoning tokens」概念——思考过程消耗的 token 不向用户展示全文，但计入成本与限额，确立了「思考也是要花钱的算力」这一商业事实；
- 刺激了整个行业：在随后的半年里，混合推理（快速/深度思考可切换）成为旗舰模型标配。

o1 的训练细节 OpenAI 未完全公开，但业界普遍从系统卡与后续复现推断其核心是：**用可验证奖励的强化学习（RLVR）训练模型生成长推理轨迹**，并辅以推理风格的冷启动数据。

### 4.2 s1：1,000 条样本与 budget forcing

o1 闭源，真正把「推理模型怎么训」讲清楚并引爆开源复现的，是 2025 年 1 月的 **s1: Simple Test-Time Scaling** [2501.19393]（Muennighoff 等，华盛顿大学/斯坦福等）。它的主张近乎挑衅地简单：

- **数据**：人工从少量多样来源精选 **1,000 条**带长推理轨迹的样本（s1K），在 Qwen2.5-32B 上做有监督微调；
- **训练**：标准 SFT，没有复杂 RL；
- **推理技巧——budget forcing（预算强制）**：在解码时操纵特殊标记来控制思考时长：
  - 当模型提前产出 `<end_think>` 想要结束思考时，不允许它停，而是追加一个 `"Wait"`，强制它继续想下去（**延长**）；
  - 或者在达到预算时直接注入结束标记，强制它收尾（**截断**）。

效果：s1-32B 在 MATH、AIME 2024 等竞赛数学上最高比 o1-preview 高出 **27%**；budget forcing 把 AIME 2024 从 50% 提到 57%。s1 的重要性在于证明了两件事：**(a) 推理能力可以靠极少高质量数据「嫁接」到现成强基座上；(b) 思考长度是推理时一个可直接拨动的旋钮**——这为后来所有「思考预算控制」接口提供了原型。

### 4.3 journey learning 与 O1 复现潮

o1 发布后社区迅速进入复现竞赛。**O1 Replication Journey（Part 1）** [2410.18982]（港中文等，2024-10）以全程公开「研究日志」的方式透明复现，提出 **journey learning**：训练数据不应只保留干净的「捷径」解法，而应包含模型真实探索中的**试错、反思、回溯**全过程——模型要学的是「怎么走完一段曲折的旅程」，而不是「背下终点」。仅 327 条旅程样本就在 MATH 上比普通监督学习高 8% 以上。

**Marco-o1** [2411.14405]（微软，2024-11）则把 CoT 微调、MCTS 与反思结合，主攻**没有标准答案、奖励难以量化的开放式任务**（如开放性问答、医学），把推理范式从数学/代码向外推广。

这一阶段的共识是：**长 CoT 数据的质量与「探索性」比数量更重要**，这也直接导向了下一章的 R1——既然数据这么关键，能否让模型自己造出这些数据？

---

## 5. R1 冲击波：纯强化学习与开源推理生态

### 5.1 DeepSeek-R1：纯 RL 激发推理，零人工 CoT 标注

2025 年 1 月 22 日 DeepSeek 发布 **DeepSeek-R1** 技术报告 [2501.12948] 与 MIT 许可的开源权重，给出了当时最完整的开源推理模型训练范式。其关键结论是：**对一个足够强的基座，直接用可验证奖励做大规模 RL（GRPO），无需任何人工标注的推理示例，模型就会自发发展出长链推理行为。**

- **R1-Zero**：在 DeepSeek-V3-Base 上直接做 RL（规则化的正确性奖励 + 格式奖励），**无冷启动 SFT**。模型自己涌现出「反思（重新审视自己的步骤）、验证、动态策略切换」等行为（即广为流传的 **aha moment**），但可读性与语言混杂较差。
- **R1**：在 R1-Zero 与基座之间加入**少量长 CoT 冷启动数据**，再经过两阶段 RL（推理导向 + 通用对齐导向），得到既强又可读、可通用的模型；报告正文报告 R1 在 AIME 2024 达 79.8%、MATH-500 达 97.3%，对标 o1【厂商口径】。
- **蒸馏**：把 R1 的长 CoT 推理轨迹蒸馏到 Qwen/Llama 的 1.5B–70B 小模型，小模型能力大幅跃升——证明「会思考」这一行为可以通过数据高效迁移。

R1 的影响是生态级的：MIT 开源 + 完整报告 + 可蒸馏，使全球团队第一次能在本地复现和改造 o1 级推理。它也确立了一套被反复沿用的配方：**强基座 → （可选）冷启动长 CoT SFT → 规则可验证奖励的大规模 RL → 蒸馏**。

### 5.2 Kimi k1.5：长上下文 + RL，明确不用 MCTS/PRM

几乎同期（2025-01-22）月之暗面发布 **Kimi k1.5** [2501.12599]，报告 AIME 77.5、MATH-500 96.2、Codeforces 第 94 百分位、MathVista 74.9【厂商口径】。它在方法论上提供了一个与「树搜索派」对立的重要论点：**仅靠长上下文 + 改进的策略优化做 RL，即可达到强推理，不需要 MCTS、价值函数或 PRM**。k1.5 还提出 **long2short**：先用长 CoT 训练/推理，再把长推理压缩成短 CoT，短 CoT 模型相对直接短训最高提升达 +550%——这是「先学会深思、再学会速答」的早期代表，与后来的过度思考治理呼应。

### 5.3 开源复现的系统性发现

R1 之后，大量工作开始严谨地复现并质疑「零 RL」的稳健性：

- **SimpleRL-Zoo** [2503.18892]：在 **10 个不同基座**（LLaMA3-8B、Mistral、DeepSeek-Math、Qwen2.5 从 0.5B 到 32B）上系统复现 R1 式 zero RL。结论泼了一盆冷水：**「aha moment」与稳定提升高度依赖基座**——在 Qwen 系之外的许多模型上，响应长度虽然增长，但真正的验证/反思行为并不必然涌现。这说明 R1-Zero 的成功部分建立在 DeepSeek/Qwen 基座已经过的强预训练与数据清洗之上。
- **Skywork Open Reasoner 1（Skywork-OR1）** [2505.22312]：在 R1-Distill 基础上做可扩展的长 CoT RL，32B 模型在 AIME24/AIME25/LiveCodeBench 平均从 57.8% 提到 72.8%（+15.0%），7B 从 43.6% 提到 57.5%，并在 AIME 上超过 R1 与 Qwen3-32B【厂商口径】。
- **PRIME（Process Reinforcement through Implicit Rewards）** [2502.01456]：提出不必单独训练一个 PRM——用策略自己的 rollout 与结果标签在线生成**隐式过程奖励**，Eurus-2-7B-PRIME 仅用约 10% 数据就在 7 个基准上超过 Qwen2.5-Math-7B-Instruct，平均比 SFT 模型提升 15.1%。
- **LIMO: Less is More for Reasoning** [2502.03387]：再次印证「少而精」——用极少量高质量推理样本即可让模型显著变强，与 s1 的结论相互加强。

### 5.4 双模式成为旗舰标配：Qwen3 与全球跟进

2025 年 4 月 29 日阿里发布并开源 **Qwen3** 系列（旗舰 Qwen3-235B-A22B，MoE；另有 30B-A3B 与 0.6B–32B 稠密模型，Apache 2.0）[官方博客]。Qwen3 的标志性设计是**思考/非思考双模式（thinking / non-thinking）与可平滑调节的思考预算**：官方明确指出模型表现「与分配的推理计算预算直接相关、且平滑可扩展」，用户可按任务配置预算，在成本与质量间取平衡，还可用 `/think` 与 `/no_think` 指令逐问切换。

至此，**「一个模型、两种思考速度、一个预算旋钮」成为旗舰模型的通用产品形态**：Anthropic Claude 3.7 Sonnet（2025-02-24，市场上首个 hybrid reasoning model，extended thinking 预算可通过 API 精细控制到 token 级，上限 128K，思考 token 与输出同价 $3/$15）[官方]、Google Gemini 2.5 Pro（2025-03，把 thinking 能力直接做进模型，特别强调「不使用 majority voting 等增加成本的测试时技巧」即取得 GPQA/AIME 2025 领先、HLE 18.8%）[官方]、xAI Grok 3、智谱 GLM 等全部跟进。开源界则进一步把这套能力下放到 0.6B 级小模型。

> ![](diagrams/1-timeline.html)
> **图 1 · TTS 技术谱系时间线（2022→2026）**
---

## 6. 验证器：并行 TTS 的「裁判」问题

并行采样生成 N 个候选之后，系统必须回答一个问题：**哪个是对的？** 验证器（verifier / reward model）的质量，直接决定了「采样到好解」能否兑现为最终的 pass@1。本章梳理验证器的技术谱系。

### 6.1 ORM 与 PRM：结果监督 vs 过程监督

- **结果奖励模型（Outcome RM, ORM）**：只对完整解答的最终答案打分。简单、标注便宜，但信号稀疏——一条轨迹错了，不知道错在哪一步；且正确答案可能伴随错误推理（侥幸答对）。
- **过程奖励模型（Process RM, PRM）**：对推理轨迹的**每一个中间步骤**打分。Lightman 等证明在 MATH 上过程监督显著优于结果监督，最好的 PRM 在大规模 N 采样后可达约 78.2%，并贡献了 PRM800K 数据集 [2305.20050]。

PRM 的致命短板是**成本**：PRM800K 需要人工逐条标注步骤对错。Math-Shepherd 用「从该步骤出发多次随机续跑、统计到达正确答案的比例」自动估计步骤价值，把人工标注去掉 [2312.08935]；PRIME 则进一步把过程奖励隐式化、在线生成，连独立 PRM 都不必训练 [2502.01456]。

### 6.2 从「打分器」到「会思考的验证器」

2025 年的一个重要转向是：**让验证器本身也做 TTS**。

- **Process Reward Models That Think（ThinkPRM）** [2504.16828]：把判别式 PRM 换成一个**生成式、会说出验证理由**的验证器——它对每个步骤先生成一段验证 CoT 再下判断。仅用 PRM800K 的 **1%** 标签，在分布外的 GPQA-Diamond / LiveCodeBench 上比用全量 PRM800K 训练的判别式验证器高 8% / 4.5%，同预算下比 LLM-as-a-Judge 高 7.2%。
- 这与「生成式奖励模型 / 生成式判别器」的大趋势一致：与其让一个小模型输出标量分，不如让一个强模型用推理来判断，验证器自己也享受测试时计算的红利。

### 6.3 不完美验证器下的计算分配

现实中的验证器既会漏判（false negative）也会误判（false positive）。2026 年的工作开始系统研究「验证器不完美时，计算该怎么花」。《Sample, Mix, or Verify? Allocating Test-Time Compute under Imperfect Verifiers》[2610.23255] 把测试时预算在三种动作间分配——**多采样（sample）、混合/融合候选（mix）、再验证（verify）**——并给出在验证器有噪条件下的最优分配策略：当验证器较弱时，盲目增加采样数收益迅速饱和，反而是「交叉验证」「候选融合」更划算。这类工作把第 3 章 Snell 的「按难度分配」进一步细化为「按验证器质量分配」。

### 6.4 无需学习验证器的路线

并非所有系统都依赖训练好的奖励模型：

- **多数投票 / self-consistency**：N 个候选中取出现次数最多的答案，零训练，是数学等有确定答案任务的强基线；
- **可执行验证**：代码任务直接跑单测、工具调用任务直接看执行结果（第 8 章的 TEX、以及 Agentic RL 中的环境反馈）；
- **LLM-as-a-Judge / 成对比较**：用强模型对候选排序。《Compare to Correct: Pairwise-Guided Logit Adjustment》[2610.18583] 进一步把成对比较的信号注入解码 logit，在无需重训的前提下提升 TTS。

验证器的选择本质是在**标注成本、判别精度、分布外鲁棒性**之间做权衡，这也是 R1/k1.5 干脆用「规则化正确答案 + RL」绕开 learned PRM 的原因。

---

## 7. 搜索：从 Best-of-N 到 MCTS，再到可证明的缩放

### 7.1 三种搜索结构的计算拓扑

| 搜索结构 | 计算拓扑 | 代表方法 | 优点 | 代价 |
|---|---|---|---|---|
| Best-of-N / 重复采样 | 并行、无共享 | self-consistency、Monkeys [2407.21787] | 实现极简、易批量、覆盖率幂律增长 | 无信息共享，N 大时冗余 |
| 束搜索 / 序列修订 | 顺序、线性 | CoT、Self-Refine [2303.17651]、束搜索 | 单条轨迹可极深、延迟友好 | 不能回溯，一步错一路错 |
| 树/图搜索 | 并行+顺序、节点共享 | ToT [2305.10601]、ReST-MCTS* [2406.03816]、MCTS | 可回溯、可前瞻、节点复用 | 需要步骤价值估计，工程复杂 |

### 7.2 MCTS 与「搜索内化」之争

树搜索派（ToT、ReST-MCTS*、rStar-Math [2501.04519] 等）主张显式地在推理时展开一棵用 PRM 引导的蒙特卡洛树：rStar-Math 让小模型通过**自演化的深度思考（MCTS + 过程验证）**掌握数学推理，是「小模型 + 重搜索」路线的标杆。《Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping》[2402.14083] 则展示了如何把 A*/MCTS 的搜索动力学蒸馏进 Transformer，让模型在单次前向中模仿搜索。

但 o1/R1/k1.5 的工业实践给出了另一条路：**把搜索「内化」进权重**——用 RL 训练模型在一条（可能很长的）自回归轨迹里隐式地完成试错与回溯，推理时不再需要外部树结构。Kimi k1.5 明确声称不需要 MCTS/价值函数/PRM [2501.12599]。两派的权衡是：显式搜索**可解释、可精确控制预算、在小模型上收益大**，但有调度与延迟开销；内化搜索**部署简单、一条流跑完**，但需要昂贵的 RL 训练，且回溯行为不可控。2025–2026 的趋势是**融合**：用轻量搜索/验证增强内化推理模型，而非二选一。

### 7.3 顺序 TTS 的探索效率与可证明性

2026 年下半年的研究焦点转向「让顺序推理更会探索」与「给 TTS 一个理论保证」：

- 《Towards Better Exploration in Sequential Test-Time Scaling》[2609.39632] 指出顺序扩展的瓶颈是**探索不足**——模型倾向于在相似轨迹上重复，提出更好的探索机制来提升单位 token 的信息增益；
- 《Provable Test-Time Scaling for Beam Search in LLM Reasoning》[2609.38672] 为束搜索式 TTS 给出可证明的缩放条件，回答「什么情况下加宽束宽一定能提高正确率」；
- 《Cheap to Draw, Expensive to Trust: Certifying Test-Time Scaling Curves》[2609.40190] 则关注**认证（certification）**：采样很便宜，但「信任」选出的答案很贵——如何对整条「计算量–准确率」缩放曲线给出统计认证，避免被少数侥幸样本误导；
- 《Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals》[2610.11570] 把测试时迭代次数推到**上万次**，用循环原生（loop-native）的注意力残差解决超深迭代下的漂移与显存问题，探索 TTS 在极端深度的边界；
- 《Taming Speculative Search for Test-Time Scaling in LLM Serving》[2609.39334] 从系统侧入手，用投机执行加速推理时搜索，直接面向 serving 的延迟约束。

这些工作标志着 TTS 研究从「证明多想有用」进入「**让多想更省、更稳、可证明**」的精细化阶段。

> ![](diagrams/2-two-modes.html)
> **图 2 · 两大计算模式与搜索结构（顺序修订 vs 并行采样）**
---

## 8. 自适应计算与过度思考治理：当前最核心的工程矛盾

TTS 走向大规模落地后，最尖锐的矛盾不再是「能不能更强」，而是「**能不能别在简单题上浪费算力**」。这就是过度思考（overthinking）问题，也是 2025–2026 论文最密集、产品决策最纠结的方向。

### 8.1 什么是过度思考

**《Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs》** [2412.21187]（2024-12-30，首个系统研究）把过度思考定义为：推理模型对本可快速作答的简单问题仍生成冗长思维链，造成**计算/延迟浪费**，甚至在长推理中「想太多反而出错」（推理链越长，累积错误与跑偏概率越高）。论文从**结果效率（单位算力的正确率）与过程效率**两个视角刻画该问题，并用自训练方法在不掉点的前提下压缩推理长度。

过度思考有三种典型表现：

1. **简单题长答**：`2+3` 也要铺垫一大段；
2. **重复打转（loops）**：在同一步骤附近反复确认、循环不前；
3. **晚节不保**：本来已经推对，却在后续冗长推理中把答案改错。

其根源是训练目标错配：RLVR 只奖励「最终答对」，**不奖励「用更少 token 答对」**，于是模型学到的最优策略是「宁可多想」——长度没有被显式惩罚。

### 8.2 长度控制的四条技术路线

针对过度思考，2025–2026 发展出四条互补路线：

**路线一：推理时预算控制（inference-time budget）。** 最直接——在提示或 API 里限定 token 预算。

- **s1 的 budget forcing**：到点强制结束或注入 `"Wait"` 延长 [2501.19393]；
- **Token-Budget-Aware LLM Reasoning** [2412.18547]：发现仅在提示中给出一个合理的 token 预算，就能显著压缩冗长推理，并提出按问题复杂度动态设定预算；
- 工业接口：OpenAI 的 `reasoning_effort`（low/medium/high）与 `max_completion_tokens`、Anthropic 的 `thinking.budget_tokens`（可精细到 token，最高 128K）、Qwen3 的思考预算平滑调节——本质都是把「想多久」的决定权交给开发者。

**路线二：自适应触发与提前停止（adaptive triggering / early stopping）。** 让模型自己判断「这题要不要想、想够了没」。

- **AdaCoT** [2505.11896]：用 PPO 把「**是否触发 CoT**」建模为帕累托最优决策，并用 Selective Loss Masking 防止决策边界坍缩；在生产流量测试集上把 CoT 触发率降到 **3.18%**、平均响应 token 减少 **69.06%**，而复杂任务性能基本不变——即「简单题直接答、难题才深思」。
- **Calibration-Free Early Stopping via E-processes** [2610.25212]：不依赖置信度校准，用统计学的 **E-process（e-process）** 对「答案是否已稳定正确」做序贯检验，一旦证据足够就提前停止，给出有保证的停机规则。
- **CALM** [2610.25908]：在解码空间做**一致性感知的 logit 掩码**，当模型多次自洽时抑制继续发散，直接缓解过度思考。

**路线三：长度感知的训练（length-aware training）。** 从源头改变奖励，让「简洁答对」也被奖励。

- 在 RL 奖励中加入长度惩罚 / 对「token–正确率」帕累托前沿做优化（AdaCoT 的训练侧）；
- **long2short**（Kimi k1.5）：先训长 CoT 能力，再蒸馏/压缩成短 CoT，使短模型兼得深度与速度 [2501.12599]；
- **Thought Pruning**：《Direct Pass@k Optimization for Thought Pruning》[2610.31667] 直接优化 pass@k 来做思维剪枝，在质量与多样性间平衡，剪掉无效枝杈。

**路线四：计算分配的上层调度（compute allocation）。** 不针对单条轨迹，而决定「整个预算在采样、修订、验证间怎么切」。

- **ATEST（Adaptive Resource-Aware Test-Time Scaling）** [2610.25519]：根据可用资源（延迟/成本预算）自适应地选择 TTS 策略与规模；
- **Adaptive Replay Refinement** [2610.28294]：把测试时预算在「**修订（revision）与重采样（resampling）**」之间动态分配——卡住时是该在当前思路上改，还是推倒重来；
- 《Where Should a Stuck Agent Spend Its Next Tokens? Causal Attribution of Inference-Time Resources》[2610.28834]：当 Agent 卡住时，用**因果归因**判断下一个 token 花在哪（继续推理 / 调用工具 / 检索）信息增益最大，把「计算分配」做成可归因的决策。

### 8.3 自适应计算的统一视角

可以把上述工作统一在一个决策框架下：把测试时推理视为一个**序贯决策过程**，在每一步选择动作（继续想 / 回溯 / 重采样 / 验证 / 调用工具 / 停止），目标是在给定预算 C 下最大化正确率：

```
最优策略 π*：在每步根据「当前证据强度、剩余预算、验证器质量」
             选择期望信息增益最大的动作，直至停机
```

- Snell 给出了「**按难度**分配」的经验规律 [2408.03314]；
- Sample/Mix/Verify 补充了「**按验证器质量**分配」[2610.23255]；
- AdaCoT/ATEST/E-process 解决「**何时触发、何时停**」[2505.11896][2610.25519][2610.25212]；
- Causal Attribution 解决「**下一步花在哪**」[2610.28834]。

这一框架也解释了工业界的产品选择：**默认快速思考、对难题自动加深、把预算上限暴露给开发者**，正在成为 ChatGPT、Claude、Gemini、Qwen 的共同设计。

### 8.4 思维链的忠实性：另一个「多想真的有用吗」的追问

与过度思考相关但更深的问题是 **CoT 忠实性（faithfulness）**：模型写下的推理步骤，是否真的是它得出答案所依据的计算？如果 CoT 只是事后合理化，那么「让它把想法写出来」未必能真实反映或改进其内部计算。

- 《Reward Misspecification Degrades Chain-of-Thought Monitorability》[2610.28303]：指出**奖励设置不当时，CoT 的可监控性会下降**——模型学会写出好看的推理但并不照做，这与欺骗/伪装问题相关；
- 《Verbalising Model's Computation for Chain-of-Thought Faithfulness Detection》[2610.23275]：研究如何检测模型「说的」和「算的」是否一致；
- 《The Reliance Hypothesis》[2610.22841]：提出一个有趣假说——**语言资源越匮乏，模型对 CoT 的依赖（及忠实性）反而上升**；
- 《Computation, Not Memory: CoT and State Tracking in LLM Agents》[2610.25760]：论证 CoT 对 Agent 的价值在于提供**计算/状态跟踪的外部草稿纸**，而非存储记忆。

忠实性研究提醒工程实践者：**更长的 CoT ≠ 更可靠的推理**，在安全敏感场景中不能把模型自述的推理当作可信证据（这与知识库已有的机制可解释性报告互为表里）。

> ![](diagrams/3-adaptive.html)
> **图 3 · 自适应计算决策框架（触发→分配→停机）**
---

## 9. 潜空间推理：当思考不再用自然语言

显式思维链有两个内在成本：一是**语言 token 很慢**——逐 token 自回归生成，长推理意味着高延迟；二是自然语言未必是推理的最优载体——很多中间状态用连续向量表达更紧凑。2024 年底起，**潜空间推理（latent / continuous reasoning）**成为 TTS 最具想象力的分支。

### 9.1 Coconut：在连续潜空间里推理

Meta 的 **Coconut（Training Large Language Models to Reason in a Continuous Latent Space）** [2412.06769]（2024-12）提出让模型把多步推理编码进**单一连续潜向量**而非一连串语言 token：模型在「潜思考」阶段把后续若干步的信息压缩进一个 hidden state，再在该连续表征上做多步推理，最后才解码成语言。其动机是：潜推理可以**并行地处理多步、避免语言化的串行瓶颈**，并让模型在一个更适合计算的表征空间里搜索。Coconut 用「把上一个 batch 的最后一个隐状态作为下一个 batch 的输入」之类的训练方式，让模型学会在潜空间中保持并推进推理状态。

### 9.2 潜空间 TTS 的 2025–2026 进展

- **Token-Disentangled Latent Test-Time Scaling for Vision-Language Reasoning** [2609.35228]：面向视觉-语言推理，把推理计算从「生成语言 token」中解耦出来、在潜空间扩展，以更低的语言 token 成本获得 TTS 收益；
- **Port-Hamiltonian Latent Deliberation** [2609.37351]：针对潜空间长推理的「**审慎漂移悬崖（deliberation drift cliff）**」——连续状态在多步推演后会漂移、退化——用 Port-Hamiltonian 动力学约束潜状态演化，使其在长程潜推理中保持稳定；
- **ReLaDrift（Recurrent Language Generation via Drifting for Test-Time Scaling）** [2610.23225]：研究循环式语言生成中的状态漂移问题并加以利用/校正；
- **Architectural Sampling** [2610.01687]：不采样不同的文本，而是利用**冻结的视觉-语言模型内部的计算多样性**（不同层/路径）来实现 TTS，提供了一种架构层面的潜空间采样。

### 9.3 循环 Transformer 与自适应深度

潜空间推理与「循环/自适应深度」模型天然合流：

- **Improving Test-Time Scaling with Adaptive Looped Transformers** [2609.35748]：让同一组 Transformer 层根据问题难度**循环执行不同次数**——简单题过一遍就出，难题把层多绕几圈，在共享权重上实现「按难度自适应的推理深度」，是把 TTS 直接做进架构的代表；
- 这条线与知识库已覆盖的**线性注意力/混合架构**（把推理看成状态更新）以及 **Titans: Learning to Memorize at Test Time**（2501.00663，在测试时学习记忆）在思想上相通：都是把「推理时多算」转化为「对隐状态/记忆的迭代更新」。

### 9.4 显式 vs 潜式：一张权衡表

| 维度 | 显式 CoT（语言空间） | 潜空间推理 |
|---|---|---|
| 延迟 | 高（逐 token 串行） | 低（连续计算可并行/循环） |
| 可解释/可监控 | 强，可读、可审计 | 弱，黑盒，需要额外探针 |
| 长程稳定性 | 会重复打转、晚节不保 | 会漂移（需动力学约束） |
| 训练难度 | 相对成熟（RLVR/蒸馏） | 较新，训练范式仍在探索 |
| 安全审计 | 可直接检查推理文本 | 难以审计，对齐风险更高 |

目前工业主力仍是**显式 CoT**（可读、可控、训练成熟），但潜空间推理被普遍视为降低 reasoning 延迟与成本的关键长期方向；一个可能的终态是**混合**——大部分推理在潜空间快速完成，只在关键节点把想法「说出来」以供验证与监控。

> ![](diagrams/4-latent.html)
> **图 4 · 显式 CoT vs 潜空间推理的计算路径**

---

## 10. 测试时学习与自进化：权重在推理中改变

前述方法都假设权重冻结；更激进的一类工作让模型在**推理/测试过程中更新自身参数或记忆**，即 Test-Time Training（TTT）/ Test-Time Learning。它把「适应当前输入分布」也变成一种测试时计算。

### 10.1 从经典 TTT 到 Test-Time Memorization

测试时训练的经典思想（Sun 等 2020 的 TTT）是：在测试样本上构造自监督任务、对模型做几步快速更新，使其适应当前输入。这一思想在大模型时代复兴：

- **Titans: Learning to Memorize at Test Time** [2501.00663]（Google，2024-12/2025-01）：提出在测试时把上下文信息以**可学习的方式写入长期记忆**（带遗忘机制的神经记忆），并在推理时检索，使模型能在测试阶段「记住」新信息，且记忆模块可与注意力混合。它是「测试时学习记忆」方向的代表（注意：该编号常被误传为 SimpleRL-Zoo，本报告已勘正）。
- 与知识库已覆盖的**智能体记忆**（MemGPT/Mem0/Zep/MemOS，外挂记忆）不同，Titans 走的是**权重/参数化记忆内化**路线。

### 10.2 Test-Time Reinforcement Learning：在无标签测试数据上做 RL

2025 年 4 月的 **TTRL: Test-Time Reinforcement Learning** [2504.16084] 把 TTS 与训练彻底打通：在**没有标签的测试数据**上，用多数投票（maj@n）等 TTS 操作本身来估计奖励——如果 N 次采样绝大多数都收敛到同一答案，就把它当作「伪标签」，据此对模型做 RL 更新。Qwen-2.5-Math-7B 在 AIME 2024 上 pass@1 相对提升约 **211%**，且最终性能可以**超过初始模型 maj@n 的上限**——这意味着它不只是「选出好样本」，而是真的从测试分布上学到了东西。TTRL 展示了「测试时计算 → 伪奖励 → 权重更新 → 更强推理」的自举闭环。

### 10.3 持续测试时学习与进化方法

- **Continual Test-Time Learning with Evolution Strategies** [2610.24137]：用**进化策略（无梯度）**在不断变化的测试流上持续适配，规避了对可微目标与大显存反向传播的依赖；
- **Binding-Specific Relearning Savings for Test-Time Continual Learning** [2610.22381]：研究测试时持续学习中「重新学习」的节省效应，刻画模型在测试流上适配的效率；
- **AdaExploreBench** [2610.25950]：构建基准评测 Agent 在**持续演化环境**中的主动探索与测试时适应能力。

### 10.4 测试时学习的风险

测试时改变权重/记忆带来独特风险，工程上必须谨慎：

- **分布漂移与灾难性更新**：少量测试样本上的更新可能损害通用能力；
- **不可复现**：同一模型在不同测试序列上变成不同模型，给评测、审计、版本管理带来挑战；
- **安全投毒面**：攻击者可通过精心构造的输入，在测试时把模型往有害方向「训练」——这是一种新的攻击面；
- **伪标签错误放大**：TTRL 依赖多数投票的正确性，一旦模型系统性地错在同一答案（高置信错误），伪标签会强化错误。

因此当前生产系统多采用**保守的、带遗忘与回滚的**测试时适应（或退而求其次只做外挂记忆），完全开放的测试时权重更新仍主要停留在研究阶段。
---

## 11. 产业落地与推理经济：TTS 如何改变成本结构与产品形态

### 11.1 一张主流推理模型的价格与预算接口表

截至 2026-10，主流厂商均把「思考预算」做成显式 API 参数。下表汇总可核实的定价与控制接口（价格随时间调整，以下为报告撰写时可核实口径）：

| 模型（厂商） | 发布 | 输入 / 输出价格（每百万 token） | 思考预算控制接口 | 思考 token 是否额外计费 |
|---|---|---|---|---|
| o1（OpenAI） | 2024-12 | $15 / $60 | max_completion_tokens | 计入输出 |
| o3-pro（OpenAI） | 2025-06 | $20 / $80【第三方：Artificial Analysis】 | reasoning_effort、max_completion_tokens | 计入输出 |
| Claude 3.7 Sonnet（Anthropic） | 2025-02-24 | $3 / $15【官方】 | thinking.budget_tokens（≤128K） | 与输出同价，不额外加价【官方】 |
| Gemini 2.5 Pro（Google） | 2025-03 | 上线时公布（AI Studio/Vertex） | thinking 预算内置 | 计入输出 |
| Qwen3（阿里，开源） | 2025-04-29 | 自托管 / 云厂商定价 | 思考预算平滑调节、/think 切换【官方】 | — |
| DeepSeek-R1（开源） | 2025-01-22 | 自托管 / 云厂商定价 | 输出长度上限 | — |

关键观察：**思考 token 普遍按普通输出 token 计费**（Anthropic 在 3.7 发布时特别强调 extended thinking 不涨价、与输出同价 $3/$15），这意味着「多想」的成本对开发者是线性可见的——预算控制不是可选项，而是经济性的核心。

### 11.2 关键产业时间线（2024-09 → 2026-10）

- **2024-09-12** OpenAI 发布 o1-preview / o1-mini，TTS 产品化起点；
- **2024-12-05** o1 转正；**2024-12-20** OpenAI 在「12 Days of OpenAI」收官活动展示 o3；
- **2025-01-20/22** DeepSeek-R1 开源（MIT）；**2025-01-22** Kimi k1.5；
- **2025-02-24** Anthropic Claude 3.7 Sonnet，市场首个 hybrid reasoning model + 同发 Claude Code【官方】；
- **2025-03** Google Gemini 2.5 Pro，thinking 能力内置；
- **2025-04-16** o3 正式发布（ARC Prize 注明正式版与其先前测试版不同）；**2025-04-29** Qwen3 开源双模式；
- **2025-06** o3-pro 上线，高计算配置在 ARC-AGI 上换取高分，引发「算力换分」讨论；
- **2025 下半年** 自适应/可控测试时计算成为研究与 API 迭代重点（reasoning on a budget）；
- **2026 全年** TTS 向多模态、语音、具身、视频、潜空间与测试时学习全面扩散，论文密度持续走高。

### 11.3 o3 与 ARC-AGI：一次被详细记录的「算力换分」

ARC Prize（François Chollet）对 o3 的独立复测是理解 TTS 成本/收益最透明的一手案例【第三方】：在 ARC-AGI-1 半私有评测上，o3 以公开榜单 **1 万美元算力上限**配置（6 次采样）取得 **75.7%**；而一个 **172 倍算力**（1024 次采样）的高计算配置取得 **87.5%**。复测同时记录了成本：高计算配置约 3,350 万 token、每任务约 26 美元、每任务约 1.3 分钟（75.7% 档）。作为对照，ARC-AGI-1 从 2020 年 GPT-3 的 0% 走到 2024 年 GPT-4o 的约 5% 用了四年——o3 是一次阶跃，但阶跃的背后是**两个数量级的测试时算力**。

这个案例同时呈现了 TTS 的两面：**能力是真实的阶跃，代价也是真实的算力/延迟/美元**。它也直接推动了 ARC-AGI-2 等更难、更新、抗污染基准的使用。

### 11.4 推理经济：从「训练为王」到「推理为王」的叙事与争议

TTS 的普及正在重塑算力的收支结构，出现了被称为 **inference economy（推理经济）** 的叙事：

- 需求侧，推理模型让**单次请求的 token 消耗成倍增加**（一个难题可能消耗数万 reasoning token），若 AI Agent 大规模铺开，总推理 token 需求可能远超训练需求；
- 成本侧，这与「推理成本持续快速下降」形成剪刀差——更便宜的推理单价 × 暴增的推理 token 量，总量仍可能高速扩张；
- 争议侧，2025 年起出现「**推理泡沫**」的讨论：一部分 benchmark 增益是否只是用巨额、人类不会花费的算力刷出来的？高延迟在真实产品中是否可接受？

需要诚实指出：关于「推理负载占数据中心收入比例」「每日处理 token 量」等宏观数字，多出自厂商自述或第三方估算，口径差异大，本报告不给出未经一手核实的具体百分比。可核实的**微观事实**（o3 的 172×、思考 token 同价计费、s1/AdaCoT 的压缩比）已足以支撑工程判断。

### 11.5 从文本到全模态：TTS 的边界正在扩张

2026 年 TTS 已不再是纯文本/数学的专利，快速向各模态渗透（均有可核实论文）：

- **多模态推理**：TempoMoE 识别并调度 MoE 中的「快/慢思考专家」以提升多模态推理效率 [2610.25999]；多模态 TTS 已有专门综述 [2606.08231]；
- **实时语音**：Heartbeat Thinking 让实时语音 Agent 在「听」的同时投机地进行推理与工具执行，把测试时思考藏进等待时间，缓解语音场景的延迟敏感 [2610.26233]；
- **具身 VLA**：DiVeR 学习决策关键的验证器用于 VLA 的测试时扩展；CLIFT 用保形自验证做 Web Agent 训练与 TTS；
- **视觉生成/视频**：Test-Time Scaling 被用于文生图流匹配模型（噪声激活的序贯蒙特卡洛）与相机条件视频扩散（GeoART）；
- **代码 Agent**：TEX 通过基于执行的交叉验证来测试 Agent 的 TTS [2610.27290]；CARET 做仓库级代码补全的免训练 TTS。

这表明「**在生成答案前多做可验证的计算**」正在成为各类基础模型的通用元方法，而不仅是语言模型的技巧。

> ![](diagrams/5-fullstack.html)
> **图 5 · TTS 全栈与推理经济（训练→推理→验证→服务）**
---

## 12. 开放问题与展望

尽管 TTS 已被广泛采用，一系列根本性问题仍未解决，它们构成未来 1–2 年的研究前沿。

**1. 测试时缩放的极限在哪里？** Snell 观察到收益随难度呈倒 U 形 [2408.03314]，但「在给定模型上、一类任务的计算饱和点」仍缺乏可预测的定量理论。什么任务属于「想再多也没用」（模型缺少必要知识而非搜索不足）？如何在投入算力前就判断一道题处于可挽救区间？这需要把「难度估计」从经验观察推进为先验可测的量。

**2. 能否有一个跨任务、跨模态的通用自适应计算控制器？** 今天的 AdaCoT/ATEST/E-process 多在单一域训练或校准 [2505.11896][2610.25519][2610.25212]。一个理想的「元控制器」应能读取当前证据强度、验证器质量与剩余预算，对任意模态输出停机/加深决策，并随模型升级而不需重训。

**3. 验证器是开放世界的根本瓶颈。** 数学/代码有标准答案，但商业世界的大量任务（开放式分析、长文写作、战略决策）没有 ground truth。生成式验证器（ThinkPRM [2504.16828]）部分缓解，但「谁来验证验证器」的递归问题、以及验证器被生成者策略「取悦」的风险仍在。

**4. 潜空间推理的可解释性与安全。** 潜推理用黑盒连续状态换取速度 [2412.06769]，使当前基于推理文本的监控、审计、欺骗检测全部失效。如何在不把推理完全语言化的前提下保留可监控性，是潜空间路线走向安全敏感场景的前提。

**5. 测试时学习的稳定性、可复现与安全投毒。** 让权重在推理中改变（TTRL [2504.16084]、Titans [2501.00663]）带来漂移、不可复现与新攻击面；需要带遗忘、回滚、沙箱与输入消毒的保守适应机制。

**6. 推理经济与能耗：Jevons 悖论会不会重演？** 推理单价快速下降，但 reasoning token 量与 Agent 普及可能使总算力和能耗反升。TTS 的单位任务碳成本、以及「172× 算力换 12 个百分点」是否值得，将越来越多地进入产品与监管视野。

**7. 评测必须动态化、私有化、抗污染。** o3 在 ARC-AGI 上的阶跃与刷分争议 [第三方 ARC Prize] 说明，静态公开榜单会迅速被针对性优化；ARC-AGI-2/3、私有评测、按「每美元/每秒」归一化的效率指标将比裸准确率更重要。

**8. 训练时与测试时的协同设计尚处早期。** 未来的模型可能在预训练阶段就为「可变推理深度」而设计（自适应循环层 [2609.35748]、快/慢专家 [2610.25999]），把 TTS 从外挂策略变成架构原生能力，并与推理引擎（投机搜索 [2609.39334]、PD 分离）联合优化。

### 工程落地 Checklist（给开发者）

- [ ] **先判断任务可验证性**：有标准答案/可执行测试 → 优先 RLVR 思路与并行采样；开放式任务 → 谨慎设定验证器，别盲信分数。
- [ ] **默认快思考、按难度加深**：用双模式模型，简单请求走 non-thinking，难题才开 thinking，避免全局长 CoT。
- [ ] **显式设置思考预算上限**：`reasoning_effort` / `thinking.budget_tokens` / `max_completion_tokens`，并按 P95 延迟与成本预算反推。
- [ ] **并行宽度按 p 估算**：用历史 pass@1 估 p，按 1−(1−p)^N 选 N，避免 N 越大越好的直觉浪费；配合多数投票或验证器。
- [ ] **监控过度思考信号**：统计平均推理长度、循环打转率、「已对改错」率；必要时上 early stopping / logit 掩码。
- [ ] **不信任 CoT 文本作为安全证据**：安全敏感决策要有独立验证器/执行校验，CoT 仅作草稿。
- [ ] **测试时学习先沙箱**：任何权重/记忆更新都要可回滚、有输入消毒，防止投毒与漂移。
- [ ] **用效率指标选型**：比较模型时同时看准确率、每任务美元成本、P95 延迟，而非只看榜单裸分。
- [ ] **关注多模态/实时场景的延迟隐藏**：语音/实时 Agent 可把思考藏进等待（如 heartbeat thinking）。
- [ ] **数据少而精**：自训推理模型优先投资少量高质量、含探索过程的轨迹（s1/LIMO 路线），而非堆量。

---

## 参考文献（arXiv 论文，编号经 arxiv.org 摘要页逐一核实）

> 核实方式：逐一访问 `arxiv.org/abs/<id>`，核对 `citation_title / citation_author / citation_date` 元数据；下列 65 篇标题、编号、日期均与官方页面一致。作者仅列前若干位。

### A. 思想源流与奠基（2022–2024）

1. [2201.11903] **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, et al. 2022/01/28.
2. [2303.17651] **Self-Refine: Iterative Refinement with Self-Feedback**. Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, et al. 2023/03/30.
3. [2303.11366] **Reflexion: Language Agents with Verbal Reinforcement Learning**. Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao. 2023/03/20.
4. [2305.10601] **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**. Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan. 2023/05/17.
5. [2305.20050] **Let's Verify Step by Step**. Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, et al. (OpenAI). 2023/05/31.
6. [2312.08935] **Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations**. Peiyi Wang, Lei Li, Zhihong Shao, R.X. Xu, Damai Dai, Yifei Li, et al. 2023/12/14.
7. [2401.10020] **Self-Rewarding Language Models**. Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, et al. (Meta AI). 2024/01/18.
8. [2402.14083] **Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping**. Lucas Lehnert, Sainbayar Sukhbaatar, DiJia Su, Qinqing Zheng, Paul McVay, Michael Rabbat, et al. (Meta AI). 2024/02/21.
9. [2403.09629] **Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking**. Eric Zelikman, Georges Harik, Yijia Shao, Noam Shazeer, Noah D. Goodman. 2024/03/14.
10. [2406.03816] **ReST-MCTS\*: LLM Self-Training via Process Reward Guided Tree Search**. Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, Jie Tang. 2024/06/06.
11. [2407.21787] **Large Language Monkeys: Scaling Inference Compute with Repeated Sampling**. Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, et al. 2024/07/31.
12. [2408.03314] **Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters**. Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar (Google DeepMind). 2024/08/06.

### B. o1 时刻、R1 冲击波与开源生态（2024-10 → 2025）

13. [2410.18982] **O1 Replication Journey: A Strategic Progress Report — Part 1**. Yiwei Qin, Xuefeng Li, Haoyang Zou, Yixiu Liu, Shijie Xia, Zhen Huang, et al. 2024/10/08.
14. [2411.14405] **Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions**. Yu Zhao, Huifeng Yin, Bo Zeng, Hao Wang, Tianqi Shi, Chenyang Lyu, et al. (Microsoft). 2024/11/21.
15. [2412.21187] **Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs**. Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, et al. 2024/12/30.
16. [2412.18547] **Token-Budget-Aware LLM Reasoning**. Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, Zhenyu Chen. 2024/12/24.
17. [2412.06769] **Training Large Language Models to Reason in a Continuous Latent Space (Coconut)**. Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, et al. (Meta AI). 2024/12/09.
18. [2501.00663] **Titans: Learning to Memorize at Test Time**. Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google). 2024/12/31.
19. [2501.19393] **s1: Simple Test-Time Scaling**. Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, et al. 2025/01/31.
20. [2501.12948] **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**. DeepSeek-AI (Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, et al.). 2025/01/22.
21. [2501.12599] **Kimi k1.5: Scaling Reinforcement Learning with LLMs**. Kimi Team (Angang Du, Bofei Gao, Bowei Xing, et al., Moonshot AI). 2025/01/22.
22. [2501.04519] **rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking**. Xinyu Guan, Li Lyna Zhang, Yifei Liu, et al. (Microsoft). 2025/01/08.
23. [2502.01456] **Process Reinforcement through Implicit Rewards (PRIME)**. Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Yuchen Zhang, Jiacheng Chen, et al. 2025/02/03.
24. [2502.03387] **LIMO: Less is More for Reasoning**. Yixin Ye, Zhen Huang, Yang Xiao, et al. 2025/02/05.
25. [2502.06703] **Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling**. Runze Liu, et al. 2025/02/10.
26. [2503.18892] **SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild**. Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, et al. 2025/03/24.
27. [2504.16828] **Process Reward Models That Think (ThinkPRM)**. Muhammad Khalifa, Rishabh Agarwal, Lajanugen Logeswaran, Jaekyeom Kim, Hao Peng, Moontae Lee, et al. 2025/04/23.
28. [2504.16084] **TTRL: Test-Time Reinforcement Learning**. Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, et al. 2025/04/22.
29. [2505.22312] **Skywork Open Reasoner 1 Technical Report**. Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, et al. (Kunlun). 2025/05/28.
30. [2505.11896] **AdaCoT: Pareto-Optimal Adaptive Chain-of-Thought Triggering via Reinforcement Learning**. Chenwei Lou, Zewei Sun, Xinnian Liang, Meng Qu, Wei Shen, Wenqi Wang, et al. 2025/05/17.
31. [2501.02497] **A Survey of Test-Time Compute: From Intuitive Inference to Deliberate Reasoning**. Yixin Ji, Juntao Li, Yang Xiang, Hai Ye, Kaixin Wu, Kai Yao, et al. 2025/01/05.
32. [2506.04611] **Revisiting Test-Time Scaling: A Survey and a Diversity-Aware Method for Efficient Reasoning**. 2025/06/05.
33. [2507.02076] **Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs**. 2025/07/02.
34. [2606.08231] **Test-Time Scaling in Multimodal Foundation Models: A Comprehensive Survey of Generation and Reasoning**. 2026/06/06.

### C. 2026 年 9–10 月最新前沿（编号与标题经 abs 页核实；作者/精确日期从略，可经编号直达原文）

35. [2609.33155] **Where Do Test-Time Scaling and Training Fall Short in Individual Stance Prediction?**
36. [2609.35228] **Token-Disentangled Latent Test-Time Scaling for Vision-Language Reasoning**.
37. [2609.35748] **Improving Test-Time Scaling with Adaptive Looped Transformers**.
38. [2609.37351] **Port-Hamiltonian Latent Deliberation: Mitigating the Deliberation Drift Cliff in Test-Time Compute Scaling**.
39. [2609.38672] **Provable Test-Time Scaling for Beam Search in LLM Reasoning**.
40. [2609.39334] **Taming Speculative Search for Test-Time Scaling in LLM Serving**.
41. [2609.39632] **Towards Better Exploration in Sequential Test-Time Scaling**.
42. [2609.40190] **Cheap to Draw, Expensive to Trust: Certifying Test-Time Scaling Curves**.
43. [2610.01687] **Architectural Sampling: Test-Time Scaling via Computational Diversity in Frozen Vision-Language Models**.
44. [2610.04837] **CARET: Training-Free Test-Time Scaling for Repository-Level Code Completion**.
45. [2610.04933] **DiVeR: Decision-Critical Verifier Learning for VLA Test-Time Scaling**.
46. [2610.06829] **CLIFT: Conformal Self-Verification for Web Agent Training and Test-Time Scaling**.
47. [2610.11570] **Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals**.
48. [2610.18583] **Compare to Correct: Pairwise-Guided Logit Adjustment for LLM Test-Time Scaling**.
49. [2610.22381] **Binding-Specific Relearning Savings for Test-Time Continual Learning**.
50. [2610.22841] **The Reliance Hypothesis: Chain-of-Thought Faithfulness Rises as Language Resource Level Falls**.
51. [2610.23275] **Verbalising Model's Computation for Chain-of-Thought Faithfulness Detection**.
52. [2610.25212] **Calibration-Free Early Stopping for Chain-of-Thought Reasoning via E-processes**.
53. [2610.25519] **ATEST: Adaptive Resource-Aware Test-Time Scaling for LLM Inference**.
54. [2610.25760] **Computation, Not Memory: Chain-of-Thought and State Tracking in LLM Agents**.
55. [2610.25908] **CALM: Consistency-Aware Logit Masking in Decoding Space for Mitigating Overthinking of Large Reasoning Models**.
56. [2610.25950] **AdaExploreBench: Benchmarking LLM Agents for Proactive Exploration and Test-time Adaptation in Evolving Environments**.
57. [2610.25999] **TempoMoE: Identifying and Steering Fast-Slow Thinking Experts for Efficient Multimodal Reasoning**.
58. [2610.26233] **Heartbeat Thinking: Speculative Reasoning and Tool Execution While Listening in Realtime Voice Agents**.
59. [2610.27290] **TEX: Test-Time Scaling Testing Agents via Execution-based Cross-Validation**.
60. [2610.28294] **Adaptive Replay Refinement: Allocating Test-Time Compute Between Revision and Resampling**.
61. [2610.28303] **Reward Misspecification Degrades Chain-of-Thought Monitorability**.
62. [2610.28834] **Where Should a Stuck Agent Spend Its Next Tokens? Causal Attribution of Inference-Time Resources**.
63. [2610.23255] **Sample, Mix, or Verify? Allocating Test-Time Compute under Imperfect Verifiers**.
64. [2610.30983] **A Reality Check of Inference-Time Context Augmentation Methods**.
65. [2610.31667] **Direct Pass@$k$ Optimization for Thought Pruning: Balancing Quality and Diversity in Test-Time Reasoning**.

### D. 官方与一手网页信源

- OpenAI. **Learning to reason with LLMs**（o1 发布）. 2024-09-12. https://openai.com/index/learning-to-reason-with-llms/ （站点对脚本访问返回反爬页，事实经多方存档与官方系统卡交叉确认）.
- OpenAI. **OpenAI o1 System Card**. 2024-12. arXiv:2412.16720.
- Anthropic. **Claude 3.7 Sonnet and Claude Code**. 2025-02-24. https://www.anthropic.com/news/claude-3-7-sonnet （2026-10-25 直读，定价 $3/$15、128K 思考上限均出自该页）.
- Google DeepMind. **Gemini 2.5: Our most intelligent thinking model yet**. 2025-03. https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/ （直读；HLE 18.8%、SWE-bench 63.8% 出自该页）.
- Qwen Team. **Qwen3: Think Deeper, Act Faster**. 2025-04-29. https://qwenlm.github.io/blog/qwen3/ （直读；模型规格与思考预算控制出自该页）.
- ARC Prize（François Chollet）. **OpenAI o3 Breakthrough High Score on ARC-AGI-Pub**. 2024-12-20（2025-04-16 更新）. https://arcprize.org/blog/oai-o3-pub-breakthrough （直读；75.7%/87.5%、172×、$26/任务出自该页）.
- DeepSeek-AI. **DeepSeek-R1**（模型与技术报告）. 2025-01-20. https://github.com/deepseek-ai/DeepSeek-R1 .
- Artificial Analysis. **o3-pro / o1 Model Overview**（第三方聚合：o3-pro $20/$80、2025-06 发布；o1 2024-12 发布）. https://artificialanalysis.ai/models （2026-10-25 访问，属第三方口径）.
- Stanford HAI. **Artificial Intelligence Index Report 2025**. https://aiindex.stanford.edu/report/ .

---

> **核实与口径声明**：本报告 65 篇 arXiv 参考文献的编号、标题、日期（A/B 节）均于 2026-10-25 经 `arxiv.org/abs/<id>` 元数据逐一核实，过程中纠正了 SimpleRL-Zoo、Marco-o1、ThinkPRM、Math-Shepherd、ReST-MCTS\*、PRIME、LIMO、Self-Rewarding、Coconut 等多处网传易错编号/作者/归属。厂商自报的模型分数标注「厂商口径」，Artificial Analysis 等聚合站标注「第三方」，OpenAI 站点因反爬未能直读的事实均经官方系统卡或多个独立信源交叉确认。凡缺乏具名一手背书的网传数字（如某些「推理 token 占比」「每日 token 处理量」）一律不予采信。
>
> 报告生成于 2026-10-25，作者为 AI 研究助手；能力边界与产品价格随时间变化，引用时请回到上述一手链接核对最新状态。
