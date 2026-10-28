# 反思式提示进化：当提示成为可学习的程序

**——GEPA 与自动语言程序优化（Language Program Optimization）前沿技术报告**

> **一句话摘要**：2025 年 7 月，UC Berkeley / Stanford / Databricks 团队提出 GEPA——一个完全不碰模型权重、只让 LLM 用自然语言「反思真实执行轨迹」来进化提示的优化器，在六个任务上平均击败需要上万次 rollout 的 GRPO 强化学习（最高 +20%），rollout 用量最多省 35 倍。2026 年下半年，围绕 GEPA 的改进、应用、诊断与替代论文集中爆发，标志着「提示」已从手工调参的模板，正式成为一种**可被机器自动搜索、编译与进化的程序**。本报告系统梳理这一方向的技术谱系、算法机理、实证证据、失败模式与实践方法。

> **关键词**：GEPA（Genetic-Pareto）、反思式提示进化（Reflective Prompt Evolution）、自动提示优化（APO/OPRO/Promptbreeder）、语言程序（Language Program）、DSPy 编译器、TextGrad 文本梯度、MIPROv2、System-aware Merge、Pareto 选择、提示级过拟合（Prompt-level Overfitting）、免梯度优化（Gradient-Free Optimization）、RLVR/GRPO 对比

| 项目 | 内容 |
| --- | --- |
| 报告方向 | 反思式提示进化 / 自动语言程序优化 |
| 核心论文 | GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning（arXiv:2507.19457） |
| 文献规模 | 34 篇，均经 arXiv 摘要页逐一核实（编号 / 标题 / 作者 / 日期） |
| 架构图 | 5 张暗色 SVG（范式坐标 / GEPA 主循环 / Pareto 选择 / 技术谱系 / 2026 生态） |
| 成文日期 | 2026-10-28 |
| 证据约定 | 凡引用具体数字均标注来源；「厂商/作者口径」指数据来自对应论文自述，非第三方复测 |

---

## 目录

1. 引言：大模型适应下游任务的三种范式
2. 问题定义：什么是「语言程序」，什么在被优化
3. 技术谱系：从 APE 到 GEPA 的三代演进（2022–2025）
4. GEPA 深度拆解：反思变异、Pareto 选择与 System-aware Merge
5. 实证证据：六任务基准与跨模型迁移
6. 扩展应用：把优化器当作推理时搜索（kernel 生成）
7. 2026 跟进潮：改进、应用、诊断、替代四条路线
8. 关键张力与失败模式：提示级过拟合、选择噪声与耦合效应
9. 实践指南：什么时候该用、怎么用、如何避坑
10. 展望与范式之争：提示优化会取代 RL 吗
11. 参考文献

---

## 1. 引言：大模型适应下游任务的三种范式

当我们希望一个通用大模型在具体任务（多跳问答、工具调用、写 CUDA kernel、操作 GUI）上表现得更好，工程上存在三条主流路径。它们的区别，本质上是**「在什么空间里、用什么信号、以什么方式做优化」**。

**路径一：参数优化（梯度 / 强化学习）。** 直接修改模型权重 Θ：SFT 用示范轨迹做监督微调，RLHF/RLVR 用偏好或可验证奖励做策略优化。2025 年以来，以 DeepSeek-R1 为代表的 RLVR（Reinforcement Learning with Verifiable Rewards）路线大放异彩，GRPO（Group Relative Policy Optimization，DeepSeekMath，arXiv:2402.03300）成为事实标准算法。它的代价是：需要成百上千、甚至数万次 rollout，需要可训练的权重访问与 GPU 资源，且优化结果锁死在被训练的那一个模型上。

**路径二：测试时计算（冻结权重，多花推理算力）。** 不训练、也不改任何外部状态，而是在推理时让模型多想、多采样、做树/图搜索与自验证（即 Test-Time Scaling，本知识库已有专题）。它提升的是单次任务的求解成功率，但不会沉淀出可复用的任务知识——下一个同类问题仍要重新思考一遍。

**路径三：语言/上下文优化（冻结权重，优化提示本身）。** 模型权重 Θ 完全冻结，把承载任务知识的**自然语言文本**——系统提示、少样本示例、工作流描述、甚至「语言程序」的代码结构——当作可优化的参数 Π，用 LLM 自身的生成与反思能力在离散文本空间里搜索最优解。优化产物是一段人类可读、可审计、可跨模型迁移的文本。

> ![](diagrams/1-paradigms.html)

GEPA 的核心主张，正是路径三在**样本效率**上对路径一的挑战：自然语言是一种比稀疏标量奖励稠密得多、可解释得多的学习介质。当 LLM 能用语言诊断「我上一次为什么失败」时，它往往只需几次 rollout 就能提炼出一条高层规则，而 GRPO 需要成千上万次 rollout 才能把同样的知识「压」进权重（论文原文：*"the interpretable nature of language often provides a much richer learning medium for LLMs, compared to policy gradients derived from sparse, scalar rewards"*，arXiv:2507.19457）。

需要强调的是，路径三并非 GEPA 的凭空发明。早在 2022 年底 APE 就证明 LLM 可以自动写出人类水平的提示；2023 年 OPRO、Promptbreeder、EvoPrompt 把进化算法引入提示空间；2024 年 DSPy 提出「语言程序编译」、TextGrad 提出「文本反向传播」、MIPROv2 解决多模块联合优化。GEPA 的贡献是把**对真实轨迹的自然语言反思**与 **Pareto 多样性选择**结合，把这条路线的样本效率与最终性能推到了新高度，并在 2026 年引发了一波密集跟进。这也是本报告以 GEPA 为主线、但完整覆盖整个语言程序优化谱系的原因。

---

## 2. 问题定义：什么是「语言程序」，什么在被优化

### 2.1 复合 AI 系统与语言程序

现代 LLM 应用很少是「一次 prompt 调用」。一个深度研究 agent 可能包含：查询规划模块、检索模块、证据综合模块、引文校验模块；一个写 kernel 的 agent 包含：代码生成、编译器反馈解析、性能剖析、迭代改写。GEPA 将这类系统形式化为**复合 AI 系统（Compound AI System）Φ**，它由若干模块（module）串接而成，每个模块都带有一个或多个自然语言提示 π。DSPy 则更早地把这种结构称为**语言程序（Language Program）**：用声明式代码定义「文本如何在模块间流动」，而把每个模块里「具体怎么对 LM 说话」留给编译器自动决定（arXiv:2310.03714）。

一个被优化的系统可以写成：

```
Φ = ⟨ Π, Θ_frozen ⟩

Π = {π_1, π_2, ..., π_|M|}   # 各模块的自然语言提示（可优化参数）
Θ                             # 底层 LLM 权重（冻结，不可优化）
```

GEPA 的优化目标，是在**只改动 Π、绝不触碰 Θ** 的前提下，找到使任务指标 μ 最大化的提示集合：

```
Π* = argmax_Π  E_{(x,m) ~ D} [ μ( Φ_Π(x), m ) ]
```

其中 `(x, m)` 是任务实例（输入 x、参考答案/判定 m），μ 是评测指标（如准确率、pass@1、AUROC）。

### 2.2 两个关键函数：评测指标 μ 与反馈函数 μ_f

GEPA 相对早期提示优化器的一个重要设计，是区分了两种信号：

- **评测指标 μ（metric）**：返回一个标量分数，用于候选之间的选择与最终排序（例如答案是否正确、测试是否通过、kernel 向量利用率）。
- **反馈函数 μ_f（feedback function）**：不仅返回分数，还返回**丰富的自然语言反馈**——编译器报错信息、未满足的评分细则（rubric）、工具调用的异常、参考答案与输出的逐条差异。GEPA 论文特别指出，除了 LLM 自己产生的「执行轨迹」（execution trace：推理、工具调用及其输出），环境在计算奖励时产生的「**评测轨迹**」（evaluation trace：如编译器在给出 0 分之前吐出的错误信息）是被前人忽视的第二类高价值诊断信号。

这一区分直接决定了 GEPA 的反思质量：优化器看到的不是冷冰冰的「0 分」，而是「第 3 个工具调用参数类型错误」「结论声称 X 但引用的证据只支持 Y」这样可归因、可行动的具体信息。

### 2.3 数据划分：feedback / pareto / test

为防止「在训练题上把提示调到背答案」，GEPA 采用严格的数据划分（Algorithm 1）：

- **D_feedback**：用于执行、采集轨迹与反馈、做小批量（minibatch）改进的训练集；
- **D_pareto**：一个规模为 n_pareto 的独立验证集，用于在「实例维度」上记录每个候选的得分向量、做 Pareto 选择与最终候选挑选；
- **D_test（held-out）**：优化过程完全不可见的留出测试集，仅在论文汇报时使用。

这种「在一个集合上改、在另一个不相交的集合上选」的纪律，是后文讨论提示级过拟合时的核心防线——而 2026 年的多篇跟进工作表明，这条防线仍会被小样本选择噪声击穿。

---

## 3. 技术谱系：从 APE 到 GEPA 的三代演进（2022–2025）

> ![](diagrams/4-genealogy.html)

自动提示优化并非一条直线，而是多条思想支流的汇合。按「变异信号从哪来、候选如何选择」可粗分为三代。

### 3.1 第一代：把提示当候选，让 LLM 批量提案 + 打分（2022–2023 初）

**APE（Automatic Prompt Engineer，arXiv:2211.01910，2022-11）。** 开山之作。给定少量输入-输出示例，APE 先让 LLM **逆向归纳**（propose 指令：从「示例对」反推出「可能生成这些示例的指令」）批量产生指令候选，再用 zero-shot 表现等打分函数筛选。在 24 个 NLP 任务中，19 个任务上自动指令达到或超过人类标注员手写指令。它确立了这个领域的基本动作：**提案（propose）→ 评估（score）→ 选择（select）**。

**Self-Refine（arXiv:2303.17651，2023-03）与 Reflexion（arXiv:2303.11366，2023-03）。** 两者都引入了「自我反馈」，但作用对象不同：Self-Refine 让同一个 LLM 依次扮演生成器、批评者、改写者，对**单条输出**反复迭代，不需要任何训练集；Reflexion 则面向 agent，把一次完整试错的环境反馈转成**言语反思**存入情节记忆，指导下一轮尝试，HumanEval pass@1 报告达到 91%。Reflexion 的「 verbal reinforcement 」——不更新权重、只把经验写成文字——是 GEPA 反思机制的直接前身。

这一代的局限：候选之间相互独立，没有「种群」与「遗传」，经验无法在多代之间累积；且反馈多为笼统的自我批评，缺少与真实失败案例的绑定。

### 3.2 第二代：种群、进化算法与元提示（2023 下半年）

2023 年 9 月前后，三篇标志性论文把**进化计算**引入提示空间，提示不再是孤立候选，而是一个可繁殖、可变异、可选择的**种群**。

**OPRO（Large Language Models as Optimizers，arXiv:2309.03409，2023-09）。** 让 LLM 直接充当优化器：把「历史解 + 各自得分」组成元提示（meta-prompt），LLM 据此生成更好的新解，打分后追加回元提示循环。在 GSM8K 上最佳提示比人工提示最高 +8%，在 Big-Bench Hard 上最高 +50%。OPRO 的选择信号只有**标量分数**——这正是后来 GEPA/MAGE 证明的关键短板：只看分数、不看「为什么错」，在复杂任务上难以持续改进。

**Promptbreeder（arXiv:2309.16797，2023-09，Google DeepMind）。** 最激进的**自指（self-referential）**机制：种群里不仅进化「任务提示」，还进化「用来变异任务提示的变异提示」本身。变异提示与任务提示共同演化，理论上可以改进「改进方法」本身。它展示了语言空间里自我改进的递归可能，但也因搜索空间巨大、结果方差高而难以稳定复现。

**EvoPrompt（arXiv:2309.08532，2023-09）。** 用 LLM 的语言能力「落地」经典进化算法算子——遗传算法的交叉/变异（GA）与差分进化（DE），完全不用梯度。31 个数据集上显著优于人工提示（BBH 最高 +25%）。

同期还有两条重要支线：

- **InstructZero（arXiv:2306.03082，2023-06）**：针对只能黑盒访问的闭源 LLM，不直接优化离散指令，而是在一个开源 LLM 上优化低维**软提示**，由其解码出指令投喂黑盒模型，再用贝叶斯优化改进软提示。它首次系统处理了「目标模型不可训练、甚至不可见」的约束。
- **APO / ProTeGi（arXiv:2305.03495，2023-05，v2 2023-10）**：Reid Pryzant 等人提出受数值梯度下降启发的「文本梯度」——用小批量数据让 LLM 写出批评性自然语言「梯度」，沿语义反方向编辑提示，并用**束搜索（beam search）+ bandit 选择**展开，初始提示最高提升 31%。这是「文本梯度」一词的来源，比 TextGrad 更早。
- **PromptAgent（arXiv:2310.16427，2023-10）**：把提示优化建模为**战略规划**，用蒙特卡洛树搜索（MCTS）在庞大的专家级提示空间中导航：提示为状态、针对错误的修改为动作、模拟未来奖励，自动产出媲美专家手写的提示。

### 3.3 第三代：语言程序编译、文本反向传播与反思式进化（2024–2025）

第三代的核心转变是：优化对象从「单条提示」升级为「**多模块语言程序**」，并开始系统解决跨模块信用分配、优化器工程化等问题。

**DSPy（arXiv:2310.03714，2023-10）。** 虽与第二代同期，但思想属于新一代：它把 LM 流水线抽象为声明式模块构成的「文本变换图」，再像传统编译器一样，针对给定指标**自动编译**——bootstrap 指令、检索并选择少样本演示——整条流水线。它第一次让「写 LM 应用」从「手写提示字符串」变成「写程序 + 编译器优化」，是整个语言程序优化方向的工程底座。GEPA、MIPROv2 都以 DSPy 的模块抽象为载体。

**TextGrad（arXiv:2406.07496，2024-06）。** 把 LLM 的批评性反馈当作「文本梯度」，沿**文本计算图反向传播**：下游结果被批评 → 批评作为梯度回传给上游文本变量 → LLM 执行「参数更新」。它让提示、解决方案等不同文本变量能在一个统一的反向传播框架里被优化，是 GEPA 的主要对比基线之一。

**MIPRO / MIPROv2（arXiv:2406.11695，2024-06）。** 直面多阶段程序的**跨模块信用分配**：在无模块级标签、无梯度条件下，联合优化每个模块的自由指令与少样本演示；用程序感知/数据感知的方式生成指令提案，用随机小批量评估学习一个代理（surrogate）模型来预测候选表现，并对「如何生成提案」本身做元优化。在 GEPA 出现前，MIPROv2 是最强的开源提示优化器，也是 GEPA 论文中最关键的对照组。

**GEPA（arXiv:2507.19457，2025-07）。** 集大成者。相比 MIPROv2 主要优化「指令 + 演示」、OPRO 只看标量分数，GEPA 的两个决定性设计是：① **反思式变异**——把真实执行轨迹 + 评测轨迹（报错/未满足的 rubric）喂给反思 LM，让它做模块级信用归因后再改写提示；② **Pareto 候选选择**——不总是变异「平均分最高」的候选（易陷局部最优），而是在「实例维度的 Pareto 前沿」上按出现频率采样，保证专门擅长不同子任务的候选都有繁殖机会。此外它还提出 **System-aware Merge** 作为模块级交叉算子。

### 3.4 一张表看清代表性方法

| 方法 | 年份/编号 | 变异信号 | 选择机制 | 是否需训练集 | 优化对象 |
| --- | --- | --- | --- | --- | --- |
| APE | 2022 / 2211.01910 | 从示例逆向归纳提案 | 打分函数排序 | 少量示例 | 单条指令 |
| Self-Refine | 2023 / 2303.17651 | 自我批评 | 迭代至收敛 | 否 | 单条输出 |
| Reflexion | 2023 / 2303.11366 | 环境反馈→言语反思 | 情节记忆 | 环境反馈 | agent 记忆 |
| APO/ProTeGi | 2023 / 2305.03495 | 文本梯度 | 束搜索 + bandit | 是 | 单条指令 |
| InstructZero | 2023 / 2306.03082 | 软提示→贝叶斯优化 | BO 采集函数 | 少量评分 | 黑盒模型指令 |
| OPRO | 2023 / 2309.03409 | 元提示历史轨迹 | 标量分数迭代 | 是 | 单条指令 |
| EvoPrompt | 2023 / 2309.08532 | GA/DE 进化算子 | 种群适应度 | 是 | 离散提示 |
| Promptbreeder | 2023 / 2309.16797 | 自指变异提示 | 训练集适应度 | 是 | 提示+变异提示 |
| PromptAgent | 2023 / 2310.16427 | 错误反思 | MCTS | 是 | 单条指令 |
| DSPy | 2023 / 2310.03714 | 编译器 bootstrap | 指标驱动搜索 | 是 | 整条程序 |
| TextGrad | 2024 / 2406.07496 | 文本梯度反传 | 计算图迭代 | 是 | 文本变量 |
| MIPROv2 | 2024 / 2406.11695 | 数据感知提案 | surrogate + 元优化 | 是 | 指令+演示 |
| **GEPA** | **2025 / 2507.19457** | **轨迹+评测反思** | **Pareto 前沿采样** | **是** | **多模块程序** |

---

## 4. GEPA 深度拆解：反思变异、Pareto 选择与 System-aware Merge

> ![](diagrams/2-gepa-loop.html)

GEPA 的完整算法由论文 Figure 4 中的 Algorithm 1（主循环）与 Algorithm 2（Pareto 选择）给出，附录 D 补充了 Merge 算子（Algorithm 3/4）。下面按主循环的每一步拆解。

### 4.1 主循环（Algorithm 1）

```
输入：系统 Φ、训练集 D_train、指标 μ、反馈函数 μ_f、预算 B
超参：minibatch 大小 b、Pareto 集大小 n_pareto
1. 将 D_train 划分为 D_feedback 与 D_pareto（|D_pareto| = n_pareto）
2. 候选池 P ← [Φ]，父代记录 A ← [None]
3. 在 D_pareto 的每个实例上评估初始 Φ，得到得分向量
4. while 预算 B 未耗尽:
     k ← SelectCandidate(P, S)          # Pareto 采样选父代
     j ← SelectModule(Φ_k)              # round-robin 选要改的模块
     M ← 从 D_feedback 抽 b 条 minibatch
     在 M 上运行，用 μ_f 收集分数、轨迹、模块级文本反馈
     π'_j ← UpdatePrompt(π_j, 反馈, 轨迹)   # 反思 LM 改写
     Φ' ← 复制 Φ_k，仅把模块 j 换成 π'_j
     σ, σ' ← M 上改写前后的平均分
     if σ' 提升:
         把 Φ' 加入 P，记录父代 k
         在整个 D_pareto 上评估 Φ'，更新其得分向量
5. 返回在 D_pareto 上平均分最高的 Φ*
```

三个工程细节值得注意：

1. **两级评估闸门**。新提示先在廉价的小 minibatch（b 条）上与父代做配对比较，只有提升了才值得在整个 D_pareto 上重新评估。这把宝贵的 rollout 集中在「有希望」的候选上，是样本效率的重要来源。
2. **模块轮选（round-robin）**。每次只改一个模块，由 round-robin 策略决定改哪个，避免优化器反复盯着同一个模块。
3. **遗传树的信号继承**。每个候选记录父代，并同时携带「父代的学习信号 + 自己的 rollout」，知识沿遗传树累积——这也是 Merge 能够判断「两个候选各自改了哪个模块」的前提。

### 4.2 反思式变异：UpdatePrompt 在做什么

反思变异是 GEPA 的灵魂。反思 LM 在一次调用中看到：

- **当前提示** π_j；
- **语言程序轨迹**：模块 j 的输入、输出与中间推理，以及整条程序的执行过程；
- **分数与文本反馈**：μ_f 返回的数值分，以及编译器报错、失败 rubric、逐条答案差异等评测轨迹；

它的任务是：① 把成功/失败**归因**到具体提示要素（隐式信用分配）；② 提出修订后的指令。论文附录 C 给出了完整的反思/更新元提示。GEPA 强调，自然语言轨迹让「把错误追溯到具体模块决策」成为可能，从而可以对单个模块做**大而精准**的更新，而不必盲目重写整段提示。

这一机制与 OPRO 的对比是理解 GEPA 的关键：OPRO 的元提示里只有「历史解 + 标量分」，LLM 要凭空猜测「为什么这个分低」；GEPA 则把「为什么」直接以报错和失败案例的形式摆在台面上。2026 年 MAGE 的消融为这一点提供了独立证据：**只依赖分数的 OPRO、只给抽象批评的 Self-Refine 都无法稳定改进提示，唯有「扎根于失败案例的反思（failure-grounded reflection）」不可或缺**（arXiv:2607.11944）。

### 4.3 Pareto 候选选择（Algorithm 2）

> ![](diagrams/3-pareto.html)

如果每一代都只变异「平均分最高」的候选，种群会迅速收敛到一个局部最优，专门擅长某些子任务的「偏科」候选会被过早淘汰。GEPA 的解法是在**实例维度**上做 Pareto 选择：

```
对 D_pareto 中每个实例 i:
    s*[i] ← 所有候选在 i 上的最高分
    P*[i] ← 在 i 上拿到最高分的候选集合
C ← 所有 P*[i] 的并集（即「至少在某一个实例上最好」的候选）
在 C 内部剔除被其他候选全面支配（dominated）者
f[Φ] ← Φ 在多少个实例的 Pareto 集合中出现
以 ∝ f[Φ] 的概率随机采样下一个要变异的候选
```

直觉上：一个候选哪怕平均分不高，只要它在**某些实例上是全场最佳**，它就站上了 Pareto 前沿、获得繁殖机会；而它被选中的概率，取决于它「擅长的实例有多少」。这同时保证了**多样性**（偏科专家不会灭绝）与**择优压力**（擅长越多实例的候选越容易被选中继续进化）。这也是 GEPA 名字「Genetic-Pareto」的由来。

### 4.4 System-aware Merge：模块级交叉（附录 D.1）

变异（mutation）只从单亲产生后代；GEPA 还提供了一个双亲**交叉（crossover）**算子 Merge，专门用于把两个不同候选在**不同模块上的互补改进**拼成一个系统。其设计极具「系统感知」色彩（Algorithm 3/4）：

1. 随机抽两个候选 i、j，若二者有直系亲缘关系则跳过；
2. 找它们的共同祖先 a，沿「祖先 → 两个后代」的差异判断**每个模块分别被谁改过**：
   - 若模块 m 在祖先与 i 中相同、但被 j 改过，则把 j 的版本并入；
   - 反之把 i 的版本并入；
   - 若 i、j 对同一模块做了**互不相同且都不同于祖先**的修改，则取二者中得分更高的版本；
3. 只有当这种模块组合「确实有意义」（Desirable：至少存在一个模块可互补）时才执行，并避免重复尝试同一组合。

换句话说，Merge 不是在字符串层面拼接提示，而是先利用遗传树识别「这两个候选各自在哪个模块上学到了东西」，再在模块粒度上做互补组合。这正是「system-aware」的含义，也使 GEPA 能优化包含多个提示的复合系统，而非单条指令。

但 Merge 也是 GEPA 中最不稳定的部件。原论文已观察到其收益因模型而异；2026 年的机制级诊断（详见第 7、8 章）表明，Merge 的不稳定主要来自**配置敏感**而非算子缺陷，且无法用提示的静态特征预测某次 Merge 是否会被接受。

---

## 5. 实证证据：六任务基准与跨模型迁移

### 5.1 实验设置

论文在六个覆盖不同能力维度的任务上评估：多跳问答 HotpotQA、工具调用基准 IFBench、多跳事实核查 HoVer、工具使用程序理解 PUPA、竞赛数学 AIME-2025、以及 LiveBench-Math。被优化/被执行的模型是开源的 Qwen3 8B 和闭源的 GPT-4.1 Mini，采用统一推理设置；采用标准的 train / validation（D_pareto）/ test（held-out）划分。对照优化器包括 MIPROv2（含只优化指令的 MIPROv2-No-Demos）、TextGrad、Trace（OptoPrime），以及 LoRA 版 GRPO（24,000 rollouts）。

### 5.2 Qwen3 8B：GEPA vs GRPO vs MIPROv2

以下为论文 Table 1（作者实测口径，单位为留出测试集 % 分数）：

| 优化方式 | HotpotQA | IFBench | HoVer | PUPA | AIME | LiveMath | 聚合 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 42.33 | 36.90 | 35.33 | 80.82 | 27.33 | 48.70 | 45.23 |
| GRPO（24k rollouts） | 43.33 | 35.88 | 38.67 | 86.66 | 38.00 | 51.26 | 48.91 |
| MIPROv2 | 55.33 | 36.22 | 47.33 | 81.55 | 20.00 | 46.60 | 47.84 |
| **GEPA** | **62.33** | **38.61** | **52.33** | **91.85** | 32.00 | 51.95 | **54.85** |
| GEPA+Merge | 64.33 | 28.23 | 51.67 | 86.26 | 32.00 | 51.95 | 52.40 |

关键观察：

- **GEPA 聚合 +9.62，是唯一大幅超过 GRPO（+3.68）与 MIPROv2（+2.61）的方法**；在 HotpotQA（+20 vs baseline）、HoVer（+17）、PUPA（+11）上优势明显。
- **样本效率悬殊**。以 IFBench 为例，GEPA 仅用 **678 次 rollout** 就找到 38.61% 的提示，超过 GRPO 用 **24,000 次 rollout** 得到的 35.88%。六个任务上 GEPA（含 Merge）平均总预算约 3,936 rollouts，而 GRPO 固定为 24,000——论文摘要据此报告「最多省 35 倍 rollout」。
- **AIME 是唯一 GRPO 明显占优的任务**（38.00 vs GEPA 32.00）。这提示在需要深层、新颖数学发现的任务上，把知识压进权重的 RL 仍有其优势；提示空间的显式规则未必能替代参数内化。
- **Merge 在 Qwen3 上不稳定**：HotpotQA 提升到 64.33，但 IFBench 暴跌至 28.23，聚合反而低于纯 GEPA。这一观察直接引发了 2026 年的 Merge 诊断研究。

### 5.3 GPT-4.1 Mini：闭源模型开箱即用 + 跨模型迁移

论文 Table 2 显示 GEPA 同样适用于只能 API 访问的闭源模型（留出测试集 %）：

| 优化方式 | HotpotQA | IFBench | HoVer | PUPA | AIME | LiveMath | 聚合 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 38.00 | 47.79 | 46.33 | 78.57 | 49.33 | 58.20 | 53.03 |
| Trace（OptoPrime） | 60.33 | 51.19 | 46.00 | 74.18 | 45.33 | 60.74 | 56.30 |
| MIPROv2-No-Demos | 38.00 | 52.04 | 51.33 | 91.85 | 48.67 | 60.97 | 57.14 |
| MIPROv2 | 58.00 | 49.15 | 48.33 | 83.37 | 51.33 | 61.84 | 58.67 |
| TextGrad | 62.33 | 48.64 | 47.67 | 85.68 | 46.67 | 63.84 | 59.14 |
| **GEPA** | **69.00** | 52.72 | 51.67 | 94.47 | **59.33** | 64.13 | **65.22** |
| GEPA+Merge | 65.67 | 55.95 | 56.67 | 96.46 | 59.33 | 64.13 | 66.36 |
| GEPA-Qwen-Opt（跨模型） | 65.67 | 49.83 | 54.67 | 90.05 | 52.67 | 59.31 | 62.03 |

两个重要结论：

1. **GEPA 聚合 +12.19（Merge 后 +13.33），全面领先 TextGrad（+6.11）、MIPROv2（+5.64）、Trace（+3.27）**。在 GPT-4.1 Mini 上 Merge 反而整体有益（与 Qwen3 相反），再次说明 Merge 收益的模型依赖性。
2. **跨模型迁移成立且效果惊人**。最后一行「GEPA-Qwen-Opt」是**完全用较弱的 Qwen3-8B 优化、再原样搬到 GPT-4.1 Mini 上执行**的提示，仍获得 +9.00 聚合增益，甚至超过所有直接在 GPT-4.1 Mini 上优化的基线（MIPROv2/TextGrad/Trace）。这意味着：可以在便宜的开源模型上搜索提示，再把成果部署到昂贵的闭源模型上——这一发现被 2026 年的跨档成本优化（arXiv:2608.10694）系统化和放大。

### 5.4 一个必须说明的证据边界

上述数字均来自 GEPA 论文同一组作者的自述实验，且多数任务的测试集规模为每基准数百条、报告的是单次或少量运行的点估计。2026 年的独立复现（见第 7、8 章）显示，提示优化的增益**高度依赖任务、模型与随机种子**，验证集增益可能在留出集反转。因此对「GEPA 平均 +6% 击败 GRPO」「最多省 35 倍 rollout」等结论，应理解为**特定实验设置下的作者口径**，而非在所有场景下保证成立的普适定律。

---

## 6. 扩展应用：把优化器当作推理时搜索（kernel 生成）

GEPA 论文第 5 节给出了一个超出「任务适配」范畴的用法：**把优化器本身当作推理时搜索策略**。做法是把待解的一批任务（例如一组待转成 CUDA 的 PyTorch 模块）直接当作训练集传给 GEPA，让 D_train 与 D_pareto 都包含这全部任务，从而允许优化器「过拟合」这批任务、迭代地为每个问题提出更好解，并把从一个任务 rollout 中学到的教训迁移到其他任务。论文在两个硬件平台上做了初步实验：

**NPU kernel（AMD XDNA2，NPUEval 早期版）。** 一个最多迭代 10 次的顺序改写 agent（Sequential10），仅用 GPT-4o 时平均向量利用率只有 4.25%；加入基于技术手册的 RAG 提升到 16.33%；再用 MIPROv2 提升到 19.03%；而**把 GEPA 用于不带 RAG 的 Sequential10，平均利用率跃升到 30.52%，部分 kernel 高达 70%**；由 GEPA 产出的单条提示也能让 Sequential10 达到 26.85%。

**CUDA kernel（KernelBench，arXiv:2502.10517，V100）。** 在 35 个代表性任务、三个难度级别上，GEPA + GPT-4o 把 GPT-4o 接近 0% 的 fast_1 分数随搜索预算提升到 20% 以上（agent 最多顺序改写 5 次）。

这组实验有两个方法论亮点：

- **用失败信息驱动领域知识检索**：μ_f 根据 rollout 的报错（如编译错误）检索技术手册的相关章节，把架构最佳实践按需注入提示进化，而非把整本手册塞进上下文。
- **在缓存下消除生成随机性**：固定 temperature/缓存，使观测到的提升可归因于「提示更新 + 多样化探索」这一推理缩放，而非采样的偶然。

作者也明确这些是「早期结果、有待系统研究」。但它揭示了一个重要可能性：**反思式提示优化与 Test-Time Scaling 并非互斥**——前者可以作为后者的「外层搜索策略」，在一批任务间沉淀和复用显式知识。这与本知识库 Test-Time Scaling 报告中「权重冻结」的谱系形成了自然衔接。

---

## 7. 2026 跟进潮：改进、应用、诊断、替代四条路线

> ![](diagrams/5-ecosystem.html)

GEPA 论文（2025-07，2026-02 更新 v2）之后，尤其在 2026 年下半年，相关工作密集出现。仅 2026 年 10 月一个月，arXiv 上就有近 10 篇标题直接涉及 GEPA / 反思式提示优化的论文。按其与 GEPA 的关系，可分为四条路线。需要再次提醒：以下量化结果除特别注明外，均为**各论文作者在其实验设置下的自述口径**，且不少工作把 GEPA 作为要击败的基线，存在「主场优势」，横向比较时应谨慎。

### 7.1 改进路线：让 GEPA 更强、更便宜、更适配

**GEPA-Curator（arXiv:2610.37651，2026-10-27）。** 在 GEPA 外层加「经验筛选（curation）」：先过滤掉噪声大、学习信号不可靠的任务，并检索该任务过去的成功轨迹作为额外上下文，再进入反思优化。在 GoBrowse、ALFWorld、AppWorld、τ-bench 上（用 Gemini 3.1 Pro、Claude Opus 5 作优化模型），相对 GEPA 最高 +20.7%、相对 ACE 最高 +21.2%。

**Adaptive-GEPA（arXiv:2609.38762，2026-09-30）。** 解决「同一端点收到异质请求」的问题：在同一搜索预算下同时进化一个 **router + 专家程序库**（router 指令、专家描述、程序代码都是可读文本）。在 4 个任务族混合、Qwen3-8B、约 18,000 次 scored calls 下，即使没有族标签也自动演化出 4 个专家，路由在全部 651 个测试请求上与真实任务划分一致；族均分 52.6→70.6，高于 GEPA full-program 的 62.5 与 GRPO 的 54.0。

**Ranking-PE（arXiv:2609.40361，2026-09-30）。** 面向类别极不平衡的多模态临床诊断，把 GEPA 基于二值正确性的分数矩阵，替换为（正例，负例）实例对上的**成对排序**，使列均值由准确率变为经验 AUROC（基于 Wilcoxon-Mann-Whitney 恒等式），并在 Pareto 判定、逐例反馈、最终选择三层统一替换，不增加模型调用。MIMIC 上三种疾病，在微调 Qwen3-VL-8B 上比准确率版配方 +5.8 AUROC 个百分点，在 MedGemma-4B 上 +16.2 pp。消融还指出：医学级视觉骨干是提示搜索无法替代的前提。

**跨档成本优化（arXiv:2608.10694，2026-08-11）。** 把演化优化中 LLM 的三个角色解耦：高体量的 answering（适应度打分）放到最便宜模型档，强模型只用于稀少的 reflection/variation，再把便宜档搜出的提示向上迁移到强目标模型部署。在 4 任务、4 家族 11 个模型上，提示质量匹配或超过同档优化，同时 96% 以上搜索 token 放在最便宜档，搜索成本降低 5.6–14 倍；在推理档每次打分都产生长 CoT 时，降幅达 25–54 倍。

**Batching GEPA（arXiv:2610.36774，2026-10-27）。** GEPA 原作者团队参与的工程改进：把「每步只提一个候选」改为「每步并行提出并评估一批候选」。多数批处理运行的墙钟时间减半甚至更短（最快约 1/4–1/3），且留出分数更高——LiveBench-Math 68.9%→72.1%（2×2），HoVer 49.0%→60.0%（8×1）。论文还给出有限 worker 池下的运行时模型，并首次讨论「固定优化预算如何在各并行轴之间分配」。

**BudgetAPO（arXiv:2610.05671，2026-10-05）。** 面向付费/限流 API 的「紧预算」（仅数百次调用）：先用短探针测出任务噪声以自适应决定评估切片大小，固定切片使每次接受/拒绝成为配对比较，反思算子联合改写推理策略与输出格式。在 7 个基准、5 个 subject model 上均排名第一；250 次调用时仅 13% 的运行退回 seed，而 GEPA 为 86%；在 GPT-OSS-20B 上 GEPA 需约 5 倍调用数才能达到 BudgetAPO 100 次调用的分数。

### 7.2 应用路线：把反思进化用到真实领域

**多模态时序持续学习（arXiv:2610.27468，2026-10-20）。** 把时间序列预测的持续学习表示为一段由结果驱动反思、不断演化的 guidance 文本，挂在冻结的 LLM 预测器提示中，无需重训练。Llama-3.3-70B 携带约 900 词演化 guidance，在留出数据上 GM MASE 0.776，优于 TSFM（TabPFN-TS-3 为 0.898），更新成本是「几美元 vs GPU-day」，并可无损迁移到更新的模型。

**深度研究 agent 的提示优化（arXiv:2610.12601，2026-10-08）。** 把 GEPA 用于多 agent 深度研究系统的两个系统提示，并在训练/验证/测试**主题完全不相交**的划分下评估，专门区分「可迁移增益」与「提示级过拟合」。留出 citation reward 0.702→0.781（+0.079，bootstrap 95% CI [0.051, 0.107]），三种子复现均值 +0.107。其关键因果发现是：**逐维度分解反馈是迁移增益的来源，把反馈压成单一标量奖励会使留出迁移变为负**。

**生产系统的元反思进化 CoCoEvolve（arXiv:2610.12794，2026-10-08）。** 提出「元反思（meta-reflection）」：变异 agent 不只看当前 frontier 候选的失败，而从**整个演化史**检索证据、对在线生产系统主动探测后再提出并验证变更。在 DABStep 公开榜第一（held-out hard accuracy 92.8%，对比 Meta-Harness 78.4%、GEPA 30.4%，同条件）；data-eng-bench solve rate 59.1%→77.4%；HotpotQA 上比 GEPA 高 11 个百分点。

### 7.3 诊断路线：独立复现与机制剖析

**本地模型双模型研究（arXiv:2610.26020，2026-10-19）。** 在两个本地部署的 reasoning-native 配置（DeepSeek V4 Flash、Qwen3.8-27B）上复现 GEPA 与 SkillOpt，结论相当冷静：HotpotQA（每条件 n=500）上两模型均无改善——Qwen 主实验中 GEPA 甚至直接选回 seed、行为与基线完全一致，SkillOpt 的验证增益在留出集反转；但在 MATH-500（n=250）上，GEPA 把 Qwen exact match 0.828→0.904、DeepSeek 0.852→0.908。它据此提出按 content/reasoning、answer-realization/evaluator alignment、serving-completion/reliability 分解「文本空间 headroom」。

**Merge 机制级诊断（arXiv:2610.34027，2026-10-25）。** 针对 GEPA 不稳定的 Merge，在 5 数据集、2 模型、9 种 merge 时机/算子配置下做机制级诊断，证明不稳定性主要来自**配置敏感**而非算子本身缺陷：merge 在 gpt-4.1-mini 上平均有益、在 Qwen3-8B 上平均有害；静态提示特征无法预测 merge 的接受决策。据此给出基于行为的探针 GEPA-MergeProbe（靠对高风险任务「弃权」避免大的 merge 失败），但探针尚不能稳定超过 NoMerge。

**MAGE 与耦合效应（arXiv:2607.11944，2026-07-11）。** 用一个可控消融框架研究多组件提示优化的交互，报告了此前未被记载的「**提示优化耦合效应（POCE）**」：多个随机优化信号在封闭反思环内同时作用时，会在提升性能的同时放大方差，且无法通过孤立分析各组件预测。三大发现：扎根失败的反思不可或缺（OPRO、Self-Refine 均失败）；MAGE 在 GSM8K-Hard 上 46.4% vs GEPA 34.0%（+12.4，5 种子，gpt-4o-mini），方差相当。

**RSEA 留出选择（arXiv:2606.28374，2026-06-17）。** 让自进化 agent 携带三层自然语言状态（策略 / 技能 / playbook），每代从自身轨迹重写三层，且**仅当候选在不相交留出集上不退化时才提交**（严格 keep-better gate）。在 ALFWorld、GAIA、τ-bench、WebShop 上与 ReAct/Reflexion/GEPA/AWM/ACE 等六个基线、同一本地骨干公平对比，核心结论是「**没有任何 artifact 普遍获胜**」：RSEA 在 ALFWorld 最强（单次 69.3% vs ReAct 64.6%，retry 后 79.4%），但在其他任务上各有赢家。

### 7.4 替代路线：质疑「迭代反思循环」本身

**CASD：代码 agent 即提示优化器（arXiv:2609.26261，2026-08-13）。** 无需环境访问与验证数据：给现成 coding agent 一个静态轨迹语料，让它编写并执行分析代码、计算全语料统计、识别系统性失败模式，**一次性**蒸馏出优化提示，论证迭代搜索循环并非必要。在 ALFWorld、τ²-bench retail/telecom、SpreadsheetBench-Verified 上，单次 CASD 在 3/4 上超过 GEPA、全部 4 个上超过 validation-gated SkillOpt；相对基线平均 +16.6 pp（GEPA +10.9、SkillOpt +5.3）；单次离线分析约 1.60 美元，比 validation-gated 搜索便宜 22 倍以上。

**RLPE（arXiv:2610.19062，2026-10-13）。** 用 RL 训练一个 9B 模型专门去分析另一任务模型的轨迹、针对观察到的失败写提示。提示只生成一次、可复用，推理时无需该训练模型，且只需黑盒 query 访问，兼容闭源模型。在 Enigmata、BIG-Bench Extra Hard、PlanBench 上，该 9B 模型在 3 个 benchmark 中的 2 个写出比前沿模型（gpt-5.6-sol、claude-opus-5）和 GEPA/TextGrad 更好的提示，并能泛化到比训练时大 6 倍、跨家族的任务模型；论文还报告其未经显式训练即涌现出迭代改提示的能力。

**FLARE（arXiv:2608.02919，2026-08-03）。** 把先进反思机制与少量 few-shot 参考示例融合，直接挑战 GEPA「反思进化优于 RL 与 few-shot」的论点。在 GPT-5 系列模型上，每个任务-模型对都胜过 GEPA；HotPotQA 最高 +14.2（GPT-5-Chat 52.2 vs GEPA 42.2），tool calling 87.0% vs 81.0%。

### 7.5 跟进潮的整体信号

把四条路线放在一起，可以读出几个明确信号：① GEPA 已成为该方向的**事实基线与对标中心**，几乎所有新工作都以它为主要对照；② 研究焦点正从「能不能优化」转向「**在有限预算/噪声/异质请求/真实生产约束下如何稳健优化**」；③ 独立复现普遍比原报告保守，「没有免费午餐、没有普遍赢家」成为共识；④ 优化的产物正从单条提示扩展到 router、专家库、多层状态、代码分析等更丰富的语言/结构 artifact。

---

## 8. 关键张力与失败模式：提示级过拟合、选择噪声与耦合效应

提示优化看似「零成本涨点」，但 2026 年的跟进工作系统揭示了它的失败模式。理解这些，比记住某个 +X% 更重要。

### 8.1 提示级过拟合（Prompt-level Overfitting）

在训练集上反复改提示，最直接的风险是把提示「调」成只在这些题上有效——要么记住具体答案，要么学到训练集特有的表层模式。GEPA 用独立的 D_pareto 选择、再用 D_test 汇报来防范，但留出集只能挡住「直接记忆训练题」，挡不住更隐蔽的问题。深度研究 agent 的 topic-disjoint 实验（arXiv:2610.12601）给出了细致剖析：获胜提示中近一半可被归为「记忆的答案」，但消融证明这部分即使在原训练题上也不得分（即并非真正的作弊）；而**验证集与留出集之间巨大的差距，主要来自噪声评判 + 小样本下的选择，而非记忆答案**。这是一个反直觉但重要的结论：很多看似「过拟合」的现象，其实是「在噪声中选错了候选」。

### 8.2 小样本选择噪声：在噪声里挑冠军

当评测本身有噪声（LLM 评判不稳定、任务成功率有随机性）、而候选差异又很小时，在几百条样本上选出的「最高分」候选，很可能只是运气好。BudgetAPO 把这一点讲得最直白：紧预算下 GEPA 在 250 次调用时有 86% 的运行最终退回 seed（arXiv:2610.05671）——即搜索过程看似产生了候选，最后却发现没有一个稳健地超过初始提示。多模态时序研究也发现：**环内（in-loop）改进会把真实的环外增益高估 2–3 倍**，必须用一个优化过程从未触碰的独立 monitor 来仲裁（arXiv:2610.27468）。对策在各工作中高度一致：固定评估切片做配对比较、扩大评估样本、用独立留出/monitor 集、报告多种子与置信区间而非单次点估计。

### 8.3 耦合效应与方差放大

MAGE 报告的 POCE 指出：当变异、选择、自适应评估等多个随机信号在一个封闭环内同时运作，它们的交互会在抬升均值的同时放大方差，且这种交互无法从单组件的孤立分析中预测（arXiv:2607.11944）。Promptbreeder 式的自指进化则把这种方差推到极致——搜索空间巨大、结果对种子高度敏感。实践含义是：提示优化系统需要像对待训练一样对待「稳定性」，用多种子评估、严格的 keep-better 门控（RSEA）来抑制方差，而不是只报最好的一次运行。

### 8.4 Merge 的配置敏感与模型依赖

System-aware Merge 在 GPT-4.1 Mini 上平均有益、在 Qwen3-8B 上平均有害（GEPA 原论文 Table 1/2 已现，arXiv:2610.34027 系统证实）。机制诊断表明问题不在算子思想，而在其收益高度依赖 merge 时机、算子配置与被优化模型，且无法用提示的静态特征预测。这意味着「开不开 Merge」不能拍脑袋，需要在目标模型上用留出集实测；在预算紧张或模型较弱时，NoMerge 往往是更安全的默认。

### 8.5 任务/模型依赖：headroom 在哪里

本地双模型研究（arXiv:2610.26020）提供了最清醒的边界：同一个 GEPA，在 MATH-500 上把两模型提升约 6–8 个百分点，在 HotpotQA 上却完全无增益、甚至直接选回 seed。这说明提示优化只在「**文本空间里确实存在可利用的 headroom**」时才有效——即失败主要源于指令/策略表述不清，而非模型本身缺知识或缺能力。若错误源于模型不会（知识/能力缺口）、或评判与目标不一致（evaluator misalignment）、或服务本身补全不可靠，再强的提示搜索也无能为力。RSEA 在四个基准上「没有任何 artifact 普遍获胜」的结论（arXiv:2606.28374）是这一判断的集中体现。

### 8.6 失败模式对照表

| 失败模式 | 典型表现 | 根因 | 主要对策 | 出处 |
| --- | --- | --- | --- | --- |
| 提示级过拟合 | 验证集涨、留出集不涨 | 记忆/表层模式 + 小样本选择 | topic-disjoint、独立留出集 | 2610.12601 |
| 选择噪声 | 多次运行结果不一致、退回 seed | 评判噪声 × 小样本 | 配对比较、加样本、多种子+CI | 2610.05671 |
| 环内增益虚高 | 环内远超环外实测 | 用同一批数据改+评 | 独立 monitor 仲裁 | 2610.27468 |
| 耦合/方差放大 | 均值涨但方差同步涨 | 多随机信号闭环交互 | keep-better 门控、多种子 | 2607.11944 |
| Merge 失效 | 交叉后分数反降 | 配置/模型依赖 | 目标模型实测、默认 NoMerge | 2610.34027 |
| 无 headroom | 优化器选回 seed | 能力缺口/评判错位 | 先诊断错误类型再决定是否优化 | 2610.26020 |

---

## 9. 实践指南：什么时候该用、怎么用、如何避坑

综合 GEPA 原论文与 2026 年的经验证据，给出一份可操作的落地建议。

### 9.1 先判断：该不该上提示优化

在上任何自动优化器之前，先回答三个问题：

1. **错误是「不会说」还是「不会做」？** 抽样看失败案例：若模型其实具备知识、只是没被引导到正确的策略/格式（如没做分步、工具参数填错、没核对引用），提示优化空间大；若模型反复在同类知识点上出错、换什么说法都不会，则应回到 SFT/RL 或换更强模型。
2. **有没有可靠、可规模化的反馈信号？** GEPA 的威力来自 μ_f 的丰富文本反馈。若任务连「对错」都难以自动判定，先解决评测，再谈优化。
3. **是否有合格的留出数据？** 至少需要不相交的 train / validation / test，且评测样本足以压过噪声。数据不足时，提示优化大概率是在拟合噪声。

### 9.2 推荐流程

```
1. 建立可自动评分的评测集（train/val/test 三划分，test 封存）
2. 写一个合理的初始语言程序（模块化：规划/检索/综合/校验分开）
3. 实现丰富的反馈函数 μ_f：不仅给分，还回传报错/未满足 rubric/逐条差异
4. 先用小预算跑 GEPA（minibatch 两级闸门），Merge 默认关闭
5. 每个候选都在固定 val 切片上与父代做配对比较
6. 用多种子运行，只接受在独立 val/monitor 上稳定不退化的候选
7. 在封存 test 上一次性汇报；报告均值、方差与置信区间
8. 产物（提示文本）纳入版本管理，跨模型部署前先小流量验证
```

### 9.3 成本与模型选择

- **在便宜模型上搜索，向强模型迁移**：跨模型迁移已被验证（GEPA-Qwen-Opt，arXiv:2507.19457），跨档成本优化可把搜索成本压低 5.6–14 倍（arXiv:2608.10694）。预算紧张时优先采用。
- **紧预算用配对比较式优化器**：只有数百次 API 调用时，BudgetAPO 类方法显著优于直接跑 GEPA（arXiv:2610.05671）。
- **异质流量考虑 router + 专家库**：同一端点服务多类差异很大的请求时，Adaptive-GEPA 的自动路由比单一共享提示更优（arXiv:2609.38762）。
- **没有验证环境/只有静态日志时**：可考虑 CASD 式一次性代码分析，成本低且在部分 agentic 任务上优于迭代搜索（arXiv:2609.26261）。

### 9.4 避坑清单

- 不要只报单次运行的最好结果；务必多种子 + 置信区间。
- 不要用同一批数据既改提示又评增益；必须留独立仲裁集。
- 不要默认开启 Merge；在目标模型上实测后再决定。
- 不要把验证集增益当成部署增益；上线前用封存 test 与小流量双重确认。
- 不要把作者口径的 +X% 外推到自己的任务；先做小规模 A/B。
- 不要忽视评测本身的偏差：judge 模型与目标不一致时，优化方向可能整体跑偏。

---

## 10. 展望与范式之争：提示优化会取代 RL 吗

GEPA 标题里「Outperform Reinforcement Learning」的措辞，天然带有范式挑战的意味。但综合全部证据，更准确的判断不是「取代」，而是「**分层与融合**」。

**第一，二者优化的知识形态不同，各有擅长。** 提示优化产出的是显式、可读、可迁移的规则，样本效率高、成本低、可审计，特别适合快速任务适配、策略纠偏与跨模型复用；RL 产出的是内隐、压进权重的能力，在需要深层模式内化、新颖数学发现（如 AIME）、以及无法用语言简单概括的技能上仍占优。GEPA 自己在 AIME 上输给 GRPO，已经说明了边界。

**第二，二者正在融合，而非互斥。** 2026 年的工作展示了多种融合形态：RLPE 用 RL 训练一个专门「写提示」的小模型，再用它服务于免梯度优化；GEPA 被用作推理时搜索的外层策略，与 Test-Time Scaling 叠加；CoCoEvolve 用演化史检索 + 主动探测把反思升级为「元反思」。一个自然的趋势是：**用 RL/梯度方法优化「优化器/反思器」，用免梯度的语言进化优化具体任务系统**，两层各司其职。

**第三，「语言即学习介质」的价值会随模型变强而变化。** 一种观点认为，随着模型自身推理与自我纠错能力增强，外部脚手架（包括显式提示）可能逐步内化、甚至「消失」（本知识库 Agent Harness 报告记录了这一范式之争）。但另一种观点指出：只要任务知识需要在不同模型、不同时间、不同团队间低成本迁移与审计，可读的语言 artifact 就有独立价值——它是连接「模型能力」与「工程系统」的稳定接口。2026 年提示优化在生产系统中的持续落地，更支持后一种判断。

**第四，方法论的严谨化是下一阶段的主题。** 从「报一个好看的 +X%」到「topic-disjoint、多种子、置信区间、独立 monitor、公平同骨干对比」，这个领域正在经历一次评测纪律的升级。未来真正值得关注的，不只是又一个更高分的优化器，而是：在噪声、异质、长周期、真实生产约束下，**哪些语言/结构 artifact 能稳定地产生可迁移、可审计的增益**——以及如何把这些方法工程化为人人可用的「LM 程序编译器」。

> **结语**：从手写提示到 APE 的自动提案，从 OPRO 的元提示到 Promptbreeder 的自指进化，从 DSPy 的编译、TextGrad 的文本反传到 GEPA 的反思式 Pareto 进化，这条线索讲述的是同一个故事的深化——**自然语言正在成为一种可编程、可编译、可自动优化的计算介质**。它不否定梯度学习，而是为「如何让机器学会一项任务」提供了一条更轻、更透明、更可迁移的道路。对实践者而言，最重要的不是站队，而是看清每种范式的 headroom 与边界，在正确的问题上使用正确的优化层。

---

## 11. 参考文献

> 以下 34 篇文献均经 arXiv 摘要页（arxiv.org/abs/&lt;编号&gt;）逐一核实编号、标题、作者与日期；点击编号可直达原文。量化数据除注明外均为各论文作者自述口径。

**核心论文**

1. Agrawal, L. A., Tan, S., Soylu, D., et al. **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.** 2025（v2 2026-02）. [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)
2. Zhou, Y., Muresanu, A. I., Han, Z., et al. **Large Language Models Are Human-Level Prompt Engineers (APE).** 2022. [arXiv:2211.01910](https://arxiv.org/abs/2211.01910)
3. Madaan, A., Tandon, N., Gupta, P., et al. **Self-Refine: Iterative Refinement with Self-Feedback.** 2023. [arXiv:2303.17651](https://arxiv.org/abs/2303.17651)
4. Shinn, N., Cassano, F., Berman, E., et al. **Reflexion: Language Agents with Verbal Reinforcement Learning.** 2023. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
5. Pryzant, R., Iter, D., Li, J., et al. **Automatic Prompt Optimization with "Gradient Descent" and Beam Search (ProTeGi).** 2023（v2 2023-10）. [arXiv:2305.03495](https://arxiv.org/abs/2305.03495)
6. Chen, L., Chen, J., Goldstein, T., et al. **InstructZero: Efficient Instruction Optimization for Black-Box Large Language Models.** 2023. [arXiv:2306.03082](https://arxiv.org/abs/2306.03082)
7. Yang, C., Wang, X., Lu, Y., et al. **Large Language Models as Optimizers (OPRO).** 2023. [arXiv:2309.03409](https://arxiv.org/abs/2309.03409)
8. Guo, Q., Wang, R., Guo, J., et al. **EvoPrompt: Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers.** 2023. [arXiv:2309.08532](https://arxiv.org/abs/2309.08532)
9. Fernando, C., Banarse, D., Michalewski, H., et al. **Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution.** 2023. [arXiv:2309.16797](https://arxiv.org/abs/2309.16797)
10. Wang, X., Li, C., Wang, Z., et al. **PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization.** 2023. [arXiv:2310.16427](https://arxiv.org/abs/2310.16427)
11. Khattab, O., Singhvi, A., Maheshwari, P., et al. **DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines.** 2023. [arXiv:2310.03714](https://arxiv.org/abs/2310.03714)
12. Yuksekgonul, M., Bianchi, F., Boen, J., et al. **TextGrad: Automatic "Differentiation" via Text.** 2024. [arXiv:2406.07496](https://arxiv.org/abs/2406.07496)
13. Opsahl-Ong, K., Ryan, M. J., Purtell, J., et al. **Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs (MIPROv2).** 2024. [arXiv:2406.11695](https://arxiv.org/abs/2406.11695)
14. Cheng, C.-A., Nie, A., Swaminathan, A. **Trace is the Next AutoDiff: Generative Optimization with Rich Feedback, Execution Traces, and LLMs.** 2024. [arXiv:2406.16218](https://arxiv.org/abs/2406.16218)

**对比基线与相关系统**

15. Shao, Z., et al. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models（GRPO）.** 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
16. Wang, Z. Z., Mao, J., Fried, D., et al. **Agent Workflow Memory (AWM).** 2024. [arXiv:2409.07429](https://arxiv.org/abs/2409.07429)
17. Yang, L., et al. **Agentic Context Engineering (ACE): Evolving Contexts for Self-Improving Language Models.** 2025. [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)
18. Ouyang, T., et al. **KernelBench: Can LLMs Write Efficient GPU Kernels?** 2025. [arXiv:2502.10517](https://arxiv.org/abs/2502.10517)

**2026 跟进：改进 / 应用 / 诊断 / 替代**

19. Yu, Z., Lee, N., Lee, M., et al. **GEPA-Curator: Curated Learning for Self Improving Agents.** 2026. [arXiv:2610.37651](https://arxiv.org/abs/2610.37651)
20. Zhao, C., Yang, R., Liu, C., et al. **When Does GEPA Merge Help? A Mechanism-Level Diagnostic.** 2026. [arXiv:2610.34027](https://arxiv.org/abs/2610.34027)
21. Mombaerts, L., Taws, J., Gargano, J., et al. **Training-Free Continual Learning for Multimodal Forecasting via Prompt Evolution.** 2026. [arXiv:2610.27468](https://arxiv.org/abs/2610.27468)
22. Giummarra, A., Baglieri, C., Di Maria, A., et al. **Understanding When Reflective Prompt and Skill Optimization Helps on Locally Served Reasoning Models: A Two-Model Study of GEPA and SkillOpt.** 2026. [arXiv:2610.26020](https://arxiv.org/abs/2610.26020)
23. Singhal, R., Maamari, K., Tambwekar, P., et al. **RLPE: Teaching LLMs to Prompt with Reinforcement Learning.** 2026. [arXiv:2610.19062](https://arxiv.org/abs/2610.19062)
24. Yoo, S. W. W., Zeng, C., Somayajula, A., et al. **Meta-Reflective Evolution Effectively Improves Production AI Systems（CoCoEvolve）.** 2026. [arXiv:2610.12794](https://arxiv.org/abs/2610.12794)
25. Lumer, E., Sen, S., Subbiah, V. K., et al. **Reflective Prompt Optimization of a Deep-Research Agent: A Topic-Disjoint, Multi-Seed Protocol for Measuring Transferable Gains and Prompt-Level Overfitting.** 2026. [arXiv:2610.12601](https://arxiv.org/abs/2610.12601)
26. Liu, H., Wang, Z., Yan, H., et al. **How Should a Prompt Optimizer Spend a Tight Budget? BudgetAPO with Noise-Adaptive Evaluation.** 2026. [arXiv:2610.05671](https://arxiv.org/abs/2610.05671)
27. Xia, T., Liu, M., Liang, Y., et al. **Ranking-Aware Prompt Optimization for Multimodal Clinical Diagnosis (Ranking-PE).** 2026. [arXiv:2609.40361](https://arxiv.org/abs/2609.40361)
28. Chen, T., Zhang, Y., Wang, R., et al. **Adaptive-GEPA: Make Your Harness Fit Heterogeneous Requests.** 2026. [arXiv:2609.38762](https://arxiv.org/abs/2609.38762)
29. Singh, A., Gautam, S., Gupta, P., et al. **Coding Agents are Strong Prompt Optimizers (CASD).** 2026. [arXiv:2609.26261](https://arxiv.org/abs/2609.26261)
30. Oved, T., Pony, R., Naparstek, O., et al. **Optimize Cheap, Deploy Strong: Cost-Aware Cross-Tier Transfer for Evolutionary Optimization.** 2026. [arXiv:2608.10694](https://arxiv.org/abs/2608.10694)
31. Sundararaman, D., Gandhi, B., Garg, A., et al. **FLARE: Few-shot Learning-based Adaptive Reflective Engine.** 2026. [arXiv:2608.02919](https://arxiv.org/abs/2608.02919)
32. Singh, P. **MAGE: Understanding Stability-Performance Trade-offs in Multi-component Prompt Optimization.** 2026. [arXiv:2607.11944](https://arxiv.org/abs/2607.11944)
33. Nguyen, M., Nguyen, Q., Vuong, P., et al. **Recursive Self-Evolving Agents via Held-Out Selection (RSEA).** 2026. [arXiv:2606.28374](https://arxiv.org/abs/2606.28374)
34. Zhang, J., Agrawal, L. A., Tan, S., Lee, D., et al. **Batching the Reflective Optimization Loop: Parallel Proposals Make GEPA Faster and Better.** 2026. [arXiv:2610.36774](https://arxiv.org/abs/2610.36774)

---

*本报告基于公开论文与 arXiv 一手页面整理，所有结论以可核实来源为准；如发现编号或数据有误，欢迎指正。*
