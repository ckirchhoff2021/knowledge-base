# 隐式思维链（Latent CoT）前沿技术报告：当大模型学会「在心里想」

> 从 Pause Tokens、Coconut、循环 Transformer 到 HybridAgent——推理正在从自然语言空间撤回连续隐空间

> 关键词：Latent CoT · Implicit CoT · Continuous Thought · Hidden Reasoning · Looped Transformer · Coconut · CODI · Reasoning Compression · Hybrid Reasoning · Silent Thinking · Adaptive Depth

---

## TL;DR

**一句话论点**：2024 年底以来，大模型推理正在发生一场「媒介革命」——中间推理步骤不再必须被写成一个个自然语言 token，而是可以压缩进模型的连续隐状态（hidden state）里完成；2026 年这条路线从学术技巧走向可证明理论、可审计机制、RL 训练配方和开源 Agent 产品，成为与 Test-Time Scaling 正交的独立前沿。

1. **问题**：显式思维链（Explicit CoT）有三重内在成本——逐 token 自回归的**延迟税**、自然语言低带宽的**表达税**、思维全程可读的**泄密税**。R1 类模型动辄数千思考 token，使「会思考」与「用得起」之间的裂缝越来越大。
2. **范式**：隐式思维链（Latent / Implicit CoT）把推理轨迹放进非语言的高维隐空间：每个「思考步」是一个稠密向量或一次跨层循环，而非一串词。权威综述（arXiv:2505.16782，2026-08 v3）将其统一为**两轴分类学**——沿序列维度展开的 token-wise horizontal（Coconut/CODI/SIM-CoT）与沿网络深度循环的 layer-wise vertical（HRM/looped transformer/自适应深度）。
3. **实证收益（作者口径）**：多模态 CoLT 仅用 3 个潜思考步，推理时间降低 **10.1×**、文本解码时间降低 **22.6×** [2606.31986]；开源 HybridAgent 在 11 个基准上平均性能 **+10.4%**、token 减少 **35.5%–54.7%**、推理加速 **1.8–2.6×** [2610.18861]；冻结骨干外挂编码器的 INTACT 把 Llama-3.2-3B 的 GSM8K 从 75.8% 提到 79.3%、端到端延迟降低 19% [2610.19568]。
4. **理论成熟**：2026-05 的 Log-ICoT 首次证明多层 Transformer 能用 poly(n) 样本内化 k-parity、训练阶段数从 O(k) 降到 O(log k)，且推理零开销 [2605.28600]；形式化分离结果表明潜思维支持更高效的并行计算，而显式 CoT 在近似计数/随机采样上不可替代 [2509.25239]——两者不是替代关系，而是能力互补。
5. **2026 信任危机**：最新审计发现，多个 Coconut 式模型在逻辑任务上「答案在潜链开始前就已可读、删掉整条潜链预测几乎不变」[2610.21182]；机制研究则在短程任务中找到了忠实的「桥接状态」、在长程任务中观察到潜链坍缩 [2602.00449]。**「隐状态里发生了计算」不等于「计算真的在推理」**，因果干预（do-intervention）、cache 载体置换测试成为新方法论标配。
6. **终态判断**：纯隐式路线在通用推理上仍稳定低于显式 CoT（小模型基准上差距可达数个点），主流收敛方向是**混合推理（Hybrid Reasoning）**——大部分思考在隐空间快速完成，在工具调用、关键决策、审计节点切换回文本。本报告基于 **119 篇编号、标题、日期均经 arXiv 摘要页 meta 逐一核实的论文**（2023-10 → 2026-11），系统梳理源流、分类学、训练配方、理论边界、机制审计、压缩姊妹路线、混合 Agent 与多模态落地，并给出工程选型清单。

> 热度佐证：腾讯云开发者社区《2026 年 5 月 AI 论文盘点》将「推理内化」列为当月第二大热点（约占重点论文 20%）；Hugging Face Daily Papers 聚合库 2026-10–11 持续出现 latent reasoning 新工作。本报告不依赖二手汇总的任何编号，全部以 arXiv 一手元数据为准（其中二手文给出的 2605.28600、2605.28820 编号经核实均真实存在，但 2605.28820 的准确标题为 *From Pixels to Words — Towards Native One-Vision Models at Scale*，与二手文转述略有出入）。

## 目录

1. 为什么是现在：显式 CoT 的三重税
2. 定义与边界：四个易混术语与五条分界线
3. 思想源流（2023–2026）：从「多给几个 token」到「在隐空间思考」
4. 统一分类学：水平轴 × 垂直轴 × 三种表征初始化
5. 水平路线旗舰拆解：Coconut、CODI、SIM-CoT 与软 token 家族
6. 垂直路线：把深度变成可伸缩的资源
7. 训练范式全谱：预训练内化、SFT、RL、蒸馏与课程
8. 理论：潜思维能做什么、不能做什么
9. 2026 信任危机：潜链真的在推理吗？
10. 姊妹路线：显式 CoT 压缩（仍在语言空间里做减法）
11. 混合推理：2026 年的收敛解——HybridAgent、TAC-CoT、SWITCH
12. 应用版图：多模态、具身、语音、多智能体与推荐
13. 评测体系、开放问题与工程落地 Checklist
14. 结语
- 参考文献

---

## 1. 为什么是现在：显式 CoT 的三重税

### 1.1 延迟税：思考是串行生成出来的

自回归语言模型的推理成本与输出长度近似线性。R1 风格的推理模型把「先想后答」产品化之后，单题生成数百到数千思考 token 成为常态：每一个 token 都要走完整的前向传播、加载全部 KV Cache、等待上一个 token 落盘。在 Agent 场景中问题更尖锐——模型在每次工具调用前后都写一大段内心独白，而真正与环境交互的只有其中少数 action token。2026-11 的一篇实证工作直接扰动教师轨迹中的思考部分后发现：**在多数 Agent 基准上，改写甚至丢弃教师的思考文本，对学生的动作表现几乎没有影响**；只有当「思考中的信息没有在对应动作里披露」时，思考监督才重要 [2611.05709]。这从数据侧质疑了「把每一步思考都写出来」的必要性。

### 1.2 表达税：自然语言不是推理的最优载体

显式 CoT 的第二个约束更本质：语言是为**人际沟通**优化的离散、低带宽信道，而不是为机器内部计算优化的。一句话里大量 token 只负责语法连贯（the、is、then），真正承载推理跃迁的词很少；反过来，许多中间状态（多个候选分支的叠加、连续的置信度、空间几何关系）很难用一串词无损表达。Coconut 的作者一针见血：「大多数词 token 主要保证文本连贯，对推理并非必要；而一些关键 token 需要复杂规划，对 LLM 反而是难题」[2412.06769]。连续向量天然可以表达**叠加态**——一个潜向量同时编码多个候选下一步，这是离散词序列做不到的。

### 1.3 泄密税：可读的思维链既是馈赠也是敞口

显式 CoT 曾被寄予「透明可监督」的厚望：安全团队可以读模型的思考来判断意图。但 2025 年起的一系列工作表明这件事两头落空：一方面，模型并不总是「说出所想」——Anthropic 等对前沿推理模型的测量显示，在使用了提示中线索（reasoning hints）的样本里，CoT 明确披露线索的比例经常低于 20%，结果导向的 RL 初期提升忠实度、之后反而侵蚀它 [2505.05410]；另一方面，完整可读的思考链会通过提示注入、侧信道、日志留存把模型的专有策略与用户隐私直接暴露。隐式推理提供了一个相反的选项：**推理过程默认不可读，只在需要审计的节点显式化**——这是 HybridAgent 把 latent thoughts 设计为「directly inspectable」（通过退出标签按需导出）的直接动机 [2610.18861]。

### 1.4 2026 年的引爆点

三重税在 2024 年还只是学术动机，2026 年变成了工程刚需，三股力量汇合：

- **推理账单成为主战场**：Test-Time Scaling 把「多算」变成能力来源之后，业界迅速撞上算力/延迟墙，「同等准确率下缩短思考」从优化项变成生存项（本知识库《大模型推理优化》《Test-Time Scaling》两报告已详述宏观背景）。
- **方法栈跑通**：2024-12 Meta 的 Coconut [2412.06769] 给出极简范式（末层隐状态直接回灌为下一步输入），2025 年 CODI、CCoT、SIM-CoT、HRM 解决了「怎么训」的问题，2026 年 Switch-GRPO、Log-ICoT、INTACT 解决了「怎么用 RL、怎么证明、怎么不动骨干」的问题。
- **Agent 与多模态提供了杀手场景**：Agent 要低延迟地多轮决策、多模态模型本来就在连续视觉空间里运算，「思考用向量、说话/动作才上 token」的混合形态在这两个场景收益最直接——2026-10 开源的 HybridAgent（togethercomputer，MIT 协议，8B/14B 检查点）是第一个把完整配方公开到可复现程度的代表 [2610.18861]。

> ![](diagrams/1-timeline.html)
> **图 1 · 隐式思维链技术谱系时间线（2023→2026）**：从 Pause Tokens 的「多算几步」直觉，到 Coconut 的连续思想范式，再到 2026 年的可证明理论、机制审计与混合 Agent 产品化。

## 2. 定义与边界：四个易混术语与五条分界线

### 2.1 术语厘清

| 术语 | 典型出处 | 推理载体 | 是否生成可见 token |
|---|---|---|---|
| **Explicit CoT** | Wei et al. 2022 | 自然语言离散 token | 是，完整可读 |
| **Implicit CoT（隐式 CoT）** | Deng et al. 2023 [2311.01460] | 模型隐状态 / 内化映射 | 否，直接出答案 |
| **Latent CoT（潜式 CoT）** | Coconut 2024 [2412.06769]、综述 [2505.16782] | 连续向量序列（潜思考步）或跨层循环 | 否（或仅少量特殊标签） |
| **Reasoning Compression（推理压缩）** | LightThinker 等 [2604.03679] | 仍在语言空间，但把长链摘要/删减成短链或特殊 token | 是，但显著变短 |

需要说明三点：

1. **Implicit 与 Latent 在 2025 年后基本合流**。早期 Implicit CoT 特指「把显式 CoT 蒸馏进直接回答的模型」[2311.01460, 2405.14838]，Latent CoT 特指「在连续空间显式维护推理状态」[2412.06769]；到 2025 年下半年两篇综述 [2507.06203, 2509.02350] 已把二者作为同一领域的两种叫法，本报告统一称「隐式思维链」，引用具体工作时保留其原名。
2. **Latent ≠ 随机噪声**。潜思考向量是确定（或可采样）的稠密计算状态，可以被线性探针、logit lens、辅助解码器读出部分语义；SIM-CoT 等方法甚至在训练时强制每个潜向量能重构出对应的一句推理文本 [2509.20317]。
3. **「压缩」有两个层次**，不要混淆：在**语言空间**里把 1000 个思考 token 压成 200 个（第 10 章），与把推理整体搬进连续空间、用 3–6 个潜向量替代（第 5 章），是技术上可串联、但范式不同的两条路线。

### 2.2 与知识库已有方向的五条分界线

- **vs Test-Time Scaling（已有报告）**：TTS 回答「 inference 时要不要多算、怎么分配算力」；Latent CoT 回答「多算的那部分计算**用什么载体**进行」。二者正交——潜空间里同样可以做顺序扩展与并行采样（SoftCoT++ [2505.11484]、连续空间 Pass@N [2510.12167]、LatentEvolve [2509.24771]）。TTS 报告第 9 章对本方向只有一节概述，本报告是其完整展开。
- **vs 稀疏/线性注意力（已有报告）**：注意力优化改的是「token 之间怎么连」，服务长上下文；Latent CoT 改的是「推理步以什么形态存在」，服务思考效率。一个潜思考序列仍然走标准注意力，二者可叠加。
- **vs 推理优化/KV Cache 压缩（已有报告）**：量化、驱逐、MLA 压缩的是**已生成 token 的缓存**；Latent CoT 直接**不生成**那些 token，从源头减少 KV 与解码步数。
- **vs 上下文工程（已有报告）**：上下文工程管理窗口内已有 token 的写入/选择/压缩；Latent CoT 发生在模型内部表征层，窗口里根本看不到这些「思考」。
- **vs 扩散语言模型（已有报告）**：DLM 在连续/掩码空间做**去噪生成**，目标是输出本身；Latent CoT 用连续状态做**中间推理**、最终答案仍多为自回归文本。两者有交叉（连续潜空间扩散式推理已出现 [2610.19871]），但问题定义不同。

## 3. 思想源流（2023–2026）：从「多给几个 token」到「在隐空间思考」

### 3.1 2023–2024 上半年：先证明「额外计算槽」有用

- **Pause Tokens（2023-10）**[2310.02226]：Google 的奠基性实验。在训练序列里插入可学习的 `<pause>` 占位 token，让模型在回答前先「空转」几步。结果显示 pause 模型在 SQuAD、推理任务上显著优于同参基线——第一次干净地证明：**收益来自额外计算本身，而非额外语言内容**。
- **Thinking Tokens（2024-05）**[2405.08644]：系统研究「沉思 token（contemplation tokens）」的调度，固定长度离散嵌入即可带来增益。
- **Quiet-STaR（2024-03）**[2403.09629]：让模型在每句话后面自生成一段「内心 rationale」再继续，用 REINFORCE 式的事后泛化收益筛选，把通用思考能力蒸馏进预训练模型——是「思考不必说出口」的早期形态。
- **Implicit CoT via KD（2023-11）**[2311.01460]：第一条明确叫「Implicit CoT」的路线：教师生成显式 CoT，学生用一个额外的隐步骤对齐教师的隐状态、直接输出答案。
- **From Explicit CoT to Implicit CoT（2024-05）**[2405.14838]：提出最朴素也最有效的内化配方——先训练显式 CoT 模型，再**逐步删除**中间推理步并微调，GPT-2 Small 在多步任务上保持性能的同时推理大幅简化。这种「消退（fading）」课程成为后来 Log-ICoT 可证明分析的对象。
- **Distilling System 2 into System 1（2024-07）**[2407.06023]：用自监督把 System-2 的慢思考（CoT、重述、System 2 Attention）「编译」进 System-1 的单遍快速回答，建立了「慢推理 → 快推理」的蒸馏叙事。

### 3.2 2024 下半年：连续思想范式确立

- **CCoT（2024-12-17）**[2412.13171]：Compressed Chain-of-Thought 首次提出生成**变长、连续、有内容**的 contemplation token——它们是显式推理的压缩稠密表征，而非固定占位向量，推理延迟显著下降。
- **多模态潜语言建模（2024-12）**[2412.08635]：以 next-token diffusion 在潜空间做多模态语言建模，是连续推理与扩散范式最早的交汇点之一。
- **Coconut（2024-12-09，Meta）**[2412.06769]：范式确立之作。直接把上一步的**末层隐状态**作为下一步的输入嵌入回灌（continuous thought），不经过词表解码。关键洞察是连续思想可以编码多个候选下一步，让模型在规划阶段做**广度优先搜索（BFS）**而非过早 commit 到单条语言路径；在 ProsQA 等需要大量搜索的逻辑任务上超过显式 CoT，但在 GSM8K 这类计算任务上反而落后（GPT-2 设定下 34.1% vs CoT-SFT 41.8%，综述表 1 复现值 [2505.16782]）——这个「擅长探索、拙于执行」的反差在一年半后被理论解释（见第 8 章）。
- **Hidden Chain-of-Thought Decoding（2024-09）**[2409.08561]：用隐式 CoT 解码加速并提升推理，是同期并行的独立尝试。
- **Uncovering Latent CoT Vectors（2024-09）**[2409.14026]：首次在现成模型里用探针「挖出」与推理相关的潜向量，预示了后来的机制审计方向。
- **语音侧先声（2024-09）**[2409.17353]：把 ASR 的 CoT 内化进语音 LLM，端到端语音到语音对话绕过「先转文字」的管线，是隐式推理在工业场景最早的落地形态之一。

### 3.3 2025：方法大爆发与两篇综述收口

2025 年是「怎么训、怎么稳、怎么扩」的方法之年，代表工作包括：CODI 自蒸馏连续思想 [2501.19201 之后的期刊版归属见下]、Heima 隐藏思考 [2501.19201]、TwT 多教师习惯性蒸馏 [2503.24198]、PonderLM/PonderLM-2 预训练阶段连续沉思 [2505.20674, 2509.23184]、Inner Thinking Transformer 动态深度 [2502.13842]、HRM 分层推理 [2506.21734]、DART 把自回归推理蒸馏成非自回归 Silent Thought [2506.11752]、SIM-CoT 步级监督 [2509.20317]、SynAdapt 用合成连续 CoT 学自适应推理 [2508.00574]、Fast Thinking 离散码本 [2509.23633]、Learning to Ponder 自适应潜推理 [2509.24238]、MarCos 全连续马尔可夫思想链 [2509.25020]。两篇综述——*A Survey on Latent Reasoning* [2507.06203] 与 *Implicit Reasoning in LLMs: A Comprehensive Survey* [2509.02350]——以及本报告主要依据的 *Reasoning Beyond Language* [2505.16782]——在年中至年末为领域建立了共同语言。

### 3.4 2026：理论、审计与产品化三线并进

- **理论线**：Log-ICoT 可证明内化 [2605.28600]、潜 CoT 能力与基本极限 [2602.01148]、潜链因果结构 [2602.08783]、潜监督的信息论分析 [2606.20075]。
- **机制审计线**：「潜 CoT 真的在推理吗」的统一分类审计 [2610.21182]、顺序任务机制研究 [2602.00449]、SCIT cache 载体因果测试 [2608.27265]、最小对探针 [2610.28498]、循环模型监控 [2610.18127]、WIDER 潜秩坍缩 [2609.07406]。
- **工程/产品线**：INTACT 冻结骨干外挂 [2610.19568]、SWITCH 可切换潜推理 + on-policy RL [2606.13106]、CoLT 多模态 22.6× [2606.31986]、TAC-CoT 工具增强隐式 CoT [2611.01926]、HybridAgent 开源混合 Agent [2610.18861]，以及向推荐（OneLatent [2610.30675]）、机器人（PhysWord [2610.26854]、LaST-R1 [2604.28192]）、视频 Agent（LARVA [2610.35229]）、统一多模态（[2610.42149]）的全面扩散。

## 4. 统一分类学：水平轴 × 垂直轴 × 三种表征初始化

本节采用 PolyU 综述 [2505.16782] 的两轴框架（该综述由香港理工大学 Xiaoyu Shen 等撰写，2025-05 首发、2026-08 更新至 v3，CC BY-NC-SA 4.0，是目前引用最完整的分类来源）。

> ![](diagrams/2-taxonomy.html)
> **图 2 · 隐式思维链统一分类学**：水平轴（沿序列生成潜思考）× 垂直轴（跨层循环深化）× 三种表征初始化（隐状态 / 词嵌入加权 / 特殊向量）。

### 4.1 水平轴（Token-wise Horizontal）

在序列维度上先生成一串潜思考 c₁, c₂, …, cₖ，再输出答案 token。关键设计分三阶段：

1. **表征初始化（Representation Initialization）**——潜思考从哪来；
2. **模型优化（Model Optimization）**——预训练 / SFT（间接或直接监督）/ RL；
3. **推理探索（Inference Exploration）**——顺序扩展（单链精修、按难度自适应步数）或并行扩展（多潜链采样，类比 self-consistency；并行概率潜推理进一步保证多条轨迹的多样性 [2610.29385]）。

### 4.2 垂直轴（Layer-wise Vertical）

不增加可见序列长度，而是让同一组层在生成每个 token 前**循环施加多次**，把「深度」变成可按 token 难度伸缩的资源：

- **Encoder-based**：以 HRM [2506.21734] 为代表，两层级（tier）结构在循环中做粗到细推理；RELAY 把每个循环迭代对齐到一个 CoT 步，生成可靠的长程推理链再反哺自回归模型训练。
- **Decoder-based（i）端到端自适应深度循环**：Coconut 式循环结构 [2412.06769]、轻量路由器逐 token 控制递归深度（ITT [2502.13842]、Bae et al.）、或把每次循环输出物化成潜「思想 token」改善信息流（Mohtashami et al.）。
- **Decoder-based（ii）循环内的潜语义监督**：用概率加权嵌入构造平滑中间目标（Zeng et al.）、用分类器给出直接「潜奖励」牵引轨迹（Du et al.）。

垂直路线最吸引人的性质是**参数量不变、计算深度可调**：多项分析显示，重复施加同一组 Transformer 层在总参数几乎不变的情况下显著提升多步推理 [2502.17416]，且循环 Transformer 具备一定的并行计算能力，形式上比固有串行的显式 CoT 更高效 [2509.25239]。

### 4.3 三种表征初始化

| 初始化方式 | 机理 | 代表 |
|---|---|---|
| **隐状态直传（Hidden State）** | 上一步末层隐状态直接（或经投影网络）回灌为下一步输入嵌入 | Coconut [2412.06769]、CODI、PCCoT、SIM-CoT、System-1.5、KaVa（KV-cache 蒸馏） |
| **词嵌入加权（Weighted Embedding）** | 在模型输入嵌入空间对现有词表嵌入做加权叠加（soft token / 概念叠加），不离开词表邻域 | Multiplex Thinking [2601.08808]、Mixture of Tokens [2509.21482]、PonderLM [2505.20674]、Latent-SFT [2510.15522] |
| **特殊向量（Special Vector）** | 新增可学习占位序列/潜 token，或用 VQ-VAE、SAE 从外部量化出潜词 | latent tokens [2505.12629]、R-Capsule [2509.22131]、Fast Thinking 码本 [2509.23633]、VQ 潜词、SAE 概念向量 |

三种方式的权衡：隐状态直传信息容量最大但最难训（要跨 batch 回灌、易漂移）；词嵌入加权最贴近预训练分布、训练最稳，但表达受词表凸包约束；特殊向量最灵活、可离散化便于检索与审计，但需要额外词表与对齐成本。

## 5. 水平路线旗舰拆解

### 5.1 Coconut：连续思想与「潜空间 BFS」

**机制**：标准自回归在第 t 步输入 token 嵌入 e_t，输出隐状态 h_t 再解码为词；Coconut 在「潜思考模式」下跳过解码，直接把 h_t 当作 c_t 喂回输入层，连续 k 步后才恢复词解码。**训练**采用多任务课程：从只有 1 个连续思想的简单任务开始，逐步增加潜步数，让模型先学会短链再学长链（直接长链训练不稳定）。**能力画像**极具诊断价值——在 ProsQA（可能世界搜索）上 97.0%，在 GSM8K（精确算术）上仅 34.1% [2602.01148 引述]：连续叠加态天然适合「同时保留多个分支」，却不适合「高精度串行执行」。

### 5.2 CODI：自蒸馏 + 投影对齐

CODI（Continuous Chain-of-Thought via Self-Distillation，2025）针对 Coconut「输出隐状态与输入嵌入空间不对齐」的问题，让上一步隐状态先经过一个**投影网络**再回灌，并通过自蒸馏让潜链对齐显式 CoT 路径的隐状态。综述复现实验中 CODI 把 GPT-2 上的 GSM8K 从 Coconut 的 34.1% 拉到 43.7%（超过显式 CoT-SFT 的 41.8%），在 LLaMA-3-1B-it 上 55.6% [2505.16782 表 1]。这是隐式路线首次在小模型算术基准上追平显式 CoT 的代表结果（注意：仍远小于当代大模型 + RL 的水平）。

### 5.3 SIM-CoT：用步级监督治「潜空间同质化」

2025-09 的 SIM-CoT [2509.20317] 指出隐式 CoT 放大计算预算时的核心病症：潜步数一多，训练就失稳崩溃——潜表征彼此趋同、丧失语义多样性。药方是**训练时挂一个辅助解码器**，强制每个隐式 token 能重构出对应的那一步显式推理文本（逐步对齐）；推理时移除解码器，零额外开销。它同时附赠了可解释性接口。2026-09 的 WIDER [2609.07406] 进一步把这一病症命名为**潜秩坍缩（latent rank collapse）**：连续潜状态坍缩到一个共享主方向；其光谱正则器惩罚轨迹在主方向上的投影，即插即用，机制分析确认有效秩提升、主方向能量下降。2026-10 的 Sep-Latent [2610.22924] 则发现分隔符 token 是天然的压缩锚点，可缓解固定长度潜序列的表征坍缩。

### 5.4 软 token 家族：不离开词表的连续推理

另一条更「温和」的路线不回灌隐状态，而是在词嵌入空间构造软 token：

- **PonderLM** [2505.20674]：生成每个复杂句子成分前，按预测分布对全部词嵌入做加权和，「沉思」若干次再真正出词；PonderLM-2 [2509.23184] 把这套机制推进预训练阶段。
- **Multiplex Thinking** [2601.08808]：每个思考步采样 K 个候选 token，把它们的嵌入聚合成一个「多路复用 token」，保留词表先验与采样动态，在连续载体上维持可处理的概率分布。
- **Mixture of Tokens / MoT** [2509.21482]：在 RLVR 中不再每步只采样离散 token，而是保留整分布信息做组相对优化。
- **Latent-SFT（Chain of Superposition）**[2510.15522]：用 Latent-Vocab 把隐状态约束在预训练词表空间内，把潜推理构造为多条推理路径的叠加，解决分布错位与链定义模糊。
- 机制研究 *LLMs are Single-threaded Reasoners* [2508.03440] 则提醒：软思考内部目前仍近似单线程，一个 token 路径无法分叉成多条独立推理线程——「叠加」更多是加权混合而非真正并行。

### 5.5 代表性方法横评（小模型受控设定，来自综述表 1，作者各自原始口径）

| 方法 | GPT-2 / GSM8K | LLaMA-3-1B-it / GSM8K | 平均可见 token 数 |
|---|---|---|---|
| No-CoT-SFT | 19.1% | 44.1% | 2.2 |
| CoT-SFT | 41.8% | 66.7% | 25 |
| Coconut | 34.1% | 45.3%（复现） | 8.2（6 潜步） |
| CODI | 43.7% | 55.6% | 6 潜步 + 答案 |
| PCCoT | 49.48% | 53.35% | 24 潜步 |
| SIM-CoT (+CODI) | 42.6% | 56.1% | 13.2 |
| System-1.5 | 46.66% | — | #Steps=2（CoT 为 26） |

读表纪律：这是不同作者在不同骨干/数据上的自报值，只说明「隐式路线在小模型上已可与短 CoT 掰手腕」，**不能**外推为「潜推理已普遍超越显式 CoT」——大模型、长程、开放域上显式 + RL 仍是主力。

## 6. 垂直路线：把深度变成可伸缩的资源

### 6.1 Looped Transformer 的理论与实证

*Reasoning with Latent Thoughts: On the Power of Looped Transformers* [2502.17416]（Princeton 等）给出关键经验结论：对加法、p-hop 归纳、数学题，**k 层模型循环 L 次的表现接近 kL 层非循环模型**，且显著优于 k 层模型——推理需要的是大深度，不必是多参数。这与「虚拟深度缩放定律」（Virtual Logic Depth [2506.18233]）相互呼应：推理能力沿有效计算深度缩放；循环语言模型上的规模化实验进一步确认，潜推理收益随循环数增加而稳定增长 [2510.25741]。

### 6.2 HRM：分层、多时标、单前向

Hierarchical Reasoning Model [2506.21734]（2025-06，受大脑分层与时标处理启发）用两层级循环架构在**单次前向传播**内完成顺序推理任务，不生成显式 CoT，同时保持训练稳定与效率，在迷宫等需要长程规划的任务上表现突出。它代表了垂直路线的理想形态：无自回归瓶颈、深度随循环数自由伸缩。

### 6.3 逐 token 自适应深度与预训练内化

- **Inner Thinking Transformer（ITT）**[2502.13842]：把部分层重新设计为可循环的「内部思考」，困难 token（梯度尖峰标记）自动获得更多循环，实现逐 token 动态深度。
- **Skip a Layer or Loop it?**[2507.07996]：推理时对冻结的预训练 LLM 做「跳层或循环」的深度自适应，无需重训。
- **Token-Level Adaptive Latent CoT 预训练**[2602.08220]（2026-02）：把内化推进**预训练阶段**——模型在发射每个 token 前生成可变长度潜 CoT，难 token 长、易 token 短（甚至为零），该行为从一阶段通用文本预训练中自然涌现，并通过逐 token 自适应停机同时降低训练与推理 FLOPs；Llama 架构上困惑度与下游准确率一致改善。
- **Recursive Language Models**[2512.24601]：把长提示当作外部环境，模型程序化地检视、分解并递归调用自身，可处理长两个数量级的输入，是「垂直计算 + 外部记忆」的推理时形态。

### 6.4 冻结骨干外挂：INTACT 的工程友好路线

2026-10 的 INTACT [2610.19568] 提供了对产业最友好的形态：**不动骨干一根毫毛**。一个小型外部编码器读取冻结模型的隐状态，迭代映射出一段潜推理前缀，随后冻结模型自行写出剩余推理与答案；编码器先蒸馏（复现书面推理的效果、同时行为贴近原模型）再用答案正确性做 RL 强化。在 1B–8B 四个冻结骨干上，GSM8K 与跨域平均准确率均提升（Llama-3.2-3B 75.8%→79.3%，Qwen2.5-3B 77.6%→79.7%），端到端每题延迟 −19%；多领域语料训练后在数学、科学、常识 12 个基准上无任务特化即改善。这意味着 latent reasoning 可以作为**旁路插件**卖给已有模型，而非要求从头重训。

## 7. 训练范式全谱

> ![](diagrams/3-training.html)
> **图 3 · 隐式 CoT 训练范式全谱**：预训练内化 → SFT（间接/直接监督）→ RL（GRPO 穿透潜计算）→ 蒸馏（显式教师→隐式学生）→ 课程（显式消退）。

### 7.1 预训练内化

PonderLM-2 [2509.23184] 与 Token-Level Adaptive Latent CoT [2602.08220] 证明潜思考可以在通用语料预训练中习得，而非只能后训练嫁接；LTA-thinker [2509.12875] 在复杂推理上用潜思想增强训练。优势是与通用语言建模目标天然兼容，挑战是潜步数调度与训练稳定性。

### 7.2 SFT：间接监督与直接监督

- **间接监督**：只有最终答案损失，潜轨迹自由生长——简单但信号稀疏、易语义漂移。
- **直接监督**（综述归纳 [2505.16782]）：
  - *重构监督*：辅助解码器把每个潜向量解码回对应推理步/高层计划（SIM-CoT [2509.20317]、CoLT [2606.31986]、R-Capsule [2509.22131]）；
  - *对比损失*：把特殊潜 token 拉近其代表全文的池化表征（Hidden CoT 系列）；
  - *MSE 隐状态匹配*：学生逐步匹配教师末层隐状态（ICoT [2311.01460]、System-1.5）；
  - *稠密 KV-cache 模仿*：学生整条潜 KV-cache 模仿教师压缩版 cache（KaVa）；
  - *潜头预测下一压缩嵌入*：CoLaR 式链式预测。
- **后训练精修**：[2506.08552] 给出轻量潜推理嵌入更新方法，只动推理嵌入、不动全模型。

### 7.3 RL：让梯度穿透循环潜计算

隐式 CoT 长期难以直接套用 on-policy RL——潜状态没有离散 token 边界，策略比（policy ratio）无从定义。2026 年的突破是 **SWITCH**[2606.13106]：引入两个离散边界 token `<swi>` 进入、`</swi>` 退出潜推理，边界使策略比良定义，Switch-GRPO 让梯度穿透循环潜计算；配合「可见→潜」课程，在同规模上持续超越此前隐状态循环方法。机制分析确认：`<swi>` 是学到的局部切换策略而非风格装饰，潜步执行的是问题相关、因果重要的计算，且计算集中在进入潜模式的一次隐状态转移上。RL 线的其他支点：Seek in the Dark 的潜空间实例级策略梯度 [2505.13308]、LaST-R1 把「潜推理后行动」用于机器人 RL 后训练 [2604.28192]、HybridRL 学习「何时潜想、何时文本行动」[2610.18861]。

### 7.4 蒸馏与课程

- **显式→隐式蒸馏**：DART [2506.11752] 用轻量 Reasoning Evolvement Module 把自回归 CoT 蒸馏成非自回归 Silent Thought，推理只走 ST 通路、无额外延迟；TwT [2503.24198] 多教师习惯性蒸馏；LoRi [2606.05315] 发现隐状态轨迹具低秩结构，在共享低秩张量子空间做一二阶矩对齐，蒸馏更省。
- **消退课程（fading）**：从显式 CoT 开始逐步撤掉思考步。朴素做法一次撤一个，需 O(k) 阶段；**Log-ICoT**[2605.28600] 提出按几何块撤并给出证明：L 层 Transformer 在 Log-ICoT 课程下用 poly(n) 样本、L=log₂k 个阶段学会 k-parity，样本效率与显式 CoT 持平而推理零开销（详见第 8 章）。
- **分布蒸馏的新前沿**：Rationale Distribution Distillation via Flow Matching [2610.19871] 用流匹配蒸馏 rationale 的**分布**而非单条轨迹，做分布级连续潜推理。

## 8. 理论：潜思维能做什么、不能做什么

### 8.1 可证明的内化：Log-ICoT

*Transformers Provably Learn to Internalize Chain-of-Thought* [2605.28600]（2026-05）是迄今最强的理论结果。背景：显式 CoT 提示把 parity 学习等任务的样本复杂度从指数降到多项式，但推理时要付出全部思考步成本；ICoT 经验有效却缺乏理论。论文证明：在其提出的 **Log-ICoT 课程**（按几何级数成块删除思考 token，而非逐个删除）下，L 层 Transformer 以 poly(n) 样本、L=log₂k 个训练阶段学会 k-parity，**样本效率匹配显式 CoT、推理开销归零**，并把此前单层的保证推进到多层架构；实验可视化了推理如何被逐层吸收进更深层。

### 8.2 探索—执行权衡：潜 CoT 的能力与基本极限

*Capabilities and Fundamental Limits of Latent CoT* [2602.01148]（2026-02）解释了 Coconut「ProsQA 97.0% vs GSM8K 34.1%」之谜：存在一个**探索—执行权衡（Exploration-Execution Trade-off）**，由「决策确定性（decisional certainty）」支配——高确定性利于精确执行但抑制探索，低确定性利于搜索却导致误差累积。论文提出 Symbolic Index 量化决策承诺度并建立其与执行稳定性/探索能力的因果关系，还证明**课程学习在理论上是必要的**：直接训练会因分布问题而可证明地失败。

### 8.3 显式与潜式的形式化分离

*A Formal Comparison Between CoT and Latent Thought* [2509.25239]（2025-09）证明两者能力不等价：**潜思维支持比固有串行的 CoT 更高效的并行计算**；反过来，**CoT 能通过随机解码做近似计数与采样**，这是确定性潜变换的短板。加上 *Reasoning by Superposition* 对 Coconut 的理论刻画 [2505.12514] 与循环 Transformer 的能力分析 [2502.17416]，理论图景给出明确指引：**不是「潜式取代显式」，而是按任务性质混合**——搜索/规划类交给潜空间并行，精确枚举/随机推断保留显式采样。

## 9. 2026 信任危机：潜链真的在推理吗？

这是本方向 2026 年最重要、也最健康的转折。高准确率 + 探针能解码出中间答案，**并不能**证明模型用潜链推出了答案。

### 9.1 反方证据：潜链可能是「事后装饰」

- **Does Latent CoT Actually Reason?**[2610.21182]（2026-10，UCSD McAuley 组等）：在覆盖 3 种训练配方、3 类推理任务、2 个骨干的 21 个模型-任务配置上，组合「删链 + 内容替换 + 探针 + 轨迹分析」。发现：在逻辑任务上，**Coconut 常常在潜链开始之前就已携带可读答案**，状态过早固化，删掉整条潜链预测几乎不变；逐步监督则能产生有序的推理布局。结论是「latent CoT」这个标签下混着多种计算形态，必须按因果效力而非可读性分类。
- **Do Latent-CoT Models Think Step-by-Step?**[2602.00449]：在中间状态已知的严格顺序多项式迭代任务上解剖 CODI：短程、低跳数时确实形成忠实的「桥接状态」；但随深度/难度增加，模型无法可靠维持完整潜 rollout，要么把计算压成部分的后段捷径，要么在更难区间退化。
- 更早的怀疑论：隐式推理只在远超常规过拟合的 grokking 式训练后出现（Wang et al. 2024）、模型是在生成链的过程中逐步拿到（子）答案而非读题时就算好 [2412.01113]、纯提示诱导不出潜 CoT、必须训练 [2411.15862]、隐式推理常是固定模式训练出的捷径 [2503.07604]、深层并无系统化的高级推理推进 [2505.13898]；另一项对照实验发现，即便插入无语义 token，只要其带来的激活分布发生偏移就能产生推理增益 [2510.01032]——这再次提示「潜链增益」可能主要来自计算/激活形态而非可读内容。

### 9.2 正方与精细化证据

- **Internal Chain-of-Thought**[2505.14530]：在 15 个两步复合任务上用层级上下文掩码、跨任务补丁与 LogitLens，证明预训练 LLM 天然存在「逐层顺序分解并执行子任务」的内部 CoT。
- **Dynamics Within Latent CoT**[2602.08783]：把潜步建模为结构因果模型（SCM）中的变量，逐步 do-干预，区分哪些步因果必要、哪些只是相关。
- **SCIT**[2608.27265]：Suffix Cache Interchange Test 构造精确的「来源—接收者」反事实，置换 cache 段并拆分 K/V 组件，定位反事实计算到底由哪个 Transformer 对象承载——在 CODI-GPT2 与 SIM-CoT 复现版上发现算术反事实主要经特定 cache 成分传递。
- **最小对探针**[2610.28498]：指出传统探针方法的陷阱——逐句读取时探针几乎读不出题目中的数字（Ouro 0.36 / Qwen 0.19），但跨问题共享编码里该数字清晰可读（约 0.9）；「探针失败 ≠ 信息不存在」。
- **循环模型五透镜检查**[2610.18127]：对 8 个循环 LM 做机制可解释性体检，评估迭代变换是否真的增益、对提示扰动是否稳定。
- **SEAL**[2504.07986]：免训练的可转向推理校准，在推理时直接引导隐状态方向即可修正推理偏差，为「不重训也能干预潜推理」提供了轻量证据。
- **监督信息量的定量绑定**：[2606.20075] 提出 Unified Latent Probe 量化潜轨迹与显式推理步之间的互信息，发现清晰的「**信息—性能绑定**」：准确率取决于潜链保留的信息保真度；刚性几何压缩会压垮推理空间，生成式重构是更好的语义锚。

### 9.3 对报告使用者的方法论含义

评估任何 latent reasoning 系统，至少要求三件事：**（1）删链/换链的反事实测试**（链对答案有没有因果贡献）；**（2）跨样本共享编码的探针**（避免单样本探针假阴性）；**（3）随任务难度的阶梯测量**（短程忠实不代表长程不坍缩）。只展示「潜向量能解码出中间答案」的营销材料，在 2026 年的证据标准下已不构成有效论证。

## 10. 姊妹路线：显式 CoT 压缩

在语言空间内做减法是与 latent CoT 并行、产业采用更快的路线，2026 年论文极为密集。它不改变推理载体，但直接砍 token：

- **方法族**：长度感知策略优化（HMPO 中位数长度策略 [2606.01934]、长 CoT 细粒度组策略 [2602.10048]）、经验引导自适应压缩（ExpThink [2605.07501]）、混合策略蒸馏 [2605.08776]、知识引导状态感知压缩 [2604.09150]、规则先验把多步推理压成一步（RuPLaR [2605.09346]）、熵引导选择 [2511.14258, 2607.28707]、token 显著性（Stream of Thought ripple [2608.31066]）、TL;DR 重加权 [2506.02678]、步级偏好优化 SSPO [2508.12604]、局部偏好学习 [2611.04814]、剪枝模型的长短对比压缩性改进 [2610.34336]。
- **记忆化方向**：LightThinker++ [2604.03679] 把中间思考动态压成紧凑语义表征并引入显式自适应内存管理——峰值 token −70%、推理时间 −26%（精度几乎无损），长程 Agent 任务中内存占用保持稳定；ARSM [2609.32852] 用假设-动作-结果微链 + 动态状态机做免训练的 Agent 上下文压缩。
- **可信度审计**：*Shorter, but Still Trustworthy?*[2604.04120] 专门评估压缩后 CoT 的可信度，提醒压缩与忠实度并非免费兼得。
- **多模态压缩**：MCPO 模态对比偏好优化 [2609.04947]、VIG 视觉信息增益奖励 [2608.21883]、Think-Compress-Answer [2610.25640]。

这条路线与 latent CoT 的关系是「同一成本压力的两级响应」：先在语言空间挤掉冗余 token，再把剩余核心计算搬进隐空间；HybridAgent 类系统往往两者都用。

## 11. 混合推理：2026 年的收敛解

> ![](diagrams/4-hybrid.html)
> **图 4 · 混合推理（Hybrid Reasoning）形态**：一次生成内文本模式与潜模式动态切换——潜空间快速思考，工具调用/观测/答案与审计节点回到文本，KV 工作记忆跨轮无损保持。

### 11.1 HybridAgent：第一个完整开源的混合 Agent 配方

*Hybrid Reasoning Agents* [2610.18861]（2026-10-13，Jiaru Zou、James Zou 等，代码在 togethercomputer 组织下，MIT 协议，发布 8B/14B 检查点与训练数据）正面回答了一个问题：**如何构建一个「超越语言思考、却仍通过语言行动」的 Agent？**

- **架构**：轻量 **HybridLink** 适配器在文本与潜模式间无损切换——模型输出 latent 进入标签后，末层隐状态直接回灌推进潜步；检测到退出标签即恢复文本。连续**工作记忆**在多轮之间保留无损的混合上下文（潜状态 + 文本共存于同一条 KV 轨迹）。
- **训练两阶段**：**HybridSFT** 教会骨干与 HybridLink 协调、稳定工具集成的混合生成；**HybridRL** 学习「什么时候在潜空间想、什么时候用文本行动」的策略。
- **结果（作者口径，尚无独立复现）**：3 个模型规模、11 个基准（数学、科学、医学、搜索、Agentic 推理），相对前沿 LLM Agent 与 Agentic 框架（含文本与潜式方法）平均性能 **+10.4%**，token 减少 **35.5%–54.7%**，推理加速 **1.8–2.6×**；论文还报告了更高倍数的端到端效率（摘要中 2.7×–3.7× 一档）。
- **可审计性设计**：潜思考并非永久黑盒——通过退出标签可把任意潜段按需导出检查，这是对「潜推理导致监控失效」批评的直接工程回应。
- **复现门槛**（论文与仓库口径）：SFT 需 8–16 GPU，RL 需 32 GPU；所有基准数字均为作者自报。

### 11.2 TAC-CoT：工具调用只保留文本

*TAC-CoT* [2611.01926]（2026-11-01）把混合思想推到更极致：用紧凑潜推理段替代全部显式推理轨迹，**文本 token 只留给工具调用、观测与最终答案**。在 Agentic 数学与 QA 基准上准确率最高 +8.8 分，模型前向传播次数最多减少 6.8×，并在单工具/多工具场景同步提升工具调用质量。

### 11.3 SWITCH：离散边界让混合可 RL、可解剖

SWITCH [2606.13106]（见 7.3）用 `<swi>`/`</swi>` 两个边界 token 同时解决混合模式的训练与分析问题，是 HybridAgent 之外另一条被验证的技术路线。它的机制发现值得产品设计者注意：**潜模式的有效计算集中在「进入潜模式的那一次隐状态转移」**——这暗示潜段并非越长越好，少数几次高质量转移可能就承载了主要收益。

### 11.4 为什么混合是均衡解

理论（第 8 章）与审计（第 9 章）从两头收敛到同一结论：

1. 纯潜式在精确执行、长程维持、结果可验证上仍有系统性短板；
2. 纯文本式在延迟、成本、信息带宽上代价沉重；
3. Agent 的真实工作流天然分段——**内部规划**（适合潜空间）与**外部交互**（工具协议、人类可读答案、审计证据必须是文本）交替出现；
4. 混合边界 token 同时提供了 RL 的数学良定义性与安全的监控挂钩点。

可以预期，2027 年前后「reasoning effort」旋钮的含义会从「想多长」升级为「**想多少放潜空间、多少说出来**」。

## 12. 应用版图

### 12.1 多模态推理与视觉生成

视觉是 latent reasoning 最自然的主场——图像本就是连续表征，硬把视觉推理翻译成文字再翻译回来既慢又损。

- **CoLT**[2606.31986]：多模态模型用 3 步潜思考替代上千文本 token，外部轻量解码器提供双向步级监督（前向：潜思考解码为下一步文本；后向：给定文本上下文把解码器隐状态对齐到潜思考）+ 内部连贯监督，推理时全部移除。8 个基准上超过 CODI、SIM-CoT 及依赖辅助图像的潜视觉方法；推理时间 −10.1×、文本解码时间 −22.6×。
- **MCOUT**[2508.12587]：多模态连续思想链直接在对齐音/视/文的连续空间推理。
- **Machine Mental Imagery**[2506.17218]：让 VLM 用潜视觉 token 调用「心理意象」，无需真的渲染图像即获得视觉想象力。
- **CrystaL**[2602.20980]：MLLM 中视觉潜变量的自发涌现研究；Latent Visual Reasoning [2509.24251]、Reasoning in the Dark [2510.12603]（交错视觉-文本潜推理）、潜空间视觉推理 [2510.23925]。
- **统一多模态**：[2610.42149] 探索连续空间潜推理作为统一多模态模型的推理底座。
- **视频 Agent**：LARVA [2610.35229] 联合学习潜推理与视觉观测，观察者不再传字幕而是传紧凑连续表征，让小模型同时学会「怎么答」与「下一步看哪」。

### 12.2 具身智能与自动驾驶

- **LaST-R1**[2604.28192]：机器人基础模型的 RL 后训练框架，在动态环境中先做潜推理再行动，突破静态模仿学习的泛化瓶颈。
- **PhysWord**[2610.26854]：用原子动作动词作语义锚组织潜空间中的物理动力学，把潜推理结构化成可复用的「物理词」原语。
- **MIRAGE**[2606.04627]：移动 Agent 从截图与语言目标出发，用连续潜推理替代长文本 CoT，学习屏幕可供性与多步导航。
- **自动驾驶世界模型**：Latent CoT World Modeling [2512.10226] 在动作对齐的潜表征上预测未来状态、优化规划。

### 12.3 语音与全双工对话

语音是最早内化 CoT 的工业场景：[2409.17353] 把 ASR CoT 内化进语音 LLM 实现端到端语音到语音；跨语言半隐式 CoT [2504.20835] 提升非核心语言指令跟随。隐式推理与全双工语音模型（知识库已有专报告）结合，可在不打断流式生成的前提下完成帧级内部计算。

### 12.4 多智能体协作

当 Agent 之间用自然语言通信时，存在三重结构性损耗：推理成本高、离散化信息有损、语言歧义冗余。潜通信路线——*Beyond Tokens* [2606.05711] 给出统一框架，LatentMAS [2511.20639] 实现端到端免训练的纯潜空间协作：各 Agent 先自回归生成潜思想，再在连续空间直接交换，消除语言编解码损耗。共享潜工作记忆提供近似无损的协作带宽。

### 12.5 信息检索与推荐

- 密集检索中用「审慎思考链」生成更丰富的文档表征 [2502.12974]；
- 序列推荐的 Think Before Recommend [2503.22675]、LARES [2505.16865]、强化潜推理推荐 [2505.19092]、潜交叉推理打通搜索与推荐 [2508.04152]、生成式推荐隐式推理 [2606.14142]；
- **OneLatent**[2610.30675]：基础推荐模型的潜推理，摆脱显式 CoT 的模板与长度开销。

### 12.6 结构化数据与在线学习

Latent CoT Improves Structured-Data Transformers [2605.11262] 证明潜推理对表格/结构化数据任务同样有效；Continuous Latent Contexts [2605.09867] 用紧凑连续潜上下文支撑 Transformer 的高效在线学习，呼应测试时学习方向。

## 13. 评测体系、开放问题与工程落地 Checklist

### 13.1 评测三层

1. **行为层**：准确率、token 数、前向次数、端到端延迟（必须与匹配的文本基线在同上下文预算下对比）；
2. **因果层**：删链/换链测试 [2610.21182]、SCM do-干预 [2602.08783]、SCIT cache 置换 [2608.27265]；
3. **信息层**：潜轨迹—显式推理互信息（ULP [2606.20075]）、有效秩/主方向能量 [2609.07406]、跨样本共享编码探针 [2610.28498]、潜段监控 [2610.18127]。

### 13.2 开放问题

- **训练稳定性与规模化**：现有强结果多在 ≤14B、受控任务上；潜链能否在数百 B 规模与通用任务上稳定工作，仍待验证。
- **泛化脆弱性**：潜推理对未见过的问题结构仍易退化 [2602.01148]，捷径学习的阴影未除 [2503.07604]。
- **长程坍缩**：潜秩坍缩、漂移悬崖在长潜链上普遍存在，WIDER 类正则只是缓解；课程学习虽能把隐式推理推向更长视界，但实验显示同样存在难以逾越的能力上限 [2610.25612]。
- **可解释性与安全的张力**：默认不可读利于隐私与防注入，却使常规安全监控失效；HybridAgent 的按需导出、SWITCH 的边界锚点是初步答案，但「审计多少、何时审计」尚无标准。
- **潜式安全攻防**：潜空间里是否存在新型欺骗（不经过文本的隐藏目标）？现有机制可解释性工具能否迁移？这是与知识库《机制可解释性与欺骗检测》方向的交叉前沿。
- **计费与合同**：潜思考也要消耗算力，但用户看不到 token——「谁为思考付费」已进入云合同讨论（相关法律信息学论文 2026-10 出现，本报告仅标注趋势、不引用未核实编号）。

### 13.3 工程落地 Checklist

- [ ] 先做**语言空间 CoT 压缩**（第 10 章），ROI 通常最高、风险最低；
- [ ] 评估内化前，先建立「同上下文预算」的文本基线与删链反事实测试流程；
- [ ] 优先尝试**冻结骨干外挂**（INTACT 式）或**旁路适配器**（HybridLink 式），避免一上来重训骨干；
- [ ] 潜步数从 1–3 步起步、用显式→潜式课程扩展，禁止直接长链训练；
- [ ] 训练时保留辅助解码器/探针做步级监督（SIM-CoT/CoLT 式），推理时再移除；
- [ ] 加潜秩正则（WIDER 式）预防长链坍缩；
- [ ] 混合模式用**离散边界 token**（SWITCH 式），同时拿到 RL 良定义性与审计挂钩点；
- [ ] 工具调用、观测、最终答案保持文本，只把内部规划放入潜空间；
- [ ] KV/工作记忆设计支持潜状态与文本共存、跨轮复用；
- [ ] 建立潜段**按需导出**的审计接口，明确哪些节点必须显式化（合规、高风险动作）；
- [ ] 所有加速/省 token 数字标注「作者口径/内部实测」，上线前做独立复测；
- [ ] 端侧/延迟敏感场景优先（语音、移动 GUI Agent），批量离线场景谨慎（显式 CoT 仍更稳）。

## 14. 结语

隐式思维链的深层意义不止于「省 token」。它代表大模型推理正在找回一个被「语言」遮蔽的事实：**计算不必经过符号序列才能发生**。从 2023 年 Pause Tokens 的朴素直觉，到 Coconut 的连续思想、HRM 的循环深度，再到 2026 年可证明的内化理论、毫不留情的因果审计与开源混合 Agent，这条路线在三年里完成了从技巧到范式的蜕变。

而 2026 年这场健康的信任危机也划出了成熟的边界：潜空间不是魔法，潜链可能只是装饰，纯隐式尚不能独当一面。最可能的未来不是「沉默取代言说」，而是**模型学会在两种媒介间自如切换**——在连续向量里快速搜索，在离散语言里验证、行动、负责。当推理同时拥有「心声」与「言辞」，大模型才真正接近人类「想在心里、说在嘴上」的思考方式。

---

## 参考文献

> 核实声明：以下 **119 篇 arXiv 论文**的编号、标题、日期均于 2026-11-11 经 `arxiv.org/abs/<id>` 摘要页 `citation_title` / `citation_date` meta 逐一核实（实际核验 120 个编号，另含正文勘误脚注中的 arXiv:2605.28820）；0 缺失、0 错号。厂商/作者自报数据均标注「作者口径」。另引用 2 篇网页信源（二手汇总，仅用于热度佐证，不作为任何技术结论依据）。

### A. 奠基与早期（2023–2024 上半年）

1. [2310.02226] **Think before you speak: Training Language Models With Pause Tokens**. Sachin Goyal et al. (Google DeepMind et al.). 2023/10/03.
2. [2311.01460] **Implicit Chain of Thought Reasoning via Knowledge Distillation**. Yuntian Deng et al. (Harvard / CMU). 2023/11/02.
3. [2403.09629] **Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking**. Eric Zelikman et al. (Stanford). 2024/03/14.
4. [2405.08644] **Thinking Tokens for Language Modeling**. 2024/05/14.
5. [2405.14838] **From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step**. 2024/05/23.
6. [2407.06023] **Distilling System 2 into System 1**. 2024/07/08.

### B. 连续思想范式（2024 下半年）

7. [2409.08561] **Expediting and Elevating Large Language Model Reasoning via Hidden Chain-of-Thought Decoding**. 2024/09/13.
8. [2409.14026] **Uncovering Latent Chain of Thought Vectors in Language Models**. 2024/09/21.
9. [2409.17353] **Internalizing ASR with Implicit Chain of Thought for Efficient Speech-to-Speech Conversational LLM**. 2024/09/25.
10. [2411.15862] **Do LLMs Really Think Step-by-step In Implicit Reasoning?** 2024/11/24.
11. [2412.01113] **LLMs Faithfully and Iteratively Compute Answers During CoT: A Systematic Analysis With Multi-step Arithmetics**. 2024/12/02.
12. [2412.06769] **Training Large Language Models to Reason in a Continuous Latent Space (Coconut)**. Shibo Hao et al. (Meta AI). 2024/12/09.
13. [2412.08635] **Multimodal Latent Language Modeling with Next-Token Diffusion**. 2024/12/11.
14. [2412.13171] **Compressed Chain of Thought: Efficient Reasoning Through Dense Representations (CCoT)**. 2024/12/17.

### C. 方法与架构（2025）

15. [2501.19201] **Efficient Reasoning with Hidden Thinking (Heima)**. 2025/01/31.
16. [2502.12974] **Learning Refined Document Representations for Dense Retrieval via Deliberate Thinking**. 2025/02/18.
17. [2502.13842] **Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking (ITT)**. 2025/02/19.
18. [2502.17416] **Reasoning with Latent Thoughts: On the Power of Looped Transformers**. 2025/02/24.
19. [2503.07604] **Implicit Reasoning in Transformers is Reasoning through Shortcuts**. 2025/03/10.
20. [2503.22675] **Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation**. 2025/03/28.
21. [2503.24198] **TwT: Thinking without Tokens by Habitual Reasoning Distillation with Multi-Teachers' Guidance**. 2025/03/31.
22. [2504.07986] **SEAL: Steerable Reasoning Calibration of Large Language Models for Free**. 2025/04/07.
23. [2504.20835] **Enhancing Non-Core Language Instruction-Following in Speech LLMs via Semi-Implicit Cross-Lingual CoT Reasoning**. 2025/04/29.
24. [2505.05410] **Reasoning Models Don't Always Say What They Think**. 2025/05/08.
25. [2505.11484] **SoftCoT++: Test-Time Scaling with Soft Chain-of-Thought Reasoning**. 2025/05/16.
26. [2505.12514] **Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought**. 2025/05/18.
27. [2505.12629] **Enhancing Latent Computation in Transformers with Latent Tokens**. 2025/05/19.
28. [2505.13308] **Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space**. 2025/05/19.
29. [2505.13898] **Do Language Models Use Their Depth Efficiently?** 2025/05/20.
30. [2505.14530] **Internal Chain-of-Thought: Empirical Evidence for Layer-wise Subtask Scheduling in LLMs**. 2025/05/20.
31. [2505.16782] **Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning**. Xinghao Chen, Anhao Zhao, Heming Xia et al. (Hong Kong Polytechnic University). 2025/05/22（v3: 2026/08/31）.
32. [2505.16865] **LARES: Latent Reasoning for Sequential Recommendation**. 2025/05/22.
33. [2505.19092] **Reinforced Latent Reasoning for LLM-based Recommendation**. 2025/05/25.
34. [2505.20674] **PonderLM: Pretraining Language Models to Ponder in Continuous Space**. 2025/05/27.
35. [2506.02678] **TL;DR: Too Long, Do Re-weighting for Efficient LLM Reasoning Compression**. 2025/06/03.
36. [2506.08552] **Efficient Post-Training Refinement of Latent Reasoning in Large Language Models**. 2025/06/10.
37. [2506.11752] **DART: Distilling Autoregressive Reasoning to Silent Thought**. 2025/06/13.
38. [2506.17218] **Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens**. 2025/06/20.
39. [2506.18233] **Beyond Parameters: Exploring Virtual Logic Depth for Scaling Laws**. 2025/06/23.
40. [2506.21734] **Hierarchical Reasoning Model (HRM)**. 2025/06/26.
41. [2507.06203] **A Survey on Latent Reasoning**. 2025/07/08.
42. [2507.07996] **Skip a Layer or Loop it? Test-Time Depth Adaptation of Pretrained LLMs**. 2025/07/10.
43. [2508.00574] **SynAdapt: Learning Adaptive Reasoning in Large Language Models via Synthetic Continuous Chain-of-Thought**. 2025/08/01.
44. [2508.03440] **LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking**. 2025/08/05.
45. [2508.04152] **Bridging Search and Recommendation through Latent Cross Reasoning**. 2025/08/06.
46. [2508.12587] **Multimodal Chain of Continuous Thought for Latent-Space Reasoning in Vision-Language Models (MCOUT)**. 2025/08/18.
47. [2508.12604] **SSPO: Self-traced Step-wise Preference Optimization for Process Supervision and Reasoning Compression**. 2025/08/18.
48. [2509.02350] **Implicit Reasoning in Large Language Models: A Comprehensive Survey**. 2025/09/02.
49. [2509.12875] **LTA-thinker: Latent Thought-Augmented Training Framework for Large Language Models on Complex Reasoning**. 2025/09/16.
50. [2509.20317] **SIM-CoT: Supervised Implicit Chain-of-Thought**. 2025/09/24.
51. [2509.21482] **Learning to Reason with Mixture of Tokens**. 2025/09/25.
52. [2509.25239] **A Formal Comparison Between Chain of Thought and Latent Thought**. 2025/09/25.
53. [2509.22131] **R-Capsule: Compressing High-Level Plans for Efficient Large Language Model Reasoning**. 2025/09/26.
54. [2509.23184] **PonderLM-2: Pretraining LLM with Latent Thoughts in Continuous Space**. 2025/09/27.
55. [2509.23633] **Fast Thinking for Large Language Models (Latent Codebooks)**. 2025/09/28.
56. [2509.24238] **Learning to Ponder: Adaptive Reasoning in Latent Space**. 2025/09/29.
57. [2509.24251] **Latent Visual Reasoning**. 2025/09/29.
58. [2509.24771] **LatentEvolve: Self-Evolving Test-Time Scaling in Latent Space**. 2025/09/29.
59. [2509.25020] **Deep Thinking by Markov Chain of Continuous Thoughts (MarCos)**. 2025/09/29.
60. [2510.01032] **Meaningless Tokens, Meaningful Gains: How Activation Shifts Enhance LLM Reasoning**. 2025/10/01.
61. [2510.12167] **Towards Inference-time Scaling for Continuous Space Reasoning**. 2025/10/14.
62. [2510.12603] **Reasoning in the Dark: Interleaved Vision-Text Reasoning in Latent Space**. 2025/10/14.
63. [2510.15522] **LLM Latent Reasoning as Chain of Superposition (Latent-SFT)**. 2025/10/17.
64. [2510.23925] **Latent Chain-of-Thought for Visual Reasoning**. 2025/10/27.
65. [2510.25741] **Scaling Latent Reasoning via Looped Language Models**. 2025/10/29.
66. [2511.14258] **Entropy-Guided Reasoning Compression**. 2025/11/18.
67. [2511.20639] **Latent Collaboration in Multi-Agent Systems (LatentMAS)**. 2025/11/25.
68. [2512.10226] **Latent Chain-of-Thought World Modeling for End-to-End Driving**. 2025/12/11.
69. [2512.24601] **Recursive Language Models**. 2025/12/31.

### D. 2026 前沿：理论、审计、训练与压缩

70. [2601.08808] **Multiplex Thinking: Reasoning via Token-wise Branch-and-Merge**. 2026/01/13.
71. [2602.00449] **Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks (CODI)**. 2026/01/31.
72. [2602.01148] **Capabilities and Fundamental Limits of Latent Chain-of-Thought**. 2026/02/01.
73. [2602.08220] **Pretraining with Token-Level Adaptive Latent Chain-of-Thought**. 2026/02/09.
74. [2602.08783] **Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure**. 2026/02/09.
75. [2602.10048] **Long Chain-of-Thought Compression via Fine-Grained Group Policy Optimization**. 2026/02/10.
76. [2602.20980] **CrystaL: Spontaneous Emergence of Visual Latents in MLLMs**. 2026/02/24.
77. [2604.03679] **LightThinker++: From Reasoning Compression to Memory Management**. 2026/04/04.
78. [2604.04120] **Shorter, but Still Trustworthy? An Empirical Study of Chain-of-Thought Compression**. 2026/04/05.
79. [2604.09150] **Think Less, Know More: State-Aware Reasoning Compression with Knowledge Guidance for Efficient Reasoning**. 2026/04/10.
80. [2604.28192] **LaST-R1: Reinforcing Robotic Manipulation via Adaptive Physical Latent Reasoning**. 2026/04/30.
81. [2605.07501] **ExpThink: Experience-Guided Reinforcement Learning for Adaptive Chain-of-Thought Compression**. 2026/05/08.
82. [2605.08776] **Reasoning Compression with Mixed-Policy Distillation**. 2026/05/09.
83. [2605.09346] **RuPLaR: Efficient Latent Compression of LLM Reasoning Chains with Rule-Based Priors — From Multi-Step to One-Step**. 2026/05/10.
84. [2605.09867] **Continuous Latent Contexts Enable Efficient Online Learning in Transformers**. 2026/05/11.
85. [2605.11262] **Latent Chain-of-Thought Improves Structured-Data Transformers**. 2026/05/11.
86. [2605.28600] **Transformers Provably Learn to Internalize Chain-of-Thought (Log-ICoT)**. 2026/05/27.
87. [2606.01934] **HMPO: Hybrid Median-length Policy Optimization for Chain-of-Thought Compression**. 2026/06/01.
88. [2606.04627] **MIRAGE: Mobile Agents with Implicit Reasoning and Generative World Models**. 2026/06/03.
89. [2606.05315] **LoRi: Low-Rank Distillation for Implicit Reasoning**. 2026/06/03.
90. [2606.05711] **Beyond Tokens: A Unified Framework for Latent Communication in LLM-based Multi-Agent Systems**. 2026/06/04.
91. [2606.13106] **Demystifying Hidden-State Recurrence: Switchable Latent Reasoning with On-Policy Reinforcement Learning (SWITCH)**. 2026/06/11.
92. [2606.14142] **Implicit Reasoning for Large Language Model-based Generative Recommendation**. 2026/06/12.
93. [2606.20075] **What Makes Effective Supervision in Latent Chain-of-Thought: An Information-Theoretic Analysis (ULP)**. 2026/06/18.
94. [2606.31986] **CoLT: Teaching Multi-Modal Models to Think with Chain of Latent Thoughts**. 2026/06/30.
95. [2607.28707] **Demystifying Entropy-based Selection for Chain-of-Thought Compression in Large Reasoning Models**. 2026/07/30.
96. [2608.21883] **VIG: Visual Information Gain as a Reward Signal for Multimodal Chain-of-Thought Compression**. 2026/08/22.
97. [2608.27265] **SCIT: Testing Causal Cache Carriers in Latent Chain-of-Thought Models**. 2026/08/27.
98. [2608.31066] **Every Token Leaves a Ripple in the Stream of Thought: Eliciting Model-Internal Token Saliency for Chain-of-Thought Compression**. 2026/08/31.
99. [2609.04947] **MCPO: Modality-Contrastive Preference Optimization for Multimodal Chain-of-Thought Compression**. 2026/09/04.
100. [2609.07406] **Think Wider: Mitigating Latent Rank Collapse in Implicit Chain-of-Thought Reasoning (WIDER)**. 2026/09/07.
101. [2609.32852] **ARSM: Auto-Regressive State Machine for Agentic Reasoning Compression**. 2026/09/26.
102. [2610.18127] **Monitoring Latent Reasoning in Looped Language Models**. 2026/10/13.
103. [2610.18861] **Hybrid Reasoning Agents (HybridAgent / HybridLink / HybridSFT / HybridRL)**. Jiaru Zou, Yunzhe Qi, Sirui Chen, Federico Bianchi, Shang Zhu, Pan Lu, Jingrui He, James Zou. 2026/10/13.
104. [2610.19568] **INTACT: Inducing Latent Reasoning on a Frozen Backbone**. 2026/10/14.
105. [2610.19871] **Rationale Distribution Distillation via Flow Matching for Distributional Continuous Latent Reasoning**. 2026/10/14.
106. [2610.21182] **Does Latent Chain-of-Thought Actually Reason? A Single, Unifying Taxonomy for Latent Reasoning**. Jingjie Qiu et al. (UC San Diego et al.). 2026/10/15.
107. [2610.22924] **Sep-Latent: Separator Tokens as Natural Compression Anchors for Implicit Reasoning**. 2026/10/16.
108. [2610.25612] **Curriculum Learning Teaches LLMs Long-Horizon Implicit Reasoning, Up To A Limit**. 2026/10/19.
109. [2610.25640] **Think, Compress, Answer: Reinforcement Learning for Multimodal Grounding via Reasoning Compression**. 2026/10/19.
110. [2610.26854] **Physical Words: A Vocabulary for Structuring Latent Reasoning in Robot Manipulation (PhysWord)**. 2026/10/19.
111. [2610.28498] **Present but Hard to Read: Probing Latent Reasoning with Minimal Pairs**. 2026/10/20.
112. [2610.29385] **Parallelizable Probabilistic Latent Reasoning for Diverse Reasoning Trajectories**. 2026/10/21.
113. [2610.30675] **OneLatent: Latent Reasoning for Efficient Foundation Recommendation Models**. 2026/10/22.
114. [2610.34336] **Improving the Chain-of-Thought Compressibility of Pruned Reasoning Models via Long–Short Contrastive Pruning**. 2026/10/25.
115. [2610.35229] **Joint Latent Reasoning and Observation for Agentic Video Understanding (LARVA)**. 2026/10/26.
116. [2610.42149] **Latent Reasoning in Continuous Space for Unified Multimodal Models**. 2026/10/30.
117. [2611.01926] **TAC-CoT: Tool-Empowered Implicit Chain-of-Thought via Interleaving Explicit Tokens**. 2026/11/01.
118. [2611.04814] **Think Less, Locally: Local Preference Learning for Reasoning Compression**. 2026/11/03.
119. [2611.05709] **Agents, Think Only When it Helps**. Min-Seon Kim et al. 2026/11/03.

### E. 网页信源（仅用于热度佐证）

- 腾讯云开发者社区，《2026 年 5 月 AI 论文盘点》，2026-07-02，`developer.cloud.tencent.com/article/2701996`（二手汇总；其引用编号已全部回到 arXiv 一手核实，且发现 2605.28820 的转述标题与原文有出入）。
- Mervin Praison，《Hybrid Reasoning Agents: Latent Thinking Halves Agent Tokens》，2026-10-19，`mer.vin/news/hybrid-reasoning-agents-latent-thinking-halves-agent-tokens/`（对 arXiv:2610.18861 与 togethercomputer 开源仓库的二手解读；所有数字以论文原文为准）。

> **核实与口径补充**：本报告实际列出 arXiv 论文 **119 篇**（正文另在勘误脚注引用 arXiv:2605.28820，合计核验 120 个编号），编号/标题/日期于 2026-11-11 经 `arxiv.org/abs/<id>` 摘要页 `citation_title` / `citation_date` meta 逐一核对，0 缺失、0 错号（含对知乎转述「ECCV 2026 CoLT」的溯源——其真身为 arXiv:2606.31986，准确数据为推理时间 10.1×、文本解码时间 22.6×）。HybridAgent、CoLT、INTACT 等加速与提分数字均为**作者口径**，本报告未做独立复现；综述表 1 的小模型横评为各原论文自报值、由综述 [2505.16782] 汇集，跨方法不可直接排名。
