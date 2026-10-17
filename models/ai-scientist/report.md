# AI Scientist 前沿技术报告：从 AI 工具到自主科研智能体——AI4S 的系统架构、能力边界与科学发现新范式

> **信源说明**：本报告基于 20 篇 arXiv / Nature 原始论文（arXiv 编号均经 `arxiv.org/abs` 页面逐一核实）、Anthropic / Google DeepMind 官方一手公告（2026-09/05 直读核实）与科技媒体交叉验证。凡仅有第三方转述、未获一手证实者，正文均显式标注「待独立验证」，不作为既成事实。
> **报告日期**：2026-10-17 ｜ **主题分类**：大模型技术（models）｜ **图表**：4 张暗色手绘 SVG

---

## 执行摘要

2026 年，大模型的竞争焦点正从「在数字空间里生成语言与图像」上移到一个更具野心的目标：**让 AI 自主提出假设、设计并执行实验、分析数据、撰写乃至评审科学论文**。这一被统称为 **AI Scientist（AI 科学家）/ 自主科研智能体** 的方向，是智源研究院《2026 十大 AI 技术趋势》中位列第四的「AI4S 北极星」，也是 2026 年 9 月全球科技舆论的绝对中心。

本报告的七个核心判断：

1. **赛道已从「单点 AI 工具」进入「端到端自主科研流水线」阶段**。以 Sakana AI Scientist 为标志，系统首次把「想法生成 → 代码实现 → 实验执行 → 结果分析 → 论文写作 → 模拟评审」串成无人（或少人）值守的闭环。

2. **但「自主科学发现」的宣传与系统真实可靠性之间存在显著落差**。独立第三方复现 Sakana 系统时发现 42% 的实验因代码错误失败，并伴随幻觉数值、错误引用、自相矛盾的结论。当前系统更接近「高度自动化的研究助理」，而非可独立担纲的科学家。

3. **要区分三层，不可混为一谈**：① **自主科研智能体（AI Scientist）**——多为演示性质；② **科学基础模型（Scientific Foundation Models）**——如 AlphaFold、GNoME，已有同行评审与实验验证，是真正被广泛采用的一层；③ **深度研究智能体（Deep Research）**——通用检索 + 长文写作，科研只是其用途之一。

4. **2026 年 9 月出现「AI 重大科学声明」密集期**：Anthropic 的 ART 酶系统（09-23）、费马大定理 Lean 形式化（09-05）、N=4 超对称 Yang-Mills 九圈振幅（09-25），OpenAI 的 Navier-Stokes 千禧年难题声明（09-08）。本报告第 8 章对其逐条做证据审计。

5. **三个流传甚广的误述必须纠正**：Sakana 2026 年《Nature》论文是**人类作者撰写的关于系统的论文**，不是「AI 写的论文发了 Nature」；Claude 费马项目是**把 Wiles 已有证明形式化为 Lean**，不是重新证明费马大定理；ART 项目官方口径是「**约 950 个 agents、21 小时、2.1 亿 token**」，且该酶**生物学功能未知**、人类完成了湿实验。

6. **闭环到物理世界靠自驱动实验室（Self-Driving Labs, SDL）**：Berkeley A-Lab 17 天尝试 58 个目标、合成 41 种新化合物；利物浦移动机器人 8 天跑完 688 个实验。材料与化学是「干智能 + 湿实验」融合最先跑通的领域。

7. **中国把「科学智能（AI4S）」视为战略机遇**：鄂维南院士 2018 年起推动 AI for Science、2021 年创立北京科学智能研究院；华为盘古气象、深势科技 DPA 大原子模型、字节系新生探途（2.9 亿美元天使轮）等构成国产版图。

---

## 目录

0. 为什么是现在：从「预测下一个词」到「自主发现新知识」
1. 概念与边界：AI Scientist、AI4S、科学基础模型与 Deep Research
2. 思想源流与技术谱系（1965–2026）
3. 系统架构深潜：端到端科研流水线的六层
4. 三大旗舰：Sakana AI Scientist、Google Co-Scientist、FutureHouse
5. 关键能力机制：树搜索、多智能体锦标赛、文献 RAG 与自动评审
6. 闭环到物理世界：自驱动实验室与湿实验
7. 科学基础模型层：被真正验证与采用的一层
8. 2026 年 9 月「突破时刻」逐条证据审计
9. 评测体系与能力边界
10. 安全、治理与科研诚信
11. 产业地图与中国布局
12. 开放问题与工程落地 Checklist
- 参考文献

---

## 0. 为什么是现在：从「预测下一个词」到「自主发现新知识」

### 0.1 范式转移的大背景

智源研究院在《2026 十大 AI 技术趋势》中给出的主线判断是：人工智能的演进核心正从「追求参数规模的语言学习」迈向「对物理世界底层秩序的深刻理解与建模」，院长王仲远将其概括为从「预测下一个词」（next-token prediction）跨越到「预测世界的下一个状态」（next-state prediction, NSP）。在这一主线下，趋势 4 明确提出：**「AI Scientist 成为 AI4S 北极星，国产科学基础模型悄然孕育」**——AI 在科研中的角色正从辅助工具升级为自主研究的「AI 科学家」，科学基础模型与自动化实验室的结合将极大加速新材料与药物研发。

这一判断并非孤例。2024–2026 年间，三条相对独立的技术曲线在「自主科研」这一点上汇合：

- **基础模型能力曲线**：模型在长程推理、代码生成、工具调用、多模态理解上的能力跨过可用门槛，使「把研究流程拆成 agent 可执行的子任务」首次成为可能；
- **智能体工程曲线**：多智能体编排、树搜索、检索增强、计算机使用（computer use）等 agent 技术成熟，让系统能驾驭「读文献—写代码—跑实验」这类长周期、需反馈的工作流；
- **实验自动化曲线**：云实验室、机器人实验平台、自驱动实验室在化学与材料领域规模化，为「数字智能」提供了可被程序化调用的「物理执行器」。

### 0.2 为什么这件事重要

科研是人类知识生产的根本环节。如果 AI 能显著压缩「假设—实验—验证」的周期与成本，其影响将不止于效率：它可能改变科学发现的速率分布、改变科研人员的工作方式，甚至改变「谁是科学家、谁是作者」的基本界定。正因赌注如此之大，2026 年 9 月每一项「AI 攻克数学/生物难题」的声明都会引发全球性关注——也正因如此，对这些声明做严格的证据分级、剥离营销叙事，是本报告的首要责任。

同时需要清醒：**「自主科研」目前最成熟的价值兑现，仍集中在有客观验证器（verifier）的环节**——代码能跑测试、数学能被形式化检查、分子能被仪器表征。而在需要品味、范式判断与真实世界新颖性的环节，AI 仍高度依赖人类。这一「验证器依赖」是理解当前所有 AI Scientist 系统能力边界的总钥匙。

---

## 1. 概念与边界：AI Scientist、AI4S、科学基础模型与 Deep Research

公共讨论中「AI 科学家」「AI for Science」「科研大模型」「Deep Research」常被混用。为避免概念滑动，本报告先做严格切分。

### 1.1 三层模型

| 层次 | 名称 | 核心任务 | 自主性 | 当前成熟度 |
|---|---|---|---|---|
| L1 | **科学基础模型**（Scientific Foundation Model） | 针对某类科学对象做高维预测/生成：蛋白结构、分子性质、材料能量、天气场 | 单一能力，人来驱动 | 高：有同行评审与实验验证，已被广泛采用 |
| L2 | **深度研究智能体**（Deep Research Agent） | 围绕给定问题做大规模检索、综合、长文写作，附引用 | 任务级自主，问题由人给定 | 中高：产品化、通用 |
| L3 | **自主科研智能体**（AI Scientist） | 自主提出问题/假设，并完成实验、分析、写作、评审的端到端闭环 | 流程级（部分开放-ended） | 中低：多为演示/研究原型，可靠性受限 |

三者并非替代关系，而是**嵌套调用**关系：一个 L3 的 AI Scientist 在做文献综述时会调用 L2 式的深度检索，在预测蛋白结构或材料能量时会调用 L1 的科学基础模型，在需要物理验证时调用自驱动实验室。

### 1.2 关键术语

- **AI for Science（AI4S，科学智能）**：最上位的概念，指将 AI 深度嵌入科研全流程、让 AI 成为科学探索主动参与者的新范式，既包括「AI 驱动科学发现」，也包括「科学原理反哺新一代 AI」。
- **AI Scientist**：本报告主角，指以 LLM 智能体为核心、追求科研流程端到端自动化的系统，目标是开放-ended（开放式）的科学发现。
- **Self-Driving Lab（SDL，自驱动实验室）**：以机器人 + AI 主动学习实现「假设—实验—表征」物理闭环的实验室，是 AI Scientist 在湿实验世界的执行末端。
- **AI co-scientist**：Google 对其多智能体科研伙伴的命名，强调「协作」而非「替代」。
- **Test-time compute（推理时计算）**：通过在推理阶段投入更多搜索/采样/辩论来提升产出质量，是 Co-Scientist 等系统提升假设质量的核心杠杆（见第 5 章）。

### 1.3 一条必须守住的边界：「关于系统的论文」≠「系统写的论文」

评估 AI Scientist 时最常见的概念偷换，是把「人类研究者撰写的、描述并评估某 AI 系统的论文」说成「AI 自己写的论文」。本报告在涉及 Sakana 2026 年《Nature》论文时会反复强调这一区分（见 4.1、8.3）。同理，「形式化一个已有证明」≠「首次证明」，「在玩具模型上用已知方法多算一圈」≠「发现新物理」。**严谨的措辞本身，就是这个领域技术素养的一部分。**

---
## 2. 思想源流与技术谱系（1965–2026）

「让机器做科学」不是 2024 年才出现的念头。理解六十年的源流，有助于看清今天的 AI Scientist 哪些是真正的跃迁、哪些是旧问题的新外衣。

### 2.1 符号主义时代：专家系统与机器发现（1965–1990s）

- **DENDRAL（1965，Stanford）**：由 Edward Feigenbaum、Joshua Lederberg、Bruce Buchanan 等发起，被视为第一个知识型专家系统，能根据质谱数据推断分子结构。它确立了「把领域知识编码为规则、让机器做专家级推断」的范式。
- **BACON 系列（1970s–80s，Langley 等）**：符号化的「科学发现程序」，能从数据中归纳出数值规律（如重新「发现」开普勒定律、理想气体定律），展示了机器自动形成假说的早期可能，但依赖人工精心构造的输入与算子。

这一阶段的根本局限是：知识靠人工编码、系统封闭、无法处理真实世界的感知与不确定性。

### 2.2 机器人科学家：物理闭环的先声（2000s–2020）

- **Adam / Eve（Ross King 团队）**：Adam 是首个通过「假设形成 + 物理实验」循环做出新科学发现的机器（酵母功能基因组学），相关成果 2009 年以 *The Automation of Science* 为题发表于《Science》；Eve 则用于早期药物筛选，奠定了现代自驱动实验室的架构雏形。
- **利物浦移动机器人化学家（2020，Nature）**：Burger、Cooper 等用改装的移动机器人在近乎标准的人类实验室里自主操作分配器与气相色谱，**8 天完成 688 个实验**，优化光催化分解水产氢，最终配方产氢量比起始条件提高 **6 倍**（DOI: 10.1038/s41586-020-2442-2）。

### 2.3 科学基础模型爆发（2020–2024）

- **AlphaFold 2（2021，Nature）**：蛋白结构预测达到实验精度，2024 年 Hassabis、Jumper 与 David Baker 因此共获诺贝尔化学奖；
- **盘古气象（2023，Nature）**、**GNoME（2023，Nature）**、**A-Lab（2023，Nature）**、**ChemCrow 与 Coscientist（2023，Nature）** 密集出现；
- **AlphaFold 3（2024-05）**、**AlphaProof/AlphaGeometry 2（2024-07，IMO 银牌级）**、**AlphaProteo（2024-09）**。

这一层的共同特征是：**单一能力、有客观真值、经同行评审与实验验证**——它为后来的「自主科学家」提供了可调用的强大「器官」，但本身并不自主。

### 2.4 端到端自主科研智能体（2024–2026）

- **2024-08**：Sakana AI 发布 *The AI Scientist*（arXiv:2408.06292），首个端到端 ML 科研流水线，每篇论文成本低于 15 美元；
- **2025-01/02**：Agent Laboratory（arXiv:2501.04227）、ScienceAgentBench（arXiv:2410.05080）、Google Co-Scientist 早期分享（arXiv:2502.18864）；
- **2025-04**：Sakana AI Scientist-v2（arXiv:2504.08066），以 Agentic Tree Search 摆脱人类代码模板，1 篇全 AI 生成论文通过 ICLR workshop 盲审；
- **2026-03/05**：Sakana 关于系统的人类署名论文登《Nature》（arXiv:2606.15497）；Google Co-Scientist 正式发布并登《Nature》；
- **2026-09**：Anthropic ART 酶系统、费马大定理 Lean 形式化、N=4 SYM 九圈振幅；OpenAI Navier-Stokes 声明（证据状态见第 8 章）。

### 2.5 谱系给出的启示

六十年脉络里反复出现三个主题：**知识从哪里来（人工编码 vs 学习）、假说是如何形成并被检验的（符号归纳 vs 搜索/采样）、验证在哪里闭合（纯数字 vs 物理实验）**。今天的 AI Scientist 用基础模型 + 智能体 + 自动化实验室，第一次同时在这三点上取得实质进展；但「开放-ended 的可靠科学判断」这一核心难题，从 BACON 到 Sakana 始终未被真正解决。

---

## 3. 系统架构深潜：端到端科研流水线的六层

尽管各家实现不同，一个典型的 AI Scientist 系统可抽象为六层流水线，外加两条横切的治理机制。

> 📊 **架构图**：见图 1（端到端自主科研流水线六层与双闭环）。
>
> diagrams/1-pipeline.html

### 3.1 第 1 层：选题与想法生成（Ideation）

输入通常是一个宽泛的研究领域（v1 还需人类提供代码模板，v2 起仅需一个 Markdown 研究主题）。系统通过文献检索 + LLM 发散，产出候选研究想法，并做初步的新颖性与可行性筛选。**这是质量与风险最集中的一层**：独立复现发现系统会把「micro-batching for SGD」这类已知概念误判为新颖（见 9.2）。

### 3.2 第 2 层：文献与知识检索（Literature / RAG）

调用检索引擎与论文库，做综述、找基线、定位方法。PaperQA2 等专门系统在这一层表现突出：回答每个问题平均检索并使用约 14.5 篇论文，并能在证据不足时回答「信息不足」（abstain），从而把 precision 做到约 85%（见 5.3）。

### 3.3 第 3 层：实验设计与代码实现（Implementation）

LLM 编写实验代码（v1 基于 aider-chat 风格的代码编辑），v2 起由 Experiment Manager agent 负责扩展、调试、放弃失败路径。这一层的可靠性是当前最大短板——复现显示每轮代码修改平均仅增加约 8% 字符，适应性有限。

### 3.4 第 4 层：实验执行（Execution）

在沙箱中运行代码、收集指标；在化学/材料场景则进一步调用云实验室或机器人平台执行湿实验。**关键在于执行结果是否可被客观判定**：单元测试、数学形式检查、仪器表征构成了「验证器」。

### 3.5 第 5 层：结果分析与迭代（Analysis Loop）

把实验输出（含图表）反馈给智能体，v2 引入视觉语言模型（VLM）评估图表质量，形成「执行—观察—修正」的内循环；在树搜索框架下，这一循环被组织成对研究路径的扩展与剪枝（见 5.1）。

### 3.6 第 6 层：写作与自动评审（Writing & Review）

自动生成论文全文（方法、实验、图表、结论），再由 Automated Reviewer 打分、提出修改意见，形成「作者—评审」外循环。Sakana 报告其自动评审器对人类评审决策的平衡准确率约 69%，F1 超过 NeurIPS 2021 一致性实验测得的人类间一致性水平（厂商口径，见 4.1、8.3）。

### 3.7 两条横切机制

- **多智能体编排**：不同角色（生成、反思、排序、演化、评审、监督）由专门 agent 承担，靠 supervisor 协调；
- **计算预算与搜索治理**：把有限的 token / 时间预算分配到最有希望的路径上（树搜索的剪枝、锦标赛的淘汰），本质是「推理时计算」的调度问题。

### 3.8 架构层面的核心矛盾

整条流水线呈现一个结构性矛盾：**越靠前的环节（选题、新颖性判断）越依赖品味与范式知识、越没有客观验证器，却越是决定科学价值的地方；越靠后的环节（跑测试、形式检查）越容易自动化，却只决定「对不对」而非「新不新、重要不重要」**。当前 AI Scientist 恰恰在易自动化的后半段表现好、在决定价值的前半段表现弱——这决定了它们暂时仍是「放大器」而非「替代者」。

---
## 4. 三大旗舰：Sakana AI Scientist、Google Co-Scientist、FutureHouse

### 4.1 Sakana AI Scientist：端到端自动化的开创者与争议中心

**v1（2024-08，arXiv:2408.06292）**。由 Sakana AI 联合 UBC、Vector Institute、Oxford 等推出（Chris Lu、Cong Lu、Robert Tjarko Lange、Jakob Foerster、Jeff Clune、David Ha）。它首次把想法生成、代码实现、实验执行、可视化、论文写作、模拟评审串成端到端闭环，在扩散模型、Transformer 语言模型、学习动力学三个机子领域验证，**每篇完整论文成本低于 15 美元**（厂商口径），并声称自动评审器在论文评分上接近人类评审。

**v2（2025-04，arXiv:2504.08066）**。Yamada 等引入三项关键升级：① **摆脱人类代码模板**，仅给一个 Markdown 研究主题；② 用 **Agentic Tree Search / Best-First Tree Search（BFTS，最佳优先树搜索）** 并行探索多条研究路径（思路借鉴国际象棋引擎），由 Experiment Manager agent 扩展、剪枝、调试、放弃死路；③ 加入 **VLM 视觉反馈**改进图表。第三方估算其单篇成本约 23–28 美元（非厂商审计，仅供参考）。

**ICLR 2025 workshop 实验**。Sakana 向 ICLR 2025「I Can't Believe It's Not Better」workshop 提交 3 篇未经人工编辑的 AI 生成论文，**1 篇通过盲审**，平均 6.33/10（三位审稿人给 6、7、6）。实验事先向组织方与 ICLR 披露、取得 UBC 伦理许可，论文在被接收后**按约定主动撤回**。必须强调：这是**接收率约 70% 的 workshop、而非主会**；审稿人只知道批次里可能有 AI 论文，不知道具体是哪篇。

**2026 年《Nature》论文（arXiv:2606.15497，DOI: 10.1038/s41586-026-10265-5）**。Yamada、Lange、Lu 等的 *Towards End-to-End Automation of AI Research* 于 2026 年发表于《Nature》。**这是一篇由人类作者撰写、描述并评估 AI Scientist 系统的论文，不是「AI 写的论文登上 Nature」**。论文提出所谓「科学的缩放定律」：基础模型越强，自动生成稿件质量越高（报告相关性 P < 0.00001）；自动评审器平衡准确率约 69%，F1 超过 NeurIPS 2021 实验中的人类间一致性；AI 生成稿件平均分 6.33/10，报告称高于约 55% 的人类 workshop 论文。论文同时坦承系统仍会产生幼稚想法、引用错误与方法严谨性不足。

> ⚠️ **口径提示**：上述 P 值、69%、55% 等数字来自论文/厂商口径，且比较基线是 workshop 论文而非主会，不能外推为「AI 科研能力已超过人类研究者」。

### 4.2 Google AI Co-Scientist：多智能体「想法锦标赛」

Google DeepMind 的 **Co-Scientist**（Gottweis 等，*Accelerating scientific discovery with Co-Scientist*，arXiv:2502.18864，2026 年发表于《Nature》655:487–496）定位为科研人员的**协作伙伴**而非替代者。其架构是一个受监督的多智能体系统，包含六类专门 agent（官方口径）：

- **Generation（生成）**：提出候选假设；
- **Reflection（反思）**：批判并改进假设；
- **Ranking（排序）**：用 Elo 式评分对假设两两比较；
- **Evolution（演化）**：在锦标赛中让假设相互杂交、变异、迭代；
- **Proximity（邻近性）**：评估假设与研究目标/已有证据的贴近度；
- **Meta-review（元综述）**：综合形成最终方案；外加 **Supervisor（监督者）** 统筹。

系统通过「**tournament of ideas（想法锦标赛）**」做假设的对抗式演化，并接入网络搜索、ChEMBL、UniProt，部分合作中调用 AlphaFold。论文在 **203 个研究目标**上观察到：**增加 test-time compute，假设的 Elo 评分持续提高**——这是「推理时计算可转化为科学假设质量」的直接证据；并在 15 个专家策划的复杂开放目标、以及 3 个生物医学场景（急性髓系白血病药物重定位、肝纤维化新靶点、抗微生物耐药机制）中验证。

官方披露的合作案例中，最常被引用的是与 Stanford（Gary Peltz）团队的肝纤维化药物再利用：候选物在实验中**阻断了约 91% 的瘢痕相关反应**（厂商/合作方口径，样本与实验设计以论文为准）。企业/机构预览合作方包括 Daiichi Sankyo、Bayer Crop Science、美国国家实验室 Genesis Mission 等。

### 4.3 FutureHouse：非营利的生物科研智能体集群

**FutureHouse** 是一家 501(c)(3) 非营利组织，目标是用 AI agent 自动化生物学等复杂科学研究，CEO/联合创始人为 Sam Rodriques，科学负责人/联合创始人为 Andrew White（PaperQA 系列作者之一）。其代表工作包括：

- **PaperQA2**（arXiv:2409.13740，*Language agents achieve superhuman synthesis of scientific knowledge*）：文献问答/综述智能体，在 LitQA2（248 道多选题）上 precision 约 85.2%、accuracy 约 66.0%，并以约 21.9% 的比例主动回答「信息不足」；对照人类专家 precision 约 64.3%、accuracy 约 63.1%（注意人类方差极大）。每题平均检索使用约 14.5 篇论文。
- **Robin**：2026-05 公告的端到端科学发现多智能体系统（*Demonstrating end-to-end scientific discovery with Robin*）；
- **DISCO**（2026-04）：为自然界不存在的化学反应设计酶；**OXtal**（2025-12）：生成式分子晶体结构预测。

FutureHouse 路线的特点是**深扎生物学、强调与实验学家协作、非营利开放**，与 Sakana 的「全自动 ML 研究」、Google 的「企业级协作平台」形成三种鲜明范式。

### 4.4 三极对比

| 维度 | Sakana AI Scientist | Google Co-Scientist | FutureHouse |
|---|---|---|---|
| 核心范式 | 端到端全自动 ML 研究流水线 | 受监督多智能体 + 想法锦标赛 | 生物科研 agent 集群、人机协作 |
| 自主主张 | 最强（开放-ended、少人值守） | 中（co-scientist，人在环中） | 中（与实验学家协作） |
| 招牌机制 | Agentic Tree Search + VLM 反馈 | Elo 排序 + 假设演化 + test-time scaling | 高引用准确的文献 RAG + 专业工具 |
| 验证方式 | 自动评审器 + workshop 盲审实验 | 生物医学湿实验合作验证 | 文献基准 + 生物/化学任务 |
| 主要风险 | 严谨性、幻觉、复现率 | 假设质量的可推广性 | 领域较窄、规模化待验 |
| 开放程度 | 开源（v1/v2 代码） | 企业预览/Gemini for Science | 非营利、论文与工具开放 |

---

## 5. 关键能力机制：树搜索、多智能体锦标赛、文献 RAG 与自动评审

### 5.1 Agentic Tree Search：把科研变成可剪枝的搜索

v2 的核心思想是把「做研究」建模为**在研究空间中的树搜索**：根节点是给定主题，每个分支是一条具体技术路线，节点展开意味着写代码、跑实验，节点的价值由实验结果估计。系统并行展开多条分支，用 Best-First 策略把算力集中到最有希望的路径，对失败/低收益路径及时剪枝。这与国际象棋引擎、以及推理模型中的 process reward / 树搜索同源，区别在于**这里的「环境」是真实的代码执行与实验反馈，且单步成本高、反馈噪声大**。

### 5.2 多智能体分工与「想法锦标赛」

单体 LLM 做科研的最大问题是缺乏自我对抗、容易过早收敛。Co-Scientist 的解法是用**角色异构 + 锦标赛选择**制造种群压力：生成 agent 保证多样性，反思 agent 施加选择压力，Elo 排序给出可比的适应度，演化 agent 重组优质假设。这与进化算法、以及 LLM 多智能体辩论（multi-agent debate）一脉相承，关键创新是把 test-time compute 与假设质量的单调提升用实验曲线联系起来。

### 5.3 高保真文献 RAG 与「弃权」机制

科研场景对引用准确性的要求远高于一般问答——一个错误引用就可能污染整条推理链。PaperQA2 的经验有两点尤其值得借鉴：① **强制溯源**，每个论断都要落到具体论文段落，平均每题调用约 14.5 篇文献；② **允许并鼓励弃权（abstention）**，在证据不足时回答「信息不足」，用约 21.9% 的弃权率把 precision 拉到约 85%。这说明在高风险知识任务中，「知道自己不知道」是一种可被设计出来的能力，而非仅是模型的内生属性。

### 5.4 自动评审：既是裁判也是瓶颈

自动评审器是闭环的关键，也是最受质疑的一环。Sakana 报告其平衡准确率约 69%、F1 超过历史测得的人类间一致性；但其后续的 **Multi-Layered Review（MLR）** 系统在受控植入核心矛盾的 benchmark 上，4 个 review 集成检测率约 73.43%（单次约 60.79%），而在真实撤稿论文（WithdrarXiv）上精确匹配检测率仅约 16.11%，50 篇隐藏 prompt injection 的论文中仅 8 篇被明确识别（第三方转述，**受控 benchmark 数字不可外推到真实审稿场景**）。这组反差说明：**在「被人为构造出来的错误」上表现好，不等于在「真实、隐蔽、分布外的学术不端」上可靠**。

### 5.5 推理时计算（test-time scaling）的角色

贯穿上述机制的一条暗线是 test-time compute：树搜索的分支数、锦标赛的轮数、文献检索的广度、自我反思的迭代次数，本质上都是「在推理阶段多花算力换质量」。Co-Scientist 用实验表明假设 Elo 随算力单调上升，这为「科研智能体的 scaling law」提供了初步证据；但第 9 章的基准数据也显示，这种提升在真实数据驱动发现任务上远未到「可靠」水平，且成本随能力快速上升。**算力能买到更好的候选，却买不到范式级的品味。**

---
## 6. 闭环到物理世界：自驱动实验室与湿实验

数字世界里的「实验成功」可以靠跑测试判定，但化学、材料、生物的真理最终要由物理世界裁决。自驱动实验室（SDL）就是 AI Scientist 的「物理执行器」。

> 📊 **架构图**：见图 2（干智能—湿实验双闭环与自驱动实验室构成）。
>
> diagrams/2-sdl.html

### 6.1 Berkeley A-Lab：17 天自主合成新材料

UC Berkeley 与劳伦斯伯克利国家实验室的 **A-Lab**（Szymanski 等，*An autonomous laboratory for the accelerated synthesis of novel materials*，《Nature》624:86–91，2023，DOI: 10.1038/s41586-023-06734-w）把材料数据库、第一性原理计算、文献自然语言模型、主动学习与机器人合成/原位表征整合为闭环。系统连续运行 **17 天，尝试 58 个目标材料，成功合成 41 种新化合物**（目标成功率约 70.7%）。这是「AI 提假设 + 机器人做实验 + 仪器给反馈」在材料领域最有代表性的硬成果。

### 6.2 云实验室：Emerald Cloud Lab 与 Strateos

**Emerald Cloud Lab（ECL）** 与 **Strateos** 提供可编程的远程自动化实验平台。ECL 的 Symbolic Lab Language（SLL）可被 LLM agent 在上下文中学习并编程——Coscientist 已演示让 GPT-4 直接编写 SLL 脚本调度仪器。云实验室把「湿实验」抽象成了类似「调用 API」的可编程能力，是打通干智能与湿实验的关键中间层。

### 6.3 自驱动实验室网络与机器人科学家谱系

- **Acceleration Consortium（多伦多大学，Alán Aspuru-Guzik）**：全球自驱动材料发现实验室网络的核心枢纽，获约 **2 亿加元**级别资助，推动 SDL 标准化与「data 可移植」。
- **Adam / Eve / Genesis（Ross King）**：从 2009 年《Science》的 Adam 到设计目标为每天并行检验约 1,000 个假设的 Genesis（含千个计算机控制的微型生物反应器），这条谱系把「假设驱动的物理闭环」推向高通量。
- **Cosmic Lab、Insilico Medicine** 等：前者探索太空/远程自动化，后者用 Pharma.AI 发现的特发性肺纤维化候选药 INS018_055 进入 II 期临床（具体结果以官方/试验注册为准）。

### 6.4 SDL 的瓶颈：不是模型，而是湿实验数据的产生速度

多家券商与一线机构在 2026 年形成一个共识性判断：AI 制药/材料研发当前的「决速步骤」（rate-limiting step）**已从模型速度转移到高质量湿实验数据的产生速度**。机器人通量、仪器接口标准化、实验数据的 FAIR 化（可发现、可访问、可互操作、可复用）与跨实验室数据可移植，成为比算法更硬的护城河。这也解释了为何礼来–英伟达愿意五年投入逾 10 亿美元建设干湿一体实验室。

---

## 7. 科学基础模型层：被真正验证与采用的一层

如果说 L3 的「自主科学家」仍多是演示，那么 L1 的科学基础模型已经扎实地进入了真实科研与产业流程。这一层是 AI Scientist 可调用的强大「器官」，也是当前 AI4S 价值兑现最充分的部分。

> 📊 **架构图**：见图 3（科学基础模型版图：蛋白/分子/材料/气象/数学）。
>
> diagrams/3-foundations.html

### 7.1 结构与生命：AlphaFold 3 与 AlphaProteo

- **AlphaFold 3（2024-05，Nature，DOI: 10.1038/s41586-024-07487-w）**：由 DeepMind 与 Isomorphic Labs 推出，把预测对象从蛋白拓展到蛋白–配体、蛋白–核酸等「生命所有分子」的结构与相互作用。AlphaFold 系列已为 Hassabis、Jumper 赢得 2024 年诺贝尔化学奖。
- **AlphaProteo（2024-09，arXiv:2409.08022）**：首个设计新型高强度蛋白结合物（binder）的 AI 系统。在 7 个目标蛋白上实验成功，**结合亲和力比现有最佳方法强 3–300 倍**；BHRF1 目标约 88% 候选结合成功，并首次为 VEGF-A 设计出成功 binder（第 8 个目标 TNFɑ 失败）。Francis Crick Institute 独立验证，部分 SARS-CoV-2 binder 可阻断病毒感染细胞（厂商口径）。

### 7.2 材料：GNoME 与气象：盘古 / GenCast / WeatherNext

- **GNoME（Merchant 等，Nature 624:80–85，2023，DOI: 10.1038/s41586-023-06735-9）**：用图网络 + 主动学习发现约 **220 万个**相对以往工作稳定的晶体结构，其中约 **381,000 个**位于更新后的能量凸包（数量级比以往扩大一个数量级），**736 个**结构已被独立实验实现；最终模型能量预测误差约 11 meV/atom、结构命中率 >80%。
- **气象**：华为**盘古气象**（Nature 2023，DOI: 10.1038/s41586-023-06185-3）开创 AI 中期天气预报；DeepMind 的 **GenCast**（概率预报）与 **WeatherNext / WeatherNext 3（2026-09）** 构成新一代天气模型梯队。

### 7.3 数学：AlphaProof / AlphaGeometry 2

DeepMind 的 **AlphaProof**（强化学习 + Lean 形式推理）与 **AlphaGeometry 2** 在 **IMO 2024** 六道题中解出 **4 道、得 28/42 分**，相当于银牌上端（金牌线 29 分）；AlphaGeometry 2 解第 4 题仅用约 19 秒。AlphaProof 方法论于 2025-11 在《Nature》发表（s41586-025-09833-y）。这是「AI + 形式化数学」第一次在顶级人类竞赛中拿到奖牌级成绩，也是 2026 年一系列 Lean 证明热潮的技术先声。

### 7.4 为什么这一层更可信

科学基础模型层有三个「自主科学家」层暂不具备的特征：**任务边界清晰、有客观真值（实验/竞赛/第一性原理）、经同行评审与独立复现**。它们不负责提出「该研究什么」，但在「给定问题给出高精度预测」上已稳定超越传统方法。理解这一层与 L3 的分工，才能对「AI 会不会取代科学家」给出不夸张也不保守的回答：**AI 先成为科学家手中越来越强的仪器，再逐步向「提出问题」这一核心逼近。**

---

## 8. 2026 年 9 月「突破时刻」逐条证据审计

2026 年 9 月，多项「AI 攻克重大科学难题」的声明在十余天内密集出现，构成本方向舆论热度的顶点。本章对每一条做证据分级——这是本报告最重要的一章。

> 📊 **架构图**：见图 4（9 月突破时间线与证据等级矩阵）。
>
> diagrams/4-evidence.html

### 8.1 Anthropic ART 酶系统（09-23）——一手公告已核实，功能仍未知

Anthropic 官方公告 *Claude discovers a novel enzyme system with CRISPR-like repeats*（本报告已直读官网，HTTP 200）宣布成立生命科学研究组，并用 Claude 在大规模基因组数据中发现一类新型酶系统 **ART（array-associated reverse transcriptases，阵列相关逆转录酶）**：含逆转录酶、重复非编码 DNA 阵列与一个功能未知的附属蛋白，结构上类似 CRISPR 的重复阵列，位于 jumbo phage，底层 RT 在既往研究（2021 年 MarsHill 报告）中已被识别，但 Claude 似乎首先注意到其重复阵列与搭档基因。

**官方口径关键数字（已一手核实）**：约 **950 个 agents**、**21 小时**、约 **2.1 亿（210 million）token**；agents 收集了逾 **20 万个 RT**、挑出约 **3,500 个新候选系统**、收敛到 **20 个**最有说服力的候选并产出供人类阅读的报告。Feng Zhang 在审阅预印本后评价其「genuinely intriguing」。

**必须随结论一并陈述的限制**：① 该酶**生物学功能未知**，尚未证明 RT 有活性、重复阵列 RNA 是其底物；② **人类完成了定义方向、选择候选与湿实验**，官方强调这是「agents 与人类在每一步协作」的新工作方式；③ 预印本托管于 Anthropic 自有网站、**未经同行评审**。因此准确表述是：**「Claude 协助发现了一个结构上类似 CRISPR、但功能未知的候选酶系统」**，而非「Claude 发明了新 CRISPR」。

> 注：部分第三方转述为「949 个独立 agent / 2.156 亿 token / 119 任务 / 19 报告 / 搜索 19 亿蛋白簇」，这些更细数字本报告未能从官方页逐一核对，**以官方「约 950 agents、21 小时、2.1 亿 token」为准**，其余标注为第三方口径。

### 8.2 Claude 费马大定理 Lean 形式化（09-05）——一手已核实，是「形式化」非「新证明」

Anthropic 官方研究页 *Formalizing Fermat's Last Theorem*（本报告已直读，HTTP 200）称：一个 agent 团队在约 **11 天**内基本自主地用 Lean 语言完成了费马大定理的**计算机可检查证明**，写下约 **1,300 万行 Lean、证明约 29,500 个中间定理**；该证明规模是 Mathlib 的 **5 倍以上**；整个项目消耗约 **60 亿（six billion）output tokens**，模型为通用内部研究模型（第三方称大致相当于 Claude Fable 5）。

**学术传承必须讲清**：这是**把 Wiles 已有证明形式化为 Lean 4**，遵循 Darmon–Diamond–Taylor 的简化阐述，建立在 Kevin Buzzard 2024 年发起的 Imperial FLT 社区项目等人类基础设施之上，并由 Buzzard 审阅。因此准确表述是：**「Claude 完成了费马大定理迄今首个完整的 Lean 机器可验证形式化」**，而非「Claude 证明了费马大定理」——数学定理本身 1994 年已由 Wiles 证明。其真正意义在于展示了 agent 驾驭超大规模、长周期形式化工程的能力（第三方提到独立 Lean kernel 校验逾百万声明零错误、仅用三个基础公理，这些细节以官方/GitHub 为准）。

### 8.3 OpenAI Navier-Stokes 千禧年难题声明（09-08）——官方页 403，待独立验证

OpenAI 据报道发布 *On the Navier-Stokes Millennium Prize Problem*，称一个内部协调的多智能体系统证明：初始静止的光滑流体在光滑外力下可在有限时间形成奇点且能量保持有限，声称覆盖 Clay 千禧年问题表述中的 C 与 D、含全空间与环面情形。

**证据状态（务必注意）**：本报告尝试直读 OpenAI 官方页时返回 **HTTP 403**，GitHub 仓库亦未审计；上述细节来自第三方（CellCog 等）转述。围绕该声明还存在**署名/贡献争议**（人类与 agent 的贡献界定）、以及「是否真正达到 Clay 问题严格标准」的外部数学质疑，且**未经同行评审**。因此本报告将其整体标注为：**「OpenAI 自称的结果，待数学界独立验证」**，不作为既成事实。千禧年问题的成立与否，最终取决于数学共同体的长期检验，而非新闻标题。

### 8.4 Claude N=4 SYM 九圈振幅（09-25/26）——第三方溯源较完整，属「已知方法 + 更多算力」

Anthropic Science 博客（Matt von Hippel 的 guest post）称 Claude（Fable 5.1）从一句话提示出发，算出平面 N=4 超对称 Yang-Mills 六粒子（hexagon）MHV 振幅的**九圈**结果，比 Dixon–Liu 2023 年的八圈纪录多一圈，由 Lance Dixon 独立核验。第三方溯源给出的成本约为：bootstrap 计算用 Python+SymPy 约 100 美元，完整路线对用户约 1,000–2,000 美元。

**边界**：N=4 SYM 是数学上的 **toy model（玩具模型）**，不直接等同于真实粒子物理发现；von Hippel 本人称这是**「用已知方法 + 更多算力」而非新思想**；中科院（Song He 团队）同周用 GPT-6 辅助得到大部分相同结果，物理结果将由人类发表。原始博客本报告未直读，故标注为**第三方较完整溯源、非新物理**。

### 8.5 审计小结

把四条放在一起，可得到一个冷静的结论：**2026 年 9 月确实标志着 AI agent 驾驭超大规模科研工程（千万行形式化、跨海量数据的模式发现）的能力跨上新台阶，但没有一条构成「AI 独立做出范式级科学发现」的证据。** 两项 Anthropic 成果是一手核实的「人机协作」成就，一项 OpenAI 声明尚待独立验证，一项振幅计算是已知方法的算力延伸。**剥离「AI 攻克难题」的标题党，真正的新闻是「协作式科研工程的自动化边界被显著推远」。**

---
## 9. 评测体系与能力边界

### 9.1 三类评测基准

| 基准 | 来源 | 任务形态 | 规模 | 关键结果 |
|---|---|---|---|---|
| **ScienceAgentBench**（arXiv:2410.05080，ICLR 2025） | OSU 等 | 真实数据驱动科学发现：提假设、设计实验、调试、分析 | 44 篇同行评审论文、102 个任务、4 学科、9 位专家验证 | 三次尝试下最佳非 o1 agent 独立成功率约 **32.4%**（加专家知识 34.3%）；o1-preview 约 **42.2%** 但成本高 |
| **MLE-bench**（arXiv:2410.07095，OpenAI） | OpenAI | 真实 Kaggle 竞赛端到端 ML 工程 | 75 个 Kaggle 竞赛，人类铜/银/金牌基线 | 最佳组合 o1-preview + AIDE 在约 **16.9%** 的竞赛中至少达到铜牌 |
| **ResearchAgent 评测**（arXiv:2404.07738，ICML 2024） | KAIST | 基于文献迭代生成研究问题/方法/实验设计 | 300 篇核心论文 + 50,091 篇参考文献 | 完整系统在问题/方法/设计上约 **4.52/4.28/4.18**（满分 5），前 3 轮提升后饱和 |

此外还有面向文献问答的 **LitQA2**（PaperQA2，248 题）、面向 ML 工程的各类 agent 竞技场，以及真实撤稿论文集 **WithdrarXiv**（arXiv:2503.05259）用于检验审稿系统。

### 9.2 独立复现揭示的硬伤

Beel、Kan、Baumgart 的 *Evaluating Sakana's AI Scientist*（arXiv:2502.14297，已一手核实标题与作者）给出了最系统的反面证据：

- 12 个被提议的实验中 **5 个（约 42%）因代码错误失败**；
- 文献综述粗糙，把已知概念（micro-batching for SGD）误判为新颖；
- 一个以「优化能效」为目标的实验，报告准确率提升却消耗更多算力，**结论与目标自相矛盾**；
- 每轮代码修改平均仅增加约 8% 字符；论文引用中位数仅 5 篇，34 个引用中仅 5 个来自 2020 年及以后；
- 出现缺失图表、重复章节、「Conclusions Here」占位符、幻觉数值等问题；估算单篇成本约 6–15 美元、人工介入约 3.5 小时。

### 9.3 评测方法论的陷阱

评估 AI Scientist 时要警惕四个陷阱：① **用自动评审器给自动生成的论文打分**——裁判与选手同源，存在系统性偏袒；② **把 workshop（高接收率）结果说成主会水平**；③ **受控 benchmark 外推**——在人工植入的错误上 73% 的检测率，不等于真实审稿准确率（真实撤稿仅约 16%）；④ **成本口径不一**——厂商「<15 美元」往往不含失败重试与人工时间。**严谨的评测应引入独立人类专家、真实任务分布、并把失败成本与人工介入计入总账。**

### 9.4 当前能力边界（一句话总结）

在**有客观验证器、边界清晰、单轮反馈快**的任务上（跑测试、形式证明、工具化文献检索、仪器表征），AI agent 已达到可用甚至惊艳的水平；在**需要范式判断、真实新颖性、跨学科品味、长程因果负责**的任务上，它们仍不可靠、须由人类兜底。能力边界的形状，基本由「验证器是否存在」这一条线勾勒。

---

## 10. 安全、治理与科研诚信

### 10.1 生物与化学两用（dual-use）风险

科研智能体的特殊风险在于：**它的能力恰好对准 CBRN（化学、生物、放射、核）领域**。Sakana 自己就指出，若 AI Scientist 接上生物/化学云实验室，可能并非出于恶意、仅因优化「新颖性」这一目标函数而产出危险的病原或物质。这是一种典型的**规格误用（specification gaming）**：系统忠实执行了「更新颖」，却背离了人类真正想要的「安全且有益」。Coscientist 论文正因 dual-use 关切，在监管成型前**扣留完整代码与提示**；AlphaProteo 则采取分阶段共享并参与 NTI AI Bio Forum。

### 10.2 负责任扩展政策（RSP）与能力分级

**Anthropic 的 Responsible Scaling Policy（RSP，官方一手）** 在 CBRN、网络安全、模型自主性三大领域做前沿风险评估，建立 ASL-2/3/4 分级、红线能力（Red Line Capabilities）、专家红队与独立评估（如 METR）机制，主张当模型能力跨过预设安全阈值时触发相应的部署护栏。这类「先定义危险能力等级、再据此决定部署」的思路，正成为前沿实验室治理生物风险的共同模板。需要注意：**自动评审/自检在隐蔽的分布外风险上同样会失灵**（见 9.2 的 prompt injection 漏检），因此不能把安全完全外包给模型自己。

### 10.3 科研诚信与「谁是作者」

- **幻觉进入科学记录**：自动生成论文中的伪造结果、幻觉方法、错误引用，若未经拦截进入预印本/期刊，将污染科学文献；
- **署名争议**：OpenAI Navier-Stokes 声明引发「人类与 agent 各贡献了什么」的质疑；更宏观地，ICLR 等学术机构尚无统一的 AI 作者规范，Sakana 建议的系统性水印又易被绕过；
- **评审被操纵**：论文中隐藏 prompt injection 可能劫持自动评审流程（50 篇中仅 8 篇被明确检测到），这对日益自动化的同行评审是直接警告。

### 10.4 治理的方向

综合各方实践，较稳健的治理组合是：**危险能力的事前分级与发布闸门 + 高风险实验的人在环中与访问控制 + 强制溯源/可复现的工件（数据、代码、环境）+ 独立第三方评估 + 明确的署名与 AI 使用披露规范**。科研领域因其自我纠错传统与同行评审机制，本应是 AI 治理的试验田；但 2026 年的现实是，能力发布的速度仍快于诚信与治理规则的成型速度。

---
## 11. 产业地图与中国布局

### 11.1 海外版图

- **系统层**：Sakana AI（AI Scientist v1/v2/Nature 论文）、Google DeepMind（Co-Scientist、Gemini for Science）、FutureHouse（PaperQA2/Robin/DISCO/OXtal）、OpenAI（Deep Research、Navier-Stokes 声明）；
- **科学基础模型 + 制药**：Isomorphic Labs（AlphaFold 商业化，2026-05 完成 **21 亿美元 B 轮**）、Insilico Medicine（INS018_055 进入 II 期）；
- **实验/算力基础设施**：Emerald Cloud Lab、Strateos、Acceleration Consortium（约 2 亿加元）、礼来–英伟达（五年逾 10 亿美元干湿一体实验室）。

### 11.2 中国布局：把「科学智能」当战略机遇

鄂维南院士是中国 AI4S 的关键推动者。据北大深研院科学智能学院转载《瞭望东方周刊》的专访（官方一手），他 2018 年提出推动 AI for Science，2021 年创立**北京科学智能研究院**（全球首个以科学智能命名、专注该领域的新型研发机构），定义 AI4S 为「将 AI 深度嵌入科研全流程、让 AI 成为科学探索主动参与者」的新范式，强调「科学智能是中国科技创新历程中前所未有的契机」，并指出 AI 下半场关键是**数据基础设施**（芯片、系统软件、数据库、HPC、算法的完整底层体系）。

国产代表力量：

| 方向 | 代表 | 要点 |
|---|---|---|
| 气象 | 华为**盘古气象** | Nature 2023，AI 中期天气预报开创者 |
| 原子/分子模拟 | **深势科技 DPA** | 覆盖 90+ 元素的大原子模型，源自鄂维南团队，北京科学智能研究院成果 |
| 生命科学 | 百图生科 BioMap、华深智药、上海 AI Lab | 蛋白/生命科学基础模型 |
| AI 制药 | **新生探途 Anew Labs**（字节分拆） | 2026-09 天使轮 **2.9 亿美元**、投后 15 亿美元，字节持股 56%；产品 AnewFold 等 |
| 科研/深度研究 | Kimi Researcher、Qwen Deep Research、智谱 AutoGLM | 深度研究智能体（细节以各公司官方为准） |

智源《2026 十大 AI 技术趋势》将 AI Scientist 列入趋势，并特别呼吁「我国需整合力量，加快构建自主的科学基础模型体系」。

### 11.3 竞争格局的判断

中美在 AI4S 上的差异更多在**组织方式**而非单点智能：美国靠 DeepMind/OpenAI/Anthropic 等前沿实验室 + 大型药企/云厂深度绑定 + 云实验室商业化；中国靠院士牵头的新型研发机构 + 科技巨头（华为/字节）切入 + 科研数据基础设施的国家推动。**真正的决胜点正在从「模型多强」转向「湿实验数据闭环 + 底层算力/软件栈 + 跨机构协作网络」的系统工程。**

---

## 12. 开放问题与工程落地 Checklist

### 12.1 六大开放问题

1. **开放-ended 可靠判断**：如何让系统在无验证器的环节（选题、范式判断、重要性评估）也可靠？这是最硬的科学问题。
2. **验证器依赖的边界**：在数学/代码上跑通的自改进闭环，能否安全迁移到知识/创造性任务而不发生模型崩溃？
3. **评审同源偏差**：如何避免「用同源模型既当选手又当裁判」带来的系统性虚高？
4. **湿实验数据瓶颈与标准化**：SDL 的通量、接口与数据可移植性能否跟上模型进步？
5. **安全与开放的平衡**：如何在扣留危险能力（dual-use）与保持科研开放共享之间取得平衡？
6. **科研生产关系**：署名、credit、激励与同行评审制度如何适应「人机协作科研」？

### 12.2 工程落地 Checklist（若要在自己的领域构建科研智能体）

- [ ] 先盘点你的领域**哪里有客观验证器**（测试、仿真、形式检查、仪器），优先在这些环节闭环；
- [ ] 文献链路强制**溯源到具体段落**，并为系统设计「**弃权/信息不足**」出口；
- [ ] 用**独立于生成模型的评审/校验**（异源模型或人类专家），避免裁判选手同源；
- [ ] 把**失败重试成本与人工介入时间**计入总账，不被「<15 美元」口径误导；
- [ ] 树搜索/多智能体要配**预算与剪枝治理**，监控 token 消耗与路径收敛；
- [ ] 湿实验/外部执行加**访问控制与人在环中**，对 dual-use 设发布闸门；
- [ ] 保存**可复现工件**（数据、代码、环境、会话转录），并明确 AI 使用与署名披露；
- [ ] 建立**难例回流**通道，把人类兜底的案例变成下一轮改进数据。

### 12.3 结语

2026 年的 AI Scientist，更像一台正在快速进化的「科研粒子加速器」：它不替科学家决定该撞出什么，但能把假设到证据的路径压缩数个数量级。9 月的喧嚣提醒我们，既要看见协作式科研工程边界被显著推远的真实进步，也要警惕「AI 独立攻克难题」的叙事泡沫。**在可预见的未来，最有成效的图景不是「AI 取代科学家」，而是「会用 AI 的科学家 + 可靠的自主科研基础设施」共同重塑知识生产的速率。**

---

## 参考文献

> 说明：以下 arXiv 编号均已经 `arxiv.org/abs/<编号>` 页面元数据（标题/作者/日期）逐一核实；Nature/Science 论文经 DOI 核实。访问/核实日期：2026-10-17。

1. Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., Ha, D. **The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery.** arXiv:2408.06292 (2024). https://arxiv.org/abs/2408.06292
2. Yamada, Y., et al. **The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search.** arXiv:2504.08066 (2025). https://arxiv.org/abs/2504.08066
3. Yamada, Y., Lange, R. T., Lu, C., Lu, C., et al. **Towards End-to-End Automation of AI Research.** arXiv:2606.15497; *Nature* (2026), DOI: 10.1038/s41586-026-10265-5. https://arxiv.org/abs/2606.15497
4. Schmidgall, S., et al. **Agent Laboratory: Using LLM Agents as Research Assistants.** arXiv:2501.04227 (2025). https://arxiv.org/abs/2501.04227
5. Chen, Z., et al. **ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery.** arXiv:2410.05080; ICLR 2025. https://arxiv.org/abs/2410.05080
6. Baek, J., et al. **ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models.** arXiv:2404.07738; ICML 2024. https://arxiv.org/abs/2404.07738
7. Chan, J. S., et al. **MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering.** arXiv:2410.07095; ICLR 2025. https://arxiv.org/abs/2410.07095
8. Boiko, D. A., MacKnight, R., Kline, B., Gomes, G. dos P. **Autonomous chemical research with large language models.** *Nature* 624, 570–578 (2023). DOI: 10.1038/s41586-023-06792-0
9. Bran, A. M., Cox, S., Schilter, O., Baldassari, C., White, A. D., Schwaller, P. **ChemCrow: Augmenting large-language models with chemistry tools.** arXiv:2304.05376; *Nature Machine Intelligence* 6, 525–535 (2024). https://arxiv.org/abs/2304.05376
10. Skarlinski, M. D., et al. **Language agents achieve superhuman synthesis of scientific knowledge.** arXiv:2409.13740 (2024). https://arxiv.org/abs/2409.13740
11. Gottweis, J., Weng, W.-H., Daryin, A., Tu, T., et al. **Accelerating scientific discovery with Co-Scientist.** arXiv:2502.18864; *Nature* 655, 487–496 (2026). https://arxiv.org/abs/2502.18864
12. Szymanski, N. J., et al. **An autonomous laboratory for the accelerated synthesis of novel materials.** *Nature* 624, 86–91 (2023). DOI: 10.1038/s41586-023-06734-w
13. Tom, G., et al. **Self-Driving Laboratories for Chemistry and Materials Science.** *Chemical Reviews* 124, 9633–9732 (2024). DOI: 10.1021/acs.chemrev.3c00674
14. Abramson, J., et al. **Accurate structure prediction of biomolecular interactions with AlphaFold 3.** *Nature* (2024). DOI: 10.1038/s41586-024-07487-w
15. Jänes, J., et al. **De novo design of high-affinity protein binders with AlphaProteo.** arXiv:2409.08022 (2024). https://arxiv.org/abs/2409.08022
16. Merchant, A., et al. **Scaling deep learning for materials discovery (GNoME).** *Nature* 624, 80–85 (2023). DOI: 10.1038/s41586-023-06735-9
17. Bi, K., et al. **Accurate medium-range global weather forecasting with 3D neural networks (Pangu-Weather).** *Nature* 619, 533–538 (2023). DOI: 10.1038/s41586-023-06185-3
18. **AlphaProof / AlphaGeometry 2: AI achieves silver-medal standard solving International Mathematical Olympiad problems.** Google DeepMind (2024); *Nature* (2025), s41586-025-09833-y. https://deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level
19. Beel, J., Kan, M.-Y., Baumgart, M. **Evaluating Sakana's AI Scientist: Bold Claims, Mixed Results, and a Promising Future?** arXiv:2502.14297 (2025). https://arxiv.org/abs/2502.14297
20. **WithdrarXiv: A Real-World Benchmark for Retracted Paper Detection / Withdrawal Checking.** arXiv:2503.05259 (2025). https://arxiv.org/abs/2503.05259
21. Schmidgall, S., et al. / Sakana AI. **ShinkaEvolve: Towards Open-Ended And Sample-Efficient Program Evolution.** arXiv:2509.19349 (2025). https://arxiv.org/abs/2509.19349
22. Yamada, Y., et al. **Digital Red Queen: Adversarial Program Evolution in Core War with LLMs.** arXiv:2601.03335 (2026). https://arxiv.org/abs/2601.03335

**一手公告/机构信源**：

- Anthropic. **Claude discovers a novel enzyme system with CRISPR-like repeats.** (2026-09-23) https://www.anthropic.com/news/claude-discovers-novel-enzyme-system
- Anthropic. **Formalizing Fermat's Last Theorem.** (2026-09-05) https://www.anthropic.com/research/formalizing-fermats-last-theorem
- Google DeepMind. **Co-Scientist: a multi-agent AI partner to accelerate research.** (2026-05-19) https://deepmind.google/blog/co-scientist-a-multi-agent-ai-partner-to-accelerate-research
- FutureHouse. https://www.futurehouse.org
- Anthropic. **Reflections on our Responsible Scaling Policy.** https://www.anthropic.com/news/reflections-on-our-responsible-scaling-policy
- 鄂维南专访（《瞭望东方周刊》，北大深研院科学智能学院转载，2025-10-29）https://ai4s.pkusz.edu.cn/info/1006/1315.htm
- 财联社/科创板日报. **新生探途天使轮**（2026-09-17）https://finance.eastmoney.com/a/202609173877125351.html
- 新华网. **智源发布 2026 十大 AI 技术趋势**（2026-01-08）http://www.news.cn/tech/20260108/b99127c88a4640bbab49e5c6294264d2/c.html

> **待独立验证声明**：OpenAI *On the Navier-Stokes Millennium Prize Problem*（官方页本报告访问返回 403，未经同行评审，细节来自第三方）；N=4 SYM 九圈振幅的原始 Anthropic Science 博客（第三方较完整溯源，属已知方法 + 更多算力）。
