# 合成数据前沿技术报告：破除「2026 数据枯竭魔咒」

### —— 从模型崩溃（Model Collapse）到修正缩放定律（Rectified Scaling Law）

> 当人类文本的存量可见尽头，模型自己生成的数据，究竟是通向 AGI 的「免费燃料」，还是让 AI 一代代变笨的「自食毒果」？本报告系统梳理 2022–2026 年大模型合成数据的理论、算法、产业与治理。

📅 2026-10-19 ｜ 方向：大模型 / 数据范式 ｜ 智源《2026 十大 AI 技术趋势》第 7 位

---

## 摘要（TL;DR）

1. **硬约束已经到来。** Epoch AI 估算人类公开文本的质量加权有效存量约 **300 万亿（300T）token**，按当前消耗速度将在 **2026–2032 年间耗尽**；叠加版权诉讼、平台封闭与高质量语料稀缺，「数据墙」从预言变成现实约束。
2. **合成数据已从「增广」变为「支柱」，但主要发生在后训练环节。** 可核实的最强官方数字是 NVIDIA Nemotron-4-340B **对齐数据 98% 为合成**；Llama 3、Phi-3/4、DeepSeek、Qwen3、Kimi K2 均在 SFT/推理/代码/Agent 数据上大规模使用合成数据。网络流传「2026 预训练 50–70% 为合成」的说法**无具名官方报告背书，属第三方推测**，本报告不予采信为事实。
3. **合成数据有「原罪」：模型崩溃。** Shumailov 等人发表于 *Nature* 的研究证明，不加区分地用模型生成内容递归训练，会让分布尾部（长尾知识）逐代消失、误差单调累积。
4. **但崩溃并非必然。** 后续研究给出了清晰的边界条件：**真实数据「累积」而非「替换」**、多源生成、保持足够人类锚点比例、外加可执行验证与自校正，可使测试误差获得独立于迭代次数的有限上界。
5. **合成数据遵循「修正缩放定律」。** 其收益存在拐点：SynthLLM 测得收益约在 **300B token 后明显递减**，8B 模型约在 **1T token** 达峰；这意味着合成数据不是无限提款机，而是需要精算配比、质量分级与溯源治理的工程学科。
6. **方法谱系已高度成熟：** 自改进（Self-Improve/STaR/RFT）、自奖励（Self-Rewarding/SPIN）、深度自演化（rStar-Math）、指令合成（Self-Instruct/Evol-Instruct/Magpie）、合成教科书（phi 系列）、教师蒸馏（Orca/R1 蒸馏）、可执行验证（CodeT/DCLM/GRPO）、世界模型仿真（Genie/GameNGen/DriveDreamer/GenSim）。
7. **评测污染是合成数据时代的伴生难题**，LiveCodeBench、LiveBench 等「时间切分、持续更新」的防污染基准成为新的可信度来源。

---

## 目录

1. [为什么是现在：数据枯竭的硬约束](#1-为什么是现在数据枯竭的硬约束)
2. [什么是合成数据：定义与统一分类](#2-什么是合成数据定义与统一分类)
3. [模型崩溃：合成数据的「原罪」](#3-模型崩溃合成数据的原罪)
4. [崩溃并非必然：边界条件与抗崩溃理论](#4-崩溃并非必然边界条件与抗崩溃理论)
5. [修正缩放定律：合成数据如何继续 scale](#5-修正缩放定律合成数据如何继续-scale)
6. [技术路线谱系：六大方法族](#6-技术路线谱系六大方法族)
7. [产业落地：前沿模型怎么用合成数据](#7-产业落地区前沿模型怎么用合成数据)
8. [生态、市场与治理](#8-生态市场与治理)
9. [评测污染与可信评测](#9-评测污染与可信评测)
10. [开放问题与风险](#10-开放问题与风险)
11. [工程落地 Checklist](#11-工程落地-checklist)
12. [结语](#12-结语)
13. [参考文献](#13-参考文献)

---

## 1. 为什么是现在：数据枯竭的硬约束

### 1.1 从「数据红利」到「数据赤字」

过去十余年深度学习的核心叙事是**缩放定律（Scaling Law）**：模型越大、数据越多，性能越好。这一叙事隐含一个前提——互联网上有无穷无尽的高质量人类文本。但 2022 年以来，这个前提被三组力量同时动摇：

- **存量见顶：** 适合训练的高质量人类文本并非无限，其总量可以被估算，且消耗速度远快于新增速度；
- **版权与合规收紧：** 主流媒体与平台开始主张数据权利、封锁爬虫、发起诉讼；
- **质量门槛提高：** 前沿模型对「教科书级」「推理级」数据的需求，远超对普通网页噪声的需求，而这类数据本就稀缺。

### 1.2 Epoch AI 的关键估算

AI 预测研究机构 Epoch AI 在《Will We Run Out of Data? Limits of LLM Scaling Based on Human-Generated Data》中给出了被广泛引用的估算【第三方研究估算】：

| 指标 | 估算值 | 置信区间 / 备注 |
|---|---|---|
| 人类公开文本**存量** | 约 **300T token** | 90% CI：100T–1000T（质量加权、考虑重复训练后） |
| 耗尽时间 | **2026–2032** | 80% CI |
| 5× 过训练情景 | 约 **2027** 用完 | — |
| 100× 过训练情景 | 约 **2025** 用完 | — |
| Llama 3-70B | 约 **10× 过训练** | 即在 Chinchilla 最优 token 数上训练约 10 倍数据 |

> 所谓「过训练（overtraining）」，指在有限数据上以多于 Chinchilla 计算最优点的 token 数进行训练。这正是数据受限时代的普遍做法。

其推理链条是：互联网可用文本总量有限 → 头部实验室每年算力增长远快于数据新增 → 为喂饱大模型被迫在同一份数据上重复训练（多个 epoch）→ 边际收益递减，最终撞上「数据墙」。

### 1.3 供给侧的三重收缩

**（1）通用网页语料的质量与体量。** Common Crawl 每月抓取约 **21 亿网页、355–364 TiB** 未压缩内容【官方口径，2026 年月度公告】，看似庞大，但其中高质量、低重复、适合训练的比例很低——LAION 式过滤流程往往只保留约 **10%** 的爬取数据。Mozilla 2024 年一项研究称，**64%** 受调 LLM 预训练使用了至少一个 Common Crawl 过滤版本【第三方研究】。

**（2）版权语料下架。** 以 Books3 为代表：该数据集含约 **196,640 本图书**，曾是 The Pile、RedPajama 的组成部分，**2023 年 10 月**因版权投诉从 Hugging Face 下架【诉讼文书 / 第三方】。

**（3）平台封闭与爬虫封锁。** Reddit、Stack Overflow、X（原 Twitter）转向收费授权；2025 年 7 月 Cloudflare 率先将 AI 抓取改为**默认 opt-in（默认拦截）**，超 100 万客户选择拦截 AI 爬虫；中文互联网平台也普遍强化数据壁垒。

### 1.4 为什么合成数据成为必然答案

在「人类数据稀缺」与「模型能力仍需继续增长」的矛盾之间，合成数据是最直接的出路：它**边际成本低、可规模化、可定向生成长尾与推理样本、可通过程序验证**。Gartner 早在 2022 年 6 月就预测「2030 年前合成数据将在 AI 模型中超过真实数据」【第三方预测】。问题随之从「要不要用」变成「怎么用才不会让模型越训越笨」——这正是本报告后半部分的核心。

> 架构图：数据墙与三重约束（diagrams/1-data-wall.html）


---

## 2. 什么是合成数据：定义与统一分类

### 2.1 定义

**合成数据（Synthetic Data）** 指由模型、程序或仿真器**人工生成**、而非直接采集自人类真实活动原始记录的训练（或评测）数据。在大模型语境下，它通常以「文本指令—回答对」「推理链（CoT）」「代码 + 单测」「多模态图文/视频对」「Agent 工具调用轨迹」等形式存在。

需要区分三个易混概念：

- **数据增强（Augmentation）：** 在真实数据基础上做改写、回译、加噪等，真实数据仍是主体；
- **蒸馏（Distillation）：** 用强「教师」模型的输出（含软标签或推理轨迹）训练弱「学生」，教师输出本质上是合成数据；
- **自训练 / 自改进（Self-Training / Self-Improve）：** 模型用**自身**生成的数据再训练自己——这是最激进、也最容易触发崩溃的形态。

### 2.2 统一视角：按「谁生成」与「用在哪」两个正交轴分类

**轴一：按数据来源（谁生成）**

| 类别 | 生成者 | 代表方法 | 主要风险 |
|---|---|---|---|
| **自生成（Self-generated）** | 模型自己 | Self-Improve、STaR、RFT、SPIN、Self-Rewarding | 分布收窄、模型崩溃 |
| **教师模型生成（Distillation）** | 更强的外部模型 | Orca、R1 蒸馏、大模型生成 SFT | 教师偏见、能力天花板 |
| **指令演化 / 模板合成** | 模型 + 规则 | Self-Instruct、Evol-Instruct、Magpie | 同质化、指令分布漂移 |
| **合成教科书 / 程序化生成** | 模型生成或人工设计的语法/模板 | phi 系列、Cosmopedia | 覆盖面、真实感 |
| **仿真器 / 世界模型生成** | 生成式环境、游戏引擎、扩散世界模型 | Genie、GameNGen、DriveDreamer、GenSim、Cosmos | sim2real gap、物理一致性 |
| **Agent 轨迹重放** | 智能体在真实/沙箱环境中执行 | Kimi K2 智能体数据、工具调用 trace | 轨迹质量、奖励稀疏 |

**轴二：按训练环节（用在哪）**

- **预训练（Pretraining）：** 合成正文、知识改写、合成代码/数学语料（风险最高，因规模巨大且决定基础能力）；
- **监督微调（SFT）：** 合成指令—回答、合成推理链（目前合成数据应用最成熟、占比最高的环节）；
- **强化学习（RL）：** 可验证奖励任务（数学答案、代码执行）、合成偏好对、过程奖励；
- **评测（Evaluation）：** 合成考题、合成检索/安全测试任务（需警惕评测失真）。

### 2.3 数据飞轮 vs 递归自消耗

合成数据有两种截然相反的循环结构，理解二者差异是理解全报告的关键：

- **数据飞轮（Data Flywheel，良性）：** 更多真实用户 → 产生更多**真实交互与反馈** → 筛选出高质量信号 → 训练更好的模型 → 吸引更多用户。真实世界始终在向循环注入新信息。
- **递归自消耗（Recursive Self-Consumption，恶性）：** 模型 A 生成数据 → 训练模型 B → B 再生成数据 → 训练模型 C……循环中**没有外部新信息注入**，近似误差逐代累积，最终走向模型崩溃。

> 核心判据：**循环里是否持续有「真实数据 / 外部真值」作为锚点注入。** 有锚点是飞轮，无锚点是自食。

---

## 3. 模型崩溃：合成数据的「原罪」

### 3.1 里程碑：从 arXiv 到 Nature

合成数据最著名的风险，由牛津/剑桥等机构的 Shumailov 等人系统提出：

- 早期版本：《The Curse of Recursion: Training on Generated Data Makes Models Forget》（arXiv:2305.17493，2023-05 首发，2024-04 更新）；
- 正式发表：《AI models collapse when trained on recursively generated data》，***Nature* 631**(8022), 755–759，2024-07-24 在线（DOI: 10.1038/s41586-024-07566-y）。

### 3.2 什么是模型崩溃

**模型崩溃（Model Collapse）** 指：当生成模型被一代代地用**自身（或同类模型）产出的数据**训练时，其对真实数据分布的近似误差逐代累积，导致——

1. **分布尾部消失：** 低概率但真实存在的「长尾」事件被系统性低估直至遗忘；
2. **高概率模式被过度强化：** 输出趋向高度同质、「流畅但狭隘」；
3. **误差单调上升：** 在「用合成数据替换真实数据」的设定下，测试误差随递归代数单调增加，且作者强调这一退化可能是**不可逆**的。

该现象并非语言模型独有：论文在 **高斯混合模型（GMM）、变分自编码器（VAE）和大语言模型（LLM）** 上都观察到了崩溃。

### 3.3 直觉：为什么会丢尾巴

每个生成模型都只是真实分布 $p_{\text{real}}(x)$ 的一个近似 $p_{\theta}(x)$。用它采样得到的数据，已经丢失了一部分真实分布信息（尤其采样数量有限时，低频事件更容易漏采）。再用这份「失真样本」训练下一代，近似误差被二次放大。高概率区因为样本多、反复出现而被牢牢记住；低概率区因为样本稀少、一旦漏采就永久缺席。如此递归，分布像被反复复印的图像——中间越来越浓、边缘越来越淡，最终只剩一团模糊的「均值」。

### 3.4 两个常被忽视的细节

- **崩溃是「静默」的：** 崩溃模型的输出依然语法流畅、语气自信，困惑度等表层指标可能看不出问题，直到下游真实任务性能显著下滑才被发现；
- **污染比例可以很低：** 有从业者与后续研究指出，即使较小比例的合成数据混入，也可能引起词汇、句法、语义多样性的可测量退化（具体阈值取决于数据与模型，详见下一章的边界条件——注意「低至千分之一即崩溃」是强主张，应与「累积真实数据可避免崩溃」一并理解，不能脱离条件引用）。

### 3.5 安全维度的次生风险

合成数据收窄还会削弱安全护栏。有测量显示，在**单一小型生成器**产出的窄分布数据上做微调，可能在保持表层输出质量的同时移除安全防护，部分研究中相当比例的越狱提示回应落入危险区域（此类数字随生成器与评测而变，应理解为「单源 + 无锚点」条件下的风险，而非合成数据的普遍属性）。这与第 10 章的开放问题相呼应。

---

## 4. 崩溃并非必然：边界条件与抗崩溃理论

*Nature* 论文引发巨大关注后，一个关键问题被提出来：**模型崩溃是宿命，还是特定（错误）用法的结果？** 2024 年的多篇后续工作给出了令人宽慰但有条件的答案——**崩溃主要发生在「用合成数据替换真实数据」且缺乏外部校正时；改变训练结构，可以避免。**

### 4.1 最重要的结论：「累积」而非「替换」

Gerstgrasser、Schaeffer 等人在《Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data》（arXiv:2404.01413）中给出了直接反驳：

- **做法差异：** 「替换（replacement）」是每代只用合成数据、丢弃上一代真实数据；「累积（accumulation）」是把每一代新生成的合成数据**叠加**到始终保留的原始真实数据上；
- **实验：** 在 TinyStories 上训练 9M GPT-2，以及 12M/42M/125M Llama-2；替换会逐代抬高测试交叉熵，而**累积**真实 + 合成数据可使误差持平甚至下降；
- **理论：** 在线性模型下可证明，累积设置中测试误差具有一个**独立于递归迭代次数的有限上界**——真实数据像锚一样，持续把分布拉回真值。

> 实践含义极其简单却关键：**永远不要丢弃真实数据；合成数据只能「加」，不能「换」。**

### 4.2 稳定性条件：真实数据比例要足够大

Bertrand、Bose 等人的《On the Stability of Iterative Retraining of Generative Models on their own Data》（arXiv:2310.00429）从优化稳定性角度刻画了条件：

- 当**初始生成模型足够接近真实分布**、且**干净真实数据相对合成数据的比例足够大**时，迭代再训练是稳定的；
- 论文用 normalizing flows 与扩散模型在 CIFAR-10、FFHQ 上验证了该理论。

这把「能不能递归训练」转化为一个可量化的工程问题：**锚点数据的占比与初始模型的保真度，共同决定循环是否稳定。**

### 4.3 统计边界：存在「最大安全合成比例」

Seddik 等人的《How Bad is Training on Synthetic Data? A Statistical Analysis of Language Model Collapse》（arXiv:2404.05090）给出统计刻画：

- 仅用合成数据训练时，崩溃在统计上不可避免；
- 混入真实数据时，**存在一个可避免崩溃的最大合成数据比例阈值**，该阈值随模型与数据场景而变。

这解释了为何「合成占比 90% 无锚点」危险，而「合成占比 50% 且保留真实锚点」往往安全——**安全与否取决于配比，而非是否使用合成数据本身。**

### 4.4 自校正循环：用「校正函数」主动纠偏

Gillman 等人的《Self-Correcting Self-Consuming Loops for Generative Model Training》（arXiv:2402.07087）不依赖「多留真实数据」，而是引入一个**校正函数（correction function）**：

- 理想校正函数可让自消耗循环**指数级地更稳定**；
- 在人体动作生成任务中，即使合成与真实数据比例高达 **100%**，配合校正仍可避免崩溃。

### 4.5 多源生成：别让一个模型「自产自销」

除上述理论外，实证研究一致表明**多源（multi-source）生成显著优于单源**：

- 单源生成（一个模型产出全部合成数据）的困惑度分布更窄，是分布崩溃的直接信号；
- 多源生成在响应内/响应间词汇多样性上保持更好；有实验中多源微调提高了合成输出的困惑度（多样性↑），同时降低了在真实人类文本上的困惑度（建模更准）；
- 实用做法：跨模型家族（开源 + 商业 API）、采样温度调度（低温保事实、高温增多样）、系统提示/角色多样化。

### 4.6 小结：抗崩溃四原则

| 原则 | 做法 | 理论依据 |
|---|---|---|
| **真实锚点永不离场** | 累积而非替换，保留 T0/T1 人类数据 | arXiv:2404.01413（误差有限上界） |
| **锚点比例足够大** | 设定并程序化执行最大合成比例 | arXiv:2310.00429、2404.05090 |
| **多源、多样化生成** | 跨家族 + 温度/角色调度 | 多源实证 |
| **主动校正与验证** | 校正函数 + 可执行验证（见第 6 章） | arXiv:2402.07087、CodeT 等 |

---

## 5. 修正缩放定律：合成数据如何继续 scale

### 5.1 从「经典缩放」到「修正缩放」

经典缩放定律把损失描述为模型参数量 $N$、真实 token 数 $D$、算力 $C$ 的平滑幂函数。但合成数据引入了两个新变量：**合成数据的质量（取决于生成器）** 与**合成数据的相对比例**。当人们试图用合成数据替代稀缺真实数据继续 scale 时，损失曲线不再是简单幂律，而呈现「先改善、后饱和、甚至反转」的形状——这被称为**修正缩放定律（Rectified Scaling Law）**。

### 5.2 关键实证：SynthLLM

Qin、Dong 等人的《Scaling Laws of Synthetic Data for Language Models》（arXiv:2503.19551，2025-03 首发，2025-10 更新）是该方向的代表性工作：

- 提出 **SynthLLM** 框架，系统测量合成数据规模与模型性能的关系；
- 合成数据遵循 **rectified scaling law**，收益存在明显拐点；
- **收益约在 300B（3000 亿）合成 token 之后显著递减**；
- **不同大小模型的「达峰/饱和」所需合成数据量不同：约 8B 模型在 1T token 附近达峰，更小的 3B 模型反而需要约 4T token**（小模型从合成数据中吸收收益更慢）；
- 在 4.5B 合成 token 设置下，MATH 准确率预测为：3B 模型 **60.0%**、8B 模型 **71.3%**。

> 解读：合成数据不是「越多越好」的线性资源，而像一种有「最佳剂量」的补剂——剂量不足收益不够，剂量过大则边际收益趋零甚至因分布收窄而反降。

### 5.3 数据过滤本身也要 scale

Goyal、Maini 等人的《Scaling Laws for Data Filtering — Data Curation cannot be Compute Agnostic》（arXiv:2404.07177）指出一个常被忽视的耦合关系：

- **数据过滤/策展策略不能脱离训练算力来谈**——同一个过滤器，在不同训练算力预算下优劣可能反转；
- LAION 式流程仅保留约 **10%** 爬取数据；论文对不同质量数据池的效用、重复曝光的效用衰减、多池组合进行建模，**无需联合训练即可预测组合后的性能**。

含义：合成 + 过滤的配方必须与目标训练算力**联合优化**，不存在「一招通用的黄金过滤器」。

### 5.4 数据受限时的重复训练边界

Muennighoff、Rush 等人的《Scaling Data-Constrained Language Models》（arXiv:2305.16264）直接研究「数据不够只能重复用」的计算最优策略：

- 实验覆盖最多 **900B 训练 token、9B 参数模型、约 400 次训练**；
- 固定算力下，**最多约 4 个 epoch** 的数据重复，相对使用等量唯一数据的性能损失可忽略；
- 超过 4 个 epoch 后，继续增加算力的价值最终降至零。

这为「在真实数据耗尽前，靠重复 + 合成能撑多久」给出了量化答案：**适度重复几乎免费，但不能无限重复；合成数据是接续这一缺口的主力，但本身也受第 5.2 节的拐点约束。**

### 5.5 一张图理解三重约束

综合本章，前沿训练同时受三条曲线约束（见图 1「数据墙与三重约束」）：

1. **真实数据存量** 有天花板（Epoch AI：约 300T token）；
2. **重复训练** 在约 4 epoch 内近似无损、之后价值归零；
3. **合成数据** 遵循修正缩放定律，在数百 B 到数 T token 处出现收益拐点。

可行解空间 = 用足真实锚点 + 适度重复 + 在拐点内精算合成配比与质量分级。

---

## 6. 技术路线谱系：六大方法族

> 架构图：六大方法族 2022–2026 演进谱系（diagrams/3-taxonomy.html）

合成数据方法在 2022–2026 年间形成了清晰的演进谱系。下面按「自改进 → 自奖励/自博弈 → 深度自演化 → 指令合成 → 合成教科书 → 教师蒸馏」六族梳理（可执行验证与世界模型仿真因自成体系，分别在第 6.7、6.8 节与第 7 章展开）。

### 6.1 自改进族：用无标签问题 + 自产 CoT

**《Large Language Models Can Self-Improve》**（Huang, Gu et al., arXiv:2210.11610，2022-10）

- 核心思想：无需人工标注，仅用一批**无标签问题**，让模型自行生成多条 CoT 解答，通过**自洽（self-consistency）投票**筛选高置信答案，再用这些样本微调自己；
- 关键数据（540B 模型）：GSM8K **74.4%→82.1%**、DROP **78.2%→83.0%**、OpenBookQA **90.0%→94.4%**、ANLI-A3 **63.4%→67.9%**。

**STaR（Self-Taught Reasoner）**（Zelikman, Wu et al., arXiv:2203.14465，2022-03）

- 在自生成推理基础上增加**「合理化（rationalization）」**：当模型答不对时，给它正确答案，让它反推出能得到该答案的推理过程，再把最终答对的样本留作训练；
- 迭代「生成 → 筛选 → 微调」，在 CommonsenseQA 上可与参数量约 **30 倍**的 SOTA 微调模型表现相当。

**RFT（Rejection sampling Fine-Tuning）**（Yuan et al., arXiv:2308.01825，2023-08）

- 系统分析数学推理的缩放关系，发现**预训练损失比参数量更能预测推理性能**；
- 强调收集**多条不同的正确推理路径（distinct reasoning paths）**比堆数量更重要；
- 多模型 rejection samples 将 LLaMA-7B 的 GSM8K 准确率从 SFT 的 **35.9%** 提升到 **49.3%**。

### 6.2 自奖励 / 自博弈族：连「评委」都是自己

**Self-Rewarding Language Models**（Yuan, Pang et al., arXiv:2401.10020，2024-01）

- 让模型同时充当「答题者」和「LLM-as-a-Judge 评委」：自己生成指令与回答、自己打分、再用 DPO 迭代提升；
- Llama-2-70B 经 3 轮迭代后，在 AlpacaEval 2.0 上超过 Claude-2、Gemini Pro 与 GPT-4-0613；指令遵循与奖励质量同步上升。

**SPIN（Self-Play Fine-Tuning）**（Chen, Deng et al., arXiv:2401.01335，2024-01）

- 把自训练构造成一场**博弈**：模型要区分「自生成响应」与「人类真实响应」，在自我对弈中把弱分布推向人类分布；
- HuggingFace Open LLM Leaderboard **58.14→63.16**，MT-Bench **5.94→6.78**；
- 理论上训练目标的全局最优恰在「模型策略 = 目标数据分布」时达到，实证可超过补充 GPT-4 偏好的 DPO 模型。

> 自奖励/自博弈的魅力是摆脱对外部评委与人类偏好的依赖，但也最贴近「递归自消耗」的危险区——实践中通常需要真实参考分布（人类响应）作为博弈目标，本质仍是「有锚点」的自我提升。

### 6.3 深度自演化族：MCTS + 过程奖励

**rStar-Math**（Guan, Zhang et al., arXiv:2501.04519，2025-01）

- 用**蒙特卡洛树搜索（MCTS）**在推理步骤空间搜索，配合**过程奖励模型（PRM）**逐步打分，自动构造大量带正确步骤标注的深度思考数据，并进行多轮自演化；
- 4 轮自演化，为 **747K** 数学问题生成数百万解答；
- Qwen2.5-Math-7B：MATH **58.8%→90.0%**；Phi3-mini-3.8B：**41.4%→86.4%**；AIME 平均解出 **53.3%（8/15）**。

意义：证明**小模型**也能靠「搜索 + 过程验证 + 自演化」达到接近前沿大模型的数学推理水平，且整条数据生产线可由程序验证把关（呼应第 6.7 节的「验证比生成容易」）。

### 6.4 指令合成族：从零「无中生有」造指令

**Self-Instruct**（Wang, Kordi et al., arXiv:2212.10560，2022-12）

- 用少量人工种子指令，让模型自己生成新指令、输入与输出，经简单过滤后回灌微调，几乎零人工成本启动指令对齐；
- Vanilla GPT-3 在 Super-NaturalInstructions 上绝对提升 **33 个百分点**；人类评估中与 InstructGPT-001 仅差约 **5 个百分点**。

**WizardLM / Evol-Instruct**（Xu, Sun et al., arXiv:2304.12244，2023-04）

- 不增加指令数量，而是让模型把一条**简单指令逐步「演化」**（加深、加约束、换场景、具体化）成复杂指令；
- GPT-4 自动评估中，WizardLM 在 **29 项技能中的 17 项**达到 ChatGPT **90% 以上**能力；高复杂度指令的人类评估更受偏好。

**Magpie**（Xu, Jiang et al., arXiv:2406.08464，2024-06）

- 巧妙利用对齐模型的「左模板（pre-question template）」与自回归特性：**不给任何用户提示**，仅靠模板 token 就能诱导模型自己吐出指令和响应；
- 用 Llama-3-Instruct 生成 **400 万条**指令—响应，筛出 **30 万条**高质量样本；仅用 Magpie 做 SFT 的 Llama-3-8B 在部分任务上可媲美用约 **1000 万数据**训练的官方 Instruct 版。

### 6.5 合成教科书族：「质量」碾压「数量」

**Textbooks Are All You Need（phi-1）**（Gunasekar, Zhang et al., arXiv:2306.11644，2023-06）

- 核心主张：**「教科书质量」的合成数据**能让小模型获得惊人能力；
- phi-1 仅 **1.3B 参数**，用 **6B token** 高质量网络数据 + **1B token** 合成教科书/习题，在 **8 张 A100** 上训练 **4 天**；
- HumanEval pass@1 **50.6%**、MBPP **55.5%**。

**phi-1.5（Textbooks Are All You Need II）**（Li, Bubeck et al., arXiv:2309.05463，2023-09）把该范式从代码扩展到常识推理与自然语言任务，进一步验证「小模型 + 高质量合成语料」路线。后续 Phi-3、Phi-4 的工程化落地见第 7 章。

### 6.6 教师蒸馏族：站在巨人输出的肩膀上

**Orca**（Mukherjee, Mitra et al., arXiv:2306.02707，2023-06）

- 用 GPT-4 的**逐步解释轨迹、思维过程与复杂指令**做教师，ChatGPT 作助教，渐进训练 13B 学生模仿「推理过程」而非只抄答案；
- 采样 **500 万**条 ChatGPT 回复，再采 **100 万**条 GPT-4 回复；Orca-13B 在 Big-Bench Hard 上相对 Vicuna-13B 提升**超过 100%**，AGIEval 提升 **42%**。

**Orca 2**（Mitra, Del Corro et al., arXiv:2311.11045，2023-11）

- 进一步教小模型**按任务选择解题策略**（逐步推理 / 先回忆再答 / 直接回答）；
- 在 15 个基准、约 100 项任务、36,000+ prompt 上评测，零样本推理相对 LLaMA-2-Chat-13B 宏观平均提升 **47.54%**。

**DeepSeek-R1 蒸馏**（arXiv:2501.12948）

- 把 R1 的高质量推理样本（约 **80 万条**）蒸馏到 Qwen、LLaMA 等 1.5B–70B 基座；
- R1-Zero 在 AIME 2024 上 pass@1 从 **15.6%→77.9%**，自洽采样达 **86.7%**；R1 本体 AIME 2024 / MATH-500 为 **79.8% / 97.3%**；1.5B 蒸馏模型 AIME 2024 达 **28.9%**、MATH 达 **83.9%**。

> 蒸馏的边界：学生通常难以超越教师，且教师的偏见/错误会被继承。Phi-4 的有趣之处正在于通过「多智能体生成 + 自修订 + 指令反转」，在 STEM QA 上**反超教师 GPT-4**，说明合成数据可以超越单纯「复制教师」（见第 7 章）。

### 6.7 可执行验证族：让「事实」自己说话

合成数据最难的不是生成，而是**知道生成得对不对**。当存在「程序化真值（programmatic oracle）」时，验证可以做到客观、廉价、可大规模——这是当前最可靠的质量闸门。

**CodeT**（Chen, Zhang et al., arXiv:2207.10397，2022-07）

- 让代码模型同时生成候选程序和测试用例，**实际执行代码**，用「代码—测试输出一致性」和「候选间一致性」双重投票选优；
- HumanEval pass@1 提升至 **65.8%**，相对 code-davinci-002 绝对 +**18.8** 个点。

**DeepSeekMath / GRPO**（Shao, Wang et al., arXiv:2402.03300，2024-02）

- 数学网页数据选择管线 + **组相对策略优化（GRPO）**：对同一问题采样一组输出，用组内相对奖励算优势，无需独立 Critic；
- 7B 在 120B 数学 token 上继续预训练，MATH 不投票 **51.7%**、64 样本自洽投票 **60.9%**。GRPO 此后成为 RLVR（可验证奖励 RL）的主力算法。

**数据策展平台 DCLM / DataComp：**

- **DCLM（DataComp-LM）**（Li, Fang et al., arXiv:2406.11794）：提供来自 Common Crawl 的 **240T token** 标准化语料、53 项下游评测、412M–7B 规模的统一实验平台；DCLM-Baseline 用 **2.6T token** 从零训 7B，MMLU 5-shot 达 **64%**，比 MAP-Neo 高 6.6 点且算力少 40%；
- **DataComp**（Gadre, Ilharco et al., arXiv:2304.14108）：面向图文多模态，候选池 **128 亿**图文对、38 个测试集；DataComp-1B 训练的 CLIP ViT-L/14 ImageNet 零样本 **79.2%**。

**语义去重 SemDeDup**（Abbas, Tirumala et al., arXiv:2303.09540）

- 用预训练 embedding 识别「语义相近、字面不同」的近重复；在 LAION 子集可删除 **50%** 数据而性能损失很小，近似训练时间减半。

> **「验证比生成容易」的不对称性是合成数据工程最大的杠杆：** 过量生成 3–5 倍候选，再用执行器 / 规则 / 交叉模型 / 去重无情过滤，留下小而精的数据集，胜过大量未验证数据。

### 6.8 世界模型仿真族

用生成式环境 / 世界模型直接产出「带动作标注的合成经验」，是机器人与自动驾驶领域的合成数据路线（Genie、GameNGen、DriveDreamer、MagicDrive、GenSim）。因其与世界模型方向强相关，统一在第 7.4 节展开。

> 架构图：抗崩溃合成数据工厂五阶段流水线（diagrams/2-pipeline.html）


---

## 7. 产业落地：前沿模型怎么用合成数据

本章信息均来自各模型**官方技术报告/官方博客**，厂商自报数据标注【官方口径】；不同环节的合成占比差异很大，切勿混用。

### 7.1 后训练/对齐环节：合成数据已是绝对主力

| 模型 | 关键披露（均为【官方口径】） | 来源 |
|---|---|---|
| **NVIDIA Nemotron-4-340B** | Instruct 版**对齐数据 98% 为合成**；10 万行合成对话仅配 **1% 人工**即可让 Llama-3-70B 对齐后追平/超过官方 Instruct 版 | 开发者博客，2024-08 |
| **Meta Llama 3**（8B/70B/405B） | 预训练约 **15T token**；代码侧生成 **270 万+ 合成 SFT 样本**，其中约 **100 万条**带执行反馈的合成编程对话；405B 用自身数据再训无益甚至降分，故引入**执行反馈作为真值** | arXiv:2407.21783 |
| **Microsoft Phi-3**（3.8B） | 重度过滤网页 + LLM 合成数据；phase-2 混入合成数据教逻辑与小众技能；small/medium（7B/14B）训 **4.8T token** | arXiv:2404.14219 |
| **Microsoft Phi-4**（14B） | 全程策略性合成：多智能体提示、自修订工作流、**指令反转**；STEM QA 上**反超教师 GPT-4**，超越单纯蒸馏 | arXiv:2412.08905 |
| **Kimi K2**（1T 总参/32B 激活 MoE） | 预训练 **15.5T token**（含知识/数学数据改写）；后训练核心是大规模**智能体数据合成管线**（工具生成、智能体多样化、评分规准、多轮轨迹、真实执行）+ RLVR 与自批评奖励 | arXiv:2507.20534 |
| **DeepSeek-R1** | R1-Zero 为**纯规则奖励 RL、无 SFT**；R1 = 冷启动 SFT → RL → 采样生成 SFT 混入 V3 多域数据 → 二次 RL → 蒸馏到 Qwen/Llama；AIME 2024 **78.8%**、MATH-500 **97.3%** | arXiv:2501.12948 |
| **Qwen3** | **36T token、119 种语言**；用 Qwen2.5/Math/Coder 合成**数万亿** token（教材/问答/指令/代码）；后训练用 QwQ-32B 对每 query 生成 N 候选 + 人工复核 | arXiv:2505.09388 |
| **DeepSeek-V3** | 预训练 **14.8T** 多元 token；后训练**从 R1 系列蒸馏推理能力** | arXiv:2412.19437 |

**两个关键观察：**

1. **「自我数据再训」对超大模型会失效。** Llama 3 405B 明确发现用自身输出再训练无益甚至降分——这与修正缩放定律的拐点、以及模型崩溃的风险一致。前沿模型的解法是引入**外部真值**（代码执行结果、可验证答案）而非简单自产自销。
2. **合成数据可以「青出于蓝」。** Phi-4 通过精心设计的生成—修订流程反超教师模型，说明合成数据的价值不止于蒸馏复制，而在于构造教师也少见的、覆盖更全的训练分布。

### 7.2 预训练环节：谨慎、未充分披露

- 与后训练的高合成占比不同，**预训练阶段的合成占比普遍披露不足**。Google Gemini 1.5/2.5 技术报告中，合成数据主要用于**评测与安全**（合成 needle-in-a-haystack 检索，召回 >99%；合成对话做提示注入测试），**未披露预训练合成比例**【官方口径】；
- Meta Llama 4（Scout/Maverick，2025-04）采用 MoE + 早期融合原生多模态，可联合预训练大量无标注文本/图像/视频，但官方**未公布合成数据具体数字**；
- 因此，网络上「2026 年预训练 50–70% 为合成数据」的说法**缺乏具名官方来源**（其可检出处为个人博客且表格自标「illustrative」），应视为第三方推测，不能作为事实。

### 7.3 中国的开源数据布局

智源 OpenSeek 开源 **>4TB** 高质量中英文数据与合成方案【官方口径，2025-02】：通用知识标签体系、简单 QA 合成、Long-CoT 的 Backward/Forward/RAG 合成，以及 RL 数据采样 + 奖励模型/规则过滤；配套 IndustryCorpus 1.0（3.4TB、18 行业）。这反映国内在「开源数据 + 可复现合成配方」上的体系化努力。

### 7.4 世界模型仿真：直接合成「带动作的经验」

在机器人与自动驾驶领域，合成数据不是文本，而是**由仿真器/世界模型生成的视觉帧与动作序列**：

| 工作 | 合成了什么 | 关键数据 |
|---|---|---|
| **Genie**（arXiv:2402.15391） | 从无标注网络视频学习的**生成式可交互环境**，潜在动作实现逐帧可控 | 11B 模型；约 **30,000 小时**平台游戏视频；可由文本/图/草图生成可玩环境 |
| **GameNGen**（arXiv:2408.14837） | 扩散模型实时模拟 DOOM，按历史帧+动作预测下一帧 | 单 TPU **20 FPS**；下一帧 PSNR **29.4**；连玩 5 分钟人类难辨真假 |
| **DriveDreamer**（arXiv:2309.09777） | 真实驾驶驱动的世界模型，两阶段扩散生成可控驾驶视频 | nuScenes FID/FVD **14.9/340.8**；合成图使 BEVFusion mAP +**3.0** |
| **MagicDrive**（arXiv:2310.02601） | 多视角街景生成，支持相机位姿/地图/3D框/文本控制 | 加入合成数据后相机-only BEVFusion mAP **32.88→35.40** |
| **GenSim**（arXiv:2310.01361） | LLM 自动生成机器人仿真任务、课程与专家演示 | GPT-4 把任务扩展 **10 倍至 100+**；sim2real 长程任务较基线 +**25%** |
| **NVIDIA Cosmos** | 物理 AI 世界基础模型，用物理感知视频生成合成训练数据 | 平台 2025-01 起迭代；2026-07 下载量破 **1000 万**【官方口径】 |

> 这条路线把「世界模型」与「合成数据」两个趋势合流：**世界模型既是被训练出来的智能，又是生产训练数据的工厂。** 其核心瓶颈是 sim-to-real gap 与长时程物理一致性（详见世界模型报告与第 10 章）。

> 架构图：世界模型作为合成数据引擎的 sim2real 闭环（diagrams/4-world-model.html）


---

## 8. 生态、市场与治理

### 8.1 数据标注与合成数据产业

**人工标注 / RLHF 阵营：**

- **Scale AI**：2025-06 Meta 以约 **143 亿美元**获其 49% 无表决权股份、估值约 **290 亿美元**，创始人 Alexandr Wang 被 acquihire（政府业务分拆）【权威媒体报道，Reuters 等】；
- **Surge AI**：Edwin Chen 2020 年创立，2025-07 首次外部募资约 **10 亿美元**，报道估值 **150–250 亿美元**；第三方（Sacra）估其 2024 年化收入约 **12 亿美元**【第三方估算】。

**合成数据平台：**

- **Gretel**：2025-03 被 **NVIDIA 收购**，媒体称金额约 **3.2 亿美元**、约 80 人并入【TechCrunch 引 Wired】；
- **MOSTLY AI**：完成 2500 万美元融资，2024-10 获美国国土安全部（DHS）隐私增强合成数据合同【第三方】；
- 开源/工具侧：**NVIDIA NeMo Curator**（大规模数据策展）、**Argilla**（人工抽检 UI）、EleutherAI 数据配方、语义去重工具等。

**数据溯源的兴起：** 前沿实践开始为每个数据分片附带数据卡（如 `data-card`），记录**生成器模型、验证器版本、种子哈希、质量分级**，以便执行配比、调试分布漂移、满足合规披露。

### 8.2 版权：决定合成数据合法性的判例线

合成数据的法律边界在 2023–2026 年被一系列案件塑造：

| 时间 | 案件 / 事件 | 要点 |
|---|---|---|
| 2023-12 | **NYT v. Microsoft/OpenAI**（SDNY）起诉 | 索赔「数十亿」；2026-09 双方就合理使用提交交叉简易判决动议 |
| 2025-06 | **Bartz v. Anthropic**，法官 Alsup 分拆裁决 | **合法取得**图书用于训练属合理使用；但从 LibGen/PiLiMi 留存**盗版书**不受保护 |
| 2025-08 | Anthropic 就 Bartz 案达成 **15 亿美元和解** | 覆盖 48.2 万+本书、销毁盗版数据集；2026-07 最终批准 |
| 2025-11 | **Getty v. Stability AI** 英国高院判决 | 驳回次级版权侵权——「**模型权重中无副本**」；认定有限商标侵权 |
| 2025-11 | 荷兰 BREIN 迫使 Common Crawl 移除 **200 万+** 荷兰语新闻 | 平台侧内容清理 |
| 2026-04/06 | News/Media Alliance（900+ 站点）、DCN（AP/NYT 等）向 Common Crawl 发要求函 | 主张「版权不是 opt-out 制度」，属要求函非诉讼 |

> 趋势：司法正在区分「**合法获取的数据用于训练**」（倾向容忍）与「**盗版/侵权来源数据**」（不保护），并开始正面回答「模型权重里是否存在版权副本」这一技术法律问题。

### 8.3 监管与合规披露

- **EU AI Act / GPAI：** 2025-07-24 EU AI Office 发布训练数据摘要模板，通用目的 AI（GPAI）的披露义务于 2025-08-02 生效，**须披露合成数据及其来源模型**；存量模型宽限至 2027-08-02【官方口径】；
- **美国加州 AB 2013**：2026-01-01 生效，要求公开披露合成数据的使用【立法】；
- **中国：** 2025-01-13 发改委、国家数据局、财政部、人社部**四部门**联合印发《关于促进数据标注产业高质量发展的实施意见》，系国家层面首次系统谋划数据标注产业【官方口径】；国家数据局亦在明确数据作为生产要素的定位。

监管方向清晰：**合成数据不再是「不能说的秘密」，披露来源、比例与质量分级正成为高风险模型的硬性义务。**

---

## 9. 评测污染与可信评测

合成数据放大了一个伴生难题：**当训练（或合成生成器）已经见过评测题，分数就不再可信。**

### 9.1 污染的两条路径

1. **直接污染：** 评测集本身被包含在预训练语料或教师模型的训练数据中，模型「背下答案」；
2. **合成间接污染：** 用前沿模型合成训练数据时，教师模型可能已在训练中见过评测题，其输出把「答案知识」蒸馏给学生；合成评测若与训练数据同源，也会系统性偏向生成器。

Deng 等人的《Investigating Data Contamination in Modern Benchmarks for Large Language Models》（arXiv:2311.09783）系统检测了现代基准中的污染情况，揭示污染比想象中普遍，且会虚高报告性能。

### 9.2 解法：时间切分 + 持续更新的「活基准」

**LiveCodeBench**（Jain, Han et al., arXiv:2403.07974）

- 只收集**评测 cutoff 之后**竞赛平台新发布的编程题，从时间上保证训练数据不可能包含；
- 来自 LeetCode、AtCoder、CodeForces；论文版含 **400 道** 2023-05 至 2024-05 发布的题，覆盖代码生成、自我修复、代码执行、测试输出预测。

**LiveBench**（White, Dooley et al., arXiv:2406.19314）

- **按月更新**，题目来自近期数学竞赛、新 arXiv 论文、新闻与新数据集，形成「污染受限」基准；
- 尽量用客观 ground truth 自动评分，减少人类/LLM 裁判偏差；规模覆盖 0.5B–405B，顶级模型准确率也低于 70%。

### 9.3 对合成数据实践的要求

- 维护一个**纯真实、严格 holdout、永不进入任何生成器/训练集**的评估集；
- 优先用「时间切分」「新鲜数据」而非静态老基准判断真实进步；
- 对代码/数学等可验证领域，直接用执行/答案真值评分；
- 报告分数时披露训练 cutoff 与合成生成器身份，便于判断污染风险。

---

## 10. 开放问题与风险

1. **预训练阶段的安全合成比例仍无共识。** 后训练高合成占比已被验证有效，但预训练（决定基础能力与世界知识）能用多少合成数据而不损伤泛化，缺乏公开、可复现的定论；厂商也普遍不披露。
2. **「信息从哪里来」的根本约束。** 自生成在信息论上不产生新信息——纯自消耗的天花板由真实锚点与外部真值决定。合成数据能「重新组织、放大、验证」已有信息，却难以凭空创造关于真实世界的新知识。
3. **教师偏见与能力继承。** 蒸馏会继承教师的错误、偏见与安全缺陷；多模型共识可缓解但无法消除。
4. **多样性度量仍粗糙。** 词汇/n-gram、困惑度、JS 散度、MMD、潜空间覆盖率各有盲区，缺乏能可靠预警崩溃的统一指标。
5. **仿真到现实的鸿沟（sim2real）。** 世界模型合成的视觉/动作数据受物理一致性、长时程漂移、域差影响，自动驾驶与机器人尤甚。
6. **安全护栏被合成数据侵蚀。** 单源、无锚点的合成微调可能在表层质量不降的情况下削弱对齐（见第 3.5 节），需要专门的安全回归测试。
7. **法律与溯源不确定性。** 「用受版权数据训练的模型产出的合成数据是否侵权」在不同法域仍分化；数据溯源（生成器/验证器/种子/分级）正从最佳实践走向合规刚需。
8. **评测可信度。** 合成时代的能力宣称必须配套防污染基准与披露规范，否则「分数通胀」会误导整个领域的资源配置。

---

## 11. 工程落地 Checklist

一个「抗崩溃」的生产级合成数据管道（对应图 2「抗崩溃合成数据工厂」）：

**① 种子与锚点**
- [ ] 以可获得的最高质量真实数据为种子，积极清洗、去重（SemDeDup 语义去重）；
- [ ] 明确保留 T0/T1 人类锚点，**只增不换**，程序化执行「最大合成比例」；
- [ ] 为每个样本打溯源标签：人工 / 人工编辑 / 合成（附生成器 ID）/ 未知。

**② 多源、过量生成**
- [ ] 跨模型家族生成（开源 + 商业 API），避免单一系列刻印偏见；
- [ ] 温度/角色/系统提示调度，低温保事实、高温增多样；
- [ ] 过量生成目标量的 3–5 倍，供后续无情过滤。

**③ 多层验证（验证比生成容易）**
- [ ] 格式/schema 校验（廉价、确定性）；
- [ ] 可执行验证：代码跑单测、数学对答案（SymPy/规则）、证明过 Lean；
- [ ] 跨模型一致性 / 批评模型拒无据内容；
- [ ] 近重复与语义去重、统计异常点检测；
- [ ] 人工只抽检被标记样本 + 固定比例随机样本。

**④ 分级混合与精算配比**
- [ ] 按质量分级（T0 人类 → T3 批量合成）加权采样；
- [ ] 依据修正缩放定律控制合成 token 在收益拐点内（参考：收益约 300B 后递减），并与训练算力联合优化；
- [ ] 数据受限时重复 epoch 控制在约 4 以内。

**⑤ 持续监控与防污染**
- [ ] 逐批追踪 JS 散度/MMD、潜空间覆盖率、词汇增长曲线，设漂移告警；
- [ ] 在纯真实 holdout 上做下游回归（holdout 降而合成基准稳 = 崩溃信号）；
- [ ] 维护时间切分的新鲜评测（LiveCodeBench/LiveBench 式），披露 cutoff 与生成器。

---

## 12. 结语

合成数据不是绕过数据墙的「作弊码」，而是一门需要锚点、验证与精算的工程学科。2022–2026 年的研究已经把两条看似矛盾的结论统一起来：

- 一方面，**无锚点的递归自消耗确实会让模型崩溃**，这是有 *Nature* 论文和统计理论支撑的真实风险；
- 另一方面，**只要保留真实锚点、累积而非替换、多源生成、外加可执行验证，崩溃并非必然**，合成数据还遵循可量化的修正缩放定律，能在拐点内安全地延续 scaling。

真正决定上限的，始终是循环中**真实信息与外部真值的注入量**。在人类文本存量可见尽头的时代，最健康的范式不是「模型吃自己」，而是「模型与真实世界持续闭环」——可执行的环境、可验证的任务、仿真器与人类反馈共同充当那个永不枯竭的锚点。这也正是合成数据与世界模型、智能体、RLVR 在 2026 年深度合流的根本原因。

---

## 13. 参考文献

> 说明：arXiv 编号均经 arxiv.org 摘要页 `citation_title / citation_author / citation_date` 元数据逐一核实（含主代理亲自抽查 11 个关键/最新编号）；*Nature* 论文经 DOI/Crossref 核实。厂商技术报告中的自报数据，正文已标注【官方口径】；第三方估算与媒体报道亦分别标注。

### 一、数据枯竭背景

1. Sevilla, Villalobos, et al. **Will We Run Out of Data? Limits of LLM Scaling Based on Human-Generated Data.** arXiv:2211.04325（2022-11，Epoch AI）. https://epoch.ai/publications/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data
2. Common Crawl. Monthly snapshots & blog（单月约 21 亿网页 / 355–364 TiB）. https://commoncrawl.org/blog
3. Epoch AI. Compute and data stock estimates（人类公开文本有效存量约 300T token，2026–2032 耗尽）. https://epoch.ai

### 二、模型崩溃理论

4. Shumailov, Shumaylov, et al. **The Curse of Recursion: Training on Generated Data Makes Models Forget.** arXiv:2305.17493（2023-05，2024-04 更新）.
5. Shumailov, et al. **AI Models Collapse When Trained on Recursively Generated Data.** *Nature* 631(8022), 755–759（2024-07）. DOI: 10.1038/s41586-024-07566-y. https://www.nature.com/articles/s41586-024-07566-y
6. Gerstgrasser, Schaeffer, et al. **Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.** arXiv:2404.01413（2024-04）.
7. Bertrand, Bose, et al. **On the Stability of Iterative Retraining of Generative Models on Their Own Data.** arXiv:2310.00429（2023-09，2024-04 更新）.
8. Seddik, Chen, et al. **How Bad Is Training on Synthetic Data? A Statistical Analysis of Language Model Collapse.** arXiv:2404.05090（2024-04）.
9. Gillman, Freeman, et al. **Self-Correcting Self-Consuming Loops for Generative Model Training.** arXiv:2402.07087（2024-02，2024-06 更新）.

### 三、修正缩放定律与数据受限训练

10. Qin, Dong, et al. **Scaling Laws of Synthetic Data for Language Models.** arXiv:2503.19551（2025-03，2025-10 更新）.
11. Goyal, Maini, et al. **Scaling Laws for Data Filtering — Data Curation Cannot Be Compute Agnostic.** arXiv:2404.07177（2024-04）.
12. Muennighoff, Rush, et al. **Scaling Data-Constrained Language Models.** arXiv:2305.16264（2023-05，2025-06 更新）.

### 四、自改进 / 自奖励 / 深度自演化

13. Huang, Gu, et al. **Large Language Models Can Self-Improve.** arXiv:2210.11610（2022-10）.
14. Zelikman, Wu, et al. **STaR: Bootstrapping Reasoning with Reasoning.** arXiv:2203.14465（2022-03，2022-05 更新）.
15. Yuan, Yuan, et al. **Scaling Relationship on Learning Mathematical Reasoning with Large Language Models (RFT).** arXiv:2308.01825（2023-08，2023-09 更新）.
16. Yuan, Pang, et al. **Self-Rewarding Language Models.** arXiv:2401.10020（2024-01，2025-03 更新）.
17. Chen, Deng, et al. **Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (SPIN).** arXiv:2401.01335（2024-01，2024-06 更新）.
18. Guan, Zhang, et al. **rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking.** arXiv:2501.04519（2025-01）.

### 五、指令合成与合成教科书

19. Wang, Kordi, et al. **Self-Instruct: Aligning Language Models with Self-Generated Instructions.** arXiv:2212.10560（2022-12，2023-05 更新）.
20. Xu, Sun, et al. **WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions (Evol-Instruct).** arXiv:2304.12244（2023-04，2025-05 更新）.
21. Xu, Jiang, et al. **Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing.** arXiv:2406.08464（2024-06，2024-10 更新）.
22. Gunasekar, Zhang, et al. **Textbooks Are All You Need (phi-1).** arXiv:2306.11644（2023-06，2023-10 更新）.
23. Li, Bubeck, et al. **Textbooks Are All You Need II: phi-1.5 Technical Report.** arXiv:2309.05463（2023-09）.

### 六、教师蒸馏

24. Mukherjee, Mitra, et al. **Orca: Progressive Learning from Complex Explanation Traces of GPT-4.** arXiv:2306.02707（2023-06）.
25. Mitra, Del Corro, et al. **Orca 2: Teaching Small Language Models How to Reason.** arXiv:2311.11045（2023-11）.
26. DeepSeek-AI, Guo, et al. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.** arXiv:2501.12948（2025-01，2026-01 更新）.

### 七、可执行验证、质量过滤与去重

27. Chen, Zhang, et al. **CodeT: Code Generation with Generated Tests.** arXiv:2207.10397（2022-07，2022-11 更新）.
28. Shao, Wang, et al. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO).** arXiv:2402.03300（2024-02，2024-04 更新）.
29. Li, Fang, et al. **DataComp-LM (DCLM): In Search of the Next Generation of Training Sets for Language Models.** arXiv:2406.11794（2024-06，2025-04 更新）.
30. Gadre, Ilharco, et al. **DataComp: In Search of the Next Generation of Multimodal Datasets.** arXiv:2304.14108（2023-04，2023-10 更新）.
31. Abbas, Tirumala, et al. **SemDeDup: Data-Efficient Learning at Web-Scale Through Semantic Deduplication.** arXiv:2303.09540（2023-03）.

### 八、世界模型 / 仿真合成数据

32. Bruce, Dennis, et al. **Genie: Generative Interactive Environments.** arXiv:2402.15391（2024-02）.
33. Valevski, Leviathan, et al. **Diffusion Models Are Real-Time Game Engines (GameNGen).** arXiv:2408.14837（2024-08，2025-04 更新）.
34. Wang, Zhu, et al. **DriveDreamer: Towards Real-World-Driven World Models for Autonomous Driving.** arXiv:2309.09777（2023-09，2023-11 更新）.
35. Gao, Chen, et al. **MagicDrive: Street View Generation with Diverse 3D Geometry Control.** arXiv:2310.02601（2023-10，2024-05 更新）.
36. Wang, Ling, et al. **GenSim: Generating Robotic Simulation Tasks via Large Language Models.** arXiv:2310.01361（2023-10，2024-01 更新）.

### 九、评测污染与可信评测

37. Deng, Zhao, et al. **Investigating Data Contamination in Modern Benchmarks for Large Language Models.** arXiv:2311.09783（2023-11）.
38. Jain, Han, et al. **LiveCodeBench: Holistic and Contamination-Free Evaluation of Large Language Models for Code.** arXiv:2403.07974（2024-03，2024-06 更新）.
39. White, Dooley, et al. **LiveBench: A Challenging, Contamination-Limited LLM Benchmark.** arXiv:2406.19314（2024-06，2025-04 更新）.

### 十、前沿模型技术报告（官方口径信源）

40. Grattafiori, Dubey, et al. **The Llama 3 Herd of Models.** arXiv:2407.21783（2024-07）.
41. Abdin, et al. **Phi-3 Technical Report.** arXiv:2404.14219（2024-04）.
42. Abdin, Bubeck, Eldan, et al. **Phi-4 Technical Report.** arXiv:2412.08905（2024-12）.
43. NVIDIA. **Nemotron-4 340B Technical Report.** arXiv:2406.11704（2024-06）；NVIDIA Developer Blog, 2024-08-16（对齐数据 98% 合成）.
44. Kimi Team, et al. **Kimi K2: Open Agentic Intelligence.** arXiv:2507.20534（2025-07）.
45. DeepSeek-AI, et al. **DeepSeek-V3 Technical Report.** arXiv:2412.19437（2024-12）.
46. Qwen Team. **Qwen3 Technical Report.** arXiv:2505.09388（2025-05）.
47. Google DeepMind. **Gemini 1.5** arXiv:2403.05530（2024-03）；**Gemini 2.5** arXiv:2507.06261（2025-07，2025-12 更新）.
48. 北京智源人工智能研究院. **OpenSeek 开源数据与合成方案**（>4TB 中英文数据；IndustryCorpus 1.0）. 智源社区，2025-02-18.
49. NVIDIA. **Cosmos World Foundation Models** 技术博客与平台（2025-01 起；2026-07 下载量破 1000 万）. https://developer.nvidia.com/cosmos

### 十一、法律与治理

50. *The New York Times Co. v. Microsoft Corp. & OpenAI*, S.D.N.Y. No. 1:23-cv-11195（2023-12 起诉）.
51. *Bartz et al. v. Anthropic*, N.D. Cal.（Alsup, J.，2025-06 分拆裁决；2025-08 达成 15 亿美元和解）.
52. *Getty Images v. Stability AI*, [2025] EWHC 2863 (Ch)（英国高等法院，2025-11）.
53. European Commission, AI Office. **GPAI Training Data Summary Template & transparency obligations**（2025-07 发布，2025-08-02 生效）.
54. 中国发改委、国家数据局、财政部、人社部. **《关于促进数据标注产业高质量发展的实施意见》**（2025-01-13）.
55. California State Legislature. **AB 2013**（合成数据使用披露，2026-01-01 生效）.

*报告完成日期：2026-10-19。*
