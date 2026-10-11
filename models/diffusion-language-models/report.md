# 扩散语言模型（DLM）前沿技术报告

> 当 Transformer 不再只从左往右写：系统梳理 2015–2026 年扩散语言模型（Diffusion Language Models）的数学原理、四条技术路线、Block-AR 工程外壳、采样器与推理压缩、后训练方法，以及 LLaDA、Mercury、Dream、Seed Diffusion、Cola DLM、DiffusionGemma、Nemotron 等前沿系统的实战数据与能力边界。

📅 **2026-10-11** ｜ **信源**：37 篇 arXiv 原始论文（编号经 arXiv 官方 API 逐一核实标题、作者与日期，含 2026 年 10 月最新在版论文）+ 各模型官方技术报告 / GitHub / 产品页交叉验证 ｜ **时间锚点**：2026-10-11

---

## 目录

1. [选题背景：为什么现在必须关注扩散语言模型](#1-选题背景为什么现在必须关注扩散语言模型)
2. [范式对比：自回归 vs 扩散](#2-范式对比自回归-vs-扩散)
3. [数学基础：扩散如何生成离散文本](#3-数学基础扩散如何生成离散文本)
4. [技术路线分类学：四条路线与共同外壳](#4-技术路线分类学四条路线与共同外壳)
5. [Block-AR：融合两端的工程外壳](#5-block-ar融合两端的工程外壳)
6. [采样器与推理压缩：把"最多 48 步"压到平均 12 步](#6-采样器与推理压缩把最多-48-步压到平均-12-步)
7. [后训练：SFT 之外，扩散模型怎么做偏好对齐与 RL](#7-后训练sft-之外扩散模型怎么做偏好对齐与-rl)
8. [前沿系统逐一剖析](#8-前沿系统逐一剖析)
9. [能力与局限：为什么扩散生成方式带来新能力也带来新毛病](#9-能力与局限为什么扩散生成方式带来新能力也带来新毛病)
10. [趋势判断：2026 年底之后看什么](#10-趋势判断2026-年底之后看什么)
11. [参考资料](#11-参考资料)

---

## 1. 选题背景：为什么现在必须关注扩散语言模型

### 1.1 选题流程与信源

本报告按照"知乎热榜 → 专题检索 → 全网信源"三层流程选题，时间锚点 2026-10-11：

1. **知乎热榜**：知乎官方榜单页（zhihu.com/billboard）对非登录爬虫返回 robots 禁止，本报告遵守 robots 协议，改用第三方聚合站 tophub.today 抓取当日热榜（存档于 `outputs/web-crawler/zhihu-hot-tophub/page.md`）。AI 相关话题在榜，舆论关注点集中在"下一代模型架构是否还有非 Transformer/非自回归的机会"。
2. **专题检索**：CSDN 于 2026-10-09 发布的年度盘点《2026 十大 LLM 研究突破》将字节跳动 Seed 团队的 **Cola DLM（连续潜空间扩散语言模型）列为"架构革命"方向首位**，与线性/高效注意力、状态空间模型等并列为自回归 Transformer 之外最受关注的候选架构。
3. **全网信源 / arXiv 核实**：2025 年 2 月 LLaDA-8B 首次证明扩散语言模型可以从零训练到 8B 并与 LLaMA3-8B 对标；2025 年 6 月 Inception Labs 的商用扩散模型 Mercury 以最高 1109 tokens/s 登上代码补全速度榜首；2025 年下半年 Dream 7B、Seed Diffusion 相继开源；2025 年 12 月 LLaDA2.0 把规模推到 100B；2026 年 5–8 月字节 Cola DLM、Google DeepMind DiffusionGemma、NVIDIA Nemotron-Labs 接连入场。arXiv 上 2026 年 10 月仍每周都有 DLM 新论文（本报告引用的最新两篇 2610.10786、2610.09416 上线于 2026-10-07，距截稿仅 4 天）。

### 1.2 知识库去重

经盘点本地 knowledge-base 仓库，`models/` 目录已覆盖 post-training、LLM RL 算法、Agentic RL、位置编码、JEV 决策等方向，`agents/` 目录覆盖 Harness-RL、自进化 Agent、Loop Engineering 等，**无任何一篇涉及扩散语言模型 / 非自回归生成范式**。本报告填补该空缺。

### 1.3 一句话定义

> **扩散语言模型（Diffusion Language Model, DLM）**：把文本生成建模为一个"从噪声到文本"的迭代去噪过程——模型先面对一整条（或一大块）充满噪声/掩码的序列，然后在数十步内**并行地、反复地**修正所有位置的 token，直到整段文本成形。它与 GPT 式"从左到右逐 token 生成"的自回归（AR）范式形成鲜明对照。

需要先澄清一个常见误解：DLM 不是把图像扩散模型（Stable Diffusion）直接套到文字上。文字是离散符号，直接加高斯噪声没有数学意义。过去五年学界的主要工作，正是在解决"扩散如何与离散词表共存"这一核心难题，并由此分化出多条技术路线（见第 4 节）。

---

## 2. 范式对比：自回归 vs 扩散

> 🖼️ **配图 1（见 HTML 版）**：自回归串行解码与扩散并行去噪的范式对比 —— `diagrams/1-paradigm-compare.html`

### 2.1 生成方式的根本差异

**自回归模型（AR）** 依据链式法则分解序列概率，逐 token 采样：

```
p(x₁,ₓ₂,…,x_L) = Πᵢ p(xᵢ | x₁,…,xᵢ₋₁)

生成：x₁ → x₂ → x₃ → … → x_L     （L 次串行前向，每次只产出 1 个 token）
```

**扩散模型** 则学习一个逆向马尔可夫链：从一个简单的初始分布（全掩码 / 均匀噪声 / 高斯噪声）出发，经过 T 步去噪恢复出数据分布：

```
前向（加噪，训练时人为定义）：x₀ → x₁ → … → x_T（纯噪声）
反向（学习得到，生成时使用）：  x_T → x_T₋₁ → … → x₀

生成：每一步对所有 L 个位置同时做一次预测与修正（T 次前向，每次产出/修正一整块）
```

当 T ≪ L 时（例如生成长度 1024 的文本，用 12–64 步去噪），扩散模型的前向次数远少于 AR，这就是其速度优势的根源。代价是：每个去噪步都是对全序列的一次"精算"，单步计算量更大，且生成质量依赖去噪步数与采样器设计。

### 2.2 六个维度的系统对比

| 维度 | 自回归（GPT 式） | 扩散（DLM） |
|---|---|---|
| **生成顺序** | 严格从左到右，token 间串行 | 无固定顺序，整块并行、反复修正 |
| **前向次数** | ≈ 输出长度 L（每前向 1 token） | 去噪步数 T（每前向修正一整块），T 通常 12–64 |
| **注意力** | 因果（causal）掩码 | 双向（bidirectional）注意力，可看到块内全部位置 |
| **KV Cache** | 天然适配，增量解码成熟 | 块内无法直接复用；需 Block-AR 等工程设计（第 5 节） |
| **上下文依赖** | 左侧词决定右侧词，无法回看修改 | 全局双向约束，可"先写个草稿再整体润色"，支持自校正 |
| **可控性** | 需外挂引导/提示工程 | 可在任意中间步对任意位置施加约束（填空、infilling、结构控制更自然） |
| **计算分配** | 每个 token 等算力 | 可按难度自适应分配步数（难样本多算，简单样本少算） |
| **成熟度（2026）** | 绝对主流，生态完备 | 已商用验证（Mercury/DiffusionGemma），但绝对能力仍略落后于同源 AR |

### 2.3 一个直觉类比

- **AR 模型像"不打草稿的作家"**：从第一个字写到最后一个字，一旦落笔（采样出某 token）就不能改。写得越快越受限于"手速"（内存带宽：每生成一个 token 都要把全部模型权重/KV 读一遍）。
- **扩散模型像"反复修改的画家"**：先铺一整块模糊的底色，然后退后几步看整体，每一步同时调整画面各处，十几轮后成稿。它的"手速"是并行的，时间花在"看整体"（计算量）而不是"逐笔搬运"（内存搬运）。

这也解释了为什么 DLM 在**单请求低延迟**场景优势最大：AR 单请求解码是典型的内存带宽受限任务，GPU 算力大量闲置；扩散每步做整块双向计算，把执行推向计算受限区间，喂饱了 GPU。但在高并发批处理时，AR 的内存瓶颈被大 batch 摊薄，扩散每 token 更高的 FLOPs 反而可能成为劣势（见 9.3 节 DiffusionGemma 的实测拐点）。

---

## 3. 数学基础：扩散如何生成离散文本

**导读：** 本节给出尽量精简但完整的数学框架。核心结论：图像扩散的"分数匹配（score matching）"在离散空间没有直接对应物，学界用两条思路解决——把状态空间做成连续嵌入再扩散（路线③），或者直接在离散状态上构造转移核并优化 ELBO（路线①②）。MD4/MDLM 的重要贡献是证明掩码扩散的连续时间 ELBO 可以化简为一个极其简单的加权交叉熵形式。

### 3.1 前向加噪过程（人为定义，无需学习）

设数据序列为 x₀ = (x₀¹,…,x₀ᴸ)，每个位置取词表 V 中的一个类别。前向过程把数据逐渐摧毁为纯噪声：

- **连续域（图像）**：q(xₜ|xₛ) = N(xₜ; αₜxₛ, σₜ²I)，即随时间推移信噪比按 schedule 下降。
- **离散域（文本）**：常见三种核（kernel）：
  - **掩码（吸收态）核**：每个 token 以随时间增长的概率被替换为 [MASK]，最终全部为 [MASK]。这是 LLaDA / Dream / SEDD / MD4 / DiffusionGemma 共用的选择，最简单稳定。
  - **多项式（multinomial）核**：token 以小概率被替换成词表中**均匀随机**的另一个词，最终趋近均匀分布（D3PM、Argmax Flows；DiffusionGemma 报告将其归为另一参数化分支）。
  - **CTMC（连续时间马尔可夫链）**：在连续时间上定义类别间的随机跳转率矩阵（rate matrix），D3PM（2021）即引入；2026 年的 Flow LM 可视为其连续状态版延伸。

以掩码核为例，t∈[0,1]，定义掩码比例随时间单调增，记为 m(t)（如 m(t)=t 或余弦 schedule），则：

```
q(xₜ | x₀)：每个位置以概率 m(t) 被置为 [MASK]，否则保持 x₀
```

### 3.2 反向去噪过程（模型需要学习的部分）

模型 pθ(xₛ|xₜ)（s<t，即往数据方向走一步）需要在给定当前带噪序列 xₜ 与时间 t 时，预测每个位置的干净 token。Transformer 输入"部分掩码的序列 + 时间步嵌入 + 条件 c（指令等）"，输出每个位置在词表上的分类分布。注意力是**双向**的：被预测的位置可以互相参考。

训练目标是数据对数似然的变分下界（ELBO）。MD4/MDLM（Shi et al., 2024）证明：对于吸收态扩散，连续时间 ELBO 可化简为

```
L = E_{t∼U[0,1], q(xₜ|x₀)} [ w(t) · CE( pθ(·|xₜ,t), x₀ ) ]
```

即**只是一个时间加权的交叉熵积分**——每个训练样本先随机采样噪声水平 t，按 m(t) 随机掩码，再让模型预测所有被掩码位置的原 token。这与 BERT 的 MLM 形式接近，但关键区别在于：① 掩码比例覆盖整个 [0,1] 而非固定 15%；② 它对应一个有严格变分解释、可生成、可计算似然的生成模型，而非判别式预训练任务。

"分数"在离散空间的对应物是数据分布的**比率**：SEDD（Lou/Meng/Ermon, 2023）提出 score entropy，通过估计 log p(data)/p(reference) 的比率构造去噪核，首次让离散扩散在同等规模下生成困惑度比此前语言扩散范式降低 25–75%，并超越 GPT-2，奠定了"离散扩散真的能用"的早期信心。

### 3.3 连续嵌入路线的数学

路线③代表 Diffusion-LM（Li et al., 2022）先把 token 映射到连续向量，对向量加高斯噪声、做连续扩散，最后把向量映射回最近的词嵌入（rounding）。其 ELBO 与图像扩散相同（高斯转移核 + 分数/速度参数化），难点在最后一跳不可微映射与语义保证。SED、CDCD 等后续工作改进 rounding 与自条件（self-conditioning）策略，但这条路线在大规模下基本被离散路线超越。2026 年以 Cola DLM 为代表的工作改为在**层次化潜空间**而非词嵌入空间做连续扩散（见 4.3、8.6 节）。

### 3.4 生成时：反向核、采样顺序与引导

生成从全掩码开始，按 schedule 逐步揭示/替换 token，常见策略：

- **置信度优先（unconfident-first / semi-arbitrary）**：每步先确定模型最有信心的一批位置，其余保持掩码；
- **随机顺序**：每步随机选一批位置填充；
- **DMax / 激进并行**：每步尽量多填，换取最少前向次数；
- **置信度 + 温度退火 + 熵阈值早停**：DiffusionGemma 采用温度 0.8→0.4 退火、熵低于 0.005 的位置提前锁定，平均 12 步、最多 48 步完成一个 256-token 块。

条件生成可套用 classifier-free guidance 的掩码扩散版本（LLaDA 给出了具体公式）；Diffusion-of-Thought（DoT, 2024）则把去噪过程本身当作"思维过程"，初步展示了扩散推理模型的自校正能力与自我一致性（self-consistency）投票潜力。

---

## 4. 技术路线分类学：四条路线与共同外壳

十年演进脉络见配图 2：2015 年 Sohl-Dickstein 等人的奠基论文首次提出用扩散概率模型做离散数据；2021 年 D3PM、Argmax Flows 建立离散扩散的现代形式；2022 年 Diffusion-LM 开启连续嵌入路线；2023 年 SEDD 首次在语言建模上追平 GPT-2；2024 年 MD4 简化目标、LLaDA 把规模拉到 8B；2025 年 Block Diffusion、Mercury、Dream、Seed Diffusion、LLaDA2.0(100B) 完成工程与规模验证；2026 年 Cola DLM、iLLaDA(12T tokens)、DiffusionGemma、Nemotron-Labs 把竞争推向"潜空间 / 热启动转换 / 三模统一"。

> 🖼️ **配图 2（见 HTML 版）**：2015–2026 扩散语言模型十年演进时间线 —— `diagrams/2-evolution-timeline.html`

下面按"状态空间在哪里扩散"把十年工作归为四条建模范式，加上所有大规模系统共用的 Block-AR 工程外壳（第 5 节展开）：

> 🖼️ **配图 3（见 HTML 版）**：DLM 四路线 × Block-AR 外壳 × 采样器的分类学 —— `diagrams/3-taxonomy.html`

### 4.1 路线①：离散掩码扩散（Masked / Absorbing Diffusion）

- **状态空间**：词表 ∪ {[MASK]}，前向只做"token → [MASK]"。
- **反向**：双向 Transformer 预测被掩码位置的原 token。
- **目标**：加权交叉熵（MD4 化简形式）。
- **代表**：SEDD、MD4、LLaDA / LLaDA1.5 / LLaDA2.0、Dream 7B、Seed Diffusion、iLLaDA、DiffusionGemma。
- **优点**：目标最简单、训练最稳、与现有 Transformer 工具链兼容性最好、容易从 AR 模型热启动转换；**缺点**：纯全序列掩码扩散不支持 KV cache、只能定长生成，需 Block-AR 补救。

这是 2025–2026 年绝对主流的路线，几乎所有大规模/商用系统都在此列。

### 4.2 路线②：多项式 / CTMC 扩散

- **状态空间**：词表本身（无吸收态），前向按概率把 token 替换为随机词，极限为均匀分布；连续时间版即 CTMC。
- **代表**：D3PM（2021）、Argmax Flows（2021）、CDCD（2022）。
- **现状**：理论优雅（保留全部 token 的类别概率流），但训练方差大、大规模效果长期不如掩码路线。2025–2026 年其思想以两种方式复活：一是作为部分系统的辅助参数化；二是 NVIDIA Nemotron-Labs 等在联合目标中把"扩散项"与 AR 项一起训练（见 8.7 节）。

### 4.3 路线③：连续嵌入扩散（Continuous Embedding Diffusion）

- **状态空间**：词嵌入向量空间，前向加高斯噪声。
- **代表**：Diffusion-LM（2022）、SED（2022）、CDCD（2022，兼具离散视角）。
- **优点**：可直接复用图像扩散的全套连续数学，梯度引导的可控生成非常自然（Diffusion-LM 在句法结构等细粒度控制上显著领先）。
- **致命难点**：最后一步 embedding→token 的 hard rounding 不可微且易错配；连续空间里"靠近某词向量"并不保证语义正确。该路线停留在小规模研究阶段。

### 4.4 路线④：连续潜空间 / 流匹配扩散（Latent Diffusion + Flow Matching）

2026 年出现的新分支，以字节 Seed 的 **Cola DLM**（arXiv 2605.06548，2026-05）为代表：

1. 先用一个 **Text VAE** 学会稳定的"文本 ↔ 连续潜变量"映射（解决路线③的 rounding 痛点：映射是学出来的、带解码器的，而非最近邻硬匹配）；
2. 再用一个**块因果 DiT**（block-causal Diffusion Transformer）在连续潜空间建模全局语义先验，训练采用 **flow matching**；
3. 生成时先在潜空间完成"全局语义运输（latent prior transport）"，再由条件解码器落成文本。

论文的关键观点：从统一的马尔可夫路径视角看，Cola DLM 的扩散过程搬运的是**全局语义先验**而非 token 级观测，从而把"全局语义组织"与"局部文字实现"解耦。在约 2B 参数、严格匹配的 AR 与 LLaDA 基线、最高约 2000 EFLOPs 的 scaling 曲线上，其验证了连续潜空间路线的良好扩展性，并天然便于向其他连续模态（图像/音频）统一延伸。模型以 Apache 2.0 开源，并提供 OpenAI 兼容 API。

### 4.5 路线⑤（共同工程外壳）：Block-AR / 块级自回归

严格说这不是独立的建模范式，而是所有大规模 DLM 共用的工程形态（第 5 节专述）：**因果编码器逐块处理已确定的干净前缀（可用 KV cache），双向扩散解码器在当前块内并行去噪；块与块之间自回归推进。** 块长 = 1 时退化为纯 AR，块长 = 全文时退化为纯扩散——它是一条连续可调的插值光谱（Block Diffusion, arXiv 2503.09573 给出了形式化与训练配方）。

---

## 5. Block-AR：融合两端的工程外壳

> 🖼️ **配图 4-A（见 HTML 版）**：Block-AR 推理流水线 —— `diagrams/4-frontier-systems.html`

纯扩散的两个工程死穴：① 不能用 KV cache，长文本推理反复全序列双向计算；② 只支持定长生成。2025 年 3 月的 **Block Diffusion**（Cornell/斯坦福等，arXiv 2503.09573）给出了统一框架：

```
已确定前缀（干净，因果注意力，KV cache 可复用）
        │
        ▼
[ 当前噪声块（block，典型 64–256 token） ]
   双向注意力并行去噪，块内位置互相可见
        │
        ▼
   左边缘条件（left conditioning）条件下锁定整块
        │
        ▼
   前一块的 KV 进入 cache，推进到下一块
```

关键性质：

1. **连续插值**：块长 b=1 即纯 AR；b=L 即纯扩散。调大块长 → 并行度高、延迟低但单步重；调小块长 → 接近 AR 的似然质量与灵活性。
2. **支持任意长度生成**：像 AR 一样逐块推进，突破定长限制。
3. **数据驱动的噪声 schedule**：训练时按经验分布选择每个样本的起始掩码比例与时间分布，降低梯度方差（这是 Block Diffusion 训练配方的核心技巧之一，配合方差估计与最优 MC 预算分配）。
4. **双注意域**：对已确定前缀用因果掩码 + KV cache；对当前块用双向掩码。Cola DLM 的"块因果 DiT"、NVIDIA TwoTower 的"冻结 AR 上下文塔 + 双向去噪塔"本质上都是这一结构的不同实现。

正是这个外壳让扩散模型进入了可部署区间：DiffusionGemma 以 256 token 为一个 canvas，Mercury、Dream、Seed、LLaDA2.0 全部采用块级生成。

### 5.1 架构趋同：从"替代 AR"到"复用 AR 资产"

早期 DLM（SEDD、LLaDA 初版）从零训练双向模型。2025 下半年起的明显趋势是**尽可能复用已有的 AR 预训练资产**：

| 方法 | 如何复用 AR 资产 | 额外训练量 |
|---|---|---|
| Dream 7B | 用 AR-LLM 初始化，再做离散扩散训练；上下文自适应 token 级噪声重排 | 未披露完整比例 |
| LLaDA2.0 | 从 AR-MoE 系统转换，三阶段块级 WSD：暖身（块长渐增）→ 全序列扩散（稳定）→ 回到小块扩散（衰减） | 转换训练而非从头训 |
| DiffusionGemma | 从 Gemma 4（25.2B 总参 / 3.8B 激活 MoE）SFT 转换 | **< 起始模型训练 token 预算的 10%** |
| Nemotron TwoTower | 基于 Nemotron-3-Nano-30B-A3B（混合 Mamba-Transformer MoE），**冻结** AR 上下文塔，只训去噪塔 | 约 2.1T tokens |
| Nemotron Tri-Mode | 用 AR-扩散联合目标训练同一网络 | 3B/8B/14B 家族 |

> **注意**：Dream 7B 的核心训练技巧是"AR 初始化 + 上下文自适应 token 级噪声重排（context-adaptive token-level noise rescheduling）"；DiffusionGemma 的 <10% token 预算是其技术报告摘要明示的数字。这些"低转换成本"是 2026 年扩散范式获得工业界入场券的直接原因——不需要赌上整条预训练管线。

### 5.2 注意：纯双向从零训练仍未死

与"热启动"路线并行，人民大学/清华等团队的 **iLLaDA**（arXiv 2606.25331，2026-06）坚持全程双向注意力、从头训练掩码扩散目标，预训练规模达 **12T tokens**（+ 25B token 指令数据 12 epoch SFT），并引入变长生成与置信度评分。iLLaDA-Base 较 LLaDA-Base 在 BBH +21.6、ARC-C +14.9；iLLaDA-Instruct 在 MATH +14.5、HumanEval +16.5，并在多项基准上与 Qwen2.5-7B 持平。这说明双向非自回归预训练本身可以走出一条有竞争力的路，热启动只是当前性价比更高的工程选择。

---

## 6. 采样器与推理压缩：把"最多 48 步"压到平均 12 步

DLM 推理成本 ≈ 去噪步数 × 每步全块计算。因此 2025 年下半年起，**采样器与步数压缩**成为最活跃的工程方向，可归入四类：

**① 更激进的并行揭示。** DMax（arXiv 2604.08302，2026-04）主张每步尽量多解掩码；Lookahead Unmasking（2511.05563，2025-11）利用相邻步预测的相关性提前揭示高置信 token；SAID（2606.04974，2026-06）做"脚手架感知"的迭代解码。

**② 时间并行 / 少步蒸馏。** 把原本串行的去噪轨迹改成时间维并行（Parallel-in-Time Sampling, arXiv 2607.00773，2026-07）；Discrete MMD 蒸馏（2603.20155，2026-03）用最大均值差异把多步教师压成少步学生，超越逐 token 蒸馏。

**③ 系统级优化。** Flash-dLLM（2609.26796，2026-09）针对 dLLM 的访问模式重新设计 IO 感知 KV cache 与并行解码，降低显存占用；dVoting（2602.12153，2026-02）加速扩散模型的自洽性投票。

**④ 训练-采样联合优化（影响最大）。** DiffusionGemma 的第二阶段把**奖励优化与采样器蒸馏合并为同一个 RL 过程**（详见第 7 节）：模型被激励"在更少步数、更便宜的采样器下拿高奖励"，训练出对少步解码鲁棒的策略；再配合 canvas=256、温度 0.8→0.4、熵阈值 0.005 的早停规则，实现最多 48 步、平均 12 步完成一块，平均每次前向产出约 20 token，单卡 H100（FP8, batch=1）约 1,500 TPS。

### 6.1 速度数据汇总（各模型自报，注意硬件/协议差异）

| 模型 | 速度（自报或第三方） | 硬件与条件 |
|---|---|---|
| Mercury Coder Mini / Small | **1109 / 737 tokens/s**（Artificial Analysis 独立评测） | H100 |
| Mercury 2（推理型，2026-02） | 约 600 / 547 tokens/s（不同端点实测） | API 实测 |
| Seed Diffusion Preview | **2146 tokens/s**，代码模型速度-质量 Pareto 前沿 | H20 |
| LLaDA2.0-Flash 100B | 375 tokens/s | 8×B200 |
| DiffusionGemma | 约 1,500 tokens/s（~20 TPF）；AR 模式 204、AR+MTP 303 | 单卡 H100，FP8，bs=1 |
| Nemotron TwoTower | 生成吞吐 2.42× AR 基线，保留 98.7% 质量 | 30B-A3B MoE |
| Nemotron Tri-Mode-8B | 每次前向 token 数 6× Qwen3-8B；SPEED-Bench 吞吐 4× | GB200 + SGLang |

> ⚠️ 跨模型 TPS 不能直接排名：硬件代际（H100/H20/B200/GB200）、batch、是否流式首 token、测量口径不同。Seed Diffusion 的 2146 TPS 是代码模型在 H20 上的自报值；DiffusionGemma 的 ~1500 是单请求场景。

### 6.2 质量代价与并发拐点

速度并非免费。以 DiffusionGemma 为例（同源 Gemma 4 MoE，2026-07 报告）：

| 指标 | 扩散模式 | 同源 AR |
|---|---|---|
| AIME（数学竞赛） | 69.1 | 88.3 |
| LiveCodeBench v6 | 69.1 | 77.1 |
| HumanEval | 94.5 | — |
| 平均输出长度 | 近 2× 缩短（更直接） | 基线 |
| 并发 > 32 路 | **AR（含投机解码）反超扩散** | — |

即：扩散模式用前沿推理基准上十几分的能力差，换来了单请求 5–7× 的吞吐与更简洁的输出；且其优势随并发升高而衰减——单请求时 AR 卡在内存带宽墙，高 batch 时 AR 已把同一批 token 的权重读摊薄，扩散每 token 更高的计算量开始吃亏。这是所有想落地 DLM 的团队必须实测的拐点。

---

## 7. 后训练：SFT 之外，扩散模型怎么做偏好对齐与 RL

AR 模型的后训练工具箱（SFT / DPO / PPO / GRPO）不能直接照搬：扩散模型的似然是 ELBO 估计量，蒙特卡洛方差大，偏好对的 log-ratio 噪声会淹没梯度。2025–2026 年形成了三代方法：

**第一代：SFT 直接适配。** LLaDA、Dream 的 Instruct 版本证明，在指令数据上继续做"掩码-预测"训练就能得到可用的指令遵循能力；iLLaDA 在 25B token 指令语料上训练 12 个 epoch。

**第二代：VRPO（方差缩减的偏好优化，LLaDA 1.5，arXiv 2505.19223，2025-05）。** 清华/蚂蚁等团队形式化分析了掩码扩散模型基于 ELBO 的似然估计方差，给出偏好优化梯度的偏差与方差界，并引入两项无偏方差缩减：**最优蒙特卡洛预算分配**与**对偶采样（antithetic sampling）**。效果（相对 SFT 版 LLaDA）：GSM8K +4.7、HumanEval +3.0、MBPP +1.8、IFEval +4.0、Arena-Hard +4.3。这是 DLM 偏好对齐第一个被广泛引用的完整方案。LLaDA2.0 则在更大规模上使用 SFT + DPO 完成对齐。

**第三代：奖励 × 采样器联合 RL（SD·RL，DiffusionGemma，2026-07）。**

> 🖼️ **配图 4-B（见 HTML 版）**：SD·RL 联合训练循环 —— `diagrams/4-frontier-systems.html`

其两阶段流水线为：① SFT 教会热启动模型双向去噪；② 把强化学习与**采样器蒸馏**放进同一个目标——采样器在训练中不断变得更激进（更少步），模型则被奖励信号驱动去适应"少步采样器下的生成"，两者形成**隐式课程（implicit curriculum）**：早期多步、低难度，后期少步、高难度。当生成熵降到阈值以下即提前停止该轨迹。这与 AR 世界"先训模型再投机解码/蒸馏"的串行流程不同，是扩散范式独有的联合优化空间。

此外，2026 年 10 月的新工作还显示"AR 后训练权重可以反哺扩散模型"（arXiv 2610.08108，Enhancing Diffusion Language Models with Autoregressive Post-Training Weights），AR 与扩散的权重/知识迁移正从单向（AR→扩散初始化）走向双向。

---

## 8. 前沿系统逐一剖析

### 8.1 LLaDA 家族（清华 GSAI / 蚂蚁等，开源）

- **LLaDA-8B**（2502.09992，2025-02）：首个在 8B 规模从零预训练（预训练+SFT 范式）并与 LLaMA3-8B 同台竞技的掩码扩散模型。优化严格的似然下界，双向注意力。代表性现象：**缓解反转诅咒（reversal curse）**——在反转诗歌补全任务上超过 GPT-4o，这是双向建模带来的直接红利。
- **LLaDA 1.5**（2505.19223，2025-05）：VRPO 对齐版（见第 7 节）。
- **LLaDA2.0**（2512.15745，2025-12）：把规模推到 **100B 总参**的 MoE 家族（mini 16B / flash 100B），均开源。核心是三阶段**块级 WSD**（Warmup-Stable-Decay）转换训练：暖身阶段块长渐进增大 → 稳定阶段全序列扩散 → 衰减阶段回到小块扩散；强调"知识继承、渐进适配、效率优先"，从 AR 模型系统转换而非从头训。Flash 100B 在 8×B200 上约 375 TPS。
- **iLLaDA**（2606.25331，2026-06）：8B，全程双向、从零、12T tokens 预训练（见 5.2）。权重与代码在 github.com/ML-GSAI/LLaDA。

### 8.2 Mercury / Mercury 2（Inception Labs，商用 API）

- **Mercury Coder**（2506.17298，2025-06）：第一个达到商业规模的扩散语言模型产品线，Transformer 参数化、并行预测多 token。Mini / Small 两档，经 Artificial Analysis 独立评测在 H100 上达 **1109 / 737 TPS**，相对速度优化的前沿模型平均快至 10×，质量可比；在 Copilot Arena 上质量第二、**速度总榜第一**。HumanEval 约 88–90。提供公开 API 与免费 playground。
- **Mercury 2**（2026-02 发布，未检索到独立 arXiv 编号）：官方称首个"推理型（reasoning）扩散语言模型"，把扩散并行生成与长思维链结合；公开 API 实测约 600 / 547 TPS（不同端点/场景）。

> 产业意义：Mercury 证明 DLM 的商业模式成立——按"速度体验"差异化定价（其 API 曾以显著低价提供），切的是代码补全这种对首 token 延迟和吞吐极敏感的场景。

### 8.3 Dream 7B（港大等，开源）

arXiv 2508.15487（2025-08）。发布时"最强开源扩散语言模型"，离散扩散、迭代并行去噪。两大训练要点：**AR-LLM 初始化**与**上下文自适应 token 级噪声重排**。能力亮点：通用/数学/代码全面超越此前 DLM；任意顺序生成、infilling、质量-速度可调；规划能力突出。Base 与 Instruct 双版本开源。后续 **Dream-Coder 7B**（2509.01142，2025-09）专攻代码；2026-10 的 Plan-and-Patch（2610.10786）以 DreamReasoner-8B 为扩散规划器，在 Natural Plan 上无任务特定训练即取得 **53.7% 的计划修复成功率，约为 AR 规划器（27.0%）的两倍**，ALFWorld/TextCraft 上计划生成延迟降低 39–46%——直接印证了扩散"局部重写、保留前后文"的结构化优势（见 9.2）。

### 8.4 Seed Diffusion（字节 Seed，预览版）

arXiv 2508.02193（2025-08）。离散状态扩散的大规模代码模型，主打高速推理：H20 GPU 上 **2,146 tokens/s**，论文称显著快于同期 Mercury 与 Gemini Diffusion，在代码模型速度-质量 Pareto 前沿建立新 SOTA。其叙事明确把 Mercury Coder、Gemini Diffusion 并列为"非顺序并行生成缓解逐 token 解码延迟"的同期证据——也侧面证实 Google 的 **Gemini Diffusion** 是 2025 年工业界的重要参与者（Google 未公开完整技术报告，本报告不引用无法核实的具体数字）。

### 8.5 Cola DLM（字节 Seed，开源）

arXiv 2605.06548（2026-05），见 4.4 节。要点重述：Text VAE + 块因果 DiT + flow matching 的层次化**连续潜空间**扩散；扩散过程是"潜变量先验运输"而非 token 恢复；约 2B 参数、8 个基准、~2000 EFLOPs scaling 实验；Apache 2.0 开源并提供 OpenAI 兼容 API。它代表的是与主流掩码路线不同的"第二增长曲线"：在潜空间获得全局语义建模与连续模态扩展性。

### 8.6 DiffusionGemma（Google，开源权重实验模型）

arXiv 2608.00146（2026-07-31，v1；2026-08 更新）。本报告写作时信息最完整、工程细节最透明的前沿 DLM：

- **基座**：Gemma 4 MoE，25.2B 总参 / 3.8B 激活；扩散微调总 token 预算 **< 原 AR 模型训练量的 10%**。
- **结构**：256 token 一个 canvas 的块扩散；保留基座的思考模式、多模态输入与长上下文能力。
- **训练**：SFT（学双向去噪）→ SD·RL（奖励 × 采样器蒸馏联合，隐式课程，熵降早停）。
- **推理**：最多 48 步、平均 12 步；温度 0.8→0.4；熵阈值 0.005；平均 ~20 token/前向；单卡 H100（FP8, bs=1）约 1,500 TPS（对照：AR 204、AR+MTP 303 TPS）。
- **能力代价**：AIME 69.1（AR 88.3）、LiveCodeBench v6 69.1（AR 77.1）、HumanEval 94.5；输出长度平均缩短近 2×；>32 路并发后 AR 反超；偶发"口吃（stuttering）"式重复。
- **关键彩蛋**：扩散微调后的模型仍能以 AR 方式生成，性能仅小幅下降——这指向**同一权重混合扩散-AR 解码**的可行路径。

### 8.7 Nemotron-Labs（NVIDIA，2026 年 6–7 月，开源）

- **TwoTower**（2606.26493）：把"上下文表示"与"迭代去噪"拆成两座塔——**冻结的 AR 上下文塔**因果处理干净前缀，**可训练的双向扩散去噪塔**通过交叉注意力精修噪声块。基于 Nemotron-3-Nano-30B-A3B（混合 Mamba-Transformer MoE），约 2.1T tokens 训练；保留 AR 基线 **98.7%** 质量，生成墙钟吞吐 **2.42×**。权重在 Hugging Face NVIDIA 合集开放。
- **Tri-Mode**（2607.05722）：单一架构、AR-扩散联合目标训练，可在 **AR / 扩散 / 自投机（self-speculation）** 三模间切换：扩散负责 lookahead 规划，AR 提供从左到右的语言先验；自投机模式下扩散起草、AR 验证，接受率与实机效率均优于 MTP。speed-of-light 分析显示最优采样器下扩散每次前向可比自投机多产出 76.5% 的 token。3B/8B/14B（含 instruct 与视觉语言版）家族，8B 版每次前向产出 token 数为 Qwen3-8B 的 6×，GB200 + SGLang 上 SPEED-Bench 吞吐 4×。

> NVIDIA 两篇工作的共同信号：工业界不再纠结"扩散还是 AR"，而是把两者做成**可按部署条件（并发、延迟目标、模态）动态切换的统一模型**。

### 8.8 其余值得跟踪的 2026 年工作

- **Flow Language Models**（连续状态扩散，如 2610.09416）：去噪全程在连续表示上演化、仅末端解码为 token；在迷宫规划/数独上，少步数区间推理效率高于同规模离散扩散（Maze15 达 95% 准确率少用 36.5% 参数），连续"叠加态"被理论与干预实验证明有利于多候选证据保持。
- **扩散 MLLM**（如 2610.00953"Two Clocks in Diffusion MLLMs"）：发现多模态扩散模型中"答案先于推理过程稳定"的双时钟现象，为扩散推理的可解释性提供新切口。

---

## 9. 能力与局限：为什么扩散生成方式带来新能力也带来新毛病

**导读：** DLM 不是"更快的 GPT"。并行去噪改变了生成的归纳偏置，带来三类新能力与三类新缺陷。

### 9.1 自适应计算与质量-速度旋钮

去噪步数是一个可在推理时连续调节的旋钮：简单请求 8–12 步快速出稿，复杂推理 32–48 步弹性加算。Dream、DiffusionGemma 都明确支持"多算则优、少算则快"。这比 AR"每个 token 固定一次前向"弹性更大——**同一个模型可以按任务难度分配测试时计算量**，与 2025–2026 年可变思维链长度、自适应草稿等"测试时计算动态分配"趋势同频。

### 9.2 双向上下文 → 全局规划与局部修补

双向注意力让模型在生成**之前**就对整段内容的结构建立把握：

- **反转诅咒缓解**：LLaDA-8B 在反转诗歌补全任务上超过 GPT-4o（2502.09992）。
- **任意位置生成 / 填充（infilling）**：给定前后缀填中间，对代码补全（上下文中已有后续代码）、模板写作、翻译对齐极友好。
- **计划修补**：Plan-and-Patch（2610.10786）中扩散规划器只重写计划的失效区域、保留前后步骤，Natural Plan 修复成功率 53.7% vs AR 27.0%；任务训练后 ALFWorld/TextCraft 计划生成延迟降低 39–46%。
- **结构化输出与可控生成**：从 Diffusion-LM 的梯度控制到现代 dLLM 的任意顺序生成，约束解码（固定模板、表格、代码骨架）可以"先定结构、再填内容"。

### 9.3 缺陷一：高并发下的吞吐劣势

如 6.2 节数据：单请求场景扩散 5–7× 的优势随 batch 增大而收窄，DiffusionGemma 实测 **超过 32 路并发时 AR（含投机解码）反超**。原因是单请求时 AR 受内存带宽限制、扩散把任务推向算力受限区间；批处理时权重读取被摊薄，AR 每 token 更低的计算量优势显现。Nemotron 三模模型正是为平掉这个劣势而设计——低并发走扩散、高并发切 AR。

### 9.4 缺陷二："口吃"与收敛失败

扩散解码可能卡在某些位置反复震荡、重复或迟迟不锁定（DiffusionGemma 报告称 stuttering），在 DMax 等高并行揭示策略下更明显。熵阈值早停、置信度采样、lookahead 系列方法都在缓解但未根除。这也是数学/代码尖峰任务仍有明显差距的原因之一（AIME 69.1 vs 88.3）——**双向自校正不是万能的，全局并行决策在需要严格因果链的任务上仍处于追赶位**。

### 9.5 缺陷三：似然评价与工具链不成熟

- 序列似然只能通过 ELBO/MC 估计，PPO/DPO 等依赖精确 logp 的算法适配成本高（VRPO 即为此而生）；
- 定长问题虽被 Block-AR 解决，但块内全双向计算的显存占用随块长平方增长（Flash-dLLM 的 IO 感知设计即针对此）；
- 采样器超参（步数、schedule、温度、阈值）空间大，不同任务需调参；
- 多轮对话的历史上下文处理、prefill 效率仍在快速演进，vLLM/SGLang 原生支持在 2026 年才起步。

### 9.6 能力总览

| 能力维度 | DLM 相对 AR | 说明 |
|---|---|---|
| 单请求吞吐 / 首字延迟 | ✅ 显著优 | 并行块生成；Mercury/Seed/DiffusionGemma 均验证 |
| 自适应计算 | ✅ 优 | 步数旋钮，按难度分配测试时计算 |
| 全局结构 / 规划 / 填充 | ✅ 优 | 双向上下文；infilling、计划修补证据充分 |
| 可控生成 / 约束输出 | ✅ 优 | 任意顺序、先结构后内容 |
| 反转诅咒等双向记忆 | ✅ 优 | LLaDA 已验证 |
| 数学 / 代码尖峰质量 | ⚠️ 略劣 | AIME/LCB 差十几分；HumanEval 已追平（94.5） |
| 高并发吞吐 | ⚠️ 略劣 / 可切换 | >32 并发 AR 反超；三模架构可自动切换 |
| 流畅度（重复 / 口吃） | ⚠️ 略劣 | 收敛失败尚未根除 |
| 似然精度 / 工具链成熟度 | ❌ 劣 | ELBO 方差、推理引擎支持处于早期 |
| 多模态扩展 | 🔜 待验证 | Cola / Flow LM 连续空间天然统一；Nemotron VLM 已出 |

---

## 10. 趋势判断：2026 年底之后看什么

1. **混合化是确定性方向，而非扩散取代 AR。** 最强证据来自架构内部：Block-AR 块长光谱、DiffusionGemma 同权重保留 AR 能力、Nemotron 三模动态切换、AR 后训练权重反哺扩散（2610.08108）。未来的主流前沿模型很可能是"一个权重、多种解码模式"，由路由器按请求延迟目标与实时并发决定走 AR 还是扩散。

2. **潜空间 / 连续状态是第二战场。** Cola DLM 用 Text VAE 绕开词嵌入 hard rounding，Flow LM 证明连续叠加态利于少步推理。连续潜空间的更大诱惑是**多模态统一**：文本、图像、音频本就是连续扩散的主场，文本潜变量扩散为统一生成栈提供了接口。

3. **后训练从"适配 AR 算法"走向"扩散原生联合优化"。** VRPO 解决似然方差，SD·RL 把采样器蒸馏并入 RL 目标。下一步大概率出现扩散原生的可验证奖励长链路 RL（数学/代码/Agent），以及把"步数"作为显式策略变量的 RL  formulation。

4. **Agent 规划是最被看好的应用落点。** 计划需要全局结构、执行中需要局部修补——这正是双向并行生成的结构性优势。Plan-and-Patch 的"修复成功率 2×、延迟 −40%"是早期强信号；2026-10-07 同周出现的 Flow LM 推理论文也集中在规划类任务（迷宫、数独）。

5. **推理系统将围绕"块"重构。** KV cache 布局（Flash-dLLM）、时间并行采样、离散蒸馏、与投机解码的融合（扩散起草 + AR 验证）会成为推理引擎的标准模块；速度评测口径（TPF、TPS、并发拐点、首块延迟）需要行业统一标准，当前各家自报数字不可直接横比。

6. **风险与开放问题。** 尖峰推理能力差距能否靠规模+后训练填平？口吃等收敛病理是否有理论上的根本解法？扩散模型的扩展定律在 100B（LLaDA2.0）之后是否继续成立？这些决定了 DLM 是成为"第二主流架构"还是停留在"高速专用架构"。

### 给不同读者的结论

- **应用开发者**：现在就可以在代码补全、模板/结构化写作、高吞吐客服、草稿生成等场景试用开源 DLM（Dream、LLaDA2.0、Cola、DiffusionGemma、Nemotron），重点实测自己流量结构下的**并发拐点**；长链路严谨推理仍以 AR 前沿模型为主。
- **模型团队**：不必从零豪赌双向预训练（iLLaDA 证明可行但昂贵），AR 热启动转换（<10% token 预算）是当前性价比路线；采样器应纳入训练目标而非事后优化。
- **基础设施团队**：块级 KV cache、时间并行、扩散-投机混合解码是接下来两年的明确工程机会。

---

## 11. 参考资料

> 所有 arXiv 编号均于 2026-10-11 通过 arXiv 官方 API（export.arxiv.org/api）逐一核实标题、作者与日期；标注"官方/网页信源"者为机构发布页面。

**奠基与理论（2015–2024）**

1. Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*, 2015 — arXiv:1503.03585 — https://arxiv.org/abs/1503.03585
2. Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models* (DDPM), 2020 — arXiv:2006.11239 — https://arxiv.org/abs/2006.11239
3. Ghazvininejad et al., *Mask-Predict: Parallel Decoding of Conditional Masked Language Models*, EMNLP 2019 — arXiv:1904.09324 — https://arxiv.org/abs/1904.09324
4. Austin et al., *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM), 2021 — arXiv:2107.03006 — https://arxiv.org/abs/2107.03006
5. Hoogeboom et al., *Argmax Flows and Multinomial Diffusion*, 2021 — arXiv:2102.05379 — https://arxiv.org/abs/2102.05379
6. Savinov et al. (DeepMind), *Step-unrolled Denoising Autoencoders for Text Generation* (SUNDA), ICLR 2022 — arXiv:2112.06749 — https://arxiv.org/abs/2112.06749
7. Li et al., *Diffusion-LM Improves Controllable Text Generation*, NeurIPS 2022 — arXiv:2205.14217 — https://arxiv.org/abs/2205.14217
8. Chen et al., *Self-Conditioned Embedding Diffusion for Text Generation* (SED), 2022 — arXiv:2211.04236 — https://arxiv.org/abs/2211.04236
9. Dieleman et al., *Continuous Diffusion for Categorical Data* (CDCD), 2022 — arXiv:2211.15089 — https://arxiv.org/abs/2211.15089
10. Lou, Meng, Ermon, *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution* (SEDD), 2023 — arXiv:2310.16834 — https://arxiv.org/abs/2310.16834
11. Shi et al., *Simplified and Generalized Masked Diffusion for Discrete Data* (MD4/MDLM), 2024 — arXiv:2406.04329 — https://arxiv.org/abs/2406.04329
12. Ye et al., *Diffusion of Thought: Chain-of-Thought Reasoning in Diffusion Language Models*, 2024 — arXiv:2402.07754 — https://arxiv.org/abs/2402.07754

**大规模化与工程化（2025）**

13. Nie et al., *Large Language Diffusion Models* (LLaDA-8B), 2025-02 — arXiv:2502.09992 — https://arxiv.org/abs/2502.09992
14. Arriola et al., *Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models*, 2025-03 — arXiv:2503.09573 — https://arxiv.org/abs/2503.09573
15. Zhu et al., *LLaDA 1.5: Variance-Reduced Preference Optimization* (VRPO), 2025-05 — arXiv:2505.19223 — https://arxiv.org/abs/2505.19223
16. Inception Labs, *Mercury: Ultra-Fast Language Models Based on Diffusion*, 2025-06 — arXiv:2506.17298 — https://arxiv.org/abs/2506.17298 ；API/playground：https://platform.inceptionlabs.ai/
17. Ye et al., *Dream 7B: Diffusion Large Language Models*, 2025-08 — arXiv:2508.15487 — https://arxiv.org/abs/2508.15487
18. Song et al., *Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference*, 2025-08 — arXiv:2508.02193 — https://arxiv.org/abs/2508.02193
19. Li et al., *A Survey on Diffusion Language Models*, 2025-08 — arXiv:2508.10875 — https://arxiv.org/abs/2508.10875
20. *Dream-Coder 7B: An Open Diffusion Language Model for Code*, 2025-09 — arXiv:2509.01142 — https://arxiv.org/abs/2509.01142
21. *Lookahead Unmasking Elicits Accurate Decoding in Diffusion Language Models*, 2025-11 — arXiv:2511.05563 — https://arxiv.org/abs/2511.05563
22. Bie et al., *LLaDA2.0: Scaling Up Diffusion Language Models to 100B*, 2025-12 — arXiv:2512.15745 — https://arxiv.org/abs/2512.15745

**2026 前沿**

23. *dVoting: Fast Voting for dLLMs*, 2026-02 — arXiv:2602.12153 — https://arxiv.org/abs/2602.12153
24. *Beyond Single Tokens: Distilling Discrete Diffusion Models via Discrete MMD*, 2026-03 — arXiv:2603.20155 — https://arxiv.org/abs/2603.20155
25. *DMax: Aggressive Parallel Decoding for dLLMs*, 2026-04 — arXiv:2604.08302 — https://arxiv.org/abs/2604.08302
26. Guo et al.（字节 Seed）, *Continuous Latent Diffusion Language Model* (Cola DLM), 2026-05 — arXiv:2605.06548 — https://arxiv.org/abs/2605.06548
27. Reda et al. (NVIDIA), *Nemotron-Labs-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context*, 2026-06 — arXiv:2606.26493 — https://arxiv.org/abs/2606.26493 ；权重：https://huggingface.co/collections/nvidia/nemotron-labs-twotower
28. *SAID: Accelerating Diffusion-Based LMs via Scaffold-Aware Iterative Decoding*, 2026-06 — arXiv:2606.04974 — https://arxiv.org/abs/2606.04974
29. Nie et al., *Improved Large Language Diffusion Models* (iLLaDA, 12T tokens), 2026-06 — arXiv:2606.25331 — https://arxiv.org/abs/2606.25331 ；代码/权重：https://github.com/ML-GSAI/LLaDA
30. Fu et al. (NVIDIA), *Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying AR, Diffusion, and Self-Speculation Decoding*, 2026-07 — arXiv:2607.05722 — https://arxiv.org/abs/2607.05722
31. *Accelerating Discrete Diffusion Models with Parallel-in-Time Sampling*, 2026-07 — arXiv:2607.00773 — https://arxiv.org/abs/2607.00773
32. DiffusionGemma Team (Google), *DiffusionGemma Technical Report*, 2026-07/08 — arXiv:2608.00146 — https://arxiv.org/abs/2608.00146
33. *Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs*, 2026-09 — arXiv:2609.26796 — https://arxiv.org/abs/2609.26796
34. Kumar et al., *Plan-and-Patch: Diffusion Language Models for Agentic Planning*, 2026-10-07 — arXiv:2610.10786 — https://arxiv.org/abs/2610.10786
35. Bai et al., *Efficient Reasoning with Flow Language Models*, 2026-09/10 — arXiv:2610.09416 — https://arxiv.org/abs/2610.09416
36. *Enhancing Diffusion Language Models with Autoregressive Post-Training Weights*, 2026-10 — arXiv:2610.08108 — https://arxiv.org/abs/2610.08108
37. *Two Clocks in Diffusion MLLMs: When Answers Stabilize Before Rationales Unfold*, 2026-10 — arXiv:2610.00953 — https://arxiv.org/abs/2610.00953

**官方与产业信源**

38. Inception Labs 官方：Mercury/Mercury 2 产品页与文档 — https://www.inceptionlabs.ai/ ；API：https://platform.inceptionlabs.ai/
39. Google DeepMind, *DiffusionGemma* 发布说明与开放权重（Gemma 开放模型计划；具体权重地址以 Google 官方 Hugging Face / Kaggle 发布页为准，本报告撰写环境无法访问 huggingface.co 核验，故不给出可能失效的直链）
40. 字节跳动 Seed 团队，Cola DLM 项目页 / 仓库（Apache 2.0，OpenAI 兼容 API；具体地址以字节 Seed 官方 GitHub/Hugging Face 发布为准）
41. Artificial Analysis 对 Mercury Coder Mini/Small 的独立吞吐评测（1109/737 TPS，H100） — https://artificialanalysis.is/
42. CSDN 博客《2026 十大 LLM 研究突破盘点》（2026-10-09，将 Cola DLM 列为架构革命首位） — https://blog.csdn.net/qq_56999332/article/details/161073516
43. 知乎热榜 AI 话题（经 tophub.today 聚合页采集，知乎直抓受 robots 限制） — https://tophub.today/n/mproPpoq6O

> **数据口径说明：** ① 各模型 TPS/TPF 数字来自各自论文/技术报告的硬件、精度、batch、上下文设定，跨模型不可直接横比；② DiffusionGemma 的步数/熵阈值/并发拐点等来自其技术报告（2608.00146）正文表格；③ 2610.08108 等 2026 年 10 月极新论文以 arXiv 页面信息为准，结论强度弱于经过同行评审的工作；④ Gemini Diffusion 因无公开技术报告，仅在他文引用语境中提及，不引用无法核实的数字。

---

## 附：与知识库已有报告的关联

- 与《大模型强化学习算法》（post-training / llm-rl-algorithms）：VRPO、SD·RL 是扩散范式下偏好优化的对应物，可对照 PPO/DPO 的似然假设阅读。
- 与《Agentic RL》：Plan-and-Patch 代表扩散模型进入多轮 Agent 环境的早期尝试，是 Agentic RL 之外的另一条"模型架构驱动"路线。
- 与《Transformer 位置编码》：块因果注意力是位置编码/注意力掩码设计的新场景。
