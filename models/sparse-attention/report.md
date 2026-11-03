# 稀疏注意力前沿技术报告：当「每个 token 看全部」不再成立——从固定模式、查询感知到原生可训练的三代演进

> 关键词：稀疏注意力（Sparse Attention）、原生稀疏注意力（NSA / DSA / MSA）、MoBA 块注意力混合、KV Cache 稀疏化、查询感知稀疏（Quest / SnapKV）、学习式门控（SeerAttention）、长上下文、注意力量化（SageAttention）、硬件对齐内核、Prefill/Decode 分离

> 信源声明：本报告基于 **47 篇 arXiv 原始论文/技术报告**（编号已通过 export.arxiv.org 官方 API 逐一核实，标题与作者一一对应；其中 A–E 组 43 篇核心技术论文，F 组 4 篇综述与评测）、DeepSeek/MiniMax 官方技术报告原文，以及腾讯云、知乎专栏、Kili Technology、Artificial Analysis、七牛云等 8 条官方/二手信源交叉验证。凡厂商自报性能数据，均标注「厂商口径」；媒体/第三方估算标注「第三方」。报告日期：2026-11-02。

## 摘要

自 2017 年《Attention is All You Need》以来，Softmax 自注意力的 **O(L²) 二次复杂度**就是长上下文头顶的紧箍咒。过去八年，破局思路沿三条路线展开：**稀疏注意力**（只算一部分 token 对）、**线性/循环注意力**（去掉 Softmax、把历史压进固定状态）、**KV 压缩**（在显存维度做文章）。本知识库已用专文覆盖后两条路线（《线性注意力与混合架构》《大模型推理优化》），本报告补上缺失的、也是 **2025 年末至 2026 年产业共识收敛最快的一块拼图：稀疏注意力**。

2025 年 2 月是一个分水岭：DeepSeek 发布 **NSA（Native Sparse Attention，arXiv:2502.11089，梁文锋署名）**，月之暗面发布 **MoBA（Mixture of Block Attention，arXiv:2502.13189）**，第一次把「动态、可学习、从预训练阶段就稀疏、且对硬件友好」四件事同时做全。此后短短一年，稀疏注意力从论文迅速走进旗舰生产系统：**DeepSeek-V3.2（2025-12，DSA）、MiniMax M3（2026-06，MSA，注意力计算在 1M 上下文下降 28.4×）、DeepSeek-V4.1-Flash（2026-09，CSA2 + FP4 KV，全局 KV 占用压到 890 字节/token）、GLM-5.1（HyScale 训练系统）** 相继采用。一个极具信号意义的事件是：MiniMax 在 M1 时代押注线性注意力（Lightning Attention），又在 M2 系列连续四代以「稀疏注意力尚不成熟」为由退回全注意力，最终在 M3 上以全新的双分支 MSA 重回稀疏——头部厂商在两代旗舰之间用脚投票，指向了同一个方向。

本报告系统梳理：（1）稀疏注意力的第一性原理——它到底在省「算力」还是「显存」，Prefill 与 Decode 两阶段的稀疏有何本质不同；（2）**三代技术谱系**——固定模式稀疏（Sparse Transformer/Longformer/BigBird）→ 免训练动态/查询感知稀疏（H₂O/Quest/MInference/SnapKV）→ 原生可训练/学习式稀疏（NSA/MoBA/MSA/DSA/SeerAttention）；（3）四大原生稀疏机制的架构与超参数逐项拆解；（4）稀疏注意力与线性注意力、KV 压缩的三维正交关系（可叠加而非互斥）；（5）硬件对齐与内核工程——为什么「理论稀疏」常常「实际更慢」；（6）评测体系、旗舰横评、开放问题与工程落地 Checklist。

## 目录

0. 为什么是现在：头部厂商为何在 2026 集体对注意力动刀
1. 第一性原理：两笔账、两个阶段、两个稀疏度
2. 三代技术谱系（2019–2026）
3. 原生可训练稀疏深度拆解：NSA / MoBA / MSA / DSA·CSA2
4. 免训练推理稀疏：驱逐、查询感知、学习门控与量化近似
5. 三条路线的统一地图：稀疏 × 线性 × KV 压缩
6. 硬件对齐与内核工程：从「理论稀疏」到「实际加速」
7. 评测体系与方法论陷阱
8. 旗舰系统横评与产业地图
9. 开放问题与未来展望
10. 工程落地 Checklist
11. 结论
12. 参考文献

---

## 0. 为什么是现在：头部厂商为何在 2026 集体对注意力动刀

### 0.1 一个被 Agent 工作负载改变的成本结构

2025 年之前，长上下文还主要是「卖点」；2026 年它成了「默认负载」。DeepSeek-V4.1-Flash 技术报告（arXiv:2609.19969）开篇给出的判断是：**长程智能体（long-horizon agents）的普及使模型工作负载日益「输入偏重」（input-heavy）**——Agent 连续运行数小时，把历史对话、工具返回、整文件引用不断堆进上下文，Prefill 计算昂贵，巨大的 KV Cache 持续挤压 HBM、SSD 容量与数据传输带宽。计算、存储、带宽三者共同构成了进一步降低部署成本的主要瓶颈。

这与本知识库《大模型推理优化》报告中的结论一致：KV Cache 是推理显存与成本的最大单一来源。但那篇报告侧重的是**显存维度的治理**（驱逐、量化、MLA 压缩、PD 分离）；本报告聚焦的是**注意力计算本身的稀疏化**——让每个 query 不再与全部历史 token 做 Softmax。

### 0.2 2025–2026 的标志性事件链

| 时间 | 事件 | 意义 |
|---|---|---|
| 2025-02 | DeepSeek **NSA**（2502.11089）、Kimi **MoBA**（2502.13189）前后脚发布 | 首次同时实现「动态 + 可学习 + 原生训练 + 硬件对齐」 |
| 2025-12 | **DeepSeek-V3.2**（2512.02556）搭载 **DSA** 稀疏注意力量产 | 稀疏注意力进入万亿参数级生产旗舰 |
| 2026-06 | **MiniMax M3 / MSA**（2606.13392）发布，1M 上下文注意力计算降 28.4×（厂商口径） | 曾放弃稀疏的厂商以新机制回归，产业共识形成 |
| 2026-09 | **DeepSeek-V4.1-Flash**（2609.19969）CSA2 + FP4 KV，全局 KV 890 字节/token | 稀疏 + 跨层复用 + 低比特三路叠加 |
| 2026-10 | **HyScale**（2610.32057）为 GLM-5.1 的 1M token 学习式稀疏训练提速 | 训练系统侧开始专门优化「选择」这一新瓶颈 |
| 2026 全年 | arXiv 单月数十篇稀疏注意力新论文（见第 9 章） | 学术密度进入爆发期 |

新华社 2026 年 1 月刊发、清华大学官网转载的《2026 年中国 AI 发展趋势前瞻》中，中国信通院明确把「以 DeepSeek 的 NSA、月之暗面的 MoBA 等为代表的稀疏注意力机制」列为「提升模型推理效率的重要技术路径」，并引述判断：行业已不再单纯依靠参数规模，**精细化机制、算法架构、训练方法的优化成为主要提升方向**。这标志着稀疏注意力从社区热点上升为产业与政策层面的共识方向。

### 0.3 MiniMax 的「转向—回退—再转向」说明了什么

最能说明问题的是 MiniMax 的路线摇摆（据 Kili Technology 2026-06 对 M3 的数据故事梳理与 MiniMax 官方博客）：

- **MiniMax-01 / M1（2025-01 起，arXiv:2501.08313）**：采用 Lightning Attention（线性注意力）+ 标准 Softmax 的混合架构；
- **M2、M2.1、M2.5、M2.7（2026 年上半年）**：连续四代**移除**稀疏/线性注意力、回到全注意力，工程博客的理由是「稀疏注意力基础设施尚不成熟，高效注意力要确定性地击败全注意力还有一段路」；
- **M3（2026-06）**：以全新的双分支 **MSA** 重回稀疏——但这已不是当年被放弃的那种稀疏：它是**可学习、固定注意力预算、在未压缩 KV 上跑精确 Softmax** 的新机制。

这条曲线说明三点：（1）早期固定模式/启发式稀疏确实会在复杂推理与编程任务上掉点；（2）线性注意力「精确检索弱」的硬伤在 Agent/代码场景被放大；（3）**保留 Softmax、只动态选择参与计算的 token 子集**的原生稀疏路线，在 2026 年走到了「不掉点且可量产」的拐点。

> 图 1 给出稀疏注意力的问题定义、两笔账，以及三代技术演进的总览。
>
> diagrams/1-overview.html

---

## 1. 第一性原理：两笔账、两个阶段、两个稀疏度

### 1.1 两笔账：省算力与省显存是两件事

标准缩放点积注意力（单头）：

```
O = softmax(Q Kᵀ / √dₕ) V        Q,K,V ∈ R^{L×d}
```

注意力矩阵 `QKᵀ` 是 `L×L` 的，由此产生两笔不同的成本：

- **算力账（compute）**：`QKᵀ` 与 `softmax(·)V` 共 `O(L²·d)` 次乘加。序列翻倍，注意力计算量约变四倍。**稀疏注意力主要砍这笔账**——跳过大部分 `qᵢ·kⱼ` 的计算。
- **显存账（memory）**：自回归解码需缓存全部历史 K/V，空间 `O(L·d)`，且每生成一个 token 都要从 HBM 读取全部 KV。**KV 压缩（量化/驱逐/MLA）主要砍这笔账**。

> 关键认知：这两笔账落在不同维度，**互不冲突、可以叠加**。DeepSeek-V4.1-Flash 正是「CSA2 稀疏（省算力）+ 跨层 KV 复用与 FP4 KV（省显存）」三路并用的典型（见第 3.4 节）。把「稀疏注意力」和「KV 压缩」混为一谈，是理解这一领域最常见的错误。

此外还要区分第三类技术：**FlashAttention（arXiv:2205.14135）是 IO 感知的精确算法**，它通过分块（tiling）减少 HBM 读写，**不改变 O(L²) 的渐近复杂度、不丢任何精度**。稀疏注意力通常需要构建在 FlashAttention 风格的块稀疏内核之上才能真正跑出速度（见第 6 章）。

### 1.2 两个阶段：Prefill 稀疏与 Decode 稀疏的瓶颈不同

这是 2026 年文献中最重要、也最容易被忽略的一组区分：

| 维度 | Prefill（预填充） | Decode（解码） |
|---|---|---|
| 形态 | 一次处理整个 prompt，L 个 query × L 个 key，**算力受限（compute-bound）** | 逐 token 生成，1 个 query × L 个 key，**访存受限（memory-bound）** |
| 主要浪费 | 大量 token 对的注意力矩阵乘 | 每步都要从 HBM 读取全部 KV |
| 稀疏的收益 | 减少 FLOPs（如 MInference、MSA prefill） | 减少 KV 读取量（如 Quest、H₂O、SnapKV） |
| 代表方法 | MInference、NSA、MoBA、MSA、SpargeAttn | Quest、H₂O、PyramidKV、DuoAttention、SeerAttention |

- **Prefill 稀疏**：注意力矩阵天然呈现结构化模式（MInference 发现的 A 形、竖斜线、块稀疏三种模式），可以离线/在线地只算少数块，直接削减矩阵乘 FLOPs。
- **Decode 稀疏**：每步只有一个 query，瓶颈不是矩阵乘而是**把 L 长度的 KV 从显存搬进来**。此时稀疏的价值在于「只加载 top-k 个关键 KV 页」，Quest 观察到「一个 token 是否关键高度依赖于当前 query」，因此用 query 对 KV 页做动态估计，只加载最关键的页，自注意力最高加速 2.23×、端到端推理延迟最多降低 7.03×（论文口径，arXiv:2406.10774）。

一套机制能否同时服务好两个阶段，是原生稀疏（NSA/MoBA/MSA）与纯推理时稀疏（Quest 等）的分水岭——前者从训练时就稀疏，训练/Prefill/Decode 全生命周期对齐；后者免训练、即插即用，但只在推理时生效。

### 1.3 两个稀疏度：计算稀疏与访存稀疏

NSA 论文特别强调一个在 GQA/MQA 架构下容易被忽视的陷阱：单个 query 头的选择是稀疏的（**计算稀疏**），但在 GQA 分组中，多个 query 头共享一组 KV，实际需要读取的 KV 是**组内所有头选择结果的并集**——这个并集可能一点也不稀疏（**访存稀疏失效**）。因此，只证明「每个头只算少数 token」是不够的，必须同时证明「组内选择的并集仍然小」，才能在现代 GQA 架构上真正减少显存访问。这也是 NSA 强调「算术强度平衡（arithmetic intensity-balanced）」与硬件对齐设计的根本原因。

### 1.4 形式化：稀疏注意力在做什么

给定 query `qₜ`，全注意力访问全部历史块集合 `B = {B₁,…,Bₘ}`；稀疏注意力引入一个**选择器/门控** `g`，先选出一个子集 `Sₜ ⊂ B`（|Sₜ| = k ≪ m），只在子集上做精确 Softmax：

```
全注意力：   oₜ = Σ_{j=1}^{t} softmax(qₜᵀkⱼ/√dₕ) vⱼ
稀疏注意力： Sₜ = Select(g, qₜ, {Bⱼ})            ← 关键：怎么选、谁来选、何时选
            oₜ = Σ_{j ∈ Sₜ} softmax(qₜᵀkⱼ/√dₕ) vⱼ   ← 保留 Softmax，精度天然占优
```

不同方法的全部差异，几乎都浓缩在 `Select` 这一步的三个问题上：

1. **选择依据**：固定位置（局部/全局/随机）？内容相似度？query 感知？可学习门控？
2. **选择粒度**：token、页（page）、块（block）、还是头（head）？
3. **选择时机**：纯离线（免训练）？预训练阶段学习（原生）？还是事后轻量蒸馏（适配）？

这三个问题的不同回答，正好对应第 2 章的三代谱系。

---

## 2. 三代技术谱系（2019–2026）

### 2.1 第一代：固定模式稀疏（2019–2021）——证明「稀疏可行」，但模式是手写的

最早一批工作针对的是长文档建模，思路是**用人工设计的固定连接模式替代全连接**，每个 token 只看「局部邻域 + 少量全局锚点」。

- **Sparse Transformer**（Child et al., OpenAI, arXiv:1904.10509, 2019）：把注意力模式分解为**跨步（strided）局部注意力 + 局部卷积式注意力**，用于生成长序列（音频、图像、文本），首次把全注意力的二次复杂度降到 O(L√L) 量级，是整个领域的奠基作。
- **Reformer**（Kitaev et al., arXiv:2001.04451, 2020）：用 **LSH（局部敏感哈希）** 把相近的 query/key 哈希到同一桶，只在桶内做注意力，并配合可逆层省显存。
- **Longformer**（Beltagy et al., arXiv:2004.05150, 2020）：面向长文档的「**滑动窗口 + 少数全局 token**」模式，局部注意力 O(L)、全局注意力覆盖任务关键位置（如 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>），在长文档 QA 上显著优于 BERT 截断。
- **BigBird**（Zaheer et al., Google, arXiv:2007.14062, 2020）：「**局部 + 全局 + 随机**」三组固定连接，并从图论上证明这种稀疏图满足图的扩展性质（作为图注意力网络是万能的）、且是 Turing 完备的，为固定模式稀疏提供了理论支撑。
- **Routing Transformer**（Roy et al., arXiv:2003.05997, 2020）：开始引入**内容路由**——基于 k-means 聚类把 query/key 分到同簇，簇内做注意力，是「动态选择」的雏形。

第一代的共同特征与局限：连接模式**位置先验、人工设计、与输入内容无关**（Routing Transformer 除外）。它们证明了「只看一小部分 token 也能工作」，但固定模式无法适应「关键信息恰好不在预设位置」的输入，泛化受限，且主要用于特定长文档/生成场景，没有进入通用大模型骨干。

### 2.2 第二代：免训练动态/查询感知稀疏（2023–2024）——即插即用，但只在推理时生效

大模型时代，研究重心转向**不改动预训练、直接在推理时动态挑选关键 KV**。这批方法利用一个被反复验证的经验现象——**注意力高度集中在少数关键 token 上，且重要性在层间/时间上相对稳定**。

- **H₂O（Heavy-Hitter Oracle）**（Zhang et al., arXiv:2306.14048, 2023）：基于「heavy-hitter」（累计注意力得分高的 token）做 KV 驱逐，在显存受限时动态淘汰低贡献 KV，显著降低显存而性能损失小。
- **StreamingLLM**（Xiao et al., arXiv:2309.17453, 2023）：发现模型对序列**最开始几个 token（attention sink，注意力汇聚点）**有系统性高注意力；只保留「起始 token + 滑窗」即可让模型流式处理超长输入，免训练。代价是它并不真正记住窗口外内容，适合流式无限生成而非全文检索问答。
- **Scissorhands**（Liu et al., arXiv:2305.17118, 2023）：提出「重要性持续性假设（Persistence of Importance）」——当前重要的 token 在后续仍重要，据此在测试时压缩 KV。
- **FastGen / Adaptive KV**（Ge et al., arXiv:2310.01801, 2023）：自适应地按头决定保留哪些 token（「Model tells you what to discard」）。
- **Quest**（Tang et al., MIT-HAN-Lab, arXiv:2406.10774, 2024）：**query 感知**的 Decode 稀疏——追踪每个 KV 页内 key 的最小/最大值，用当前 query 估计页的关键度，只加载 top-k 页。
- **PyramidKV**（Zhang et al., arXiv:2406.02069, 2024）：按「金字塔式信息汇聚」动态分配 KV 预算——浅层多留、深层少留。
- **SnapKV**（Li et al., arXiv:2404.14469, 2024）：在生成**开始之前**，用最后几个 query 位置的注意力模式识别该关注哪些 KV，prefill 阶段就完成选择。
- **DuoAttention**（Xiao et al., arXiv:2410.10819, 2024）：把注意力头分为「检索头（retrieval）」与「流式头（streaming）」，分别用不同策略，兼顾长上下文检索与流式效率。
- **MInference 1.0**（Jiang et al., Microsoft, arXiv:2407.02490, 2024）：针对 **Prefill**，离线为每个头选定 A 形/竖斜线/块稀疏三种模式之一，在线动态构建稀疏索引 + 专用 GPU 内核，大幅加速长序列 prefill，无需改动预训练或微调。
- **SpargeAttn**（Zhang et al., arXiv:2502.18137, 2025）：**免训练**的稀疏注意力，通过精心设计的近似为任意模型的推理（prefill+decode）提速。
- **RetrievalAttention**（Liu et al., arXiv:2409.10516, 2024）：用向量检索来定位相关 KV，加速长上下文推理。

第二代的共同特征：**免训练、即插即用、基于启发式重要性信号或 query 动态选择**。优点是对现有模型零侵入；局限是（1）只在推理时生效，训练仍用全注意力，训练—推理注意力模式不对齐；（2）启发式选择在特定任务上可能误删关键信息，精度保证依赖经验而非端到端学习。

### 2.3 第三代：原生可训练/学习式稀疏（2025–2026）——让模型自己学会「看哪里」

第三代的核心跃迁是：**把「选择」变成可学习、可端到端训练的一部分，从预训练阶段就使用与推理完全相同的稀疏机制**，并做硬件对齐，使稀疏在训练、Prefill、Decode 全生命周期一致。

- **NSA**（Yuan et al., DeepSeek, arXiv:2502.11089, 2025-02）：压缩 + 选择 + 滑窗三分支 + 可学习门控，硬件对齐、原生可训练（详见 3.1）。
- **MoBA**（Lu et al., Moonshot, arXiv:2502.13189, 2025-02）：把 MoE 路由思想搬到注意力，query 无参数门控动态选块，可在全/稀疏注意力间无缝切换（详见 3.2）。
- **SeerAttention**（Zhang et al., Microsoft, arXiv:2410.13276, 2024-10/持续更新）：用**可学习门控网络**预测块级稀疏，仅需以自蒸馏方式轻量训练门控参数，配合块稀疏 FlashAttention 内核，长上下文 prefill 精度/延迟双优。
- **DSA（DeepSeek Sparse Attention）**：随 **DeepSeek-V3.2**（arXiv:2512.02556, 2025-12）量产，大幅降低长上下文计算复杂度而保持性能。
- **MSA（MiniMax Sparse Attention）**（Lai et al., arXiv:2606.13392, 2026-06）：索引分支 + 主分支双分支、固定 2048 token 注意力预算，KL 对齐训练（详见 3.3）。
- **CSA2（Compressed Sparse Attention 2）**：随 **DeepSeek-V4.1-Flash**（arXiv:2609.19969, 2026-09）推出，跨层 KV 复用 + FP4 KV（详见 3.4）。
- **NAMOH（Mixture-of-Head）**（Fu et al., arXiv:2609.38832, 2026-09）：从「头」维度做稀疏——每个 token 只激活 H 个头中的 K 个，头选择同时决定活跃参数与可用上下文，把「参数缩放」与「上下文缩放」统一起来。
- **SeerAttention-R**（Microsoft, arXiv:2506.08889, 2025-06）：面向**长推理链（long reasoning）**的稀疏注意力适配。
- **训练系统 HyScale**（Deng et al., arXiv:2610.32057, 2026-10）：指出「选择」本身成为学习式稀疏训练的新瓶颈（每个 query 要给所有历史 token 打分），用 HyTopK 免打分表精确 top-k + 选择重放为 GLM-5.1 的 1M token 训练提速。

> 图 2 用时间线呈现三代谱系的代表工作与核心取舍。
>
> diagrams/2-lineage.html

---

## 3. 原生可训练稀疏深度拆解：NSA / MoBA / MSA / DSA·CSA2

### 3.1 NSA：三分支动态层次稀疏（DeepSeek）

NSA 的设计目标是同时满足三点：**全局语境感知、局部精确、以及在现代硬件上真正跑得快**。它把历史 token 组织成时间块，通过三条并行注意力路径处理，再用可学习门控融合。

**三条分支：**

1. **压缩分支（Compressed）——粗看全局**：按压缩块大小 `l=32` 把远处 token 压缩成少量摘要 token，用低成本扫一遍全局大意；
2. **选择分支（Selected）——精看重点**：在全部历史块中按相关性挑出 `n=16` 个块（选择块大小 `l′=64`，含固定激活的 1 个初始块和 2 个局部块），只对这些块做细粒度注意力，绝大部分块被跳过；
3. **滑窗分支（Sliding Window）——细看近邻**：窗口大小 `w=512`，滑动步长 `d=16`，保证相邻 token 的局部信息不丢。

三条分支的输出由**可学习门控**加权合并。论文还刻意让压缩/选择分支与局部模式「隔离」，避免模型走局部捷径而不去学习长程选择。

**关键超参数（论文原文，27B 骨干评测配置）**：压缩块 `l=32`、滑窗步长 `d=16`、选择块 `l′=64`、选择块数 `n=16`、滑窗 `w=512`；GQA 分组 `h=16`，query/key 维度 `dₖ=192`，value 维度 `dᵥ=128`。

**两大创新：**

- **算术强度平衡的算法 + 硬件对齐内核**：专门设计 Triton 内核，使稀疏计算模式在 GPU 上真正跑出速度，而不是「理论稀疏、实际更慢」；
- **端到端原生可训练**：从预训练就用稀疏，训练与推理同一套机制。在 **27B 参数、260B token** 的骨干上，NSA 在通用基准、长上下文任务与基于指令的推理上**保持或超过全注意力**，预训练损失稳定平滑且更低。

**实测效率（论文口径）**：64k 上下文下，前向最高 **9.0×**、后向 **6.0×** 加速；解码时每注意力操作的访存量随上下文增长远低于全注意力（8k→64k 下访存仅从 2048 增至 5632 等价 token，而理论全注意力对应 4×→11.6× 的访存增长）。

### 3.2 MoBA：把 MoE 路由搬到注意力（Moonshot）

MoBA 遵循「**更少结构（less structure）**」原则——不引入 sink/窗口这类预定义位置偏置，而是让模型根据内容自主决定看哪里。它把长度 `N` 的上下文切成 `n` 个块（块大小 `B=N/n`），每个 query 通过**无参数门控**（query 与块代表向量算相关性，不引入待训练参数）路由到最相关的 **top-k 个块**做注意力。

**工程细节与关键参数：**

- 「当前块」（含 query 自身的块）的路由会破坏因果性（块内 mean pooling 会看到未来），需用因果掩码特殊处理；
- 在 1M 连续预训练阶段：块大小设为 **4096、top-K=12**，注意力稀疏度达 `1 − 4096×12/1M = ` **95.31%**；在 1M 连续预训练完成后，再激活 MoBA 训练 100B token；
- 支持上下文从 8K 到 **10M** 的长度缩放，通过变化块大小保持恒定稀疏率（固定 64 个 MoBA 块、top-k=3）。

**两大特性：**

- **全/稀疏无缝切换**：同一套权重，需要精度时切回全注意力、需要速度时切稀疏，降低了「上稀疏掉点」的风险；
- **已部署**：MoBA 已用于支撑 Kimi 的长上下文请求。**Prefill 1M token 时加速比最高 6.5×（论文口径）**，且在所有上下文长度上都比全注意力更快，呈现亚二次复杂度。

### 3.3 MSA：双分支固定预算稀疏（MiniMax）

MSA 是 MiniMax 为 M3 设计的新机制，其哲学是「**不近似注意力，只在精选子集上跑全精度 Softmax**」。

**双分支结构（块大小 128 token）：**

1. **索引分支（Index Branch）**：轻量分支，为每个 query 选出 **top-16 个最相关 KV 块**，把注意力窗口封顶在 **16×128 = 2048 token**——无论总上下文是 10K 还是 1M，注意力预算恒定；
2. **主分支（Main Branch）**：只在被选中的块上运行**精确 Softmax 注意力**。

索引分支用**与主分支对齐的 KL 散度损失**训练，使「什么是相关」在预训练中被学习，而非固定启发式。

**验证与实测：**

- 在一个 **109B 参数、3T token 从头训练**的验证模型上，MSA 在下游基准上追平标准 GQA（论文口径）；
- 1M 上下文下，每 token 注意力计算减少 **28.4×（厂商口径，arXiv:2606.13392）**；
- 内核采用 **KV-outer gather**（每个块只读一次、连续访存），据 Tech Times 报道比 Flash-Sparse-Attention、flash-moba 等开源方案快约 4×（第三方/媒体口径）；
- **与 MLA 的取舍**：MSA 在**未压缩 KV** 上操作，显存占用更高，但长上下文深处细微信息的检索保真度更高；MLA 把 KV 压进低维潜空间省显存。二者代表「检索精度 vs 显存」的不同权衡。

### 3.4 DSA 与 CSA2：DeepSeek 的量产与极限压缩

**DeepSeek-V3.2（2025-12，arXiv:2512.02556）的 DSA**：作为三大技术突破之首，DSA「在长上下文场景大幅降低计算复杂度，同时保持模型性能」。V3.2 配合可扩展 RL 框架与大规模 Agent 任务合成流水线，性能对标 GPT-5（厂商口径）。

**DeepSeek-V4.1-Flash（2026-09，arXiv:2609.19969）的 CSA2**，把稀疏推到新的极限，是「三笔账一起算」的范本：

- **模型本体**：552B 骨干参数的多模态 MoE，支持最高 **1M token**；采用**因果编码器-解码器（CED）非对称架构**——前 20 层因果编码器处理输入、后 20 层解码器生成输出，decode 每 token 激活 16B、prefill 仅激活 8B，对输入偏重的 Agent 负载成本效率极高；
- **CSA2（Compressed Sparse Attention 2）**：**跨层 KV Cache 复用**（相邻层共享/复用 KV）+ 稀疏注意力；
- **FP4 KV**：KV 以 FP4 低比特缓存；
- **成果（厂商口径）**：全局 KV 占用（始终在 HBM）降到 **890 字节/token，约为 DeepSeek-V4-Flash 的 1/4**；通过 **SWA Bounded Replay** 部署优化，持久化 KV 占用（SSD/主机内存）约为 V4-Flash 的 **1/8**；在 KV 占用大幅缩小的同时性能显著优于基线。

> 图 3 横向对比 NSA / MoBA / MSA / CSA2 四大原生机制的分支结构、选择粒度与超参数。
>
> diagrams/3-native-mechanisms.html

---

## 4. 免训练推理稀疏：驱逐、查询感知、学习门控与量化近似

原生稀疏需要从头训练或大规模续训；对于**已经训练好、不想重训**的模型，免训练方法是落地首选。本章按机理分四族。

### 4.1 驱逐派：H₂O 与 StreamingLLM

- **H₂O（arXiv:2306.14048）**：把 KV 驱逐建模为「保留 heavy-hitter」——累计获得高注意力的 token 更可能影响未来输出，显存受限时淘汰累计贡献最低者。它是推理时 KV 预算治理的代表性算法。
- **StreamingLLM（arXiv:2309.17453）**：揭示 attention sink 现象，只保留「初始几个 token + 滑窗」即可流式处理超 400 万 token 输入，无需微调。但要清醒：它维持的是**流式生成的稳定性**，并不对窗口外内容做真实检索，不能用于需要全文证据的问答。
- **Scissorhands（arXiv:2305.17118）**：以「重要性持续性假设」为依据做测试时 KV 压缩。

### 4.2 查询感知派：Quest、SnapKV、PyramidKV

这一族的核心洞察是「**token 的关键度依赖于 query，不能用全局固定排名一刀切**」。

- **Quest（arXiv:2406.10774）**：Decode 时按 KV **页**（而非单 token）管理——记录每页 key 的 min/max，用当前 query 估计该页能达到的最大注意力，只加载 top-k 关键页。自注意力最高 2.23× 加速、端到端延迟最多降 7.03×，长依赖任务精度损失可忽略（论文口径）。
- **SnapKV（arXiv:2404.14469）**：在生成前用 prompt 尾部若干「验证位置」的 query 对 KV 的注意力作为重要性信号，prefill 阶段就完成 KV 筛选，无需逐生成步重判。
- **PyramidKV（arXiv:2406.02069）**：按层做金字塔式预算分配——浅层保留较多 KV（底层汇聚信息）、深层大幅压缩，契合「信息沿层汇聚」的观察。
- **DuoAttention（arXiv:2410.10819）**：区分检索头/流式头，检索头需要从全文召回、流式头只需近期 KV，按头分配不同 KV 策略。

### 4.3 学习门控派：SeerAttention（轻量蒸馏适配）

SeerAttention（arXiv:2410.13276）介于「免训练」与「原生」之间：它不重训整个模型，而是**单独训练一个轻量门控网络**——门控由可学习线性层从 K/V 块产出打分，预测块级注意力稀疏；门控参数以**自蒸馏**方式（让稀疏输出对齐原模型全注意力输出）快速收敛，再配合块稀疏 FlashAttention 内核。这为「给现成模型加可学习稀疏」提供了一条成本远低于从头原生训练的路径。**SeerAttention-R（arXiv:2506.08889）**进一步把该思想用于长推理链场景。

### 4.4 量化近似派：SageAttention（注意力量化，与稀疏正交）

严格说注意力量化不属于「选择 token 子集」，但它常与稀疏配合，是推理加速的重要拼图：

- **SageAttention（arXiv:2410.02367）**：对 Q/K 做高精度 8-bit 量化后再算注意力，即插即用；OPS 比 FlashAttention2/xformers 高约 2.1×/2.7×，端到端指标几乎无损（论文口径）。
- **SageAttention2（arXiv:2411.10958）**：彻底平滑离群值 + per-thread INT4 量化，在 Hopper GPU 上速度追平 FlashAttention3(FP8) 而精度更高。
- **SageAttention2++（arXiv:2505.21136）**：更高效的工程实现。

> 注意：量化近似改变的是「每次点积的数值精度与速度」，稀疏改变的是「算多少次点积」，二者正交、可叠加。这也是为什么 V4.1-Flash 能同时使用稀疏（CSA2）与 FP4 KV。

---

## 5. 三条路线的统一地图：稀疏 × 线性 × KV 压缩

把视野拉到整个「高效注意力/高效推理」版图，存在三条常被混淆、实则正交的路线。本知识库已分别为后两条立传，此处给出统一对照。

| 路线 | 代表 | 保留 Softmax？ | 主要省 | 历史信息形态 | 精确检索（大海捞针） |
|---|---|---|---|---|---|
| **稀疏注意力** | NSA / MoBA / MSA / DSA / Quest | ✅ 保留（只选子集） | 算力（+部分显存） | 仍是显式 KV 列表，只是不全部参与 | 强（选中即精确） |
| **线性/循环注意力** | Lightning / Mamba / DeltaNet / KDA | ❌ 去掉 | 算力 | 压缩进固定大小状态 | 弱（状态容量有限） |
| **KV 压缩** | MLA / MQA·GQA / KIVI / FP4 KV | ✅（MLA/GQA 仍是 softmax） | 显存 | 低维潜空间 / 共享头 / 低比特 | 中（取决于压缩比） |
| **IO 感知精确** | FlashAttention 三代 | ✅ | 显存读写（不改复杂度） | 完整 KV | 强（无损） |

三条判据帮助厘清关系：

1. **稀疏 vs 线性**：稀疏保留 Softmax、检索精度天然占优，代价是选择本身有成本、最坏情况仍可能退化；线性彻底改计算方式、理论 O(n)，但固定状态在精确召回上吃亏。MiniMax 从 Lightning（线性）转向 MSA（稀疏），本质是在 Agent/代码这类强检索场景用「保留 Softmax」换检索精度。
2. **稀疏 vs KV 压缩**：前者砍算力、后者砍显存，维度不同、**可叠加**。DeepSeek 自家就是 MLA（显存）与 NSA/DSA（算力）一起上。
3. **头维度是新战场**：NAMOH（Mixture-of-Head，arXiv:2609.38832）证明还可以从「激活哪些头」做稀疏，让参数缩放与上下文缩放统一——增加 H 而固定 K，既加参数又缩短每头历史、不增总 KV 存储。

> 图 4 用三维坐标呈现这些方法在「算力—显存—检索精度」空间中的位置与可叠加关系。
>
> diagrams/4-landscape.html

---

## 6. 硬件对齐与内核工程：从「理论稀疏」到「实际加速」

稀疏注意力最残酷的一课是：**渐近复杂度更优，不等于墙上时间（wall-clock）更短**。不规则的稀疏访存若处理不当，GPU 大量时间耗在 gather、分支与等待上，反而比稠密的 FlashAttention 慢。这也是早期稀疏被 MiniMax 判「不成熟」的原因。硬件对齐要解决四件事。

### 6.1 块对齐与算术强度平衡

- **块大小必须与 GPU 张量核（tensor core）的 tile 对齐**：NSA 要求内核块 `Bₖ` 整除选择块 `l′`，使内层循环能跑满张量核；
- **算术强度（arithmetic intensity）= 每字节访存对应的计算量**。稀疏容易「访存多、算得少」而沦为访存受限；NSA 通过三分支负载设计让算术强度平衡，使稀疏计算在硬件上真正受惠。

### 6.2 GQA 下的「并集」陷阱

如 1.3 节所述，GQA 组内多 query 头共享 KV，单头选择稀疏不代表组内访存稀疏。必须以「组内选择并集」为优化目标，才能在 GQA/MQA 这一现代主流架构上真正减少 KV 读取。

### 6.3 专用内核与 gather 模式

- **MSA 的 KV-outer gather**：以 KV 块为外层循环，每个块只读一次、连续访存，再分发给选了它的 query，避免重复读取；
- **块稀疏 FlashAttention**：SeerAttention、MoBA 都构建在 FlashAttention 风格的块稀疏内核上，把不规则索引转化为对若干连续块的稠密注意力调用；
- **MInference 三模式内核**：针对 A 形/竖斜线/块稀疏分别优化 GPU 内核。

### 6.4 训练侧的新瓶颈：「选择」本身很贵

HyScale（arXiv:2610.32057）指出了一个 2026 年才凸显的问题：学习式稀疏训练时，**每个 query 要先给所有历史 token 打分才能选出 top-k，而真正的稀疏注意力只读 k 个**——选择步骤反而成了训练瓶颈；现有系统还会物化完整打分表、并在重计算时重复选择。其解法：

- **HyTopK**：流式地算出精确 top-k，不物化完整打分表（把选择输出视为「一组索引」而非「一张激活表」）；
- **选择重放（selection replay）**：反向重计算时直接复用前向的选择结果；
- 据此为 GLM-5.1 在 1M token 序列上的训练显著提速。

这标志着稀疏注意力的工程重心已从「推理内核」延伸到「训练系统」，与本知识库《大模型推理优化》中 PD 分离、《AI 编译器与异构底座》中 LLM 自动写 kernel 的趋势相互呼应。

---

## 7. 评测体系与方法论陷阱

### 7.1 长上下文/稀疏的主流基准

- **RULER（arXiv:2404.06654）**：通过可配置的检索长度、多键检索、聚合、多变量追踪等任务，测量模型「真实有效上下文长度」，专门暴露「标称窗口 ≠ 有效窗口」，是长上下文/稀疏方法最常用的压力测试。
- **LongBench（arXiv:2308.14508）**：双语、多任务长上下文理解基准（单/多文档 QA、摘要、代码等）。
- **Needle In A Haystack（NIAH）**：在长文本中插入一个事实，测单点精确召回，对稀疏方法「是否恰好漏掉关键块」极为敏感。
- **InfiniteBench / PG-19**：MInference 等用于超长 prefill 与长程语言建模评测。

### 7.2 方法论陷阱

1. **加速比口径不可横比**：NSA 的 9.0×（64k，前向）、MoBA 的 6.5×（1M prefill）、MSA 的 28.4×（1M 注意力计算）测的上下文长度、硬件、基线（全注意力 vs FlashAttention）、统计对象（端到端 vs 仅注意力内核）都不同，**不能直接排名**，只能看趋势。
2. **「性能基本持平」要看任务分布**：平均指标持平可能掩盖特定任务（多跳检索、长链推理）上的损失；需报告最差情况与分任务结果，而非只看平均分。
3. **训练—推理对齐**：只在推理时稀疏的方法（第二代）与原生稀疏（第三代）评测的不是一回事——前者测「启发式不丢信息的边界」，后者测「模型学会选择后的能力」，不宜混在一张表里比高低。
4. **评测污染**：超长基准可能进入新模型训练数据；需配合 RULER 这类可程序化生成、可变长度/位置的任务降低污染风险。
5. **EEI 排序的稳定性**：2026 年的注意力综述（arXiv:2608.15459）用效率/表达力/可解释性（EEI）框架比较 21 种方法，并以 20 万次 Monte Carlo 扰动证明：细粒度排名在 67–70% 的样本中会发生超过一位的变化，结论只支持**粗粒度分层比较**，不支持精确名次。这为整个领域的「榜单崇拜」提供了清醒剂。

---

## 8. 旗舰系统横评与产业地图

### 8.1 四大原生机制横评

| 机制 | 厂商 | 结构 | 选择粒度 | 关键预算 | 标志性数据（口径） |
|---|---|---|---|---|---|
| **NSA** | DeepSeek | 压缩+选择+滑窗三分支+门控 | 块（64） | 选 16 块、窗 512 | 64k 前向 9.0×；27B/260B token 验证 |
| **MoBA** | Moonshot | 单门控动态选块（无参数门控） | 块（4096） | top-12，稀疏 95.31% | 1M prefill 6.5×；可切全注意力；已上线 |
| **MSA** | MiniMax | 索引+主双分支（KL 对齐） | 块（128） | top-16，封顶 2048 | 1M 注意力计算 −28.4×（厂商口径） |
| **CSA2** | DeepSeek | 跨层 KV 复用+稀疏+FP4 | 层 + 块 | 890 字节/token | 全局 KV 1/4、持久 KV 1/8（对比 V4-Flash） |

### 8.2 采用稀疏注意力的旗舰模型

| 模型 | 时间 | 注意力 | 规模/上下文 | 要点 |
|---|---|---|---|---|
| DeepSeek-V3.2 | 2025-12 | DSA | 长上下文 | 三大突破之首，配合可扩展 RL，对标 GPT-5（厂商口径） |
| MiniMax M3 | 2026-06 | MSA | 428B MoE / 23B 激活 / 1M | 100T+ token、原生多模态；AA 智能指数 55（第三方） |
| DeepSeek-V4.1-Flash | 2026-09 | CSA2 + FP4 | 552B / 1M | CED 非对称架构，prefill 仅激活 8B |
| GLM-5.1 | 2026 | 学习式稀疏 | 1M 训练 | 由 HyScale 系统支撑训练 |
| Kimi 长上下文服务 | 2025 起 | MoBA | 1M+ | 已部署支撑生产请求 |

### 8.3 产业信号与中国布局

- 稀疏注意力是 2026 年**中国头部厂商集体引领**的架构方向：DeepSeek（NSA/DSA/CSA2）、Moonshot（MoBA）、MiniMax（MSA）、智谱 GLM（学习式稀疏）均已落地；新华社/中国信通院将其列为重点技术路径。
- 美国侧以 **Microsoft（MInference、SeerAttention）** 及学界（MIT-HAN-Lab 的 Quest）为代表，更多从「现成模型推理加速」切入。
- 配套生态正在成熟：专用内核（块稀疏 FlashAttention）、训练系统（HyScale）、部署指南（NVIDIA 已发布 M3 官方部署指南）、以及 vLLM/SGLang/llama.cpp 的社区适配。
- 需注意许可差异：M3 采用 **MiniMax Community License**（非 MIT/Apache，商用需授权），「开放权重 ≠ 可自由商用」。

---

## 9. 开放问题与未来展望

1. **稀疏是不是终局？** 线性注意力火过又被稀疏盖过；注意力地基层八年改了无数次。稀疏保留 Softmax 赢得了检索精度，但是否会被「稀疏 + 线性」更深融合或全新机制取代，仍无定论。
2. **选择的质量如何给形式化保证？** 2026-10 已出现《Certified Sparse Attention via Perturbed Retrieval》（arXiv:2610.24991）这类**可认证稀疏**的尝试，以及《Sparse Attention Is Matrix Approximation, Not Choosing from a Bag of Values》（arXiv:2610.10871）对理论框架的反思——把稀疏理解为矩阵近似而非「从一袋值里挑」，可能催生更好的误差界。
3. **非对称/头维度稀疏的潜力**：《More Value per Key: Asymmetric Sparse Attention》（arXiv:2610.04753）与 NAMOH 头稀疏（arXiv:2609.38832）刚起步，参数缩放与上下文缩放能否真正统一，是值得跟踪的前沿。
4. **长推理链下的稀疏**：推理模型（o1/R1 类）会产生数万 token 的思维链，注意力在长链上如何分布、稀疏是否伤害推理，SeerAttention-R（arXiv:2506.08889）开了头但远未解决。
5. **训练稳定性与缩放律**：原生稀疏在超大规模（万亿参数、百万 token）下的训练稳定性、以及「稀疏预算 k 应如何随模型/上下文规模缩放」，尚缺系统的缩放定律。
6. **跨层/跨模态泛化**：跨层 KV 复用（CSA2）的极限、视频/音频等稠密模态上稀疏模式是否与文本一致（VideoNSA, arXiv:2510.02295 已在探索），都需更多证据。
7. **理论与实证的落差**：固定模式稀疏有 BigBird 式图论保证但灵活性差；动态/学习式稀疏效果好却缺误差界。如何为「学习到的选择」提供可证明的召回保证，是连接两代的核心理论难题。

---

## 10. 工程落地 Checklist

**选型判断**

- ☐ 先分清瓶颈在 **Prefill（算力受限）还是 Decode（访存受限）**：前者选 MInference/MSA/NSA 类 prefill 稀疏，后者选 Quest/H₂O/PyramidKV 类 KV 读取稀疏。
- ☐ 能从头训练/大规模续训 → 优先**原生稀疏（NSA/MoBA/MSA 路线）**，训练—推理全生命周期对齐；不能重训 → 用**免训练方法**或 SeerAttention 式轻量门控适配。
- ☐ 强检索/多跳/代码场景优先**保留 Softmax 的稀疏**而非线性注意力，避免精确召回掉点。

**叠加优化**

- ☐ 稀疏（省算力）与 KV 压缩（MLA/FP4，省显存）**叠加使用**，并评估 GQA 组内「选择并集」的真实访存稀疏度。
- ☐ 稀疏内核构建在 **FlashAttention 风格块稀疏内核**上；块大小与张量核 tile 对齐，评估算术强度，避免「理论稀疏、实际更慢」。
- ☐ 训练侧评估「选择」开销，必要时采用 HyTopK/选择重放等免打分表方案。

**评测与风险**

- ☐ 用 **RULER + NIAH + LongBench** 组合测真实有效上下文，报告分任务与最差情况，不只看平均分。
- ☐ 加速比标注清楚上下文长度/硬件/基线/统计口径，不跨研究横比排名。
- ☐ 核查模型许可（社区许可 vs MIT/Apache）与评测污染；关键业务用自有领域数据做独立验收。

---

## 11. 结论

稀疏注意力在 2025 年 2 月因 NSA 与 MoBA 迎来范式转折，又在随后一年内通过 DeepSeek-V3.2/V4.1、MiniMax M3、GLM-5.1 等旗舰迅速完成「从论文到量产」的闭环。它的胜出并非偶然：相较线性注意力，它**保留 Softmax、因而保住了大模型最依赖的精确内容寻址能力**；相较固定模式稀疏，它**让模型自己学会「看哪里」**，摆脱了手写位置先验；同时它与 KV 压缩、注意力量化正交可叠加，给系统留出了持续压缩的空间。

MiniMax「线性 → 全注意力 → 原生稀疏」的路线曲线，是整个行业认知收敛的缩影：真正走到拐点的不是「稀疏」这个老概念，而是「**动态可学习 + 原生训练对齐 + 硬件对齐内核**」三者齐备的新稀疏。面向 2027，可认证的选择质量、头/非对称维度稀疏、长推理链与跨模态稀疏、以及学习式选择的缩放定律，将决定这条路线是成为终局，还是再次被更新的范式吸收。无论结果如何，「让每个 token 只看它该看的地方」已经是当前大模型架构演进中最确定、最值得投入的方向之一。

---

## 12. 参考文献

> 以下 43 篇文献编号均于 2026-11-02 通过 export.arxiv.org 官方 API（id_list 直查与标题检索）逐一核实，标题与作者对应一致。

**A. 奠基与固定模式稀疏（第一代）**

1. Vaswani, A., et al. *Attention Is All You Need*. arXiv:1706.03762, 2017. https://arxiv.org/abs/1706.03762
2. Child, R., et al. *Generating Long Sequences with Sparse Transformers*. arXiv:1904.10509, 2019. https://arxiv.org/abs/1904.10509
3. Kitaev, N., et al. *Reformer: The Efficient Transformer*. arXiv:2001.04451, 2020. https://arxiv.org/abs/2001.04451
4. Beltagy, I., Peters, M. E., Cohan, A. *Longformer: The Long-Document Transformer*. arXiv:2004.05150, 2020. https://arxiv.org/abs/2004.05150
5. Zaheer, M., et al. *Big Bird: Transformers for Longer Sequences*. arXiv:2007.14062, 2020. https://arxiv.org/abs/2007.14062
6. Roy, A., et al. *Efficient Content-Based Sparse Attention with Routing Transformers*. arXiv:2003.05997, 2020. https://arxiv.org/abs/2003.05997

**B. 免训练动态 / 查询感知稀疏（第二代）**

7. Zhang, Z., et al. *H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*. arXiv:2306.14048, 2023. https://arxiv.org/abs/2306.14048
8. Xiao, G., et al. *Efficient Streaming Language Models with Attention Sinks (StreamingLLM)*. arXiv:2309.17453, 2023. https://arxiv.org/abs/2309.17453
9. Liu, Y., et al. *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time*. arXiv:2305.17118, 2023. https://arxiv.org/abs/2305.17118
10. Ge, S., et al. *Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs (FastGen)*. arXiv:2310.01801, 2023. https://arxiv.org/abs/2310.01801
11. Tang, Y., et al. *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference*. arXiv:2406.10774, 2024. https://arxiv.org/abs/2406.10774
12. Zhang, Z., et al. *PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling*. arXiv:2406.02069, 2024. https://arxiv.org/abs/2406.02069
13. Li, Y., et al. *SnapKV: LLM Knows What You are Looking for Before Generation*. arXiv:2404.14469, 2024. https://arxiv.org/abs/2404.14469
14. Xiao, Y., et al. *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads*. arXiv:2410.10819, 2024. https://arxiv.org/abs/2410.10819
15. Jiang, H., et al. *MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention*. arXiv:2407.02490, 2024. https://arxiv.org/abs/2407.02490
16. Zhang, J., et al. *SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference*. arXiv:2502.18137, 2025. https://arxiv.org/abs/2502.18137
17. Liu, D., et al. *RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval*. arXiv:2409.10516, 2024. https://arxiv.org/abs/2409.10516
18. Zhang, Y., et al. *Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference*. arXiv:2403.09054, 2024. https://arxiv.org/abs/2403.09054

**C. 原生可训练 / 学习式稀疏（第三代）**

19. Yuan, J., et al. *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (NSA)*. arXiv:2502.11089, 2025. https://arxiv.org/abs/2502.11089
20. Lu, E., et al. *MoBA: Mixture of Block Attention for Long-Context LLMs*. arXiv:2502.13189, 2025. https://arxiv.org/abs/2502.13189
21. Zhang, Y., et al. *SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs*. arXiv:2410.13276, 2024. https://arxiv.org/abs/2410.13276
22. DeepSeek-AI, et al. *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*. arXiv:2512.02556, 2025. https://arxiv.org/abs/2512.02556
23. Lai, X., et al. *MiniMax Sparse Attention (MSA)*. arXiv:2606.13392, 2026. https://arxiv.org/abs/2606.13392
24. DeepSeek-AI, et al. *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*. arXiv:2609.19969, 2026. https://arxiv.org/abs/2609.19969
25. Fu, Z., Wang, R., Li, M. *Scaling Parameter and Context in Attention: Native Sparse Attention from Mixture-of-Head (NAMOH)*. arXiv:2609.38832, 2026. https://arxiv.org/abs/2609.38832
26. Deng, Y., et al. *HyScale: Scaling Long-Context Training for Learned Sparse Attention*. arXiv:2610.32057, 2026. https://arxiv.org/abs/2610.32057
27. *SeerAttention-R: Sparse Attention Adaptation for Long Reasoning*. arXiv:2506.08889, 2025. https://arxiv.org/abs/2506.08889
28. *VideoNSA: Native Sparse Attention Scales Video Understanding*. arXiv:2510.02295, 2025. https://arxiv.org/abs/2510.02295
29. *FSA: An Alternative Efficient Implementation of Native Sparse Attention Kernel*. arXiv:2508.18224, 2025. https://arxiv.org/abs/2508.18224

**D. 2026 前沿：可认证、非对称与理论反思**

30. *Certified Sparse Attention via Perturbed Retrieval*. arXiv:2610.24991, 2026. https://arxiv.org/abs/2610.24991
31. Wan, F., et al. *Sparse Attention Is Matrix Approximation, Not Choosing from a Bag of Values*. arXiv:2610.10871, 2026. https://arxiv.org/abs/2610.10871
32. Elata, N., et al. *More Value per Key: Asymmetric Sparse Attention for Faster LLM Decoding*. arXiv:2610.04753, 2026. https://arxiv.org/abs/2610.04753
33. *Subquadratic Attention with a Hierarchical Tree-Structured Indexer*. arXiv:2610.42112, 2026. https://arxiv.org/abs/2610.42112
34. *LazyFlash: Forward-Dense, Backward-Sparse Attention for Accelerated Long-Context Fine-Tuning*. arXiv:2610.31911, 2026. https://arxiv.org/abs/2610.31911
35. *SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking*. arXiv:2609.13141, 2026. https://arxiv.org/abs/2609.13141
36. *Joint Subspace Sparse Attention for Efficient Long-Context Decoding*. arXiv:2610.19026, 2026. https://arxiv.org/abs/2610.19026

**E. 相关：量化近似、线性注意力、KV 压缩与 IO 感知（对照）**

37. Zhang, X., et al. *SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration*. arXiv:2410.02367, 2024. https://arxiv.org/abs/2410.02367
38. *SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization*. arXiv:2411.10958, 2024. https://arxiv.org/abs/2411.10958
39. *SageAttention2++: A More Efficient Implementation of SageAttention2*. arXiv:2505.21136, 2025. https://arxiv.org/abs/2505.21136
40. Dao, T., et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. arXiv:2205.14135, 2022. https://arxiv.org/abs/2205.14135
41. Liu, H., et al. *Ring Attention with Blockwise Transformers for Near-Infinite Context*. arXiv:2310.01889, 2023. https://arxiv.org/abs/2310.01889
42. DeepSeek-AI, et al. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. arXiv:2405.04434, 2024. https://arxiv.org/abs/2405.04434
43. MiniMax, et al. *MiniMax-01: Scaling Foundation Models with Lightning Attention*. arXiv:2501.08313, 2025. https://arxiv.org/abs/2501.08313

**F. 综述与评测**

44. *Efficient Attention Mechanisms for Large Language Models: A Survey*. arXiv:2507.19595, 2025. https://arxiv.org/abs/2507.19595
45. *Not All Attention Is Equal: A Quantitative Survey of the EEI Trade-off*. arXiv:2608.15459, 2026. https://arxiv.org/abs/2608.15459
46. Hsieh, C.-P., et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* arXiv:2404.06654, 2024. https://arxiv.org/abs/2404.06654
47. Bai, Y., et al. *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding*. arXiv:2308.14508, 2023. https://arxiv.org/abs/2308.14508

**G. 官方与二手信源（交叉验证，非 arXiv）**

48. DeepSeek-AI. *Native Sparse Attention*（NSA 论文 HTML 全文，超参数与 9.0×/6.0× 加速数据核对）. https://arxiv.org/html/2502.11089v2
49. Moonshot AI. *MoBA: Mixture of Block Attention for Long-Context LLMs*（论文 HTML 全文，95.31% 稀疏度与 6.5× 数据核对）. https://arxiv.org/html/2502.13189v1
50. 新华社（清华大学官网转载）.《2026 年中国 AI 发展趋势前瞻》, 2026-01-31. https://www.tsinghua.edu.cn/info/1182/124190.htm
51. 乐小野.《一文读懂稀疏注意力，DeepSeek、Kimi、MiniMax 为什么集体动了 Transformer 的根》, 腾讯云开发者社区, 2026-06. https://cloud.tencent.com/developer/article/2684229
52. Kili Technology. *Data Story: How MiniMax M3 Reversed Its Own Engineering Decision*, 2026-06-30（MSA 双分支、28.4×、109B 验证模型与许可信息）. https://kili-technology.com/blog/data-story-minimax-m3
53. MiniMax. *MiniMax M3* 官方模型页与博客. https://www.minimax.io/blog/minimax-m3
54. Artificial Analysis. *MiniMax-M3*（智能指数 55、AA-Omniscience 尝试率等第三方评测）. https://artificialanalysis.ai/models/minimax-m3
55. 七牛云.《Kimi K3.1 Pro vs DeepSeek V4.1 Flash：2026 年国产旗舰如何选？》, 2026-10-28（CED 架构与产业动态，第三方）. https://news.qiniu.com/archives/1793156564314

> 说明：正文称「43 篇 arXiv」指 A–E 组核心技术论文（第 1–43 条）；F 组另含 2 篇综述与 2 篇评测（第 44–47 条），G 组为 8 条官方/二手信源（第 48–55 条）。全部 arXiv 编号经官方 API 核实。