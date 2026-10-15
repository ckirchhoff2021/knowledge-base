# 线性注意力与混合架构前沿技术报告：当注意力不再「二次」

> 关键词：线性注意力（Linear Attention）、状态空间模型（SSM/Mamba）、门控循环、Delta 规则、Gated DeltaNet、Kimi Linear / KDA、混合架构（Hybrid Architecture）、KV 缓存、长上下文、Test-Time Training

> 信源声明：本报告基于 **27 篇 arXiv 原始论文**（编号已通过 arxiv.org 摘要页逐一核实，标题与作者一一对应）、各厂商官方技术报告，以及作者公开博客交叉验证。凡厂商自报性能数据，均标注「厂商口径」。报告日期：2026-10-15。

## 摘要

过去八年，大模型几乎等同于「Softmax 自回归 Transformer」。但自注意力对序列长度的**二次复杂度**，使其在长上下文场景中显存与算力成本急剧膨胀——KV 缓存随上下文线性增长，成为推理成本的主要来源。2023 年底以来，以 **Mamba（选择性状态空间模型）** 为代表的线性时间序列模型重新点燃了「取代或部分取代 Softmax 注意力」的希望；2024 年 Mamba-2 提出的**结构化状态空间对偶（SSD）**，在数学上把 SSM 与带掩码的线性注意力统一为同一对象；而 **DeltaNet / Gated DeltaNet** 通过把经典的 **Delta 规则**（误差校正更新）并行化，显著提升了固定大小状态下的精确召回能力。

2025 年 10 月，月之暗面（Moonshot AI）发布的 **Kimi Linear** 技术报告成为标志性事件：在**完全相同的训练配方**下，其提出的 **Kimi Delta Attention（KDA）** 与 MLA 分层混合架构，在短上下文、长上下文与强化学习三种场景中**全面超越**纯全注意力基线，同时把 KV 缓存降低约 **75%**，在 100 万 token 上下文下解码吞吐最高提升 **6 倍**，并开源了权重与内核。这是线性注意力第一次在公平对比中「打赢」全注意力。此后 KDA 被进一步放大到 2.8T 参数的 Kimi K3 旗舰模型，标志着线性/混合架构从「备选」走向「主战场」。

本报告系统梳理：（1）线性注意力的数学本质及其与 RNN、SSM 的对偶关系；（2）2017–2026 年的技术演进谱系；（3）五类状态更新规则（朴素线性、门控衰减、Delta 规则、门控 Delta、测试时训练）的原理与取舍；（4）Kimi Linear/KDA、MiniMax-01、Jamba、RWKV、Gated DeltaNet、TTT/Titans 等前沿系统的架构与实测数据；（5）评测体系、工程落地与开放问题。

## 目录

0. 为什么是现在：二次注意力的墙
1. 统一视角：线性注意力 = 可并行训练的现代 RNN
2. 技术演进谱系（2017–2026）
3. 五类状态更新规则深度拆解
4. 前沿系统与旗舰架构
5. 评测体系与方法论陷阱
6. 工程落地：选型、部署与成本
7. 开放问题与未来展望
8. 结论
9. 参考文献

---

## 0. 为什么是现在：二次注意力的墙

### 0.1 Softmax 注意力的复杂度瓶颈

标准缩放点积注意力（单头）可写为：

```
并行形式：  O = softmax(Q Kᵀ / √dₕ) V          ∈ R^{L×d}
循环形式：  oₜ = Σⱼ₌₁ᵗ softmax(qₜᵀkⱼ/√dₕ) vⱼ  ∈ R^d
```

其中 `L` 为序列长度，`d` 为头维度。注意力矩阵 `QKᵀ` 是 `L×L` 的，因此：

- **计算复杂度**：`O(L²·d)`，随序列长度二次增长；
- **KV 缓存**：自回归解码时需缓存全部历史 Key/Value，空间 `O(L·d)`，且每生成一个新 token 都要读取全部历史。

即便有 FlashAttention（降低显存读写、不改变渐近复杂度）、GQA/MQA/MLA（压缩 KV 头数）等工程优化，注意力「随长度增长」的本质没有改变。在 Agent、长文档理解、代码库分析、视频/音频等动辄数十万到上百万 token 的场景下，KV 缓存成为**推理显存与成本的最大单一来源**。

### 0.2 三条破局思路

1. **稀疏注意力（Sparse Attention）**：只计算部分 token 对（局部、块稀疏、路由式），亚二次但依赖特定模式，召回易受损；
2. **线性 / 循环模型（Linear / Recurrent）**：把历史压缩进**固定大小状态**，计算与显存对长度线性甚至常数；
3. **混合架构（Hybrid Architecture）**：堆叠少量 Softmax 注意力层（负责精确召回）与多数线性/循环层（负责长程吞吐），兼顾质量与效率。

2023 年底至今的主线，正是第 2、3 条路线在「**可并行训练**」约束下的成熟与统一。

> 图 1 直观对比了二次注意力与线性/循环注意力在复杂度、KV 缓存与对偶形式上的差异。
>
> diagrams/1-complexity.html

---

## 1. 统一视角：线性注意力 = 可并行训练的现代 RNN

### 1.1 去掉 Softmax，用结合律重组

线性注意力的核心做法是**去掉 softmax**（早期工作还包含特征映射与归一化项，后续研究表明这些并非必需）：

```
并行形式：  O = (Q Kᵀ) V
循环形式：  oₜ = Σⱼ₌₁ᵗ (qₜᵀ kⱼ) vⱼ
```

去掉 softmax 本身不立即降低复杂度，但带来了关键的数学性质——**线性（结合律）**。利用矩阵乘法结合律：

```
(Q Kᵀ) V = Q (Kᵀ V)
```

定义矩阵值状态 `Sₜ = Σⱼ₌₁ᵗ vⱼ kⱼᵀ ∈ R^{d×d}`，则：

```
Sₜ = Sₜ₋₁ + vₜ kₜᵀ，      oₜ = Sₜ qₜ
```

这揭示了线性注意力的本质：**一个带矩阵值状态的线性 RNN**。推理时：

- 自回归复杂度从 `O(L²d)` 降到 `O(L·d²)`；
- 状态空间从 `O(L·d)` 降到 `O(d²)`，与序列长度无关。

### 1.2 状态扩展：为什么需要 d×d 的矩阵状态

传统 RNN（LSTM/GRU）隐状态维度通常与输入同量级，因为基于矩阵乘的状态更新代价高昂；但 RNN 只能靠循环状态记住全部历史，**状态容量是检索能力的瓶颈**。Mamba 明确提出「硬件高效的状态扩展」，而线性注意力式的**外积更新** `vₜkₜᵀ` 被证明是把状态从 `O(d)` 高效扩展到 `O(d²)` 的最优方式之一（Mamba-2 也采用这一策略）。

### 1.3 训练并行化：分块（Chunkwise）与并行扫描

线性 RNN 的历史短板是**训练无法并行**（必须按时间顺序递推）。现代线性注意力通过两条路解决：

- **分块并行（Chunkwise Parallel）**：把序列切成块，块内用并行的注意力式计算，块间用循环状态传递，得到亚二次复杂度且对 GPU 友好（flash-linear-attention 库即此思路）；
- **并行扫描（Parallel Scan）**：线性递推可用前缀扫描在 `O(log L)` 个并行步内算完（SSM/Mamba 路线）。

关键结论：**线性递推可并行，非线性递推不可并行**。保持状态更新线性，就同时拿到了「训练像注意力一样并行」与「推理像 RNN 一样省显存」两大优势。

### 1.4 SSM 与注意力的对偶：Mamba-2 的 SSD

Mamba-2（Dao & Gu, 2024）证明 SSM 与注意力在数学上高度相关，提出**结构化状态空间对偶（Structured State-Space Duality, SSD）**：某种带掩码的线性注意力可视为 SSM 的特例，反之亦然。这让 SSM 能直接复用为注意力优化的、面向 Tensor Core 的 matmul 内核，工程上大幅提速。

至此，**线性注意力、门控 RNN、SSM、（后续的）TTT 收敛为同一个「可并行训练的线性循环状态空间」框架**——它们是同一对象在「并行形式 / 循环形式」下的不同呈现。

---

## 2. 技术演进谱系（2017–2026）

> 图 2 给出了完整时间线。这里按四个阶段提炼关键转折。
>
> diagrams/2-evolution.html

### 2.1 理论奠基期（2017–2022）

- **2017 · Transformer**：Softmax 注意力确立范式，也立起「二次复杂度」这一靶子。
- **2020 · Linear Transformers（Katharopoulos et al.）**：首次去掉 softmax，用结合律把注意力写成线性 RNN。但状态只能加、不能删，长序列记忆过载，质量明显落后。
- **2021 · Fast Weight Programmers（Schlag, Irie & Schmidhuber）**：把线性注意力的状态矩阵解释为被即时写入的「快速权重」，与 1992 年 Schmidhuber 的快速权重编程器统一，为 Delta 规则与 TTT 路线埋下伏笔。

这一阶段的核心矛盾：**线性很快，但打不过 Softmax**。

### 2.2 现代 RNN 崛起（2023）

- **RetNet（2023.07，微软）**：提出 retention，结合指数衰减与并行/循环双重形式，主张做 Transformer 的「继任者」。
- **RWKV（2023.05）**：用线性注意力式时间混合 + 通道混合，纯 RNN 也能以 Transformer 方式并行训练；后加入 Linux 基金会。
- **Mamba（2023.12，Gu & Dao）**：选择性 SSM（S6），让状态更新的 `B、C、Δ` 成为**输入相关**的动态参数，可按内容选择记忆/遗忘；配合硬件感知扫描内核，首次让线性时间模型在语言建模上接近 Transformer。
- **GLA（2023.12，Yang et al.）**：门控线性注意力，用数据相关遗忘门 + 硬件高效训练，是后续大量工作的基础算子。

### 2.3 对偶、Delta 与混合成型（2024）

- **HGRN2（2024.04）**：用外积把分层门控 RNN 的状态从 `d` 扩到 `d²` 而不增参数，并给出线性注意力解释。
- **Zoology（2023.12 / 2024）**：系统揭示「召回（recall）」是次二次模型的命门，并给出简单线性注意力在召回—吞吐间的平衡点。
- **Jamba（2024.03）/ Griffin（2024.02）**：前者按「3–3–5」比例混合 Mamba、注意力与 MoE；后者用 GLA 门控循环 + 局部注意力。混合架构成型。
- **Mamba-2 / SSD（2024.05）**：SSM↔注意力对偶，可复用 matmul 内核。
- **xLSTM（2024.05）**：扩展 LSTM（sLSTM/mLSTM），复兴门控记忆并引入矩阵状态。
- **DeltaNet（2024.06，NeurIPS'24）**：把 Delta 规则沿序列长度并行化，在 MQAR 联想召回上近乎满分。
- **TTT（2024.07）**：把隐状态变成一个在测试序列上被自监督梯度更新的小模型。
- **GDN（2024.12）**：Gated DeltaNetworks，把 Delta 规则装进 Mamba-2 块，在同等规模下稳定优于 Mamba-2。

### 2.4 规模化与「反超」（2025–2026）

- **MiniMax-01（2025.01）**：每 8 层 7 层 Lightning Attention + 1 层 softmax，456B 总参 / 45.9B 激活，推理支持 4M token。
- **Titans（2025.01，Google）**：测试时记忆，引入神经长期记忆模块，在固定上下文记忆之外学习「该记什么」。
- **RWKV-7 Goose（2025.03）**：动态状态演化，主张突破注意力/线性注意力的 TC0 表达力上限。
- **Jamba-1.5（2024.08 / 2025 部署）**：256K 上下文，RULER 平均 95.7%，ExpertsInt8 量化。
- **Kimi K2（2025.07）**：MLA + 线性注意力混合的万亿级 MoE，开源。
- **Jet-Nemotron（2025.08，NVIDIA）**：用 Post Neural Architecture Search 改造 Qwen3，配合新 JetBlock，256K 生成吞吐最高 53.6×。
- **Kimi Linear / KDA（2025.10）**：**首次在公平对比下全面超越全注意力**，成为本报告选定的标志性节点。
- **2026 · HALO/HypeNet**：仅用 2.3B token（不到预训练数据的 0.01%）就把 Qwen3 蒸馏成混合架构，大幅降低中小团队进入门槛；Gated DeltaNet 2 等继续增强表达力；KDA 被放大到 2.8T 的 Kimi K3。

---

## 3. 五类状态更新规则深度拆解

> 所有线性循环层都维护一个矩阵状态 `S ∈ R^{d×d}`，差别在更新算子。图 3 给出横向对比。
>
> diagrams/3-update-rules.html

### 3.1 朴素线性注意力：只能加

```
Sₜ = Sₜ₋₁ + vₜ kₜᵀ
```

形式化地，线性注意力实现的是键值联想记忆 `S = Σ vᵢkᵢᵀ`。假设所有 key 单位归一化，检索 `kⱼ` 时：

```
S kⱼ = vⱼ + Σᵢ≠ⱼ (kᵢᵀkⱼ) vᵢ
                 └── 检索误差 ──┘
```

要让误差为零，需要所有 key 两两**正交**；而 `d` 维空间最多容纳 `d` 个正交向量。这解释了为何增大头维度有效，也说明朴素线性注意力存在根本的「记忆过载」——只能加不能删，串扰随长度累积。

### 3.2 门控 / 指数衰减：能遗忘

```
Sₜ = αₜ ⊙ Sₜ₋₁ + vₜ kₜᵀ
```

引入数据相关的遗忘门 `αₜ`，旧记忆按比例衰减后再写入。不同模型对门矩阵的参数化不同（外积结构以省参数）：

- 衰减快速权重：`Gₜ = βₜ αₜᵀ`
- GLA：`Gₜ = 1·αₜᵀ`
- Mamba-1：`Gₜ = exp(−(Δₜ1ᵀ) ⊙ exp A)`
- Mamba-2：`Gₜ = γₜ 11ᵀ`

门控显著缩小了与 Softmax 的差距，但「精确改写/复制」仍偏弱。

### 3.3 Delta 规则：按误差定向改写

Delta 规则是经典的误差校正学习原则：按「目标 − 当前预测」的差值调整参数。用于线性注意力：

```
Sₜ = Sₜ₋₁ − βₜ (Sₜ₋₁ kₜ − vₜ) kₜᵀ
```

其中 `βₜ` 是学习率，`Sₜ₋₁kₜ` 是当前对 key `kₜ` 的读出（旧值）。也可直观写成「先擦旧、再写新」：

```
vₜⁿᵉʷ = (1−βₜ) vₜᵒˡᵈ + βₜ vₜ
Sₜ = Sₜ₋₁ − vₜᵒˡᵈ kₜᵀ + vₜⁿᵉʷ kₜᵀ
```

`βₜ=0` 时记忆不变，`βₜ=1` 时完全覆盖旧值。DeltaNet 可由对每步在线 MSE 损失做一步梯度下降精确导出：

```
ℒₜ(S) = ½ ‖S kₜ − vₜ‖²
Sₜ = Sₜ₋₁ − βₜ ∇ℒₜ(Sₜ₋₁)
```

在 MQAR（多查询联想召回）这一衡量次二次模型上下文召回的合成基准上，DeltaNet 取得近乎满分，且 MQAR 表现与真实语言建模中的「联想召回命中」强相关。

### 3.4 门控 + Delta：当前最强通用循环层

把「先衰减、再误差校正」结合：

```
Sₜ = αₜ ⊙ Sₜ₋₁ − βₜ ( (αₜ ⊙ Sₜ₋₁) kₜ − vₜ ) kₜᵀ
```

- **GDN（Gated DeltaNetworks）**：在 Mamba-2 块内把（多值）注意力算子替换为 Delta 规则，在语言建模、召回与长序列上以相同或更低算力稳定优于 Mamba-2。
- **Gated DeltaNet（2024.12，Yang et al.）**：在 DeltaNet 基础上加入**多通道遗忘门**，把单通道衰减扩展为每通道独立门控，显著提升状态利用效率；在作者公平对比中成为通用循环层的最强基线。
- **KDA（Kimi Delta Attention，2025.10）**：在 Gated DeltaNet 基础上引入**更细粒度的数据相关门控**。其核心洞察是：线性注意力本质是一个有限状态 RNN，隐状态是稀缺资源；更精细的门控让有限状态被更高效地用于跨超长序列的保留与召回。配合定制的分块算法（一种对角加低秩 DPLR 转移矩阵的特化形式），在 GPU 上无需奇异内核即可高速运行。

### 3.5 测试时训练 TTT：把更新推广为通用学习

```
Wₜ = Wₜ₋₁ − η ∇ℓ(Wₜ₋₁; xₜ)，   oₜ = f(xₜ; Wₜ)
```

隐状态本身是一个模型 `f`（TTT-Linear 用线性模型，TTT-MLP 用两层 MLP），更新规则是自监督学习的一步（如重建被遮挡的输入）。在 125M–1.3B 规模上，TTT-Linear 追平或超过 Transformer 与 Mamba；与 Mamba 不同，它能像 Transformer 一样随上下文增多持续降低困惑度。Google **Titans** 进一步在测试时引入可学习的神经长期记忆。

值得注意的是理论汇合：**TTT-Linear 在去掉非线性组件、mini-batch 大小为 1 时与 DeltaNet 数学等价**。Delta 规则与 TTT 是同一思想（把状态更新视为一次学习）的两个切面。

---

## 4. 前沿系统与旗舰架构

> 图 4 横向对比四大旗舰的层排列模式、规模与关键指标。
>
> diagrams/4-flagship.html

### 4.1 Kimi Linear / KDA：标志性的「反超」

**论文**：Kimi Team, *Kimi Linear: An Expressive, Efficient Attention Architecture*（arXiv:2510.26692，2025-10-30）。

- **架构**：KDA（Kimi Delta Attention）+ MLA（Multi-Head Latent Attention，与 DeepSeek 同款）**分层混合**；预训练一个 **48B 总参 / 3B 激活**的模型。
- **位置编码**：KDA 与 MLA 层均采用 **NoPE**（位置信息由循环/转移结构隐式承载）。
- **公平对比设计**：在**完全相同的训练配方**（数据、优化器、token 数）下，仅替换注意力模块，比较 KDA+MLA 混合 vs 纯 MLA。
- **关键结果（厂商口径）**：
  - 在短上下文、长上下文与强化学习（RL）**三种 regime 下全面超越**纯全 MLA 基线；
  - **KV 缓存降低约 75%**；
  - 在 **100 万 token** 上下文下，解码吞吐最高提升 **6 倍**；
- **开放程度**：开源 KDA 内核、vLLM 实现，以及预训练与指令微调权重（Hugging Face），可在商用硬件上运行 3B/48B 模型。

**为什么重要**：这是线性注意力多年来第一次在受控、公平对比中不是「接近」而是「打赢」全注意力。它改变了每一次模型训练的算力预算方程；75% 的 KV 缓存下降对长上下文 Agent、RAG、代码库分析是直接的成本节约；开源的内核+权重提供了从论文到可运行模型的完整路径。

**规模验证**：KDA 在 9 个月后被放大到 **Kimi K3**（2.8T 总参 / 104B 激活的 MoE，原生多模态，1M 上下文）。公开分析指出 K3 大量使用 KDA 层（社区据 K3 论文统计约 69 个 KDA 层 + 24 个 Gated MLA 层），并加入注意力残差（Attention Residuals）等改进——这是在超大模型而非 3B 演示模型上验证了该架构。Kimi K2（arXiv:2507.20534）此前已在万亿级 MoE 上采用 MLA + 线性注意力混合。

### 4.2 MiniMax-01：线性注意力 + MoE 的数百 B 商用基座

**论文**：*MiniMax-01: Scaling Foundation Models with Lightning Attention*（arXiv:2501.08313，2025-01-14）。

- **层模式**：每 **8 层 = 7 层 Lightning Attention（线性）+ 1 层 softmax 注意力**；Lightning Attention 用平铺（tiling）策略把计算拆为块内（局部）与块间（全局），实现线性复杂度。
- **规模**：**456B 总参 / 45.9B 激活**，32 专家 MoE；RoPE。
- **上下文**：推理支持高达 **4M token**（当时无模型可及）。
- **系统配套**：Varlen Ring Attention（变长、去 padding）、LASP+（线性注意力序列并行）、定制 CUDA 内核。
- **同期多模态**：MiniMax-VL-01 用 512B 视觉-语言 token 训练。

### 4.3 Jamba / Jamba-1.5：混合 Mamba-Transformer-MoE

**论文**：*Jamba*（arXiv:2403.19887）与 *Jamba-1.5: Hybrid Transformer-Mamba Models at Scale*（arXiv:2408.12570）。

- **架构**：Attention : Mamba ≈ **1 : 7**，并叠加 MoE；Jamba-1.5-Large **398B 总参 / 94B 激活**（16 专家 top-2），有效上下文 **256K**（当时开放权重最长）。
- **关键结果（厂商口径）**：RULER 在到 256K 的各长度上均 ≥93.9%，**平均 95.7%**，长上下文优于 GPT-4-1106、Gemini-1.5-pro；**256K 下 KV 缓存仅约 9GB**（LLaMA-3.1-70B 同长度需约 80GB），数量级下降。
- **ExpertsInt8**：因 >85% 权重在 MoE 层，将 MoE/MLP 权重以 INT8 存储、计算前反量化为 BF16，在 vLLM 的 fused_moe 内核中实现；免校准、秒级量化，使 Jamba-1.5-Large 在 **8×80GB GPU** 上处理 256K 而质量无损。
- **训练稳定性**：提出 Activation Loss（`L_total = L_LM + α·L_activation`，`α=1e-5`），把激活幅值控制在 2K–3K，避免 FP16 推理时的数值问题。
- **一个值得记录的工程取舍**：团队发现混合场景下 **Mamba-1 + 注意力**的组合反而优于 Mamba-2——交错的注意力层补偿了 Mamba-2 在孤立使用时的部分优势。

### 4.4 RWKV 系列：纯 RNN 路线

- **RWKV**（arXiv:2305.13048）：时间混合（线性注意力式）+ 通道混合，无注意力、常数显存、生成速度恒定，可像 GPT 一样并行训练。
- **RWKV-5/6 Eagle/Finch**（arXiv:2404.05892）：引入**矩阵值状态**与动态循环。
- **RWKV-7 Goose**（arXiv:2503.14456，2025-03-18）：**动态状态演化（Dynamic State Evolution）**，主张突破注意力/线性注意力范式的 **TC0 表达力上限**；开源 0.1B–14B 全尺寸，大上下文算力需求比 Transformer 低 **10–100×**。
- **短板**：对 prompt 格式敏感；回看/检索类任务偏弱——这恰是「保留少量注意力层」的混合路线（Jamba/MiniMax/Kimi）相对纯 RNN 的价值所在。

### 4.5 Gated DeltaNet / GDN / DeltaNet：算法基石

- **DeltaNet**（arXiv:2406.06484，NeurIPS'24）：并行化 Delta 规则，MQAR 近乎满分。
- **GDN**（arXiv:2412.06464，*Gated Delta Networks: Improving Mamba2 with Delta Rule*）：把 Delta 规则写进 Mamba-2 块，作者给出的多组公平对比显示其在语言建模与召回上稳定优于 Mamba-2。
- **Gated DeltaNet**：在 DeltaNet 上加入多通道遗忘门（社区与后续工作如 Gated DeltaNet 2 持续增强表达力）。

### 4.6 TTT / Titans / Griffin / xLSTM / Jet-Nemotron / HALO

- **TTT**（arXiv:2407.04620）：隐状态即模型，更新即自监督一步。
- **Titans**（arXiv:2501.00663，*Learning to Memorize at Test Time*）：在测试时学习的神经长期记忆。
- **Griffin**（arXiv:2402.19427，Google DeepMind）：GLA 门控循环 + 局部注意力混合。
- **xLSTM**（arXiv:2405.04517）：sLSTM/mLSTM，复兴并扩展 LSTM 记忆。
- **Jet-Nemotron**（arXiv:2508.15884，NVIDIA，NeurIPS 2025）：两大创新——**Post Neural Architecture Search（PostNAS）**先定位预训练 Transformer 中关键注意力层的最优放置、再搜索更优块设计；**JetBlock** 用动态卷积 + 硬件感知搜索增强线性注意力。把 Qwen3 等改造后，256K 生成吞吐最高 **53.6×**（H100，最大 batch），Jet-Nemotron-2B/4B 分别比 Qwen3-1.7B-Base 快 21×/47×。
- **HALO / HypeNet**（arXiv:2601.22156，清华 + ModelBest，2026-01）：跨架构蒸馏流水线，先用注意力权重迁移初始化、隐藏状态对齐，再做**高效注意力层选择**（决定保留哪些层不转换）、知识蒸馏与微调；配合 **HyPE**（RoPE + NoPE 的混合位置编码）实现强长度泛化。把 Qwen3 转成 HypeNet 仅需 **2.3B token（< 预训练数据的 0.01%）**，性能与原模型相当、长上下文更优。对比此前蒸馏方法（Mamba-in-the-Llama 20B、RAD 20B、Jet-Nemotron 400B token），数据效率是数量级提升。

### 4.7 横向对比总表

| 系统 | 时间 | 混合算子 / 比例 | 规模（总/激活） | 位置编码 | 标志性指标（厂商口径） |
|---|---|---|---|---|---|
| **Kimi Linear** | 2025.10 | KDA + MLA 分层 | 48B / 3B | NoPE | KV −75%；1M 解码 6×；全 regime 反超 |
| **MiniMax-01** | 2025.01 | 7 Lightning + 1 softmax | 456B / 45.9B | RoPE | 推理 4M；线性注意力+MoE 数百 B |
| **Jamba-1.5** | 2024.08 | Attn:Mamba≈1:7 + MoE | 398B / 94B | — | RULER 95.7%；256K KV 仅 9GB |
| **RWKV-7** | 2025.03 | 纯 RNN，动态状态演化 | 0.1B–14B | 隐式 | 大上下文算力低 10–100× |
| **GDN** | 2024.12 | Mamba2 块 + Delta 规则 | 中小规模 | — | 同规模稳定优于 Mamba-2 |
| **TTT / Titans** | 2024–25 | 隐状态即模型 + 神经记忆 | 至 1.3B+ | — | 随上下文持续降困惑度 |
| **Jet-Nemotron** | 2025.08 | PostNAS + JetBlock | 2B / 4B | — | 256K 吞吐最高 53.6× |
| **HALO/HypeNet** | 2026.01 | 蒸馏 + HyPE | Qwen3 同档 | RoPE+NoPE | 仅 2.3B token 完成转换 |

---

## 5. 评测体系与方法论陷阱

### 5.1 合成诊断基准：便宜但要看对维度

- **MQAR（多查询联想召回）**：字母↔数字映射后批量召回，直接测上下文联想召回；DeltaNet 在此近乎满分。MQAR 与真实语言建模的召回命中强相关，是筛选架构的低成本前置实验。
- **Zoology**（arXiv:2312.04927）：系统测量并改进高效语言模型的「召回」，指出召回是次二次模型的主要误差来源，并给出简单线性注意力在召回—吞吐间的平衡。
- **MAD 类多任务记忆诊断**：在压缩、模糊召回、上下文召回、记忆、含噪召回、选择性复制等多个维度上分别打分。公开数据中，不同架构「偏科」明显——例如 DeltaNet 在上下文/含噪召回与选择性复制上接近满分，但在「记忆（长序列被动保留）」维度偏弱；Mamba 在选择性复制上偏弱。**单一平均分可能掩盖架构特性，需逐维度解读**。

### 5.2 长上下文基准

- **RULER / Needle-in-a-Haystack（NIAH）**：检索类任务历来是线性注意力的阿喀琉斯之踵。Jamba-1.5 在 RULER 上 256K 平均 95.7%；但社区也反复强调：**效率数字漂亮的同时，必须在真实大海捞针任务上与同尺寸全注意力对比**，这正是 KDA 仍需持续验证的开放点。
- **LongBench / SCROLLS**：更贴近真实任务的长上下文套件。
- **长度泛化（length generalization）**：在短上下文训练、长上下文测试。HALO 的 HyPE（RoPE+NoPE + 注意力 logits 缩放）即专门为混合架构的长度泛化设计。

### 5.3 方法论陷阱

1. **公平对比**：必须控制数据、优化器、训练 token 与算力。Kimi Linear 的说服力正来自「同配方仅换模块」；反之，许多「超越」声明因训练预算不对等而不可比。
2. **准确率 vs 计算开销**：TTS/线性架构都要求把「准确率—FLOPs—墙钟时间—显存」联合评估，单看准确率或单看速度都会误导。
3. **厂商口径**：厂商技术报告中的峰值加速比（如 6×、53.6×）通常在最大 batch、特定上下文、特定硬件下测得，**真实业务负载下收益会打折**，需自行压测。
4. **平均分别掩盖偏科**：要用逐维度/逐长度的细粒度指标。
5. **蒸馏 vs 架构创新的混淆**：当一个模型同时使用前沿模型蒸馏与新架构时，需消融才能归因（Kimi Linear 讨论中即有此争论）。

---

## 6. 工程落地：选型、部署与成本

### 6.1 三条路线怎么选

- **纯 Softmax Transformer**：质量与精确召回最稳，生态最成熟；但长上下文 KV 成本高。适合上下文不长、对召回要求极高、不愿承担新架构风险的场景。
- **纯线性/RNN（RWKV）**：显存常数、生成速度恒定、超长上下文最省、端侧友好；但精确召回/回看偏弱、对格式敏感。适合超长流式、端侧、对成本极敏感且召回要求可通过工程缓解的场景。
- **混合架构（Kimi/MiniMax/Jamba/HALO）**：保留少数（约 1/8–1/4）注意力层负责精确召回，多数线性层负责吞吐，是当前**质量—效率的最佳平衡**，也是本报告推荐的默认方向。

### 6.2 关键工程杠杆

1. **注意力层放置（layer placement）**：并非所有注意力层同等重要。PostNAS、HALO 的注意力层选择、SMART 的输出分布偏移、KL-LS 的 KL 度量都在解决「保留哪几层」——这是混合架构质量的最大杠杆。
2. **KV 缓存压缩**：KDA −75%、Jamba 数量级下降，直接降低长上下文单并发显存与可服务并发数。
3. **量化**：Jamba ExpertsInt8（MoE INT8、免校准）、K3 的原生 MXFP4，使超大 MoE 可在更少/更主流 GPU 上部署。
4. **推理框架**：KDA/JetBlock 均提供 vLLM 实现；线性层的分块预填充（chunked prefilling）、前缀/radix 缓存、PD 分离等可叠加。
5. **低成本获得混合模型**：不必从零预训练——可用 HALO（2.3B token）或 PostNAS/Jet-Nemotron 从已有 Transformer（如 Qwen3）改造，大幅降低验证与落地成本。

### 6.3 落地 Checklist

- [ ] 明确真实上下文长度分布与召回要求（多数业务其实不需要 1M）。
- [ ] 用 MQAR/Zoology 类低成本基准做架构初筛。
- [ ] 在**自有数据、目标硬件**上做准确率—吞吐—显存—延迟（含 p99）联合压测，勿只信厂商峰值。
- [ ] 对混合模型重点验证 NIAH/RULER 等精确召回，不被平均分迷惑。
- [ ] 优先评估开源混合权重 + vLLM 路径（Kimi Linear、Jamba、Jet-Nemotron、HypeNet），避免自研 CUDA 内核。
- [ ] 若改造自有模型，先用 HALO/PostNAS 做注意力层选择，再决定转换比例。
- [ ] 关注位置编码（NoPE / RoPE / HyPE）对长度外推的影响并做外推测试。
- [ ] 预留数值稳定性方案（如 Jamba 的 Activation Loss）与量化回归测试。

---

## 7. 开放问题与未来展望

1. **精确召回仍是试金石**。固定大小状态在理论上无法无损保存任意长历史（正交向量上限、记忆过载）。KDA 等在基准上大幅改善，但真实、对抗性的大海捞针与多跳检索仍需在更大规模、更多场景中持续验证。
2. **状态容量与状态更新的极限**。`d²` 矩阵状态 + 更强更新（Delta、门控 Delta、TTT-MLP）能逼近全注意力到什么程度？更高维/结构化状态（如三维状态、Mixture-of-Memories）是否带来下一阶收益，是活跃方向。
3. **非线性状态更新的并行化**。TTT-MLP 等更强更新受限于内存 I/O；如何在保持表达力的同时获得硬件高效的并行形式，仍未完全解决。
4. **与 RL / 推理模型的协同**。Kimi Linear 特别报告了在 RL regime 下同样反超全注意力——线性/混合架构对长链路思维链、Agent 多轮轨迹的影响值得深入研究（与本知识库《Agentic RL》报告形成呼应）。
5. **蒸馏与架构创新的归因**。如何干净地分离「蒸馏自前沿模型」与「新架构本身」的贡献，需要更严格的消融协议。
6. **位置编码与长度外推**。NoPE、RoPE、HyPE 各有取舍；混合架构中注意力层用 RoPE、RNN 层用 NoPE 的不一致如何系统性解决，仍在演进。
7. **生态与标准化**。内核（flash-linear-attention、各家 vLLM 插件）、量化（ExpertsInt8、MXFP4）、训练并行（Ring/LASP）正在快速标准化；一旦工具链成熟，混合架构可能成为新默认。
8. **与扩散语言模型等新范式的关系**。扩散语言模型解决「生成方向」问题，线性/混合架构解决「序列算子复杂度」问题，二者正交，未来可能组合（参见本知识库《扩散语言模型》报告）。

**趋势研判**：线性/混合架构不会「彻底取代」Softmax 注意力——少数注意力层在可预见未来仍是精确召回的保障；但「**多数线性层 + 少数注意力层**」很可能成为下一代基础模型的默认骨架，KDA 从 3B 验证到 2.8T 旗舰的 9 个月，已经给出了强烈信号。

---

## 8. 结论

1. 线性注意力的本质是**带矩阵值状态的线性 RNN**，靠结合律把二次注意力改写为线性递推；现代分块并行与并行扫描让它「训练像注意力、推理像 RNN」。
2. 进步的主线是**状态更新规则的进化**：从「只能加」（朴素线性）到「能遗忘」（门控）、「能按误差改写」（Delta）、「门控 + Delta」（GDN / Gated DeltaNet / KDA），再到「通用学习」（TTT/Titans）。更新越强，固定状态下的召回越接近全注意力。
3. **混合是工程最优解**：保留约 1/8–1/4 注意力层负责精确召回，多数线性层负责吞吐。MiniMax-01、Jamba、Kimi Linear 已在数百 B 到万亿级规模验证。
4. **Kimi Linear/KDA 是标志性拐点**：首次在公平对比下全面超越全注意力（KV −75%、1M 解码 6×），并被放大到 2.8T 的 K3；配合开源权重与内核，标志该方向从「备选」进入「主战场」。
5. 对从业者：默认关注混合架构，用低成本合成基准初筛、在自有负载上联合压测、优先走开源 + 蒸馏改造路径，并把精确召回与长度外推作为验收红线。

---

## 9. 参考文献

> 以下 27 篇文献编号均已通过 arxiv.org 摘要页逐一核实（标题、作者、日期对应一致），核实时间 2026-10-15。

1. Vaswani et al. *Attention Is All You Need.* arXiv:1706.03762, 2017. https://arxiv.org/abs/1706.03762
2. Katharopoulos et al. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* arXiv:2006.16236, 2020. https://arxiv.org/abs/2006.16236
3. Schlag, Irie & Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* arXiv:2102.11174, 2021. https://arxiv.org/abs/2102.11174
4. Sun et al. *Retentive Network: A Successor to Transformer for Large Language Models.* arXiv:2307.08621, 2023. https://arxiv.org/abs/2307.08621
5. Peng et al. *RWKV: Reinventing RNNs for the Transformer Era.* arXiv:2305.13048, 2023. https://arxiv.org/abs/2305.13048
6. Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752, 2023. https://arxiv.org/abs/2312.00752
7. Yang et al. *Gated Linear Attention Transformers with Hardware-Efficient Training (GLA).* arXiv:2312.06635, 2023. https://arxiv.org/abs/2312.06635
8. Qin et al. *Hierarchically Gated Recurrent Neural Network for Sequence Modeling (HGRN).* arXiv:2311.04823, 2023. https://arxiv.org/abs/2311.04823
9. Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* arXiv:2312.04927, 2023. https://arxiv.org/abs/2312.04927
10. Dao & Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2).* arXiv:2405.21060, 2024. https://arxiv.org/abs/2405.21060
11. Qin et al. *HGRN2: Gated Linear RNNs with State Expansion.* arXiv:2404.07904, 2024. https://arxiv.org/abs/2404.07904
12. Lieber et al. *Jamba: A Hybrid Transformer-Mamba Language Model.* arXiv:2403.19887, 2024. https://arxiv.org/abs/2403.19887
13. De et al. *Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.* arXiv:2402.19427, 2024. https://arxiv.org/abs/2402.19427
14. Beck et al. *xLSTM: Extended Long Short-Term Memory.* arXiv:2405.04517, 2024. https://arxiv.org/abs/2405.04517
15. Yang et al. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length (DeltaNet).* arXiv:2406.06484, 2024（NeurIPS 2024）. https://arxiv.org/abs/2406.06484
16. Sun et al. *Learning to (Learn at Test Time): RNNs with Expressive Hidden States (TTT).* arXiv:2407.04620, 2024. https://arxiv.org/abs/2407.04620
17. Jamba Team. *Jamba-1.5: Hybrid Transformer-Mamba Models at Scale.* arXiv:2408.12570, 2024. https://arxiv.org/abs/2408.12570
18. Yang, Kautz & Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule (GDN).* arXiv:2412.06464, 2024. https://arxiv.org/abs/2412.06464
19. MiniMax et al. *MiniMax-01: Scaling Foundation Models with Lightning Attention.* arXiv:2501.08313, 2025. https://arxiv.org/abs/2501.08313
20. Behrouz et al. *Titans: Learning to Memorize at Test Time.* arXiv:2501.00663, 2025. https://arxiv.org/abs/2501.00663
21. Peng et al. *RWKV-7 "Goose" with Expressive Dynamic State Evolution.* arXiv:2503.14456, 2025. https://arxiv.org/abs/2503.14456
22. Peng et al. *Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence (RWKV-5/6).* arXiv:2404.05892, 2024. https://arxiv.org/abs/2404.05892
23. Kimi Team. *Kimi K2: Open Agentic Intelligence.* arXiv:2507.20534, 2025. https://arxiv.org/abs/2507.20534
24. Gu et al. *Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search.* arXiv:2508.15884, 2025（NeurIPS 2025）. https://arxiv.org/abs/2508.15884
25. Kimi Team. *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692, 2025. https://arxiv.org/abs/2510.26692
26. Arora et al. *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff.* arXiv:2402.18668, 2024. https://arxiv.org/abs/2402.18668
27. Chen et al. *Hybrid Linear Attention Done Right: Efficient Distillation and Effective Architectures for Extremely Long Contexts (HALO/HypeNet).* arXiv:2601.22156, 2026. https://arxiv.org/abs/2601.22156

### 参考的其他信源（非 arXiv）

- Songlin Yang. *DeltaNet Explained (Part I–III)*（作者官方博客，含线性注意力/DeltaNet 推导与 MAD/MQAR 数据）. https://sustcsonglin.github.io/blog/2024/deltanet-1
- flash-linear-attention 开源库（GLA / DeltaNet / Gated DeltaNet / KDA 等内核实现）. https://github.com/sustcsonglin/flash-linear-attention
- RWKV 官方 Wiki（RWKV-4/5/6/7 版本、论文链接与模型状态）. https://wiki.rwkv.com
- NVIDIA / NVlabs. *Jet-Nemotron* 官方仓库（PostNAS/JetBlock 说明与吞吐数据，厂商口径）. https://github.com/NVlabs/Jet-Nemotron
- Developers Digest. *Kimi Linear: An Attention Architecture That Outperforms Full Attention*（HN 讨论与 K3 采用 KDA 的社区分析，二次信源）. https://www.developersdigest.tech/blog/kimi-linear-attention-architecture-hn-analysis
