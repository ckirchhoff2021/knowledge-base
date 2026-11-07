# Muon 优化器前沿技术报告：矩阵正交化与新一代大模型训练底座

> **主题**：当 AdamW 不再是唯一答案——从 Keller Jordan 的 Muon 博客，到 Moonlight 的 ~2× 算力效率，再到万亿参数 Kimi K2 的 MuonClip，系统梳理「矩阵正交化」优化器家族的原理、规模化工程、生态格局与开放争议。
>
> **写作日期**：2026-11-08 ｜ **信源纪律**：全部 arXiv 编号经 `export.arxiv.org` API 逐一核实；产业事实以一手技术报告/官方博客为准；对厂商自报分数明确标注。

---

## 摘要

2024 年 10 月，Keller Jordan 等人发表博客《Muon: An optimizer for hidden layers in neural networks》，提出一种把权重矩阵的**动量做正交化（orthogonalization）**后再更新的优化器。它不再像 Adam 那样逐元素自适应、维护一阶与二阶两个矩，而是直接利用权重的二维矩阵结构，通过 Newton–Schulz 迭代近似计算矩阵极分解（polar decomposition），让更新矩阵的奇异值谱趋于均衡。

短短一年内，这条路线完成了「小模型玩具 → 大模型标配」的跨越：

- **Moonshot AI（月之暗面）** 在 2025 年 2 月的论文《Muon is Scalable for LLM Training》中，识别出 **weight decay** 与 **逐参数更新尺度对齐**两个规模化关键，在缩放定律实验中测得 Muon 达到 AdamW 同等性能仅需约 **52% 的训练 FLOPs（约 2× 算力效率）**，并据此训练了 3B/16B 的 MoE 模型 **Moonlight**（5.7T tokens）。
- 2025 年 7 月，Moonshot 在万亿参数（1T 总参 / 32B 激活）的 **Kimi K2** 上进一步提出 **MuonClip** 优化器，用 **QK-Clip** 机制约束注意力 logits 爆炸，在 **15.5T tokens 预训练中实现零 loss spike**。
- 生态侧迅速涌现出 NorMuon、Dion/Dion3、MuonBP、Hierarchical Muon、MuLoCo、Turbo-Muon、Muon^p 等一批变体；理论侧建立了「Muon ≈ 谱范数约束下的最速下降/隐式谱范数正则」的解释，同时也出现了「Muon 在凸 Lipschitz 函数上不收敛」这样的反例性批评。

本报告给出 Muon 的**第一性原理推导**（谱平衡 → Newton–Schulz → 极分解 → 谱范数几何）、**规模化的两笔工程账**（weight decay、consistent update RMS、分布式实现）、**万亿参数稳定性方案**（QK-Clip），以及一张覆盖 2025–2026 全生态的对照表，并诚实讨论其 wall-clock 开销、理论争议与落地边界。

**一句话结论**：Muon 不是「更快的 Adam」，而是把优化的几何单位从「一维向量」换成「二维矩阵算子」；它在 token 效率上已被反复验证，但在单步 wall-clock 成本、跨阶段迁移与理论收敛性上仍是一个快速演进、尚未完全收敛的前沿方向。

---

## 1. 引言：优化器——被忽视的「第四种效率杠杆」

### 1.1 大模型训练的效率账本

训练一个大模型的成本，可以粗略写成：

```
总成本 = 6 × N（参数量） × D（训练 token 数） / (单卡算力 × MFU × 卡数)
```

过去三年，行业提升效率的努力主要集中在三条轴上：

1. **架构**：MoE 把「容量」与「计算」解耦；稀疏注意力把 O(L²) 降下来；线性注意力/Mamba 改变序列建模方式。
2. **数据**：高质量配比、合成数据、数据筛选与课程学习，用更少的 token 达到同等损失。
3. **系统**：量化（FP8/FP4）、FlashAttention、PD 分离、并行策略与编译器，提升 MFU。

**第四条轴——优化器——长期被默认是 AdamW，少有争议。** 自 2014 年 Adam（Kingma & Ba）与 2017 年 AdamW（Loshchilov & Hutter）以来，几乎所有主流大模型（GPT、LLaMA、DeepSeek、Qwen）都以 AdamW 为预训练与后训练的默认优化器。优化器被视为「调个学习率即可」的基础设施，而非效率来源。

Muon 的意义正在于：它重新打开了「优化器本身可以改变 token 效率」的想象空间。如果一个优化器能让模型用一半的 FLOPs 达到同样的损失，其价值等同于一次免费的架构升级。

### 1.2 为什么是 2024–2025 这个时间点

Muon 并非凭空出现，它依赖几条思想线索的汇合：

- **矩阵符号函数 / 极分解的优化视角**：Bernstein 等人 2024 年 9 月的《Old Optimizer, New Norm: An Anthology》（arXiv:2409.20325）把深度学习优化解释为「不同范数约束下的最速下降」，为「正交化更新」提供了几何语言。
- **Newton–Schulz 迭代的廉价化**：用 5 次矩阵多项式迭代即可近似正交化，且可在 bf16/fp16 下运行，使单步开销可控。
- **小模型快速迭代文化**：Keller Jordan 的 modded-nanogpt 等项目让「在几小时内验证一个优化器想法」成为可能，极大加速了优化器实验的循环。

从 2024 年 10 月的博客，到 2025 年 2 月 Moonshot 的规模化论文，再到 2025 年 7 月万亿参数 K2 的工业验证，Muon 用约 9 个月走完了「想法 → 论文 → 旗舰模型」的全过程——这本身也是 2025 年 AI 研究节奏的一个缩影。

### 1.3 本报告的组织与信源纪律

- 第 2 章补齐前置知识（SGD → Adam → AdamW）。
- 第 3 章给出 Muon 的第一性原理。
- 第 4 章拆解规模化的两大关键与分布式实现。
- 第 5 章详解 Kimi K2 的 MuonClip/QK-Clip。
- 第 6 章给出 2025–2026 生态全景。
- 第 7–9 章讨论争议、落地 Checklist 与展望。
- 参考文献中所有 arXiv 编号均经 arXiv 官方 API 核实（截至 2026-11-08），厂商自报的 benchmark 数字均标注「厂商报告」。

---

## 2. 前置知识：从 SGD 到 AdamW，优化器在做什么

### 2.1 一阶方法的基本形态

给定参数 `W` 与损失 `L`，最简单的 SGD 更新为：

```
W_t = W_{t-1} − η · ∇L(W_{t-1})
```

带动量的 SGD 积累历史梯度：

```
M_t = μ · M_{t-1} + ∇L(W_{t-1})
W_t = W_{t-1} − η · M_t
```

动量的作用是平滑随机噪声、并在一致方向上加速。

### 2.2 Adam：逐元素自适应

Adam 在动量（一阶矩）之外，再维护一个梯度平方的指数滑动平均（二阶矩），为**每个标量参数**自适应地调整步长：

```
m_t = β1 · m_{t-1} + (1−β1) · g_t          # 一阶矩
v_t = β2 · v_{t-1} + (1−β2) · g_t²         # 二阶矩
W_t = W_{t-1} − η · m̂_t / (√v̂_t + ε)      # m̂, v̂ 为偏差修正
```

直觉上，梯度大的方向步长被压低，梯度小的方向步长被放大，从而在不同参数方向之间取得平衡。

### 2.3 AdamW：解耦权重衰减

原始 Adam 把 L2 正则混在梯度里，与自适应步长耦合后效果失真。AdamW（Loshchilout & Hutter, 2017）把**权重衰减解耦**：

```
W_t = W_{t-1} − η · ( m̂_t/(√v̂_t + ε)  +  λ · W_{t-1} )
```

这一形式成为大模型时代的事实标准。

### 2.4 Adam 的两个隐含特征（也是 Muon 的攻击点）

理解 Muon，关键是看清 AdamW 的两个隐含设计：

1. **把参数当作「一维向量」处理**：无论权重本身是不是一个有结构的矩阵，Adam 都把它拍平成一串标量，逐元素独立自适应。权重矩阵作为「作用于隐空间的线性算子」的二维结构被完全忽略。
2. **更新方向由逐元素二阶矩决定**：自适应发生在坐标粒度，而不是矩阵算子粒度。

Muon 的核心主张是：**神经网络的权重本质是矩阵（线性算子），优化应当尊重并利用这一结构，在矩阵/算子层面做「自适应」，而非在标量层面。**

---

## 3. Muon 的第一性原理：矩阵正交化

本章是全报告的理论核心。我们依次回答四个问题：什么是正交化？为什么要正交化？如何廉价地实现正交化？它在几何上意味着什么？

### 3.1 更新规则一览

Muon 只作用于可表示为矩阵的参数（如注意力的 Q/K/V/O 投影、MLP 权重、MoE 路由矩阵），更新规则为：

```
M_t = μ · M_{t-1} + ∇L(W_{t-1})           # 梯度动量
O_t = NewtonSchulz(M_t)                   # 正交化（近似极分解）
W_t = W_{t-1} − η_t · O_t                 # 更新
```

与带动量 SGD 的唯一区别，就是多了一步 `NewtonSchulz` 正交化。这一步是 Muon 的全部秘密。

> 图 1 · Muon 的一步更新：把梯度矩阵正交化后再更新权重，以及奇异值谱被拉平的示意。见 diagrams/1-muon-mechanism.html

### 3.2 什么是正交化：极分解与 SVD

对任意矩阵 `M`，做奇异值分解：

```
M = U Σ V^⊤
```

其中 `U`、`V` 的列正交，`Σ` 是对角奇异值矩阵。**矩阵极分解（polar decomposition）**定义为：

```
polar(M) = U V^⊤
```

等价地：

```
polar(M) = (M M^⊤)^{−1/2} M
```

`polar(M)` 是一个**正交矩阵（列/行均单位且彼此正交）**，它把 `M` 的所有奇异值都变成 1，只保留方向信息、抹除幅度信息。

Muon 用 `polar(M)` 代替 `M` 作为更新方向，意味着：**无论原始梯度动量的奇异值相差多大，更新矩阵的奇异值都被拉平到 1。**

### 3.3 为什么要正交化：谱平衡与「避免只在少数方向学习」

Jordan 等人的直觉是：如果直接用动量 `M` 更新，当 `M` 的奇异值高度不均（少数方向奇异值极大、多数方向极小）时，权重会**沿着少数主导方向更新**，而大量其他方向几乎得不到学习——这会限制优化的有效维度。

正交化把更新的奇异值谱拉平，确保：

- 更新矩阵**满数值秩（full numerical rank）**；
- 每一个奇异方向都获得同等的更新幅度；
- 优化在更多样的方向上推进。

Moonshot 在 Moonlight 论文中用 **SVD 熵（SVD entropy）** 实证了这一点：对权重矩阵的奇异值分布计算熵，Muon 训练出的权重在几乎所有矩阵组（注意力 QO/KV、专家、共享专家、路由器、dense 层）上 SVD 熵都显著高于 AdamW——即 Muon 权重的谱更「平坦」、方向更丰富。其中 **MoE 路由器权重的 SVD 熵差异最大**，暗示 MoE 类模型从 Muon 中受益更多。论文还观察到 Muon 在**数学与代码任务**上优势尤为明显。

### 3.4 如何廉价实现：Newton–Schulz 迭代

精确计算极分解需要 SVD，代价高且不易并行。Muon 改用 **Newton–Schulz 迭代**近似。先归一化：

```
X_0 = M / ||M||_F          # 使奇异值落入 [0, 1]
```

然后迭代（Muon 实践中用 5 次、bf16/fp16）：

```
X_k = a·X_{k-1} + b·(X_{k-1} X_{k-1}^⊤) X_{k-1}
                + c·(X_{k-1} X_{k-1}^⊤)² X_{k-1}
```

原始 Muon 的系数为 `a = 3.4445, b = −4.7750, c = 2.0315`。

**为什么这个迭代能逼近极分解？** 注意到每次迭代对 SVD 的作用是：奇异向量 `U、V` 保持不变，奇异值被一个奇数次多项式 `p(x) = a·x + b·x³ + c·x⁵` 映射。选择系数使该多项式在 `[0,1]` 上把任意 `x` 都推向 1，则 5 次迭代后：

```
X_5 ≈ U V^⊤ = polar(M)
```

Moonshot 论文中报告：`N=10` 次迭代虽正交化更精确，但并不会带来更好的模型表现，因此取 `N=5` 以节省开销；动量系数取 `0.95`。

> **工程细节**：实践中 Muon 采用 Nesterov 风格动量，即把 `μ·M + g`（而非 `M`）送入 Newton–Schulz。

### 3.5 几何视角：谱范数下的最速下降

Bernstein & Newhouse（2024，arXiv:2409.20325）给出了一个更深刻的统一视角：**不同优化器 = 不同范数约束下的最速下降（steepest descent）。**

- Adam 对应一种动态的、由 max-of-max 范数导出的约束；
- Muon 对应（大 `p` 的）Schatten-`p` 范数；当正交化精确计算时，即为**谱范数（spectral norm，即矩阵最大奇异值）**。

在谱范数单位球内最小化内积 `<G, O>`，解正是：

```
O = argmin_{||O||_2 ≤ 1} <G, O> = U V^⊤
```

权重作为作用于（局部）欧氏空间的线性算子，对其施加的范数理应是**诱导算子范数（谱范数）**，而非把矩阵拍平后的向量范数。在这个意义上，Muon 的范数选择比 Adam 更「名正言顺」。

这一几何视角后来被严格化为「Muon 隐式地在做谱范数约束优化」（见 6.3 节），构成了 Muon 理论的基石。

---

## 4. 规模化：Moonlight 论文的两大关键与分布式实现

Jordan 最初的 Muon 在小规模语言模型上效果惊艳，但能否扩展到十亿、百亿参数与万亿 token，尚未被证明。Moonshot AI 的《Muon is Scalable for LLM Training》（arXiv:2502.16982，2025-02-24）系统性地回答了这个问题。他们发现，原始 Muon 直接放大后增益会消失，原因与两个关键修复有关。

### 4.1 关键一：加入 Weight Decay

原始 Muon **没有 weight decay**。Moonshot 发现，当模型变大、token 变多时，权重和层输出的 RMS 持续增长，超出 bf16 的高精度范围，损害性能。

把 AdamW 式的解耦 weight decay 引入 Muon：

```
W_t = W_{t-1} − η_t · ( O_t + λ · W_{t-1} )
```

实验（800M 参数、100B tokens，约 5× 过训练）显示：

- 无 weight decay 的原版 Muon 初期收敛更快，但随训练推进部分权重增长过大，长期性能受限；
- 加 weight decay 后，在过训练区间的验证损失同时优于原版 Muon 与 AdamW。

### 4.2 关键二：Consistent Update RMS（逐参数更新尺度对齐）

Adam 的一个隐含性质是更新 RMS（均方根）理论上维持在约 1 附近（实际因 β1<β2、ε>0 通常小于 1）。而 Muon 的更新 RMS 随参数矩阵形状变化。论文给出引理：

> **引理**：对形状为 `[A, B]` 的满秩矩阵参数，其理论 Muon 更新 RMS 为 `√(1/max(A,B))`。

这在放大时会造成两类问题：

- 当 `max(A,B)` 过大（如 dense MLP 大矩阵），更新过小，限制表示能力；
- 当 `max(A,B)` 过小（如把 GQA/MLA 的每个 KV head 当独立参数），更新过大，导致训练不稳定。

**修复方法**：对每个矩阵的 Muon 更新乘以 `√max(A,B)`，抵消引理效应，使不同形状矩阵的更新 RMS 一致。论文比较了三种实现（Baseline、Update Norm、Adjusted LR），最终选用成本最低的 **Adjusted LR**：

```
W_t = W_{t-1} − η_t · ( 0.2 · O_t · √max(A,B) + λ · W_{t-1} )
```

其中 `0.2` 是为了把 Muon 的更新 RMS 匹配到 AdamW 实测的 `0.2~0.4` 区间。这样做的额外好处是：**Muon 可以直接复用为 AdamW 调好的学习率与 weight decay，无需重新调参**，从而「开箱即用」。

### 4.3 缩放定律：约 2× 算力效率

论文在 Llama 架构的一系列 dense 模型上（399M→1.5B，8K 上下文）做了严格的缩放定律实验，AdamW 基线经过网格搜索保证公平。拟合曲线为：

| 优化器 | LM loss（seqlen=8K） |
|---|---|
| Muon | `2.506 × C^{−0.052}` |
| AdamW | `2.608 × C^{−0.054}` |

核心结论：**在 compute-optimal 设置下，Muon 达到 AdamW 同等性能只需约 52% 的训练 FLOPs（约 2× 算力效率）。**

### 4.4 Moonlight：把规模化跑通的模型

基于上述改进，Moonshot 训练了 **Moonlight**：采用 deepseek-v3-small 架构，2.24B 激活 / 15.29B 总参数（含 emb 为 3B/16B），用 Muon 训练 5.7T tokens。

在 1.2T token 的同架构对照中（Moonlight-A 为 AdamW 版本），Muon 版本在代码（HumanEval 37.2 vs AdamW 29.3）、数学（MATH 19.8 vs 16.1）上明显领先。完整训练后的 Moonlight 在 MMLU（70.0）、BBH（65.2）等指标上，以小得多的训练 FLOPs 逼近或超越同尺度模型，推进了「性能–训练 FLOPs」的 Pareto 前沿。

### 4.5 分布式 Muon：ZeRO-1 风格的工程实现

Muon 需要完整梯度矩阵才能做正交化，因此朴素 ZeRO-1（按元素切分优化器状态）不直接适用。Moonshot 提出 **Distributed Muon**：

1. 在 DP 组上 `reduce-scatter` 梯度；
2. 用本地分片动量更新；
3. **DP Gather**：把分片梯度聚合成完整梯度矩阵；
4. 在完整矩阵上跑 Newton–Schulz；
5. 只保留本地分片对应的更新，丢弃其余；
6. `all-gather` 更新后的参数。

其资源账为：

- **显存**：Muon 只有一个动量缓冲，AdamW 有两个，故 Muon 优化器额外显存约为 AdamW 的一半；
- **通信**：bf16 下新增 gather 使总通信量为 AdamW 的 `(1, 1.25]` 倍（多 DP 时接近下界 1）；
- **延迟**：Newton–Schulz 只占前后向时间的约 1%~3%，且可通过计算/通信重叠进一步隐藏，实际集群中相对 AdamW 无明显延迟开销。

---

## 5. 万亿参数稳定性：Kimi K2 与 MuonClip

如果说 Moonlight 证明了 Muon 能在 16B MoE 上规模化，那么 2025 年 7 月的 **Kimi K2**（arXiv:2507.20534）则把它推到了**万亿参数**这一前所未有的尺度，并暴露出一个新问题：注意力 logits 爆炸。

### 5.1 K2 的架构与训练设定

- **MoE 架构**：1T 总参数 / 32B 激活；384 个专家，每 token 选 8 个；61 层（仅 1 层标准 dense）；隐维 7168；MLA 注意力；词表 16 万。
- **预训练**：15.5T tokens，global batch 67M tokens，WSD 学习率调度（前 10T 恒定 2e-4，后 5.5T cosine 衰减），weight decay 0.1；末段退火 + YaRN 把上下文扩展到 128K。

### 5.2 新问题：Muon 训练中的注意力 logits 爆炸

Moonshot 用原版 Muon 训练一个 9B 激活/53B 总参的中等 MoE 时发现：**最大注意力 logits 很快超过 1000**，进而引发明显的 loss spike，偶尔发散。这是 Moonlight 规模上未充分暴露、但在更大尺度上成为主要矛盾的稳定性问题。

### 5.3 QK-Clip：在更新后约束 Q/K 权重

Moonshot 提出 **QK-Clip**：以本 batch 的最大注意力 logit 为引导信号，当其超过阈值 τ 时，在**优化器更新之后**重缩放 query/key 投影权重。

定义每头最大 logit：

```
S_max^h = (1/√d) · max_{X∈B} max_{i,j} Q_i^h (K_j^h)^⊤
```

朴素做法对所有头统一裁剪：

```
W_q^h ← γ^α · W_q^h
W_k^h ← γ^{1−α} · W_k^h
γ = min(1, τ/S_max),   α 通常取 0.5
```

实践中只有少数头 logits 爆炸，因此采用**逐头缩放** `γ_h = min(1, τ/S_max^h)`，把对训练的干预降到最小。对 MLA 则区分处理：头专属的 qC/kC 各乘 `√γ_h`，头专属旋转 qR 乘 `γ_h`，而共享旋转 kR 保持不动以免跨头影响。

关键点：**QK-Clip 不改变当前步的前向/反向计算**，只是用最大 logit 作为「控制权重增长力度」的引导信号。

### 5.4 MuonClip 与零 loss spike

Moonshot 把 **Muon（含 weight decay、consistent RMS）+ QK-Clip** 整合为单一优化器 **MuonClip**（K2 取 `τ=100`）。实测：

- 训练初期最大 logits 被钳制在 100，约 30% 训练步后自然衰减到稳定区间；
- **15.5T tokens 全程训练 loss 平滑、无任何 spike（未经平滑/子采样的原始曲线）**。

Moonshot 报告（厂商报告口径）K2 在非思考模型中达到开源 SOTA：Tau2-Bench 66.1、ACEBench(En) 76.5、SWE-Bench Verified 65.8、SWE-Bench Multilingual 47.3。这些数字是厂商在特定采样/工具设置下的结果，横比时需注意口径。

### 5.5 为什么 K2 的成功是「多重因素的巧合」

后来 Dion3 论文（见 6.2）复盘指出，K2 能把 Muon 规模化，其实依赖一组有利条件的对齐：

1. 早期 PyTorch/Megatron-LM 的 DP 切分策略恰好对 Muon 友好（大连续 buffer、仅跨 DP 边界的张量需额外 gather）；
2. 细粒度 MoE + 仅 1 个 dense 层，使万亿参数尺度下大多数矩阵仍然很小，Newton–Schulz 开销可控；
3. 流水线并行 + 专家并行的主策略天然以最小通信把 Muon 计算分布到各设备。

这意味着 K2 的成功路径并不完全通用——这也是后续一批工作（Dion3、Hierarchical Muon 等）要解决「让 Muon 在任意架构/并行策略下都便宜」的原因。

---

## 6. 生态全景：2025–2026 的变体、理论与竞争格局

Muon 的火爆催生了一个快速膨胀的生态。我们把代表性工作分为四组：正交化本体改进、更小/更便宜的更新、分布式与通信、理论与批评。

> 图 2 · 从博客到万亿参数的规模化时间线与 2025–2026 生态四支线。见 diagrams/2-scaling-landscape.html

### 6.1 正交化本体的改进（不改更新结构，只改正交化步骤）

| 工作 | arXiv | 核心思想 |
|---|---|---|
| **Polar Express**（Amsel et al., ICLR 2026） | OpenReview yRtgZ1K8hO | 最优矩阵符号方法及其在 Muon 上的应用，给出更优系数 |
| **Chebyshev 多项式加速**（Grishina et al.） | 2506.10935 | 用 Chebyshev 型多项式加速 Newton–Schulz 正交化 |
| **Turbo-Muon**（Boissin et al.） | 2512.04632 | 近正交预条件，加速基于正交性的优化 |
| **Muon^p** | 2606.13867 | 引入分数谱幂（fractional spectral powers），推广谱变换 |
| **系数微调**（Cesista/Jordan/You） | 博客 | 通过优化 Newton–Schulz 系数再挤出 1–2% 效率 |

### 6.2 更小、更便宜的更新（缩小正交化的输入）

| 工作 | arXiv | 核心思想 |
|---|---|---|
| **Dion**（Ahn et al., Essential AI） | 2504.05295 | 分布式正交更新：对动量做低秩近似 + 误差反馈（error feedback），只正交化小矩阵 |
| **Trion**（Modoranu et al., ICLR 2026） | OpenReview TkHjRwbMNl | 用 DCT 列做动态子空间选择，比幂迭代更简单却更好 |
| **Dion3**（Amsel et al., Microsoft/NYU/Princeton） | 2608.11612 | 全栈优化：Gram Newton–Schulz + CuteDSL 对称核 + 行子采样 + megabatching，优化器步时间最多降 6× |
| **MuonBP**（Khaled et al., ICLR 2026） | OpenReview mHouLSUQP5 | 块周期正交化：交替使用块级与全量步骤 |
| **Hierarchical Muon**（Tang et al.） | 2606.27216 | 分块（tiled）Newton–Schulz 更新 |
| **SCALE 极简设计**（Glentis et al.） | 2506.16659 | 列归一化 + 策略性动量分配，以更低显存达到与 Muon 相当甚至更好的效果 |
| **NorMuon**（Chen et al.） | 2510.05491 | 让 Muon 更高效、可扩展，对正交化与归一化做组合改进 |
| **Practical Efficiency of Muon**（Lingle et al.） | 2505.02222 | 系统评估 Muon 预训练的实际（wall-clock）效率，提示 token 效率与时间效率的差距 |

**Dion3 值得单独展开**，因为它直接攻击 Muon 最大的工程痛点——Newton–Schulz 的立方时间开销与分片通信。四项叠加改进：

1. **Gram Newton–Schulz**：不在大矩阵上迭代，而在小的对称 Gram 矩阵 `XX^⊤` 上迭代，数学等价但大幅减少 FLOP（典型 α=4 时比标准实现省 55%~68%）；
2. **CuteDSL 对称 GEMM 核**：利用 Gram 矩阵的对称性，Hopper/Blackwell 上相对 cuBLAS 约 2×；
3. **行子采样更新**：每步只选一部分（建议 1/4 或 1/8）L1 范数最大的行做正交化与更新，其余行不动，并配合误差反馈；
4. **Megabatching**：把同形状矩阵打包进单次 all-to-all，把通信轮数降到 O(1)。

Dion3 报告：在 7B/4×GH200 上把 Muon 优化器步时间从 **26× AdamW 降到 4× AdamW**，大模型上整体约 6× 加速；且意外发现调对后的行子采样（f=1/8）损失还略低于 NorMuon。其学习率迁移规则为 `η ∝ 1/√f`。

### 6.3 理论：谱范数约束与收敛性

| 工作 | arXiv | 贡献 |
|---|---|---|
| **Muon Optimizes Under Spectral Norm Constraints**（Chen, Li, Liu） | 2506.15054 | 证明带解耦 weight decay 的 Muon 隐式求解 `‖W‖₂ ≤ 1/λ` 的约束优化，经凸对偶/KKT 连接核范数更新与谱范数约束 |
| **A Note on the Convergence of Muon**（Li & Hong） | 2502.02900 | 早期收敛性分析 |
| **On the Convergence Analysis of Muon** | 2505.23737 | 收敛性分析 |
| **On the Convergence of Muon and Beyond** | 2509.15816 | 推广的收敛性结果 |
| **Gluon**（Riabinin et al.） | 2505.13416 | 桥接 LMO 类优化器的理论与实践，给出逐层平滑度/自适应步长 |
| 有限步 Newton–Schulz 的保证（多篇, 2026） | 2610.03306 / 2609.39595 / 2610.34158 | 研究有限次迭代 + Nesterov 动量下的训练损失保证与平稳性 |

### 6.4 批评与反例（必须正视）

| 工作 | arXiv | 观点 |
|---|---|---|
| **Muon Does Not Converge on Convex Lipschitz Functions** | 2605.08980 | 给出凸 Lipschitz 函数上 Muon 不收敛的反例，质疑其普适收敛性 |
| **A Quadratic Lens on Muon's Polar Update** | 2610.30280 | 二次问题视角下分析极更新的收益、陷阱与 Newton–Schulz 的角色 |
| **When Does Muon Help Agentic RL?** | 2607.16169 | 实证 Muon 在智能体 RL 中何时有用、何时无用，提示收益有条件 |

### 6.5 更广阔的优化器赛道：Muon 不是唯一挑战者

Muon 处在一个更大的「后 Adam」优化器复兴浪潮中，理解它也需看到邻居：

- **Shampoo**（Gupta, Koren, Singer，arXiv:1802.09568）：张量预处理的二阶方法，是 SOAP 的前身。
- **SOAP**（arXiv:2409.11321）：用 Adam 改进并稳定 Shampoo。
- **Sophia**（Liu et al.，arXiv:2305.14342）：可扩展的随机二阶优化器。
- **MARS**（Yuan et al.，arXiv:2411.10438）：用方差缩减（variance reduction）提升大模型训练。
- **COSMOS**（arXiv:2502.17410）：混合自适应优化器，在高维子空间用 Muon 类预处理以省显存。
- **MuLoCo**（arXiv:2505.23725）：把 Muon 作为 DiLoCo 的局部/内部优化器，配合误差反馈可把 delta 量化到 2 bit、通信最多省 8×。

---

## 7. 争议与局限：为什么「2× 算力效率」不等于「全面取代 AdamW」

尽管 Muon 的 token 效率被反复验证，但冷静看，它距离「全面取代 AdamW」仍有几个真实障碍。

### 7.1 Token 效率 ≠ Wall-clock 效率

> 图 3 · Muon 的两种效率必须分开看：token/FLOPs 效率明确占优，但单步 wall-clock 更贵，净收益取决于 NS 暴露时间占比。见 diagrams/3-efficiency-tradeoff.html

Muon 的「约 2× 算力效率」是在 **compute-optimal、以训练 FLOPs 计**的缩放定律口径下得到的。但 Newton–Schulz 是 AdamW 所没有的**额外立方时间矩阵运算**，单步 wall-clock 更贵。Dion3 的实测很说明问题：未经优化的 Muon 在 7B 上优化器步时间可达 **26× AdamW**。

优化器步占总训练时间的比例随设置差异极大：Dion3 给出两个案例——Kimi K2 式高效流水线下，Newton–Schulz 仅约占 **2%**（大量被流水线下一阶段反向传播掩盖）；而 Llama3-70B SFT 式小 batch、密集模型场景下可高达 **17%**。因此：

- 在 MoE + 细粒度专家 + 流水并行的场景，Muon 几乎是「免费」的 token 效率；
- 在 dense + 小 batch SFT/RL 场景，正交化开销可能侵蚀甚至抵消其 token 收益。

这也是 Dion3、NorMuon、Hierarchical Muon 等系统工作的价值所在——它们把这笔 wall-clock 账一点点压下来。

### 7.2 预训练–微调的优化器错配

Moonshot 在 Moonlight 论文中报告了一个微妙现象：**预训练与 SFT 用不同优化器时会错配**——Muon 预训练 + Muon SFT 效果最好；但 AdamW 预训练的模型用 Muon 做 SFT，Muon 并不明显优于 AdamW，反之亦然。这意味着：

- 无法简单地把海量的 AdamW 预训练 checkpoint「捡起来」用 Muon 微调就享受红利；
- 优化器选择在训练流水线中具有路径依赖，其机理尚待研究。

### 7.3 理论收敛性的反例

2026 年的《Muon Does Not Converge on Convex Lipschitz Functions》（arXiv:2605.08980）在凸 Lipschitz 函数上给出了 Muon 不收敛的反例。这不否定其在实际大模型上的经验效果，但提醒：Muon 的理论保证是有条件的，「谱范数最速下降」的优雅几何并不能自动推出普适收敛。结合有限精度、有限次 Newton–Schulz 迭代，实际算法与理想化极更新之间仍有差距，这也是 2026 年一批理论工作聚焦的点。

### 7.4 收益是「有条件」的

《When Does Muon Help Agentic RL?》（arXiv:2607.16169）显示 Muon 在智能体 RL 中并非总是有用。结合 Moonlight 论文「数学/代码收益最大」「MoE 路由器谱熵差异最大」的观察，更合理的心智模型是：**Muon 的收益依赖于问题结构（Hessian 是否低秩/分块对角、权重矩阵是否有可利用的二维结构），而非处处占优。**

### 7.5 工程成熟度与生态默认值

AdamW 的最大优势是十年积累的默认超参、框架支持与调试经验。Muon 仍在快速演进，系数、归一化、裁剪阈值、分片策略尚无统一标准，落地需要更强的工程能力。对多数团队而言，「先用好 AdamW + 合理架构/数据」往往比「追新优化器」性价比更高。

---

## 8. 工程落地 Checklist

给打算在训练中尝试 Muon 的团队一份可操作清单：

**A. 先判断是否值得用**

- [ ] 我的场景是不是 **MoE / 细粒度专家 / 流水并行**？若是，Muon 开销占比低、收益大，优先尝试。
- [ ] 我的核心目标是不是 **数学/代码/推理 token 效率**？这些方向经验收益最大。
- [ ] 若我是 **dense 大模型 + 小 batch SFT/RL**，先估算优化器步占比（可能高达十几个百分点），谨慎评估 wall-clock 账。
- [ ] 不要指望在 AdamW 预训练 checkpoint 上直接换 Muon SFT 就有红利（优化器错配）。

**B. 正确配置 Muon**

- [ ] 只对 **2D 矩阵参数**用 Muon；embedding、LM head、RMSNorm gamma、bias 等非矩阵参数继续用 AdamW。
- [ ] 务必加 **weight decay**（Moonlight 经验 0.1），防止权重/层输出 RMS 失控。
- [ ] 做 **consistent update RMS**：更新乘 `0.2 · √max(A,B)`，并据此直接复用 AdamW 的学习率/weight decay。
- [ ] Newton–Schulz 取 **5 次迭代**、半精度（bf16；追求稳定性可试 fp16），动量约 0.95。
- [ ] SwiGLU 的 up/gate、拼接的 QKV 建议**拆开分别正交化**（实测损失更好）；是否按注意力头再拆需按架构实验（GLM 式 MLA 可能适合，常规多头可能变差）。

**C. 大规模稳定性**

- [ ] 监控**每头最大注意力 logit**；超过阈值（如 τ=100）即触发 **QK-Clip** 式更新后重缩放，优先逐头裁剪。
- [ ] MLA 场景区分 qC/kC/qR/kR 分别处理，避免跨头污染。
- [ ] 关注 loss spike 与 grad norm；QK-Clip 不应改变当前步前向/反向。

**D. 分布式与系统**

- [ ] 用 **Distributed Muon / FSDP 方案**：DP gather 完整矩阵 → 正交化 → 只保留本地分片。
- [ ] 采用 **megabatching** 把同形状矩阵打包，降低通信轮数；考虑 Gram Newton–Schulz / 对称核 / 行子采样压缩开销。
- [ ] 跨 pod / 低带宽场景可考虑只同步选中子矩阵（压缩数据并行）或 MuLoCo 式 delta 量化。
- [ ] 用 CUDA Graph 捕获降低行选择带来的大量 kernel launch 开销。

**E. 验证纪律**

- [ ] 与**经过网格搜索的强 AdamW 基线**对比，而非随手调的 AdamW。
- [ ] 同时报告 **token/FLOPs 效率**与**单步 wall-clock**，避免只讲缩放定律。
- [ ] 厂商自报 benchmark 仅作参考，以自己工作负载上的实测为准。

---

## 9. 展望：Muon 之后，优化器往哪走

短期（6–12 个月）可预见的演进方向：

1. **Wall-clock 成本持续被系统工程压平**：Gram Newton–Schulz、对称 GEMM 核、行子采样、megabatching、分块正交化会组合成新的「默认 Muon 实现」，把优化器步占比稳定压到个位数。
2. **Muon 与二阶/预条件方法融合**：COSMOS、NorMuon、Turbo-Muon 已显示「正交化 × 逐神经元自适应 × 预条件」可叠加；Muon 更可能成为混合优化器中的一个组件，而非孤立算法。
3. **理论与实践的缝隙被填补**：有限步 Newton–Schulz、有限精度、Nesterov 动量下的收敛/损失保证，以及对「不收敛反例」边界的刻画，将是理论主线。
4. **从预训练走向 RL/Agent**：Muon 在 agentic RL 中「何时有用」的条件会被更系统地研究，可能与 RL 算法、轨迹结构产生协同。
5. **更激进的几何推广**：Muon^p 的分数谱幂、Schatten 一般范数、Lion-K/LMO 框架下换不同凸映射，可能催生具有不同正则/鲁棒性的新优化器。

**更深一层的判断**：Muon 真正持久的贡献，或许不是某个具体算法，而是重新确立了一条原则——**优化器应当尊重参数的张量/算子结构，在正确的几何空间里做自适应**。无论 Muon 本身是否成为终局，这条原则大概率会沉淀为后 Adam 时代优化器设计的共识。

---

## 参考文献

> 编号核实方式：调用 arXiv 官方 API（`https://export.arxiv.org/api/query`），按 `id_list` 与标题检索双重确认；标题、日期以 API 返回为准（核实日期 2026-11-08）。部分 ICLR/ICML 2026 论文以 OpenReview 链接给出。

### A. Muon 核心论文

1. Keller Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, J. Bernstein. **Muon: An optimizer for hidden layers in neural networks**（博客，2024-10）. https://kellerjordan.github.io/posts/muon/
2. J. Liu, J. Su, X. Yao, Z. Jiang, G. Lai, Y. Du, Y. Qin, W. Xu, E. Lu, J. Yan, Y. Chen, H. Zheng, Y. Liu, S. Liu, B. Yin, W. He, H. Zhu, Y. Wang, J. Wang, M. Dong, Z. Zhang, Y. Kang, H. Zhang, X. Xu, Y. Zhang, Y. Wu, X. Zhou, Z. Yang. **Muon is Scalable for LLM Training**. arXiv:2502.16982（2025-02-24）. https://arxiv.org/abs/2502.16982
3. Kimi Team. **Kimi K2: Open Agentic Intelligence**. arXiv:2507.20534（2025-07-28）. https://arxiv.org/abs/2507.20534
4. J. Bernstein, L. Newhouse. **Old Optimizer, New Norm: An Anthology**. arXiv:2409.20325（2024-09-30）. https://arxiv.org/abs/2409.20325

### B. 变体与系统

5. K. Ahn, B. Xu, Y. Fan, P. Sharma, Z. Zhan, J. Langford. **Dion: Distributed Orthonormalized Updates**. arXiv:2504.05295（2025-04-07）. https://arxiv.org/abs/2504.05295
6. N. Amsel, J. Zhang, K. Ahn, A. Naeimi, A. Feng, B. Chen, T. Dao, J. Langford. **Dion3: Full-stack orthogonal updates**. arXiv:2608.11612（2026-08-12）. https://arxiv.org/abs/2608.11612
7. Z. Tang, T. Xu, Y. Saad, Y. Xi. **Hierarchical Muon: Tiled Newton–Schulz Updates for Efficient Muon Optimization**. arXiv:2606.27216（2026-06-25）. https://arxiv.org/abs/2606.27216
8. E. Grishina, M. Smirnov, M. Rakhuba. **Accelerating Newton–Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials**. arXiv:2506.10935（2025-06-12）. https://arxiv.org/abs/2506.10935
9. T. Boissin, T. Massena, F. Mamalet, M. Serrurier. **Turbo-Muon: Almost-Orthogonal Pre-Conditioning for Fast Muon Updates**. arXiv:2512.04632（2025-12-04）. https://arxiv.org/abs/2512.04632
10. **Muon^p: Muon with Fractional Spectral Powers**. arXiv:2606.13867（2026-06-11）. https://arxiv.org/abs/2606.13867
11. A. Glentis, J. Li, A. Han, M. Hong. **Memory-Efficient LLM Pretraining via Minimalist Optimizer Design**. arXiv:2506.16659（2025-06-20）. https://arxiv.org/abs/2506.16659
12. A. Khaled, K. Ozkara, T. Yu, M. Hong, Y. Park. **MuonBP: Faster Muon via Block-Periodic Orthogonalization**. ICLR 2026. https://openreview.net/forum?id=mHouLSUQP5
13. I. Modoranu, M. Safaryan, E. Schultheis, A. Chumachenko, D. Alistarh. **Trion: FFT-based Dynamic Subspace Selection for Low-Rank Adaptive Optimization of LLMs**. ICLR 2026. https://openreview.net/forum?id=TkHjRwbMNl
14. N. Amsel, D. Persson, C. Musco, R. Gower. **The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm**. ICLR 2026. https://openreview.net/forum?id=yRtgZ1K8hO
14a. **NorMuon: Making Muon more efficient and scalable**. arXiv:2510.05491（2025-10-07）. https://arxiv.org/abs/2510.05491
14b. J. Lingle et al. **Practical Efficiency of Muon for Pretraining**. arXiv:2505.02222（2025-05-04）. https://arxiv.org/abs/2505.02222

### C. 理论、收敛与批评

15. L. Chen, J. Li, Q. Liu. **Muon Optimizes Under Spectral Norm Constraints**. arXiv:2506.15054（2025-06-18）. https://arxiv.org/abs/2506.15054
16. J. Li, M. Hong. **A Note on the Convergence of Muon**. arXiv:2502.02900（2025-02-05）. https://arxiv.org/abs/2502.02900
17. **On the Convergence Analysis of Muon**. arXiv:2505.23737（2025-05-29）. https://arxiv.org/abs/2505.23737
18. **On the Convergence of Muon and Beyond**. arXiv:2509.15816（2025-09-19）. https://arxiv.org/abs/2509.15816
19. A. Riabinin, E. Shulgin, K. Gruntkowska, P. Richtárik. **Gluon: Making Muon & Scion Great Again!（Bridging Theory and Practice of LMO-based Optimizers for LLMs）**. arXiv:2505.13416（2025-05-19）. https://arxiv.org/abs/2505.13416
20. **Muon Does Not Converge on Convex Lipschitz Functions**. arXiv:2605.08980（2026-05-09）. https://arxiv.org/abs/2605.08980
21. **A Quadratic Lens on Muon's Polar Update: Benefits, Pitfalls, and the Role of Newton–Schulz**. arXiv:2610.30280（2026-10-21）. https://arxiv.org/abs/2610.30280
22. **When Does Muon Help Agentic Reinforcement Learning?** arXiv:2607.16169（2026-07-17）. https://arxiv.org/abs/2607.16169
23. **Training-Loss Guarantees for Muon with Finite-Step Newton–Schulz Orthogonalization**. arXiv:2610.03306（2026-10-02）. https://arxiv.org/abs/2610.03306

### D. 相邻优化器与分布式

24. V. Gupta, T. Koren, Y. Singer. **Shampoo: Preconditioned Stochastic Tensor Optimization**. arXiv:1802.09568（2018-02-26）. https://arxiv.org/abs/1802.09568
25. **SOAP: Improving and Stabilizing Shampoo using Adam**. arXiv:2409.11321（2024-09-17）. https://arxiv.org/abs/2409.11321
26. H. Liu, Z. Li, D. Hall, P. Liang et al. **Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training**. arXiv:2305.14342（2023-05-23）. https://arxiv.org/abs/2305.14342
27. H. Yuan, Y. Liu, S. Wu, X. Zhou et al. **MARS: Unleashing the Power of Variance Reduction for Training Large Models**. arXiv:2411.10438（2024-11-15）. https://arxiv.org/abs/2411.10438
28. **COSMOS: A Hybrid Adaptive Optimizer for Memory-Efficient Training of LLMs**. arXiv:2502.17410（2025-02-24）. https://arxiv.org/abs/2502.17410
29. B. Thérien, X. Huang, A. Defazio, I. Rish et al. **MuLoCo: Muon is a practical inner optimizer for DiLoCo**. arXiv:2505.23725（2025-05-29）. https://arxiv.org/abs/2505.23725

### E. 背景与官方一手信源

30. D. Kingma, J. Ba. **Adam: A Method for Stochastic Optimization**. ICLR 2015. https://arxiv.org/abs/1412.6980
31. I. Loshchilov, F. Hutter. **Decoupled Weight Decay Regularization（AdamW）**. ICLR 2019. https://arxiv.org/abs/1711.05101
32. DeepSeek-AI. **DeepSeek-V3 Technical Report**. arXiv:2412.19437（2024-12-27）. https://arxiv.org/abs/2412.19437
33. **GLM-5: From Vibe Coding to Agentic Engineering**. arXiv:2602.15763（2026-02-17）. https://arxiv.org/abs/2602.15763
34. OpenAI. **gpt-oss-120b & gpt-oss-20b Model Card**. arXiv:2508.10925（2025-08-08）. https://arxiv.org/abs/2508.10925
35. Essential AI. **Layer Sharding for Large-Scale Training with Muon**（官方研究博客）. https://www.essential.ai/research/infra
36. Microsoft Research / Dao-AILab. **dion** 与 **gram-newton-schulz**（开源实现）. https://github.com/microsoft/dion ｜ https://github.com/Dao-AILab/gram-newton-schulz

---

> **免责声明**：本报告为技术调研，所引 benchmark 多为厂商或论文自报结果，口径（采样次数、工具设置、评测版本）可能不同；具体选型请以自身工作负载上的可复现实验为准。arXiv 编号虽经 API 核实，但论文可能已有更新版本，请以 arXiv 页面最新版为准。
