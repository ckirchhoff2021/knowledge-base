# Transformer 架构与位置编码（Position Embedding）技术调研报告

> 从《Attention Is All You Need》到 200 万 Token 长上下文：系统拆解 Transformer 架构，重点剖析位置编码（Positional Encoding / Position Embedding）的常用方案、它为什么不可或缺，以及它与上下文长度（Context Length）之间的深层关系。
>
> 📅 2026-09-30 ｜ 信源：9 篇 arXiv 原始论文（编号经 arXiv API 逐一核实）+ 知乎社区高赞文章交叉验证

---

## 目录

1. [背景：从循环网络到注意力范式](#1-背景从循环网络到注意力范式)
2. [Transformer 原理详解](#2-transformer-原理详解)
3. [位置编码为什么重要](#3-位置编码为什么重要)
4. [位置编码常用方案全景](#4-位置编码常用方案全景)
5. [上下文长度与位置编码：外推、内插与长文本扩展](#5-上下文长度与位置编码外推内插与长文本扩展)
6. [核心实现：RoPE 与长上下文扩展伪代码](#6-核心实现rope-与长上下文扩展伪代码)
7. [结论与选型建议](#7-结论与选型建议)
8. [参考资料](#8-参考资料)

---

## 1. 背景：从循环网络到注意力范式

**导读：** Transformer 不是凭空诞生的，它是为了解决 RNN 家族"无法并行、长距离衰减"两大顽疾而提出的。本章先讲清楚旧路线的缺陷，再定位 Transformer 的历史角色。

### 1.1 旧路线的两个结构性缺陷

在 2017 年之前，序列建模（机器翻译、语言模型）的主流骨架是循环神经网络（RNN）及其改进版 LSTM、GRU：

1. **时序依赖，无法并行。** RNN 必须按时间步依次计算——第 t 个词的隐状态 h_t 依赖 h_{t-1}，训练时一个长度为 n 的句子至少要串行 n 步，GPU 的大规模并行算力被浪费。
2. **长距离信息衰减。** 信息要从句子开头传到结尾，必须穿过所有中间隐状态。即使 LSTM 用门控缓解了梯度消失，相距上百个词的两个 token 之间依然难以直接交互。

### 1.2 Transformer 的定位：全连接的"消息传递"

2017 年 6 月，Google 团队发表论文《Attention Is All You Need》（NeurIPS 2017，arXiv:1706.03762），提出了 **Transformer**。其核心主张是：**彻底抛弃循环和卷积结构，序列中任意两个 token 都通过注意力机制直接交互。**

> **直觉类比：** RNN 像一条**流水线传话游戏**——每个人只能把话传给下一个人，传到末尾早已走样；Transformer 像一间**圆桌会议室**——每个人（token）随时可以直接听到房间里任何人的发言，并根据内容决定更关注谁。

这一设计带来三个决定性优势：

| 维度 | RNN / LSTM | Transformer |
|---|---|---|
| 并行性 | 必须按时间步串行 | 所有位置同时计算，全并行 |
| 长距离依赖路径 | O(n)，信息逐级传递 | O(1)，任意两 token 直接相连 |
| 可扩展性 | 难以靠堆 GPU 加速 | 完美适配 GPU/TPU，规模可随算力增长 |

正是这种"可规模化并行"的特性，让 Transformer 走出机器翻译，成为 BERT、GPT 系列、T5、LLaMA、Qwen、DeepSeek 等几乎所有现代大模型的统一底座，并且被推广到视觉（ViT）、语音、多模态、蛋白质结构（AlphaFold）等领域。

### 1.3 本报告聚焦的核心问题

Transformer 的注意力机制有一个"先天盲点"：**它本身看不到 token 的顺序。** 为了解决这个问题引入的"位置编码"，看似只是架构中的小组件，却直接决定了模型能否处理长文本、能否"训短推长"。本报告在讲透整体架构的基础上，重点回答：

- 位置编码注入了什么信息？为什么不可或缺？
- 从正弦编码到 RoPE、ALiBi，各方案的原理与取舍是什么？
- 上下文长度和位置编码到底是什么关系？为什么扩展上下文如此依赖它？

---

## 2. Transformer 原理详解

**导读：** 本章从输入到输出完整走一遍 Transformer：分词与嵌入 → 缩放点积注意力 → 多头注意力 → 前馈网络 → 残差与归一化 → 编解码器结构。

### 2.1 总体结构：Encoder–Decoder

原始 Transformer 为机器翻译设计，采用编码器–解码器（Encoder–Decoder）堆叠结构，两侧各由 N=6 个相同的层堆叠而成：

- **编码器（Encoder）：** 接收完整输入序列，每层包含两个子层——多头自注意力（Multi-Head Self-Attention）和逐位置前馈网络（Position-wise Feed-Forward Network，FFN）。
- **解码器（Decoder）：** 自回归地生成输出序列，每层包含三个子层——**带掩码的**多头自注意力、对编码器输出的交叉注意力（Cross-Attention）、FFN。

现代大模型通常只取其中一半：

| 架构路线 | 来源 | 代表模型 | 特点 |
|---|---|---|---|
| Encoder-only | BERT | BERT、RoBERTa | 双向注意力，适合理解/分类/抽取 |
| Decoder-only | GPT | GPT、LLaMA、Qwen、DeepSeek | 因果掩码，自回归生成，当代 LLM 主流 |
| Encoder-Decoder | 原始 Transformer | T5、BART | 输入双向编码、输出自回归，适合翻译/摘要 |

### 2.2 输入侧：分词与词嵌入

模型不能直接读文字，需要两步转换：

1. **分词（Tokenization）：** 通过 BPE（Byte-Pair Encoding）、SentencePiece 等子词算法，把文本切成 token 序列（子词是介于"字符"和"单词"之间的单元），并映射为整数 id。
2. **词嵌入（Token Embedding）：** 查表把每个 id 映射为一个 d_model 维稠密向量（原始论文 d_model=512）。整个序列表示为矩阵 X，形状为 (n, d_model)。

此时向量只携带"这是什么词"的语义信息，**还不携带"它在第几个位置"的信息**——这正是后面位置编码要补上的。

### 2.3 缩放点积注意力（Scaled Dot-Product Attention）

注意力是 Transformer 的心脏。自注意力（Self-Attention）让序列中的每个 token 与序列内所有 token（包括自己）计算关联程度。它通过三个角色实现：

- **Query（查询，Q）：** 当前 token "想找什么信息"；
- **Key（键，K）：** 每个 token "能提供什么信息"；
- **Value（值，V）：** 每个 token "实际被聚合的内容"。

三者都由输入 X 分别乘以三个可学习矩阵 W_Q、W_K、W_V 得到：

```
Q = X · W_Q      # 形状 (n, d_k)
K = X · W_K
V = X · W_V
```

计算过程分为四步：

1. **打分：** 用 Q 与 K 的点积衡量两两 token 的相似度，得到注意力分数矩阵 Q·K^T（形状 n×n）。
2. **缩放：** 除以 sqrt(d_k)，防止点积随维度变大而数值过大、把 softmax 推入梯度极小的饱和区。
3. **归一化：** 对分数做 softmax，得到每行和为 1 的注意力权重。
4. **加权聚合：** 用权重对 V 加权求和，得到每个位置融合了全局信息的新表示。

```
             Q · K^T
Attention(Q,K,V) = softmax( ----------- ) · V
              sqrt(d_k)
```

### 2.4 掩码（Mask）：因果性与填充

- **因果掩码（Causal / Look-Ahead Mask）：** Decoder-only 模型生成第 t 个 token 时只能看见它之前（含自身）的 token。实现上把注意力分数矩阵的上三角（未来位置）置为 -∞，softmax 后这些位置权重为 0。这保证了自回归生成的合法性。
- **填充掩码（Padding Mask）：** 屏蔽为对齐批次长度而补充的占位符 pad，让模型不把它们当作真实内容。

### 2.5 多头注意力（Multi-Head Attention）

单组 Q/K/V 只能学习一种关联模式。Transformer 把向量切成 h=8 个"头"（原始论文），**每头在独立的低维子空间（d_k = d_model/h = 64）中并行做注意力**，再把各头输出拼接后经线性层 W_O 融合：

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

不同的头可以关注不同类型的关系：有的头关注句法搭配（如主谓一致），有的头关注指代消解或局部相邻词。多头让模型在同一层拥有多副"观察眼镜"。

### 2.6 逐位置前馈网络（FFN）

注意力层负责 token 之间的信息混合，FFN 则对**每个位置独立地**做相同的非线性变换，承担特征加工与知识存储：

```
FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2     # 两层全连接 + ReLU
```

内层维度通常放大 4 倍（512 → 2048）。现代模型用 GELU、SwiGLU 等激活函数替代 ReLU；大模型中大部分参数量集中在 FFN。

### 2.7 残差连接与层归一化

每个子层外都包了一条残差连接（Residual Connection）和层归一化（Layer Normalization），保证深层网络可训练：

```
# Post-LN（原始 Transformer）
output = LayerNorm(x + Sublayer(x))
```

残差让梯度有"高速公路"直达浅层，缓解深层退化。原始论文采用 Post-LN（先残差后归一化），但 Post-LN 对学习率敏感、深层容易训练不稳；现代 LLM（GPT-2 起）普遍改用 **Pre-LN**（先归一化后进子层）：

```
# Pre-LN（现代主流，训练更稳）
output = x + Sublayer(LayerNorm(x))
```

### 2.8 一层 Transformer Block 的完整数据流

```
输入 X
 │
 ├─ ① 层归一化（Pre-LN）
 ├─ ② 投影出 Q / K / V
 ├─ ③（位置编码在此作用于 Q、K，或等效地作用于注意力分数）
 ├─ ④ 多头缩放点积注意力 + 掩码
 ├─ ⑤ 输出投影 W_O
 ├─ ⑥ 残差相加
 ├─ ⑦ 层归一化
 ├─ ⑧ FFN（先升维 4× 再降维）
 └─ ⑨ 残差相加 → 送入下一层
```

解码器最终把最后一层输出经线性层映射到词表大小的 logits，再 softmax 得到下一个 token 的概率分布，以交叉熵（Cross-Entropy）为损失、teacher forcing 方式训练。

---

## 3. 位置编码为什么重要

**导读：** 这是理解整份报告的关键章节。注意力机制天生"无序"，我们先从数学性质讲清它为什么看不见顺序，再说明位置信息具体包含哪三类内容。

### 3.1 根本原因：自注意力是"置换等变"的

自注意力对输入 token 的排列顺序不敏感。把输入序列中任意两个词的位置交换，只要对应地交换输出，结果完全一致——这在数学上称为**置换等变性（Permutation Equivariance）**。

考虑这样一个事实：

```
句子 A：狗咬人
句子 B：人咬狗
```

纯自注意力（无位置编码）在这两个句子上得到的输出集合是**相同**的——因为它本质上只是对一组词向量做了加权求和（一个"词袋"，Bag-of-Words），权重只由词与词的内容相似度决定，与谁在前谁在后无关。但这两个句子的含义天差地别。

> **直觉类比：** 自注意力给你的是一袋打乱的拼图碎片，每块碎片都能看到其他碎片，却不知道各自本该放在哪个坐标。位置编码就是给每块碎片盖上的"坐标印章"。

因此，**必须从外部显式注入位置信息**，Transformer 才能区分词序、理解语法和语义。

### 3.2 位置编码需要承载的三类信息

一个好的位置编码，要让模型回答三个层次的问题：

1. **顺序（Order / 绝对位置）：** 谁在前、谁在后？"我打他"和"他打我"的施受关系由此区分。
2. **相对距离（Relative Distance）：** 两个 token 相隔多远？相邻词往往组成短语（局部性），远距离关联（如代词与先行词）需要被识别但权重通常应更弱。
3. **方向（Direction）：** 是在"左边"还是"右边"？对自回归解码器尤其重要，模型需要知道哪些是历史、哪些是未来。

此外，研究者普遍希望位置编码还满足两个工程性质：

- **平移不变性/相对性：** 同样的相对距离（如相隔 3 个词）在句子开头和句子中间应具有相似的表示；
- **外推性（Extrapolation）：** 在较短序列上训练后，能否直接处理更长的序列。

### 3.3 位置编码与上下文长度的第一层关系

"模型最多能处理多长的序列"（上下文长度）在架构上直接受位置编码约束：

- 位置编码必须为每个可能的位置提供一个表示。若是**可学习的绝对位置编码**，训练时只为前 L 个位置准备了向量，第 L+1 个位置根本"无表可查"。
- 即使是公式生成的位置编码（正弦、RoPE），模型也只在 [0, L) 范围内"见过并学过"如何使用这些位置信号；超出训练长度后，位置信号的分布进入陌生区域，注意力分数可能剧烈异常，导致性能崩塌。

所以，**上下文长度不只是一个能填多少 token 的数字，它首先是位置编码定义并被训练覆盖的范围。** 第 5 章将系统展开如何扩展这个范围。

---

## 4. 位置编码常用方案全景

**导读：** 本章按"绝对位置 → 相对位置 → 旋转编码 → 线性偏置"的演进脉络，逐一剖析主流方案，并在 4.6 给出横向对比表。

### 4.1 正弦绝对位置编码（Sinusoidal PE）

原始 Transformer（2017）采用一组固定（不可学习）的正余弦函数，为位置 pos 生成 d_model 维向量。第 i 维的频率按几何级数变化：

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

**设计直觉：**

- 低维（小 i）对应**高频**信号，波长很短，用于区分邻近位置；
- 高维（大 i）对应**低频**信号，波长很长（最长 10000×2π），用于区分远距离位置。
- 每个位置由此得到一个独一无二的"二进制式"频率指纹。

**关键性质——相对位置可由线性变换表达：** 利用三角恒等式，PE(pos+k) 可以表示为 PE(pos) 的线性函数，因此该编码理论上允许模型学习相对位置关系，并且因为是公式生成的，对更长位置天然有定义（具备初步外推的形式基础）。

**不足：** 它通过"加法"把位置与词义混合（直接加到词向量上），表示相对位置的方式仍然**间接**；实践中直接外推到远超训练长度时效果也会明显下降。

### 4.2 可学习绝对位置编码（Learned / Learnable PE）

BERT、GPT-2 等模型不再用固定公式，而是把位置嵌入做成一张**可学习的查找表**：为最大长度 L_max 内的每个位置设置一个 d_model 维向量，随训练一起更新。

```
最终输入 = TokenEmbedding(token_id) + PositionEmbedding(position_id)
其中 position_id ∈ {0, 1, ..., L_max - 1}
```

**优点：** 简单、灵活，模型可以自由学习在特定位置上"什么表示最有用"，在训练长度内效果好。

**缺点：**

1. **硬长度上限：** 表只覆盖到 L_max（如 BERT 的 512、GPT-2 的 1024），超出后没有对应向量，无法直接处理。
2. **缺乏相对位置归纳偏置、几乎不能外推：** 每个位置向量独立学习，位置 1025 完全陌生，泛化到更长序列需要额外训练。

### 4.3 相对位置编码（Relative Position Encoding）

核心思想转变：**不关注"我在第几位"，只关注"我和你相隔多远"。** 这更符合语言的局部性，也天然具备平移不变性。

**代表一：Shaw et al.（2018，arXiv:1803.02155）。** 在计算注意力时，不仅用语义上的键，还为每种相对距离（裁剪到最大窗口 k 内）引入可学习的相对向量，分别作用于注意力权重和加权内容。

**代表二：T5 的相对位置偏置（Raffel et al., 2019，arXiv:1910.10683）。** 在注意力分数上、softmax 之前加一个**只取决于相对距离桶（bucket）的可学习标量偏置**，并对每个头使用不同偏置。距离被分桶（近距离细粒度、远距离对数粗粒度），超长距离归入同一个桶。

```
Attention(Q,K,V) = softmax( Q·K^T / sqrt(d_k) + B ) · V
其中 B[i][j] = b_{ bucket( relative_position(i,j) ) }   # 每个头一套
```

**优点：** 直接编码相对距离、平移不变、效果稳定。
**局限：** 依赖固定的桶/裁剪窗口，超出分桶范围后位置区分度饱和，外推到极长序列仍有挑战；且是在分数上打补丁，与词向量的位置信息分离。

### 4.4 旋转位置编码（RoPE, Rotary Position Embedding）

RoPE 由苏剑林（追一科技）在 2021 年论文《RoFormer》（arXiv:2104.09864）中提出，是**当代主流大模型（LLaMA、Qwen、DeepSeek、GPT 系列多数开源模型等）的事实标准**。它的出发点极具巧思：**用绝对位置编码的形式，实现相对位置编码。**

#### 4.4.1 核心思想：对 Q、K 做"按位置的旋转"

注意力权重由 q·k（Query 与 Key 的内积）决定。如果在做内积之前，先根据位置对 q 和 k 各自做一个旋转变换，让旋转角依赖位置，那么内积就会自然带上相对位置信息：

```
q_m' = R(m) · q_m      # 把位置 m 的 Query 旋转角度 mθ
k_n' = R(n) · k_n      # 把位置 n 的 Key   旋转角度 nθ
```

借助复数/旋转的性质，变换后二者的内积**只依赖于相对位置 (m − n)**：

```
q_m' · k_n'  =  q_m · R(m - n) · k_n
```

这就是 RoPE 最关键的恒等式：**通过对绝对位置的旋转，在注意力内积中得到了纯粹的相对位置依赖。**

#### 4.4.2 具体实现：两两成对，角度频率递减

把 d 维向量的相邻两维视为一个二维平面上的坐标 (x_{2i}, x_{2i+1})，在每个二维平面内旋转。第 i 个平面对应的角频率为：

```
θ_i = base ^ ( -2i / d ),   i = 0, 1, ..., d/2 - 1,   通常 base = 10000

对位置 pos：
[x_{2i}'  ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [x_{2i}  ]
[x_{2i+1}'] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [x_{2i+1}]
```

实际工程中常用等价的高效写法：

```
x' = x · cos(pos·θ) + rotate_half(x) · sin(pos·θ)
```

频率结构与正弦编码一脉相承：**低维平面高频（管近处），高维平面低频（管远处）**；最高频波长很短，最低频波长约为 base×2π。

#### 4.4.3 为什么 RoPE 成为主流

1. **相对位置天然编码：** 内积只依赖相对位移，平移不变，符合语言规律。
2. **远程衰减：** 位置差越大，内积中旋转带来的周期性抵消越多，注意力自然随距离衰减，形成"软窗口"，无需额外惩罚项。
3. **不增加可学习参数、即插即用：** 旋转矩阵完全由位置和维度决定，可直接施加在已有的 Q、K 上。
4. **良好的可扩展基础：** 通过缩放角度（PI/NTK/YaRN，见第 5 章）就能低成本扩展上下文，工程生态成熟。

### 4.5 线性偏置（ALiBi, Attention with Linear Biases）

ALiBi 由 Press et al. 在 2021 年提出（ICLR 2022，arXiv:2108.12409），标题即点明卖点——**Train Short, Test Long（训短测长）**。

它**完全不使用位置嵌入**：不向词向量加位置，也不旋转 Q、K，而是在注意力分数上、softmax 之前，直接加一个与距离成正比的**惩罚项**：

```
Attention(Q,K,V) = softmax( Q·K^T / sqrt(d_k)  +  m_h · (i - j) ) · V
```

- (i − j) 是 query 与 key 的相对距离（key 在越远的"左边"，惩罚越大）；
- m_h 是每个注意力头各自固定的斜率（slope），不同头有不同强度，按几何级数选取，类似一组不同宽度的"软窗"；
- 惩罚为负，距离越远，注意力分数被压得越低。

**最大优势——极强的外推性：** 因为惩罚是一条随距离线性延伸的简单直线，模型在短上下文中学到的"距离—惩罚"规律可以直接延伸到训练时从未见过的远处，从而在不微调或少量微调的情况下处理更长输入。

**局限：** 强烈的局部偏置可能在一定程度上限制对远距离信息的利用；超长上下文场景中，线性惩罚如何与稀疏/全局注意力配合需要额外设计。Baichuan2-13B 等模型曾采用 ALiBi。

### 4.6 横向对比总表

| 方案 | 年份 | 注入位置 | 注入方式 | 额外参数 | 相对位置 | 原生外推性 | 代表模型 |
|---|---|---|---|---|---|---|---|
| Sinusoidal 正弦 | 2017 | 词向量 | 加法（固定公式） | 无 | 间接（线性可表达） | 一般 | 原始 Transformer |
| Learned 可学习 | 2018 | 词向量 | 加法（查找表） | L_max×d | 弱 | 差（硬上限） | BERT、GPT-2 |
| Relative（Shaw / T5 bias） | 2018/2019 | 注意力分数/键 | 相对向量或分桶偏置 | 有 | 直接 | 受桶/窗口限制 | T5、DeBERTa |
| **RoPE 旋转** | 2021 | Q、K | 按位置旋转 | 无 | 直接（内积恒等） | 较好，易扩展 | LLaMA、Qwen、DeepSeek |
| **ALiBi 线性偏置** | 2021 | 注意力分数 | 距离线性惩罚 | 无 | 直接 | **强（训短测长）** | BLOOM、Baichuan2-13B |

> **消歧提示：** "PE"在资料中既可能指 Positional Encoding（位置编码这一机制），也常与 Position Embedding 混用；此外部分资料把"RoPE 缩放类方法（PI/NTK/YaRN）"也笼统称为"位置编码"，阅读时需注意区分"基础编码方案"与"上下文扩展算法"两个层次。

### 4.7 其他值得了解的方案

- **XPos（2022，arXiv:2212.10554，《A Length-Extrapolatable Transformer》）：** 在 RoPE 基础上增加指数形式的位置衰减，进一步改善长度外推。
- **KERPLE：** 把 ALiBi 的固定斜率改为可学习的核化幂律/指数衰减，自适应距离模式。
- **NoPE（无位置编码）：** 部分研究（尤其语音、某些非自回归或数据本身带强顺序结构的场景）尝试不使用显式位置编码；对通用文本 LLM，主流结论仍是需要显式位置信息。
- **CoPE / Sandwich 等：** 在多层或输出侧探索更灵活的位置注入，属研究性变体。

---

## 5. 上下文长度与位置编码：外推、内插与长文本扩展

**导读：** 本章正面回答"上下文长度跟位置编码有什么关系"。长上下文有两个独立瓶颈，位置编码决定其中的"表示"瓶颈；本章讲清外推与内插的分野，以及 PI → NTK → YaRN → LongRoPE 的技术演进和关键数据。

### 5.1 长上下文的两个瓶颈

把模型上下文从 L 扩展到 L'（L' > L）会同时撞上两堵墙：

1. **计算/显存瓶颈：** 标准注意力的计算量和显存随序列长度呈 **O(n²)** 增长（n×n 的注意力矩阵）。这由 FlashAttention、稀疏注意力、滑窗注意力（Sliding Window）、Ring Attention 等技术解决。
2. **位置表示瓶颈（本报告焦点）：** 模型只在位置区间 [0, L) 上被训练过。如何让它在 [0, L') 上仍能正确使用位置信号？这正是 PI/NTK/YaRN/LongRoPE 要解决的问题，**与用什么注意力算法相互独立**。

换言之，即便 FlashAttention 让算力不再是问题，**RoPE 等位置编码若不扩展，模型在 L 之外依然会"读不懂位置"而崩溃。**

### 5.2 外推 vs 内插：两种根本路线

- **外推（Extrapolation）：** 位置 0…L 保持不变，让模型直接面对从未见过的 L…L' 新位置。问题是新位置上的角度/分数进入训练分布之外，易产生灾难性的大注意力分数。
- **内插（Interpolation）：** 把更长的位置坐标**压缩映射**回 [0, L) 的熟悉区间。所有位置都落在模型见过的范围内，因此稳定得多，代价是位置分辨率被稀释（相邻 token 的位置差变小）。

Position Interpolation 论文（arXiv:2306.15595）从理论上给出关键结论：**内插的注意力分数上界比外推小约 600 倍**，从数学上解释了为什么"压缩回去"远比"直接延伸"稳定。

### 5.3 Position Interpolation（PI，线性内插）

做法最简单：对输入位置索引做线性缩放。设扩展倍数 s = L'/L：

```
# 原本
angle = pos · θ_i
# PI：把位置除以 s，等价于把所有角频率除以 s
angle = (pos / s) · θ_i
```

这样最远位置 L' 旋转后的角度恰好等于原来 L 的角度，整段被均匀压回训练区间。论文一手数据（arXiv:2306.15595 摘要）：

- 将基于 RoPE 的 LLaMA（7B 到 65B）上下文扩展到 **32768**，**只需 1000 步以内**的少量微调；
- 在 passkey 检索、语言建模、长文档摘要任务上效果良好，同时**在原始长度内的任务质量基本保持**。

**PI 的问题：** 当扩展倍数很大时，所有频率被等比例压缩，**高频维度也被压慢**，导致相邻位置难以区分——即"高频信息损失"，近处 token 的分辨率下降。

### 5.4 NTK-Aware 插值：高频外推、低频内插

NTK（Neural Tangent Kernel）方案源自社区讨论（Reddit/GitHub，作者正是后来 YaRN 的作者），核心洞察是：**不该对所有频率"一刀切"。**

```
直觉：高频维度负责近距离，要尽量保住分辨率 → 让它近似"外推"
      低频维度负责远距离，本来变化就慢 → 放心"内插"压缩
```

实现上不显式除以 s，而是**修改 RoPE 的底数 base**：

```
base' = base · s ^ (d / (d - 2))
θ_i' = base' ^ (-2i / d)
```

通过放大 base，角频率整体改变，但改变量随维度 i 非均匀：低频维度被大幅压缩（内插），高频维度几乎不动（外推）。这就兼顾了近处的区分度和远处的稳定性。

- **静态 NTK：** base 固定放大，推理即生效，几乎不需微调，适合中小倍数扩展。
- **Dynamic NTK（动态）：** 推理时根据当前序列实际长度动态调整 base，短序列不压缩、长序列才加大缩放，因此能在不损短文本的同时按需外推。Qwen 等模型采用 dynamic NTK，并配合 logN 注意力缩放、滑窗注意力。

**LLaMA 2 Long 的工程做法（改 base 路线）：** 把 base 从 10000 直接放大到 500000，配合长文本继续训练，即可将上下文扩展到 128K。

> ⚠️ 下表为知乎文章转述的论文数据，引用前建议核对原文。来源：知乎《RoPE外推优化——支持192K上下文长度》（作者：绝密伏击），评测样本长度 32k，指标为困惑度 PPL（越低越好）：

| 配置 | Books | CC | Wikipedia |
|---|---|---|---|
| RoPE（base=10000） | 6.548 | 6.816 | 3.802 |

### 5.5 YaRN：分频段 + 注意力温度，训短测长

YaRN（Yet another RoPE extensioN，arXiv:2309.00071）对 NTK 思想做了严谨化和系统化，被 DeepSeek-V3、Qwen 系列、gpt-oss 等当代模型采用。其要点：

1. **分频段（NTK-by-Part / RoPE-by-Part）：** 不再整体改 base，而是把维度按波长分成三段区别对待：
   - **高频维度（波长远小于上下文）：** 基本不缩放，保住近距离分辨能力；
   - **低频维度（波长远大于上下文）：** 完全内插（除以 s）；
   - **中间频段：** 用一条平滑的斜坡函数（ramp）在"不缩放"与"完全内插"之间过渡，避免硬切换造成突变。
2. **注意力温度缩放（Attention Scaling）：** 内插后注意力分布会变得过于平坦，YaRN 引入温度项（与扩展倍数相关，典型为 1/(0.1·ln(s) + 1)），对注意力 logits 重新锐化，恢复模型的区分度。
3. **动态（Dynamic YaRN）：** 推理时按实际长度决定缩放，可在微调长度之外继续外推约 2 倍，实现"train short and test long"。

一手数据（arXiv:2309.00071 摘要原文）：YaRN 扩展上下文窗口时，**比此前方法少用约 10 倍的 token、少约 2.5 倍的训练步数**；用 YaRN 可让 LLaMA 有效利用并外推到远超其原始预训练长度的上下文，且能外推到超出微调数据集长度之外。

### 5.6 LongRoPE：非均匀缩放，迈向 200 万

LongRoPE（2024，arXiv:2402.13753）进一步发现：PI 的"均匀缩放"是次优的，存在两类可利用的**非均匀性**。

1. 通过高效搜索，为每个频率维度找到**各自的最优缩放因子**（而非统一的 1/s），为微调提供好得多的初始化——**在不微调的情况下就能实现约 8 倍扩展**；
2. 采用**渐进式扩展**：先微调得到 256K 的模型，再在此基础上做第二次位置内插，最终达到 **2048K（约 200 万 token）**；
3. 通过短上下文恢复微调，保证模型在原始短窗口上的性能不掉。

一手数据（arXiv:2402.13753 摘要原文）：**首次**将预训练 LLM 的上下文窗口扩展到 **2048K**，在 256K 训练长度下**仅需约 1K 微调步**，并保持原始短上下文性能。

### 5.7 位置编码与上下文长度关系小结（演进谱系）

```
基础位置编码                长上下文扩展（RoPE 系）
─────────────              ──────────────────────────
Sinusoidal(2017)           PI(2023)：线性内折全部频率 ──→ 稳定但损高频
Learned(2018)      ──┐
Relative/T5(2019)    ├─→ RoPE(2021) ──→ NTK(2023)：高频外推/低频内折
ALiBi(2021) ───────┘                    ├─→ YaRN(2023)：分段+温度，训短测长
                                        └─→ LongRoPE(2024)：非均匀+渐进，2M
```

上下文长度的量级演进（业界典型值）：

| 时期 | 典型上下文 | 关键手段 |
|---|---|---|
| 2018 BERT/GPT | 0.5K–1K | 可学习/正弦，无外推设计 |
| 2019–2020 GPT-2/T5 | 1K–2K | 相对位置偏置 |
| 2023 初 LLaMA | 2K–4K | RoPE（base=10000） |
| 2023 GPT-4 / Claude | 32K–128K | PI / NTK / YaRN + 长文微调 |
| 2024–2025 长文模型 | 128K–1M | YaRN、非均匀缩放、稀疏/滑窗注意力 |
| 2024 LongRoPE | 2M | 非均匀 + 渐进式二次内插 |

**核心关系总结：**

1. 位置编码**定义了位置表示的有效范围**，上下文长度首先受此范围约束；
2. 训练长度内靠模型"学会使用"位置信号，长度之外靠**外推/内插算法**续命；
3. 缩放（PI/NTK/YaRN）解决"位置表示"瓶颈，FlashAttention/稀疏注意力解决"O(n²) 算力"瓶颈，二者缺一不可；
4. 趋势是从"均匀压缩 + 全量微调"走向"非均匀/分频段压缩 + 极少微调甚至免微调"，让长上下文越来越廉价。

---

## 6. 核心实现：RoPE 与长上下文扩展伪代码

**导读：** 本章给出可直接对照实现的 Python 风格伪代码，帮助把前面的公式落到工程。

### 6.1 RoPE

```python
import torch

def precompute_freqs(dim, max_seq_len, base=10000):
    # 每个二维平面的角频率 θ_i，共 dim/2 个
    idx = torch.arange(0, dim, 2).float()
    freqs = 1.0 / (base ** (idx / dim))          # shape (dim/2,)
    pos = torch.arange(max_seq_len).float()
    angles = torch.einsum('i,j->ij', pos, freqs) # shape (max_seq_len, dim/2)
    return angles.cos(), angles.sin()

def rotate_half(x):
    # 把后半段与前半段配对取负，构造旋转的 sin 乘子
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    # q,k: (batch, seq, n_heads, dim)；cos,sin: (seq, dim/2)
    cos = torch.cat([cos, cos], dim=-1)          # 广播到完整维度
    sin = torch.cat([sin, sin], dim=-1)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k

# 在 attention 内部：
# q, k = apply_rope(q, k, cos[:seq_len], sin[:seq_len])
# scores = q @ k.transpose(-1, -2) / sqrt(d_k)
```

### 6.2 PI / NTK / YaRN 的缩放因子

```python
def rope_scaling(kind, dim, pos, base=10000, scale=1.0,
                 original_max=4096, ramp=0.625):
    i = torch.arange(0, dim, 2).float()
    freqs_base = base ** (-i / dim)

    if kind == "pi":                       # 线性内插：全部频率 /scale
        freqs = freqs_base / scale

    elif kind == "ntk":                    # NTK：整体改 base，非均匀
        base_new = base * scale ** (dim / (dim - 2))
        freqs = base_new ** (-i / dim)

    elif kind == "yarn":                   # YaRN：按波长分段
        wavelength = 2 * torch.pi / freqs_base
        # 高频(波长短)→不缩放(1.0)，低频(波长长)→完全内插(1/scale)
        low  = ramp * original_max
        high = (1.0 - ramp) * original_max * scale
        smooth = (wavelength - low) / (high - low)
        smooth = smooth.clamp(0, 1)
        factor = 1.0 - smooth * (1 - 1.0 / scale)
        freqs = freqs_base * factor

    return pos[:, None] * freqs[None, :]   # 最终角度

# YaRN 注意力温度（logits 锐化），scale 为扩展倍数：
# attn_scale = 0.1 * math.log(scale) + 1
# scores = scores / attn_scale
```

---

## 7. 结论与选型建议

**导读：** 用一组可执行的结论和工程 checklist，收束全文。

### 7.1 编号结论

1. **Transformer 靠"全连接注意力 + 全并行"取代了 RNN**，O(1) 的 token 交互路径和可大规模并行是它统治序列建模的根本原因。
2. **自注意力天然无序（置换等变），位置编码是让模型理解语言的必需品，而非可选装饰。** 它要同时承载顺序、相对距离与方向三类信息。
3. **方案演进的主线是"从绝对到相对、从加词向量到作用于注意力计算"**：正弦/可学习 → 相对位置偏置 → RoPE、ALiBi。
4. **RoPE 是当代 LLM 的主流选择**（内积恒等注入相对位置、远程衰减、无参数、易扩展）；**ALiBi 以"训短测长"的强外推见长**。
5. **上下文长度首先是位置编码被定义和训练覆盖的范围。** 超出范围时，PI/NTK/YaRN/LongRoPE 解决"位置表示"，FlashAttention/稀疏注意力解决"O(n²) 算力"，两者正交且都不可少。
6. **内插比外推稳定得多（理论上界约小 600 倍）；分频段（高频外推、低频内插）+ 注意力温度 + 非均匀缩放**是长上下文低成本扩展的关键诀窍。

### 7.2 工程选型 Checklist

- ☐ 新做自回归 LLM：优先 **RoPE**（生态、工具链、长文扩展方案最成熟）。
- ☐ 强需求"训短测长"、预算极少：评估 **ALiBi / Dynamic YaRN**。
- ☐ 扩展 ≤ 2–4 倍且可少量微调：**PI**（最简单）。
- ☐ 扩展更大且要保住近位分辨：**NTK-Aware / Dynamic NTK**。
- ☐ 扩展 8 倍以上、追求性价比：**YaRN（分段 + 温度）+ 少量长文微调**。
- ☐ 冲刺 100K–2M：**LongRoPE 式非均匀 + 渐进扩展**，并叠加 FlashAttention / 滑窗 / 稀疏注意力解决算力。
- ☐ 扩展后务必验证两类基准：**长文任务**（passkey、长文摘要/检索）与**原始短窗任务**（防止"长了但变笨"）。
- ☐ 检查 base、scale、original_max、ramp 等超参在推理代码与训练代码中**完全一致**，否则位置错位。

### 7.3 注意事项与诚实性声明

- 本报告 arXiv 论文编号（1706.03762、1803.02155、2104.09864、2108.12409、2306.15595、2309.00071、2402.13753、2212.10554、1910.10683）均已通过 **arXiv 官方 API 逐一核实**；PI、YaRN、LongRoPE 的关键定量结论摘自**论文摘要原文**。
- 5.4 节 LLaMA 2 Long 的 PPL 表格为**知乎文章转述**（已在表下标注作者与文章），引用到正式材料前请二次核对论文原文。
- 业界"上下文长度"数字（如某模型支持 192K/1M）会随版本快速变化，正文给出的是量级与趋势，具体以各模型官方说明为准。

---

## 8. 参考资料

### 8.1 原始论文（arXiv 编号经官方 API 核实）

1. Vaswani et al. **Attention Is All You Need.** NeurIPS 2017.
   https://arxiv.org/abs/1706.03762
2. Shaw et al. **Self-Attention with Relative Position Representations.** NAACL 2018.
   https://arxiv.org/abs/1803.02155
3. Raffel et al. **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer（T5）.** 2019.
   https://arxiv.org/abs/1910.10683
4. Su et al. **RoFormer: Enhanced Transformer with Rotary Position Embedding.** 2021.
   https://arxiv.org/abs/2104.09864
5. Press et al. **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation（ALiBi）.** ICLR 2022.
   https://arxiv.org/abs/2108.12409
6. Sun et al. **A Length-Extrapolatable Transformer（XPos）.** 2022.
   https://arxiv.org/abs/2212.10554
7. Chen et al. **Extending Context Window of Large Language Models via Positional Interpolation（PI）.** 2023.
   https://arxiv.org/abs/2306.15595
8. Peng et al. **YaRN: Efficient Context Window Extension of Large Language Models.** ICLR 2024.
   https://arxiv.org/abs/2309.00071
9. Ding et al. **LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens.** 2024.
   https://arxiv.org/abs/2402.13753

### 8.2 官方代码与一手资料

- YaRN 官方代码仓库：https://github.com/jquesnelle/yarn
- RoPE 原始中文解读，苏剑林·科学空间：https://spaces.ac.cn/archives/8265

### 8.3 知乎文章（社区信源，交叉验证用）

1. 假如给我一只AI. **位置编码之路：SIN->ALiBi->RoPE ->PI->NTK->YARN.**（671 赞）
   https://zhuanlan.zhihu.com/p/1894384438206505105
2. 绝密伏击. **RoPE外推优化——支持192K上下文长度.**（129 赞，5.4 节 PPL 数据来源）
   https://zhuanlan.zhihu.com/p/678755776
3. Chaos AI Studio. **看图学大模型：从绝对位置编码到旋转位置编码(RoPE).**（265 赞）
   https://zhuanlan.zhihu.com/p/687525842
4. kaiyuan. **彻底搞懂RoPE计算原理：从1D到3D.**（270 赞）
   https://zhuanlan.zhihu.com/p/2023493768003724514
5. 姚远. **漫谈 Transformer 中的绝对位置编码、相对位置编码和融合位置编码(RoPE).**
   https://zhuanlan.zhihu.com/p/17311602488
6. lcltopismine3. **RoPE(旋转位置编码)完整数学原理与实现.**（70 赞）
   https://zhuanlan.zhihu.com/p/1968030005012443434
7. MrYXJ. **LLM时代Transformer中的Positional Encoding.**（85 赞）
   https://zhuanlan.zhihu.com/p/664214907
8. 小冬瓜AIGC. **【手撕 YaRN】LLM 训短推长，一直外推一直爽！**
   https://zhuanlan.zhihu.com/p/1944715652926538312
9. 然荻. **一文从位置编码诞生到理解长文本扩展!!Sinusoidal|ALiBi|ROPE|YaRN.**
   https://zhuanlan.zhihu.com/p/1909642625289533001
10. 紫气东来. **LLM(23)：LLM 中的长文本问题.**（443 赞）
    https://zhuanlan.zhihu.com/p/640641794
11. 周弈帆. **让预训练 Transformer 生成更长的文本/图像：位置编码长度外推技术.**
    https://zhuanlan.zhihu.com/p/11433578489
12. Kevin吴嘉文. **论文笔记 | 探索 LLM 的上下文长度外推.**
    https://zhuanlan.zhihu.com/p/674699556
13. AI产品经理大群. **一文读懂：Transformer(详细但无代码).**（556 赞）
    https://zhuanlan.zhihu.com/p/607423406

---

*报告完。核心事实经一手论文与 arXiv API 核实，转述数据已在正文显式标注。*
