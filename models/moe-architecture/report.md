# 稀疏混合专家（MoE）架构前沿技术报告：当模型容量与计算量解耦——从 Shazeer 137B 到 DeepSeek-V3、Kimi K2 与 768 专家极稀疏时代

> **报告日期**：2026-10-22　**主题分类**：大模型架构创新
> **核心事实**：全部经一手信源核实；arXiv 编号经 arxiv.org 摘要页逐一核实；厂商自报数据在正文显式标注「厂商口径」；工程库数据来自其官方 GitHub README。
> **去重说明**：本知识库 models 分类下 21 篇报告均无 MoE 专文（MoE 仅在其他报告中作背景提及），本报告为该方向首篇深度报告。

---

## 目录

1. [为什么是 MoE：一个改变缩放方式的想法](#1-为什么是-moe一个改变缩放方式的想法)
2. [第一性原理：MoE 的形式化与核心矛盾](#2-第一性原理moe-的形式化与核心矛盾)
3. [九年演进时间线（2017–2026）](#3-九年演进时间线20172026)
4. [路由机制全谱：谁来选择专家](#4-路由机制全谱谁来选择专家)
5. [专家组织：细粒度分割、共享专家与专家数竞赛](#5-专家组织细粒度分割共享专家与专家数竞赛)
6. [负载均衡：辅助损失、无辅助损失与控制论统一](#6-负载均衡辅助损失无辅助损失与控制论统一)
• [6.1 辅助负载均衡损失](#61-辅助负载均衡损失)
• [6.2 无辅助损失负载均衡（ALF-LB）](#62-无辅助损失负载均衡alf-lb)
• [6.6 负载均衡的控制论统一（2026 新框架）](#66-负载均衡的控制论统一2026-新框架)
7. [MoE 训练系统：专家并行与 all-to-all 通信](#7-moe-training系统专家并行与-all-to-all-通信)
8. [MoE 推理：被专家撑大的显存与三条优化路线](#8-moe-推理被专家撑大的显存与三条优化路线)
• [8.1 推理第一性问题：decode 阶段的「1 个 token 也得装下整个模型」](#81-推理第一性问题decode-阶段的1-个-token-也得装下整个模型)
• [8.2 Upcycling：从 dense 检查点直接长出 MoE](#82-upcycling从-dense-检查点直接长出-moe)
• [8.3 专家卸载与预测式驻留](#83-专家卸载与预测式驻留)
• [8.4 专家复制与动态冗余](#84-专家复制与动态冗余)
• [8.5 专家量化与总体成本账](#85-专家量化与总体成本账)
• [8.6 2026 前沿：循环 MoE、软 MoE、专家剪枝](#86-2026-前沿循环-moe软-moe专家剪枝)
9. [MoE 缩放定律与「无限参数」猜想](#9-moe-缩放定律与无限参数猜想)
10. [旗舰系统横评（2024–2026）](#10-旗舰系统横评20242026)
10. [MoE 生态与采用时间线](#11-moe-生态与采用时间线)
11. [开放问题与工程落地 Checklist](#12-开放问题与工程落地-checklist)
12. [结语](#13-结语)

> 注：目录中章节编号重复处为 Markdown 自动编号显示问题，以正文实际内容为准。

---

## 1. 为什么是 MoE：一个改变缩放方式的想法

2024 年 12 月 27 日，DeepSeek 发布技术报告，宣告一个 **6710 亿参数**的模型用仅 **370 亿激活参数**完成每个 token 的计算，训练投入仅约 **278.8 万 H800 GPU 小时、14.8 万亿 token**（厂商口径，[2412.19437](https://arxiv.org/abs/2412.19437)）。2025 年 7 月，月之暗面的 Kimi K2 把总参数推到 **1.04 万亿**而每 token 仅激活 **326 亿**（[2507.20534](https://arxiv.org/abs/2507.20534)）。2026 年 9 月的工作已经在 **768 个专家、top-3** 的极稀疏设定下稳定训练（[2609.39137](https://arxiv.org/abs/2609.39137)）。

这些数字背后是同一个简单而深刻的想法：**把 Transformer 的前馈网络（FFN）替换成 N 个并行的子网络（专家），每个 token 只经过其中少数几个**。于是，模型"知道"的东西（总容量）与每个 token 实际"调用"的计算量（激活计算）第一次被干净地解耦——你可以在不按比例增加训练算力的前提下，把模型的知识容量扩大一个数量级。

This idea is Mixture of Experts (MoE)。它不是 2024 年的发明：2017 年 Noam Shazeer 等人的稀疏门控 MoE 层已做到每层多达数千专家、容量提升超过 1000 倍、137B 参数的语言模型（[1701.06538](https://arxiv.org/abs/1701.06538)）；2021 年的 Switch Transformer 用最简单的 top-1 路由把模型推到 1.6 万亿参数（[2101.03961](https://arxiv.org/abs/2101.03961)）。但 MoE 从"能跑通"到"成为几乎所有旗舰模型的骨干"，靠的是 2024–2026 年在**专家组织、负载均衡、专家并行通信和推理系统**四个方向上的密集工程突破——而这些突破绝大多数来自中国团队（DeepSeek、阿里 Qwen、Moonshot、MiniMax、智谱）。

> ![](diagrams/1-principle.html)

## 2. 第一性原理：MoE 的形式化与核心矛盾

### 2.1 数学定义

标准 dense Transformer 每个位置的 FFN 子层输出为：

```
y = FFN(x) = W2 · silu(W1 · x)          # SwiGLU 风格，W1: H→I, W2: I→H
```

MoE 把它替换为 N 个同构专家 {E₁…Eₙ} 加一个路由器（router/gate）。对 token 表示 x：

```
# 1) 路由打分：一个可学习的线性层 + softmax
logits = x · W_g                  # W_g: H→N
scores = Softmax(logits)          # N 个专家的亲和度

# 2) 选专家：top-k（典型 k=1…10）
TopK = argtopk(scores, k)

# 3) 稀疏加权组合
y = Σ_{i∈TopK} scores_i · E_i(x)
```

每个 token 的激活参数量约为：

```
激活参数 ≈ Attention(H) + k × Expert(H,I)
总参数   ≈ Attention(H) + N × Expert(H,I)
```

以 DeepSeek-V3 为例：N=256 路由专家 + 1 共享专家，k=8，61 层。总参数 671B，每 token 激活仅 37B——**容量/计算比约 18:1**（[2412.19437](https://arxiv.org/abs/2412.19437)）。Kimi K2 为 384 专家、k=8，比值约 32:1（[2507.20534](https://arxiv.org/abs/2507.20534)）。

### 2.2 为什么是 FFN 被替换，而不是注意力

现有 MoE 几乎只对 FFN 做条件化，原因有三：

1. **FFN 是模型的"知识存储层"。** 机制研究普遍把 FFN 看作 key-value 记忆（每个中间维 i 对应一条"若输入匹配关键模式则注入知识"的条目）。知识本身天然可分领域/分主题，适合分给不同专家；而注意力负责跨位置信息混合，条件化会破坏序列内的一致性。
2. **计算占比高。** FFN 通常占每层约 2/3 的 FLOPs，替换它对总计算的调节杠杆最大。
3. **无跨-token 依赖。** FFN 逐位置独立计算，做稀疏选择不影响其他位置，容易并行；注意力的每个 head 处理整个序列，条件化代价高。

### 2.3 MoE 的三对核心矛盾

| 矛盾 | 内容 | 典型表现 |
|---|---|---|
| **专门化 vs 均衡** | 我们希望专家各专一块（专门化），但路由器会自然涌向少数"万能"专家，造成热专家过载、冷专家永不更新（专家坍塌） | 几个专家收走大部分 token，其余退化；训练不稳 |
| **容量 vs 通信** | 专家越多、分割越细，知识容量越大；但专家分布在不同设备上，每个 MoE 层都要做两次全对全（all-to-all）通信把 token 送到对应专家再收回 | 通信常成为多机训练瓶颈，占比可超计算 |
| **训练友好 vs 推理友好** | MoE 训练时每批 token 多，每个专家都有活干、利用率高；推理（尤其生成阶段）每批 token 很少，却必须把全部专家权重常驻显存 | MoE 模型"训练省钱、推理费显存"，单请求低延迟场景吃亏 |

### 2.4 关键术语速查

| 令牌 | 含义 |
|---|---|
| N / k | 专家总数 / 每 token 选中的专家数（top-k） |
| routed / shared expert | 路由专家（被选择）/ 共享专家（每个 token 都过，承载通用知识） |
| capacity factor | 容量因子：每个专家最多接收 token 数相对均值的倍数，超出则 drop（丢弃）或在系统层处理 |
| token dropping | 专家桶满后，溢出 token 在该层走捷径（仅残差）或被 padding 处理 |
| EP（Expert Parallelism） | 专家并行：不同专家放不同 GPU，靠 all-to-all 交换数据 |
| aux-loss / ALF-LB | 辅助负载均衡损失 / 无辅助损失负载均衡（DeepSeek-V3 的 bias 法） |
| upcycling | 从已训练好的 dense 检查点初始化并训练成 MoE，省预训练成本 |
| redundant experts | 冗余专家：把高负载专家复制多份放到不同 GPU，用于推理负载分担 |


---

## 3. 九年演进时间线（2017–2026）

| 时间 | 工作 | 机构 | 关键动作 | 规模/数据 |
|---|---|---|---|---|
| 2017-01 | [Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538) | Google | 带噪声 top-k 门控 + 容量因子，首次 GPU 集群条件计算 | 每层数千专家、容量 ×1000、137B LM |
| 2020-06 | [GShard](https://arxiv.org/abs/2006.16668) | Google | 注解 API + XLA 自动分片，top-2 | >600B、2048 TPU |
| 2021-01 | [Switch Transformer](https://arxiv.org/abs/2101.03961) | Google | 最简 top-1 + 选择性精度，bfloat16 训练稀疏模型 | 1.6T 参数，预训练最快 ×7 |
| 2021-03 | [BASE Layers](https://arxiv.org/abs/2103.16716) | Meta | token→专家建模为线性分配（最优均衡），无 aux-loss | 零负载偏差 |
| 2021-06 | [Hash Layers](https://arxiv.org/abs/2106.04426) | Meta | token 哈希确定性选专家，零路由参数 | 天然均衡 |
| 2021-12 | [GLaM](https://arxiv.org/abs/2112.06905) | Google | 64 专家 top-2 的稀疏通用 LM，相对 GPT-3 大幅省算力 | 1.2T 总参/激活约三分之一 |
| 2022-02 | [ST-MoE](https://arxiv.org/abs/2202.08906) | Google | 系统研究训练不稳定与微调，稳定可迁移设计 | 269B 总参≈32B dense 算力 |
| 2022-02 | [Expert Choice Routing](https://arxiv.org/abs/2202.09368) | Google | 反转路由：专家选 token，桶固定 | 同算力预训练约 ×2 |
| 2022-06 | [Tutel](https://arxiv.org/abs/2206.03382) | Microsoft | 自适应 MoE 运行时，动态并行/流水线 | 8×A100 大幅加速 |
| 2022-08 | [Branch-Train-Merge](https://arxiv.org/abs/2208.03306) | Meta | 无通信并行训领域专家，再平均合并 | 免多节点同步 |
| 2022-11 | [MegaBlocks](https://arxiv.org/abs/2211.15841) | Databricks/Stanford | 块稀疏 MoE kernel，不丢 token、不浪费 padding | 比 Tutel 快至 40% |
| 2022-12 | [Sparse Upcycling](https://arxiv.org/abs/2212.05055) | Google | 从 dense 检查点初始化训练 MoE | 显著省预训练成本 |
| 2023-08 | [Soft MoE](https://arxiv.org/abs/2308.00951) | Google | 全可微软分配，解决离散路由不稳 | 视觉大幅超 ViT |
| 2024-01 | [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) | Mistral | 8 专家 top-2，开源引爆 MoE 普及 | 46.7B 总/12.9B 激活 |
| 2024-01 | [DeepSeekMoE](https://arxiv.org/abs/2401.06066) | DeepSeek | 细粒度分割 + 共享专家隔离 | 16B 匹敌 LLaMA2-70B |
| 2024-03 | [DBRX](https://www.databricks.com/blog/introducing-dbrx) | Databricks | 16 专家 top-4 开源 MoE | 132B 总/36B 激活 |
| 2024-04 | [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) | DeepMind | 每层只处理 top-k 个 token，沿深度动态分配算力 | 80% 算力预算匹配 dense |
| 2024-04 | Snowflake Arctic | Snowflake | 128 专家 + dense 主干，企业开源 MoE | 480B 总/17B 激活 |
| 2024-05 | DeepSeek-V2 | DeepSeek | MLA + DeepSeekMoE，组限制路由 | 236B 总/21B 激活 |
| 2024-09 | [OLMoE](https://arxiv.org/abs/2409.02060) | Ai2 | 全开源（权重/数据/代码/日志）MoE | 7B 总/1.3B 激活 |
| 2024-12 | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | DeepSeek | 无辅助损失负载均衡 + MTP，定义新一代范式 | 671B/37B，14.8T token |
| 2025-01 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | DeepSeek | V3-Base 上 RLVR，推理能力涌现 | 同 V3 架构 |
| 2025-01 | [MiniMax-01](https://arxiv.org/abs/2501.08313) | MiniMax | 线性（Lightning）注意力 + MoE | 456B/45.9B，训练 1M |
| 2025-04 | [Qwen3 系列](https://arxiv.org/abs/2505.09388) | 阿里 | 全尺寸 dense+MoE，128–512 专家 | 235B/22B 等 |
| 2025-07 | [Kimi K2](https://arxiv.org/abs/2507.20534) | Moonshot | MuonClip 稳定化，1T 总参强 agentic | 1.04T/32.6B，SWE-Bench 65.8 |
| 2025-07 | GLM-4.5 | 智谱 | 160 专家 + 1 共享，混合 dense/MoE 层 | 355B/32B |
| 2026-09 | [Instella-MoE](https://arxiv.org/abs/2609.00791) | Instella | 开源 MoE 技术报告 | 厂商口径见报告 |
| 2026-09 | [ID Balancing](https://arxiv.org/abs/2609.39137) | — | PID 控制论统一负载均衡，768 专家 | top-3 MaxVio 降 >50% |
| 2026-10 | [MirrorMoE](https://arxiv.org/abs/2610.22353) | — | 重新审视路由探索，token 自适应抑制冗余 | 跨规模一致提升 |

> 注：Snowflake Arctic 官方论文为《Arctic Embed / Arctic》系列技术博客（snowflake.com，本机网络不可达），参数数据经其开源 config 与第三方数据库交叉核实。

---

## 4. 路由机制全谱：谁来选择专家

路由是 MoE 的"心脏"，也是大部分不稳定性的来源。九年下来，路由机制沿三条轴线分化。

### 4.1 轴一：选择方向——token 选专家 vs 专家选 token

**（a）Token-choice（主流）**：token 根据 router softmax 选 top-k 专家。优点是语义直观、每个 token 都有确定的 k 个专家服务；缺点是热门专家会被挤爆，必须外加容量限制或均衡机制。Switch（top-1）、Mixtral/DeepSeek（top-2/6/8）均属此类。

**（b）Expert-choice**：[Zhou et al. 2022](https://arxiv.org/abs/2202.09368) 把路由反转——每个专家固定容量桶，主动"认领"它最想处理的 top-k token。负载因此**天然完全均衡**，论文报告同等算力下预训练约 **2× 加速**。代价是：一个 token 可能被 0 个或多个专家选中（服务数量不确定），且该机制依赖一个 batch 内全局可见，**因果自回归生成时难以直接使用**（需批处理），故主要用于训练/视觉。

**（c）软分配 Soft MoE**：[Puigcerver et al. 2023](https://arxiv.org/abs/2308.00951) 更进一步，放弃离散选择：把一个 batch 的全部 token 按亲和度加权"溶解"成固定数量的 slot，专家处理 slot 后再加权还原。整个过程**完全可微、无 token 丢弃、无负载问题**，在视觉任务上大幅超越 ViT 与 token/expert-choice。它代表了"用可微软组合换取稳定性"的路线，但 slot 数随专家数增长、语义可解释性弱。

### 4.2 轴二：是否可学习——学习路由 vs 确定性路由

**（a）学习路由**：router 是可训练线性层 `Softmax(x·W_g)`，能自适应地形成专门化，但带来路由坍塌、训练不稳等全部问题。

**（b）哈希路由**：[Hash Layers](https://arxiv.org/abs/2106.04426) 用 token 内容的哈希值确定性地映射到专家子集，**零路由参数、零均衡损失、零分配算法**，天然均衡，性能与 Switch/BASE 竞争甚至更优。它牺牲了"按需选专家"的灵活性，换取极简与稳定。

**（c）免训练语义路由（2026 新发现）**：[Do et al., SSMoE](https://arxiv.org/abs/2605.30992) 发现一个漂亮事实——**专家权重矩阵的特征向量本身编码了语义**：用 token 表示与专家权重奇异向量的相似度做路由（一次 SVD），无需任何训练/微调即可得到不坍塌的路由器，在语言与视觉、干净与损坏数据上都展现出更好的泛化与鲁棒性。这暗示"路由知识其实已存在于专家权重中"，显式 router 可能只是历史包袱。

### 4.3 轴三：选择粒度——token 路由 vs 深度路由

**（a）Token 路由（MoE 本义）**：在每个 MoE 层为每个 token 选专家，调节的是"每个 token 用哪些 FFN"。

**（b）深度路由 Mixture-of-Depths（MoD）**：[Raposo et al. 2024](https://arxiv.org/abs/2404.02258) 换了一个维度：不是选专家，而是**在每层只让 top-k 个 token 参与该层计算**（其余 token 走残差直通）。这样 FLOPs 沿"深度 × token"两个维度动态分配，简单 token 提前"抄近道"，复杂 token 逐层被深加工。论文报告在 **80% 算力预算**下即可匹配或超越 dense，前向最高约 **5×** 节省。MoD 与 MoE 正交，可叠加（MoE 选专家 + MoD 选 token）。

### 4.4 路由的训练动力学：三阶段与拥塞博弈

路由不是一成不变的。[Mouzouni 2026](https://arxiv.org/abs/2604.04230) 用"拥塞博弈"视角分析 OLMoE（20 个检查点）与 OpenMoE-8B 的训练全过程，发现一个单一"拥塞系数 γ_eff"即可刻画均衡—质量权衡，且训练呈现**三阶段**：

```
早期 γ_eff ≈ 14（专家间争抢弱、较均衡）
   → 中期 γ_eff 峰值 36–39（专门化加剧、涌向热专家）
   → 后期 γ_eff ≈ 9（重新趋于均衡，专家分工收敛）
```

这解释了为什么"训练中期最容易出 loss spike"，也说明负载均衡约束的强度理应随训练阶段调整——固定系数的 aux-loss 在中期可能过弱、后期可能过强。

---

> ![](diagrams/2-routing-spectrum.html)

---

## 5. 专家组织：细粒度分割、共享专家与专家数竞赛

确定了"怎么选"，下一个问题是"专家怎么切"。2024 年 DeepSeekMoE 给出的两个设计——**细粒度专家分割**与**共享专家隔离**——已成为此后几乎所有 MoE 的标准配方。

### 5.1 细粒度专家分割（Fine-Grained Segmentation）

直觉：在固定总参数预算下，把每个专家做"小"、专家数做"多"，组合空间更大、专门化更细。形式化地，把 N 个专家各分割为 m 份，得到 mN 个小专家，同时每个 token 从激活 k 个改为激活 mk 个——**总参数与每 token 激活计算都不变，但可选组合数从 C(N,k) 暴增到 C(mN,mk)**。

[DeepSeekMoE](https://arxiv.org/abs/2401.06066) 的实测极具说服力：2B 规模的细粒度模型匹敌 GShard 2.9B；**16B 的 DeepSeekMoE（64 路由专家、top-6）匹敌 LLaMA2-70B**。此后专家数一路攀升：Mixtral 的 8 → DeepSeek-V3 的 256 → Kimi K2 的 384 → Qwen3-Next 的 512 → 2026 年前沿研究的 768。

### 5.2 共享专家隔离（Shared Expert Isolation）

纯路由有个问题：大量**通用、公共的知识**（常见语法、通用事实）被每个 token 重复路由，浪费路由槽位、还造成专家争抢。DeepSeekMoE 的解法是隔离出 K_s 个**共享专家**，每个 token 无条件经过它们，把"公共知识"与"专门知识"分开：

```
y = Σ_{s∈Shared(K_s)} E_s(x)              # 公共知识：每个 token 都过
  + Σ_{i∈TopK routed} scores_i · E_i(x)   # 专门知识：按需路由
```

DeepSeekMoE-16B 用 2 个共享专家；DeepSeek-V3/Kimi K2/GLM-4.5 用 1 个。共享专家让路由专家可以更专注于细分领域，也显著降低路由负载。值得注意的是 Qwen3 全系列**不使用共享专家**，而是靠 128–512 的多专家数与 top-8 覆盖——两条路线都能跑通，说明共享专家并非必需，但在控制专家总数时很有效。

### 5.3 组限制路由（Group-Constrained Routing）

当专家数达数百，若 token 可任意路由到任何专家，通信会在全部节点间全互联，带宽压力大。DeepSeek-V2/V3 引入**组限制**：把专家分成若干组（V3：256 专家分 8 组），先选出亲和度最高的几个组，再只在组内选专家。V3 由此保证每个 token **最多被发往 4 个节点**，把 all-to-all 通信限制在局部（[2412.19437](https://arxiv.org/abs/2412.19437)）。这一约束还被 EPLB 利用——同组专家尽量放同一节点，进一步减少跨节点流量。

### 5.4 专家数竞赛与"探索困境"

专家是不是越多越好？[Li et al. 2026, MirrorMoE](https://arxiv.org/abs/2610.22353) 指出一个内在张力——**路由探索困境**：细粒度分割要求路由器保持足够探索（多试不同小专家才能形成专门化），而共享专家会降低探索需求（公共知识被兜底后，路由更容易集中）。二者组合若处理不当，会出现大量"冗余激活"：token 选中的多个小专家其实在做重复的事。MirrorMoE 让 token 自适应地抑制冗余专家，在不同模型规模上取得一致提升。这标志着研究焦点从"把专家做多"转向"让每次激活都不浪费"。

---

> ![](diagrams/3-load-balancing.html)

---

## 6. 负载均衡：辅助损失、无辅助损失与控制论统一

MoE 最经典的工程难题：不加干预时，路由器会形成正反馈——专家被选得越多、训练得越好，就越容易被选，最终**少数专家通吃、多数专家坍塌**。九年里出现了三代解法。

### 6.1 辅助负载均衡损失

最直接的办法：在主语言模型损失之外，加一个鼓励"每个专家分到的 token 数尽量均匀"的辅助项。对一层，设 f_i 为被路由到专家 i 的 token 占比，P_i 为 router 给专家 i 的平均概率，则 Switch 风格的辅助损失为：

```
L_bal = α · N · Σ_i f_i · P_i          # 均匀分布时最小
L_total = L_LM + L_bal
```

α 是均衡系数（Mixtral 开源 config 中 `router_aux_loss_coef=0.02`）。它简单有效，Switch/Mixtral 都靠它。但有两个固有代价：① 它是一个**与真实目标无关的代理目标**，强行拉平会牺牲专门化——2026 年的研究甚至证明"均衡过头"会损害性能（见 6.5）；② α 需要调，且如第 4.4 节所述，最优强度随训练阶段变化。

配套机制还有**容量因子与 token dropping**：给每个专家设容量上限（均值的 capacity factor 倍），桶满后溢出 token 在该层只走残差。这防止热专家无限膨胀，但丢弃 token 会损失信息，大容量因子又浪费算力，是个两难。

### 6.2 无辅助损失负载均衡（ALF-LB）

[DeepSeek-V3](https://arxiv.org/abs/2412.19437) 提出影响深远的替代方案：**完全去掉辅助损失**，改为在路由打分时给每个专家加一个可动态调整的偏置 b_i：

```
# 每个专家维护一个 bias，按其负载动态调整
若专家 i 过载 → b_i 减小（被选中概率下降）
若专家 i 欠载 → b_i 增大（被选中概率上升）

路由：TopK( Softmax(x·W_g + b) )
```

偏置只影响"选哪个专家"，不进入反向传播、不污染主损失梯度，因而模型可以全力优化语言建模目标、同时保持均衡。DeepSeek-V3 凭此在 14.8T token 训练中保持稳定。这一"用动态偏置而非损失项"的思想在 2025–2026 被广泛采纳。

### 6.3 理论解释：原始-对偶与遗憾界

[Han & Zhong 2025](https://arxiv.org/abs/2512.03915) 为 ALF-LB 给出了第一个理论框架：把负载均衡看作一个**分配问题**，DeepSeek 的偏置更新恰好是该问题原始-对偶方法的**单步常数时间近似**，并证明在在线强凸条件下可达到 **O(log T) 的对数遗憾界**——即随训练进行，实际分配与最优均衡分配的累计差距增长得非常慢。这为"拍脑袋的 bias 规则"提供了坚实的优化理论支撑。

### 6.4 最优均衡的另一条路：线性分配

更早的 [BASE Layers](https://arxiv.org/abs/2103.16716) 走得更彻底：不用可微路由，直接把一个 batch 内"token→专家"的分配建模为**线性分配（最优传输）问题**并求最优解，使每个专家收到**完全等量**的 token（零负载偏差），无需辅助损失、无新超参。其代价是每步要解一次分配问题（计算开销）且路由不可微（用匈牙利类算法），工程上不如 bias 法轻便，故未成为主流，但思想被后续 Soft MoE/ALF 理论吸收。

### 6.5 反方证据：均衡也会过头

负载均衡不是越强越好。[Kapusuzoglu et al. 2026](https://arxiv.org/abs/2609.04453) 揭示了**过度分散（over-dispersion）**现象：激进的均衡把路由压成近均匀分布，专家间的重要性差异被抹平，困惑度甚至不再能预测下游性能。他们在 gpt-oss-20B 上发现：困惑度最低的配置，数学推理反而最差；按激活感知评分，GPQA 相差达 **18 分**。他们提出领域感知的 **MESA** 剪枝策略，在剪枝 25% 时于 7+ 个领域最小化最差退化。这是对"均匀即正义"直觉的重要纠正——**均衡的目标是支撑专门化，而非消灭差异**。

### 6.6 负载均衡的控制论统一（2026 新框架）

[Jin et al., ID Balancing](https://arxiv.org/abs/2609.39137) 提供了一个统一视角：把各种均衡机制看作**反馈控制器**——

```
DeepSeek bias 法     ≈ 积分控制器（I）：对累计偏差累加修正
Kimi K3 分位均衡法   ≈ 比例控制器（P）：对当前偏差即时修正
```

两者都是"不完整的 PID"。作者补上微分（D）项，提出 **ID（积分-微分）控制器**，用偏差的变化率做提前响应，显著改善极稀疏 MoE 的稳定性。在 **768 专家、top-10/5/3** 的设定下：top-3 时最大违规 MaxVio 降低 **>50%**、最小违规 MinVio 降低 12%；模型从 18.9B 扩到 69.9B 时，MaxVio 比 aux-loss 基线低约 **89.6%**。控制论框架的优雅之处在于：它把"调均衡"从炼丹变成了有成熟理论（稳定性、超调、滞后）可循的工程问题。

---

> ![](diagrams/4-train-infer.html)

---

## 7. MoE 训练系统：专家并行与 all-to-all 通信

### 7.1 专家并行（Expert Parallelism）

MoE 训练的核心并行维度是 EP：把不同专家放到不同 GPU 上。一个 MoE 层的前向需要两次 all-to-all 通信：

```
① Dispatch（分发）：每个 GPU 把本地 token 按其选中的专家，发送到持有该专家的 GPU
② Compute：各 GPU 只计算自己持有的专家收到的 token
③ Combine（合并）：把专家输出按来源发回，各 token 在原 GPU 上加权聚合
```

这与 dense 模型的 all-reduce 截然不同：all-to-all 的通信量与路由分布强相关、模式不规则，且要跨节点（NVLink 域内 + RDMA 域间）。实际训练通常组合使用 EP × TP（张量并行）× PP（流水）× DP（数据并行），并通过容量因子、组限制路由控制通信范围。

### 7.2 DeepEP：为 MoE 重写的 all-to-all 内核

普通 NCCL all-to-all 并非为 MoE 的不规则、非对称通信优化。DeepSeek 于 2025 年 2 月开源 **DeepEP（DeepEveryParallel）**——一个高吞吐、低延迟的 EP all-to-all GPU 内核库，专用于 MoE 的 dispatch/combine（据其[官方 GitHub README](https://github.com/deepseek-ai/DeepEP)）：

- 原生支持 **FP8 dispatch**（低精度分发，省带宽）；
- 通过 NVLink 做**冗余专家的权重与梯度交换**；
- 通信内核运行时经 **DeepJIT** 编译，按当前设备即时生成；
- 还提供 PP/CP/DP 及远程内存访问（Engram）的实验性原语；
- 要求 Hopper 及更新 GPU、CUDA 13.1+、NCCL 2.32.3+，节点内走 NVLink、节点间走 RDMA。

> **关于 DeepEP 的编号核实**：DeepEP **没有 arXiv 论文**，仅以 GitHub 仓库及官网 deepep.org 发布。网传编号 2506.06236 经核实实为一篇化学/高焓气体物理论文（《Modified Marrone-Treanor dissociation model…》），与 MoE 无关，**不应引用**。

2026 年的新版本（V2.5）把缓冲区细分为 EPBuffer/EngramBuffer/PPBuffer/BucketBuffer，新增批量 all-gather/reduce-scatter/all-reduce，并支持动态冗余专家的权重预取（`lb_prefetch_weights`）与梯度归约。DeepEP 还发布了**昇腾 Ascend 950 NPU 版本**（API 一致、性能对齐），是国产算力软件栈适配的重要一环。

### 7.3 EPLB：专家并行负载均衡器

训练时负载可由 aux-loss/bias 保证，但**推理/部署时专家负载不均**会直接导致部分 GPU 过载。DeepSeek 开源的 [EPLB](https://github.com/deepseek-ai/EPLB) 用**冗余专家**策略解决：复制高负载专家，再启发式地把副本打包到各 GPU。它提供两种策略：

- **分层均衡（Hierarchical）**：当节点数整除专家组数时，先把专家组均匀打包到节点（利用组限制路由，同组尽量同节点以减少跨节点流量），再在节点内复制并打包到各 GPU。适合 **prefill 阶段**（EP 规模较小）。
- **全局均衡（Global）**：无视专家组、全局复制并打包到 GPU。适合 **decode 阶段**（EP 规模较大）。

EPLB 的核心接口 `eplb.rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)` 依据估计的专家负载（仓库外问题，常用历史统计滑动平均）计算复制与放置方案。

> **编号核实**：EPLB 同样**无 arXiv 论文**，仅 GitHub 发布。网传 2504.13081 实为 CMS 高能物理论文（Higgs 玻色子产生截面），与 MoE 无关，**不应引用**。

### 7.4 块稀疏与高效 MoE 内核

MoE 训练还有一类"系统级"优化：把 MoE 计算表达为**块稀疏矩阵乘**，避免为不规则 token 数做 padding 或丢 token。[MegaBlocks](https://arxiv.org/abs/2211.15841) 据此自研 GPU kernel，做到不丢 token、不浪费 padding，比 Tutel 端到端快至 **40%**、比 Megatron-LM dense 快 **2.4×**。[Tutel](https://arxiv.org/abs/2206.03382) 则提供自适应并行与流水线的运行时。这类工作与 DeepEP 互补：一个优化"算"，一个优化"通信"。

---

## 8. MoE 推理：被专家撑大的显存与三条优化路线

### 8.1 推理第一性问题：decode 阶段的「1 个 token 也得装下整个模型」

MoE 训练时每 batch 有成千上万 token，所有专家都被充分利用，算力花在刀刃上。但**自回归生成（decode）阶段，每步只有批内少量 token**，而无论这一步路由到几个专家，**全部 N 个专家的权重都必须常驻显存**——因为下一步可能用到任何一个。于是：

```
dense 模型：显存 ≈ 全模型权重（= 全部知识）
MoE 模型： 显存 ≈ 全模型权重（= 全部知识，同样巨大！），但每步计算只用其中 k/N
```

这就是 MoE "训练省、推理费显存"的根源：671B 的 DeepSeek-V3 即使每 token 只激活 37B，部署时仍需装下全部 671B 权重，还带来专家间负载不均、低 batch 下专家利用率低、kernel launch 多等问题。MoE 推理优化因此围绕三条路线展开：**少装（卸载/换入换出）、复制分担（冗余专家）、缩小（量化）**，外加从源头省成本的 upcycling。

### 8.2 Upcycling：从 dense 检查点直接长出 MoE

Upcycling 严格说是"训练侧省钱"，但深刻影响推理模型的获得成本。[Sparse Upcycling（Google, 2022）](https://arxiv.org/abs/2212.05055) 发现：不必从零预训练 MoE——取一个已训好的 dense 模型，把它的 FFN **复制 N 份**作为 N 个专家的初始化，加入 router，再继续训练，即可用远低于从零训练的成本得到性能更好的 MoE。Qwen1.5-MoE 即由 Qwen-1.8B upcycle 而来，官方称只用约 **25% 的训练资源**，推理速度为 Qwen1.5-7B 的 **1.74×**（厂商口径，模型卡）。

2026 年 upcycling 仍是热点，且更精细：

- **DivMoE**（[2610.11317](https://arxiv.org/abs/2610.11317)）：跨域专家组合的细粒度 upcycling；
- **MASKerade**（[2610.07809](https://arxiv.org/abs/2610.07809)）：用 token 路由的 mask 专家做 dense→MoE；
- **SPRI**（[2606.16456](https://arxiv.org/abs/2606.16456)）：SVD 分割的残差初始化，面向数据受限场景；
- **Dense2MoE**（[2605.26496](https://arxiv.org/abs/2605.26496)）：统一剪枝与 upcycling，推进端侧 LLM 的帕累托前沿；
- **Cluster-Aware Upcycling**（[2604.13508](https://arxiv.org/abs/2604.13508)）：聚类感知初始化以增强专家专门化。

### 8.3 专家卸载与预测式驻留

既然不必每步用全部专家，最直接的推理省显存法是**专家换入换出（expert offloading）**：把冷门专家放 CPU/更慢的内存，按路由预测提前换入。2026 年这条路线进展密集：

- **MoE-CORE**（[2610.01950](https://arxiv.org/abs/2610.01950)）：显存受限下协同的专家卸载与驻留；
- **RapidMoE**（[2610.01265](https://arxiv.org/abs/2610.01265)）：利用跨层非对称性的自适应残差卸载；
- **Mira**（[2609.38090](https://arxiv.org/abs/2609.38090)）：自适应缓存 + 预测式专家暂存；
- **CPU-GPU 协同**（[2610.23006](https://arxiv.org/abs/2610.23006)）：自适应专家分配；
- **HiNa-MoE**（[2610.05123](https://arxiv.org/abs/2610.05123)）：在带矩阵引擎的 CPU 上高性能、非侵入式跑 MoE。

核心技术矛盾是**换入延迟 vs 命中率**：预测得准可以把换入隐藏在计算之下，预测不准则 GPU 干等。这要求路由器输出可预测、跨层路由有规律——也反过来影响架构设计。

### 8.4 专家复制与动态冗余

高负载专家在多 GPU 部署时会成为瓶颈。EPLB 的**冗余专家**把热专家复制多份、分散到不同 GPU，请求按负载任选一份，等价于在推理侧做"专家级负载均衡"。DeepEP V2.5 进一步支持**动态冗余专家**：在 NVLink 上预取专家权重与量化尺度（`lb_prefetch_weights`）、归约梯度（`lb_reduce_grads`），让冗余副本可以按需动态建立而非静态放置。

### 8.5 专家量化与总体成本账

专家权重同样可量化（W4/W8、FP8/FP4），且 DeepEP 原生支持 FP8 dispatch，把通信也低精度化。MoE 的量化有个额外好处：专家独立，量化误差更难在全网络累积，且热专家可高精度、冷专家可低精度做混合精度。需要强调的是总体拥有成本：MoE 虽降低训练算力与单 token 计算，但**显存占用、节点间通信、调度复杂度**上升，在"单用户、低延迟、小并发"场景（如端侧）未必划算；它真正发光的是**高并发、大 batch 的云端服务**——此时专家被充分复用，容量优势兑现为单位 token 成本。

### 8.6 2026 前沿：循环 MoE、软 MoE、专家剪枝

- **循环 / Looped MoE**：[Scaling Laws for Looped MoE](https://arxiv.org/abs/2609.40316) 与 [SMELT](https://arxiv.org/abs/2609.01343) 让同一组专家被循环调用多次（权重共享、深度可变），用少量专家模拟更深/更大的计算，缩放定律显示其在算力匹配下很高效。
- **token 自适应软计算**：[Share First, Route What Remains](https://arxiv.org/abs/2608.10392) 统一"先共享、再路由剩余"的框架，与共享专家思想一脉相承。
- **专家剪枝**：除 MESA 外，[Reconstruction-Optimized Expert Pruning](https://arxiv.org/abs/2610.27123) 与 [STUN](https://arxiv.org/abs/2409.06211) 用结构化+非结构化剪枝裁掉冗余专家，直接缩小部署体积。

---

## 9. MoE 缩放定律与「无限参数」猜想

MoE 把容量与计算解耦后，一个自然的问题是：**参数（专家）还能怎么 scale？** 2026 年出现了一批专门刻画 MoE 缩放规律的工作：

- **超参数缩放定律**：[Hyperparameter Scaling Laws Across MoE Sparsity](https://arxiv.org/abs/2609.08690) 系统研究学习率等超参如何随稀疏度（N 与 k）变化，给出跨稀疏度的迁移规律，避免每换一个 MoE 配置就重新调参。
- **多语言 MoE 缩放**：[Scaling Multilingual MoE](https://arxiv.org/abs/2610.20247) 研究语言覆盖度如何改变容量与算力的分配——语言越多，专家容量的"有效利用率"越低，需要在专家数与语言数之间重新平衡。
- **循环 MoE 缩放定律**：[Looped MoE](https://arxiv.org/abs/2609.40316) 与 [SMELT](https://arxiv.org/abs/2609.01343) 给出"同一组专家循环 T 次"的算力—性能曲线，显示权重共享的循环调用在算力匹配下可逼近更大的非循环模型。

更大胆的是 [Infinite-Parameter LLMs](https://arxiv.org/abs/2609.18842)（2026-09）：它探讨让模型从**实时数据流中即时生成并适配权重**，把"专家/参数"视为可按需生成、用完即可弃的临时资源，而非固定的模型文件。若这条路线成立，MoE 的"固定 N 个专家"将演化为"按需生成的专家"，容量/计算解耦被推向极致——模型参数量在概念上不再有上限。这仍是早期猜想，但代表了 MoE 思想的逻辑终点。

---

## 10. 旗舰系统横评（2024–2026）

### 10.1 参数与架构对比

下表数据来自各厂商论文/官方 config（〔厂商〕），部分精确值经 Epoch AI 数据库交叉核实（〔第三方转述〕）。

| 模型 | 年份 | 总参数 | 激活参数 | 专家(路由+共享) | top-k | 层数 | 上下文 |
|---|---|---|---|---|---|---|---|
| Mixtral 8x7B | 2024 | 46.7B | 12.9B | 8 + 0 | 2 | 32 | 32K |
| Mixtral 8x22B | 2024 | ≈140.6B | ≈39.1B | 8 + 0 | 2 | 56 | 64K |
| DeepSeekMoE-16B | 2024 | 16.4B | 2.8B | 64 + 2 | 6 | 28 | 4K |
| DeepSeek-V2 | 2024 | 236B | 21B | 160 + 2 | 6 | 60 | 128K |
| DeepSeek-V3 / R1 | 2024–25 | 671B | 37B | 256 + 1 | 8 | 61 | 128K |
| Qwen1.5-MoE | 2024 | 14.3B | 2.7B | 60 + 0 | 4 | 24 | 8K |
| Qwen2-57B-A14B | 2024 | 57B | 14B | 64 + 0 | 8 | 28 | 128K |
| Qwen3-235B-A22B | 2025 | 235B | 22B | 128 + 0 | 8 | 94 | 40K(可扩128K) |
| Qwen3-30B-A3B | 2025 | 30B | 3B | 128 + 0 | 8 | 48 | 40K |
| Qwen3-Coder-480B | 2025 | 480B | 35B | 160 + 0 | 8 | 62 | 256K |
| Qwen3-Next-80B | 2025 | 80B | 3B | 512 + 0 | 10 | 48 | 256K |
| Kimi K2 | 2025 | 1.04T | 32.6B | 384 + 1 | 8 | 61 | 128K |
| MiniMax-Text-01 | 2025 | 456B | 45.9B | 32 + 0 | 2 | 80 | 1M(外推4M) |
| Snowflake Arctic | 2024 | 480B | 17B | 128(+dense主干) | 2 | 35 | 4K |
| GLM-4.5 | 2025 | 355B | 32B | 160 + 1 | 8 | 92 | 128K |
| GLM-4.5-Air | 2025 | 106B | 12B | 128 + 1 | 8 | 46 | 128K |
| OLMoE-1B-7B | 2024 | 6.9B | 1.3B | 64 + 0 | 8 | 16 | 4K |
| DBRX | 2024 | 132B | 36B | 16 + 0 | 4 | 40 | 32K |
| Grok-1 | 2024 | 314B | 未公布 | 8 + 0 | 2 | 64 | 8K |

### 10.2 能力侧亮点（均为厂商口径）

- **DeepSeek-R1**（[2501.12948](https://arxiv.org/abs/2501.12948)）：在 V3-Base 上纯 RLVR 训练，AIME 2024 pass@1 平均 **79.8%**、MATH-500 **97.3%**，开放权重推动推理模型普及。
- **Kimi K2**（[2507.20534](https://arxiv.org/abs/2507.20534)）：以 agentic 能力为核心卖点，SWE-Bench Verified **65.8**、Tau2-Bench **66.1**、ACEBench(En) **76.5**；用 **MuonClip（对 QK 做 clip）**解决 Muon 优化器在大规模下的不稳定，15.5T token 训练零 loss spike。
- **Qwen3**（[2505.09388](https://arxiv.org/abs/2505.09388)）：提供从 0.6B 到 235B 的 dense/MoE 全尺寸矩阵，思考/非思考模式可切换，MoE 型号以极高激活比性价比著称（30B 总参仅 3B 激活）。
- **Instella-MoE**（[2609.00791](https://arxiv.org/abs/2609.00791)，2026-09）与 **Qwen3.8-Next**（[2608.30320](https://arxiv.org/abs/2608.30320)，2026-08）：代表 2026 年开源 MoE 在训练稳定性与效率评估上的最新工程化总结。

### 10.3 设计哲学的两条路线

```
DeepSeek/Kimi/GLM 路线：中多专家(160–384) + 1 共享专家 + top-8，组限制路由
Qwen 路线：          多专家(128–512)、无共享专家 + top-8/10，靠专家数覆盖
```

前者用共享专家兜底通用知识、路由更聚焦；后者用更细更多的专家 + 更宽激活保证覆盖。两者在 2025–2026 都达到了旗舰水平，说明 MoE 设计尚未收敛到唯一最优，仍有充分探索空间。

---

## 11. MoE 生态与采用时间线

- **2017–2021（Google 时代）**：MoE 主要在 Google 内部（Shazeer/Switch/GShard/GLaM/ST-MoE）与 TPU 生态，外界难复现，工程门槛极高。
- **2022（开源基础设施萌芽）**：MegaBlocks、Tutel、Sparse Upcycling、Branch-Train-Merge 等让 GPU 上的 MoE 训练变得可行、低成本。
- **2024（开源引爆）**：Mixtral 8x7B（2024-01）以开放权重证明 MoE 可在平民算力上微调和部署；同年 DeepSeekMoE、DeepSeek-V2/V3、DBRX、Arctic、OLMoE、Qwen MoE 密集发布，**MoE 成为开源旗舰的默认选择**。
- **2025（中国团队主导 + agentic 转向）**：DeepSeek-R1、Qwen3、Kimi K2、MiniMax-01、GLM-4.5 等把 MoE 与推理/智能体能力结合，前沿 MoE 的架构创新与系统软件（DeepEP/EPLB）主要由中国团队贡献。
- **2026（极稀疏与系统化）**：768 专家、控制论负载均衡、专家卸载/循环 MoE、专家剪枝等把"极稀疏 MoE"的训练与推理做成系统化工程；DeepEP 上线昇腾 NPU，与国产算力栈协同。

> **采用率口径说明**：本环境无法直连 HuggingFace 与部分厂商官网，故未给出"MoE 占旗舰模型百分比"的精确第三方统计数字，以免编造；但从 Epoch AI notable models 数据库与上述时间线可见，2024 年后开放权重的旗舰语言模型中 MoE 已占多数。该定性判断有 Epoch AI 数据库（epoch.ai/data，2026-10-22 访问）支撑。

---

## 12. 开放问题与工程落地 Checklist

### 12.1 开放问题

1. **专家到底学到了什么？** 专家专门化可观测（不同专家被不同领域 token 激活），但"一个专家=一个可命名的技能/领域"仍不严格成立，专家角色会漂移、冗余。
2. **均衡与专门化的最优边界在哪？** "均衡过头"已被证明有害，但如何按训练阶段/领域动态确定最优不均衡程度，仍缺统一答案（控制论框架是好的开始）。
3. **推理显存墙能否根本破解？** 专家卸载受限于换入延迟与预测命中率；只有当路由高度可预测时才能完全隐藏换入，这与"保持探索"存在张力。
4. **极稀疏的极限？** 768 专家 top-3 已可行，但专家数继续增大时，路由学习、通信调度、kernel launch 开销会不会反噬收益？
5. **MoE 与新架构/新模态的组合**：MoE + 线性注意力（MiniMax-01）、MoE + 多模态、MoE + 世界模型/VLA 都在早期，统一的架构原则尚未形成。
6. **稳定性的理论保证**：loss spike、路由突变在实践中被各种 trick（clip、bias、MuonClip）压住，但缺乏端到端的收敛/稳定理论。

### 12.2 工程落地 Checklist

**训练侧**
- [ ] 优先从 dense 检查点 upcycling，而非从零预训练（省大量成本）
- [ ] 采用细粒度分割 + 1 个共享专家（或 Qwen 式多专家无共享），top-8 起步
- [ ] 用 ALF-LB（动态 bias）替代/弱化 aux-loss，避免污染主目标；bias 更新强度随训练阶段调整
- [ ] 组限制路由把每 token 节点数控制在小范围（如 ≤4），降低跨节点通信
- [ ] all-to-all 用专用内核（DeepEP 类），开启 FP8 dispatch；Hopper+NVLink+RDMA 为推荐底座
- [ ] 监控每专家 token 分布、拥塞系数与 router 熵，中期（专门化最剧烈时）重点防 spike

**推理侧**
- [ ] 认清 MoE 适合高并发云端、不适合低并发/端侧（除非配合卸载/剪枝）
- [ ] 用 EPLB 冗余专家 + prefill/decode 分阶段策略均衡 GPU
- [ ] 显存受限时启用预测式专家卸载，确保换入被计算隐藏
- [ ] 专家量化（热专家高精度、冷专家低精度）+ FP8 通信
- [ ] 配合连续批处理提高每批 token、提升专家利用率
- [ ] 部署前用真实流量统计专家热度，动态调整副本与驻留集合

---

## 13. 结语

MoE 的本质贡献不是"更多参数"，而是**把模型的知识容量与每个 token 的计算量干净地解耦**，从而在 dense 缩放日益昂贵时，开辟出一条"容量近乎自由、计算保持精瘦"的新缩放路径。九年里，它从 Google 内部的 TPU 奢侈品，变成由开源社区与中国团队主导、人人可用的旗舰骨干；2024–2026 年在专家组织、负载均衡、专家并行通信与推理系统上的突破，把它从"能跑通"推向"跑得稳、跑得省"。

下一阶段的胜负手在于：能否在**极稀疏**设定下同时守住路由稳定、把推理显存墙真正推倒，并理解专家所承载的知识结构。控制论负载均衡、预测式专家驻留、循环 MoE 乃至"按需生成专家"的无限参数猜想，已经勾勒出这条路径的轮廓。可以确定的是——**让模型调用海量知识、却只为用到的那一小部分付费**，这一思想会长期主导大模型架构的演进。

---

## 参考文献

> 下列 arXiv 编号均经 arxiv.org 摘要页/搜索页核实（标题、作者、首次提交日期）；〔G〕标注为厂商技术报告/官方仓库（无 arXiv 编号）。全部链接于 2026-10-22 核实可达。

**奠基与路由**
1. Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, 2017. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
2. Lepikhin et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*, 2020. [arXiv:2006.16668](https://arxiv.org/abs/2006.16668)
3. Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*, 2021. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
4. Lewis et al., *BASE Layers: Simplifying Training of Large, Sparse Models*, 2021. [arXiv:2103.16716](https://arxiv.org/abs/2103.16716)
5. Roller et al., *Hash Layers For Large Sparse Models*, 2021. [arXiv:2106.04426](https://arxiv.org/abs/2106.04426)
6. Du et al., *GLaM: Efficient Scaling of Language Models with Mixture-of-Experts*, 2021. [arXiv:2112.06905](https://arxiv.org/abs/2112.06905)
7. Zoph et al., *ST-MoE: Designing Stable and Transferable Sparse Expert Models*, 2022. [arXiv:2202.08906](https://arxiv.org/abs/2202.08906)
8. Zhou et al., *Mixture-of-Experts with Expert Choice Routing*, 2022. [arXiv:2202.09368](https://arxiv.org/abs/2202.09368)
9. Puigcerver et al., *From Sparse to Soft Mixtures of Experts*, 2023. [arXiv:2308.00951](https://arxiv.org/abs/2308.00951)
10. Li et al., *Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models*, 2022. [arXiv:2208.03306](https://arxiv.org/abs/2208.03306)
11. Raposo et al., *Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models*, 2024. [arXiv:2404.02258](https://arxiv.org/abs/2404.02258)
12. Komatsuzaki et al., *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints*, 2022. [arXiv:2212.05055](https://arxiv.org/abs/2212.05055)

**系统与基础设施**
13. Hwang et al., *Tutel: Adaptive Mixture-of-Experts at Scale*, 2022. [arXiv:2206.03382](https://arxiv.org/abs/2206.03382)
14. Gale et al., *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts*, 2022. [arXiv:2211.15841](https://arxiv.org/abs/2211.15841)
15. DeepSeek-AI, *DeepEP: Expert-Parallel All-to-All Communication Library*, 2025. 〔G〕 [github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
16. DeepSeek-AI, *EPLB: Expert Parallelism Load Balancer*, 2025. 〔G〕 [github.com/deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB)

**旗舰模型技术报告**
17. Jiang et al., *Mixtral of Experts*, 2024. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
18. Dai et al., *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*, 2024. [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)
19. DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
20. DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
21. Muennighoff et al., *OLMoE: Open Mixture-of-Experts Language Models*, 2024. [arXiv:2409.02060](https://arxiv.org/abs/2409.02060)
22. MiniMax, *MiniMax-01: Scaling Foundation Models with Lightning Attention*, 2025. [arXiv:2501.08313](https://arxiv.org/abs/2501.08313)
23. Qwen Team, *Qwen3 Technical Report*, 2025. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
24. Kimi Team, *Kimi K2: Open Agentic Intelligence*, 2025. [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)
25. Instella, *Instella-MoE Technical Report*, 2026. [arXiv:2609.00791](https://arxiv.org/abs/2609.00791)
26. *On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability*, 2026. [arXiv:2608.30320](https://arxiv.org/abs/2608.30320)

**2025–2026 前沿：均衡、稳定性、稀疏化与推理**
27. Han & Zhong, *A Theoretical Framework for Auxiliary-Loss-Free Load Balancing of Sparse MoE in Large-Scale AI Models*, 2025. [arXiv:2512.03915](https://arxiv.org/abs/2512.03915)
28. Do et al., *Eigenvectors of Experts are Training-free Non-collapsing Routers (SSMoE)*, 2026. [arXiv:2605.30992](https://arxiv.org/abs/2605.30992)
29. Kapusuzoglu et al., *When Load-Balancing Goes Too Far: Expert Pruning in Over-Dispersed MoE (MESA)*, 2026. [arXiv:2609.04453](https://arxiv.org/abs/2609.04453)
30. Mouzouni, *Three Phases of Expert Routing: How Load Balance Evolves During MoE Training*, 2026. [arXiv:2604.04230](https://arxiv.org/abs/2604.04230)
31. Li et al., *Rethinking Routing Exploration in Modern Mixture-of-Experts (MirrorMoE)*, 2026. [arXiv:2610.22353](https://arxiv.org/abs/2610.22353)
32. Jin et al., *ID Balancing: Stable Training of Extremely Sparse MoE via PID-Based Load Control*, 2026. [arXiv:2609.39137](https://arxiv.org/abs/2609.39137)
33. *Hyperparameter Scaling Laws Across MoE Sparsity*, 2026. [arXiv:2609.08690](https://arxiv.org/abs/2609.08690)
34. *Scaling Multilingual Mixture-of-Experts: How Language Coverage Changes Capacity and Compute Allocation*, 2026. [arXiv:2610.20247](https://arxiv.org/abs/2610.20247)
35. *Scaling Laws for Looped Mixture-of-Experts*, 2026. [arXiv:2609.40316](https://arxiv.org/abs/2609.40316)
36. *SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers*, 2026. [arXiv:2609.01343](https://arxiv.org/abs/2609.01343)
37. *Infinite-Parameter LLMs: Generating and Adapting Weights from Live Data*, 2026. [arXiv:2609.18842](https://arxiv.org/abs/2609.18842)
38. *MoE-CORE: Coordinated Expert Offloading and Residency for Memory-Constrained MoE Inference*, 2026. [arXiv:2610.01950](https://arxiv.org/abs/2610.01950)
39. *RapidMoE: Exploiting Cross-Asymmetry via Adaptive Residual Offloading for Large-Scale MoE Inference*, 2026. [arXiv:2610.01265](https://arxiv.org/abs/2610.01265)
40. *Mira: Memory-Efficient MoE Inference Using Adaptive Caching and Predictive Expert Staging*, 2026. [arXiv:2609.38090](https://arxiv.org/abs/2609.38090)
41. *Efficient CPU-GPU Collaborative MoE Inference via Adaptive Expert Assignment*, 2026. [arXiv:2610.23006](https://arxiv.org/abs/2610.23006)
42. *HiNa-MoE: High-Performance, Non-Intrusive MoE Inference on CPUs with Matrix Engines*, 2026. [arXiv:2610.05123](https://arxiv.org/abs/2610.05123)
43. *DivMoE: Fine-Grained MoE Upcycling via Cross-Domain Expert Composition*, 2026. [arXiv:2610.11317](https://arxiv.org/abs/2610.11317)
44. *MASKerade: Token-Routed Mask Experts for Dense-to-MoE Upcycling*, 2026. [arXiv:2610.07809](https://arxiv.org/abs/2610.07809)
45. *SPRI: SVD-Partitioned Residual Initialization for Data-Constrained MoE Upcycling*, 2026. [arXiv:2606.16456](https://arxiv.org/abs/2606.16456)
46. *Dense2MoE: Pushing the Pareto Frontier of On-Device LLMs via Unified Pruning and Upcycling*, 2026. [arXiv:2605.26496](https://arxiv.org/abs/2605.26496)
47. *Enhancing Mixture-of-Experts Specialization via Cluster-Aware Upcycling*, 2026. [arXiv:2604.13508](https://arxiv.org/abs/2604.13508)
48. *Share First, Route What Remains: A Unified Framework for Token-Adaptive MoE Computation*, 2026. [arXiv:2608.10392](https://arxiv.org/abs/2608.10392)
49. *STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning*, 2024. [arXiv:2409.06211](https://arxiv.org/abs/2409.06211)
50. *Reconstruction-Optimized Expert Pruning for Mixture-of-Experts Language Models*, 2026. [arXiv:2610.27123](https://arxiv.org/abs/2610.27123)

**综述**
51. *A Comprehensive Survey of Mixture-of-Experts: Algorithms, Theory, and Applications*, 2025. [arXiv:2503.07137](https://arxiv.org/abs/2503.07137)

**数据来源（参数交叉核实）**
- Epoch AI, *Notable AI Models* 数据库（Mixtral 等精确参数）：[epoch.ai/data/notable_ai_models.csv](https://epoch.ai/data/notable_ai_models.csv)（2026-10-22 访问，第三方转述厂商口径）
- 各模型官方 config（经 hf-mirror.com 镜像抓取，引用给官方 huggingface.co URL）：Mixtral、DeepSeekMoE-16B、DeepSeek-V2、Qwen 系列模型卡。
