# 世界模型（World Models）前沿技术报告：从「预测下一个词」到「预测世界的下一状态」

> 关键词：世界模型 · Next-State Prediction · JEPA · Genie 3 · V-JEPA 2 · Dreamer 4 · Cosmos · 悟界 Physis · 具身智能 · 生成式交互仿真
>
> 信源：30 篇 arXiv / 同行评审原始论文（编号经 arXiv 官方 API 逐一核实）+ Google DeepMind / Meta / NVIDIA / OpenAI / 智源官方信源 + 新华社、TechCrunch、虎嗅等权威媒体与知乎社区讨论交叉验证
>
> 报告日期：2026-10-12

## 目录

## 0. 摘要（TL;DR）

**世界模型（World Model）正在成为 2025–2026 年大模型领域最确定的新范式转移。** 2026 年 1 月，北京智源人工智能研究院在《2026 十大 AI 技术趋势》中将「世界模型成为 AGI 共识方向，Next-State Prediction 或成新范式」列为十大趋势之首，明确指出 AI 的核心正从「预测下一个词」（next-token prediction）跨越到「预测世界的下一状态」（next-state prediction）[30]。几乎同一时间窗口，Google DeepMind 的 **Genie 3**（2025-08，首个实时可交互世界模型，720p / 24fps / 分钟级一致性）[18][19]、Meta FAIR 的 **V-JEPA 2**（2025-06，100 万小时视频自监督预训练 + 62 小时机器人数据实现零样本规划）[12][44]、DeepMind 的 **Dreamer 4**（2025-09，首个纯靠离线数据在 Minecraft 挖到钻石的智能体）[21]、NVIDIA 全面开放的 **Cosmos 世界基座模型平台** [31]、李飞飞 World Labs 的 **Marble** 与智源的 **悟界 Emu3.5 / Physis** [28][29] 密集落地，Yann LeCun 离职 Meta 创办的 AMI Labs 与 World Labs 各自拿下约 10 亿美元级融资 [35]。

本报告的核心判断：

1. **世界模型不是单一模型，而是一族试图学习环境动力学 $P(s_{t+1}\mid s_t,a_t)$ 的方法**，当前可清晰划分为五条技术路线：① 潜空间模型式强化学习（World Models / PlaNet / Dreamer）；② 联合嵌入预测 JEPA（LeCun 路线，在表征空间而非像素空间预测）；③ 生成式交互仿真（Genie / GameNGen / Oasis / Dreamer 4）；④ 显式 3D 空间智能（World Labs Marble / 腾讯 HunyuanWorld）；⑤ 物理状态预测与世界基座模型平台（智源悟界 / NVIDIA Cosmos）。
2. **「视频生成 ≠ 世界模型」是当前最重要的认知澄清**。智源院长王仲远、LeCun 与知乎一线研究者都强调：能生成连续画面不等于掌握了物理规律；世界模型的关键判据是**动作条件性（action-conditioning）、物理一致性、反事实推演与可用于规划/控制** [29][36]。
3. **世界模型与 VLA（Vision-Language-Action）不是替代关系，而是互补的两个条件分布**：世界模型回答「给定状态与动作，未来会怎样」$P(s'\mid s,a)$，VLA 回答「给定观测与指令，该做什么」$\pi(a\mid o,l)$；二者正在 V-JEPA 2-AC、GR-2、GR00T N1 等工作中融合 [12][33][38]。
4. **最大的现实牵引力来自数据稀缺**：机器人与自动驾驶无法像 LLM 那样从互联网下载海量交互数据，世界模型通过在仿真/想象中生成近乎无限的合成数据与安全试错，被视为具身智能的「认知底座」和「数据工厂」 [21][31][38]。
5. **当前仍处早期**：分钟级时长、物理失真、误差累积、评测体系缺失（PSNR/SSIM/FVD 无法度量物理正确性）、sim-to-real 鸿沟未被证明可跨越，是所有路线共同的开放问题 [19][21][25]。

> 🖼️ 架构图：见 `diagrams/1-taxonomy.html`（图 1 · 世界模型五大技术路线分类学）

---

## 1. 为什么是现在：三重驱动力与一场范式转移

### 1.1 LLM 的「物理世界盲区」

大语言模型可以流畅描述「苹果落地」，却未必真正理解重力——它是在背诵语料中的共现模式，而非在内部推演物理因果。这一根本局限源于训练数据：LLM 的养料是互联网文本，而真实世界是**三维的、连续的、受物理定律约束且充满交互因果**的。LeCun 多年来一直断言「自回归 LLM 在通往人类级智能的路上是条死胡同」，其核心论据正是：纯文本无法建立对物理世界的可推演模型 [9][35]。

当 AI 的主战场从「数字空间的内容生成」转向「物理世界的感知、决策与行动」（机器人、自动驾驶、工业自动化），这一盲区从学术瑕疵变成了工程瓶颈。

### 1.2 三重驱动力

世界模型在 2025–2026 年爆发，是三股力量叠加的结果：

- **认知升维的需求**：智源将其概括为从「感知」到「认知与规划」的升维——Agent 不能只知道「眼前有什么」，还必须预判「接下来会发生什么」：杯子推到桌边会不会掉、叉起箱子后重心如何偏移、机械臂会不会碰到障碍 [30][38]。
- **物理交互数据的极端稀缺**：LLM 可以下载整个互联网的文本，但机器人无法下载「一万次真实抓取」。真实世界的摩擦、碰撞、遮挡、抖动、外力干扰都必须在真实或高保真仿真环境中反复学习，成本高昂且危险。世界模型能在「想象」中生成无限、多样、安全的经验 [21][31]。
- **技术与算力的成熟**：扩散模型 / DiT、视频分词器（video tokenizer）、可扩展 Transformer、大规模视频策展管线（如 Cosmos Curator 14 天处理 2000 万小时视频）使高保真、可交互、甚至实时（Dreamer 4 单卡 H100 ≥20 FPS）的世界模型首次成为可能 [21][31]。

### 1.3 标志性事件密度（2024-02 → 2026-06）

| 时间 | 事件 | 意义 |
|---|---|---|
| 2024-02 | OpenAI Sora 技术报告《Video generation models as world simulators》[22] | 首次由头部实验室正式提出「视频生成模型即世界模拟器」叙事 |
| 2024-02 | Genie 1（11B 参数，ICML 2024 最佳论文）[16] | 首个从无标注网络视频学习的生成式交互环境 |
| 2024-05 / 08 | DIAMOND [15] / GameNGen（神经网络实时模拟 DOOM）[20] | 扩散世界模型与「神经游戏引擎」验证 |
| 2024-12 | Genie 2：单图生成可玩 3D 世界，长时记忆 [17] | 2D → 3D 的通用性飞跃 |
| 2025-01 | NVIDIA CES 发布 Cosmos 世界基座模型平台 [32] | 「卖铲人」入场，物理 AI 基建化 |
| 2025-06 | V-JEPA 2：理解 + 预测 + 规划三合一 [12] | JEPA 路线首次实现真实机械臂零样本规划 |
| 2025-08 | **Genie 3：首个实时可交互世界模型** [18] | 720p/24fps、数分钟一致性、可提示事件 |
| 2025-09 | Dreamer 4：离线数据 Minecraft 挖钻石 [21] | 证明「在想象中训练」可扩展到复杂长程任务 |
| 2025-10 / 11 | 智源悟界 Emu3.5 [28]；World Labs Marble [39] | 国产世界学习者；首个可编辑可导航 3D 世界产品 |
| 2026-01 | 智源《2026 十大 AI 技术趋势》NSP 列首位 [30] | 行业共识的官方化表达 |
| 2026-02 | World Labs 完成 10 亿美元融资（NVIDIA/AMD/Autodesk 投资）[35] | 资本重注 |
| 2026-06 | 智源发布悟界 Physis-v0.1「通用世界基座模型」[29] | 提出 Next Physical State Prediction 第五路线 |

---

## 2. 定义与形式化：到底什么是「世界模型」

### 2.1 严格定义

在强化学习与控制的经典语境中，世界模型是智能体内部对**环境转移函数**（及奖励函数）的近似：

$$
\hat{P}(s_{t+1} \mid s_t, a_t), \qquad \hat{r}(s_t, a_t)
$$

即给定当前状态 $s_t$ 与动作 $a_t$，模型预测下一状态的分布。它赋予智能体三种能力：**预测**（行动前在内部模拟后果）、**规划**（在模型中搜索/想象多条动作序列并择优）、**想象学习**（完全或部分在模型生成的虚拟经验中训练策略）[4][6]。

认知科学的源头更早：Kenneth Craik 在 1943 年《The Nature of Explanation》中提出，心智中存在外部现实的「小尺度模型」（small-scale model），可以预演不同行动及其后果 [1]；Richard Sutton 的 **Dyna 架构**（1991）则首次在工程上把「真实经验学习」与「模型生成的模拟经验规划」统一进同一个智能体 [2][3]。

### 2.2 关键判据：视频生成 ≠ 世界模型

2024 年 Sora 引发了「视频生成模型是不是世界模拟器」的大讨论。OpenAI 自己的措辞是审慎的——「扩展视频生成模型是构建物理世界通用模拟器的**一条有希望的路径**」，并列举了 3D 一致性、遮挡、对象持久性等「涌现的模拟能力」 [22]。但哲学家/AI 研究者 Raphaël Millière 随即指出：该文本本质是公司博文而非同行评审论文，行为证据尚不足以证明模型真的在「运行模拟」而非学习画面的统计规律 [43]。

到 2026 年，业界已形成更严格的区分。**一个模型要被称为（严格意义上的）世界模型，通常需要满足：**

1. **动作/干预条件性（action / intervention conditioning）**：能响应外部动作或干预，回答「如果我这样做会怎样」的反事实问题，而非只能被动续接视频；
2. **物理一致性与对象持久性**：被遮挡或离开视野的物体仍保持稳定属性，交互符合因果与物理规律；
3. **长时程一致性**：状态在数秒至数分钟尺度上不漂移、不「幻觉」；
4. **可用于下游规划/控制**：预测能真正驱动决策，而不仅是看起来逼真。

智源院长王仲远明确强调「视频生成不等于世界模型」，世界模型的关键是预测下一**物理状态**而非生成下一帧像素 [29][36]。知乎高赞调研也提醒，2026 年「世界模型」是被滥用最严重的词之一——视频生成、latent next-token、3DGS+视频 pipeline 被混为一谈，需警惕 PR 通稿式叙事 [36]。

### 2.3 分层理解（避免概念膨胀）

综合学术与产业界的用法，本报告把相关工作分为四层，后文讨论将明确标注：

- **L1 · 严格物理世界模型**：动作条件下预测下一物理状态，可交互、可规划（Dreamer、Genie、V-JEPA 2-AC、Physis）；
- **L2 · 3D 世界生成模型**：生成可漫游、可编辑、可导出几何的 3D 场景（Marble、HunyuanWorld）；
- **L3 · 世界—行动模型 WAM**：未来状态预测与动作策略结合用于机器人（Motus、GR-2、Cosmos Reason）；
- **L4 · 广义视频生成模型**：Sora、Seedance、Wan、Vidu 等学习视觉时序规律，但多数**不是**动作条件的物理世界模型。

---

## 3. 思想源流：从心智模型到「在梦中训练」

### 3.1 认知科学与强化学习的源头（1943–2015）

- **Craik（1943）**：内部「小尺度模型」用于预演行动后果 [1]。
- **Sutton 的 Dyna（1990/1991）**：同一个学习算法既用真实经验更新价值/策略，也用环境模型生成的模拟经验更新——这是「在世界模型中做梦/规划」的 RL 源头 [2]。
- **Sutton & Barto《Reinforcement Learning: An Introduction》（1998）**：Dyna-Q 等架构系统化，模型生成经验与真实经验等价地用于规划 [3]。

### 3.2 Ha & Schmidhuber 2018：深度学习时代的奠基演示

2018 年的论文《World Models》[4]（项目页 worldmodels.github.io）让这个概念在深度学习圈爆火。其架构由三个极简组件构成：

- **V（Vision / VAE）**：把高维图像观测压缩为低维潜变量 $z_t$；
- **M（Memory / MDN-RNN）**：在动作条件下预测未来潜变量的概率分布（混合密度网络 RNN），在 VizDoom 中甚至学会预测「死亡」终止状态；
- **C（Controller）**：基于 $[z_t, h_t]$ 输出动作，CarRacing 实验中控制器仅 **867 个参数**，用 CMA-ES 进化策略训练。

最具冲击力的实验是：智能体可以**完全在 VAE+RNN 生成的「梦」中训练**，再迁移回真实 VizDoom 环境并有效行动。论文同时罕见地诚实预告了一个困扰该领域至今的问题——策略可能学会利用模拟器的缺陷「作弊」（exploiting the model），这正是后来模型利用（model exploitation）与分布漂移问题的先声 [4]。

### 3.3 PlaNet 与 Dreamer 谱系：潜空间想象的成熟（2018–2023）

DeepMind（彼时含 Google Brain）的 Hafner 等人把这条路线发展成一套可扩展的严谨范式：

| 模型 | arXiv | 核心贡献 |
|---|---|---|
| **PlaNet**（2018, ICML 2019） | [1811.04551](https://arxiv.org/abs/1811.04551) | 提出 RSSM（Recurrent State-Space Model），潜状态同时含**确定性 + 随机性**转移；在潜空间做快速在线规划（MPC）[5] |
| **Dreamer**（2019, ICLR 2020） | [1912.01603](https://arxiv.org/abs/1912.01603) | Actor-Critic 完全在**长期潜空间想象轨迹**上用解析反传学习行为，数据效率与最终性能大幅领先 [6] |
| **DreamerV2**（2020, ICLR 2021） | [2010.02193](https://arxiv.org/abs/2010.02193) | 世界模型改用**离散**潜变量；首个在 55 个 Atari 游戏上「在独立训练的世界模型中想象」达到人类水平 [7] |
| **DreamerV3**（2023） | [2301.04104](https://arxiv.org/abs/2301.04104) | **单一固定配置**跨 150+ 多样任务胜过专用算法；首个无人类数据、无课程从零在 Minecraft 收集到钻石的算法 [8] |

这条主线确立了「**学一个紧凑、结构化的潜空间动力学，在其中做规划与想象学习**」的正统方法，影响了此后几乎所有模型式 RL 工作。同期还有两条重要支线：**IRIS**（[arXiv:2209.00588](https://arxiv.org/abs/2209.00588)，离散分词 + 自回归 Transformer，Atari 100k 上仅用约两小时游戏经验就在 26 个游戏中 10 个超过人类）[13]、**TWM**（[arXiv:2303.07109](https://arxiv.org/abs/2303.07109)，Transformer-XL 直接关注历史状态/动作/奖励，10 万次交互即可）[14]，以及把扩散模型引入世界建模的 **DIAMOND**（[arXiv:2405.12399](https://arxiv.org/abs/2405.12399)，用 EDM 风格扩散在 Atari「梦」中建模视觉动力学，证明视觉细节对策略性能至关重要）[15]。

> 🖼️ 架构图：见 `diagrams/2-evolution-timeline.html`（图 2 · 世界模型 1943–2026 思想源流与里程碑时间线）

---

## 4. 五大技术路线全景

当代深度学习世界模型已分化为五条清晰可辨的路线。它们在「**在哪个空间做预测**」这一根本问题上分道扬镳：像素空间、离散 token 空间、抽象表征空间、显式 3D 空间，还是统一物理状态空间。

### 4.1 路线一：潜空间模型式强化学习（Latent Dynamics / Model-Based RL）

即第 3 章的 World Models → PlaNet → Dreamer 谱系。其信条是：**不要在高维像素上预测，先压缩成紧凑潜状态，再在潜空间里学习动力学、做规划与想象训练**。RSSM 的「确定性 + 随机性」双分支设计、actor-critic 在想象轨迹上反传、离散潜变量等是其核心遗产。优点是数据效率极高、理论干净；瓶颈是长期受限于简单环境（Atari/DM Control），且强依赖模型保真度，策略容易钻模型误差的空子 [4][5][8]。

### 4.2 路线二：联合嵌入预测 JEPA（LeCun 路线）

Yann LeCun 在 2022 年的立场文档《A Path Towards Autonomous Machine Intelligence》（[OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf)）中系统阐述了其反对逐像素生成的理由：真实世界充满不可预测的高频细节（每片草叶、每片树叶的确切位置），强迫模型重建像素会把容量浪费在无关细节上；正确做法是在**学到的抽象表征空间**里预测可预测的部分，并以基于能量的模型（EBM）框架衡量预测与目标表征的相容性 [9]。

JEPA（Joint-Embedding Predictive Architecture）家族沿图像→视频→可规划三级演进：

- **I-JEPA**（[arXiv:2301.08243](https://arxiv.org/abs/2301.08243)，2023）：从图像的 context block 预测同一图像 target block 的**表征**而非像素，是非生成式方法，不依赖手工视图增强 [10]；
- **V-JEPA**（[arXiv:2404.08471](https://arxiv.org/abs/2404.08471)，2024）：把特征预测目标搬到视频，仅用特征预测、不用像素重建/文本/负样本/人工标注，在约 200 万个视频上训练 [11]；
- **V-JEPA 2**（[arXiv:2506.09985](https://arxiv.org/abs/2506.09985)，2025）：集大成者，详见第 5.2 节 [12]。

需要特别澄清：JEPA 的「mask-denoising」**不是**去噪自编码器式的像素重建，而是 $s_x=f_\theta(x)$、$\hat{s}_y=g_\phi(s_x,c)$——从上下文嵌入预测目标块的联合嵌入表示，再用能量/距离度量相容性 [9][10]。

### 4.3 路线三：生成式交互仿真（Generative Interactive Environments）

这是 2024 年以来最吸引眼球的路线：直接（或经潜空间）生成**能实时响应动作的未来画面**，把世界模型做成一个「神经仿真器 / 可玩的生成环境」。

- **Genie 1**（[arXiv:2402.15391](https://arxiv.org/abs/2402.15391)，11B 参数，ICML 2024 最佳论文）：首个以无监督方式从无标注网络视频训练的生成式交互环境，可用文本/图片/草图提示生成可逐帧控制的 2D 世界；三大组件为**时空视频分词器 + 自回归动力学模型 + 潜在动作模型**（无需真值动作标签）[16]；
- **Genie 2**（2024-12 官方博客）：单张提示图 + 键鼠操作生成可玩 3D 世界，最长约 1 分钟一致（多数示例 10–20 秒），具备长时记忆与反事实轨迹生成 [17]；
- **GameNGen**（[arXiv:2408.14837](https://arxiv.org/abs/2408.14837)，2024）：首个完全由神经网络驱动、可实时交互的 DOOM 游戏引擎，扩散模型在人类对局轨迹上训练 [20]；
- **Oasis**（Decart/Etched，2024-10）：首个实时可玩的生成式 Minecraft，但当时空一致性较弱；
- **Matrix-Game**（[arXiv:2506.18701](https://arxiv.org/abs/2506.18701)，2025）：交互式世界基座模型 [27]；
- **Genie 3 / Dreamer 4**（2025）：把保真度、时长与实时性推到新高度，详见第 5 章 [18][21]。

这条路线的核心赌注是：**如果生成的环境足够真实、足够多样，在其中训练出的智能体就可能泛化到现实世界**。DeepMind 公开承认这「仍是一个尚未被证明的假设」，也是该路线最大的风险 [19][35]。

### 4.4 路线四：显式 3D 空间智能（Spatial Intelligence）

李飞飞创立的 World Labs 走了一条与 LeCun「时间预测」正交的路：**显式重建三维世界的几何结构、深度与物体相对位置**。其首款产品 **Marble**（2025-11 发布）接受文字、图片、视频或粗糙 3D 草图，输出的不是视频而是**可编辑、可导航、可导出**的 3D 世界——可旋转视角、移动物体、改变光照，并导出为高斯溅射（Gaussian Splatting）、三角网格或视频，直接拖入 Unreal Engine / Unity 使用 [39]。其关键差异在于底层维护**统一的 3D 空间表示**，因此「转身回来，世界还是那个世界」（空间一致性），而非逐帧编故事。团队联创 Ben Mildenhall 正是 NeRF 发明者 [35][39]。

国内腾讯混元的 **HunyuanWorld 1.0**（2025-07 开源，[GitHub](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0)）同属此路线：以全景图作为 world proxy，生成可沉浸、可探索、可交互的 3D 世界，支持语义分层与 mesh 导出，后续迭代到 WorldPlay 1.5、HY-World 2.0 [40]。这条路线擅长重建空间的「样子」，但对空间中「会发生什么」的动态物理理解仍较弱（Marble 行走几步后也会出现视觉变形）[35]。

### 4.5 路线五：物理状态预测与世界基座模型平台（WFM Platform）

NVIDIA 与智源代表了「平台化 / 物理状态化」的最新方向。

**NVIDIA Cosmos**（2025-01 CES 发布，2025-03 GTC 大版本更新）是一个全栈开放的**世界基座模型（World Foundation Models）平台**，含三类模型与一条数据管线 [31][32]：

- **Cosmos Predict**：从文本/图像/视频多模态输入生成未来世界状态，支持给定首尾帧预测中间多帧动作/运动轨迹；
- **Cosmos Transfer**：吃入分割图、深度图、激光雷达、位姿、轨迹等结构化输入，生成可控的照片级视频，把 Omniverse 仿真转成真实感合成数据；
- **Cosmos Reason**：开放可定制、具时空意识的推理模型，用思维链理解视频并以自然语言预测交互结果（如「行人踏入斑马线」「箱子从架子掉落」），可做高层规划器或 VLA 组件；
- **Cosmos Curator 数据管线**：官方称 14 天可处理 2000 万小时视频（传统 CPU 方案需 3 年以上）；视觉 Tokenizer 压缩率比业界方案高 8× [31][35]。

黄仁勋的定位是：「**世界基础模型之于物理 AI，就像大语言模型之于生成式 AI**」。其商业逻辑是用免费开放的软件锁定硬件生态——1X、Agility、Figure AI、Skild、Uber、Waabi、小鹏等均为早期采用者 [31][35]。

**智源悟界**则把路线进一步理论化。王仲远团队将既有路线归为四类（语言中心、像素中心、三维结构、视觉表征），并提出第五条路：在统一多模态/全模态潜空间中直接预测**下一物理状态**（Next Physical State Prediction, NPSP）。2026-06 北京智源大会发布的 **悟界 Physis-v0.1** 自称「全球首个通用世界基座模型」，输入涵盖视频、深度 RGB、3D 点云、力触反馈，在物理隐空间中预测，官方称支持 50+ 复杂物理场景的长程推理（截至本报告核实日，未公开权重/参数，需谨慎对待）[29]。此前的 **悟界 Emu3.5**（[arXiv:2510.26583](https://arxiv.org/abs/2510.26583)，总参数 34.1B，10T+ 多模态 token 训练，论文标题即 *Native Multimodal Models are World Learners*）则证明统一 next-token 预测也能涌现「世界探索、开放世界具身操作」能力 [28]；更早的 Emu3 相关工作于 2026-01 发表于 *Nature* [42]。

> 🖼️ 架构图：见 `diagrams/3-five-routes.html`（图 3 · 五大路线在「预测空间」坐标轴上的定位）

---

## 5. 旗舰系统深度拆解

### 5.1 Genie 3：首个实时可交互世界模型

2025 年 8 月 5 日，Google DeepMind 由 Jack Parker-Holder 与 Shlomi Fruchter 署名发布 **Genie 3**，官方称之为「首个实时、可交互的世界模型」[18]。经核实的官方关键参数 [18][19]：

- 输入一句文本提示即可生成可导航的动态世界，**720p 分辨率、24 fps 实时生成**（产品页标注 20–24 fps），单次可保持**数分钟（a few minutes）**一致性——相较 Genie 2 的 10–20 秒是数量级跃升；
- **环境/对象持久性**：重访时能回忆此前见过的细节，打碎的花瓶碎片会留在原地，离开再回来仍在；官方称这种记忆并非显式编程，而是规模下的**涌现属性**；
- **可提示世界事件（promptable world events）**：生成过程中可用一条提示即时介入、改变世界（如切换天气）；
- 产品页明确「Grounded in Street View」，即基于 Google Maps 街景数据做现实锚定。

技术上 DeepMind **没有发布配套论文**，仅有博客与发布会采访。可知线索：模型是**自回归、逐帧生成**的——「一次生成一帧，必须回看此前生成的内容来决定接下来发生什么」，不依赖硬编码物理引擎，物体运动/碰撞规则全部从数据中学得 [19]。官方还演示了与 **SIMA 智能体**的闭环：Genie 3 前向模拟环境，SIMA 输出动作，在仓库场景完成「走向绿色压缩机」「靠近红色叉车」等指令 [19]。

官方坦承局限：物理仍有失真（如滑雪者与雪的关系不真实）、智能体可执行动作范围有限、多智能体复杂交互困难、只能支撑数分钟而真实训练需要数小时 [19]。

### 5.2 V-JEPA 2：100 万小时视频 + 62 小时机器人数据 → 零样本规划

Meta FAIR（Assran、Bardes、LeCun、Ballas 等）2025 年 6 月的 **V-JEPA 2**（[arXiv:2506.09985](https://arxiv.org/abs/2506.09985)）是 JEPA 路线迄今最强证据，采用清晰的两阶段设计 [12]：

1. **Action-free 预训练**：在 **100 万小时以上**互联网视频 + 100 万图像上，用视觉 mask-denoising（表征空间特征预测）目标训练最大 **10 亿参数**的 V-JEPA 2 编码器；
2. **动作条件后训练**：冻结视频编码器，仅用 Droid 数据集中**不到 62 小时**无标签机器人视频，训练一个 300M 参数、block-causal 注意力的 Transformer 预测器 **V-JEPA 2-AC**，自回归预测「给定动作与历史状态后的下一帧表征」；
3. **规划部署**：在 MPC（模型预测控制）循环中用能量最小化搜索动作，**零样本**部署到两个不同实验室的 Franka 机械臂，仅凭单目 RGB 与图像目标完成抓取/放置，且**未在这些环境采集任何数据、无任务特定训练、无奖励函数**。

经核实的基准成绩：Something-Something v2 动作理解 **77.3** top-1；Epic-Kitchens-100 动作预测 **39.7** recall@5（超此前任务专用模型）；与 LLM 对齐后在 8B 规模拿下视频问答 PerceptionTest **84.0**、TempCompass **76.9** 的 SOTA [12][44]。论文同样诚实列出局限：对相机位姿敏感、长时程规划误差累积、图像目标泛化受限 [12]。

### 5.3 Dreamer 4：在「想象」中训练可扩展智能体

DeepMind 的 Hafner、Yan、Lillicrap 于 2025 年 9 月 29 日发表 **Dreamer 4**（[arXiv:2509.24527](https://arxiv.org/abs/2509.24527)），把潜空间想象范式推向复杂 3D 环境 [21]：

- **因果分词器（causal tokenizer）**：用掩码自编码 + MSE/LPIPS 把视频帧压成连续潜表示，强制时间因果以支持逐帧解码；
- **快捷强制目标（shortcut forcing）**：扩展 diffusion forcing / shortcut models，仅 4 步采样即可达到 64 步扩散强制的质量（约 16× 加速），采用 x-prediction 降低长 rollout 误差累积（FVD 从 306 降到 57）；
- **高效 Transformer**：轴向注意力（空间/时间分离）、每 4 层才做一次稀疏时间注意力、GQA、register tokens，实现**单张 H100 上 ≥20 FPS 实时交互**，上下文窗口达 **9.6 秒**（此前模型的 6 倍）；
- **三阶段训练**：① 世界模型预训练（可含大量无动作标签视频）；② 插入策略/奖励头，用行为克隆 + 多 token 预测微调；③ **想象训练**：冻结世界模型，在其生成的想象 rollout 上用 PMPO（带 KL 正则到 BC 先验）做强化学习。

标志性成果：在 Minecraft 钻石任务（需连续选择 **2 万多次**键鼠动作的长程稀疏奖励任务）上，Dreamer 4 是**首个纯靠离线数据、无任何环境交互**获得钻石的智能体，成功率 0.7%；中间里程碑石镐 >90%、铁镐 29%；仅用 VPT 约 **1/100 的标注数据**（2.5K 小时视频，其中约 100 小时动作标注即达到全监督 85% PSNR / 100% SSIM 的动作条件效果），且动作条件能从主世界泛化到仅在无标签视频中见过的下界/末地 [21]。其局限同样明显：钻石绝对成功率仍低、长时程仍漂移、UI 状态不稳定、完全在模型内优化存在 reward hacking 风险、缺乏不确定性估计 [21]。

### 5.4 三家旗舰的横向对比

| 维度 | Genie 3 [18][19] | V-JEPA 2(-AC) [12] | Dreamer 4 [21] |
|---|---|---|---|
| 出品方 | Google DeepMind | Meta FAIR | Google DeepMind |
| 预测空间 | 像素（自回归逐帧） | 抽象表征（JEPA） | 连续潜 token（扩散式快捷强制） |
| 公开程度 | 仅博客，无论文 | 完整论文 + 开源代码/权重 | 完整论文 |
| 交互/实时性 | 720p / 24fps，实时 | 非实时生成，用于 MPC 规划 | 单 H100 ≥20 FPS，可实时玩 |
| 一致性时长 | 数分钟 | 受限于规划时域（短程操纵） | 9.6 秒上下文，长 rollout 漂移 |
| 动作条件 | 键鼠/提示事件 | 机器人关节动作（62h 数据） | 键鼠（约 100h 标注） |
| 标志性验证 | SIMA 闭环导航 | 两实验室 Franka 零样本抓取 | 离线数据 Minecraft 挖钻石 |
| 哲学 | 生成式仿真，学物理 | 表征空间预测，弃像素 | 潜空间高保真想象 + 想象内 RL |

> 🖼️ 架构图：见 `diagrams/4-flagship-architecture.html`（图 4 · 三大旗舰的技术架构对比：分词器 → 动力学 → 策略/规划）

---

## 6. 世界模型 × 具身智能：与 VLA 是互补而非替代

### 6.1 两个条件分布

机器人「大脑」的两个核心问题在概率上对应两个不同的条件分布：

- **世界模型（动力学）**：$P(s_{t+1}\mid s_t, a_t)$ —— 给定状态与动作，未来会怎样；
- **VLA（策略）**：$\pi(a_t\mid o_t, l_t)$ —— 给定视觉观测 $o$ 与语言指令 $l$，应输出什么动作。

二者天然互补：世界模型提供「想象力/预判力」与合成数据，VLA 提供「感知—理解—执行」主链路。业界（如仙工智能等产业方）的共识是：**世界模型不是来终结 VLA 的，而是推动 VLA 从「感知-理解-动作」升级到「预测-推演-执行」**；真正的机器人大脑是「VLA + 世界模型 + 控制系统 + 真实数据闭环」的系统工程，任何单一模型都不是完整答案 [38]。

### 6.2 融合的代表性工作

- **V-JEPA 2-AC**：冻结的世界模型 + MPC 规划器直接产出动作，是「世界模型驱动控制」的极简形态 [12]；
- **字节 GR-2**（[arXiv:2410.06158](https://arxiv.org/abs/2410.06158)）：生成式视频-语言-动作模型，把 web-scale 知识与生成式未来预测融入机器人操作策略 [33]；
- **生数科技 × 清华 Motus**（2025-12 开源，约 8B 参数 MoT 架构，集成视频生成 + VLM + 动作专家）：定位统一 latent action world model / WAM，追求零样本泛化与跨本体适配 [34]；
- **NVIDIA GR00T N/N1、Cosmos Reason**：把世界基座模型作为机器人通用大脑/高层规划器的组件 [31]；
- **1X、Agility、Figure AI、Skild** 等机器人公司则用 Cosmos Transfer/Predict 大规模生成训练数据 [31]。

### 6.3 为什么世界模型是具身智能的数据答案

真实机器人交互数据「稀缺且昂贵」，而世界模型可以：① 在仿真中生成近乎无限的长尾场景（极端天气、罕见障碍、危险工况）；② 让策略在「想象」中安全试错，避免真机损坏；③ 作为逆动力学/动作标注器，把海量无标签人类视频变成可学习资源。这正是智源「合成数据破除数据枯竭」、黄仁勋「Physical AI Data Factory」叙事的技术内核 [30][31]。但 sim-to-real 的最终鸿沟——生成环境里训出的能力能否可靠迁移到现实——仍是整个领域「未被证明的核心假设」[19][35]。

---

## 7. 产业与生态地图（2025–2026）

### 7.1 国际：研究巨头 + 明星创业公司 + 卖铲人

- **Google DeepMind**：Genie 1/2/3 + Dreamer 1–4 + SIMA，是生成式交互仿真与潜空间 RL 双料领先者，并与 NVIDIA 合作为 Cosmos 提供 SynthID 水印 [16][18][21][31]；
- **Meta FAIR**：JEPA 全系（I-/V-/V-JEPA 2），LeCun 2025 年从 Meta 退休后创办 **AMI Labs**，首轮融资超 **10 亿美元**（创欧洲 AI 种子轮纪录），坚持 JEPA / 世界模型路线，但他本人坦言商业化产品仍需数年 [35]；
- **OpenAI**：Sora 提出「世界模拟器」叙事（2024-02），Sora 2（2025-09-30）持续推进视频物理真实感，但严格动作条件世界模型尚未作为独立产品发布 [22][23]；
- **NVIDIA**：Cosmos（Predict/Transfer/Reason）+ Omniverse + GR00T + Curator，以开放软件锁定 CUDA/硬件生态，是明确的「卖铲人」[31][32]；
- **World Labs（李飞飞）**：Marble 开创可编辑可导航 3D 世界品类，2026-02 完成 **10 亿美元**融资（NVIDIA、AMD、Autodesk 投资）[35][39]。

### 7.2 中国：从视频生成到世界基座的全面跟进

| 机构 | 代表工作 | 路线分层 |
|---|---|---|
| 智源 BAAI | 悟界 Emu3 / Emu3.5（[2510.26583](https://arxiv.org/abs/2510.26583)）、Physis-v0.1、RoboBrain Orca [28][29][42] | L1/L4，NPSP 第五路线 |
| 腾讯混元 | HunyuanWorld 1.0/1.1、WorldPlay、HY-World 2.0 [40] | L2 显式 3D |
| 字节跳动 | Seedance（视频）、Research GR-2（机器人 WAM）[33] | L3/L4 |
| 阿里通义 | Wan2.1 / Wan2.1-VACE（开源视频基座，常作 WAM backbone） | L4 |
| 生数科技 × 清华 | Vidu Q1（数字世界 WGM）+ Motus（物理世界 WAM）[34] | L3/L4 |
| 自动驾驶（小鹏、华为、Waabi 等） | 用 Cosmos/自研世界模型做驾驶仿真与长尾合成数据 [31] | L1/L3 |

需要强调：国产多数「视频生成大模型」属于 L4，不能直接等同于严格物理世界模型；真正明确宣称动作条件物理状态预测的是智源 Physis、生数 Motus 等少数工作，且 Physis 尚未公开权重，宣传与可核实能力之间应保留审慎距离 [29][36]。

### 7.3 资本与共识

世界模型在 2025H2–2026 成为资本与顶级人才共同押注的赛道：AMI Labs 与 World Labs 两家公司在数周内各拿约 10 亿美元 [35]；智源将其列十大趋势之首 [30]；黄仁勋称其为「物理 AI 的 LLM 时刻」[31]。驱动力的另一面是 LLM 边际收益递减、资本需要新叙事、具身智能崛起三者的叠加 [35]。

---

## 8. 评测：怎么判断一个世界模型「真的懂世界」

### 8.1 传统视频指标为何不够

**PSNR / SSIM** 度量像素级重建误差，**FVD** 度量生成视频分布距离，但它们都无法回答世界模型最关键的问题：预测在**物理与因果上是否正确**。一个逐像素 PSNR 很高的模型仍可能让物体穿墙、让碎片凭空消失。Dreamer 4 也承认 PSNR/SSIM 不捕捉语义任务正确性 [21]。

### 8.2 更应关注的维度与基准

- **动作条件准确率 / 反事实一致性**：给定不同动作，未来是否相应地、正确地改变；
- **对象持久性与长时程稳定性**：离开视野的物体属性是否保持；
- **物理合理性**：碰撞、重力、遮挡、材质交互是否可信；
- **下游控制成功率**：预测能否真正提升规划/控制（最硬的端到端指标，如 V-JEPA 2 的零样本抓取、Dreamer 4 的钻石成功率）；
- 专用基准：**WorldModelBench**（[arXiv:2502.20694](https://arxiv.org/abs/2502.20694)，专门以「视频模型是否像世界模型」为评判目标，从反事实/动作条件等角度测评）[25]；V-JEPA 2 所用的 Something-Something v2、Epic-Kitchens-100、PerceptionTest、TempCompass 等理解/预测基准 [12]。

评测体系的不成熟本身就是早期阶段的标志：截至目前，业界仍缺乏一个被广泛接受、覆盖「反事实正确性 + 长时程 + 下游控制」的标准化世界模型基准 [21][25]。

---

## 9. 开放问题与风险

1. **长时程一致性与误差累积**：Genie 3 的「数分钟」、Dreamer 4 的 9.6 秒上下文，距离支撑数小时训练所需的稳定环境仍有量级差距；自回归/扩散 rollout 的误差会随时间放大 [19][21]。
2. **物理保真度**：神经世界模型没有硬编码物理引擎，碰撞/接触/流体等仍频繁失真，能否完全替代传统仿真引擎存疑 [19][35]。
3. **Sim-to-Real 未被证明**：「在足够真实的生成环境中训练 → 泛化到现实」是整个生成式路线的核心赌注，迄今只有小规模、短程任务的证据 [19][12]。
4. **模型利用与奖励黑客**：策略完全在学习到的模型中优化时，会钻模型误差的空子（Ha & Schmidhuber 2018 已预警，Dreamer 4 指出尚无系统对策）[4][21]。
5. **不确定性与安全**：现有模型多为点预测，缺乏认知不确定性估计，难以在「不知道」时安全拒动；这对机器人/自动驾驶是落地生死线 [21][30]。
6. **概念泡沫与营销叙事**：「世界模型」标签被严重滥用，视频生成 demo 被包装成物理理解；需要以动作条件性、物理一致性、下游控制三项硬指标去伪存真 [36]。
7. **算力与数据门槛**：Dreamer 4 训练需 256–1024 TPU-v5p 量级 [21]；高保真可交互世界模型仍是巨头游戏。
8. **评测与理论缺失**：缺乏标准基准与对 shortcut forcing 等新目标的形式化稳定性/误差界分析 [21][25]。

---

## 10. 结论与展望

**世界模型之所以是当前最值得关注的方向，不是因为它已经成功，而在于它第一次让「AI 理解物理世界」成为一个有清晰技术路线、有旗舰系统验证、有巨头与资本全力投入的工程问题。** 五条路线——潜空间 RL、JEPA、生成式交互仿真、显式 3D、物理状态平台——并非非此即彼，而更像同一块拼图的不同碎片：

- **短期（1 年内）**：世界模型最大的确定性价值在**合成数据与仿真**——为自动驾驶、机器人批量生产长尾、可控、安全的训练数据，Cosmos 类平台与 WAM 模型率先商业化 [31][34]；
- **中期（1–3 年）**：路线融合是主旋律——JEPA 的高效表征 + 生成式模型的高保真 + 显式 3D 的空间一致性 + 物理状态先验，共同服务于 VLA 的「预测-推演-执行」闭环；评测基准与 sim-to-real 证据将成为分水岭 [12][38]；
- **长期**：若「在想象中训练、在现实中泛化」的假设被证实，世界模型将成为具身 AGI 的认知底座；若被证伪，它仍会作为强大的数据工厂与规划模块留存。LeCun「五年后没人用今天这样的 LLM」的断言虽激进，但其趋势判断值得重视：**AI 已经读完了万卷书，接下来要行千里路，而世界模型就是它的眼睛和四肢** [35]。

对研究者与工程团队的行动建议：① 用第 2.2 节的四项硬判据甄别「真世界模型」与「视频生成包装」；② 优先在数据合成、安全试错、模型预测控制三个已被验证的场景落地；③ 同时跟踪 JEPA 开源权重（V-JEPA 2）与 Cosmos 开放平台两条可直接复用的路径；④ 建立以反事实一致性和下游控制成功率为核心的内部评测，而非只看视频画质指标。

---

## 参考文献

> 说明：所有 arXiv 编号均经 arXiv 官方 API（export.arxiv.org/api）逐一核实标题与存在性；官方博客/产品页为一手信源，媒体与知乎为交叉验证信源并在正文标注。访问日期：2026-10-12。

**一、思想源流与经典世界模型**

1. Kenneth J. W. Craik. *The Nature of Explanation*. Cambridge University Press, 1943. https://www.cambridge.org/core/journals/philosophy/article/nature-of-explanation-by-k-j-w-craik-cambridge-university-press-1943-pp-viii-123-price-6s/4E205C90F90E12A9123D83EDF6D21168
2. Richard S. Sutton. *Dyna, an Integrated Architecture for Learning, Planning, and Reacting*. SIGART Bulletin, 2(4):160–163, 1991. https://doi.org/10.1145/122344.122377
3. Richard S. Sutton & Andrew G. Barto. *Reinforcement Learning: An Introduction*. MIT Press, 1998. http://incompleteideas.net/book/the-book.html
4. David Ha, Jürgen Schmidhuber. *World Models*. arXiv:1803.10122, 2018. https://arxiv.org/abs/1803.10122 ｜项目页：https://worldmodels.github.io/
5. Danijar Hafner 等. *Learning Latent Dynamics for Planning from Pixels* (PlaNet). arXiv:1811.04551, 2018（ICML 2019）. https://arxiv.org/abs/1811.04551
6. Danijar Hafner 等. *Dream to Control: Learning Behaviors by Latent Imagination* (Dreamer). arXiv:1912.01603, 2019（ICLR 2020）. https://arxiv.org/abs/1912.01603
7. Danijar Hafner 等. *Mastering Atari with Discrete World Models* (DreamerV2). arXiv:2010.02193, 2020（ICLR 2021）. https://arxiv.org/abs/2010.02193
8. Danijar Hafner 等. *Mastering Diverse Domains through World Models* (DreamerV3). arXiv:2301.04104, 2023. https://arxiv.org/abs/2301.04104

**二、JEPA 路线（LeCun / Meta FAIR）**

9. Yann LeCun. *A Path Towards Autonomous Machine Intelligence*. 2022. https://openreview.net/forum?id=BZ5a1r-kVsf
10. Mahmoud Assran 等. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA). arXiv:2301.08243, 2023. https://arxiv.org/abs/2301.08243
11. Adrien Bardes 等. *Revisiting Feature Prediction for Learning Visual Representations from Video* (V-JEPA). arXiv:2404.08471, 2024. https://arxiv.org/abs/2404.08471
12. Mahmoud Assran 等（含 Yann LeCun, Nicolas Ballas）. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*. arXiv:2506.09985, 2025. https://arxiv.org/abs/2506.09985 ｜代码：https://github.com/facebookresearch/vjepa2 ｜博客：https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks

**三、生成式 / Transformer / 扩散世界模型**

13. Vincent Micheli, Eloi Alonso, François Fleuret. *Transformers are Sample-Efficient World Models* (IRIS). arXiv:2209.00588, 2022（ICLR 2023）. https://arxiv.org/abs/2209.00588
14. Jan Robine 等. *Transformer-based World Models Are Happy With 100k Interactions* (TWM). arXiv:2303.07109, 2023. https://arxiv.org/abs/2303.07109
15. Eloi Alonso 等. *Diffusion for World Modeling: Visual Details Matter in Atari* (DIAMOND). arXiv:2405.12399, 2024. https://arxiv.org/abs/2405.12399 ｜项目页：https://diamond-wm.github.io
16. Jake Bruce 等（Google DeepMind）. *Genie: Generative Interactive Environments*. arXiv:2402.15391, 2024（ICML 2024 最佳论文）. https://arxiv.org/abs/2402.15391
17. Google DeepMind. *Genie 2: A Large-Scale Foundation World Model*. 官方博客, 2024-12-04. https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
18. Jack Parker-Holder, Shlomi Fruchter 等. *Genie 3: A New Frontier for World Models*. Google DeepMind 官方博客, 2025-08-05. https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/ ｜产品页：https://deepmind.google/models/genie/
19. TechCrunch. *DeepMind Thinks Genie 3 World Model Presents a Stepping Stone Towards AGI*. 2025-08-05. https://techcrunch.com/2025/08/05/deepmind-thinks-genie-3-world-model-presents-stepping-stone-towards-agi
20. Anchit Gupta 等. *Diffusion Models Are Real-Time Game Engines* (GameNGen). arXiv:2408.14837, 2024. https://arxiv.org/abs/2408.14837
21. Danijar Hafner, Wilson Yan, Timothy Lillicrap. *Training Agents Inside of Scalable World Models* (Dreamer 4). arXiv:2509.24527, 2025-09-29. https://arxiv.org/abs/2509.24527
22. OpenAI. *Video Generation Models as World Simulators*（Sora 技术报告）. 2024-02-15. https://openai.com/index/video-generation-models-as-world-simulators/
23. OpenAI. *Sora 2*. 官方发布页, 2025-09-30. https://openai.com/index/sora-2/
24. （预留）
25. *WorldModelBench: Judging Video Generation Models As World Models*. arXiv:2502.20694, 2025. https://arxiv.org/abs/2502.20694
26. （预留）
27. *Matrix-Game: Interactive World Foundation Model*. arXiv:2506.18701, 2025. https://arxiv.org/abs/2506.18701

**四、中国生态 / 世界基座模型**

28. 北京智源人工智能研究院（BAAI）. *Emu3.5: Native Multimodal Models are World Learners*. arXiv:2510.26583, 2025-10-30. https://arxiv.org/abs/2510.26583 ｜模型：https://huggingface.co/BAAI/Emu3.5 ｜代码：https://github.com/baaivision/Emu3.5
29. 北京智源人工智能研究院. *悟界 Physis-v0.1 全球首个通用世界基座模型 / RoboBrain Orca*（2026 北京智源大会发布）. 智源社区官方, 2026-06-13. https://hub.baai.ac.cn/view/55492
30. 新华网. *智源发布 2026 十大 AI 技术趋势*. 2026-01-08. http://www.news.cn/tech/20260108/b99127c88a4640bbab49e5c6294264d2/c.html
31. NVIDIA. *NVIDIA Announces Major Release of Cosmos World Foundation Models and Physical AI Data Tools*（GTC 新闻稿）. 2025-03-18/19. https://nvidianews.nvidia.com （转载：https://finance.yahoo.com/news/nvidia-announces-major-release-cosmos-191300896.html ）｜官方：https://www.nvidia.com/en-us/ai/cosmos/ ｜代码：https://github.com/NVIDIA/Cosmos
32. NVIDIA. *NVIDIA Launches Cosmos World Foundation Model Platform to Accelerate Physical AI Development*（CES 发布）. 2025-01. https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-world-foundation-model-platform-to-accelerate-physical-ai-development
33. ByteDance Research. *GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation*. arXiv:2410.06158, 2024. https://arxiv.org/abs/2410.06158 ｜项目页：https://gr2-manipulation.github.io
34. 人民网（转证券日报）. *生数科技发布通用世界模型战略：数字世界 WGM（Vidu）+ 物理世界 WAM（Motus）*. 2026-04-02. http://finance.people.com.cn/n1/2026/0402/c1004-40694098.html ｜Motus 代码：https://github.com/Shengshu-Motus
35. 虎嗅 / APPSO（杜晨）. *五大门派"围攻"大模型*（世界模型五路线：JEPA / 空间智能 / 学习型仿真 / 英伟达基建 / 主动推断；含 AMI Labs、World Labs 融资）. 2026-04-12. https://www.huxiu.com/article/4849809.html
36. 微卷的大白. *「世界模型」浅调研（2026.5）*. 知乎专栏, 2026-05-10. https://zhuanlan.zhihu.com/p/2036554354874962873
37. gurubar. *最新 World Model × Robotics 论文解析——世界模型正在重塑机器人的大脑*. 知乎专栏, 2025-11-11/2026-03 更新. https://zhuanlan.zhihu.com/p/1974055240048345546
38. 仙工智能 SEER. *热点聚焦｜世界模型对决 VLA 大模型，具身智能最终走向何方？*. 2026-05-12. https://seer-robotics.ai/zh/media/316
39. World Labs. *Marble — Generate Editable, Navigable 3D Worlds*. 官方产品页, 2025-11. https://www.worldlabs.ai/
40. 腾讯混元 Tencent Hunyuan. *HunyuanWorld 1.0: Generating Immersive, Explorable, Interactive 3D Worlds*. GitHub, 2025-07-26. https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
41. weijian.sun. *Meta 世界模型 V-JEPA 2：自监督视频模型实现理解、预测与规划*. 知乎专栏, 2025-06-12. https://zhuanlan.zhihu.com/p/1916456899529934478
42. BAAI. *Multimodal Learning with Next-Token Prediction for Large Multimodal Models*（Emu3, Nature）. Nature 650, 327–333, 2026-01-28. https://www.nature.com/articles/s41586-025-10041-x ｜arXiv:2409.18869：https://arxiv.org/abs/2409.18869
43. Raphaël Millière. *Are Video Generation Models World Simulators?*. Artificial Cognition, 2024-03-01. https://artificialcognition.net/posts/video-generation-world-simulators
44. Meta AI. *V-JEPA 2: Advancing Video Understanding Through Self-Supervised Learning*（官方博客）. 2025-06. https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks

---

*本报告为技术调研文档，核心事实经一手信源核实，转述数据（如媒体报道的融资金额、处理速度等）已在正文显式标注信源；智源 Physis 等尚未公开权重的系统，其官方宣称能力未经独立复现，引用时请注意区分。*
