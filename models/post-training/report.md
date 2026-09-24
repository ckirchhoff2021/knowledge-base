# 大模型 Post-Training（后训练）技术调研报告

> 调研来源：知乎社区技术文章（16 篇）+ 其引用的原始论文（arXiv）交叉验证
> 报告日期：2026-09-24
> 适用读者：大模型算法工程师、算法选型决策者、希望系统理解 RLHF/DPO/GRPO 的研究者

## 目录

1. [背景](#一背景)
2. [原理详解](#二原理详解)
3. [方案设计](#三方案设计)
4. [Loss / 算法细节](#四loss--算法细节)
5. [结论与选型建议](#五结论与选型建议)
6. [参考资料](#六参考资料)

---

## 一、背景

### 1.1 问题定义：预训练模型为什么不能直接用

大语言模型（Large Language Model, LLM）的能力来自两个阶段：

- **预训练（Pre-training）**：在数万亿 token 的互联网文本上做下一 token 预测（next-token prediction），目标是压缩并拟合语料分布，模型学到的是"世界知识 + 语言统计规律"。
- **后训练（Post-Training）**：在预训练得到的基座模型（Base Model）上，用**小规模、高质量、带人工设计信号**的数据继续训练，把"文本续写器"雕琢成"能听懂指令、符合人类价值观、会推理、能干活的助手"。

基座模型直接作为助手使用有三个根本缺陷：

1. **目标错位**：预训练优化的是"像不像语料"，而用户要的是"有没有用、对不对、安不安全"。续写目标天然会产生幻觉、答非所问，甚至模仿语料中的有害内容。
2. **不懂指令**：给 Base Model 一个问题，它更可能继续"罗列相似问题"而不是回答问题。
3. **行为不可控**：输出格式、风格、语气、安全边界无法保证。

### 1.2 预训练 vs 后训练对比

| 维度 | 预训练 Pre-training | 后训练 Post-Training |
|------|--------------------|-----------------------|
| 数据规模 | 万亿级 token | 万～百万级条，体量小 3～5 个数量级 |
| 数据质量要求 | 重数量、重多样性，弱过滤 | 质量决定成败，强人工/强筛选 |
| 数据形态 | 原始网页、书籍、代码 | 指令-回答对、偏好排序、规则奖励 |
| 优化目标 | 下一 token 似然 | 指令遵循、偏好对齐、推理、任务成功 |
| 算力成本 | 极大（千卡～万卡月级） | 较小，但在线 RL 阶段依然昂贵 |
| 改变什么 | 知识与通用能力 | 行为方式、价值取向、能力"解锁" |

### 1.3 一个直觉类比

> 预训练像一个**读完了整座图书馆、自学成才的天才少年**——知识渊博，但没人教过他怎么回答问题、怎么与人相处，甚至分不清哪些知识是过时或有害的；
> 后训练则是他的**上岗培训**：SFT 是"照着老师傅的标准答案做学徒"，偏好优化（RLHF/DPO）是"客户给服务打分、按好评改进"，RLVR/GRPO 是"做有标准答案的题、做错就重做"，最终把他训练成一个可靠的专业顾问。

### 1.4 Post-Training 的四大目标

| 目标 | 英文 | 代表技术 |
|------|------|----------|
| 指令遵循 | Instruction Following | SFT（指令微调） |
| 价值与偏好对齐 | Preference / Value Alignment | RLHF、DPO 家族、RLAIF、Constitutional AI |
| 推理能力增强 | Reasoning Enhancement | RFT/STaR、RLVR、GRPO、过程奖励 |
| 领域适配与效率 | Domain Adaptation & Efficiency | 领域 SFT、PEFT（LoRA）、模型融合 |

**核心定位：Post-Training 不是"再学知识"，而是用高信噪比的信号改变模型的行为分布，并用对齐约束防止能力失控。**

> 配套架构图（浏览器打开）：[图 1 · Post-Training 范式流水线](diagrams/1-posttraining-pipeline.html)

---

## 二、原理详解

> 本章给出统一的形式化视角：所有后训练方法都可以表述为"**构造信号 → 定义目标 → 更新策略 π**"，区别只在于信号从哪来、目标怎么写。

### 2.1 形式化定义

设基座模型参数为 θ₀，策略模型（policy）为 π_θ；输入 prompt 为 x，回复序列为 y = (y₁, …, y_T)。

```
模型对回复的对数概率：
log π_θ(y | x) = Σ_t log π_θ(y_t | x, y_<t)

后训练的统一形式：
θ* = argmax_θ  E_{x ~ D}[ 信号质量(x, y; π_θ) ]
                   − β · 偏离约束(π_θ || π_ref)
```

- **信号质量**：SFT 中是"示范答案的似然"；RLHF 中是奖励模型打分 r(x,y)；RLVR 中是规则判定的对错；DPO 中隐含在偏好对里。
- **偏离约束 β·KL**：把新策略拴在参考模型 π_ref（通常是 SFT 模型）附近，防止为了刷分而退化。

### 2.2 监督微调 SFT：模仿示范

SFT（Supervised Fine-Tuning）在"指令 x — 高质量回答 y*"数据上做最大似然：

```
L_SFT(θ) = − E_(x,y*)~D_SFT [ Σ_t log π_θ(y*_t | x, y*_<t) ]
```

机制要点：

1. 只在回答（assistant）token 上计损失，prompt 部分用 mask 屏蔽。
2. 数据配比是关键：任务类型组合、难易样本比例、长短回答混合、推理过程展示程度都影响效果。
3. **少量高质量数据常优于大量普通数据**；合成 SFT 数据成本低，但会带来错误重复、风格单一、多样性下降。
4. SFT 只通过"单个 token"给信号，对模型完整生成能力的估计有偏（见 2.6）。

### 2.3 RLHF：基于人类反馈的强化学习

RLHF（Reinforcement Learning from Human Feedback）由 InstructGPT（Ouyang et al., 2022）确立为标准三阶段：

1. **SFT**：人工撰写示范答案，微调基座，得到会回答问题的初始策略。
2. **奖励模型 RM（Reward Model）训练**：对同一 prompt 采样多个回复，标注员给出排序；用 Bradley–Terry 模型把"人类偏好"学成分数函数 r_φ(x,y)。
3. **PPO 强化学习**：把 RM 当作环境奖励，用 PPO（Proximal Policy Optimization）更新策略，同时加 KL 惩罚约束。

奖励模型的 Bradley–Terry 建模：

```
人类认为 y_w 优于 y_l 的概率：
P(y_w ≻ y_l | x) = σ( r_φ(x, y_w) − r_φ(x, y_l) )

RM 损失：
L_RM(φ) = − E_(x,y_w,y_l) [ log σ( r_φ(x,y_w) − r_φ(x,y_l) ) ]
```

PPO 阶段目标（直觉形式）：

```
目标 = E[ r_φ(x, y) ] − β · KL( π_θ(·|x) || π_ref(·|x) )

PPO clipped objective（裁剪重要性采样比，限制单步更新幅度）：
r_t(θ) = π_θ(y_t|·) / π_old(y_t|·)
L_clip = E_t[ min( r_t · A_t,  clip(r_t, 1−ε, 1+ε) · A_t ) ]
```

其中 A_t 是优势函数（advantage），由 Critic（价值网络）估计；KL 项防止模型为追求高分生成无意义内容。

**RLHF 的工程代价**：PPO 全流程需要同时驻留 4 个大模型——Actor（训练）、Reference（冻结）、Critic（训练）、Reward Model（冻结），显存与工程复杂度极高，且训练不稳定、对超参敏感。

### 2.4 奖励的两种粒度：ORM 与 PRM

| 类型 | 英文 | 信号粒度 | 优点 | 问题 |
|------|------|----------|------|------|
| 结果奖励模型 | ORM, Outcome Reward Model | 只给最终结果打一个分 | 标注简单、训练稳定 | 反馈稀疏、方差大、无法定位错误步骤 |
| 过程奖励模型 | PRM, Process Reward Model | 给每个推理步骤打分 | 能定位错误、引导多步推理 | 步骤边界与正确性难标注，易被 reward hacking |

DeepSeek-R1 报告把 PRM 列入"未成功尝试"：通用推理中细粒度步骤难定义、中间步骤正确性难判定、模型型 PRM 在大规模 RL 中可能被攻击；但 PRM 在 **top-N 重排序 / 引导式搜索**中表现不错。这是 R1 的工程经验，**不能改写成"PRM 已失效"**——强 selector 不等于抗攻击的 reward。

### 2.5 DPO：去掉奖励模型与强化学习

DPO（Direct Preference Optimization，Rafailov et al., 2023）的核心洞见：**RLHF 中最优策略与奖励有解析关系，可以直接用偏好对训练策略，无需训练 RM、无需在线采样**。

推导思路（三步）：

1. KL 约束下的奖励最大化有闭式最优解：
```
π*(y|x) ∝ π_ref(y|x) · exp( r(x,y) / β )
```
2. 反解出奖励（配分函数 Z(x) 与 y 无关）：
```
r(x,y) = β · log( π_θ(y|x) / π_ref(y|x) ) + β · log Z(x)
```
3. 把它代回 Bradley–Terry 损失，Z(x) 项在差分时消去，得到 DPO 损失：
```
L_DPO(θ) = − E_(x,y_w,y_l) [
  log σ( β · log π_θ(y_w|x)/π_ref(y_w|x)
         − β · log π_θ(y_l|x)/π_ref(y_l|x) )
]
```

机制要点：

1. 只需：偏好数据（chosen/rejected）+ 冻结的参考模型，无需 RM、Critic、在线 RL。
2. 本质上是在拉开" winning 回复相对参考模型的对数概率增幅"与"losing 回复增幅"的差距。
3. **离线 DPO（Offline DPO）**是开源主流：偏好数据事先采集；但训练中策略会逐渐偏离采样时的分布，损失会错误估计模型当前能力，导致效果打折——这就是"离线偏好优化的分布漂移"。
4. **在线 DPO（Online DPO）**实时采样、实时标注偏好，理论形态最佳，但数据标注开销甚至高于 RLHF。

### 2.6 为什么 SFT / DPO 会"估计有偏"

用一个统一视角看：SFT、RLHF、DPO 都是"先估计模型自身的偏好分布，再与人类偏好对齐"，但估计粒度不同：

- SFT 只根据模型生成的**一个 token**做纠正（像"每写一步就立刻对答案"），暴露不出完整思路的漏洞。
- RLHF / DPO 基于**完整回复**给信号（像"先整卷做完再对答案"），估计更准，但成本更高。
- 刷题类比：先独立写完再对答案，能通过试错暴露思维漏洞；边写边对答案则收获更少。

### 2.7 RLAIF 与 Constitutional AI：用 AI 反馈替代人工

- **RLAIF**（Reinforcement Learning from AI Feedback）：用强模型代替人类标注偏好/打分，降低标注成本、提升规模。
- **Constitutional AI（宪法 AI，Bai et al., 2022）**：先给模型一套成文原则（"宪法"，如"不得歧视、不得协助违法"），让模型自我批评并改写有害回复（SL 阶段），再用基于宪法的 AI 偏好训练奖励模型、做 RL。
- 价值：把"人工逐条标注"压缩为"写原则 + AI 规模化执行"。局限：评审模型可能奖励"表达姿态"而非真实性，原则覆盖不到所有分布。

### 2.8 RLVR 与 GRPO：可验证奖励时代

在数学、代码、逻辑等**答案可被规则自动判定对错**的任务上，出现了 RLVR（Reinforcement Learning with Verifiable Rewards）：

- 奖励直接来自判题器（答案匹配、单元测试、形式化校验），不需要训练奖励模型，从源头避免奖励模型失真。
- 奖励类型演进：① 二元正确性奖励（对/错，简单但稀疏高方差）；② 逐步准确性奖励（给中间步骤渐进反馈）；③ 自一致性奖励（多条推理路径共识）；④ 基于偏好的奖励（开放式复杂任务）。

**GRPO（Group Relative Policy Optimization，组相对策略优化）** 由 DeepSeekMath（Shao et al., 2024）提出，后成为 DeepSeek-R1-Zero/R1 的核心算法。核心洞见：

> Baseline（用于降方差的参照值）**不必用神经网络（Critic）去预测，可以用同一问题一组采样的平均奖励做统计估计**。

机制步骤：

1. **组采样**：对问题 q，用当前策略生成 G 个输出 {o₁,…,o_G}（DeepSeekMath 中 G=64）。
2. **打分**：用奖励模型或规则判题器得到 {r₁,…,r_G}。
3. **组内标准化得优势**：A_i = (r_i − mean(r)) / std(r)，天然消除 Critic。
4. **目标函数**：对重要性采样比做裁剪，并额外加一个 token 级 KL 正则（把参考模型也省进同一个目标里）。

由此显存中只需 Actor + Reference（+奖励打分可离线），**抛弃 Critic 是 GRPO 相对 PPO 的最大效率优势**。

### 2.9 推理模型的两条路线：R1-Zero 与 R1

- **DeepSeek-R1-Zero**：不做推理 SFT（冷启动），直接从基座模型上用 GRPO + 可验证奖励大规模 RL，模型自发涌现出"反思、验证"等长链思维。问题是可读性差、语言混杂。
- **DeepSeek-R1**：先用少量高质量长 CoT 数据做**冷启动 SFT**（Cold Start）让轨迹可读、可筛，再做大规模 RL；并引入推理能力蒸馏（把 R1 的思维链蒸馏进小模型）。
- 路线启示：可验证任务上"先 RL 再说"可行，但冷启动能显著改善可读性与稳定性；蒸馏是让推理能力下沉到小模型的关键手段。

> 配套架构图（浏览器打开）：[图 2 · RLHF vs DPO vs GRPO 对比](diagrams/2-rlhf-dpo-grpo-comparison.html)

---

## 三、方案设计

> 本章给出方法谱系、变体横向对比和关键实验数据，供选型直接使用。

### 3.1 方法演进谱系（2017–2025）

| 时间 | 方法 / 工作 | 关键贡献 |
|------|-------------|----------|
| 2017 | RLHF（Christiano et al.） | 首次用人类偏好训练奖励模型做 RL |
| 2022.03 | InstructGPT（PPO 三阶段） | SFT→RM→PPO 成为工业标准 |
| 2022.03 | STaR | 采样—过滤正确 CoT—迭代微调，rationalization 回补 |
| 2022.12 | Constitutional AI / RLAIF | 成文原则 + AI 反馈规模化对齐 |
| 2023.05 | DPO | 去 RM、去在线 RL 的直接偏好优化 |
| 2023.10 | IPO | 缓解 DPO 过拟合与对 β 敏感 |
| 2024.01 | Self-Rewarding | 模型自身充当奖励模型，迭代自对齐 |
| 2024.02 | KTO / DeepSeekMath(GRPO) | 无需成对数据；去 Critic 的组相对优化 |
| 2024.03 | ORPO | SFT 与偏好优化合一，无需参考模型与单独 SFT |
| 2024.05 | SimPO | 长度归一化、彻底去参考模型 |
| 2025.01 | DeepSeek-R1-Zero / R1 | 基座直 RL、冷启动 + RLVR + 蒸馏的推理范式 |

### 3.2 主流后训练方法横向对比

| 方法 | 监督信号 | 需要 RM | 需要 Critic | 需要参考模型 | 需要成对偏好数据 | 成本 | 稳定性 | 典型场景 |
|------|----------|:---:|:---:|:---:|:---:|------|--------|----------|
| SFT | 示范答案 | 否 | 否 | 否 | 否 | 低 | 高 | 指令遵循、领域适配、冷启动 |
| RFT / STaR | 自采样 + 答案正确性过滤 | 否 | 否 | 否 | 否 | 中 | 中高 | 数学/代码推理增强 |
| RLHF (PPO) | RM 打分 | 是 | 是 | 是 | 训练 RM 时需要 | 高 | 较低 | 通用对话、开放式对齐 |
| DPO | 成对偏好 | 否 | 否 | 是 | 是 | 中低 | 高 | 偏好对齐、低成本替代 RLHF |
| GRPO (RLVR) | 规则/奖励 + 组相对优势 | 否（规则） | 否 | 是（KL 项） | 否 | 中 | 中高 | 可验证推理、训练推理模型 |
| RLAIF / CAI | AI 偏好/原则 | AI 充当 | 视算法 | 视算法 | AI 生成 | 中 | 中 | 规模化安全对齐 |

### 3.3 DPO 家族变体对比

| 算法 | 全称 | 需要 ref_model | 需要成对数据 | 需要先 SFT | 主要解决的问题 |
|------|------|:---:|:---:|:---:|------|
| DPO | Direct Preference Optimization | 是 | 是 | 是 | 基础版，替代 RLHF |
| IPO | Implicit Preference Optimization | 是 | 是 | 是 | 过拟合、对 β 敏感 |
| KTO | Kahneman–Tversky Optimization | 是 | 否（好/坏二元反馈即可） | 是 | 成对偏好数据难获取 |
| ORPO | Odds Ratio Preference Optimization | 否 | 是 | 否（SFT 已内建） | 流程冗长，想合并阶段 |
| SimPO | Simple Preference Optimization | 否（长度归一化边际奖励） | 是 | 是 | 依赖参考模型、长度偏差 |

### 3.4 关键实验数据（均来自原始论文，知乎转述项已标注）

| 工作 | 关键结果 | 来源 |
|------|----------|------|
| InstructGPT | 标注员对 **1.3B PPO-ptx 输出的偏好度超过 175B GPT-3**；小模型 + 对齐可反超更大基座 | Ouyang et al., 2022（原始论文） |
| DPO | 在情感控制、摘要、对话无害性等任务上，DPO **持平或优于 PPO**，且实现更简单、训练更稳 | Rafailov et al., 2023（原始论文） |
| DeepSeekMath GRPO | 在数学推理上以更低资源取得优于 PPO 等方法的效果（GSM8K 等基准），组大小 G=64 | Shao et al., 2024（原始论文） |
| DeepSeek-R1 | R1 在 AIME 2024 pass@1 ≈ **79.8%**、MATH-500 ≈ **97.3%**；R1 蒸馏小模型在多项基准上超越同规模模型 | DeepSeek-R1 技术报告, 2025；**具体数字来自论文与知乎转述，引用前请按原报告二次核对** |
| RM 评估（ICLR'25） | 每 prompt 收集 3–4 个回复性价比最高；ξ 相关系数在 BoN/PPO 下分别约 0.677 / 0.688 | 知乎《重新思考 Reward Model 评估》转述，建议核对原文 |

### 3.5 数据体系：三类后训练数据

| 类型 | 来源 | 用途与风险 |
|------|------|-----------|
| 人工标注数据 | 标注员写示范、排偏好 | 质量最高、成本大；覆盖有限 |
| 蒸馏数据 | 教师模型（如 R1）生成 CoT/回答 | 能力下沉快；受教师风格与错误约束 |
| 合成数据 | Self-Instruct、反向翻译（backtranslation）、STaR 自迭代 | 可规模化；易多样性塌缩、错误自我强化 |

---

## 四、Loss / 算法细节

> 本章汇总核心 Loss 与可直接落地的 GRPO 伪代码。

### 4.1 核心 Loss 一览

```
# 1) SFT（仅 response token）
L_SFT = − E[ Σ_t m_t · log π_θ(y_t | x, y_<t) ]

# 2) Reward Model（Bradley–Terry 成对排序）
L_RM  = − E[ log σ( r(x, y_w) − r(x, y_l) ) ]

# 3) PPO（最大化奖励 − KL，裁剪重要性比）
J_PPO = E[ r(x,y) ] − β · KL(π_θ || π_ref)
r_t(θ) = π_θ / π_old
L_clip = E[ min(r_t A_t, clip(r_t, 1−ε, 1+ε) A_t) ]

# 4) DPO（隐式奖励，无 RM）
L_DPO = − E[ log σ( β·(log π_θ(y_w)/π_ref(y_w)
                      − log π_θ(y_l)/π_ref(y_l)) ) ]

# 5) GRPO（组内相对优势 + token 级 KL）
A_i    = (r_i − mean({r_j})) / (std({r_j}) + 1e-6)
L_GRPO = − E[ 1/G Σ_i ( 1/|o_i| Σ_t min( r_it A_i, clip(r_it,1−ε,1+ε) A_i )
                        − β · ( log π_θ(o_it)/π_ref(o_it)
                                − log π_ref(o_it)/π_θ(o_it) ) ) ]
```

### 4.2 GRPO 伪代码（带注释，Python 风格）

```python
def grpo_step(prompts, policy, ref_model, reward_fn,
              G=8, beta=0.04, eps=0.2):
    rollouts = []
    for q in prompts:
        # ① 组采样：同一问题用当前策略生成 G 条回复
        outputs = [policy.generate(q) for _ in range(G)]

        # ② 打分：规则判题器 / 奖励模型给出标量奖励
        rewards = [reward_fn(q, o) for o in outputs]

        # ③ 组内标准化：用统计 baseline 替代 Critic
        mean = sum(rewards) / G
        std  = (sum((r - mean) ** 2 for r in rewards) / G) ** 0.5
        advantages = [(r - mean) / (std + 1e-6) for r in rewards]
        rollouts.append((outputs, advantages))

    loss = 0.0
    for outputs, advantages in rollouts:
        for o, adv in zip(outputs, advantages):
            for token in o.tokens:
                # ④ 重要性采样比 + PPO 式裁剪
                ratio = policy.prob(token) / o.old_prob(token)
                surr1 = ratio * adv
                surr2 = clip(ratio, 1 - eps, 1 + eps) * adv
                pg    = min(surr1, surr2)

                # ⑤ 无偏 KL 近似（Schulman 蒙特卡洛形式，恒非负）
                log_ratio = log(policy.prob(token) / ref_model.prob(token))
                kl = exp(log_ratio) - 1 - log_ratio

                loss += -(pg - beta * kl)

    loss.backward()   # 只更新 Actor；没有 Critic，显著省显存
    optimizer.step()
```

### 4.3 DPO 伪代码（最简形式）

```python
def dpo_loss(policy, ref, batch, beta=0.1):
    # batch 含同一 prompt 的 chosen / rejected
    pi_logratios  = policy.logp(batch.chosen)   - policy.logp(batch.rejected)
    ref_logratios = ref.logp(batch.chosen)     - ref.logp(batch.rejected)
    logits = beta * (pi_logratios - ref_logratios)
    return -logsigmoid(logits).mean()
```

> 配套架构图（浏览器打开）：[图 3 · GRPO 流程与病理](diagrams/3-grpo-flow.html)

---

## 五、结论与选型建议

### 5.1 核心结论

1. **SFT 是一切后训练的地基**：再强的 RL 也建立在"会基本回答/解题"的策略上；数据质量与配比决定上限。
2. **RLHF(PPO) 效果好但贵且不稳**：适合资源充足、开放式对话场景；其"四模型驻留 + Critic"是主要瓶颈。
3. **DPO 家族是性价比路线**：去 RM、去在线 RL，工程友好；但离线分布漂移真实存在，迭代式/在线 DPO 才能逼近 RLHF 上限。
4. **可验证任务首选 RLVR + GRPO**：规则奖励 + 去 Critic，是 2025 年以来推理模型（R1/QwQ 类）的制胜公式。
5. **冷启动 SFT 解决可读性，蒸馏解决能力下沉**：纯基座直 RL（R1-Zero）可行但产物粗糙。
6. **对齐有代价**：KL/β 过大约束能力、过小导致 reward hacking；PRM、自奖励等手段仍受攻击与标注瓶颈约束。

### 5.2 工程落地 Checklist

- ☐ 先做高质量 SFT，冻结基线；只在 response token 计 loss
- ☐ 建立离线评测集：通用能力 + 任务指标 + 安全集，防止"对齐税"悄悄掉点
- ☐ 偏好/奖励数据覆盖多场景、多难度；允许 Tie（平局）标签
- ☐ 奖励模型与策略模型保持**同一谱系（same lineage）**
- ☐ RL 阶段加 KL 约束与裁剪；监控 reward 与 KL 曲线，警惕 reward hacking
- ☐ 算力有限：用 SFT + DPO（或 ORPO/SimPO 合并阶段）起步
- ☐ 数学/代码任务：上 RLVR + GRPO，组大小按显存从 G=8 起调
- ☐ 推理模型：冷启动 SFT → GRPO/RLVR → 蒸馏到小模型
- ☐ 合成/蒸馏数据做去重、多样性与正确性过滤，避免错误自我强化

### 5.3 注意事项与诚实性声明

- 本报告主要信源为知乎社区文章与其引用的原始论文；**方法原理与 arXiv 编号均按检索结果交叉核对**。
- 第 3.4 节中 R1 的具体跑分（AIME 79.8%、MATH-500 97.3%）与 RM ξ 相关系数（0.677/0.688）**部分来自知乎转述**，写入正式材料前请按 DeepSeek-R1 原报告与 ICLR 2025 原文二次核对。
- 术语消歧：RFT 在不同资料中可指"Rejection sampling Fine-Tuning（拒绝采样微调）"或"Reinforcement Learning Fine-Tuning"；本报告采用前者。
- PRM 的"未成功尝试"仅代表 DeepSeek-R1 项目的工程结论，不代表 PRM 在重排序/引导搜索场景无效。

---

## 六、参考资料

### 6.1 原始出处 / 工业报告

1. OpenAI, InstructGPT 技术博客与论文《Training language models to follow instructions with human feedback》
2. DeepSeek-AI, 《DeepSeekMath》《DeepSeek-R1》技术报告
3. Anthropic, 《Constitutional AI: Harmlessness from AI Feedback》

### 6.2 研究论文（arXiv）

1. Christiano et al., *Deep Reinforcement Learning from Human Preferences*, arXiv:1706.03741
2. Ouyang et al., *Training language models to follow instructions with human feedback (InstructGPT)*, arXiv:2203.02155
3. Zelikman et al., *STaR: Bootstrapping Reasoning With Reasoning*, arXiv:2203.14465
4. Bai et al., *Constitutional AI*, arXiv:2212.08073
5. Rafailov et al., *Direct Preference Optimization (DPO)*, arXiv:2305.18290
6. Azar et al., *Implicit Preference Optimization (IPO)*, arXiv:2310.12036
7. Yuan et al., *Self-Rewarding Language Models*, arXiv:2401.10020
8. Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*, arXiv:2402.01306
9. Shao et al., *DeepSeekMath（含 GRPO）*, arXiv:2402.03300
10. Hong & Lee, *ORPO: Monolithic Preference Optimization without Reference Model*, arXiv:2403.07691
11. Meng et al., *SimPO: Simple Preference Optimization with a Reference-Free Reward*, arXiv:2405.14734
12. DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL*, arXiv:2501.12948

### 6.3 知乎主要参考文章

1. 边路腰刀《SFT、RLHF、DPO、IFT —— LLM 微调的进化之路》— https://zhuanlan.zhihu.com/p/710652762
2. 吕阿华《【LLM行业综述】LLM后训练技术综述（全文）》— https://zhuanlan.zhihu.com/p/30880313862
3. 琦琦《大语言模型后训练技术深度综述 (2023-2026)》— https://zhuanlan.zhihu.com/p/2003230939191457464
4. 想飞的石头《大模型 Post-Training 综述学习笔记》— https://zhuanlan.zhihu.com/p/30209020836
5. xyiz《大模型后训练与强化学习（七）：GRPO》— https://zhuanlan.zhihu.com/p/2004694228433916589
6. wuxiaojun《强化学习小白理解 GRPO（二）：GRPO 核心代码实践》— https://zhuanlan.zhihu.com/p/23349133287
7. 引线小白《学习 DeepSeek R1：一文读懂大语言模型中的强化学习（SFT、RFT、DPO、PPO、GRPO）》— https://zhuanlan.zhihu.com/p/21178712267
8. John X《DPO 系列偏好优化算法总结》— https://zhuanlan.zhihu.com/p/2066125342696268489
9. Kane《LLM 三阶段 Loss 推导：Pretraining、SFT 与 RL 到底在优化什么》— https://zhuanlan.zhihu.com/p/2077497291070620478
10. Ghost apple《从 STaR 到 DeepSeek-R1：推理能力如何被监督、筛选与蒸馏？》— https://zhuanlan.zhihu.com/p/2077362775622534042
11. JayJay《ICLR 2025 Spotlight｜重新思考 Reward Model 评估》— https://zhuanlan.zhihu.com/p/1899485712614655042
12. 王小二《强化学习与生成式 AI 的统一：从数学基础到算法演进》— https://zhuanlan.zhihu.com/p/2027758469663527764
13. 白露未晞me《锐评大模型各大研究方向》— https://www.zhihu.com/question/1958561146647876509/answer/2075524312644195693
14. sia《大语言模型之数据工程》— https://zhuanlan.zhihu.com/p/2083221865833698561
15. Bill《强化学习中有哪些重要的理论结果？》— https://www.zhihu.com/question/312164724/answer/2050852950613009683
16. 明赫佐希《如果你是大模型算法的面试官，你会问哪些问题？》— https://www.zhihu.com/question/11013588902/answer/2016558503343448756
