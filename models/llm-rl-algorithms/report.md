# 大模型常用强化学习算法技术调研报告

> 从策略梯度（Policy Gradient）到 PPO，再到 GRPO 及其变体 DAPO、Dr.GRPO、GSPO：用一篇报告讲透大模型强化学习的数学原理、算法演进、奖励范式与工程实践。
>
> 📅 2026-09-30 ｜ 信源：13 篇 arXiv 原始论文（编号经 arXiv API 逐一核实）+ 知乎高赞技术文章交叉验证

---

## 目录

1. [背景：为什么大模型需要强化学习](#1-背景为什么大模型需要强化学习)
2. [强化学习基础：把语言生成建模为 MDP](#2-强化学习基础把语言生成建模为-mdp)
3. [PPO：大模型 RL 的经典基线](#3-ppo大模型-rl-的经典基线)
4. [Critic-Free 策略梯度家族：REINFORCE → RLOO → ReMax → REINFORCE++](#4-critic-free-策略梯度家族reinforce--rlo--remax--reinforce)
5. [GRPO：当代大模型 RL 的主角](#5-grpo当代大模型-rl-的主角)
6. [GRPO 的改进变体：DAPO、Dr.GRPO、GSPO](#6-grpo-的改进变体dapodrgrpogspo)
7. [奖励范式：RLHF、RLAIF、RLVR 与 R1 训练流程](#7-奖励范式rlhfrlaifrlvr-与-r1-训练流程)
8. [核心实现伪代码](#8-核心实现伪代码)
9. [结论与选型建议](#9-结论与选型建议)
10. [参考资料](#10-参考资料)

---

## 1. 背景：为什么大模型需要强化学习

**导读：** 监督微调（SFT）解决"会照做"，但解决不了"做得好不好"和"自己探索更优解法"。强化学习（Reinforcement Learning, RL）在大模型生命周期中承担了两个不可替代的角色：对齐人类偏好、激发推理能力。

### 1.1 SFT 的三个天花板

大模型先经过预训练（语言建模）和监督微调（SFT，在高质量示范数据上学习），但 SFT 存在结构性局限：

1. **只能模仿示范，无法超越示范。** SFT 最小化与标注答案的交叉熵，老师多好、学生至多一样。
2. **高质量示范成本极高。** 复杂推理任务（数学竞赛、代码）的标注轨迹稀缺且昂贵。
3. **"唯一正确答案"式学习无法刻画偏好。** 很多问题没有标准答案——回答的帮助性、无害性、风格好坏是相对的，需要在多个候选之间比较。

### 1.2 RL 的两个不可替代角色

| 角色 | 名称 | 奖励来源 | 代表事件 |
|---|---|---|---|
| **对齐人类偏好** | RLHF（Reinforcement Learning from Human Feedback） | 奖励模型（Reward Model, RM）对人类偏好建模 | 2022 InstructGPT / ChatGPT（arXiv:2203.02155） |
| **激发推理能力** | RLVR（Reinforcement Learning with Verifiable Rewards） | 规则/程序自动验证答案对错（0/1） | 2025 DeepSeek-R1（arXiv:2501.12948） |

RL 的本质区别在于：**模型不再被告知"标准答案是什么"，而是自己生成多个候选、只拿到一个"好不好"的分数，然后通过试错调整生成策略。** 这让模型有机会探索出训练数据里没有的解法——R1 报告中观察到自我反思、验证、动态策略切换等推理行为的"涌现"。

### 1.3 本文与"非 RL 偏好优化"的边界

DPO（Direct Preference Optimization，arXiv:2305.18290）等方法用一个离线损失直接拟合偏好数据、无需采样和训练循环，常被拿来与 RL 对比。严格意义上它们**不是强化学习**（没有环境交互、没有奖励、没有策略提升）。本文聚焦真正的 RL 算法，仅在对比章节简要讨论 DPO。

---

## 2. 强化学习基础：把语言生成建模为 MDP

**导读：** 先建立"生成一个 response = 走一条轨迹"的对应关系，再推导一切大模型 RL 算法的共同祖先——策略梯度，并解释 baseline 为什么是必需的。

### 2.1 语言生成对应的 MDP

经典 RL 用马尔可夫决策过程（Markov Decision Process, MDP）建模。在 LLM RL 中：

```
状态 s_t：prompt + 已生成的前 t-1 个 token
动作 a_t：第 t 个 token（动作空间 = 词表 V，通常数万个）
策略 πθ(a|s)：模型给出的下一个 token 概率分布（即模型本身）
奖励 r_t：通常只在 response 结束时有一个总分（sparse reward）
轨迹 τ ：(s_1, a_1, s_2, a_2, ..., s_T, a_T)，即一次完整生成
```

两个 LLM 场景的特殊性值得注意：

1. **奖励极度稀疏。** 奖励模型通常只对完整回答给一个总分；RLVR 只有最终答案对/错（1/0）。中间每个 token 没有即时奖励。
2. **状态转移确定。** 在 LLM 中 s_{t+1} 就是把 a_t 拼到上下文后得到的，没有环境随机性；全部随机性来自策略采样。

### 2.2 策略梯度定理：所有算法的共同祖先

RL 的目标是最大化期望回报 `J(θ) = E_{τ~πθ}[R(τ)]`。**策略梯度定理**给出其梯度：

```
∇θ J(θ) = E_{τ~πθ} [ Σ_t R(τ) · ∇θ log πθ(a_t | s_t) ]
```

蒙特卡洛估计（REINFORCE 的基础形式）：

```
∇θ J ≈ (1/N) Σ_n Σ_t R(τ_n) · ∇θ log πθ(a_t^n | s_t^n)
```

含义直观：采样 N 条回答，**得分越高的回答，其中每个 token 的概率越被提高；得分为负则压低。**

### 2.3 为什么需要 Baseline（一个关键例子）

知乎技术文章中常用一个例子暴露朴素策略梯度的问题。考虑 prompt「把 LLM和强化学习 中的 和 改成 -」，模型采样出 3 个回答，奖励分别为：

```
Response1（啰嗦且错误 LLM--强化学习）：R1 = 1
Response2（啰嗦但正确）：              R2 = 2
Response3（简洁正确 LLM-强化学习）：   R3 = 3
```

三个奖励**全是正数**，于是朴素策略梯度会提高**所有**回答里所有 token 的概率——包括错误回答和啰嗦废话，只是提高幅度不同。这显然不合理：我们真正想学的是"相对而言哪个更好"。

解决办法是引入 **baseline（基线）b**，用 R − b 替代 R：

```
∇θ J = E [ Σ_t (R(τ) - b) · ∇ log πθ(a_t|s_t) ]
```

只要 b 不依赖动作，梯度仍然无偏。取三个奖励的均值 b = 2，则相对优势变为：

```
A1 = -1  → 压低错误回答的 token 概率
A2 =  0  → 不动
A3 = +1  → 提高最优回答
```

**baseline 的选择是后续一切算法分野的源头：**

- PPO：用一个学习出来的价值网络（Critic）当 baseline；
- RLOO：用"其他采样回答的平均奖励"当 baseline；
- GRPO：用同 prompt 多条采样的组内均值当 baseline；
- ReMax：用贪心（greedy）回答的奖励当 baseline。

### 2.4 优势函数与广义优势估计（GAE）

更精细的做法不是把总分平摊给每个 token，而是估计每个 token 的**优势（Advantage）** A_t——在 s_t 采取 a_t 比平均水平好多少。为此定义：

```
回报（return）：G_t = Σ_{t'≥t} γ^(t'-t) · r_{t'}
价值函数：V(s) = E_{τ~π}[G_t | s_t = s]        # Critic 要学的东西
时序差分残差（TD residual）：
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)

广义优势估计（GAE, λ-return）：
A_t^GAE = Σ_{l≥0} (γλ)^l · δ_{t+l}
```

- γ（gamma）是折扣因子；λ（lambda）在偏差与方差之间调节：λ=0 时 A_t=δ_t（低方差、有偏），λ=1 时是蒙特卡洛回报（无偏、高方差）。
- LLM RL 中由于奖励稀疏且多在句末，γ 常设为 1（不折扣）。

---

## 3. PPO：大模型 RL 的经典基线

**导读：** PPO（Proximal Policy Optimization，arXiv:1707.06347）是 OpenAI 在 InstructGPT（2022，arXiv:2203.02155）中用于 RLHF 的算法，长期是大模型 RL 的事实标准。本章讲清它的四个模型、clip 机制和完整损失。

### 3.1 四个模型

一次完整的 PPO 训练涉及四个模型，显存开销巨大：

| 模型 | 是否训练 | 作用 |
|---|---|---|
**Policy（Actor，策略模型）** | ✅ 训练 | 即被优化的 LLM，生成回答 |
**Reference（参考模型）** | ❄️ 冻结 | 通常是 SFT 模型，用 KL 散度约束 Policy 不要偏离太远 |
**Reward Model（奖励模型）** | ❄️ 冻结（本阶段） | 对回答打分 |
**Critic（价值模型）** | ✅ 训练 | 估计每个 token 的状态价值 V(s)，提供 baseline |

Critic 通常与 Policy 同规模——这意味着 PPO 训练至少要承担两个同尺寸模型的训练状态（梯度、优化器状态），显存和算力成本高昂。

### 3.2 重要性采样比率与 Clip 机制

RL 数据是 on-policy 的（必须由当前策略采样），但一批数据要做多个 epoch 的梯度更新，策略会变。为合法复用旧数据，引入重要性采样比率：

```
ratio_t(θ) = πθ(a_t|s_t) / πθ_old(a_t|s_t)
```

朴素目标 ratio·A 可能导致一次更新过猛。PPO 的核心是**截断（clip）目标**：

```
L_PPO = E_t [ min( ratio_t · A_t,
                  clip(ratio_t, 1-ε, 1+ε) · A_t ) ]
```

ε 通常取 0.1–0.2。机制解读：

- 当 A_t > 0（动作好）：目标随 ratio 上升，但超过 1+ε 后被截断——鼓励多生成好动作，但别一步到位；
- 当 A_t < 0（动作差）：目标随 ratio 下降而降低，但 ratio 低于 1−ε 后被截断——压低坏动作即可，不必一次压死。

**clip 是 PPO 训练稳定的关键：它把每次策略更新限制在旧策略附近一个"信任域"内。**

### 3.3 Critic 损失（Value Clip）

Critic 用 GAE 算出的回报作为回归目标：

```
L_VF = E_t [ ( Vθ(s_t) - V_targ )^2 ]
V_targ = A_t^GAE + Vθ_old(s_t)     # 即用 GAE 回报作为价值标签
```

工程实现中价值损失通常也做 clip，防止 Critic 更新过猛进而污染下一轮优势估计：

```
V_clipped = V_old + clip( Vθ(s_t) - V_old, -ε_v, ε_v )
L_VF = 0.5 · max( (Vθ - V_targ)^2, (V_clipped - V_targ)^2 )
```

在 LLM PPO 中通常只对 response token 计算 Critic 损失，不计算 prompt 和 padding token。

### 3.4 KL 约束、熵奖励与总损失

为防止策略钻奖励模型的漏洞、变得面目全非，加入对参考模型的 KL 惩罚：

```
InstructGPT 的做法：把 KL 惩罚直接加在每个 token 的奖励里
r_t' = r_t - β · log( πθ(a_t|s_t) / π_ref(a_t|s_t) )

PPO 总目标：
maximize   L_PPO - c1 · L_VF + c2 · E[ H(πθ(·|s_t)) ]
（另含 KL 项，各实现写法略有差异）
```

- β 是 KL 系数，控制"向高奖励靠拢"与"别偏离 SFT 模型"之间的平衡；
- 熵项 H 鼓励策略保持一定探索性，避免过早确定化；
- LLM PPO 中 β 通常远小于经典 RL 设定，KL 多用 token 级估计。

### 3.5 PPO 的训练循环

```
① 从数据集取 prompt，用当前 Policy 自回归采样 response（rollout/generation）
② Reward Model 对完整回答打分；KL 惩罚按 token 计入
③ Critic 给出各 token 价值，用 GAE 计算每个 token 的优势 A_t
④ 用这批 rollout 数据做多个 epoch 更新：
   Actor 最大化 clip 目标；Critic 最小化 value loss
⑤ 回到 ① 循环
```

### 3.6 PPO 的痛点

1. **Critic 成本高：** 与 Policy 同规模，显存/算力几乎翻倍，训练系统复杂。
2. **稀疏奖励下 Critic 难学：** LLM 中往往只有最后一个 token 有奖励，却要求 Critic 准确给出每个中间 token 的价值，信号微弱、易不准。
3. **超参敏感：** clip 系数、KL 系数、GAE 的 λ、学习率、rollout 长度相互耦合，"炼金术"门槛高。
4. **在线采样吞吐受限：** rollout 生成（decode）是 RL 训练速度的常见瓶颈。

---

## 4. Critic-Free 策略梯度家族：REINFORCE → RLO → ReMax → REINFORCE++

**导读：** 为了砍掉昂贵且难训的 Critic，研究者用"采样回答自身的奖励统计"构造 baseline。本章按时间顺序介绍这一家族，它们是通向 GRPO 的前奏。

### 4.1 REINFORCE 与 REINFORCE++

**REINFORCE** 即第 2.2 节的蒙特卡洛策略梯度，最简单但方差大、不稳定。

**REINFORCE++**（arXiv:2501.03262，《Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization》）把 PPO 验证过的工程技巧移植回无 Critic 的 REINFORCE：

1. 保留 PPO 式 **token 级 clip**（重要性比率裁剪），保证更新稳定；
2. **全局优势归一化**——把一个训练 batch 内所有 prompt 的优势放在一起做标准化（减全局均值、除全局标准差），缓解每个 prompt 采样数太少导致的优势估计不稳；
3. KL 正则与归一化技巧整合进目标函数。

社区实践反馈（知乎·初七9527）：REINFORCE++-baseline（减组均值 + 全局标准差归一化）在长 CoT 推理任务上稳定性好；纯 REINFORCE++ 更适合 −1/1 这种对称奖励。**定位是"比 GRPO 稳定、比 PPO 快"**，因为它不需要 Critic，同时避免了 GRPO 在小 group 下组内 std 估计噪声大的问题。

### 4.2 RLOO：Leave-One-Out 基线

RLOO（REINFORCE Leave-One-Out，系统研究见《Back to Basics》，arXiv:2402.14740）对每个 prompt 采样 N 条回答，**第 i 条回答的 baseline 是"除它之外其他 N−1 条回答的平均奖励"**：

```
A_i = R_i - (1/(N-1)) · Σ_{k≠i} R_k
```

"留一"设计的好处：baseline 与被评价的回答不相关（数学上比包含自身的均值更干净），且被证明是 REINFORCE 风格方法的强方差缩减手段。《Back to Basics》论文的核心结论是：**只要调参得当，简单的 REINFORCE 类方法在 RLHF 上可以匹配甚至超过复杂的 PPO**，质疑了 Critic 的必要性。

### 4.3 ReMax：用贪心回答当基线

ReMax（arXiv:2310.10505）不做多样本统计，而是用**模型对同一 prompt 的贪心解码（greedy，不采样）回答的奖励**作为每条采样回答的 baseline：

```
A_i = R_i - R( greedy-decode(prompt) )
```

优点是每个 prompt 只需一次额外的贪心生成（可用高效的非采样解码），显存和计算开销小；论文给出了方差缩减的理论分析。缺点是贪心回答的质量本身随训练变化、计算开销仍随每轮 prompt 数线性增长。

### 4.4 四种 Baseline 对比

| 算法 | Baseline | 需要 Critic | 每 prompt 额外开销 | 特点 |
|---|---|---|---|---|
| PPO | 学习的价值网络 V(s) | ✅ | 无（但常驻同规模模型） | 经典、稳定、成本最高 |
| RLOO | 其他 N−1 条采样的均值 | ❌ | 已含在采样组内 | 留一，方差小 |
| ReMax | 贪心回答的奖励 | ❌ | 1 次贪心生成 | 简单，额外生成成本 |
| GRPO | 组内采样均值 + 组内标准化 | ❌ | 已含在采样组内 | clip + 组相对，工程主流 |
| REINFORCE++ | 全局/组均值 + 全局归一化 | ❌ | 已含在采样组内 | 长 CoT 稳定性好 |

---

## 5. GRPO：当代大模型 RL 的主角

**导读：** GRPO（Group Relative Policy Optimization）出自 DeepSeekMath 论文（2024，arXiv:2402.03300），因 DeepSeek-R1（arXiv:2501.12948）的成功成为业界最主流的大模型 RL 算法，并被 Qwen、大量开源推理模型采用。

### 5.1 核心思想：组相对优势

GRPO 彻底去掉 Critic。对每个 prompt q，从旧策略采样 **G 条**回答（如 G=8 或 16）：

```
{o_1, o_2, ..., o_G} ~ πθ_old(· | q)
```

每条回答由奖励模型/验证器打分，得到 {r_1, ..., r_G}。然后用**组内统计**计算每条回答的优势：

```
组内均值： μ = (1/G) Σ_k r_k
组内标准差：σ = sqrt( (1/G) Σ_k (r_k - μ)^2 )

第 i 条回答的优势： A_i = (r_i - μ) / (σ + eps)
```

这就是 "Group Relative" 的含义：**不看奖励绝对值，只看这条回答相对于同一个 prompt 下其他回答好不好。**

一个直观的数学题例子：同题采样 4 条回答，奖励为 {0, 0, 1, 1}（错/对），则 μ=0.5，组内标准化后错解得负优势、正解得正优势——无需任何价值网络就把"该压低谁、该提高谁"区分清楚。

### 5.2 目标函数：Token 级 Clip + 显式 KL

GRPO 在回答的每个 token 上使用类似 PPO 的 clip 目标，同时把 KL 正则**直接写进损失**（而非加在奖励里，这是与 PPO 的一个工程差异）：

```
J_GRPO(θ) =
  E_q,E{~πθ_old} [
    (1/|o_i|) Σ_t {
      min( ρ_i,t · A_i, clip(ρ_i,t, 1-ε, 1+ε) · A_i )
    }
    - β · KL( πθ(·|q) || π_ref(·|q) )
  ]

其中 ρ_i,t = πθ(o_i,t | q, o_i,<t) / πθ_old(o_i,t | q, o_i,<t)
```

注意目标里除以 |o_i| 做**长度归一化**，使长短回答对梯度的贡献相对均衡。

**KL 的无偏估计（k3 估计子）：** GRPO 采用一个必定非负的估计量，直接用采样到的 token 计算，不需要额外构造参考模型分布：

```
对每个采样 token，令 ρ = πθ/π_ref
KL 估计 = ρ - log ρ - 1        # 由 x - log x - 1 ≥ 0 知其恒为非负
```

### 5.3 GRPO vs PPO：差异总览

| 维度 | PPO | GRPO |
|---|---|---|
| Baseline | Critic 价值网络 | 组内奖励均值 |
| 优势粒度 | 每个 token（GAE） | 每条回答一个标量，平摊给该回答所有 token |
| Critic 模型 | 需要（同规模，显存翻倍） | **不需要** |
| KL 位置 | 通常加在 token 奖励里 | 直接加在 loss 里（无偏估计子） |
| 采样 | 每 prompt 1 条（典型） | 每 prompt G 条 |
| 调参复杂度 | 高 | 相对简单 |
| 主要缺点 | 贵、Critic 难训 | 组内 std 在小 G/奖励全同时不稳；token 级 clip 有病理 |

### 5.4 GRPO 的已知病理

1. **所有回答奖励相同时优势全为 0：** 组内 σ=0，该 prompt 本批不提供学习信号（难题全错或简单题全对都如此）。
2. **小 group 下方差大：** G 小时组内均值/标准差估计噪声大。
3. **长度偏置：** Dr.GRPO 研究（见 6.2）发现 GRPO 的 token 级目标存在人为拉长回答的优化偏置，尤其会让错误回答变得冗长。
4. **低概率 token 的 clip 退化：** clip-higher 研究指出，entropy 高、概率极低的推理 token 因 ratio 几乎总在裁剪区内而得不到有效梯度。

---

## 6. GRPO 的改进变体：DAPO、Dr.GRPO、GSPO

**导读：** R1 之后，社区围绕 GRPO 的病理提出多个高质量改进。本章介绍三个代表作：字节 DAPO（开源大规模 RL 系统）、Dr.GRPO（去长度偏置）、阿里 Qwen GSPO（序列级优化）。

### 6.1 DAPO：四项关键技术

DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization，arXiv:2503.14476，字节跳动 Seed）完整开源了一个大规模 RL 系统，**用 Qwen2.5-32B base 在 AIME 2024 上达到 50 分**。其四项关键技术：

1. **Clip-Higher（解耦裁剪）：** 对正优势动作放宽上裁剪界（如 ε_high=0.28），同时保持下裁剪界 ε_low=0.2。因为坏动作（A<0）被压低时 ratio 通常变小、很少触及下裁剪界，而下界 0.2 过紧会限制好动作继续增强——尤其推理中那些低概率但关键的探索 token。
2. **Dynamic Sampling（动态采样）：** 一个 batch 中奖励全同（全对或全错）的 prompt 会产生零优势、浪费算力。DAPO 在组内优势全为 0 时**对该 prompt 重新采样**，直到组内出现信号，并把整体采样规模动态维持在目标上，显著提升有效训练样本占比。
3. **Token-Level Policy Gradient Loss（Token 级损失）：** 不再用"回答总奖励除以回答长度"做归一化，而是把 loss 在整个 batch 的所有有效 token 上平均，避免短回答被长回答稀释梯度，让长短回答获得公平的梯度规模。
4. **Overlong Reward Shaping（超长奖励塑形）：** 对因超长被截断的回答，如果其已生成部分包含正确答案，给予**软奖励（soft reward）**而非直接 0 分，缓解模型为避免截断而学到的异常策略、也减少长度惩罚噪声。

### 6.2 Dr.GRPO：去除长度膨胀偏置

论文《Understanding R1-Zero-Like Training: A Critical Perspective》（arXiv:2503.20783）做了两件事：

1. **批判性审视 R1-Zero 式训练：** 研究基座模型与 RL 的关系（发现 DeepSeek-V3-Base 已存在"Aha moment"，Qwen2.5 base 即使无 prompt 模板也有强推理能力，提示预训练偏置的影响）；
2. **发现并修复 GRPO 的长度偏置：** token 级长度归一化、标准差归一化等组合会产生优化偏置，**人为放大回答长度（尤其是错误回答）**，导致 token 效率低下。Dr.GRPO 移除这些偏置源，成为一个无偏（unbiased）的精简配方：

```
论文一手结果：7B 模型的极简 R1-Zero 配方在 AIME 2024 上达到 43.3% 准确率
```

### 6.3 GSPO：序列级重要性采样

GSPO（Group Sequence Policy Optimization，arXiv:2507.18071，阿里巴巴 Qwen 团队）的核心改变是：**重要性比率不再按 token 定义，而按整条序列的似然定义**，并据此做序列级裁剪、奖励与优化：

```
token 级 ratio（GRPO）：ρ_t = πθ(o_t|...) / πθ_old(o_t|...)
序列级 ratio（GSPO）：ρ = Π_t ρ_t = πθ(o) / πθ_old(o)
```

论文声称的优势（一手摘要）：

- 相比 GRPO 训练效率与性能更优；
- **显著稳定 MoE（混合专家）模型的 RL 训练**——MoE 中 token 级路由使 token 级 ratio 噪声大，序列级估计更稳；
- 有简化 RL 基础设施设计的潜力。该方法已用于最新 Qwen3 模型的训练。

### 6.4 变体对比总表

| 算法 | 出处 | 关键改动 | 解决的问题 | 报告中的代表结果 |
|---|---|---|---|---|
| GRPO | DeepSeekMath 2024 | 组内相对优势、去 Critic | PPO 太贵 | R1 系列推理模型 |
| DAPO | 字节 2025 | clip-higher + 动态采样 + token-level loss + 超长软奖励 | 零优势浪费、裁剪退化、截断噪声 | Qwen2.5-32B，AIME24 50 分 |
| Dr.GRPO | 2025 | 去掉长度膨胀的优化偏置 | 回答冗长、token 效率低 | 7B，AIME24 43.3% |
| GSPO | 阿里 2025 | 序列级 ratio / clip / 优化 | token 级噪声、MoE RL 不稳 | 用于 Qwen3 |

---

## 7. 奖励范式：RLHF、RLAIF、RLVR 与 R1 训练流程

**导读：** 算法是"怎么学"，奖励是"学什么"。本章梳理三大奖励来源及 R1 的完整多阶段配方。

### 7.1 三种奖励来源

| 范式 | 奖励如何产生 | 优点 | 缺点 |
|---|---|---|---|
**RLHF** | 人类对回答两两比较 → 训练奖励模型 RM → RM 打分 | 贴合真实人类偏好 | RM 训练/维护贵，可被策略钻空子（reward hacking） |
**RLAIF / CAI** | 用强模型（AI feedback）或一组原则（Constitutional AI）生成偏好/打分 | 成本低、可大规模 | 继承反馈模型的偏见，可信度依赖反馈模型 |
**RLVR** | 规则/程序验证：数学答案匹配、代码单测、格式校验（常为 0/1） | 客观、廉价、无偏、可大规模 | 只适用于可验证任务（数学、代码、STEM） |

### 7.2 Reward Hacking 与常见奖励设计

RL 训练中策略会不断寻找奖励函数的漏洞，典型对策：

- **KL 约束**：限制偏离参考模型的程度（前述 β 项），是防 reward hacking 的第一道闸；
- **长度奖励/长度惩罚**：RLHF 常加格式与简洁度奖励；RLVR 中要小心"废话奖励"和长度偏置（Dr.GRPO 的发现）；
- **规则奖励组合**（R1 风格）：

```
总奖励 = 正确性奖励（答案匹配，±1 或 0/1）
       + 格式奖励（如是否正确使用 <think>/<answer> 标签）
       + 语言一致性奖励（混合语言场景）
```

### 7.3 DeepSeek-R1 的多阶段训练流程

根据 R1 技术报告（arXiv:2501.12948）：

```
R1-Zero 路线（极简，纯 RL）：
  Base Model ──GRPO + RLVR──→ R1-Zero
  （不用任何 SFT，直接在基座上做 RL；会出现可读性/语言混杂问题）

R1 正式路线（多阶段）：
  ① Base
  ② 冷启动 SFT（用少量长 CoT 种子数据）
  ③ 面向推理的 RL（GRPO + 可验证奖励）
  ④ 拒绝采样 + SFT（用 RL 模型生成并筛选数据，扩充各领域数据）
  ⑤ 全场景 RL（RM + 规则奖励，兼顾通用与推理）──→ R1

蒸馏：把 R1 的推理轨迹用于微调小模型（1.5B–70B），小模型同样获得强推理能力
```

R1 的核心经验：**对可验证任务，奖励对了、算法稳了，模型能自主发现反思与验证等高级推理模式，而不需要大量人工推理标注。** 这直接引发了 2025 年全行业的 RLVR 浪潮。

---

## 8. 核心实现伪代码

**导读：** 给出 GRPO 训练循环的最小实现骨架，帮助把前述公式落到工程。

```python
import torch

def grpo_step(prompts, policy, ref, reward_fn,
              G=8, eps=0.2, beta=0.04):
    """一个最小的 GRPO 训练步（示意，省略分布式与细节）。"""
    rollouts, ratios, logp_old_all, rewards = [], [], [], []

    # ① Rollout：每个 prompt 采样 G 条回答
    for q in prompts:
        outs, logp = [], []
        for _ in range(G):
            o, lp = policy.generate(q, do_sample=True)   # 返回回答+token对数概率
            outs.append(o); logp.append(lp)
        r = [reward_fn(q, o) for o in outs]             # RM 或规则验证器
        rollouts.append((q, outs, logp)); rewards.append(torch.tensor(r))

    loss = 0.0
    for (q, outs, logp), r in zip(rollouts, rewards):
        # ② 组相对优势（标准化）
        A = (r - r.mean()) / (r.std() + 1e-6)

        for o, lp_old, a in zip(outs, logp, A):
            # ③ 用当前策略重算 token 对数概率，得到重要性比率
            lp_new = policy.logprob(o, q)
            rho = torch.exp(lp_new - lp_old)            # token 级 ratio

            # ④ PPO 式 clip 目标（a 是整条回答的标量优势，广播到 token）
            L = torch.min(rho * a,
                         rho.clamp(1 - eps, 1 + eps) * a).mean()

            # ⑤ KL 无偏估计子：rho_ref - log rho_ref - 1
            lp_ref = ref.logprob(o, q)
            rho_ref = torch.exp(lp_new - lp_ref)
            kl = (rho_ref - torch.log(rho_ref) - 1).mean()

            loss += -L + beta * kl

    # ⑥ 反传更新（实际系统中配合分布式引擎：vLLM rollout、
    #    FSDP/Megatron 训练、KL/优势的 token 级 mask 等）
    loss.backward()
    policy.optimizer.step()
```

主流开源 RL 框架：**OpenRLHF、verl（字节，DAPO 基座）、TRL（Hugging Face）、SLIME（智谱）、NeMo-Aligner（NVIDIA）**，均已内置 PPO/GRPO 及常见变体。

---

## 9. 结论与选型建议

**导读：** 用编号结论和工程 checklist 收束全文。

### 9.1 编号结论

1. **大模型 RL 的一切算法都源于策略梯度 + baseline。** baseline 怎么选，决定了算法家族：学习的价值网络（PPO）、留一均值（RLOO）、贪心奖励（ReMax）、组内均值（GRPO）、全局归一化（REINFORCE++）。
2. **PPO 经典但昂贵：** Actor-Critic + GAE + token 级 KL，需要同规模 Critic，显存翻倍、超参敏感，仍是工业界稳定基线。
3. **GRPO 是当代主流：** 去 Critic、组相对优势、token clip、KL 入损失，工程性价比最高，是 R1/Qwen 等推理模型的共同选择。
4. **改进在消除 GRPO 的四类病理：** DAPO（裁剪/采样/长度/截断）、Dr.GRPO（长度膨胀偏置）、GSPO（序列级，稳 MoE）代表了 2025 年的演进方向。
5. **奖励与算法同等重要：** RLHF 靠奖励模型对齐偏好，RLVR 靠可验证规则激发推理；KL 约束是防 reward hacking 的基本手段。
6. **R1 验证了"纯 RL 激发推理"的可行性**：对可验证任务，模型能自主涌现反思/验证行为，少依赖人工推理标注。

### 9.2 工程选型 Checklist

- ☐ 预算充足、求稳、已有成熟 PPO 栈：**PPO**（OpenRLHF / TRL）。
- ☐ 主流推理模型训练（数学/代码/STEM）：**GRPO + RLVR**，G 取 8–16。
- ☐ 零优势样本多、算力浪费明显：上 **DAPO 动态采样 + clip-higher**。
- ☐ 回答越训越长、错误答案啰嗦：换 **Dr.GRPO** 或检查长度归一化/标准化偏置。
- ☐ 训练 MoE 模型、token 级方差大：评估 **GSPO** 序列级优化。
- ☐ 长 CoT 且 group 小、优势不稳：**REINFORCE++-baseline**（全局归一化）。
- ☐ 监控面板必备：reward、响应长度、entropy、KL、clip fraction（被裁剪 token 占比）、组内零优势比例。
- ☐ 门禁测试同时覆盖**目标任务提升**与**通用能力不退化**，并专门做 reward hacking 用例。
- ☐ 基础设施：rollout 与训练分离（vLLM/高并发采样 + FSDP/Megatron），优先解决生成吞吐瓶颈。

### 9.3 注意事项与诚实性声明

- 本报告 13 篇 arXiv 编号（1707.06347、2203.02155、2305.18290、2402.03300、2501.12948、2402.14740、2310.10505、2304.06767、2503.14476、2503.20783、2507.18071、2501.03262、2509.08827）均经 **arXiv 官方 API 核实**；DAPO（AIME 50 分）、Dr.GRPO（7B AIME 43.3%）、GSPO 机制等关键结论摘自**论文摘要原文**。
- 知乎 API 返回的正文不含 LaTeX 公式图片，本报告公式依据一手论文与标准定义重述；社区实践反馈（如 REINFORCE++ 的适用场景）已注明为知乎作者经验，采用前请在自己的任务上验证。
- 算法超参（ε、β、G、γ、λ）的最优值高度依赖任务与框架，报告给的是常用范围与原理，不是可直接照抄的配置。

---

## 10. 参考资料

### 10.1 原始论文（arXiv 编号经官方 API 核实）

1. Schulman et al. **Proximal Policy Optimization Algorithms.** 2017.
   https://arxiv.org/abs/1707.06347
2. Ouyang et al. **Training language models to follow instructions with human feedback（InstructGPT）.** NeurIPS 2022.
   https://arxiv.org/abs/2203.02155
3. Rafailov et al. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model（DPO，非RL对照）.** NeurIPS 2023.
   https://arxiv.org/abs/2305.18290
4. Luo et al. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models（GRPO）.** 2024.
   https://arxiv.org/abs/2402.03300
5. DeepSeek-AI. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.** 2025.
   https://arxiv.org/abs/2501.12948
6. Ahmadian et al. **Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs（RLOO）.** ACL 2024.
   https://arxiv.org/abs/2402.14740
7. Li et al. **ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models.** ICML 2024.
   https://arxiv.org/abs/2310.10505
8. Dong et al. **RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment.** 2023.
   https://arxiv.org/abs/2304.06767
9. Yu et al. **DAPO: An Open-Source LLM Reinforcement Learning System at Scale.** 2025.
   https://arxiv.org/abs/2503.14476
10. Liu et al. **Understanding R1-Zero-Like Training: A Critical Perspective（Dr.GRPO）.** 2025.
    https://arxiv.org/abs/2503.20783
11. Xu et al. **Group Sequence Policy Optimization（GSPO）.** 2025.
    https://arxiv.org/abs/2507.18071
12. Hu et al. **REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization.** 2025.
    https://arxiv.org/abs/2501.03262
13. Li et al. **A Survey of Reinforcement Learning for Large Reasoning Models.** 2025.
    https://arxiv.org/abs/2509.08827

### 10.2 开源框架

- OpenRLHF：https://github.com/OpenRLHF/OpenRLHF
- verl（字节，DAPO 训练框架）：https://github.com/volcengine/verl
- TRL（Hugging Face）：https://github.com/huggingface/trl
- NeMo-Aligner（NVIDIA）：https://github.com/NVIDIA/NeMo-Aligner

### 10.3 知乎文章（社区信源，交叉验证用）

1. 绝密伏击. **手撕大模型强化学习数学推导：REINFORCE、PPO 与 GRPO 公式全解析.**
   https://zhuanlan.zhihu.com/p/2058583217666519600
2. 余培杰. **从LLM的视角看策略梯度、PPO、GRPO（包含详细推导过程）.**（153 赞）
   https://zhuanlan.zhihu.com/p/1887162711286211485
3. 曾天真. **详解DeepSeek-R1核心强化学习算法：GRPO.**（711 赞）
   https://zhuanlan.zhihu.com/p/21046265072
4. vibe life. **LLM RL I - 从 PPO 到 GRPO：6改进、冲突、共识与适用边界.**
   https://zhuanlan.zhihu.com/p/2084486924069356995
5. 初七9527. **RLHF 对齐之 REINFORCE++ 算法 - 比 GRPO 稳定比PPO快.**（543 赞）
   https://zhuanlan.zhihu.com/p/14888098807
6. 问夏. **GRPO、GSPO、DAPO、DRGRPO.**
   https://zhuanlan.zhihu.com/p/1947023655080001638
7. 绝密伏击. **十分钟读懂 PPO、GRPO.**（124 赞）
   https://zhuanlan.zhihu.com/p/1916437378278596847
8. 绝密伏击. **DeepSeek-R1 技术报告解读.**（1134 赞）
   https://zhuanlan.zhihu.com/p/19868935152
9. 引线小白. **学习deepseek R1：一文读懂大语言模型中的强化学习(SFT、RFT、DPO、PPO、GRPO).**
   https://zhuanlan.zhihu.com/p/21178712267
10. 蜡笔小辛. **从一个 token 到 PPO、GRPO、GSPO：讲清大模型强化学习的完整脉络.**
    https://zhuanlan.zhihu.com/p/2048027180802875523
11. 月球上的人. **一文教你理解GRPO算法(附代码).**（65 赞）
    https://zhuanlan.zhihu.com/p/1977856962004791608
12. 长琴. **Reinforce++和它的KL Loss选择.**
    https://zhuanlan.zhihu.com/p/1966082771828078245
13. FUNNY AI. **清华最新发布114页大型推理模型的强化学习综述.**
    https://zhuanlan.zhihu.com/p/1951335985762767489

---

*报告完。核心事实经一手论文与 arXiv API 核实，社区经验与转述数据已在正文显式标注。*
