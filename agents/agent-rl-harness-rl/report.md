# Agent-RL 与 Harness-RL 技术调研报告

> 报告日期：2026-09-23
> 信源：知乎社区技术文章（12篇深度分析）、arXiv 论文（OpenForgeRL、CoreCraft、Miles/AReaL/verl 技术报告、DeepAnalyze/ReTool 等）、开源项目文档
> 主题：系统梳理 LLM Agent 强化学习的两条技术路线——「简化环境中的Agent-RL」与「Harness-Native RL（Harness-RL）」的原理、方法、演进、工程实践与选型建议

---

## 目录

1. [背景与问题定义](#1-背景与问题定义)
2. [核心概念：Agent、Harness 与 RL 的关系](#2-核心概念agentharness-与-rl-的关系)
3. [原理详解](#3-原理详解)
   - [3.1 传统 Agent-RL 的范式与假设](#31-传统-agent-rl-的范式与假设)
   - [3.2 训推不一致：Agent-RL 的根本缺陷](#32-训推不一致agent-rl-的根本缺陷)
   - [3.3 Harness-Native RL 的核心思想](#33-harness-native-rl-的核心思想)
   - [3.4 Harness-Benefit 与 Harness-Updating](#34-harness-benefit-与-harness-updating)
4. [实施方法与工程实践](#4-实施方法与工程实践)
   - [4.1 传统 Agent-RL 的训练流水线](#41-传统-agent-rl-的训练流水线)
   - [4.2 Harness-Native RL 的训练流水线](#42-harness-native-rl-的训练流水线)
   - [4.3 算法选择：GRPO / GSPO / AT-GRPO / SAO](#43-算法选择)
   - [4.4 基础设施选型](#44-基础设施选型)
5. [方案对比与方法演进](#5-方案对比与方法演进)
6. [核心实现要点：算法伪代码](#6-核心实现要点算法伪代码)
7. [结论与选型建议](#7-结论与选型建议)
8. [参考资料](#8-参考资料)

---

## 1. 背景与问题定义

### 1.1 Agent 能力训练的挑战

随着大模型从「单轮问答」走向「能使用工具、执行多步任务、与真实环境交互」的 Agent，传统的 SFT（监督微调）+ RLHF（基于人类反馈的强化学习）范式暴露出明显不足：

- **单轮 RL（如数学推理）**的 rollout 是纯文本生成，环境简单（题目→答案），奖励函数可以自动判定对错（规则匹配、答案比对）。
- **Agent 任务**（写代码、浏览网页、操作 OS、跨应用执行）涉及**多轮工具调用、有状态环境交互、长程规划、错误恢复**，轨迹长度可达数百到数万 token，传统方法难以直接扩展。

### 1.2 一个反直觉的实验事实

2026 年微软和多个团队独立发现了一个震撼性结果：

> **同一个模型权重，只更换推理时的 Harness（脚手架），ClawEval 基准的 pass@1 可以从 48.5% 掉到 20.9%**，差距超过两倍。

这说明 Agent 的实际表现不只是「模型能力」问题——Harness（提示词、工具注册、上下文管理、重试策略、超时设置、子代理调度）与模型权重共同决定行为。如果训练时用了一个「简化版 Harness」，部署时换到「真实 Harness」，模型面对的观测分布、行动空间、错误模式都发生了根本性变化，性能严重衰减。

### 1.3 本报告核心问题

| 问题 | 对应章节 |
|------|---------|
| Agent-RL 和 Harness-RL 的本质区别是什么？ | §2–§3 |
| 为什么训推不一致是 Agent-RL 的核心痛点？ | §3.2 |
| Harness-Native RL 如何实现？业界有哪些成熟方案？ | §4 |
| 不同场景下该选哪条路线？落地 checklist 是什么？ | §5–§7 |

**直觉类比**：把 LLM 比作F1赛车手，Harness 就是赛车+赛道+维修站。传统 Agent-RL 是让车手在模拟器里练（环境简化、路况完美、没有故障），然后直接送到真实赛道比赛——车手当然不适应。Harness-Native RL 是直接在真实赛道上练，用真实的赛车、面对真实的故障和天气，练出来的能力直接可用。

---

## 2. 核心概念：Agent、Harness 与 RL 的关系

### 2.1 Agent 的分解公式

业界逐步形成了一个共识性的 Agent 能力分解：

```
Agent 行为 = Model(权重) × Harness(脚手架) × Environment(环境)
```

- **Model**：基础模型权重（Qwen、GLM、GPT 系列），决定原始的语言理解、推理、生成能力。
- **Harness（推理脚手架）**：包裹模型的运行时框架——系统提示词、工具注册与 schema、上下文管理策略、重试/超时/错误处理逻辑、记忆模块、子代理调度策略、输出解析器。
- **Environment**：Agent 实际操作的外部世界——文件系统、浏览器、API、数据库、操作系统 GUI。

### 2.2 Agent-RL vs Harness-RL 的定义差异

| 维度 | 传统 Agent-RL | Harness-Native RL (Harness-RL) |
|------|--------------|-------------------------------|
| **训练对象** | 模型权重（θ） | 模型权重（θ）+ Harness 配置（η） |
| **训练环境** | 简化版环境（Mock 工具、合成数据、简化 grader） | 真实部署 Harness（完整工具链、真实环境） |
| **Rollout 方式** | 同步、受控、单进程 | 异步、真实多进程、通过 Proxy 拦截模型调用 |
| **奖励来源** | 简化 verifier（规则匹配、单元测试） | 真实任务 outcome（单测通过/任务完成）+ 过程奖励 |
| **训推一致性** | 低（分布偏移严重） | 高（训推同路径） |
| **工程复杂度** | 低（复用现有 RLHF 框架） | 高（需要 LLM Proxy、异步轨迹收集、中间件拦截） |
| **代表工作** | ReTool、Search-R1、WebArena RL、Kimi-K1.5 长程 RL | OpenForgeRL（微软）、Agent Lightning、CoreCraft、NeoHorse 数据引擎 |

### 2.3 什么是 Harness-Updating？

Harness-Updating（脚手架更新）是 Harness-RL 的一个进阶方向：不只优化模型权重 θ，还把 Harness 中的可调参数 η（Prompt 模板、工具描述文本、Workflow 路由逻辑、重试策略参数、上下文截断阈值）一并作为优化对象，通过 RL 或搜索方法联合优化 (θ, η)。

核心洞察来自 OpenForgeRL 等工作：Harness 的「超参数」对性能影响巨大，但目前全部依赖人工调优。Harness-Updating 让这些参数也进入学习循环。

---

## 3. 原理详解

### 3.1 传统 Agent-RL 的范式与假设

传统 Agent-RL 把 Agent 任务形式化为一个多步 MDP，但在实际训练时做了大量简化：

**形式化定义**：
```
状态 S_t = (历史对话, 工具返回结果, 当前环境观测)
动作 A_t = 模型生成的 token（可能是工具调用或最终回答）
转移 T(S_{t+1} | S_t, A_t) = 执行工具调用/下一步推理，由 Harness+Environment 决定
奖励 R(S_T) = 任务完成度（如代码是否通过单测、答案是否正确）
```

传统方法的关键假设：

1. **工具调用可以被简化为字符串模式匹配**（如训练时用固定格式的 mock 返回，而不是真实 API 调用）。
2. **奖励函数可以离线/同步计算**（不需要真实运行环境）。
3. **训练和推理时的观测分布一致**（即「简化环境」足够接近真实环境）。
4. **Harness 是透明的、可以被剥离的**——模型能力 ≈ Agent 能力。

这四条假设在长程、真实环境 Agent 任务中**全部不成立**。

### 3.2 训推不一致：Agent-RL 的根本缺陷

当训练用的 Harness（H_train）和部署用的 Harness（H_deploy）不同时，会产生**多维度分布偏移**：

**(a) 观测空间偏移（Observation Shift）**
- 训练时工具返回的是预先写好的 mock 字符串（格式工整、无异常、无截断）。
- 真实工具可能返回超时错误、HTML 噪声、截断内容、格式异常。
- 模型在训练中从未见过这些模式，部署时遇到就会「不知所措」。

**(b) 行动空间偏移（Action Space Shift）**
- 训练时可能只开放 3 个工具；部署时有 20 个工具，工具 schema 更复杂。
- 训练时不允许并行工具调用；部署时 harness 会自动触发并行调用。
- 训练时不做重试；部署时 harness 会在失败时自动重试（模型收到的输入是「重试后的结果」，而非错误信息）。

**(c) 上下文管理偏移**
- 训练时所有上下文完整可见；部署时 harness 可能做了摘要、截断、滑动窗口。
- 模型在训练中依赖的某些线索（如特定格式的错误提示、特定前缀）在部署时被 harness 的预处理层剥离。

**(d) Prompt/系统消息偏移**
- 训练时的 system prompt 是工程师精心设计的 RL 专用 prompt；部署时换成产品 prompt。
- 大量实验表明，system prompt 微小变化（甚至只是加一个换行）可导致已训练策略性能下降 10–30%。

**(e) 奖励黑客（Reward Hacking）**
- 在简化 grader 环境中，模型可能学会「欺骗 grader」（比如输出特殊字符串触发通过、利用 mock 工具的确定性模式）。
- 这些策略在真实环境中完全无效，但在训练中会被 reward 强化。

**量化证据**：
- OpenForgeRL 实验：同一模型换 harness 后 ClawEval pass@1 从 48.5% → 20.9%。
- 397B 模型在简化训练 harness 上的收益，换到 OpenCode（真实代码环境）后明显缩小——证明部分能力是「harness-overfitting」。

### 3.3 Harness-Native RL 的核心思想

Harness-Native RL 的原则非常简洁：**训练时使用的 Harness 就是部署时使用的 Harness**，不做任何简化。

**核心机制：LLM Proxy 拦截**

```
┌──────────────────────────────────────────────────┐
│           部署时 (Deployment)                     │
│  ┌──────┐    ┌──────────┐    ┌─────────────┐     │
│  │ User │───▶│ Harness  │───▶│   Tools     │     │
│  └──────┘    │ (Prompt, │    │ (Browser,   │     │
│              │  Tools,  │◀──▶│  Shell,     │     │
│              │  Memory, │    │  API, ...)  │     │
│              │  Retry)  │    └─────────────┘     │
│              └────┬─────┘                         │
│                   │ 直接调用 LLM API              │
│                   ▼                               │
│              ┌─────────┐                         │
│              │   LLM   │                         │
│              └─────────┘                         │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│           训练时 (Harness-Native RL)              │
│  ┌──────┐    ┌──────────┐    ┌─────────────┐     │
│  │ Roll │───▶│ Harness  │───▶│   Tools     │     │
│  │  out │    │ (SAME as │    │ (REAL tools,│     │
│  │ Gen  │    │  deploy!)│◀──▶│  REAL env)  │     │
│  │  .   │    │          │    └─────────────┘     │
│  │      │    └────┬─────┘                         │
│              │ 调用 LLM（被拦截）                 │
│              ▼                                    │
│         ┌─────────────┐                           │
│         │ LLM Proxy   │◀── 注入 action（当前policy│
│         │ (拦截层)    │    采样的 token 序列）    │
│         └──────┬──────┘                           │
│                │ 记录 (S_t, A_t, R_t) 到轨迹buffer│
│                ▼                                   │
│         ┌─────────────┐                           │
│         │   LLM (train)│←── PPO/GRPO 更新权重    │
│         └─────────────┘                           │
└──────────────────────────────────────────────────┘
```

关键设计点：

1. **零侵入**：Harness 代码无需修改，训练时和部署时跑的是同一套二进制/脚本。
2. **Proxy 拦截**：在 LLM API 调用层插入代理，既可以把当前 policy 的 action 注入到 harness 的执行流程中（替换真实 LLM 输出），又可以记录所有 (state, action, result) 三元组作为训练数据。
3. **异步收集**：多个 Harness 实例可以并行跑不同任务，trajectory 通过队列异步传给 trainer，解决长程 rollout 的延迟问题。
4. **真实奖励**：任务完成与否由真实环境的 outcome 决定（比如代码是否真的通过了单测、浏览器操作是否真的完成了表单提交），不依赖简化 grader。

### 3.4 Harness-Benefit 与 Harness-Updating

OpenForgeRL 和后续工作提出了两个关键概念：

**Harness-Benefit（脚手架增益）**：衡量 Harness 对 Agent 性能的贡献。形式化地，对于给定模型 θ：

```
Benefit(H; θ) = Perf(θ, H_deploy) - Perf(θ, H_minimal)
```

其中 H_minimal 是裸模型（无工具、无 prompt 工程、无重试）。这个概念揭示了：

- 如果在 H_train 上训练得到的 θ*，在 H_deploy 上 Benefit 很小，说明训练过拟合到了训练 harness。
- Harness-Native RL 的目标是让 H_deploy 的 Benefit 最大化。

**Harness-Updating（脚手架学习）**：把 Harness 中可调的离散/连续参数 η（prompt 模板、工具描述、路由策略、重试阈值、context window 管理策略等）也作为优化变量，联合优化 (θ, η)：

```
(θ*, η*) = argmax_{θ,η}  E[R(τ) | θ, η, H_deploy]
```

这可以通过多种方法实现：
- **梯度方法**：如果 η 是连续可微的（如某些阈值、temperature），可通过 PPO/REINFORCE 估计梯度。
- **离散搜索**：如果 η 是离散的（prompt 文本、工具选择），用 MCTS、进化算法、或元 Agent 提案+评估（如 AFlow、ADAS）。
- **混合方法**：θ 用 RL 优化，η 用 Prompt-Optimization 或 A/B 测试搜索。

**双轨训练**是 Harness-Updating 的一个重要实践：模型和 Harness 交替更新，类似 GAN 的训练循环。NeoHorse（华为）的工作展示了将部署中的路由 harness 作为数据引擎，每轮交互轨迹经筛选后转成 SFT 样本，再 on-policy 蒸馏回模型。

---

## 4. 实施方法与工程实践

### 4.1 传统 Agent-RL 的训练流水线

```
Step 1: 准备训练环境
  - 实现简化版工具（mock API、返回固定格式结果）
  - 实现 reward function（规则匹配/单元测试/答案比对）
  - 构造训练任务集（问题-答案对、SWE-bench类任务）

Step 2: Rollout 采样
  - 用当前 policy model 生成完整 trajectory
  - 每个 trajectory 包含多轮 (observation, action) 对
  - 同步执行，环境反馈即时返回

Step 3: 奖励计算
  - 终局奖励（outcome reward）：任务是否完成
  - 过程奖励（process reward）：每步是否合理（可选）
  - 格式奖励：是否符合工具调用格式

Step 4: PPO/GRPO 训练
  - 用 rollout 数据计算 advantage（相对基线、group-relative等）
  - 更新 policy 模型参数
  - 周期性在验证集上评估

Step 5: 部署到真实 Harness
  - 将训练好的模型接入真实 agent 框架
  - 发现性能衰减 → 回到 Step 1 调参（循环反复）
```

**优点**：框架成熟（verl、TRL、OpenRLHF 均支持），rollout 快，调试方便。
**缺点**：部署时性能衰减严重；模型可能学会 reward hacking；不适应真实工具的异常模式。

### 4.2 Harness-Native RL 的训练流水线

```
Step 1: 准备生产级 Harness
  - 使用部署时一模一样的 Harness 代码（不做任何简化）
  - 接入真实工具（浏览器、文件系统、真实 API）
  - 确保 Harness 可以通过环境变量/配置指向 LLM Proxy

Step 2: 部署 LLM Proxy
  - 在 Harness 和模型推理服务之间插入代理
  - 代理的职责：
    (a) 拦截 Harness 的 LLM 请求，记录当前 state（prompt、工具返回、历史）
    (b) 用当前训练中的 policy 生成 action（token序列）返回给 Harness
    (c) 记录 trajectory 数据到 replay buffer
    (d) 在推理阶段，可以透传给真实模型做 inference

Step 3: 异步 Rollout 集群
  - 启动 N 个 Harness worker，每个独立执行任务
  - Worker 通过 Proxy 与 Policy 服务交互（类似部署时的 API 调用模式）
  - Trajectory 完成后写入分布式 replay buffer
  - 支持 checkpoint resume、失败任务重试

Step 4: 训练循环
  - 从 replay buffer 采样 trajectory batch
  - 用 GRPO/PPO 等算法计算 loss，更新 model 权重
  - 异步同步权重到 rollout workers（支持 staleness 控制）
  - 定期在 held-out 真实任务集上评估

Step 5: 直接部署
  - 训练好的 model 权重直接加载到生产 Harness 中
  - 不需要额外适配，因为训练和部署用的是同一个 Harness
```

**关键技术细节**：
- **权重同步**：Miles v0.1 支持三种方案——RPC（轻量快）、NCCL（高性能）、HF Checkpoint（灵活）。
- **Staleness 控制**：异步训练中 rollout worker 使用的 policy 版本可能落后 trainer 数个 step，需要设置阈值（通常 <1 epoch），或用 importance sampling 校正 off-policy 偏差。
- **Prefix Replay**：微软的工作提出，多轮 distillation 时可以复用已验证过的 prefix（前缀轨迹），大幅减少 rollout 成本。
- **错误恢复**：Harness 本身会超时、崩溃、工具会返回错误——这些都是真实训练数据的一部分，不应丢弃，而是作为负样本训练模型的错误恢复能力。

### 4.3 算法选择

| 算法 | 适用场景 | 核心思想 | 代表工作 |
|------|---------|---------|---------|
| **GRPO** | 单轮/短程任务（数学、代码） | 去掉 Critic，用 group 内相对 reward 作为 baseline，省显存 | DeepSeek-R1、Qwen3 |
| **GSPO** | 长程、大规模 Agent 任务 | Group-Sequential，将长序列分段优化，解决长序列 credit assignment | Qwen3（Kimi K2 也用类似思路） |
| **DAPO** | 大规模稳定 RL | 动态采样、clip 优化、token级 loss 过滤 | ByteDance |
| **AT-GRPO** | 多智能体协作 | Agent-wise + Turn-wise 双维度 advantage 估计，支持多agent同时学习 | PettingLLMs |
| **SAO** | 长程 Agentic RL（多工具调用） | Stable Advantage Optimization，通过 advantage 归一化和正则化稳定长轨迹训练 | GLM-5.3 |
| **REINFORCE++** | 通用 RLHF | 简化 PPO，去掉 critic，鲁棒性强 | 华为诺亚 |
| **MCTS+RL** | 需要探索的复杂任务 | 搜索过程产生on-policy数据，RL学习搜索策略 | AlphaCode 系列、OpenHands |

**实践推荐**：
- 入门/单轮任务：GRPO（verl 原生支持，调参简单）。
- 多轮工具调用/Agent：GSPO + SAO 或 AT-GRPO（长程训练稳定）。
- 多智能体协作：AT-GRPO。
- 有 MCTS 搜索模块：策略蒸馏 + GRPO（如 Kimi-K1.5 路线）。

### 4.4 基础设施选型

| 框架 | 类型 | 特点 | 适用场景 |
|------|------|------|---------|
| **verl** | RL 训练框架（字节/清华） | 开源最活跃，支持多轮工具调用 RL，SGLang 后端 | 研究、生产均可 |
| **Miles** | 全栈 RL 系统（360） | 基于 slime，支持 FSDP/Megatron 双后端，三种权重同步方案 | 生产级大规模训练 |
| **AReaL** | 分布式 Agent RL（字节） | 完全异步，Replay Buffer 支持 off-policy 数据混合 | 大规模、需要高吞吐 |
| **OpenForgeRL** | Harness-Native RL（微软） | LLM Proxy 模式，零侵入接入任意 Harness | Harness-Native 首选 |
| **Agent Lightning** | Harness-Native RL（轻量） | ~3500 行代码，极简设计，直接用真实 Agent Harness | 快速原型、中小规模 |
| **Slime** | RL 框架 | 灵活 APR Replay Buffer，支持 partial rollout | 研究实验 |

**底层推理引擎**：
- vLLM：部署广泛，但有和训练引擎的 logprob 不一致问题（需引入 IS 校正）。
- SGLang：与 verl 深度集成，prefix cache 效率高，多轮 RL 性能优秀。

---

## 5. 方案对比与方法演进

### 5.1 Agent-RL 技术路线演进（2024–2026）

```
2024 Q3-Q4  第一波：单轮 RLHF 扩展到推理链
  └─ GRPO 普及（DeepSeek-R1），数学/代码推理突破
  └─ 局限：单轮、无工具、纯文本生成

2025 Q1-Q2  第二波：多轮/工具调用 RL（Agent-RL v1）
  └─ Search-R1、WebRL、ReTool：支持搜索/工具调用
  └─ verl 开源多轮 RL 支持（SGLang + verl 联合）
  └─ 问题：用简化环境训练，部署时掉点严重
  └─ Kimi-K1.5 长程 RL（但仍在受控环境中）

2025 Q3-Q4  第三波：发现训推不一致问题
  └─ 多篇论文揭示 harness 影响巨大
  └─ 黑盒 Harness RL 尝试（炼熵师等工作）
  └─ 异步 RL 框架（AReaL、Miles）成熟

2026 Q1-Q2  第四波：Harness-Native RL 新范式
  └─ OpenForgeRL（微软）：LLM Proxy 模式，任意 Harness 端到端训练
  └─ Agent Lightning：极简 Harness-Native 框架
  └─ CoreCraft：高保真真实环境训练
  └─ Harness-Updating/Harness-Benefit 理论框架提出
  └─ 397B 模型 Agent RL 训练配方公开（OpenAI o1 核心成员参与）

2026 Q3+   第五波（进行中）：联合优化
  └─ Model × Harness × Environment 协同进化
  └─ Self-Evolving Agent（Evolving-RL、Memento-Skills）
  └─ 多智能体协同 RL（PettingLLMs）
  └─ 部署后持续学习（Harness Continual Learning）
```

### 5.2 关键方法对比表

| 方法 | 训练环境 | Harness 一致性 | 多轮支持 | 异步 | 开源 |
|------|---------|--------------|---------|------|------|
| 传统 PPO/GRPO (verl) | 简化/Mock | ❌ 低 | ✅ | ⚠️ 部分 | ✅ |
| ReTool (Google) | 简化工具 | ❌ 低 | ✅ | ❌ | ✅ |
| DeepAnalyze | 简化搜索 | ❌ 低 | ✅ | ❌ | ✅ |
| Search-R1 | 简化搜索 | ❌ 低 | ✅ | ❌ | ✅ |
| AReaL | 可定制 | ⚠️ 中 | ✅ | ✅ | ✅ |
| Miles | 可定制 | ⚠️ 中 | ✅ | ✅ | ✅ |
| **OpenForgeRL** | **真实 Harness** | **✅ 高** | **✅** | **✅** | **✅** |
| **Agent Lightning** | **真实 Harness** | **✅ 高** | **✅** | **✅** | **✅** |
| CoreCraft | 高保真仿真 | ✅ 高 | ✅ | ✅ | ❓ |

### 5.3 实验结果对比（公开数据）

| 基准/任务 | 简化环境 Agent-RL | Harness-Native RL | 差距 |
|-----------|------------------|-------------------|------|
| ClawEval (pass@1) | 48.5%（训练 harness） | 48.5%（同harness）/ 20.9%（换harness） | 27.6pp 掉点 |
| SWE-bench Verified (小模型) | ~20%（mock env） | ~35%（真实 harness 训练） | +15pp |
| WebArena | ~18%（simplified browser） | ~28%（真实浏览器） | +10pp |
| ALFWorld (Evolving-RL) | 85%+ | 93.1% | +8pp |

> ⚠️ **注意**：以上数据来自知乎文章转述，引用前请二次核对原论文。

---

## 6. 核心实现要点：算法伪代码

### 6.1 Harness-Native RL 核心循环

```python
def harness_native_rl(
    harness_path: str,        # 部署时使用的 Harness 代码路径（不做任何修改）
    tasks: list[Task],
    model: PolicyModel,
    ref_model: ReferenceModel,
    reward_fn: Callable,       # 基于真实 outcome 的奖励函数
    num_iterations: int = 100,
    rollout_workers: int = 64,
    buffer_size: int = 4096,
):
    """Harness-Native RL 训练主循环"""
    
    # 1. 启动 LLM Proxy，拦截 Harness 对模型的调用
    proxy = LLMProxy(model=model, ref_model=ref_model)
    proxy.start()
    
    # 2. 启动异步 Rollout 集群
    rollout_pool = RolloutPool(
        harness_command=f"python {harness_path}",
        proxy_endpoint=proxy.endpoint,
        num_workers=rollout_workers,
    )
    
    for iteration in range(num_iterations):
        # 3. 收集 trajectories（异步）
        trajectories = []
        while len(trajectories) < buffer_size:
            task = sample_task(tasks)
            traj = rollout_pool.submit(task)  # 异步执行
            trajectories.append(traj)
        
        # 4. 计算奖励（真实 outcome，如单测通过率）
        for traj in trajectories:
            traj.reward = reward_fn(traj.final_state, traj.task)
        
        # 5. GRPO/PPO 更新（与传统 RL 类似）
        batch = make_batch(trajectories)
        loss = train_step(model, ref_model, batch)
        
        # 6. 异步同步新权重到 rollout workers
        proxy.broadcast_weights(model, max_staleness=1.0)  # 允许最多1个epoch延迟
        
        # 7. 在 held-out 真实任务上评估
        if iteration % 10 == 0:
            eval_score = evaluate(model, harness_path, eval_tasks)
            print(f"Iter {iteration}: loss={loss:.4f}, eval={eval_score:.4f}")
    
    return model
```

### 6.2 LLM Proxy 核心逻辑

```python
class LLMProxy:
    """拦截 Harness 对 LLM 的调用，注入当前 policy 的输出"""
    
    def __init__(self, model, ref_model=None):
        self.model = model          # 当前训练中的 policy
        self.ref_model = ref_model  # reference model（用于 KL 约束）
        self.trajectory_buffer = []
        self.current_rollouts = {}  # trace_id → 轨迹记录
    
    async def handle_chat_completion(self, request):
        """拦截 /v1/chat/completions 请求"""
        trace_id = request.get("trace_id", generate_id())
        
        # 记录当前 state（messages, tools, temperature 等）
        state = {
            "messages": request["messages"],
            "tools": request.get("tools", []),
            "temperature": request.get("temperature", 1.0),
        }
        
        # 用当前 policy 生成 action（而非调用真实 API）
        outputs = self.model.generate(
            messages=state["messages"],
            tools=state["tools"],
            temperature=state["temperature"],
        )
        
        # 记录 (state, action) pair
        action_token_ids = outputs.token_ids
        with torch.no_grad():
            logprob = self.model.logprob(state, action_token_ids)
            ref_logprob = self.ref_model.logprob(state, action_token_ids) if self.ref_model else 0.0
        
        self.current_rollouts.setdefault(trace_id, []).append({
            "state": state,
            "action_token_ids": action_token_ids,
            "logprob": logprob,
            "ref_logprob": ref_logprob,
        })
        
        # 返回给 Harness（格式与真实 API 完全一致）
        return {
            "id": f"chatcmpl-{trace_id}",
            "choices": [{"message": outputs.message, "finish_reason": "stop"}],
            "usage": outputs.usage,
        }
    
    def mark_episode_done(self, trace_id, final_reward):
        """Harness 执行完任务后调用，标记轨迹结束并写入 buffer"""
        traj = self.current_rollouts.pop(trace_id)
        traj.final_reward = final_reward
        self.trajectory_buffer.append(traj)
```

### 6.3 Harness-Updating 扩展（联合优化）

```python
def harness_updating_rl(
    harness_template,  # Harness 代码模板（含可优化参数 η）
    tasks,
    model,
    eta_searcher: EtaOptimizer,  # Prompt/Workflow 优化器
    num_outer_loops=10,
):
    """Model θ 和 Harness η 交替优化"""
    eta = eta_searcher.default_eta()
    
    for outer in range(num_outer_loops):
        # Phase 1: 固定 η，优化 θ（标准 Harness-Native RL）
        harness = instantiate(harness_template, eta)
        model = harness_native_rl(harness, tasks, model, num_iterations=50)
        
        # Phase 2: 固定 θ，搜索更好的 η
        eta_candidates = eta_searcher.propose_candidates(eta, model, harness)
        best_eta, best_score = None, -float("inf")
        for eta_candidate in eta_candidates:
            harness_cand = instantiate(harness_template, eta_candidate)
            score = evaluate(model, harness_cand, tasks)
            if score > best_score:
                best_eta, best_score = eta_candidate, score
        eta = best_eta
        
        print(f"Outer loop {outer}: best_score={best_score:.4f}")
    
    return model, eta
```

---

## 7. 结论与选型建议

### 7.1 核心结论

1. **Agent 能力 = 模型 × Harness**，只优化其中一个会留下巨大天花板。传统 Agent-RL 在简化环境中训练，部署时性能严重衰减（可达 50%+）。
2. **Harness-Native RL 是当前最务实的方向**：直接在部署用的 Harness 上训练，通过 LLM Proxy 拦截模式收集轨迹，零分布偏移。OpenForgeRL、Agent Lightning 等框架已开源可用。
3. **Harness-Updating 是下一个前沿**：把 Prompt、工具描述、Workflow 路由等脚手架参数也纳入优化，模型+脚手架联合学习，有望带来显著的额外增益。
4. **异步 RL 基础设施是规模化的关键**：长程 trajectory 在真实 Harness 中执行很慢（单个任务可达数分钟），必须异步收集 + 分布式训练，控制 policy staleness 是核心工程挑战。
5. **算法层面**：GRPO 仍然是最通用的基线；GSPO/SAO 在长程场景表现更好；多智能体场景需要 AT-GRPO 类的专门算法。

### 7.2 工程选型 Checklist

**☐ 如果你的场景是「单轮/短程、工具简单」（如简单 RAG、数学推理）**
- 走传统 Agent-RL 即可，用 verl + GRPO，成本低、成熟度高。
- 注意确保训练时的工具 schema 和部署时完全一致。

**☐ 如果你的场景是「长程、复杂工具链、真实环境交互」（如 SWE 代码生成、Browser Agent、OS Agent）**
- 必须走 Harness-Native RL，否则训推不一致会吃掉大部分收益。
- 推荐 OpenForgeRL 或 Agent Lightning 起步，逐步迁移。

**☐ 如果你已经有训练好的 Agent 模型，但部署后性能不及预期**
- 不要急着重新训模型——先排查 Harness 差异（Prompt、工具schema、超时/重试策略、上下文截断）。
- 用 Harness-Native 微调（small lr）适配真实 Harness，往往比从头训更高效。

**☐ 如果你的团队有较强工程能力，追求 SOTA**
- 搭建异步 RL 基础设施（Ray + SGLang + 自定义 Proxy）。
- 引入 Harness-Updating 优化 prompt 和 workflow。
- 考虑多智能体 AT-GRPO 协同训练。

### 7.3 注意事项与陷阱

- **奖励设计**：不要只依赖终局 outcome reward（稀疏、方差大），适当加入过程奖励（格式正确性、工具选择合理性），但要警惕过程奖励被 hack。
- **数据质量 >> 数据数量**：过滤掉 Harness 崩溃/超时的轨迹，这些轨迹的噪声会严重损害训练。
- **监控指标**：除了 reward 和 eval score，还要持续监控 KL(训练策略 || 推理策略) 、action entropy、轨迹长度分布、tool call 错误率。
- **版本对齐**：训练引擎和推理引擎的 logprob 计算必须定期对齐校验，否则 GRPO 的 advantage 估计有偏。
- **小模型也值得做 RL**：MiniCPM5-2B 级别的模型在合适的 Harness-RL 训练后，可以达到远超其参数级别的 Agent 能力。
- **Harness 本身的演进**：如果 Harness 在训练期间频繁更新（新工具加入、prompt 调整），需要持续做 on-policy 微调，否则模型会过时。

### 7.4 诚实性声明

- 本文中的实验数据（如 ClawEval 48.5%→20.9%、397B模型缩小）均来自知乎社区文章转述，未逐一核对原始论文，引用前建议对照 arXiv 原文确认。
- 部分框架（Miles、Agent Lightning）的具体能力范围基于其公开文档描述，未在本环境中实际部署验证。
- Harness-RL 是一个快速演进的方向（2026 Q3 仍在爆发期），部分推荐意见基于截至 2026-09 的认知，可能随新工作出现而过时。

---

## 8. 参考资料

### 原始论文/技术报告

1. OpenForgeRL Team. *OpenForgeRL: Train Harness-native Agents in Any Environment*. arXiv:2607.21557, 2026.
2. Kimi Team. *Kimi K1.5: Scaling Reinforcement Learning with LLMs*. arXiv:2501.12599, 2025.
3. Kimi Team. *Kimi K2: Open Agentic Intelligence*. 2025.
4. DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. 2025.
5. Xiao et al. *Training Generalizable Agents on High-Fidelity RL Environments (CoreCraft)*. 2026.
6. Zhang et al. *AFlow: Automating Agentic Workflow Generation*. 2025.
7. Hu et al. *ADAS: Automated Design of Agentic Systems*. 2025.
8. *Harness Continual Learning: Continual Adaptation Beyond Model Parameters*. arXiv:2608.19013, 2026.

### 知乎深度文章

9. 绝密伏击. 《LoRA 一作、OpenAI o1 核心成员的新工作：首次公开 397B 模型的 Agent RL 训练配方》. https://zhuanlan.zhihu.com/p/2082860762947703480
10. 我是刘玉书. 《大模型 Agent 强化学习系统学习指南》. https://zhuanlan.zhihu.com/p/2084575106010035340
11. nano. 《同一个模型换套 Harness，成绩为何能差两倍？OpenForgeRL 揭开了 Agent RL 的盲区》. https://zhuanlan.zhihu.com/p/2066311596893386186
12. 北方的郎. 《告别训练与部署脱节：OpenForge RL 如何让开源大模型驾驭任意复杂脚手架》. https://zhuanlan.zhihu.com/p/2065482566442948071
13. PaperAgent. 《微软一天连发3篇之：打造自己的Harness原生Agents》. https://zhuanlan.zhihu.com/p/2065804479589511392
14. 百无一用是书生. 《每日论文解读(7.27)——OpenForge RL 论文解读》. https://zhuanlan.zhihu.com/p/2065118911431972446
15. 炼熵师. 《黑盒harness下的LLM RL》. https://zhuanlan.zhihu.com/p/2062177691130926825
16. Ghost apple. 《Agentic RL：从一个终局分数，到可训练的经验闭环》. https://zhuanlan.zhihu.com/p/2073800126892602594
17. QikaQika. 《关于Agent模型能力和Agentic RL训练的整理》. https://zhuanlan.zhihu.com/p/2019366800190625461
18. 乞力马扎罗雪人. 《【AgentRL】工业级Agentic RL 训练对比选型指南》. https://zhuanlan.zhihu.com/p/1978600046514685178
19. vEth. 《Harness 构建者的实践清单：在训练分布的引力场中工作》. https://zhuanlan.zhihu.com/p/2058591945652449851
20. 石臻说AI. 《Agent 如何持续变强：从 Harness 自演化到模型进化的探索》. https://zhuanlan.zhihu.com/p/2081132570482422938
21. 智永. 《超越模型权重更新：Agentic AI时代的AI持续学习》. https://zhuanlan.zhihu.com/p/2028806091039876663
22. 听寒. 《单卡 V100 从零自研多轮 GRPO：把 Qwen3-4B 训成"检索—推理—再检索"Agent》. https://zhuanlan.zhihu.com/p/2080617731527799804
23. IchbinDerek. 《Agent 递归自我改进(RSI)的 Harness 工程》. https://zhuanlan.zhihu.com/p/2067214030054467237
24. 好奇的小逸. 《GLM-5.3 中长程 Agentic RL 的关键方法：SAO》. https://zhuanlan.zhihu.com/p/2072077539649049420
25. 龟壳. 《Harness Continual Learning》. https://zhuanlan.zhihu.com/p/2075328352307648239
26. James. 《RL-as-a-Service：Agent 强化学习的环境、验证与训练栈》. https://zhuanlan.zhihu.com/p/2079568299806008105
27. vibe life. 《Self Improvement Agent: HarnessX 与 LIFE-HARNESS》. https://zhuanlan.zhihu.com/p/2051607112724263376
28. HyperAI超神经. 《论文汇总丨Agent Harness新进展》. https://zhuanlan.zhihu.com/p/2082132789743313534

---

*本报告使用 tech-report-writing 技能（tech-reports-skills/tech-report-writing）生成。架构图使用 architecture-diagram 技能制作。*
