# Agentic RL：大模型智能体强化学习前沿技术报告

> 从单轮 RLVR 到多轮环境交互：系统梳理 2025–2026 年 Agentic RL 的核心挑战、算法谱系（StarPO/GiGPO/SPA-RL/ARPO/T²PO/DART）、奖励与环境工程、异步训练基础设施，以及 Kimi K2、MiniMax、GLM 等前沿系统的实战路线。
>
> 📅 2026-10-10 ｜ 信源：16 篇 arXiv 原始论文（编号经 arXiv API 逐一核实）+ 知乎高赞技术综述交叉验证

---

## 目录

1. [背景：从单轮 RLVR 到 Agentic RL](#1-背景从单轮-rlvr-到-agentic-rl)
2. [问题定义：Agentic RL 与单轮 RL 的五个本质差异](#2-问题定义agentic-rl-与单轮-rl-的五个本质差异)
3. [核心挑战：多轮环境下的七类病理](#3-核心挑战多轮环境下的七类病理)
4. [算法方案谱系](#4-算法方案谱系)
5. [学习信号：环境、奖励与课程](#5-学习信号环境奖励与课程)
6. [基础设施：异步数据流与大规模训练系统](#6-基础设施异步数据流与大规模训练系统)
7. [前沿系统实战：Kimi、MiniMax、GLM 路线对比](#7-前沿系统实战kiminimaxglm-路线对比)
8. [统一视角：三大不变量与工程 Checklist](#8-统一视角三大不变量与工程-checklist)
9. [参考资料](#9-参考资料)

---

## 1. 背景：从单轮 RLVR 到 Agentic RL

**导读：** 2025 年初 DeepSeek-R1 证明了单轮可验证奖励（RLVR）能激发数学/代码推理能力；而 2025 下半年到 2026 年，行业主战场已经转向让模型在真实环境中多轮行动——这就是 Agentic RL。

### 1.1 范式转移：从"答题"到"做事"

以 GRPO + RLVR 为代表的单轮 RL，优化的是**一次性生成**：给一道题，模型输出一段思维链加答案，验证器判对错，一个回合结束。它极大地提升了数学竞赛、代码竞赛等静态任务，但真实的智能体任务长这样：

```
观察状态 → 思考 → 调用工具（搜索/代码/浏览器/终端/API）
        → 拿到环境反馈 → 再思考 → 再调用 …（多轮）
        → 最终只有任务成功/失败一个稀疏信号
```

典型场景：网页购物（WebShop）、文本具身（ALFWorld）、多跳搜索问答、软件工程（修真实 GitHub issue）、多步数据分析。模型必须学会**何时调用工具、调用什么、如何根据反馈修正、何时停止**。

> **直觉类比：** 单轮 RLVR 像训练学生**闭卷考一张试卷**——交卷即出分；Agentic RL 像训练研究员**完成一个真实课题**——要查资料、做实验、失败重来、管理预算，几周后才知道课题成不成。后者的难点从"解题"变成了"在不确定环境里持续做正确决策"。

### 1.2 最重要的行业信号

正如知乎高赞综述（Snail，266 赞）所总结：过去一年各大模型公司技术报告给出的最重要信号，**"不是又出现了一个更好的 PPO/GRPO 变体"，而是真正有效的 Agentic RL 已经从单轮文本优化，转向长上下文、工具调用、部分可观测、异步执行环境中的系统性策略学习。** Agentic RL 不是单一算法，而是环境建模、学习信号、异步数据流、策略优化和基础设施协同的整套系统。

### 1.3 本报告范围

聚焦"多轮、带工具/环境交互"的强化学习（区别于本库已收录的单轮《大模型强化学习算法》报告——那篇讲 PPO/GRPO 数学原理，这篇讲它们进入多轮环境后为什么不够用、业界如何修补与重构）。

---

## 2. 问题定义：Agentic RL 与单轮 RL 的五个本质差异

**导读：** 用 MDP 语言讲清多轮智能体训练的形式化对象，五个差异直接决定了后续所有算法与系统设计。

### 2.1 轨迹（Trajectory）成为优化单位

一条 agent 轨迹包含状态、思考、动作、环境反馈的完整序列：

```
τ = (s_0, a_0, r_0^env, s_1, a_1, r_1^env, ..., s_T, a_T, R_final)
```

其中动作 a_t 既可能是**推理 token**（think），也可能是**工具调用**（search/code/click…）。环境返回的 observation 成为下一步状态的一部分。优化目标是最大化整条轨迹的期望回报，而非单个回答的奖励。

### 2.2 五个本质差异

| 维度 | 单轮 RLVR | Agentic RL |
|---|---|---|
| 回合结构 | 一问一答，单轮 | 多轮观察–思考–行动–反馈 |
| 奖励 | 答案对错，回合末给出 | **极度稀疏且延迟**，常只有最终成功/失败 |
| 环境 | 无（验证器是纯函数） | 有状态、有随机性、可能不可逆（真实 API、文件系统） |
| 轨迹长度 | 数百到数千 token | 数千到数万 token，rollout 耗时从秒级到分钟/小时级 |
| 数据分布 | 采样即训练，近似 on-policy | 长 rollout 期间策略已更新多轮，**天然 off-policy / stale** |

### 2.3 形式化：部分可观测的多轮决策

严格说，多数 agent 环境是**部分可观测马尔可夫决策过程（POMDP）**：模型只能看到 observation history，看不到环境完整状态（如搜索引擎索引、远端机器状态）。策略输入是历史的拼接：

```
a_t ~ πθ( · | s_0, a_0, o_0, ..., s_t )
```

策略梯度定理形式不变，但估计它的每一步都变难了：

- **方差爆炸**：一条轨迹只有一个 0/1 信号，却要归因给上万个 token；
- **环境方差混入**：同一轨迹换次搜索结果可能就不同，奖励噪声不全部来自策略；
- **信用分配（Credit Assignment）**：成功到底是因为第三步选对了工具，还是第七步的代码写对了？

---

## 3. 核心挑战：多轮环境下的七类病理

**导读：** 把 GRPO 直接搬到多轮 agent 环境会撞上一连串新失败模式。本章按"信号、优化、行为"三类梳理七种已被论文命名的病理。

### 3.1 稀疏奖励与长程信用分配

Agent 生成上千 token、执行多次环境交互，最后只拿到一个成功/失败信号。朴素做法把同一个轨迹级优势平摊给所有 token，等于"成功了就奖励全部动作、失败了就惩罚全部动作"——其中大量无关/错误动作得不到正确区分。这是 GiGPO、SPA-RL 等**细粒度信用分配**工作的出发点（见 4.2）。

### 3.2 回声陷阱（Echo Trap）与训练崩溃

RAGEN（arXiv:2504.20073）在多个风格化环境中发现一种反复出现的失败模式：**奖励方差骤降（reward variance cliffs）伴随梯度尖峰（gradient spikes）**，训练成功率突然崩掉。其机理是多轮训练中策略与环境形成正反馈回路——模型输出某种容易触发环境固定回应的模式，后续 rollout 不断"回声"自我强化，有效批大小坍缩、优势失去区分度。

T²PO（ICML 2026 Spotlight，arXiv:2605.02178）同样报告：即使强 baseline，在不同随机种子下也可能出现 success rate 下降、KL 散度和梯度范数同时爆炸，最终 training collapse。

### 3.3 模板坍塌 / 推理坍塌（Template / Reasoning Collapse）

RAGEN-2（arXiv:2604.06268）命名了一种更隐蔽的病理 **template collapse**：模型表面上推理格式多样，实际上依赖**与输入无关的固定模板**——看起来每轮都在"思考"，思考内容却不随问题变化。论文提出基于互信息的诊断框架来识别这种"伪多样性"。

### 3.4 犹豫与无效轮次：预算空转

T²PO 的核心观点被作者概括为"犹豫就会败北"：多轮交互中模型持续生成**低信息量内容**——既不降低不确定性，也不推进任务进展，却消耗 rollout 预算、放大训练噪声。SimpleTIR（arXiv:2509.02479）则直接识别并过滤包含**无效轮次（ineffective turns）**的轨迹来稳定训练。

### 3.5 Reasoning 与 Tool-use 的梯度冲突

人大与字节的研究（arXiv:2602.00994）用 LEAS 指标量化发现：**推理 token 与工具调用 token 的梯度方向会"打架"**。推理希望参数支持深入逻辑推导，工具使用希望参数支持准确格式化与工具选择，在共享参数空间里互相干扰，联合训练的隐含假设几乎从未被验证过。

### 3.6 交互坍塌（Interaction Collapse）

ASTER（arXiv:2602.01204）发现扩展工具集成推理时，模型会出现 **interaction collapse**：随着训练进行，工具调用频率不升反降，模型退回"纯思考"的舒适区——因为短交互轨迹方差小、更容易拿稳定梯度。论文要求用**高交互密度轨迹做冷启动**来避免。

### 3.7 分布偏移：Staleness 与 Train-Serving Mismatch

分钟级甚至小时级的 rollout 意味着：一条轨迹被采完时，采样它的旧策略 π_old 已落后当前策略许多个更新版本。直接用重要性采样比率修正，ratio 会极端化、clip 后梯度归零或爆掉；此外工具环境本身在变（搜索索引、网页结构），造成训练分布与真实部署分布的偏移（train-serving mismatch）。

### 3.8 病理总表

| 病理 | 现象 | 出处 |
|---|---|---|
| 长程信用分配难 | 轨迹级 0/1 信号平摊给上万 token | GiGPO / SPA-RL 的动机 |
| 回声陷阱 | 奖励方差悬崖 + 梯度尖峰，训练崩 | RAGEN |
| 模板坍塌 | 推理看似多样，实则与输入无关 | RAGEN-2 |
| 犹豫/无效轮次 | 低信息 token 空转 rollout 预算 | T²PO / SimpleTIR |
| 想–做梯度冲突 | reasoning 与 tool-use 梯度方向相反 | arXiv:2602.00994 (DART) |
| 交互坍塌 | 训练中工具越用越少，退回纯思考 | ASTER |
| 分布偏移 | stale policy、环境漂移 | K1.5 / 各异步系统 |

---

## 4. 算法方案谱系

**导读：** 围绕上述病理，2025–2026 年的算法工作可归为四族：轨迹级框架、细粒度信用分配、探索/稳定控制、参数解耦。

### 4.1 轨迹级策略优化：StarPO

RAGEN 提出 **StarPO（State-Thinking-Actions-Reward Policy Optimization）**，把整条状态–思考–动作–奖励轨迹作为优化单位，是多轮 agent RL 的通用框架；其稳定变体 **StarPO-S** 在轨迹层面加入稳定化机制，抑制回声陷阱中的梯度尖峰。它确立了"agent RL 的 loss 对象是 trajectory 而非单条 response"这一基本范式。

### 4.2 细粒度信用分配

#### GiGPO：组中组（Group-in-Group）

GiGPO（arXiv:2505.10978）保留 GRPO 无 critic、低显存的优点，但引入**两层组结构**实现步骤级信用分配：

```
外层组：同一 prompt/task 采样多条完整轨迹（像 GRPO）
内层组：在每个时间步 t，跨多条轨迹收集"同一状态下"的动作
        → 用步级组内相对量估计该步的细粒度优势
```

直觉：评估"第三步这步操作好不好"，不看整条轨迹最终成败，而看**在相同状态步上，这步动作相对于其他轨迹同步动作的表现**。论文一手摘要称其在保持 critic-free、低内存、稳定收敛的同时，实现了 LLM agent 的细粒度信用分配。

#### SPA-RL：逐步进展归因

SPA-RL（arXiv:2505.20732）提出通用的**奖励再分配**框架：把最终奖励分解为反映每一步**增量贡献**的中间奖励（stepwise progress attribution），让多步交互获得密集且有意义的指导信号，而非等最终结果。

### 4.3 探索控制与训练稳定

#### ARPO：基于熵的自适应采样

ARPO（arXiv:2507.19849）观察到：LLM 在多轮工具任务中会表现出**高度不确定性行为**（熵激增），现有 RL 算法不能平衡内在长程推理与多轮工具交互。它通过**基于熵的自适应采样机制**在"全局推理"与"局部工具交互"之间动态分配采样资源，论文报告**仅用一半工具调用预算**即取得更优性能。

#### T²PO：不确定性引导的探索控制

T²PO（arXiv:2605.02178，ICML 2026 Spotlight）针对多轮训练崩溃，核心思想是**监控 agent 在每一步的不确定性**，抑制"犹豫"式低信息生成：当某步既不降低不确定性、也不推进任务时，减少在这类行为上的 rollout 与梯度投入，把预算导向有信息量的思考与动作，从而稳定多轮训练。

#### SimpleTIR：过滤无效轮次

SimpleTIR（arXiv:2509.02479）是即插即用方案：识别并**丢弃包含无效轮次的轨迹**，避免"什么都没推进"的噪声片段污染梯度，同时鼓励模型发现多样的推理模式。

### 4.4 参数解耦：DART 双 LoRA

针对 reasoning 与 tool-use 的梯度冲突（3.5 节），DART（arXiv:2602.00994）给出简洁解法：**给推理和工具使用各配一个独立 LoRA 适配器**，两条梯度路径互不干扰，用单模型（共享底座 + 双适配器）达到"推理模型 + 工具模型"双模型系统的效果。这是"梯度冲突 → 参数空间解耦"思路的代表。

### 4.5 工具学习的早期算法脉络（2025 上半年）

在上述系统级工作之前，一批工作先解决了"模型能不能学会用工具"：

| 工作 | arXiv | 贡献 |
|---|---|---|
| ToRA | 2309.17452 | 自然语言推理与计算库/符号求解器结合的早期范式 |
| TORL | 2503.23383 | 无监督数据下用 RL 自主探索工具使用策略 |
| ReTool | 2504.11536 | RL 学会在推理中动态穿插实时代码执行 |
| ToolRL | 2504.13958 | 系统研究工具选择/应用的奖励设计原则 |
| OTC | 2504.14870 | 鼓励最少调用次数下答对，优化工具使用成本 |
| ZeroSearch | 2505.04588 | 用"模拟搜索"训练真实搜索能力，规避引擎成本与文档质量问题 |

### 4.6 算法对比总表

| 算法 | 针对病理 | 核心机制 | 是否需额外模型 |
|---|---|---|---|
| StarPO/StarPO-S | 回声陷阱 | 轨迹级优化 + 稳定化 | 否 |
| GiGPO | 步骤信用分配 | 两层组，步级相对优势 | 否（critic-free） |
| SPA-RL | 稀疏奖励 | 最终奖励再分配为步进展奖励 | 否 |
| ARPO | 推理/工具失衡 | 熵自适应采样 | 否 |
| T²PO | 训练崩溃、犹豫 | 不确定性引导的探索控制 | 否 |
| SimpleTIR | 无效轮次 | 轨迹过滤 | 否 |
| DART | 梯度冲突 | 推理/工具双 LoRA | 双适配器 |

---

## 5. 学习信号：环境、奖励与课程

**导读：** Agentic RL 的效果上限由学习信号决定。本章讲环境从哪来、奖励怎么设计、数据课程怎么组织。

### 5.1 可验证环境的规模化

RLVR 在单轮靠答案匹配；agent 时代需要**可执行、可验证、可大规模并行**的环境：

- **软件工程环境**：真实仓库 + issue + 测试套件，补丁是否通过隐藏测试即奖励。SWE-Universe（arXiv:2602.02361）把真实可验证环境扩展到百万量级；
- **工具沙箱**：代码解释器、浏览器、终端沙箱，动作真实执行、结果可判；
- **模拟环境**：ZeroSearch（arXiv:2505.04588）代表的思路——训练时不真的调搜索引擎，而用模型模拟搜索返回，规避成本和不可控文档，再把能力迁移到真实工具；
- **风格化环境**：RAGEN 使用的 WebShop、ALFWorld 等，用于受控研究多轮动力学。

### 5.2 奖励设计的层次

```
结果奖励：任务成功（0/1）、测试通过 —— 最稀疏但最客观
过程/格式奖励：工具调用格式正确、参数合法（如 Nemotron-Tool-N1 仅用二元格式+功能奖励即超越 GPT-4o 工具表现）
效率奖励：调用次数/预算（OTC）、token 经济性
进展奖励：SPA-RL 式的步级增量归因
规则组合：R1 风格的正确性 + 格式 + 语言一致性
```

ToolRL（arXiv:2504.13958）的结论是"Reward is All Tool Learning Needs"——**好的奖励原则本身比复杂算法更关键**。

### 5.3 冷启动与课程：避免交互坍塌

纯基座直接做 agent RL 容易学不动或学歪，标准做法是先 SFT 冷启动、再 RL，且冷启动数据的"形状"很关键：

- ASTER 要求冷启动轨迹具有**高交互密度**，防止模型一开始就养成不调工具的习惯；
- ICLR 2026 的 lightweight SFT 研究（OpenReview:yezWGJmODg）指出，让模型"为 RL 做好准备"只需轻量 SFT，过度 SFT 反而压缩探索空间；
- Kimi K2 则用**大规模 agentic 数据合成管线**自动构造带工具交互的训练轨迹。

---

## 6. 基础设施：异步数据流与大规模训练系统

**导读：** 当 rollout 从秒级拉长到分钟/小时级，瓶颈不再是 loss 设计，而是吞吐、稳定性与 agent 灵活性的三难权衡。本章讲 Agentic RL 区别于传统 LLM RL 的系统构造。

### 6.1 三难权衡

MiniMax 在 Forge 框架中把核心矛盾总结为三者不可兼得：

```
吞吐（rollout 与训练的 GPU 利用率）
  × 稳定性（staleness 控制、梯度质量）
    × agent 灵活性（任意长轨迹、任意工具、动态停止）
```

例如：等所有轨迹采完再训（batch mode）稳定但 GPU 大量闲置；边采边训（async）吞吐高但 stale 严重。

### 6.2 关键系统技术

1. **Partial Rollout Reuse（部分轨迹重用，Kimi k1.5，arXiv:2501.12599）。** 策略更新后不必丢弃未完成的长轨迹，而是配合 mirror-descent 风格的策略优化重用已采片段，大幅摊薄长 rollout 成本。K1.5 的摘要明确把**长上下文扩展 + 改进的策略优化（含 partial rollout）**列为关键要素，且不依赖 MCTS、价值函数等复杂技术。
2. **异步调度与 Actor–Rollout 分离。** 训练 actor 与生成 rollout 的引擎（常为 vLLM 类高并发推理）解耦运行，SkyRL-Agent（arXiv:2511.16108）主打异步调度、轻量工具集成和后端互操作。
3. **蜂群式弹性伸缩。** 通义 AgentJet 提出"灵活训练、动态伸缩"的分布式 Agentic RL 蜂群框架：rollout 负载随任务难度波动时弹性扩缩生成资源。
4. **Agent Swarm / PARL（Kimi K2，arXiv:2507.20534）。** K2（1T 总参 / 32B 激活 MoE）在后训练的联合 RL 阶段使用大规模智能体集群产生交互数据；其摘要还披露 MuonClip 优化器（Muon + QK-clip）解决训练不稳定，15.5 万亿 token 预训练零 loss spike。
5. **Staleness 控制。** 限制 rollout 可落后的版本数、对过旧轨迹降权或丢弃、token 级 clipping 控制重要性比率极端化。

### 6.3 典型异步闭环

```
任务池 ──→ Rollout 集群（vLLM + 工具沙箱，弹性伸缩）
                    │  完整/部分轨迹 + 奖励
                    ▼
            轨迹缓冲区（按策略版本/质量分桶，staleness 过滤）
                    │
                    ▼
            Trainer（异步取批，GiGPO/GRPO 类优化，token 级 clip）
                    │ 新权重
                    └──→ 回流 Rollout 集群（partial rollout 续采）
```

---

## 7. 前沿系统实战：Kimi、MiniMax、GLM 路线对比

**导读：** 三家技术报告代表了 Agentic RL 工程化的三种组织方式。

| 系统 | 核心打法 | 代表性技术 |
|---|---|---|
| **Kimi k1.5/K2/K2.5** | 简洁有效的 RL 框架 + 大规模 agentic 数据 | partial rollout、长上下文 RL、MuonClip、agentic 数据合成、Agent Swarm、token-level clipping、GRM rubric（K2.5 扩展到视觉 agent） |
| **MiniMax M1 / Forge** | 效率优先的 test-time 扩展 + 可扩展 agent RL 框架 | Lightning Attention 降长程成本、Forge 框架直面吞吐×稳定×灵活三难 |
| **GLM-5** | 阶段化 RL pipeline | Reasoning RL → Agentic RL → General RL **顺序推进**而非一次混训；异步 RL 基础设施 + 跨阶段蒸馏，兼顾长时程 agent 学习与通用能力保持 |

**阶段化 RL（GLM 路线）值得特别注意**：它反对"reasoning、tool-use、general 能力混在一个 RL 阶段训完"，理由与 3.5 节的梯度冲突一致——先把纯推理 RL 做扎实，再进入 agentic RL 学工具交互，最后做 general RL 对齐通用偏好，阶段之间用蒸馏传递能力。这与 DART 的参数解耦是同一矛盾的两种解法（时序解耦 vs 结构解耦）。

---

## 8. 统一视角：三大不变量与工程 Checklist

**导读：** 用知乎综述（WillWay/Snail）总结的三大不变量收束全文，并给出可执行 checklist。

### 8.1 三大不变量

无论算法和系统怎么变，有效的 Agentic RL 必须始终维持：

1. **探索不塌缩。** 策略可探索空间不能过早收敛——防止模板坍塌、交互坍塌、多样性丧失；高交互密度冷启动、熵监控、自适应采样（ARPO）都是保探索的手段。
2. **梯度非退化。** 持续制造"优势不坍缩"的训练批次——全成功/全失败的轨迹组梯度为零，需要动态采样、无效轨迹过滤、细粒度信用分配（GiGPO/SPA-RL）保证每批都有学习信号。
3. **分布可控。** 控制参数更新与真实部署之间的分布偏移——用 partial rollout、staleness 过滤、token 级 clip、版本管理应对长轨迹与异步执行引入的 off-policy drift 和 train-serving mismatch。

### 8.2 工程选型 Checklist

- ☐ 先问任务是否真需要多轮：单轮 RLVR 能解的不要上 agentic RL（系统复杂度差一个量级）。
- ☐ 环境三要素：可大规模并行、动作可执行可回滚、结果可客观验证；先建环境再谈算法。
- ☐ 冷启动数据保证工具交互密度，SFT 宁轻勿重，给 RL 留探索空间。
- ☐ 信用分配：长程任务优先试 **GiGPO**（步级组优势）或 **SPA-RL**（进展归因），不要把轨迹 0/1 平摊。
- ☐ 训练不稳先查信号：监控每批奖励方差、熵、工具调用频率、有效轮次占比、梯度范数、KL、轨迹多样性。
- ☐ 出现崩溃对症处理：回声/梯度尖峰→StarPO-S/T²PO；犹豫空转→SimpleTIR/T²PO；工具越用越少→高交互冷启动（ASTER）；想做冲突→DART 双 LoRA 或阶段化 RL。
- ☐ 基础设施上 actor/rollout 分离 + 异步调度 + partial rollout 复用，并设置 staleness 上限。
- ☐ 效率约束进奖励：调用次数/token 预算（OTC、ARPO 思路），避免模型学会"大力出奇迹"。
- ☐ 评估必须分能力：任务成功率之外，单独评估推理能力与通用能力是否在 agentic RL 后退化。

### 8.3 注意事项与诚实性声明

- 本报告 16 个 arXiv 编号（2309.17452、2501.12599、2503.23383、2504.11536、2504.13958、2504.14870、2504.20073、2505.04588、2505.10978、2505.20732、2507.19849、2509.02479、2511.16108、2507.20534、2506.13585、2602.01204、2602.00994、2604.06268、2605.02178）均经 **arXiv 官方 API 逐一核实**；RAGEN、ARPO、GiGPO、K1.5、K2 的关键结论摘自论文摘要原文。
- Kimi K2.5、GLM-5、Forge、AgentJet、SWE-Universe 等 2026 年系统的技术细节来自知乎技术综述转述（文章中给出了 arXiv/官网链接），引用到正式材料前请核对一手技术报告。
- 该方向迭代极快（月级），各系统的具体超参与模块名称以官方最新报告为准。

---

## 9. 参考资料

### 9.1 原始论文（arXiv 编号经官方 API 核实）

1. Gou et al. **ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving.** 2023. https://arxiv.org/abs/2309.17452
2. Kimi Team. **Kimi k1.5: Scaling Reinforcement Learning with LLMs.** 2025. https://arxiv.org/abs/2501.12599
3. **TORL: Scaling Tool-Integrated RL.** 2025. https://arxiv.org/abs/2503.23383
4. **ReTool: Reinforcement Learning for Strategic Tool Use in LLMs.** 2025. https://arxiv.org/abs/2504.11536
5. **ToolRL: Reward is All Tool Learning Needs.** 2025. https://arxiv.org/abs/2504.13958
6. **OTC: Optimal Tool Calls via Reinforcement Learning.** 2025. https://arxiv.org/abs/2504.14870
7. Zhou et al. **RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn RL（StarPO/回声陷阱）.** 2025. https://arxiv.org/abs/2504.20073
8. **ZeroSearch: Incentivize the Search Capability of LLMs without Searching.** 2025. https://arxiv.org/abs/2505.04588
9. **GiGPO: Group-in-Group Policy Optimization for LLM Agent Training.** 2025. https://arxiv.org/abs/2505.10978
10. **SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution.** 2025. https://arxiv.org/abs/2505.20732
11. **ARPO: Agentic Reinforced Policy Optimization.** 2025. https://arxiv.org/abs/2507.19849
12. **SimpleTIR: End-to-End RL for Multi-Turn Tool-Integrated Reasoning.** 2025. https://arxiv.org/abs/2509.02479
13. **SkyRL-Agent: Efficient RL Training for Multi-turn LLM Agent.** 2025. https://arxiv.org/abs/2511.16108
14. Kimi Team. **Kimi K2: Open Agentic Intelligence（MuonClip / Agent Swarm）.** 2025. https://arxiv.org/abs/2507.20534
15. MiniMax. **MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention.** 2025. https://arxiv.org/abs/2506.13585
16. **ASTER: Agentic Scaling with Tool-integrated Extended Reasoning（交互坍塌）.** 2026. https://arxiv.org/abs/2602.01204
17. Li et al. **Reasoning and Tool-use Compete in Agentic RL（DART 双 LoRA）.** 2026. https://arxiv.org/abs/2602.00994
18. **RAGEN-2: Reasoning Collapse in Agentic RL（模板坍塌）.** 2026. https://arxiv.org/abs/2604.06268
19. **T²PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic RL.** ICML 2026 Spotlight. https://arxiv.org/abs/2605.02178

### 9.2 知乎综述与解读（社区信源，交叉验证用）

1. Snail. **Agentic RL 训练：环境建模、学习信号、异步数据流、策略优化和基础设施的协同.**（266 赞）
   https://zhuanlan.zhihu.com/p/2014843795380904030
2. 虚怀若谷. **Agentic Post-training 技术调研（论文清单）.**（172 赞）
   https://zhuanlan.zhihu.com/p/2029952170645259049
3. WillWay. **破解 Agentic RL：三大不变量与八个关键维度.**
   https://zhuanlan.zhihu.com/p/2021149018131046968
4. Wang.hx. **ICML Spotlight｜T²PO：用不确定性引导稳定多轮 Agentic RL.**（87 赞）
   https://zhuanlan.zhihu.com/p/2034956125624381457
5. JustBeClaw. **Agentic RL 最新认知：Reasoning 和 Tool-use 的优化方向会打架.**（64 赞）
   https://zhuanlan.zhihu.com/p/2024962637880934688
6. DIREWOLF. **通义 AgentJet：分布式 Agentic RL 蜂群训练框架.**（65 赞）
7. sanmuyang. **LLM RL 2026 论文（九）SEAL.**

---

*报告完。核心事实经一手论文与 arXiv API 核实，2026 年系统细节以官方技术报告为准。*
