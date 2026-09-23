# Self-Evolving Agents × Harness-Native RL 技术调研报告

> 主题：当 Agent 的进化不再只发生在「模型权重」里，而是同时发生在「模型 + Harness」两个空间——系统梳理 Self-Evolving Agents（自进化智能体）与 Harness-Native RL（在真实脚手架上做强化学习）两条线索在 2026 年的合流：为什么必须在真实 Harness 上训练、双空间协同进化怎么搭、代表性系统（TaoLive HAT/HSA、HarnessX、SkillOpt、RRSI）的工程细节、实验数据、Misevolution 风险与落地 checklist。
>
> 调研日期：2026-09-23 ｜ 一手信源：6 篇核心论文摘要已从 arXiv 逐篇核实（含 2026-09-21 刚挂出的 RRSI）+ 知乎深度解读 20+ 篇

---

## 目录

1. [背景与问题定义](#1-背景与问题定义)
   - [1.1 一个新共识：Agent 的进化有两个空间](#11-一个新共识agent-的进化有两个空间)
   - [1.2 为什么自进化离不开 Harness-Native RL](#12-为什么自进化离不开-harness-native-rl)
   - [1.3 旧路线的三个缺陷](#13-旧路线的三个缺陷)
   - [1.4 直觉类比](#14-直觉类比)
2. [原理详解](#2-原理详解)
   - [2.1 自进化 Agent 的统一反馈环](#21-自进化-agent-的统一反馈环)
   - [2.2 What / When / How：自进化三问](#22-what--when--how自进化三问)
   - [2.3 双基质：权重参数空间 vs Harness 符号空间](#23-双基质权重参数空间-vs-harness-符号空间)
   - [2.4 Harness-Native RL 为何是自进化的跑道](#24-harness-native-rl-为何是自进化的跑道)
   - [2.5 关键消歧：反思、记忆、技能、RL 的边界](#25-关键消歧反思记忆技能rl-的边界)
3. [方案设计：四个代表系统](#3-方案设计四个代表系统)
   - [3.1 TaoLive HAT/HSA：让小模型适应变化的 Harness](#31-taolive-hathsa让小模型适应变化的-harness)
   - [3.2 HarnessX：可组合、可演化的 Harness 工厂](#32-harnessx可组合可演化的-harness-工厂)
   - [3.3 SkillOpt：把技能文档当成可训练参数](#33-skillopt把技能文档当成可训练参数)
   - [3.4 RRSI：给递归自改进套上正则化](#34-rrsi给递归自改进套上正则化)
   - [3.5 系统对比与实验数据表](#35-系统对比与实验数据表)
   - [3.6 方法演进谱系](#36-方法演进谱系)
4. [核心实现](#4-核心实现)
   - [4.1 HSA 五维扰动与三阶段训练](#41-hsa-五维扰动与三阶段训练)
   - [4.2 Cross-Harness GRPO](#42-cross-harness-grpo)
   - [4.3 SkillOpt 文本空间优化器伪代码](#43-skillopt-文本空间优化器伪代码)
   - [4.4 Misevolution：自进化的失控面](#44-misevolution自进化的失控面)
5. [结论与选型建议](#5-结论与选型建议)
6. [参考资料](#6-参考资料)

---

## 1. 背景与问题定义

### 1.1 一个新共识：Agent 的进化有两个空间

经过 2025–2026 年的演进，业界形成了一个清晰共识：**Agent = Model（模型权重）× Harness（运行时脚手架）**。而 Agent 的自我进化，自然也有两个可优化的空间：

| 进化空间 | 载体 | 更新方式 | 擅长的改进 |
|---|---|---|---|
| **参数空间**（model-centric） | 模型权重 θ | RL 梯度更新（GRPO / GSPO…） | 细粒度行为：何时调哪个工具、如何措辞、何时终止 |
| **符号空间**（environment-driven） | Harness 配置 η：提示词、技能、工具 schema、控制流、记忆、Hook | 离散结构编辑：增删替换组件 | 粗粒度策略架构：加一个工具、换一个处理器、重组提示结构 |

2025 年 8 月的综述《A Comprehensive Survey of Self-Evolving AI Agents》（arXiv:2508.07407）首次用统一反馈环把「静态基座模型」与「终身 Agent 系统」桥接起来；2026 年 2 月起的后续综述进一步把趋势概括为 **「From Model-Centric to Environment-Driven Co-Evolution（从以模型为中心到环境驱动的协同进化）」**——Agent 与它所处的环境/Harness 共同演化，而不是只调权重。

### 1.2 为什么自进化离不开 Harness-Native RL

此前的自进化工作（Reflexion 式反思、经验记忆、技能库积累）大多默认一个前提：**Harness 是固定的背景**。但这带来两个根本问题：

1. **训练分布与进化分布割裂**：进化信号来自真实运行，但如果模型训练仍在简化环境/Mock 工具上做（传统 Agent-RL），模型在真实 Harness 里产生的轨迹无法直接转化为有效的梯度信号——它在「假环境」里学到的策略，撑不起「真环境」里的进化。
2. **Harness 的离散进化与模型的参数进化是两套独立循环**：Harness 演化时产生的千万级 token 轨迹被丢弃；模型升级后，旧 Harness 又常常失效，工程师得重搭一遍脚手架。

**Harness-Native RL 提供的正是合流点**：让 RL 直接发生在真实（或高保真增强的）Harness 上，执行轨迹（trace）同时成为两种信号——Harness 的离散更新信号 + 模型的梯度训练信号。HarnessX（arXiv:2606.14249）把这条闭环正式命名为 **Cross-Harness GRPO**；TaoLive（arXiv:2608.15763）则在淘宝直播数字人业务里验证了它的工业可行性。

### 1.3 旧路线的三个缺陷

| 旧路线 | 缺陷 | 后果 |
|---|---|---|
| **固定 Harness SFT** | 模型只认识一个固定版本的脚手架 | 换 Harness 配置即退化；SFT 还会损伤通用能力（TaoLive 实测 IFEval 掉 7.7 分） |
| **传统 Agent-RL（简化环境）** | 训练时剥离真实工具/Hook/多轮状态 | 训推偏移；真实故障恢复（超时、权限拒绝、重试）学不到 |
| **松散自修订 / 手动改技能** | 无步长控制、无独立验证集、无负反馈记忆 | 技能越改越长、越改越偏；一次「看起来合理」的重写可能让真实指标倒退 |

### 1.4 直觉类比

- **模型是车手，Harness 是赛车，Harness-Native RL 是真实赛道**：在模拟器（简化环境）里练出的车手，上真实赛道会不适应；自进化要有效，必须让车手、赛车在真实赛道上一起磨合、一起升级。
- **双空间进化 = 换赛车 + 练车技交替进行**：改 Harness 是「换零件、调空气动力学」（离散、粗粒度，梯度表达不了）；模型 RL 是「练走线、练刹车点」（连续、细粒度，符号规则捕捉不了）。只练车不换车，或只换车不练车，都到不了最快圈速。

---

## 2. 原理详解

### 2.1 自进化 Agent 的统一反馈环

综合 arXiv:2508.07407 的框架，一个自进化系统的最小闭环可以写成：

```text
             ┌─────────────────────────────────────────────┐
             │                Agent System                 │
  task ──▶ 感知 ──▶ 推理/规划 ──▶ 行动（工具/Hook/技能）   │
             ▲                               │             │
             │                               ▼             │
        更新后的状态 ◀── 评估（任务结果/验证器/环境反馈）  │
             │                                             │
             └──▶ 进化信号分两路：                         │
                  (a) 参数空间：RL 更新 θ                  │
                  (b) 符号空间：编辑 η（Harness/Skill）    │
```

形式化地：

```text
给定状态 s_t，Agent 在 Harness η 下执行：
    a_t ~ π_θ( · | s_t, η )              # 注意 η 是条件输入
    r_t = Env.step(a_t)                  # 真实环境结果

每个进化周期后：
    θ' ← θ + RL_update( trajectories on η )       # 参数空间
    η' ← Edit( η, proposal_from(traces) )         # 符号空间，需经验证门控
    s_{t+1} ← Memory.write(s_t, trace, reflection)
```

关键点：**策略 π 以 Harness η 为条件（condition），而不是记住某个固定 η**。这正是 Harness-Aware Training 的核心目标。

### 2.2 What / When / How：自进化三问

自进化系列综述围绕三个问题组织全部技术：

1. **What（进化什么）**：模型权重、记忆、技能、工具、提示词、控制流（工作流）、甚至优化器本身。
2. **When（何时进化）**：每轮在线更新 vs 周期性离线进化；失败触发 vs 成功轨迹蒸馏；关键约束是更新不能破坏已有能力（防灾难性遗忘）。
3. **How（如何进化）**：反馈从哪来（任务真实结果 / 独立验证器 / 模型自评）、经验如何筛选、更新如何门控（验证集严格提升才接受）。

### 2.3 双基质：权重参数空间 vs Harness 符号空间

「为什么需要两个空间而不是一个」，HarnessX 给出的论证可以归纳成一张分工表：

| 维度 | Harness 侧（非参数，符号） | Model 侧（参数，GRPO） |
|---|---|---|
| 更新类型 | 离散结构变更（加工具、换 processor、重组 prompt） | 细粒度行为调整（何时调工具、措辞、终止判断） |
| 数据组织 | 含多模型 checkpoint 轨迹的 buffer | cross-harness grouping 后分组计算优势 |
| 表达边界 | 参数更新表达不了的架构变化 | 符号规范捕捉不了的高维 in-context 状态 |
| 角色 | 定义粗粒度策略架构 | 学会利用这套架构 |

两者互补：**Harness 决定能力释放率，模型决定能力上限。**

### 2.4 Harness-Native RL 为何是自进化的跑道

Harness-Native RL 对自进化的支撑作用体现在三点：

1. **信号保真**：只有在真实/高保真 Harness 上 rollout，trace 里才包含真实的工具失败、Hook 拦截、多轮状态转移——这些才是驱动双空间进化的有效信号。
2. **随机化覆盖**：在真实 Harness 之上做受控随机化（HSA），让训练分布覆盖未来部署会遇到的 Harness 变体，模型学的是「理解当前 Harness」而非「背下固定版本」。
3. **双信号同源**：同一条轨迹既算 RL 优势（模型信号），又被进化引擎读取提出 Harness 编辑（符号信号），两个优化方向天然对齐，不会互相打架。

### 2.5 关键消歧：反思、记忆、技能、RL 的边界

- **Reflexion（反思）**：失败后用自然语言总结原因——它是信号产生机制，本身不是完整进化；反思不落盘、不被后续使用则无效。
- **Memory（记忆）**：跨会话存储经验与状态的载体；记忆可以被污染（见 §4.4）。
- **Skill（技能）**：结构化、可复用的外部知识文档（SKILL.md）；技能是 Harness 中最常被当作「可训练外部参数」的组件（SkillOpt）。
- **RL（强化学习）**：在（高保真）环境中用结果信号更新权重——它是参数空间的进化机制；与符号空间编辑组合才构成完整协同进化。

---

## 3. 方案设计：四个代表系统

### 3.1 TaoLive HAT/HSA：让小模型适应变化的 Harness

论文《Training Agents to Evolve with Their Harness: TaoLive Digital Avatar Agent Technical Report》（arXiv:2608.15763，淘宝直播数字人，2026-08）直面一个业务矛盾：

- 大模型 zero-shot 能适应可进化 Harness，但**延迟不达标**；
- 小模型延迟达标，但会**过拟合到固定 Harness 配置**，Harness 一更新性能就掉。

解法 **HAT（Harness-Aware Training）** 的核心组件 **HSA（Harness-State Augmentation，脚手架状态增强）**：在保持任务语义不变的前提下，对 5 类 Harness 元素施加随机变换（细节见 §4.1），让模型无法依赖任何「固定写法」捷径。训练分三阶段：HSA-SFT → General On-Policy Distillation → HSA-RL（§4.1）。

### 3.2 HarnessX：可组合、可演化的 Harness 工厂

论文《HarnessX: A Composable, Adaptive, and Evolvable Agent Harness Foundry》（arXiv:2606.14249，小米 Darwin Agent Team，2026-06）针对现有 Harness 三大缺陷（手工静态、架构耦合、与模型训练割裂）给出三层架构：

1. **组合层 Composition**：把 Harness 拆成类型化原语（typed primitives），通过替换代数（substitution algebra）像搭积木一样组装，组件可独立替换。
2. **适应层 Adaptation — AEGIS**：一个基于执行轨迹的多智能体演化引擎，在「符号适配 ↔ 强化学习」的操作对应（operational mirror）上提出 Harness 编辑。
3. **协同进化层 Co-Evolution — Cross-Harness GRPO**：同一批轨迹同时驱动 Harness 更新与模型 GRPO 训练。

### 3.3 SkillOpt：把技能文档当成可训练参数

论文《SkillOpt: Executive Strategy for Self-Evolving Agent Skills》（arXiv:2605.23904，微软 + 上海交大 + 同济 + 复旦，2026-05）主张：**技能应该像冻结 Agent 的外部状态一样被「训练」**，具备权重优化同样的纪律。机制要点：

- 独立的**优化器模型**读取评分后的 rollout，输出对单个技能文档的**有界 ADD / DELETE / REPLACE 编辑**；
- 编辑仅在**严格提高预留验证集分数**时接受（held-out validation gate）；
- 配套**文本学习率预算**（控制每轮编辑量，如 L 从 4 衰减到 2）、**被拒编辑缓冲区**（负反馈记忆）、**epoch 级慢更新 + meta-skill 反思**；
- 部署期零额外模型调用（技能就是一份普通文档）。

值得关注的后续：SkillOpt 第二篇（SkillOpt-Lite）把复杂管道几乎全部拆掉只留极简循环，实测在 6 个 benchmark 上反而收敛更快、上限更高——被解读为「苦涩的教训（Bitter Lesson）」在技能优化领域的应验。

### 3.4 RRSI：给递归自改进套上正则化

论文《RRSI: Regularized Recursive Self-Improvement of Agent Harnesses》（arXiv:2609.24972，2026-09-21 刚挂出，极新）指出：Harness 的递归自改进（Meta-Harness、AHE、TTHE、HarnessX 类方法）会出现经典的**过拟合**——在训练 benchmark 上反复看反馈，最后把 Harness 优化成「专门会做这套题」的程序，分布内增益很大、分布外（OOD）增益缩小甚至消失。

RRSI 把机器学习正则化原则引入 Harness 进化，约束编辑候选的**提议与选择**：提议端使用**随时间退火的编辑预算**（temporally annealed budget），限制单轮可改的组件数；选择端加约束，保证进化不记忆训练任务。

### 3.5 系统对比与实验数据表

核心数字均来自论文摘要（arXiv 原文已核实编号与标题）：

| 系统 | 机构 | 核心机制 | 关键结果 |
|---|---|---|---|
| **TaoLive HAT** | 淘宝直播 | HSA 五维扰动 + 三阶段训练 | Live-Stream QA **94.8**（base 80.3；最强通用 LLM 93.0）；Harness-Variant QA **94.6**（base 75.4）；IFEval **83.5**（固定 Harness SFT 掉 7.7 分）；单卡 H20 P50/P95 **3.4s / 8.1s**；线上 A/B 商品页浏览正向 |
| **HarnessX** | 小米 | 替换代数 + AEGIS + Cross-Harness GRPO | 5 基准（ALFWorld/GAIA/WebShop/τ³-Bench/SWE-bench Verified）平均 **+14.5%**，最高 **+44.0%**；基线越弱增益越大 |
| **SkillOpt** | 微软等 | 有界文本编辑 + 验证门控 + 文本 LR | 技能在反馈下可控收敛，部署期零额外调用；具体分数见论文表格（本报告未逐项转录） |
| **SkillOpt-Lite** | 微软等 | 极简循环 | 6 benchmark 全面包住 SkillOpt 与初始技能，前 2–3 步即领先 |
| **RRSI** | —（2026-09-21） | 退火编辑预算 + 选择约束 | 缓解 ID/OOD 增益落差（具体数值见论文，极新） |

配套相关工作：**RLVE**（arXiv:2511.07317，ICML 2026，自适应可验证环境扩展 RL）、**Who Grades the Grader**（arXiv:2607.12790，评估指标与技能协同进化）、**SEVerA**（arXiv:2603.25111，自进化 Agent 的可验证合成）。

### 3.6 方法演进谱系

| 时间 | 节点 | 意义 |
|---|---|---|
| 2022 | ReAct（arXiv:2210.03629） | Agent Loop 原型 |
| 2023 | Reflexion（arXiv:2303.11366） | 失败反思→记忆，自改进雏形 |
| 2025-08 | 自进化 Agent 综述（arXiv:2508.07407） | 统一反馈环，「自进化」立题 |
| 2025-11 | RLVE（arXiv:2511.07317，ICML 2026） | 自适应可验证环境成为 RL 扩展资产 |
| 2026-02 | 协同进化综述（Model-Centric → Environment-Driven） | 双空间叙事成型 |
| 2026-03 | SEVerA（arXiv:2603.25111） | 自进化的可验证合成 |
| 2026-05 | SkillOpt（arXiv:2605.23904） | 技能文档成为可训练外部参数 |
| 2026-06 | HarnessX（arXiv:2606.14249） | Cross-Harness GRPO 打通双信号 |
| 2026-07 | Who Grades the Grader（arXiv:2607.12790） | 评分者本身也要协同进化 |
| 2026-08 | TaoLive HAT（arXiv:2608.15763） | 工业级部署验证（淘宝直播） |
| 2026-09-21 | RRSI（arXiv:2609.24972） | 自改进正则化，防过拟合（极新） |

---

## 4. 核心实现

### 4.1 HSA 五维扰动与三阶段训练

HSA 五维扰动（保持任务语义，专门破坏模型可能依赖的捷径）：

| 扰动维度 | 变换操作 | 破解的捷径 |
|---|---|---|
| **Skill 标识符** | 注入合成/噪声技能、随机屏蔽、重命名、用强 LLM 重写描述 | 靠固定技能 ID 路由 |
| **Skill 内容** | 规则改写、随机屏蔽、重新排序 | 靠规则位置与原文措辞 |
| **工具定义** | 工具重命名、描述重写（功能不变） | 靠工具名匹配 |
| **System Prompt** | 顶层块/块内条目重排、数值约束扰动（长度、轮数上限，业务合法范围内） | 只认熟悉模板 |
| **Hook** | 修改重试行为、消息结构、模拟 Hook 触发的修正 | 记死重试模式 |

三阶段训练流水线：

```text
Stage 1 · HSA-SFT
  对原始 Harness 应用 HSA 得到多个变体；
  教师模型（强 LLM）在原始及增强 Harness 下生成轨迹；
  按 Accuracy + Effectiveness 过滤后做 SFT（约 10K 真实直播样本）。
  对 hook-triggered retry 轨迹：多数只保留最终正确行为
  （避免学会"先犯错再重试"），少量保留完整重试轨迹以学习错误恢复。

Stage 2 · General On-Policy Distillation
  在生产级实时流模拟器中 on-policy 采样，
  恢复 SFT 阶段损失的通用能力（防 IFEval 类指标回退）。

Stage 3 · HSA-RL
  在增强 Harness 环境中做 RL，
  提升对不断变化的 Harness 的鲁棒性。
```

Stage 2 的「模拟 on-policy 环境」包含四个仿真器：直播流输入模拟（商品咨询/闲聊/购买意向/售后/混合弹幕 + 房间状态）、Harness Agent 调度（Skill 路由/Hook 检查/多轮循环）、工具执行模拟（生产目录快照 + 受控故障注入：超时、格式错误、权限拒绝）、增强 Harness 配置（保证 SFT 与 RL 覆盖相同变化范围）。

### 4.2 Cross-Harness GRPO

HarnessX 协同进化的核心思想（Python 风格伪代码）：

```python
# cross-harness GRPO：一条轨迹，两种进化信号
def cross_harness_grpo(rollouts, model, harness_foundry):
    # rollouts 来自不同 harness 变体，按 harness 配置分组
    groups = group_by_harness(rollouts)

    for harness_cfg, trajs in groups.items():
        # 关键：用该组对应的 harness 设置重新装载模型，再计算新策略概率
        model.load_harness(harness_cfg)   # 工具/提示/记忆截断规则各不同
        logp_new = model.logprob([t.actions for t in trajs])
        logp_old = [t.logp for t in trajs]

        # 重要性采样系数：exp(logp_new - logp_old)
        ratios = exp(logp_new - logp_old)
        advantages = compute_group_advantages([t.rewards for t in trajs])
        model.grpo_step(advantages, ratios)          # 信号(a)：模型参数更新

        # 信号(b)：同一批轨迹交给演化引擎，提出 harness 离散编辑
        edits = AEGIS.propose_edits(trajs, harness_cfg)
        for edit in edits:
            if validation_gate.passes(edit, held_out_tasks):
                harness_foundry.apply(edit)           # 严格提升才接受
```

工程要点：**不同样本的 Harness 不同（注册工具、提示词、记忆截断规则都不同），必须先还原到对应设置再用新模型推理**，否则概率比完全错误。这也是「Harness 作为条件输入」在算法层的具体含义。

### 4.3 SkillOpt 文本空间优化器伪代码

```python
# SkillOpt —— 技能文档的可控文本空间优化
def skillopt(skill_doc, rollout_source, epochs=3):
    best_doc = skill_doc
    best_score = evaluate(best_doc, held_out_val)
    rejected_buffer = []          # 被拒编辑：负反馈记忆，防重复尝试
    lr_budget = TextLR(initial=4, schedule=[4, 3, 2])  # 每轮编辑条数预算

    for epoch in range(epochs):
        rollouts = rollout_source.collect(best_doc)    # 评分后的真实执行轨迹
        proposals = optimizer_model.propose_edits(
            skill=best_doc,
            scored_rollouts=rollouts,
            rejected=rejected_buffer,
            budget=lr_budget.get(),                   # 有界：ADD/DELETE/REPLACE
        )

        for edit in proposals:                        # mini-batch 合并后逐个验证
            cand_doc = apply_edit(best_doc, edit)
            cand_score = evaluate(cand_doc, held_out_val)
            if cand_score > best_score:               # 验证门控：严格提升
                best_doc, best_score = cand_doc, cand_score
            else:
                rejected_buffer.append((edit, cand_score))  # 失败写入负反馈

        lr_budget.decay()

    return best_doc        # 一份普通 .md，部署时零额外调用
```

与 TextGrad 的路线差异：TextGrad 对每个变量分别算「文本梯度」，自由度高但易发散；SkillOpt 把经验汇聚到单个技能文档，用有界编辑 + 独立验证门控保证可复现，产物可独立部署，并验证了技能可跨模型、跨 Harness 迁移。

### 4.4 Misevolution：自进化的失控面

自进化不是「配好就能放手」。一篇专门研究进化失效的工作（Misevolution，知乎社区转述，**未取得 arXiv 原文独立核实**）沿四个进化维度测试失控：

| 维度 | 失控表现 |
|---|---|
| **记忆** | 记忆积累后安全对齐降级——Agent 开始接受它原本应拒绝的请求 |
| **工具** | 工具创建与复用无意中引入代码漏洞 |
| **模型权重** | 针对分布内任务的进化在分布外变成退化（RRSI 同样指出） |
| **工作流** | 控制流自修改后路径爆炸/死锁，且难以回滚 |

值得注意的是顶级模型（如 Gemini-2.5-Pro 级，知乎原文举例）也未能免疫。推论：**自进化必须配独立评估、编辑回滚与定期人工审计**；另一条工程边界是可恢复性——目前多数会话层没有 durable run cursor，进程退出后无法判断工具副作用状态，恢复未完成调用需要工具幂等、执行 receipt 与补偿机制。

---

## 5. 结论与选型建议

### 5.1 核心结论

1. **自进化与 Harness-Native RL 在 2026 年必然合流**：只有在真实/高保真 Harness 上 rollout，执行轨迹才能同时支撑模型参数更新与 Harness 离散进化，两个空间的信号同源、方向对齐。
2. **双基质分工明确**：Harness 编辑改粗粒度策略架构（加工具、换处理器、重组提示），模型 RL 改细粒度行为；Harness 决定能力释放率，模型决定能力上限，只优化一端都会留下瓶颈。
3. **「以 Harness 为条件」是训练目标的核心**：HSA 类任务保持型随机化让模型学会理解当前 Harness 而非背固定版本，三阶段训练（SFT → On-Policy 蒸馏 → RL）兼顾业务能力、通用能力与鲁棒性。
4. **进化必须有优化纪律**：有界编辑、独立验证集严格门控、文本学习率、负反馈缓冲——没有这些，技能/脚手架会越改越偏；这正是 SkillOpt 与 RRSI 的共同主张。
5. **递归自改进会过拟合且会 Misevolution**：分布内刷榜、记忆污染对齐、工具引入漏洞都是已观察到的风险，需要退火预算、OOD 评测、回滚与人工审计兜底。

### 5.2 落地 checklist

- □ 明确 Agent 双空间清单：哪些能力靠模型 RL，哪些靠 Harness/技能编辑
- □ 训练/进化的 rollout 放在真实或高保真 Harness 上，不用 Mock 环境
- □ 对 Harness 做任务保持型随机化（参考 HSA 五维），模型不依赖固定 ID/模板/工具名
- □ 三阶段推进：HSA-SFT 打基础 → On-Policy 蒸馏保通用 → HSA-RL 增鲁棒
- □ 跨 Harness 计算优势时，先按配置分组并还原对应设置再算概率比
- □ 技能编辑有界（ADD/DELETE/REPLACE + 预算），独立验证集严格提升才接受
- □ 维护被拒编辑缓冲（负反馈），同一失败编辑不重复尝试
- □ 给递归进化设退火编辑预算，限制单轮变更组件数
- □ 建 OOD 评测集，监控 ID/OOD 增益差，防 Harness 刷榜式过拟合
- □ 每次进化自动留快照可回滚；工具副作用保证幂等 + 执行 receipt
- □ 记忆写入前过滤；定期用安全测试集检查「记忆是否污染对齐」
- □ 定期人工审计技能/工具/工作流变更，关键系统保持 human-on-the-loop
- □ 先在单一业务线试点（如 TaoLive 从直播 QA 切入），验证 ROI 再推广

### 5.3 注意事项与诚实性声明

- 本报告**核心事实**（6 篇论文的标题、编号、作者机构、提交日期及摘要中的机制与数字）均经 arXiv 原文逐篇核实。
- TaoLive 的具体仿真器组件、HSA 五维表、三阶段细节的中文表述综合自知乎多篇深度解读，个别措辞（如 Stage 2 模拟器的精确实现）建议引用前查论文原文核对。
- **Misevolution 研究**（四维度失控、Gemini 举例）目前仅见知乎社区转述，未独立核实其论文出处，引用前请务必查证。
- SkillOpt-Lite、AutoDesign、AgentX、Hermes Agent 等条目来自社区文章；其中 GitHub star 数等数据时效性强，以官方仓库当前状态为准。
- RRSI（2026-09-21）极为新近，实验数字可能随版本更新变化。

---

## 6. 参考资料

### 原始出处 / 一手论文

1. 《Training Agents to Evolve with Their Harness: TaoLive Digital Avatar Agent Technical Report》. arXiv:2608.15763, 2026-08. https://arxiv.org/abs/2608.15763
2. 《HarnessX: A Composable, Adaptive, and Evolvable Agent Harness Foundry》. arXiv:2606.14249, 2026-06. https://arxiv.org/abs/2606.14249
3. 《SkillOpt: Executive Strategy for Self-Evolving Agent Skills》. arXiv:2605.23904, 2026-05. https://arxiv.org/abs/2605.23904
4. 《RRSI: Regularized Recursive Self-Improvement of Agent Harnesses》. arXiv:2609.24972, 2026-09-21. https://arxiv.org/abs/2609.24972
5. 《A Comprehensive Survey of Self-Evolving AI Agents: A New Paradigm Bridging Foundation Models and Lifelong Agentic Systems》. arXiv:2508.07407, 2025-08. https://arxiv.org/abs/2508.07407
6. 《RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments》. arXiv:2511.07317, 2025-11, ICML 2026. https://arxiv.org/abs/2511.07317
7. 《Who Grades the Grader? Co-Evolving Evaluation Metrics and Skills for Self-Improving LLM Agents》. arXiv:2607.12790, 2026-07. https://arxiv.org/abs/2607.12790
8. 《SEVerA: Verified Synthesis of Self-Evolving Agents》. arXiv:2603.25111, 2026-03. https://arxiv.org/abs/2603.25111
9. Shunyu Yao 等.《ReAct》. arXiv:2210.03629, 2022. https://arxiv.org/abs/2210.03629
10. Noah Shinn 等.《Reflexion》. arXiv:2303.11366, 2023. https://arxiv.org/abs/2303.11366

### 知乎深度解读

11. 龟壳.《Training Agents to Evolve with Their Harness》（HSA/三阶段详解）. https://zhuanlan.zhihu.com/p/2084597415307494818
12. FirSight.《Training Agents to Evolve with Their Harness: TaoLive Digital Avatar Agent Technical Report》. https://zhuanlan.zhihu.com/p/2077831105324099031
13. 大模型最新论文.《小米提出 HarnessX：自动进化的 agent 外壳》. https://zhuanlan.zhihu.com/p/2051062401626380122
14. Leezhou.《HarnessX：可组合、自适应、可演化的智能体 Harness 工厂》. https://zhuanlan.zhihu.com/p/2050234112791908374
15. 微软亚洲研究院.《SkillOpt：把智能体的"技能"当作可训练的外部参数》. https://zhuanlan.zhihu.com/p/2044477678036768645
16. 明了个明.《来自微软的自进化 SkillOpt 深度解读与反思》. https://zhuanlan.zhihu.com/p/2043415966944580626
17. PaperAgent.《微软竟然一口气发了 2 篇 SkillOpt》（SkillOpt-Lite / Bitter Lesson）. https://zhuanlan.zhihu.com/p/2065024808484616185
18. 恋猫.《谷歌 RRSI 论文：当 Harness 开始自己改自己时，首个问题是「刷 Benchmark」》. https://zhuanlan.zhihu.com/p/2086133782416049728
19. 硅基捕手维克托.《近一年 Agent 自进化的两大方向和四大趋势》（Misevolution）. https://zhuanlan.zhihu.com/p/2022259769969255910
20. vibe life.《Self Improvement Agent：Harness improvement 协议化》（HarnessX 与 LIFE-HARNESS）. https://zhuanlan.zhihu.com/p/2051607112724263376
21. vibe life.《从手工到自进化：Agent 环境扩展五篇论文》（RLVE/AgentScaler）. https://zhuanlan.zhihu.com/p/2079542661044818118
22. vibe life.《Self-Improving Agent · 两篇综述文章》. https://zhuanlan.zhihu.com/p/2064718097500655950
23. 黄浴.《自演化智体系统性综述：从以模型为中心到环境驱动的协同演化》. https://zhuanlan.zhihu.com/p/2034904235989528764
24. 陶刚.《当 Agent 开始自己教自己：一篇综述里的进化路线图》. https://zhuanlan.zhihu.com/p/2084004810383176735
25. Cici学算法.《自进化智能体协同进化综述》. https://zhuanlan.zhihu.com/p/2056466849857123080
26. JustBeClaw.《Hermes Agent 全面调研解读》（开源自改进框架）. https://zhuanlan.zhihu.com/p/2022015752258027715
27. 机器之心.《自进化设计 Harness 直接出论文 poster》（AutoDesign）. https://zhuanlan.zhihu.com/p/2074088170950546526
28. Leon.《Agent 进化 02：Harness 自进化全景，三种范式与六个系统》. https://zhuanlan.zhihu.com/p/2075899440183956551
29. 阿葱.《Harness Evolution：AgentX 如何改进自己？》. https://zhuanlan.zhihu.com/p/2081760978576974699
