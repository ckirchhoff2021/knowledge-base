# 🧠 大模型技术报告

大模型与后训练方向的论文分析，共 **32 个话题**。每个话题为自包含文件夹（`.md` 源文件 + 精排 `.html` + `diagrams/`），按方向分为四组。

---

## 架构基础

### [Transformer × 位置编码](https://ckirchhoff2021.github.io/knowledge-base/models/transformer-pe/report.html)

系统拆解 Transformer 的注意力、FFN 与编解码器，重点回答「位置编码为什么不可或缺」，并对比正弦编码、RoPE、ALiBi 与 PI/NTK/YaRN/LongRoPE 等长上下文扩展方案。

📅 2026-09-30 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](transformer-pe/report.md)

### [线性注意力与混合架构](https://ckirchhoff2021.github.io/knowledge-base/models/linear-attention/report.html)

用「线性注意力 = 可并行训练的线性 RNN」统一视角，梳理 Linear Transformer → RetNet → Mamba 的演进、门控更新规则，以及 Kimi Linear、MiniMax Lightning、Jamba 等旗舰横评。

📅 2026-10-15 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](linear-attention/report.md)

### [稀疏注意力 Sparse Attention](https://ckirchhoff2021.github.io/knowledge-base/models/sparse-attention/report.html)

当「每个 token 看全部」不再成立：算力 O(L²·d) 与显存 O(L·d) 两笔账、Prefill/Decode 两阶段，以及固定模式（Sparse Transformer/Reformer/Longformer/BigBird）→ 免训练动态（H₂O/Quest/SnapKV/MInference）→ 原生可训练（NSA/MoBA/MSA/DSA/CSA2）三代谱系；深度拆解 NSA 三分支门控、MoBA MoE 式无参数块路由、MSA 双分支固定 2048 预算、V4.1 CSA2+FP4 把全局 KV 压到 890 字节/token，梳理稀疏×线性×KV 压缩正交可叠加关系与 12 条落地 Checklist。

📅 2026-11-03 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](sparse-attention/report.md)

### [MoE 稀疏混合专家](https://ckirchhoff2021.github.io/knowledge-base/models/moe-architecture/report.html)

讲清「容量与计算解耦」的稀疏专家架构：路由策略、负载均衡、专家并行与通信开销，以及 DeepSeek、Mixtral 等主流 MoE 系统的工程取舍。

📅 2026-10-22 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](moe-architecture/report.md)

### [扩散语言模型 DLM](https://ckirchhoff2021.github.io/knowledge-base/models/diffusion-language-models/report.html)

当语言模型不再只从左往右写：掩码/多项/score-entropy/连续潜空间四条技术路线、Block-AR 工程外壳与采样压缩，覆盖 LLaDA、Mercury、DiffusionGemma 等系统实测。

📅 2026-10-11 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](diffusion-language-models/report.md)

---

## 训练与算法

### [Post-Training 后训练](https://ckirchhoff2021.github.io/knowledge-base/models/post-training/report.html)

后训练全景：SFT、RLHF(PPO)、奖励模型、DPO 家族（IPO/KTO/ORPO/SimPO）、RLAIF 与 RFT/STaR，以及 DeepSeek-R1「冷启动—大规模 RL—蒸馏」范式，含 Loss 与伪代码。

📅 2026-09-24 ｜ 🖼 架构图 ×3 ｜ 📄 [Markdown](post-training/report.md)

### [大模型强化学习算法](https://ckirchhoff2021.github.io/knowledge-base/models/llm-rl-algorithms/report.html)

从策略梯度与 PPO（clip 信任域、GAE、Critic）到 Critic-Free 家族与主流 GRPO，再到 DAPO、Dr.GRPO、GSPO 改进变体，并梳理 RLHF/RLAIF/RLVR 三种奖励范式与 reward hacking 对策。

📅 2026-09-30 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](llm-rl-algorithms/report.md)

### [Agentic RL 智能体强化学习](https://ckirchhoff2021.github.io/knowledge-base/models/agentic-rl/report.html)

聚焦多轮环境下的 RL：轨迹级优化、稀疏延迟奖励、回声陷阱/模板坍塌/交互坍塌等训练病理，以及 GiGPO、SPA-RL、DART 等算法与异步训练基础设施。

📅 2026-10-10 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](agentic-rl/report.md)

### [合成数据与修正缩放定律](https://ckirchhoff2021.github.io/knowledge-base/models/synthetic-data/report.html)

模型崩溃的机理与合成数据方法：数据放大/筛选/回炉策略、合成数据的偏差累积，以及如何用真实数据锚点修正缩放定律。

📅 2026-10-19 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](synthetic-data/report.md)

### [反思式提示进化 GEPA](https://ckirchhoff2021.github.io/knowledge-base/models/reflective-prompt-evolution/report.html)

当提示成为可学习的程序：语言程序 Φ=⟨Π, Θ_frozen⟩ 形式化、三代谱系（APE/OPRO→EvoPrompt/Promptbreeder/TextGrad→GEPA）、反思变异 + 实例级 Pareto 选择 + System-aware Merge，六任务平均胜 GRPO 6%、rollout 省 35×，以及 2026 改进/应用/诊断/替代四路线跟进与提示级过拟合治理。

📅 2026-10-28 ｜ 🖼 架构图 ×5 ｜ 📄 [Markdown](reflective-prompt-evolution/report.md)

### [Muon 优化器：矩阵正交化](https://ckirchhoff2021.github.io/knowledge-base/models/muon-optimizer/report.html)

「后 Adam」优化器方向：把权重矩阵的动量做极分解正交化（奇异值谱拉平为满数值秩），用 Newton–Schulz 5 次迭代廉价近似。Moonlight 论文补入解耦 weight decay 与 consistent update RMS 两个规模化关键，缩放定律测得 Muon 达到 AdamW 同等性能仅需约 52% FLOPs（约 2× 算力效率）；万亿参数 Kimi K2 用 MuonClip（QK-Clip，τ=100）约束注意力 logits，15.5T tokens 零 loss spike。报告还梳理 NorMuon/Dion3/Hierarchical/Turbo-Muon/Muon^p 等百变体、收敛性反例，并严格区分 token/FLOPs 效率与 wall-clock 效率（标准 Muon 优化器步可达 26× AdamW）。

📅 2026-11-26 ｜ 🖼 架构图 ×3 ｜ 📄 [Markdown](muon-optimizer/report.md)

### [当 AI 开始改写自己：权重空间编程 WSP 与递归自我改进 RSI](https://ckirchhoff2021.github.io/knowledge-base/models/self-improving-ai/report.html)

2026-11 单周十余篇新论文集中爆发的新范式：以「概念地图（WSL 底座 × WSP 能力形态 × RSI 系统闭环）」统摄，梳理 Gödel Machine × MAML/超网络 × ROME/MEMIT 三脉合流；权重空间学习 WSU/WSR/WSG 三维与权重对称群（神经元置换、注意力 GL 对称、头置换；图元网络 Graph Metanetwork、UWE 跨架构编码）；WSP 三大支柱（读 / 局部编辑 / 从零手工设定权重，WSP-Scope 仅改 1023 参数使 MATH-500 +29.45 分）与同周成熟的怀疑论（局部算术+数据集记忆、计算而非权重访问、架构天花板）；RSI 双循环与 L0–L3 分级、Darwin Gödel Machine（SWE-bench 20→50%）、运行时自改进 Exo/BOXBOX、产业落地百度 ERNIE Evolve（昆仑芯 P800 生成速度 623.6%）、Conitzer 安全博弈与权限守恒，附 14 条落地 Checklist。

📅 2026-11-10 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](self-improving-ai/report.md)

---

## Jev 决策模型

### [JEV 决策小模型](https://ckirchhoff2021.github.io/knowledge-base/models/jev-decision/report.html)

TypeSafe AI 的 System One 决策模型 Jev 深度调研：类型安全约束解码、决策流水线、变体对比，以及 Jev-like 小模型的复现训练流程与选型建议。

📅 2026-09-21 ｜ 🖼 架构图 ｜ 📄 [Markdown](jev-decision/report.md)

### [Jev 训练方案 · Qwen3.5-4B](https://ckirchhoff2021.github.io/knowledge-base/models/jev-qwen35-4b/report.html)

以 Qwen3.5-4B 为基座复现 Jev-like 决策模型的完整工程方案：单 token 约束解码 vs readout 打分头、数据构造、LoRA SFT + 校准 + 可选 GRPO、评估门禁与部署级联。

📅 2026-09-28 ｜ 🖼 架构图 ×3 ｜ 📄 [Markdown](jev-qwen35-4b/report.md)

### [开源 Jev 训练方法全景](https://ckirchhoff2021.github.io/knowledge-base/models/open-source-jev-training/report.html)

横评开源 Jev 模型各自的训练方法：软标签 SFT、proper scoring rule RL、教师分布蒸馏、多模态训练等路线的损失设计与实测指标对比。

📅 2026-10-20 ｜ 🖼 架构图 ×3 ｜ 📄 [Markdown](open-source-jev-training/report.md)

### [开源 Jev 模型全景](https://ckirchhoff2021.github.io/knowledge-base/models/open-source-jev-landscape/report.html)

HuggingFace 上 50+ 条目中七大家族 16 个代表模型的目录式盘点，含统一契约（noul/choice/score）、共同设计决策、评测纪律与选型 Checklist。

📅 2026-10-26 ｜ 🖼 架构图 ×3 ｜ 📄 [Markdown](open-source-jev-landscape/report.md)

---

## 系统与应用

### [SkillOpt 技能优化](https://ckirchhoff2021.github.io/knowledge-base/models/skillopt/report.html)

把 Agent 技能文档视为冻结模型的「外部可训练状态」，用独立优化器把轨迹转为有界 ADD/DELETE/REPLACE 编辑，仅在验证集严格提升时接受，实现无需改权重的技能迭代。

📅 2026-06-29 ｜ 📄 [Markdown](skillopt/report.md)

### [世界模型 World Models](https://ckirchhoff2021.github.io/knowledge-base/models/world-models/report.html)

从「预测下一个词」迈向「预测世界下一状态」：L1–L4 抽象层级、五大技术路线（像素生成/潜空间动力学/JEPA/显式 3D/物理状态），以及 Genie 3、V-JEPA 2、Dreamer 4 三大旗舰拆解。

📅 2026-10-12 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](world-models/report.md)

### [智能体记忆 Agent Memory](https://ckirchhoff2021.github.io/knowledge-base/models/agent-memory/report.html)

偏学术谱系：三正交轴分类学、四类认知记忆、六代架构演进（朴素 RAG → MemGPT → Mem0 → Zep 时序图谱 → A-MEM）、记忆生命周期与自进化记忆前沿。

📅 2026-10-13 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](agent-memory/report.md)

### [Agent Memory 系统设计](https://ckirchhoff2021.github.io/knowledge-base/models/agent-memory-systems/report.html)

偏系统实现（与上一篇互补）：深度拆解 Hermes 冻结快照文件记忆、Mem0 事实抽取与多信号检索、Codex 两阶段抽取-巩固流水线，横评 Claude Code/Letta/Devin/Honcho/Zep，附参考架构与选型表。

📅 2026-10-26 ｜ 🖼 架构图 ×3 ｜ 📄 [Markdown](agent-memory-systems/report.md)

### [视觉-语言-动作模型 VLA](https://ckirchhoff2021.github.io/knowledge-base/models/vla-embodied/report.html)

让大模型从「描述世界」走向「改变世界」：动作表示光谱、双系统架构、跨本体泛化与数据来源，横评 π0.5、Gemini Robotics、OpenVLA-OFT 等开源人形 VLA。

📅 2026-10-14 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](vla-embodied/report.md)

### [智能体互联协议栈 MCP/A2A](https://ckirchhoff2021.github.io/knowledge-base/models/agent-interop/report.html)

MCP、A2A 与 AGENTS.md 三大互联标准的分层关系：工具接入、Agent 间通信与上下文约定，以及它们如何共同构成 Agent 互操作底座。

📅 2026-10-16 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](agent-interop/report.md)

### [AI Scientist 自主科研](https://ckirchhoff2021.github.io/knowledge-base/models/ai-scientist/report.html)

自主科研智能体全景：从选题、文献综述、实验设计到论文撰写的自动化流水线，代表性系统、能力边界与可复现性风险。

📅 2026-10-17 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](ai-scientist/report.md)

### [机制可解释性与欺骗检测](https://ckirchhoff2021.github.io/knowledge-base/models/mech-interp/report.html)

用机制可解释性打开模型黑箱：特征/回路级分析方法，以及如何检测模型的欺骗性对齐与隐藏意图，兼顾 AI 安全治理视角。

📅 2026-10-18 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](mech-interp/report.md)

### [大模型推理优化](https://ckirchhoff2021.github.io/knowledge-base/models/inference-optimization/report.html)

推理侧工程全栈：量化、KV-Cache、投机采样、PD 分离与连续批处理，以及面向长上下文与高并发的部署优化。

📅 2026-10-20 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](inference-optimization/report.md)

### [AI 编译器与异构底座](https://ckirchhoff2021.github.io/knowledge-base/models/ai-compiler-stack/report.html)

从 AI Kernel 到芯片的全栈：算子编译、图优化、国产异构硬件适配与编译器自动调优。

📅 2026-10-21 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](ai-compiler-stack/report.md)

### [GUI Agent 电脑操作](https://ckirchhoff2021.github.io/knowledge-base/models/gui-agent/report.html)

让 Agent 操作图形界面：截图理解、元素定位、动作生成与环境反馈闭环，覆盖主流电脑/手机操作智能体与评测基准。

📅 2026-10-23 ｜ 🖼 架构图 ×6 ｜ 📄 [Markdown](gui-agent/report.md)

### [Test-Time Scaling](https://ckirchhoff2021.github.io/knowledge-base/models/test-time-scaling/report.html)

继参数缩放、后训练之后的第三条能力轴：长思维链、并行采样、树/图搜索与自验证，以及 o1、s1 budget forcing、DeepSeek-R1 与过度思考的治理。

📅 2026-10-25 ｜ 🖼 架构图 ×5 ｜ 📄 [Markdown](test-time-scaling/report.md)

### [全双工实时语音对话模型](https://ckirchhoff2021.github.io/knowledge-base/models/full-duplex-speech/report.html)

当大模型学会「边听边说」：级联 ASR-LLM-TTS → 端到端 SpeechLM → Moshi 原生全双工三代谱系、流式 codec 与 turn-taking，覆盖模型/中间件/硬件三层产业生态。

📅 2026-10-26 ｜ 🖼 架构图 ×5 ｜ 📄 [Markdown](full-duplex-speech/report.md)

### [Agent Harness 智能体脚手架](https://ckirchhoff2021.github.io/knowledge-base/models/agent-harness/report.html)

当脚手架成为能力本身：「Agent = Model + Harness」的六层框架与六大运行时职责、失败归因五度量（VCR/Model-Harness Fit）、Harness Scaling 第四缩放维度、Cache Engineering 与 Token 资源经济学，以及「harness 是否会消失」的范式之争。

📅 2026-10-27 ｜ 🖼 架构图 ×5 ｜ 📄 [Markdown](agent-harness/report.md)

### [主动式智能体 Proactive Agent](https://ckirchhoff2021.github.io/knowledge-base/models/proactive-agent/report.html)

当 AI 不再等待指令：核心问题从「能不能做（capability）」转向「该不该此刻无提示地做（intervention timing）」。以 `intervene ⇔ E[收益] − E[打扰成本] > θ` 为主线，梳理五层管线与 L0–L4 分级、1999–2017 决策论谱系与 2024–2026 LLM 浪潮为何互不相通，拆解 Proactive Agent / PRISM / TGL / ProAgent / Pro²Assist / EgoSocial / Plan-Watch-Recover 等旗舰，剖析「廉价门控/昂贵执行器」收敛、reward⊆eval、接受率陷阱与 must/may/must-not 三态窗口，并以 Claude Duet 为产业样本。

📅 2026-11-01 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](proactive-agent/report.md)

### [上下文工程 Context Engineering](https://ckirchhoff2021.github.io/knowledge-base/models/context-engineering/report.html)

当 Agent 的主战场从「写提示」转向「管理 token 全生命周期」：提示工程只写指令，上下文工程决定运行每一步窗口里的全部 token。以「标称窗口≠有效窗口」为切入点，深度拆解 Context Rot（Chroma 18 模型、完美检索仍退化；位置/难度/字面重合三股独立力量）、Write/Select/Compress/Isolate 四杠杆与中毒/分心/混淆/冲突四失败的映射、Compaction 可逆性三分（掩码/卸载/摘要）与 Prompt Cache 经济学、ACE 生成→反思→策展三阶段自改进闭环（编码 +10.6% / 金融 +8.6%）、静态 AGENTS.md 之争（Gloaguen −3% vs Lulla 运行时 −28.64%）、Governance Decay（压缩静默删除合规约束；约束存活 0% 违规 vs 被删 38%）与带外策略层，以及「压缩后约束恢复率」评测和 12 条落地 Checklist。

📅 2026-11-02 ｜ 🖼 架构图 ×6 ｜ 📄 [Markdown](context-engineering/report.md)
