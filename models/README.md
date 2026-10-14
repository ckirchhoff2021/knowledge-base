# 🧠 大模型技术报告

大模型与后训练方向的论文分析，共 11 个话题。

---

### [JEV 决策小模型](jev-decision/report.html)

TypeSafe AI 于 2026-09-15 发布的 System One 决策模型 **Jev** 深度调研：System One 架构、类型安全约束解码、决策流水线、变体对比与实测数据、Jev-like 模型复现训练流程，以及与 Agent 生态的集成和选型建议。信源为 29 篇去重知乎文章及 TypeSafe 官方博客、Vercel、TechCrunch 等外部信源交叉验证。

📅 2026-09-21 ｜ 🖼 架构图 ｜ 📄 [Markdown](jev-decision/report.md)

### [SkillOpt 论文分析](skillopt/report.html)

arXiv:2605.23904（微软 + 上海交大 + 同济 + 复旦，2026-05）解读：把 Agent 技能文档视为冻结模型的**外部可训练状态**，用独立优化器模型把打分轨迹转为有界 ADD/DELETE/REPLACE 编辑，仅在验证集严格提升时接受，实现稳定、可复现、无需改权重的技能迭代。

📅 2026-06-29 ｜ 📄 [Markdown](skillopt/report.md)

### [Post-Training 后训练技术](post-training/report.html)

系统梳理大模型后训练技术全景：SFT、RLHF(PPO)、奖励模型 ORM/PRM、DPO 家族（IPO/KTO/ORPO/SimPO）、RLAIF 与 Constitutional AI、RFT/STaR、RLVR 与 GRPO，以及 DeepSeek-R1 的冷启动—大规模 RL—蒸馏范式。含方法演进谱系（2017–2025）、横向对比、关键实验数据、核心 Loss 与 GRPO/DPO 伪代码和工程落地 checklist。信源为 16 篇知乎文章与其引用的 arXiv 论文交叉验证。

📅 2026-09-24 ｜ 🖼 架构图 3 张 ｜ 📄 [Markdown](post-training/report.md)

### [Jev 决策模型训练方案（Qwen3.5-4B）](jev-qwen35-4b/report.html)

以 Qwen3.5-4B 为基座复现 Jev-like 决策模型的完整工程方案：单 token 约束解码 vs readout 打分头双路线选型、GDN 混合架构适配与关闭 thinking、数据四元组与硬负例/无信号/拒答构造、LoRA SFT + temperature 概率校准 + 可选 GRPO、评估四件套门禁、合并/GGUF 转换与 vLLM/Ollama 部署、阈值级联与难例回流。含四周排期、资源预算与风险对策，3 张架构图。

📅 2026-09-28 ｜ 🖼 架构图 3 张 ｜ 📄 [Markdown](jev-qwen35-4b/report.md)

### [Transformer 架构与位置编码](transformer-pe/report.html)

系统拆解 Transformer 架构（缩放点积注意力、多头机制、FFN、残差与归一化、编解码器结构），并重点剖析**位置编码为什么不可或缺**：自注意力的置换等变性，以及正弦编码、可学习编码、相对位置编码（Shaw/T5）、**RoPE 旋转编码**、**ALiBi 线性偏置**的原理与取舍。进一步解释上下文长度与位置编码的深层关系，以及 PI、NTK、YaRN、LongRoPE 长文本扩展技术（内插 vs 外推、分频段缩放，直至 200 万 Token）。信源为 9 篇 arXiv 原始论文（编号经 arXiv API 逐一核实）与知乎高赞文章交叉验证。

📅 2026-09-30 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](transformer-pe/report.md)

### [大模型常用强化学习算法](llm-rl-algorithms/report.html)

从策略梯度（Policy Gradient）与 MDP 建模出发，深度剖析大模型 RL 算法：经典基线 **PPO**（重要性采样、clip 信任域、GAE、Critic、KL 惩罚），Critic-Free 家族 **RLOO / ReMax / REINFORCE++**，以及当代主流 **GRPO**（组相对优势、token clip、KL 无偏估计）与其改进变体 **DAPO**（clip-higher、动态采样、token-level loss、超长软奖励）、**Dr.GRPO**（去除长度膨胀偏置）、**GSPO**（序列级优化，稳 MoE）。并梳理 RLHF/RLAIF/RLVR 三种奖励范式、reward hacking 对策与 DeepSeek-R1 多阶段流程。信源为 13 篇 arXiv 原始论文（编号经 arXiv API 逐一核实）与知乎高赞文章交叉验证。

📅 2026-09-30 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](llm-rl-algorithms/report.md)

### [Agentic RL 智能体强化学习](agentic-rl/report.html)

聚焦多轮环境下的大模型强化学习：从单轮 RLVR 到 Agentic RL 的五个本质差异（轨迹级优化、极稀疏延迟奖励、有状态随机环境、长 rollout、天然 off-policy）；系统梳理七类训练病理——**回声陷阱**（奖励方差悬崖+梯度尖峰，RAGEN）、**模板坍塌**（推理与输入无关，RAGEN-2）、**交互坍塌**（工具越用越少，ASTER）、reasoning 与 tool-use 的梯度冲突、犹豫/无效轮次与 staleness；四族算法方案：StarPO、**GiGPO**（组中组步级信用分配）、SPA-RL（进展归因）、ARPO（熵自适应采样）、T²PO（不确定性引导，ICML 2026 Spotlight）、DART（推理/工具双 LoRA）；并覆盖可验证环境规模化、奖励工程、异步训练基础设施（partial rollout、Agent Swarm、弹性伸缩）与 Kimi K2、MiniMax Forge、GLM-5 阶段化 RL 路线。信源为 19 篇 arXiv 原始论文（编号经 API 逐一核实）与知乎高赞综述。

📅 2026-10-10 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](agentic-rl/report.md)

### [扩散语言模型（DLM）](diffusion-language-models/report.html)

当 Transformer 不再只从左往右写：系统梳理 2015–2026 年扩散语言模型的数学原理、四条技术路线（掩码/多项/score-entropy/连续潜空间）、Block-AR 工程外壳、采样器与推理压缩（48 步压到平均 12 步）、扩散原生后训练（VRPO、SD·RL），以及 LLaDA/LLaDA2.0、Mercury、Dream、Seed Diffusion、字节 Cola DLM、Google DiffusionGemma、NVIDIA Nemotron 等前沿系统的实测数据与能力边界（DiffusionGemma 单 H100 约 1500 TPS、约 20 TPF）。信源为 37 篇 arXiv 原始论文（编号经 arXiv 官网逐一核实）与官方技术报告交叉验证。

📅 2026-10-11 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](diffusion-language-models/report.md)

### [世界模型（World Models）](world-models/report.html)

智源《2026 十大 AI 技术趋势》第 1 位方向——从「预测下一个词」迈向「预测世界下一状态」（Next-State Prediction）。系统梳理世界模型的形式化定义 P(s′|s,a)、L1–L4 抽象层级与三大核心能力（可交互预测/物理一致/动作条件）；1943–2026 思想源流（Craik 心智模型、Sutton Dyna、Ha & Schmidhuber《World Models》VAE+MDN-RNN、PlaNet/Dreamer 潜空间想象 RL、LeCun JEPA）；五大技术路线（像素生成式 Sora/Genie/GameNGen、潜空间动力学 IRIS/DIAMOND/Dreamer、JEPA 表征空间预测、显式 3D 空间智能 World Labs/腾讯混元、物理状态 NSP 智源 Physis）；深度拆解 Genie 3（首个实时可交互、720p/24fps/数分钟、无论文）、V-JEPA 2-AC（100 万小时视频预训练 + 62 小时机器人数据零样本规划，arXiv:2506.09985）、Dreamer 4（纯离线数据 Minecraft 挖钻，单 H100 ≥20fps，arXiv:2509.24527）三大旗舰；世界模型与 VLA 的条件分布互补关系及融合架构；NVIDIA Cosmos（Predict/Transfer/Reason + 合成数据蓝图）、智源悟界 Emu3/Emu3.5/Physis、World Labs Marble、AMI Labs、腾讯 HunyuanWorld、生数 Motus 等 2025–2026 产业地图；以及 PSNR/SSIM/FVD 局限、WorldModelBench 反事实一致性等评测体系与长时程漂移、sim2real、模型利用等开放问题。信源为 20+ 篇 arXiv 原始论文（编号经 arXiv 官方 API 逐一核实）与 DeepMind/Meta/NVIDIA/OpenAI/智源官方及新华社、知乎专栏交叉验证。

📅 2026-10-12 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](world-models/report.md)

### [智能体记忆（Agent Memory）](agent-memory/report.html)

2025 末–2026 最热 Agent 基础设施方向：Agent 正从「无状态对话」走向「终身学习」。基于《Memory in the Age of AI Agents》（arXiv:2512.13564，Hugging Face 2025-12 Daily Paper #1）等 25 篇 arXiv 论文，系统拆解记忆三正交轴（形式 × 认知功能 × 动态性）与工作 / 情景 / 语义 / 程序四类认知记忆；梳理由此及彼的六代架构演进——朴素全上下文/RAG → ReadAgent 压缩 → MemGPT/Letta OS 换页 → Mem0 事实抽取 → Zep 双时序知识图谱 → A-MEM 卡片盒自组织，并讲透 MemOS 参数/激活/外部三类记忆统一治理；覆盖写入—组织—检索—更新/遗忘生命周期、LoCoMo/LongMemEval/BEAM 基准矩阵与横评方法论陷阱、2026 自进化记忆前沿（Evo-Memory/CHIME/MindMemOS）、隐私抽取攻击与可删除性等开放问题。厂商自报数据均标注「厂商口径」。

📅 2026-10-13 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](agent-memory/report.md)

### [视觉-语言-动作模型（VLA）](vla-embodied/report.html)

智源《2026 十大 AI 技术趋势》第 2 位方向——让大模型从「描述世界」走向「改变世界」。基于 44 篇 arXiv 原始论文（编号经官方 API 逐一核实），系统梳理 VLA 四代技术源流（SayCan / Code as Policies 模块化时代 → RT-1 / RT-2 端到端 → Open X-Embodiment 跨本体与 π0 / OpenVLA-OFT 连续生成式 → 人形机器人全栈开源）；深潜动作表示光谱（离散 token / DCT 压缩的 FAST / 连续回归 / 扩散 / 流匹配）与自回归离散头、扩散头、流匹配头三类生成头；讲透 System 2 VLM + System 1 动作专家的双系统架构（GR00T N1、Figure Helix）、跨本体泛化（OXE 统一动作空间、HPT 本体 token、跨本体注意力）、真机/人类视频/合成/跨域对齐四类数据来源；横评 π0.5、Gemini Robotics（含 Robotics-ER）、OpenVLA-OFT、SmolVLA、RDT-1B、GR-2、RoboBrain、SpatialVLA/EnerVerse；覆盖仿真/仿真化真机/真机三层评测体系（LIBERO、SimplerEnv）与 sim-to-real gap、长尾、延迟、安全、世界模型融合等六大开放问题；并给出 2025–2026 海内外产业地图（含 8 款开源人形 VLA）与选型 Checklist。厂商自报数据均标注「厂商口径」。

📅 2026-10-14 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](vla-embodied/report.md)
