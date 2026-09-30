# 🧠 大模型技术报告

大模型与后训练方向的论文分析，共 5 个话题。

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
