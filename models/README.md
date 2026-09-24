# 🧠 大模型技术报告

大模型与后训练方向的论文分析，共 3 个话题。

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
