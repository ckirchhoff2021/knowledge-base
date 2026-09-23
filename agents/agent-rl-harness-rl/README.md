# Agent-RL × Harness-RL

> LLM Agent 强化学习两条路线的系统调研：为什么「在简化环境里训 Agent」会失败，以及 Harness-Native RL 如何在真实脚手架上训练。

## 文档

### [Agent-RL 与 Harness-RL 技术调研报告](report.md)

🌐 [在线阅读](https://ckirchhoff2021.github.io/knowledge-base/agents/agent-rl-harness-rl/report.html) ｜ 📅 2026-09-23 ｜ 🖼 3 张架构图

约 2.2 万字。围绕 **Agent = Model × Harness** 拆解传统 Agent-RL 的训推不一致根因（Mock 工具、剥离 Hook/多轮状态），对照 Harness-Native RL 的 Harness-Benefit / Harness-Updating 闭环；覆盖 OpenForgeRL、CoreCraft 等系统，GRPO / GSPO / AT-GRPO / SAO 算法对比，verl / Miles / AReaL 基础设施选型，附算法伪代码与落地 checklist。信源为 12 篇知乎深度分析与 arXiv 论文交叉验证。

`Agent-RL` `Harness-Native RL` `GRPO` `训推不一致` `基础设施选型`
