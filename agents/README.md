# 🤖 Agent 技术报告

LLM Agent 方向的深度调研合集，共 **9 个话题**。每个话题为自包含文件夹（`.md` 源文件 + 精排 `.html` + `diagrams/`）。

---

### [Agent-RL × Harness-RL](https://ckirchhoff2021.github.io/knowledge-base/agents/agent-rl-harness-rl/report.html)

围绕 **Agent = Model × Harness**，拆解传统 Agent-RL 训推不一致的根因（Mock 工具、剥离 Hook/多轮状态），对照 Harness-Native RL 的闭环；覆盖 GRPO/GSPO/AT-GRPO 算法与 verl/Miles/AReaL 基础设施选型。

📅 2026-09-23 ｜ 🖼 架构图 ×3 ｜ 📄 [Markdown](agent-rl-harness-rl/report.md)

### [Self-Evolving Agents](https://ckirchhoff2021.github.io/knowledge-base/agents/self-evolving-harness-rl/report.html)

Agent 同时在「模型权重 θ」与「Harness 配置 η」双空间协同进化：统一反馈环与 TaoLive、HarnessX、SkillOpt、RRSI 四个代表系统，以及 Misevolution 失控面分析。

📅 2026-09-23 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](self-evolving-harness-rl/report.md)

### [Loop Engineering 循环工程](https://ckirchhoff2021.github.io/knowledge-base/agents/loop-engineering/report.html)

从「人给 Agent 写提示词」到「人设计让 Agent 自行运转的循环」：LoopSpec 五要素、四类循环、嵌套反馈环、builder/checker 验证者模式与工程反模式。

📅 2026-09-23 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](loop-engineering/report.md)

### [Claude Tag](https://ckirchhoff2021.github.io/knowledge-base/agents/claude-tag/report.html)

Anthropic「AI 同事」三份配套调研：技术调研讲四大能力与分层记忆、技术栈拆解组织级 Agent 运行时、开源盘点对比可自托管、不锁模型的复刻项目。

📅 2026-08 ｜ 🖼 架构图 ×6 ｜ 📄 [Markdown](claude-tag/report.md)

### [飞书数字员工记忆系统](https://ckirchhoff2021.github.io/knowledge-base/agents/feishu-memory/report.html)

长期驻留、多群并发的 AI 同事如何持续学习：参考 Hermes、Claude Tag 与 Mem0/Letta/Zep，给出五层记忆 + 四级隔离架构，覆盖蒸馏、检索、遗忘与安全治理全生命周期。

📅 2026-08-27 ｜ 🖼 架构图 ×1 ｜ 📄 [Markdown](feishu-memory/report.md)

### [拟人化 Agent 产品设计](https://ckirchhoff2021.github.io/knowledge-base/agents/anthropomorphic-agent/product-design.html)

让 AI 像「人」而非「工具」：产品调研梳理工具型→伙伴型三阶段演化与六层架构，理论框架整理媒体方程式、CASA 范式与五维拟人化设计理论（共 2 篇）。

📅 2026-08-24 ｜ 📄 [Markdown](anthropomorphic-agent/product-design.md)

### [OPD / OPSD 在线策略蒸馏](https://ckirchhoff2021.github.io/knowledge-base/agents/online-policy-distillation/report.html)

后训练第三条路：从离线方法的暴露偏差与在线 RL 的稀疏奖励切入，讲清 On-Policy Distillation 如何在学生自身轨迹上对齐教师，附完整数学推导与落地指南（共 2 篇）。

📅 2026-07 ｜ 🖼 架构图 ×3 ｜ 📄 [Markdown](online-policy-distillation/report.md)

### [Hermes Agent API 指南](hermes/API.md)

本地代码/IDE 如何调用远端 Hermes 完整 Agent 能力：厘清 `proxy`/`serve`/`dashboard`/`api_server` 四种方式，给代码用的是 OpenAI 兼容 API Server。

📅 2026-08-24 ｜ 📄 [Markdown](hermes/API.md)

### [大模型制作短剧完全指南](https://ckirchhoff2021.github.io/knowledge-base/agents/ai-short-drama/report.html)

用 LLM + 多模态模型制作真人风短剧/AI 漫剧的全流程：七层流水线、结构化剧本、角色一致性四代方案、视频/音频模型横评、成本账与三条落地路线，附分镜级 Checklist。

📅 2026-10-22 ｜ 🖼 架构图 ×3 ｜ 📄 [Markdown](ai-short-drama/report.md)
