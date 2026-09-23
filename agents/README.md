# 🤖 Agent 技术报告

LLM Agent 方向的深度调研合集，共 8 个话题。每个话题为自包含文件夹（`.md` 源文件 + 精排 `.html` + `diagrams/`）。

---

### [Agent-RL × Harness-RL](agent-rl-harness-rl/report.html)

约 2.2 万字。围绕 **Agent = Model × Harness**，拆解传统 Agent-RL 的训推不一致根因（Mock 工具、剥离 Hook/多轮状态），对照 Harness-Native RL 的 Harness-Benefit / Harness-Updating 闭环；覆盖 OpenForgeRL、CoreCraft 等系统，GRPO / GSPO / AT-GRPO / SAO 算法对比，verl / Miles / AReaL 基础设施选型，附伪代码与落地 checklist。

📅 2026-09-23 ｜ 🖼 3 图 ｜ 📄 [Markdown](agent-rl-harness-rl/report.md)

### [Self-Evolving Agents × Harness-Native RL](self-evolving-harness-rl/report.html)

Agent 同时在「模型权重 θ」与「Harness 配置 η」双空间协同进化：自进化统一反馈环与四个代表系统——TaoLive HAT/HSA 五维扰动 + 三阶段训练、小米 HarnessX 的 Cross-Harness GRPO（5 基准平均 +14.5%）、微软 SkillOpt 有界文本编辑 + 验证门控、RRSI 退火编辑预算防过拟合；含 Misevolution 失控面与 checklist。6 篇核心论文经 arXiv 逐篇核实。

📅 2026-09-23 ｜ 🖼 4 图 ｜ 📄 [Markdown](self-evolving-harness-rl/report.md)

### [Loop Engineering（循环工程）](loop-engineering/report.html)

从「人给 Agent 写提示词」到「人设计让 Agent 自行运转的循环」：LoopSpec 五要素形式化、Anthropic turn/goal/time/proactive 四类循环、Addy Osmani 5+1 工程组件、吴恩达三层嵌套反馈环、循环契约与 builder/checker 验证者模式、工程反模式与落地 checklist。信源均核实原文。

📅 2026-09-23 ｜ 🖼 4 图 ｜ 📄 [Markdown](loop-engineering/report.md)

### [Claude Tag](claude-tag/report.html)

Anthropic「AI 同事」三份配套调研（官方公告已核实）：[技术调研](claude-tag/report.html)讲四大能力、Agent Identity、Session/Harness 运行时与分层记忆；[技术栈拆解](claude-tag/tech-stack.html)逐层分析组织级 Agent 运行时的算法研究点；[开源盘点](claude-tag/open-source.md)对比自托管、不锁模型的开源复刻项目。

📅 2026-08 ｜ 🖼 6 图 ｜ 📄 [Markdown](claude-tag/report.md)

### [飞书数字员工记忆系统](feishu-memory/report.html)

长期驻留、多群并发的 AI 同事如何持续学习：参考 Hermes、Claude Tag 与 Mem0 / Letta / Zep / 腾讯 AgentMemory，给出**五层记忆 + 四级隔离**架构，覆盖写入蒸馏、预算检索、跨群知识提升、主动沉淀、遗忘与安全治理的全生命周期。

📅 2026-08-27 ｜ 🖼 1 图 ｜ 📄 [Markdown](feishu-memory/report.md)

### [拟人化 Agent 产品设计](anthropomorphic-agent/product-design.html)

让 AI 像「人」而非「工具」：[产品调研](anthropomorphic-agent/product-design.html)梳理工具型 → 伙伴型三阶段演化、六层架构与主流产品对比；[理论框架](anthropomorphic-agent/design-framework.html)整理媒体方程式、CASA 范式与外在/语言/认知/情感/社交五维设计理论。

📅 2026-08-24 ｜ 📄 [Markdown](anthropomorphic-agent/product-design.md)

### [OPD / OPSD 在线策略蒸馏](online-policy-distillation/report.html)

后训练第三条路：[技术调研](online-policy-distillation/report.html)从离线方法暴露偏差 vs 在线 RL 稀疏奖励切入，讲清 On-Policy Distillation 如何在学生自身轨迹上对齐教师；[完全指南](online-policy-distillation/guide.html)提供完整数学推导、伪代码与调参落地。

📅 2026-07–08 ｜ 🖼 3 图 ｜ 📄 [Markdown](online-policy-distillation/report.md)

### [Hermes Agent API 指南](hermes/API.md)

本地代码/IDE 如何调用远端 Hermes 完整 Agent 能力：一张表厘清 `proxy` / `serve` / `dashboard` / `api_server` 四种方式，给代码用的是 OpenAI 兼容的 API Server。来源为 Hermes 源码 + 官方文档 + 实践验证。

📅 2026-08-24 ｜ 📄 [Markdown](hermes/API.md)
