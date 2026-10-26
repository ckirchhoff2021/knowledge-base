# 🧠 大模型技术报告

大模型与后训练方向的论文分析，共 25 个话题。每个话题为自包含文件夹（`.md` 源文件 + 精排 `.html` + `diagrams/`）。

---

## 架构基础

| 报告 | 说明 | 更新时间 |
|---|---|---|
| [Transformer × 位置编码](transformer-pe/report.html) | 架构全解与长上下文扩展 | 2026-09-30 |
| [线性注意力与混合架构](linear-attention/report.html) | 线性 RNN 视角与 Mamba 等 | 2026-10-15 |
| [MoE 稀疏混合专家](moe-architecture/report.html) | 容量与计算解耦 | 2026-10-22 |
| [扩散语言模型 DLM](diffusion-language-models/report.html) | 非自回归生成新范式 | 2026-10-11 |

## 训练与算法

| 报告 | 说明 | 更新时间 |
|---|---|---|
| [Post-Training 后训练](post-training/report.html) | SFT / RLHF / DPO / GRPO 全景 | 2026-09-24 |
| [大模型强化学习算法](llm-rl-algorithms/report.html) | PPO 与 GRPO 家族详解 | 2026-09-30 |
| [Agentic RL 智能体强化学习](agentic-rl/report.html) | 多轮环境下的强化学习 | 2026-10-10 |
| [合成数据与修正缩放定律](synthetic-data/report.html) | 模型崩溃与合成数据方法 | 2026-10-19 |

## Jev 决策模型

| 报告 | 说明 | 更新时间 |
|---|---|---|
| [JEV 决策小模型](jev-decision/report.html) | System One 决策模型调研 | 2026-09-21 |
| [Jev 训练方案 · Qwen3.5-4B](jev-qwen35-4b/report.html) | 4B 基座复现方案 | 2026-09-28 |
| [开源 Jev 训练方法全景](open-source-jev-training/report.html) | 开源模型训练方法横评 | 2026-10-20 |
| [开源 Jev 模型全景](open-source-jev-landscape/report.html) | 开源模型目录更新版 | 2026-10-26 |

## 系统与应用

| 报告 | 说明 | 更新时间 |
|---|---|---|
| [SkillOpt 技能优化](skillopt/report.html) | 文本空间技能优化器 | 2026-06-29 |
| [世界模型 World Models](world-models/report.html) | 预测世界下一状态 | 2026-10-12 |
| [智能体记忆 Agent Memory](agent-memory/report.html) | 终身学习记忆架构 | 2026-10-13 |
| [Agent Memory 系统设计](agent-memory-systems/report.html) | Hermes / Mem0 / Codex 记忆架构 | 2026-10-26 |
| [视觉-语言-动作模型 VLA](vla-embodied/report.html) | 具身智能模型 | 2026-10-14 |
| [智能体互联协议栈 MCP/A2A](agent-interop/report.html) | MCP / A2A / AGENTS.md | 2026-10-16 |
| [AI Scientist 自主科研](ai-scientist/report.html) | 自主科研智能体 | 2026-10-17 |
| [机制可解释性与欺骗检测](mech-interp/report.html) | 机制解释与 AI 安全 | 2026-10-18 |
| [大模型推理优化](inference-optimization/report.html) | 量化 / 缓存 / PD 分离 | 2026-10-20 |
| [AI 编译器与异构底座](ai-compiler-stack/report.html) | Kernel 与芯片全栈 | 2026-10-21 |
| [GUI Agent 电脑操作](gui-agent/report.html) | 电脑操作智能体 | 2026-10-23 |
| [Test-Time Scaling](test-time-scaling/report.html) | 测试时计算扩展 | 2026-10-25 |
| [全双工实时语音对话模型](full-duplex-speech/report.html) | 全双工实时语音对话 | 2026-10-26 |
