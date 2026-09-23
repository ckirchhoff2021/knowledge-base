# OPD / OPSD 在线策略蒸馏

> 后训练第三条路：在学生模型自身的轨迹分布上蒸馏教师，兼顾密集 Token 监督与在线 RL 的同策略对齐。

## 文档

### [OPD 与 OPSD 技术调研报告](report.md)

🌐 [在线阅读](https://ckirchhoff2021.github.io/knowledge-base/agents/online-policy-distillation/report.html) ｜ 📅 2026-08-18 ｜ 🖼 3 张架构图

从「预训练 → SFT → RL」的三元困境切入，系统梳理 On-Policy Distillation 与 On-Policy Self-Distillation 的原理：离线方法的暴露偏差 vs 在线 RL 的稀疏奖励与信用分配，OPD 如何用 KL 对齐 + on-policy 采样弥合二者，以及在 Agent、代码、数学推理等长程场景的工业落地（20+ 篇知乎文章与原始论文交叉验证）。

### [OPD 在线策略蒸馏完全指南](guide.md)

🌐 [在线阅读](https://ckirchhoff2021.github.io/knowledge-base/agents/online-policy-distillation/guide.html) ｜ 📅 2026-07

工程向实战手册：完整数学推导、架构设计、可运行伪代码、调参指南与落地路径，覆盖从部署到优化的全流程。

`OPD` `OPSD` `后训练` `知识蒸馏` `强化学习`
