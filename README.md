<div align="center">

# 📚 AI 技术知识库
个人AI技术调研、论文分析、Agent技术、大模型研究、工具开发相关的**高质量技术文档知识库**，所有文档同时提供Markdown源文件和精排版HTML文件，支持在线直接预览。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python&style=for-the-badge" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Docs-13+-orange?style=for-the-badge" alt="Docs">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" alt="Status">
  <a href="https://ckirchhoff2021.github.io/knowledge-base/"><img src="https://img.shields.io/badge/🌐-在线预览-blueviolet?style=for-the-badge" alt="Online Preview"></a>
</p>

---

</div>

## ✨ 特性
| 🎯 高质量内容 | 🌗 深色模式 | 📱 响应式 | 🧮 公式支持 | 📊 图表渲染 |
|:---:|:---:|:---:|:---:|:---:|
| 所有文档均为深度调研，包含原理推导、架构图、伪代码、落地指南 | 自动跟随系统切换浅色/深色模式，夜间阅读护眼 | 完美适配PC/平板/手机，移动端自动优化布局 | 完整KaTeX数学公式渲染，支持行内/块级公式 | 自动渲染Mermaid架构图/流程图/时序图 |

---

## 📁 目录结构

每个话题一个独立文件夹，内含 `.md` 源文件、`.html` 精排版、以及 `diagrams/` 图表资源，方便阅读和维护：

```
knowledge-base/
├── 🤖 agents/                    # Agent 技术调研
│   ├── claude-tag/               # Claude Tag 协作型 Agent（调研+技术栈+开源实现）
│   ├── anthropomorphic-agent/    # 拟人化 Agent 产品设计（产品调研+理论框架）
│   ├── feishu-memory/            # 飞书数字员工记忆系统设计
│   ├── online-policy-distillation/ # OPD/OPSD 在线策略蒸馏技术
│   └── hermes/                   # Hermes Agent API 文档
├── 🧠 models/                    # 大模型技术与论文分析
│   ├── jev-decision/             # TypeSafe JEV 决策小模型调研报告
│   └── skillopt/                 # SkillOpt 技能优化论文分析
└── 🌱 life/                      # 心态成长、健康、生活方法论
    ├── emotion-regulation/       # 情绪修复与心态调整
    ├── insomnia-treatment/       # 长期失眠就医指南
    ├── mindfulness/              # 心平气和与感知力提升
    └── wealth-management/        # 小白理财资产配置系统
```

---

## 📖 文档列表

### 🤖 Agent 智能体技术
<table>
  <tr><th>📄 文档</th><th>🏷️ 类型</th><th>📅 更新</th><th>📝 说明</th><th>🔗</th></tr>
  <tr>
    <td>Claude-Tag 技术调研报告</td>
    <td>技术深度调研</td><td>2026-08-19</td>
    <td>Anthropic Claude Tag 协作型 Agent：四大能力、Agent Identity、运行时架构、Session/Harness 设计</td>
    <td><a href="https://ckirchhoff2021.github.io/knowledge-base/agents/claude-tag/report.html">📖</a></td>
  </tr>
  <tr>
    <td>Claude-Tag 技术栈拆解与算法研究点</td>
    <td>技术分析</td><td>2026-08-23</td>
    <td>Claude Tag 底层技术栈逐层拆解与算法研究点全景图</td>
    <td><a href="https://ckirchhoff2021.github.io/knowledge-base/agents/claude-tag/tech-stack.html">📖</a></td>
  </tr>
  <tr>
    <td>Claude-Tag 开源实现盘点</td>
    <td>开源盘点</td><td>2026-08-21</td>
    <td>Claude Tag 类开源项目对比与选型建议</td>
    <td><a href="https://ckirchhoff2021.github.io/knowledge-base/agents/claude-tag/open-source.html">📖</a></td>
  </tr>
  <tr>
    <td>拟人化 Agent 产品设计调研报告</td>
    <td>产品调研</td><td>2026-06-20</td>
    <td>拟人化 Agent 设计：工具型→助手型→伙伴型三阶段演化，心理学基础与产品形态</td>
    <td><a href="https://ckirchhoff2021.github.io/knowledge-base/agents/anthropomorphic-agent/product-design.html">📖</a></td>
  </tr>
  <tr>
    <td>拟人化 Agent 设计维度理论框架</td>
    <td>理论框架</td><td>2026-06-20</td>
    <td>外在拟人化、语言、认知、情感、社交五维设计框架</td>
    <td><a href="https://ckirchhoff2021.github.io/knowledge-base/agents/anthropomorphic-agent/design-framework.html">📖</a></td>
  </tr>
  <tr>
    <td>飞书数字员工记忆系统设计</td>
    <td>系统设计</td><td>2026-08-19</td>
    <td>飞书 Claude Tag 记忆系统架构、云端记忆、会话隔离设计</td>
    <td><a href="https://ckirchhoff2021.github.io/knowledge-base/agents/feishu-memory/report.html">📖</a></td>
  </tr>
  <tr>
    <td>OPD/OPSD 技术调研报告</td>
    <td>技术深度调研</td><td>2026-07-15</td>
    <td>在线策略蒸馏（Online Policy Distillation）技术原理、架构与落地实践</td>
    <td><a href="https://ckirchhoff2021.github.io/knowledge-base/agents/online-policy-distillation/report.html">📖</a></td>
  </tr>
  <tr>
    <td>OPD 在线策略蒸馏完全指南</td>
    <td>实战指南</td><td>2026-07-18</td>
    <td>OPD 从零部署到优化的完整工程指南</td>
    <td><a href="https://ckirchhoff2021.github.io/knowledge-base/agents/online-policy-distillation/guide.html">📖</a></td>
  </tr>
  <tr>
    <td>Agent-RL与Harness-RL技术调研报告</td>
    <td>技术深度调研</td>
    <td>2026-09-23</td>
    <td>~22000字，LLM Agent强化学习两条路线（传统Agent-RL vs Harness-Native RL）系统调研：训推不一致根因、OpenForgeRL/CoreCraft/GRPO/SAO/AT-GRPO算法对比、基础设施选型（verl/Miles/AReaL）、落地checklist，含3张暗色主题交互架构图</td>
    <td align="center"><a href="https://ckirchhoff2021.github.io/knowledge-base/Agent/html/Agent-RL与Harness-RL技术调研报告.html"><b>🔗 在线阅读</b></a></td>
  </tr>
</table>

### 🧠 大模型技术
<table>
  <tr><th>📄 文档</th><th>🏷️ 类型</th><th>📅 更新</th><th>📝 说明</th><th>🔗</th></tr>
  <tr>
    <td>JEV 决策小模型技术调研报告</td>
    <td>技术深度调研</td><td>2026-09-21</td>
    <td>TypeSafe JEV 决策模型：System One 架构、类型安全约束解码、决策流水线、Agent 集成生态与选型建议</td>
    <td><a href="https://ckirchhoff2021.github.io/knowledge-base/models/jev-decision/report.html">📖</a></td>
  </tr>
  <tr>
    <td>SkillOpt 技术分析（2605.23904）</td>
    <td>论文分析</td><td>2026-06-29</td>
    <td>SkillOpt：通过轨迹驱动编辑训练可复用自然语言技能的文本空间优化器</td>
    <td><a href="https://ckirchhoff2021.github.io/knowledge-base/models/skillopt/report.html">📖</a></td>
  </tr>
</table>

### 🌱 生活方法论
<table>
  <tr><th>📄 文档</th><th>🏷️ 类型</th><th>📅 更新</th><th>📝 说明</th><th>🔗</th></tr>
  <tr>
    <td>情绪修复与心态调整指南</td>
    <td>方法论</td><td>2026-05-10</td>
    <td>基于情绪调节理论的日常心态修复协议与 toxic people 应对策略</td>
    <td><a href="https://ckirchhoff2021.github.io/knowledge-base/life/emotion-regulation/guide.html">📖</a></td>
  </tr>
  <tr>
    <td>长期失眠就医全指南</td>
    <td>健康指南</td><td>2026-04-10</td>
    <td>失眠症状分析、治疗流程、杭州医院对比与异地医保方案</td>
    <td><a href="https://ckirchhoff2021.github.io/knowledge-base/life/insomnia-treatment/guide.html">📖</a></td>
  </tr>
  <tr>
    <td>提升感知力与心平气和指南</td>
    <td>方法论</td><td>2026-03-15</td>
    <td>情绪不受外界干扰：提升感知力与心平气和的完整方法论</td>
    <td><a href="https://ckirchhoff2021.github.io/knowledge-base/life/mindfulness/guide.html">📖</a></td>
  </tr>
  <tr>
    <td>小白理财资产配置系统</td>
    <td>技术方案</td><td>2026-02-20</td>
    <td>从零开始的个人资产配置系统设计与学习路径</td>
    <td><a href="https://ckirchhoff2021.github.io/knowledge-base/life/wealth-management/guide.html">📖</a></td>
  </tr>
</table>

---

## 🌐 在线预览
所有文档已开启 GitHub Pages 在线预览，**无需下载、无需环境，点击直接阅读**：
> 🔗 知识库首页：[https://ckirchhoff2021.github.io/knowledge-base/](https://ckirchhoff2021.github.io/knowledge-base/)
>
> 访问规则：`https://ckirchhoff2021.github.io/knowledge-base/[分类]/[话题]/[文件名].html`
