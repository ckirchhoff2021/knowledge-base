<div align="center">

# 📚 AI 技术知识库
个人AI技术调研、论文分析、Agent技术、技能开发相关的**高质量技术文档知识库**，所有文档同时提供Markdown源文件和精排版HTML文件，支持在线直接预览。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python&style=for-the-badge" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Docs-11+-orange?style=for-the-badge" alt="Docs">
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

## 🌐 快速访问
所有文档已开启GitHub Pages在线预览，**无需下载、无需环境，点击直接阅读**：
> 🔗 知识库首页：[https://ckirchhoff2021.github.io/knowledge-base/](https://ckirchhoff2021.github.io/knowledge-base/)
>
> 访问规则：`https://ckirchhoff2021.github.io/knowledge-base/[分类]/html/[文档名].html`

---

## 📁 目录结构
```
knowledge-base/
├── 🤖 Agent/                 # 智能体、强化学习、Agent技术相关文档
│   ├── markdown/             # Markdown源文件（可编辑）
│   ├── html/                 # 精排版HTML（直接阅读）
│   └── assets/               # 图片、图表等资源
├── 🧠 PT/                    # 大模型预训练、后训练、RLHF、论文分析
│   ├── markdown/
│   └── html/
├── 🛠️ Skills/                # 技能开发、工具链、工程实践
│   ├── markdown/
│   └── html/
└── 🌱 Life/                  # 心态成长、情绪管理、生活方法论
    ├── markdown/
    ├── html/
    └── assets/
```

---

## 📖 文档列表
### 🤖 Agent 智能体技术
<table>
  <tr>
    <th>📄 文档名称</th>
    <th>🏷️ 类型</th>
    <th>📅 更新时间</th>
    <th>📝 说明</th>
    <th>🔗 链接</th>
  </tr>
  <tr>
    <td>Claude-Tag-技术调研报告</td>
    <td>技术深度调研</td>
    <td>2026-08-19</td>
    <td>Claude Tag（Anthropic 团队协作型 Agent）深度调研：四大核心能力、Agent Identity、运行时架构（Session/Harness/Sandbox）、落地实践与风险，含3张交互架构图</td>
    <td align="center"><a href="https://ckirchhoff2021.github.io/knowledge-base/Agent/html/Claude-Tag-技术调研报告.html"><b>🔗 在线阅读</b></a></td>
  </tr>
  <tr>
    <td>Claude-Tag-技术栈拆解与算法研究点分析</td>
    <td>技术深度调研</td>
    <td>2026-08-22</td>
    <td>Claude Tag八层技术栈拆解、算法密度标注、记忆管线与研究点优先级分析，含3张交互架构图</td>
    <td align="center"><a href="https://ckirchhoff2021.github.io/knowledge-base/Agent/html/Claude-Tag-技术栈拆解与算法研究点分析.html"><b>🔗 在线阅读</b></a></td>
  </tr>
  <tr>
    <td>Claude-Tag-开源实现盘点</td>
    <td>开源项目盘点</td>
    <td>2026-08-21</td>
    <td>Claude Tag主流开源复刻实现盘点（Open Tag/MFS等），GitHub实测数据</td>
    <td align="center"><a href="https://ckirchhoff2021.github.io/knowledge-base/Agent/html/Claude-Tag-开源实现盘点.html"><b>🔗 在线阅读</b></a></td>
  </tr>
  <tr>
    <td>拟人化Agent产品设计调研报告</td>
    <td>技术深度调研</td>
    <td>2026-08-24</td>
    <td>~16000字，拟人化Agent五大设计维度、六层架构、心理学与HCI理论框架，含交互架构图</td>
    <td align="center"><a href="https://ckirchhoff2021.github.io/knowledge-base/Agent/html/拟人化Agent产品设计调研报告.html"><b>🔗 在线阅读</b></a></td>
  </tr>
  <tr>
    <td>拟人化Agent设计维度理论框架调研笔记</td>
    <td>调研笔记</td>
    <td>2026-08-24</td>
    <td>拟人化Agent设计理论素材：被理解感、连续人格、社交分寸等设计维度理论支撑</td>
    <td align="center"><a href="https://ckirchhoff2021.github.io/knowledge-base/Agent/html/拟人化Agent设计维度理论框架调研笔记.html"><b>🔗 在线阅读</b></a></td>
  </tr>
  <tr>
    <td>飞书数字员工记忆系统设计技术报告</td>
    <td>技术深度调研</td>
    <td>2026-08-21</td>
    <td>~14000字，飞书数字员工分层记忆系统设计：常驻层/检索层/蒸馏层、双通道写入、预算检索注入，含总体架构图</td>
    <td align="center"><a href="https://ckirchhoff2021.github.io/knowledge-base/Agent/html/飞书数字员工记忆系统设计技术报告.html"><b>🔗 在线阅读</b></a></td>
  </tr>
  <tr>
    <td>OPD-OPSD技术调研报告</td>
    <td>技术深度调研</td>
    <td>2026-08-18</td>
    <td>~3万字，StepOPSD步骤级蒸馏、Agent RL核心方案深度解析，含3张交互架构图</td>
    <td align="center"><a href="https://ckirchhoff2021.github.io/knowledge-base/Agent/html/OPD-OPSD-技术调研报告.html"><b>🔗 在线阅读</b></a></td>
  </tr>
  <tr>
    <td>OPD在线策略蒸馏技术完全指南</td>
    <td>技术落地指南</td>
    <td>2026-08-14</td>
    <td>~25000字，含数学推导、Mermaid架构图、可运行PyTorch伪代码、落地路径、调参指南、常见问题</td>
    <td align="center"><a href="https://ckirchhoff2021.github.io/knowledge-base/Agent/html/OPD在线策略蒸馏技术完全指南.html"><b>🔗 在线阅读</b></a></td>
  </tr>
</table>

### 🧠 PT 大模型技术
<table>
  <tr>
    <th>📄 文档名称</th>
    <th>🏷️ 类型</th>
    <th>📅 更新时间</th>
    <th>📝 说明</th>
    <th>🔗 链接</th>
  </tr>
  <tr>
    <td>2605.23904-SkillOpt技术分析报告</td>
    <td>arXiv论文解析</td>
    <td>2026-06-03</td>
    <td>大模型技能自动优化技术论文深度分析</td>
    <td align="center"><a href="https://ckirchhoff2021.github.io/knowledge-base/PT/html/2605.23904-SkillOpt技术分析报告.html"><b>🔗 在线阅读</b></a></td>
  </tr>
</table>

### 🛠️ Skills 技能开发
> 🔄 持续更新中...

### 🌱 Life 心态成长
<table>
  <tr>
    <th>📄 文档名称</th>
    <th>🏷️ 类型</th>
    <th>📅 更新时间</th>
    <th>📝 说明</th>
    <th>🔗 链接</th>
  </tr>
  <tr>
    <td>小白理财技术方案-从零开始的资产配置系统</td>
    <td>理财方法论</td>
    <td>2026-09-02</td>
    <td>基于知乎社区高赞讨论的系统化梳理：复利/定投数学原理、四层资金池架构、五阶段落地路线、防骗决策树伪代码，含3张技术插图与14个参考来源</td>
    <td align="center"><a href="https://ckirchhoff2021.github.io/knowledge-base/Life/html/小白理财技术方案-从零开始的资产配置系统.html"><b>🔗 在线阅读</b></a></td>
  </tr>
  <tr>
    <td>情绪修复与心态调整指南</td>
    <td>心态方法论</td>
    <td>2026-08-27</td>
    <td>情绪ABC模型与Gross调节、四阶段全天应对流程、应对糟心的人的边界策略，含3张交互流程图</td>
    <td align="center"><a href="https://ckirchhoff2021.github.io/knowledge-base/Life/html/情绪修复与心态调整指南.html"><b>🔗 在线阅读</b></a></td>
  </tr>
</table>

---

## 📝 文档规范
所有技术文档严格按照六段式标准结构编写，保证内容质量：
1. **问题背景**：技术出现的背景、解决的痛点、应用场景
2. **核心原理**：数学推导、理论基础、本质分析
3. **系统架构**：整体架构图、模块说明、数据流
4. **方案详情**：实现细节、工程优化、调参指南、常见问题
5. **伪代码/实现路径**：可运行代码、分阶段落地方案
6. **思考展望**：优劣对比、前沿方向、落地建议

---

## 🤝 说明
- 文档持续更新中，欢迎Star收藏关注
- 所有HTML文档使用`tech-doc`专业技术主题生成，排版美观、阅读体验好
- 转载请注明来源
