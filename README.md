<div align="center">

# 📚 AI 技术知识库

个人技术调研与论文分析合集 —— 深度、一手核实、可在线阅读的 **Agent · 大模型 · 生活方法论** 报告

[![Online Preview](https://img.shields.io/badge/🌐-GitHub_Pages-blueviolet?style=for-the-badge)](https://ckirchhoff2021.github.io/knowledge-base/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📂 报告导航

### 🤖 Agent 智能体

| 报告 | 说明 |
|---|---|
| [**Agent-RL × Harness-RL**](https://ckirchhoff2021.github.io/knowledge-base/agents/agent-rl-harness-rl/report.html) | Agent 强化学习两条路线：训推不一致根因、算法与基础设施选型 |
| [**Self-Evolving Agents × Harness-Native RL**](https://ckirchhoff2021.github.io/knowledge-base/agents/self-evolving-harness-rl/report.html) | 模型 θ × Harness η 双空间协同进化：HAT、Cross-Harness GRPO、SkillOpt、RRSI |
| [**Loop Engineering**](https://ckirchhoff2021.github.io/knowledge-base/agents/loop-engineering/report.html) | 从写提示词到设计循环：LoopSpec、四类循环、builder/checker |
| [**Claude Tag**](https://ckirchhoff2021.github.io/knowledge-base/agents/claude-tag/report.html) | Anthropic「AI 同事」：产品调研 + 技术栈拆解 + 开源盘点（3 篇） |
| [**飞书数字员工记忆系统**](https://ckirchhoff2021.github.io/knowledge-base/agents/feishu-memory/report.html) | 五层记忆 + 四级隔离的持续学习架构 |
| [**拟人化 Agent 产品设计**](https://ckirchhoff2021.github.io/knowledge-base/agents/anthropomorphic-agent/product-design.html) | 工具 → 伙伴：拟人化心理学基础与五维设计框架（2 篇） |
| [**OPD / OPSD 在线策略蒸馏**](https://ckirchhoff2021.github.io/knowledge-base/agents/online-policy-distillation/report.html) | 后训练第三条路：on-policy 蒸馏（调研 + 完全指南 2 篇） |
| [**Hermes Agent API 指南**](agents/hermes/API.md) | 本地调用远端 Hermes 完整 Agent 能力的四种方式 |
| [**大模型制作短剧完全指南**](https://ckirchhoff2021.github.io/knowledge-base/agents/ai-short-drama/report.html) | 一句话创意到成片七层流水线：剧本结构化 JSON/黄金三秒 · 角色一致性四代方案（参考图/首尾帧/LoRA）· I2V 模型横评（可灵/Seedance/Veo/Wan）· 先配音后定画面/音画一体 · 多 Agent 电影系统（MovieDreamer/MovieAgent/MAViS）· 成本 1300–3000 元 vs 真人 50 万 · 爆款率 0.48% 现实 · 选型 Checklist |

### 🧠 大模型技术

| 报告 | 说明 |
|---|---|
| [**JEV 决策小模型**](https://ckirchhoff2021.github.io/knowledge-base/models/jev-decision/report.html) | System One 架构、类型安全约束解码与决策流水线 |
| [**SkillOpt 论文分析**](https://ckirchhoff2021.github.io/knowledge-base/models/skillopt/report.html) | 轨迹驱动的文本空间技能优化器 |
| [**Post-Training 后训练技术**](https://ckirchhoff2021.github.io/knowledge-base/models/post-training/report.html) | 后训练全景：SFT、RLHF、DPO 家族、RLAIF/CAI、RLVR/GRPO 与 R1 冷启动—蒸馏范式 |
| [**Jev 训练方案 · Qwen3.5-4B**](https://ckirchhoff2021.github.io/knowledge-base/models/jev-qwen35-4b/report.html) | 4B 基座复现 Jev-like：LoRA SFT + 概率校准 + 约束解码，含数据/训练/转换/级联全流程 |
| [**Transformer × 位置编码**](https://ckirchhoff2021.github.io/knowledge-base/models/transformer-pe/report.html) | 架构全解：RoPE/ALiBi 原理取舍与 PI/NTK/YaRN 长上下文扩展 |
| [**大模型强化学习算法**](https://ckirchhoff2021.github.io/knowledge-base/models/llm-rl-algorithms/report.html) | PPO 深度拆解 · GRPO/DAPO/Dr.GRPO/GSPO · RLVR 与 R1 流程 |
| [**Agentic RL 智能体强化学习**](https://ckirchhoff2021.github.io/knowledge-base/models/agentic-rl/report.html) | 多轮环境训练：回声陷阱/模板坍塌/交互坍塌 · GiGPO/T²PO/DART · 异步RL系统 |
| [**扩散语言模型 DLM**](https://ckirchhoff2021.github.io/knowledge-base/models/diffusion-language-models/report.html) | 非自回归新范式：LLaDA/Mercury/Cola/DiffusionGemma · Block-AR · 采样器压缩 · SD·RL，单 H100 约 1500 TPS |
| [**世界模型 World Models**](https://ckirchhoff2021.github.io/knowledge-base/models/world-models/report.html) | 从预测下一个词到预测世界下一状态（NSP）：五大路线 · Genie 3/V-JEPA 2/Dreamer 4 · 世界模型×VLA · Cosmos/悟界 |
| [**智能体记忆 Agent Memory**](https://ckirchhoff2021.github.io/knowledge-base/models/agent-memory/report.html) | 从无状态对话到终身学习：四类认知记忆 · 六代架构（MemGPT/Mem0/Zep/A-MEM）· MemOS · 自进化记忆 |
| [**视觉-语言-动作模型 VLA**](https://ckirchhoff2021.github.io/knowledge-base/models/vla-embodied/report.html) | 从「描述世界」到「改变世界」：四代源流 · 动作表示光谱与三类生成头 · 双系统架构 · 跨本体泛化 · π0.5/Gemini Robotics/GR00T/OpenVLA · 人形机器人产业地图 |
| [**线性注意力与混合架构**](https://ckirchhoff2021.github.io/knowledge-base/models/linear-attention/report.html) | 当注意力不再「二次」：线性注意力=可并行训练的线性 RNN · Mamba/SSD 统一视角 · 五类状态更新规则（→KDA）· Kimi Linear/MiniMax-01/Jamba/RWKV-7 横评 · 跨架构蒸馏 HALO/PostNAS · KV 缓存省 75%、1M 上下文吞吐 ×6 |
| [**智能体互联协议栈 MCP/A2A**](https://ckirchhoff2021.github.io/knowledge-base/models/agent-interop/report.html) | Agent 时代的「TCP/IP」：N×M→N+M · 三层协议栈（AGENTS.md/MCP/A2A）· MCP Host-Client-Server 与三原语 · A2A Agent Card 与 Task 七态 · 工具投毒/Rug Pull/致命三要素/CVE · MCP 捐赠 Linux 基金会 AAIF |
| [**AI Scientist 自主科研智能体**](https://ckirchhoff2021.github.io/knowledge-base/models/ai-scientist/report.html) | 智源趋势第 4 位：从科研助手到自主科学发现 · 六层流水线与智能体树搜索 BFTS · Sakana/Co-Scientist/FutureHouse · 自驱动实验室干湿双闭环 · 科学基础模型版图 · 2026-09 声明证据审计（ART/费马/Navier-Stokes）· 生物安全 |
| [**机制可解释性与 AI 欺骗检测**](https://ckirchhoff2021.github.io/knowledge-base/models/mech-interp/report.html) | 智源趋势第 10 位：从幻觉到系统性欺骗 · 电路/叠加假说/IOI · SAE 撬开叠加（Claude 3 Sonnet 3400 万特征）· Circuit Tracing · 欺骗三形态（Sleeper Agents/Alignment Faking/In-context Scheming，勒索 84%、o1 追问 85% 维持欺骗）· 2026 事件审计（Hugging Face/Medicare/AISI 29.2%/Astra 叫停/FTC）· 五层防御栈 |
| [**合成数据与修正缩放定律**](https://ckirchhoff2021.github.io/knowledge-base/models/synthetic-data/report.html) | 智源趋势第 7 位：破除「2026 数据枯竭魔咒」（Epoch AI：人类文本约 300T token、2026–2032 耗尽）· 模型崩溃 Nature 理论与抗崩溃边界（累积真实锚点/多源/可执行验证）· 修正缩放定律（SynthLLM：收益约 300B token 后递减、8B 约 1T 达峰）· 六大方法族（Self-Improve/STaR/RFT/Self-Rewarding/SPIN/rStar-Math/Self-Instruct/Magpie/phi/Orca/R1 蒸馏/CodeT/DCLM）· 世界模型仿真（Genie/GameNGen/DriveDreamer/GenSim/Cosmos）· Nemotron 对齐数据 98% 合成 · 版权判例与 EU AI Act 披露 · 防污染评测 |
| [**大模型推理优化**](https://ckirchhoff2021.github.io/knowledge-base/models/inference-optimization/report.html) | 智源趋势第 8 位：推理成本两年降 280 倍 vs 推理约占数据中心收入 40% 的剪刀差 · Prefill/Decode 两阶段与内存墙 · 权重量化（SmoothQuant/GPTQ/AWQ/QServe）· KV Cache 治理（H₂O/StreamingLLM/KIVI/MLA 压缩 93.3%/ShadowKV）· FlashAttention 三代 · PagedAttention/连续批处理/Chunked Prefill · 投机解码家族（EAGLE 三代 6.5×/Medusa/Lookahead/MTP）· DistServe/Splitwise/Mooncake PD 分离 · 六大引擎横评 · FP8/FP4 与端侧推理 |
| [**AI 编译器与异构全栈底座**](https://ckirchhoff2021.github.io/knowledge-base/models/ai-compiler-stack/report.html) | 智源趋势第 9 位：CUDA 开发者一年由 590 万增至 750 万的护城河 · MLIR 多层方言 · Triton TTIR→TTGIR→PTX 与 TVM/Relax 两条 lowering 流水线 · 算子融合/布局/张量化 · 算子语言光谱（CuTe/ThunderKittens/TileLang/Exo，tile 级甜点层）· Ansor/Roller/PET 自动调优 · Alpa/GSPMD 自动并行 · LLM/Agent 自动写 Kernel（KernelBench/CuTeGen/CUDA Agent L3 更快率 92%/AKG 跨 GPU+NPU）· ROCm 10.1/TheRock · 昇腾 CANN 等国产栈盘点 · FlagOS 2.2 深度拆解（FlagTree 15+ 后端、Day0 适配 8–9 款芯片）· UALink/UEC · H20 管制时间线 |
| [**稀疏混合专家 MoE 架构**](https://ckirchhoff2021.github.io/knowledge-base/models/moe-architecture/report.html) | 当容量与计算解耦：容量∝N、计算∝k · 九年时间线（Shazeer 137B→Switch 1.6T→Mixtral/DeepSeek→Kimi K2 1T）· 路由六范式（token-choice/BASE/Hash/expert-choice/Soft MoE/MoD/免训练特征向量路由）· 细粒度分割+共享专家 · 负载均衡（辅助损失→DeepSeek 无辅助损失 bias→控制论 PID 统一）· DeepEP 专家并行 all-to-all · EPLB 冗余专家 · Upcycling · 推理显存墙与专家卸载/量化 · 缩放定律与「无限参数」猜想 · 19 款旗舰横评（671B/37B、1.04T/32.6B） |
| [**GUI Agent 电脑操作智能体**](https://ckirchhoff2021.github.io/knowledge-base/models/gui-agent/report.html) | 当大模型学会「看屏幕、点按钮」：四年四代（读 HTML→GPT-4V grounding→原生统一模型→RL 智能体）· GUI Grounding 纯视觉 vs 可访问性树（ScreenSpot/SoM/OmniParser）· 原生一体（UI-TARS 1/2、CogAgent、Aguvis、ShowUI、Qwen/Seed/Kimi/AutoGLM）vs 通用-专家模块化（Agent S/S2/BJudge 首超人类 72.6%/S4 接口自演化）· 记忆三层抽象（子目标图/ScopeMem/Trace2World 91.1%）· SFT→多轮 RL 与奖励危机（提示判官 68–70% 天花板、VeriCUA、自述泄漏盲判）· GUI/代码统一 OS-Helix、全模态、Dual Exam、医疗垂直、开放全栈、NPU 端侧 · 提示注入攻防与 2026-07–10 真实越权事件链（OpenAI 逃逸入侵 HF、Anthropic 切断评测公网、CERT-In）· OSWorld 14.9%→83.4% 与独立复测 |

| [**开源 Jev 模型训练方法全景**](https://ckirchhoff2021.github.io/knowledge-base/models/open-source-jev-training/report.html) | 已发布权重的开源 Jev-like 决策模型怎么训：统一判别架构（置换等变候选头，noul/choice/score 共用一头）· 三大范式——软标签 SFT（AgentJev-0.6B：soft CE+Brier/全参 bf16/独立校准，79.25%）、RLCD 校准强化学习（Laya 421M：Gaussian 探索+proper scoring rule+REINFORCE，76.6–77%）、真实测试结果领域续训（coding-gates：2100 次真实运行+受约束选点，门禁 100%）· 零样本派/LoRA 专家/log-prior · 10 月 TreeEncoder 共享前缀树（P50 278→44ms）· 污染事故与教师天花板 0.735 的纪律 · 选型 Checklist |

### 🌱 生活方法论

| 报告 | 说明 |
|---|---|
| [**情绪修复与心态调整**](https://ckirchhoff2021.github.io/knowledge-base/life/emotion-regulation/guide.html) | 日常心态修复协议与 toxic people 应对 |
| [**长期失眠就医指南**](https://ckirchhoff2021.github.io/knowledge-base/life/insomnia-treatment/guide.html) | 症状、治疗流程、杭州医院与异地医保 |
| [**提升感知力与心平气和**](https://ckirchhoff2021.github.io/knowledge-base/life/mindfulness/guide.html) | 情绪不受外界干扰的完整方法论 |
| [**小白理财资产配置**](https://ckirchhoff2021.github.io/knowledge-base/life/wealth-management/guide.html) | 从零开始的资产配置系统与学习路径 |
| [**小白理财实战手册**](https://ckirchhoff2021.github.io/knowledge-base/life/wealth-practice/guide.html) | 照着做的一年行动方案：开户、预算、应急金、保险、定投到止盈的完整操作 |

---

## 📁 目录约定

每个话题一个**自包含文件夹**：

```
agents/<topic>/
├── report.md        # Markdown 源文件
├── report.html      # 暗色精排版（含目录、图表）
├── README.md        # 本话题报告摘要
├── diagrams/        # 架构图源文件
└── fonts/           # 报告内嵌字体（按需）
```

## 🌐 在线阅读

全部报告经 GitHub Pages 发布，无需下载：

```
https://ckirchhoff2021.github.io/knowledge-base/[分类]/[话题]/report.html
```

> 知识库首页：[ckirchhoff2021.github.io/knowledge-base](https://ckirchhoff2021.github.io/knowledge-base/)

---

<div align="center">

Distributed under the **MIT License** ｜ 核心事实经一手信源核实，转述数据在正文显式标注

</div>
