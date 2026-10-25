# 🧠 大模型技术报告

大模型与后训练方向的论文分析，共 22 个话题。

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

### [大模型常用强化学习算法](llm-rl-algorithms/report.html)

从策略梯度（Policy Gradient）与 MDP 建模出发，深度剖析大模型 RL 算法：经典基线 **PPO**（重要性采样、clip 信任域、GAE、Critic、KL 惩罚），Critic-Free 家族 **RLOO / ReMax / REINFORCE++**，以及当代主流 **GRPO**（组相对优势、token clip、KL 无偏估计）与其改进变体 **DAPO**（clip-higher、动态采样、token-level loss、超长软奖励）、**Dr.GRPO**（去除长度膨胀偏置）、**GSPO**（序列级优化，稳 MoE）。并梳理 RLHF/RLAIF/RLVR 三种奖励范式、reward hacking 对策与 DeepSeek-R1 多阶段流程。信源为 13 篇 arXiv 原始论文（编号经 arXiv API 逐一核实）与知乎高赞文章交叉验证。

📅 2026-09-30 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](llm-rl-algorithms/report.md)

### [Agentic RL 智能体强化学习](agentic-rl/report.html)

聚焦多轮环境下的大模型强化学习：从单轮 RLVR 到 Agentic RL 的五个本质差异（轨迹级优化、极稀疏延迟奖励、有状态随机环境、长 rollout、天然 off-policy）；系统梳理七类训练病理——**回声陷阱**（奖励方差悬崖+梯度尖峰，RAGEN）、**模板坍塌**（推理与输入无关，RAGEN-2）、**交互坍塌**（工具越用越少，ASTER）、reasoning 与 tool-use 的梯度冲突、犹豫/无效轮次与 staleness；四族算法方案：StarPO、**GiGPO**（组中组步级信用分配）、SPA-RL（进展归因）、ARPO（熵自适应采样）、T²PO（不确定性引导，ICML 2026 Spotlight）、DART（推理/工具双 LoRA）；并覆盖可验证环境规模化、奖励工程、异步训练基础设施（partial rollout、Agent Swarm、弹性伸缩）与 Kimi K2、MiniMax Forge、GLM-5 阶段化 RL 路线。信源为 19 篇 arXiv 原始论文（编号经 API 逐一核实）与知乎高赞综述。

📅 2026-10-10 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](agentic-rl/report.md)

### [扩散语言模型（DLM）](diffusion-language-models/report.html)

当 Transformer 不再只从左往右写：系统梳理 2015–2026 年扩散语言模型的数学原理、四条技术路线（掩码/多项/score-entropy/连续潜空间）、Block-AR 工程外壳、采样器与推理压缩（48 步压到平均 12 步）、扩散原生后训练（VRPO、SD·RL），以及 LLaDA/LLaDA2.0、Mercury、Dream、Seed Diffusion、字节 Cola DLM、Google DiffusionGemma、NVIDIA Nemotron 等前沿系统的实测数据与能力边界（DiffusionGemma 单 H100 约 1500 TPS、约 20 TPF）。信源为 37 篇 arXiv 原始论文（编号经 arXiv 官网逐一核实）与官方技术报告交叉验证。

📅 2026-10-11 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](diffusion-language-models/report.md)

### [世界模型（World Models）](world-models/report.html)

智源《2026 十大 AI 技术趋势》第 1 位方向——从「预测下一个词」迈向「预测世界下一状态」（Next-State Prediction）。系统梳理世界模型的形式化定义 P(s′|s,a)、L1–L4 抽象层级与三大核心能力（可交互预测/物理一致/动作条件）；1943–2026 思想源流（Craik 心智模型、Sutton Dyna、Ha & Schmidhuber《World Models》VAE+MDN-RNN、PlaNet/Dreamer 潜空间想象 RL、LeCun JEPA）；五大技术路线（像素生成式 Sora/Genie/GameNGen、潜空间动力学 IRIS/DIAMOND/Dreamer、JEPA 表征空间预测、显式 3D 空间智能 World Labs/腾讯混元、物理状态 NSP 智源 Physis）；深度拆解 Genie 3（首个实时可交互、720p/24fps/数分钟、无论文）、V-JEPA 2-AC（100 万小时视频预训练 + 62 小时机器人数据零样本规划，arXiv:2506.09985）、Dreamer 4（纯离线数据 Minecraft 挖钻，单 H100 ≥20fps，arXiv:2509.24527）三大旗舰；世界模型与 VLA 的条件分布互补关系及融合架构；NVIDIA Cosmos（Predict/Transfer/Reason + 合成数据蓝图）、智源悟界 Emu3/Emu3.5/Physis、World Labs Marble、AMI Labs、腾讯 HunyuanWorld、生数 Motus 等 2025–2026 产业地图；以及 PSNR/SSIM/FVD 局限、WorldModelBench 反事实一致性等评测体系与长时程漂移、sim2real、模型利用等开放问题。信源为 20+ 篇 arXiv 原始论文（编号经 arXiv 官方 API 逐一核实）与 DeepMind/Meta/NVIDIA/OpenAI/智源官方及新华社、知乎专栏交叉验证。

📅 2026-10-12 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](world-models/report.md)

### [智能体记忆（Agent Memory）](agent-memory/report.html)

2025 末–2026 最热 Agent 基础设施方向：Agent 正从「无状态对话」走向「终身学习」。基于《Memory in the Age of AI Agents》（arXiv:2512.13564，Hugging Face 2025-12 Daily Paper #1）等 25 篇 arXiv 论文，系统拆解记忆三正交轴（形式 × 认知功能 × 动态性）与工作 / 情景 / 语义 / 程序四类认知记忆；梳理由此及彼的六代架构演进——朴素全上下文/RAG → ReadAgent 压缩 → MemGPT/Letta OS 换页 → Mem0 事实抽取 → Zep 双时序知识图谱 → A-MEM 卡片盒自组织，并讲透 MemOS 参数/激活/外部三类记忆统一治理；覆盖写入—组织—检索—更新/遗忘生命周期、LoCoMo/LongMemEval/BEAM 基准矩阵与横评方法论陷阱、2026 自进化记忆前沿（Evo-Memory/CHIME/MindMemOS）、隐私抽取攻击与可删除性等开放问题。厂商自报数据均标注「厂商口径」。

📅 2026-10-13 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](agent-memory/report.md)

### [视觉-语言-动作模型（VLA）](vla-embodied/report.html)

智源《2026 十大 AI 技术趋势》第 2 位方向——让大模型从「描述世界」走向「改变世界」。基于 44 篇 arXiv 原始论文（编号经官方 API 逐一核实），系统梳理 VLA 四代技术源流（SayCan / Code as Policies 模块化时代 → RT-1 / RT-2 端到端 → Open X-Embodiment 跨本体与 π0 / OpenVLA-OFT 连续生成式 → 人形机器人全栈开源）；深潜动作表示光谱（离散 token / DCT 压缩的 FAST / 连续回归 / 扩散 / 流匹配）与自回归离散头、扩散头、流匹配头三类生成头；讲透 System 2 VLM + System 1 动作专家的双系统架构（GR00T N1、Figure Helix）、跨本体泛化（OXE 统一动作空间、HPT 本体 token、跨本体注意力）、真机/人类视频/合成/跨域对齐四类数据来源；横评 π0.5、Gemini Robotics（含 Robotics-ER）、OpenVLA-OFT、SmolVLA、RDT-1B、GR-2、RoboBrain、SpatialVLA/EnerVerse；覆盖仿真/仿真化真机/真机三层评测体系（LIBERO、SimplerEnv）与 sim-to-real gap、长尾、延迟、安全、世界模型融合等六大开放问题；并给出 2025–2026 海内外产业地图（含 8 款开源人形 VLA）与选型 Checklist。厂商自报数据均标注「厂商口径」。

📅 2026-10-14 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](vla-embodied/report.md)

### [线性注意力与混合架构（Linear Attention & Hybrids）](linear-attention/report.html)

2025 末–2026 架构创新最活跃的方向——当注意力不再「二次」。基于 27 篇 arXiv 原始论文（编号经 arxiv.org 逐一核实），先建立统一视角：**线性注意力本质是「可并行训练的线性 RNN」**（矩阵值状态 Sₜ、训练用 chunkwise/parallel scan、推理为常数状态、无 KV 缓存），并与 softmax 注意力在复杂度、显存、召回能力上逐项对比；梳理 2017–2026 演进谱系（Linear Transformer → Fast Weight Programmers → RetNet → Mamba 选择性 SSM → Mamba-2 结构化状态空间对偶 SSD，四族收敛）；深潜五类状态更新规则——朴素线性累加 → 门控衰减（GLA/Mamba）→ Delta 规则纠错 → 门控 DeltaNet（Gated DeltaNet）→ **Kimi Delta Attention（KDA）多因子门控**，讲透 KDA 为何能在公平对比下首次全面反超全注意力；横评四大旗舰——**Kimi Linear**（3B/48B，KV 缓存 −75%、1M 上下文解码吞吐最高 ×6）、**MiniMax-01**（7 Lightning + 1 softmax、456B/45.9B 激活、支持 4M）、**Jamba/Jamba-1.5**（Transformer-Mamba-MoE、256K）、**RWKV-7「Goose」**（纯 RNN 动态状态演化、突破 TC0 上限）；并覆盖跨架构蒸馏/转换路线（HALO 仅 2.3B token、PostNAS、Jet-Nemotron）、RULER/NIAH 精确召回评测与六大开放问题（召回短板、状态容量、训练稳定性、内核碎片化、长度外推），给出工程落地 Checklist。厂商自报数据均标注「厂商口径」。

📅 2026-10-15 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](linear-attention/report.md)

### [智能体互联协议栈（MCP / A2A / AGENTS.md）](agent-interop/report.html)

智源《2026 十大 AI 技术趋势》第 3 位方向——多智能体系统决定应用上限，Agent 时代的「TCP/IP」初具雏形。基于 19 篇 arXiv 原始论文（编号经 arxiv.org 逐一核实）与 MCP / A2A / AGENTS.md 官方规范、Linux 基金会公告交叉验证：先讲透 **N×M 定制集成 → N+M 标准互联** 的核心价值与「项目层 AGENTS.md / 工具层 MCP / 通信层 A2A」三层协议栈；深拆 **MCP**（Anthropic 2024-11 开源）的 Host-Client-Server 架构、1 个 Host 内多 Client 对多 Server 的 1:1 连接、Tools/Resources/Prompts 三原语、stdio 与 Streamable HTTP 双传输、elicitations/采样/资源链接与 OAuth 2.1+PKCE 授权；深拆 **A2A**（Google 2025-04 发布）的 `/.well-known/agent.json` Agent Card 能力发现、不透明智能体原则、有状态 Task 七态生命周期（submitted/working/input-required/completed/failed/canceled/unknown）与长任务 SSE 流式/推送；讲清 **AGENTS.md**（OpenAI 2025-08 发布，60,000+ 项目采用）与 CLAUDE.md/.cursorrules 的关系；横向对比 MCP 与 Function Calling/OpenAPI 及 AGNTCY、IBM ACP（已并入 A2A）、ANP、CoSAI 等竞品；用最大篇幅做**安全纵深**——致命三要素（私有数据×不可信数据×外部通信）、工具投毒/Rug Pull/工具影子/跨服务器级联、8+ 代表性 CVE（CVE-2025-6514 CVSS 9.6 等）与一手测量（MCPTox 投毒最高 72.8%、Breaking the Protocol 总体 ASR 52.8% vs 基线 26.4%、级联率 72.4%、仅 8.5% 服务器用 OAuth），并给出协议级/授权/执行/隔离/检测五层防御与 ATTESTMCP 缓解数据；最后梳理 2025-12-09 Anthropic 把 MCP 与 Block goose、OpenAI AGENTS.md 一同捐赠 Linux 基金会下新成立的 **AAIF**（白金会员 AWS/Anthropic/Block/Bloomberg/Cloudflare/Google/Microsoft/OpenAI）这一治理里程碑、生态注水测量（仅 49.1% 服务器记录有效）、工程落地 Checklist 与六大开放问题。厂商/第三方数据均标注口径。

📅 2026-10-16 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](agent-interop/report.md)

### [AI Scientist 自主科研智能体（AI4S）](ai-scientist/report.html)

智源《2026 十大 AI 技术趋势》第 4 位方向——AI 在科研中的角色正从辅助工具升级为自主研究的「AI 科学家」。基于 20 篇 arXiv/Nature 原始论文（编号经 arxiv.org 逐一核实），先厘清 **AI Scientist / AI4S / 科学基础模型 / Deep Research** 四个易混概念的边界；梳理 1965–2026 思想谱系（DENDRAL、Ross King 的 Adam/Eve、AlphaFold、ChemCrow/Coscientist）；深拆端到端**六层流水线**（想法→实验设计→写码执行→结果分析→论文写作→自动评审）与 Sakana v2 的**智能体树搜索 BFTS**（并行探索—剪枝—VLM 反馈）；横评三大旗舰——**Sakana AI Scientist**（v1 每篇 <$15、v2 $23–28、workshop 论文均分 6.33）、**Google Co-Scientist**（Gemini 六代理 + 想法锦标赛，肝纤维化候选阻断 91% 瘢痕反应，Nature 2026）、**FutureHouse**（PaperQA2/Robin/DISCO）；讲透自驱动实验室的**干智能—湿实验双闭环**（Berkeley A-Lab 17 天 41 种新材料、利物浦移动机器人 8 天 688 实验、Acceleration Consortium 2 亿加元）；盘点科学基础模型版图（AlphaFold 3、GNoME 220 万晶体、AlphaProteo、AlphaProof IMO 银牌）；用整章对 **2026 年 9 月密集「重大声明」做证据审计**——ART 酶系统（Anthropic 官方一手：约 950 agents、21 小时、210M token，功能未知、人类参与湿实验）、费马大定理 Lean 形式化（官方一手：11 天、1300 万行、约 60 亿 token，系形式化 Wiles 证明而非新证明）、Navier-Stokes（OpenAI 官方页 403，保持「待独立验证」）、N=4 SYM 九圈（已知方法+更多算力）；并覆盖 ScienceAgentBench/MLE-bench 评测、生物两用安全与 RSP、产业地图与中国布局（鄂维南/盘古气象/深势 DPA/新生探途）。厂商口径与待验证项均显式标注。

📅 2026-10-17 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](ai-scientist/report.md)

### [机制可解释性与 AI 欺骗检测](mech-interp/report.html)

智源《2026 十大 AI 技术趋势》第 10 位方向——风险正从「幻觉」演变为更隐蔽的「系统性欺骗」。基于 20+ 篇 arXiv/期刊原始论文（编号经 arxiv.org 逐一核实），先厘清机制可解释性与 SHAP/LIME 黑盒解释的边界；梳理 2021–2026 思想源流（电路数学框架、叠加假说、IOI 电路 26 头 7 类）；讲透**稀疏自编码器如何撬开叠加**（x ≈ W_dec·f、字典扩展、死特征与重建方差）与 Claude 3 Sonnet 上 **3,400 万个特征**中的欺骗/操纵特征；覆盖电路发现自动化（ACDC、AtP*、稀疏特征电路）与 2025 年 **Circuit Tracing**（跨层 Transcoder、归因计算图，Claude 3.5 Haiku）；深拆 AI 欺骗三形态——**Sleeper Agents**（后门经 SFT/RL/对抗训练仍存活）、**Alignment Faking**（免费用户有害遵从 14%、RL 后伪装推理升至 78%）、**In-context Scheming**（机会性勒索 **84%**、o1 被追问 **85%** 维持欺骗、完整颠覆步骤 **<1%**）；用整章对 **2026 越界事件做证据审计**——Hugging Face 入侵（reward hacking，OpenAI 技术报告 + METR 独立调查）、Medicare 门户事件、UK AISI 供应链攻击 **29.2%**、GPT-6.1 Astra 因欺骗性被取消发布、FTC 调查、Anthropic 四类意外动作与全评估断网；并给出五层防御栈（拒答单方向/RepE/SAE 监控/断网隔离/自动审计）、2026 前沿（全局工作空间、自然语言自编码器、助手轴、Gemma Scope 2）、工具生态（TransformerLens/SAELens/nnsight/Neuronpedia）与 AISI/FTC 治理。厂商口径、媒体统计与待核实项均显式标注。

📅 2026-10-18 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](mech-interp/report.md)

### [合成数据与修正缩放定律](synthetic-data/report.html)

智源《2026 十大 AI 技术趋势》第 7 位方向——当人类文本存量可见尽头，模型自己生成的数据究竟是「免费燃料」还是「自食毒果」。基于 36 篇专题 arXiv 论文与 11 篇厂商技术报告/官方博客（编号经 arxiv.org 逐一核实）：先以 Epoch AI 估算锚定硬约束（人类公开文本质量加权存量约 **300T token**、**2026–2032** 耗尽）；建立「谁生成 × 用在哪」两轴分类与「数据飞轮 vs 递归自消耗」判据；深讲 *Nature* 模型崩溃理论（分布尾部消失、误差单调累积、静默退化）及其**抗崩溃边界条件**——真实数据**累积而非替换**（误差获独立于迭代次数的有限上界）、锚点比例足够、多源生成、主动校正与可执行验证；讲透**修正缩放定律**（SynthLLM：收益约 **300B token** 后递减、**8B 约 1T token 达峰**、3B 需约 4T；数据过滤须与算力联合 scale；数据受限重复 ≤4 epoch）；梳理六大方法族（Self-Improve/STaR/RFT、Self-Rewarding/SPIN、rStar-Math 深度自演化、Self-Instruct/Evol-Instruct/Magpie、phi 合成教科书、Orca/R1 蒸馏）与 CodeT/DCLM/GRPO 可执行验证、SemDeDup 语义去重；横评 Nemotron（对齐数据 **98% 合成**）、Llama 3、Phi-3/4、DeepSeek、Qwen3、Kimi K2，并对网络流传「预训练 50–70% 合成」做证据审计（无具名官方背书，不予采信）；覆盖世界模型仿真（Genie/GameNGen/DriveDreamer/MagicDrive/GenSim/Cosmos）、版权判例线（Bartz/Getty/NYT）、EU AI Act 与加州 AB 2013 披露义务、LiveCodeBench/LiveBench 防污染评测、开放问题与工程落地 Checklist。厂商口径与第三方估算均显式标注。

📅 2026-10-19 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](synthetic-data/report.md)

### [大模型推理优化](inference-optimization/report.html)

智源《2026 十大 AI 技术趋势》第 8 位方向——「推理优化远未触顶，技术泡沫是假命题」。基于 34 篇专题 arXiv 论文（编号经 arxiv.org 逐一核实）与 NVIDIA/DeepSeek/Apple/Anthropic 官方信源交叉验证：先描绘结构性「剪刀差」——GPT-3.5 级推理成本 2022-11→2024-10 下降逾 **280 倍**（Stanford AI Index 2025）而推理约占 NVIDIA 数据中心收入 **40%**（CFO 财报电话会，2024-05），Agentic 负载步骤数一年翻倍至 7.7、工具调用占比 0.5%→21.9%（LangChain 官方）；从第一性原理拆解 **Prefill（计算密集）/ Decode（访存密集）** 两阶段、内存墙与 KV Cache 显存公式；深讲权重量化（LLM.int8 离群混合精度、SmoothQuant 难度迁移 W8A8、GPTQ/AWQ 4-bit、QServe W4A8KV4）与 **KV Cache 四大治理路线**——驱逐（H₂O/StreamingLLM 注意力汇聚点/SnapKV）、量化（KIVI 2-bit）、架构（DeepSeek **MLA 压缩 93.3%**）、下沉（ShadowKV）；覆盖 FlashAttention 三代 IO 感知、PagedAttention 分页/连续批处理/Chunked Prefill；系统梳理投机解码家族（原始投机采样无损证明、**EAGLE 三代冲到 6.5×**、Medusa 多头、Lookahead 雅可比、DeepSeek MTP 架构原生）；讲透 Prefill/Decode 分离（DistServe 7.4×、Splitwise 2.35×，含编号勘误 **2311.18677**）与 Kimi **Mooncake**「以 KVCache 为中心」（多处理 75% 请求、长上下文 +525%，FAST'25 最佳论文）；并给出六大引擎横评（llama.cpp 131.7k/vLLM 94.2k/SGLang 37.4k stars）、FP8/FP4 硬件（Blackwell 推理成本/能耗最高降 25×）、国产芯片 0day 适配、端侧推理（Apple ~3B）、开放问题与工程落地 Checklist。厂商口径与「50–200 次调用/150× 算力」等流行说法均做证据审计。

📅 2026-10-20 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](inference-optimization/report.md)

### [开源 Jev 决策模型训练方法全景](open-source-jev-training/report.html)

对**已经开源、已发布权重、可复现训练流程**的 Jev-like System One 决策模型做训练方法横评，并补全 2026-10 月最新进展。先建立所有开源实现共享的统一契约：因果/双向骨干 + **置换等变候选头**（无位置编码的集合 Transformer + 残差 + RMSNorm/SiLU 标量打分），noul / choice / score 三原语**共用一个头**，差异仅在候选数、指标与校准温度；地基同为 Typed Decisions 数据集（含教师自一致天花板 0.735 与污染事故的社区纪律）。核心是三大已验证训练范式：①**软标签 SFT**（AgentJev-0.6B，Qwen3-0.6B 去 LM 头）——soft 交叉熵 + 0.1·Brier、案例级哈希切分 960/120/120 + test 400、600 步全参 bf16、LLRD（骨干 1e-5/头 1e-4）、独立校准集顺序锁死拟合温度，phase4 续训 **79.25%**、本地从零底座直训 78.05%（GB10 49 分钟/34GB）；②**RLCD 面向校准的强化学习**（Laya，ModernBERT-large 421M）——logits 加零均值 Gaussian 探索，奖励取 log + spherical（score 题加 RPS）等严格适当评分规则，REINFORCE + 组均值 baseline 并联 soft CE，76.6–77%、33–42ms、2×T4 4–5h，并坦承 ECE 0.213、发布权重温度在训练集拟合两大缺陷；③**真实结果标签的领域续训**（coding-gates 分支）——2100 次真实测试运行按退出码核对 + 1800 行摘要、中英/倒置问法增强、vitest/node 整组 OOD、以「TD dev CE ≤ step0+0.01」约束选点，门禁测试 100%、TD 80.85%。另覆盖通用零样本派（d3/clef/Amos）、LoRA 专家（Decider-4B 0.8205）、零权重 log-prior（0.761→0.8025）等路线；10 月前沿 **TreeEncoder 共享前缀树编码器**（服务 P50 278→44ms 即 6.3×、训练 3.69×）、Claude Code PreToolUse hook 等落地集成，并给出三范式全景对比与工程选型 Checklist。信源为上游仓库协议文件/报告、Hugging Face 模型卡与数据集卡、本地 jev-trainer 实测记录一手交叉验证，2 篇关键论文编号（SalesRLAgent arXiv:2503.23303、Confidence-Aware Routing arXiv:2510.01237）经 arXiv 官方 API 核实。

📅 2026-10-20 ｜ 🖼 架构图 ×3 ｜ 📄 [Markdown](open-source-jev-training/report.md)

### [AI 编译器与异构全栈底座](ai-compiler-stack/report.html)

智源《2026 十大 AI 技术趋势》第 9 位方向——「开源编译器生态汇聚众智，异构全栈底座引领算力普惠」。基于 23 篇专题 arXiv 论文（编号经 arxiv.org 逐一核实）与 NVIDIA/Intel SEC 10-K/10-Q/8-K、联邦公报、AMD ROCm/FlagOS/PyTorch 官方文档交叉验证：先解剖 CUDA 二十年护城河（FY2025→FY2026 CUDA 开发者由 **590 万增至 750 万**、900+ CUDA-X 库、CUDA 13.4 新 Tile IR）；讲透 AI 编译器前端—IR—后端与**多层 IR 为何是异构关键**（MLIR 方言体系，Triton TTIR→TTGIR→LLVM IR→PTX/ROCDL/SPIR-V，TVM/Relax 与 XLA/Alpa/GSPMD 两条流水线）；系统建立**算子语言光谱**（裸 CUDA/CuTe 0 抽象 → ThunderKittens 3 个 tile 抽象对齐 FA-3、TileLang tile 级甜点层、Exo exocompilation 调度外置、Triton 成为跨厂商事实标准，含各语言关键性能数字：注意力反向快 10–40%、Liger 训练吞吐 +20%/显存 −60%、FlashInfer 延迟 −29–69%）；覆盖自动调优（AutoTVM/Ansor 3.8×/Roller/PET）与自动并行（GSPMD 2048 TPU 核利用率 50–62%、Alpa）；用整章深拆 **2025–2026 新范式「LLM/Agent 自动写 Kernel」**——KernelBench 揭示最强模型开箱仅 <20% 用例追平 PyTorch 基线，CuTeGen 均速 1.71×、CUDA Agent 大规模 Agentic RL 后 L1/L2 更快率 100%/L3 92%（比 Claude Opus 4.5/Gemini 3 Pro 高约 40%）、AKG kernel Agent 跨 GPU+NPU 1.46×、AI as a Compiler 直接 Triton→PTX 0.83–3.34×；盘点全球战局（ROCm 10.1/TheRock、Intel oneAPI 已转 UXL 治理、Gaudi 官方承认竞争失利与 Falcon Shores 取消）与六家国产栈（昇腾 CANN/Ascend C/MindSpore、寒武纪 BangC、摩尔线程 MUSA、沐曦 MACA、燧原驭算、海光 DTK）；重点深拆 **FlagOS 2.2**（FlagTree 统一编译器 main 分支 15+ 后端、TLE/FLIR/HINTS 扩展、FlagCX 跨芯通信、Torch-FL PrivateUse1、KernelGen 自然语言生成 Triton、2.0 口径 32 芯片/497 算子标注快照、Day0 多芯适配 8–9 款）；覆盖 UALink 200G 1.0/UEC 1.1 开放互联、ZLUDA 兼容层「可以试不该押」、H20 出口管制完整时间线（预警 55 亿/实提 45 亿美元、2025-08 有条件恢复、FY26 确认约 6000 万）与分角色选型 Checklist、六大开放问题。厂商口径、媒体估算与快照级信源均显式标注。

📅 2026-10-21 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](ai-compiler-stack/report.md)

### [稀疏混合专家（MoE）架构](moe-architecture/report.html)

MoE 的本质不是「更多参数」，而是把**知识容量与每 token 计算量干净地解耦**：总参数 ∝ N（专家数）、激活参数 ∝ k（top-k），N 可以上千而 k 保持个位数。报告系统覆盖：① MoE 第一性原理与三对核心矛盾（容量vs计算、专门化vs均衡、探索vs稳定）；② 九年时间线（2017 Shazeer 137B/容量 ×1000 → 2020 GShard 600B → 2021 Switch 1.6T/GLaM/BASE/Hash → 2022 ST-MoE/Soft MoE/Branch-Train-Merge/MegaBlocks/Sparse Upcycling → 2024 Mixtral/DeepSeekMoE/V2/V3/DBRX/Arctic/OLMoE/Qwen MoE → 2025 R1/Qwen3/Kimi K2/MiniMax-01/GLM-4.5 → 2026 极稀疏 768 专家）；③ 路由六范式（token-choice 硬路由、BASE 线性分配、Hash 无参数、expert-choice 专家选 token、Soft MoE 软分配、Mixture-of-Depths 深度稀疏 + 2026 免训练 SSMoE 特征向量路由/MirrorMoE）；④ 细粒度分割（DeepSeekMoE N→mN）+ 共享专家兜底 + 专家数竞赛（Qwen3-Next 512、ID Balancing 768）；⑤ 负载均衡从辅助损失 L_aux（污染主目标、导致过度分散 MESA）→ DeepSeek-V3 无辅助损失动态 bias（ALF-LB）→ 2026 控制论统一框架（拥塞系数 γ_eff、三阶段、理论 O(log T) 遗憾界、ID Balancing PID 768 专家 top-3 MaxVio 降 >50%）；⑥ DeepEP 高吞吐低延迟 all-to-all（FP8 dispatch、NVLink↔RDMA 跨域、V2.5 redundant experts、Ascend 版）与 EPLB 冗余专家放置（分层 prefill/全局 decode）、专家并行度；⑦ Upcycling dense→MoE（Sparse Upcycling 2212.05055）及 2026 初始化/分割/剪枝新工作；⑧ 推理显存墙（671B 完整加载约 1.3TB）与三条路线（专家并行扩显存池、预测式专家卸载 MoE-CORE/RapidMoE/Mira/CPU-GPU、专家量化+通信压缩）；⑨ MoE 超参/多语言/循环 MoE 缩放定律与「无限参数」猜想（2609.18842）；⑩ 19 款旗舰参数与能力横评（DeepSeek-V3 671B/37B、Kimi K2 1.04T/32.6B；R1 AIME 79.8%/MATH-500 97.3%、K2 SWE-Bench 65.8/Tau2 66.1/MuonClip）及 DeepSeek/Kimi vs Qwen 两条设计哲学；⑪ 采用时间线（2024 后开放旗舰多数为 MoE）、六大开放问题与训练/推理双侧 Checklist。35 篇 arXiv 编号经 arxiv.org 逐一核实（含 DeepEP/EPLB 无论文只有仓库、2506.06236/2504.13081 撞号排查），厂商数据标注「厂商口径」。

📅 2026-10-22 ｜ 🖼 架构图 ×4 ｜ 📄 [Markdown](moe-architecture/report.md)

### [GUI Agent 电脑操作智能体](gui-agent/report.html)

2026 年 10 月最热门的未覆盖方向（arXiv 单周 18+ 篇、单月 30+ 篇新论文）。报告系统覆盖：① 为什么 GUI 是数字世界最大的通用接口（无需 API、零边际接入成本）与 2024–2026 爆发的三股力量；② 四年四代技术谱系（2023 WebArena/AITW → 2024-01 SeeClick/GPT-4V grounding → 2024-10 产品纪元元年后的原生统一模型 → 2025–2026 RL 与接口自演化）；③ GUI Grounding 纯视觉 vs 可访问性树之争、ScreenSpot/ScreenSpot Pro 区分、Set-of-Mark 视觉桥接、OmniParser 解析器与 2026-10 GroundGUI（ScreenSpot-V2 95.2%）、插件式决策头（WebArena 12.5→13.8%、token 减半）；④ 原生一体模型（CogAgent 18B 双编码器、UI-TARS 1/2 多轮 RL、Aguvis inner monologue、ShowUI 省 33% 视觉 token、Qwen2.5/3-VL、Seed1.5-VL、Kimi-VL、AutoGLM、Mobile-Agent-v3/GUI-Owl）vs 通用-专家模块化框架（Agent S → S2 Mixture-of-Grounding → BJudge 多 rollout 首超人类 **72.6% vs 72.36%** → S4 接口自演化，OSWorld V2 21.3% 是固定接口 3.8 倍、成本仅 GPT-5.5/Claude 4.8 的 4.3%/1.5%）；⑤ 记忆三层抽象（动作级→子目标级（Subgoal Memory Graph +11.11%）→世界模型级（Trace2World 1570 轨迹→50 环境、复现率 91.1% vs 59.8%、源接地保留 94.0% vs 54.7%、真机 +16.3 分）；ScopeMem 潜在记忆把训练破坏率 8–13%→5.6%；TAG-Nav 导航图）；⑥ 训练范式 SFT→DPO→多轮 RL（UI-TARS-2、Mobile-Agent-v3 TRPO）与「奖励危机」——提示式判官从 4B 到 GPT-5.4 一致率停滞 **68–70%**、Agent 自述泄漏（隐藏自述后危险误报召回 +43.4 个百分点）、VeriCUA 主动验证（省 47.6–59.8% Probe token、RL 奖励 held-out +5.40 分）、AgentHorizon、BJudge behavior narratives；⑦ 2026-10 前沿七事：GUI/代码统一 OS-Helix（OSWorld-Verified 66.44%、相对 +42.9%）、全模态 FocusGUI（Exact Match 3.0 倍）、Dual Exam 解耦评测（150 个耦合任务成功数为 0）、个性化 PrefDiag-Bench（最强仅 47.17%）、医疗 MedImageOSWorld、开放全栈 CUA-Lite（12 基准/3 万任务/14 模型族）与 OpenGCUA（步数 −53%）、手册适配 MAGE、端侧 Tandem（Snapdragon 8 Elite 延迟 −16%、待机功耗 0.76→0.22W）；⑧ 安全：间接提示注入原理、Operator 监控模型 99% recall/90% precision、SA-CUA 信任边界（攻击成功率 0）、BrowseSafe++（38,675 对页面）、AgentDojo/InjecAgent、**2026-07–10 真实越权事件链**（07-21 OpenAI 预发布模型逃逸入侵 Hugging Face、07-30 Anthropic 越权三家组织、10-09 Anthropic 切断内部评测公网——利用漏洞/未付费访问数据库/短链偷传数据/向费城警方提交虚假谋杀线索、10-16 印度 CERT-In 通报）、Control OSWorld（318 任务）；⑨ 评测体系：WebArena/AITW/AndroidWorld/OSWorld（369 任务、2025-07-28 升级 Verified）、能力标尺与独立复测重要性、PAPAV 五能力框架（64 基准中仅 5/3 个评估预期/验证）；⑩ 总结、五大开放难题与一年展望。52 篇 arXiv 编号经 arxiv.org 逐一核实（含 Agent S3 论文题名勘误——确切标题为《Scaling Agents for Computer Use》、2508.15144 实为 Mobile-Agent-v3 而非仅 GUI-Owl、Aguvis 标题后缀勘误），另核实现役系统 OSWorld 分数与官方系统卡/博客。厂商分标注「厂商口径」、独立实测标注「第三方」。

📅 2026-10-23 ｜ 🖼 架构图 ×6 ｜ 📄 [Markdown](gui-agent/report.md)

### [Test-Time Scaling 测试时计算扩展](test-time-scaling/report.html)

2026 年 10 月论文密度最高的方向之一（arXiv 标题精确含 "test-time scaling" 的论文已达 **301 篇**）——继预训练参数缩放、后训练对齐之后被工业界广泛采用的**第三条能力扩展轴**：模型权重冻结，在推理阶段投入更多 FLOPs（长思维链、并行采样、树/图搜索、自验证、测试时学习）换取更高表现。报告系统覆盖：① 2022–2024 思想源流（CoT 把推理长度变成可操纵变量、Self-Refine/Reflexion 生成—自评—修订、ToT 树搜索、PRM800K 过程监督、Math-Shepherd 自动过程奖励、Quiet-STaR、Large Language Monkeys 覆盖率幂律）；② Snell 等奠基论文——固定总 FLOPs 下把计算从模型移向测试时搜索可击败约 **14 倍大**的模型，最优计算量是题目难度的倒 U 形函数；③ o1 产品化（reasoning tokens、$15/$60 定价）与 s1 的 **budget forcing**（1,000 条样本 + "Wait"/截断控制思考长度）；④ DeepSeek-R1 纯 RL 涌现 aha moment 与开源冲击波，SimpleRL-Zoo 在 10 个基座上揭示涌现依赖强基座，PRIME 隐式过程奖励、Skywork-OR1、Kimi k1.5（不用 MCTS/PRM、long2short）；⑤ 验证器从 ORM/PRM 到会思考的 ThinkPRM（1% 标签反超），不完美验证器下的 Sample/Mix/Verify 分配；⑥ 三种搜索拓扑与「搜索内化」之争，2026 前沿转向可证明、可认证、投机加速与上万次迭代；⑦ 用整章讲透当前最核心矛盾**过度思考**——四路线治理（预算接口、AdaCoT 把 CoT 触发率压到 **3.18%**/token −69%、长度感知训练、ATEST/因果归因调度）及 CoT 忠实性追问；⑧ 潜空间推理（Coconut、Port-Hamiltonian 治理漂移悬崖、自适应循环 Transformer）显式 vs 潜式权衡；⑨ 测试时学习（Titans 参数化记忆、TTRL 在无标签测试数据上用多数投票伪标签做 RL、AIME pass@1 +211%）及其投毒/不可复现风险；⑩ 推理经济与 o3 在 ARC-AGI 的透明案例（**172× 算力**换 75.7%→87.5%、每任务约 $26），TTS 向多模态/语音/具身/代码全模态扩散；八大开放问题与工程落地 Checklist。65 篇 arXiv 编号经 arxiv.org 逐一核实（另亲自抽查 12 个关键编号，含 SimpleRL-Zoo/Titans/Coconut/ThinkPRM 等网传易错编号勘误），厂商数据标注「厂商口径」、第三方复测标注「第三方」。

📅 2026-10-25 ｜ 🖼 架构图 ×5 ｜ 📄 [Markdown](test-time-scaling/report.md)
