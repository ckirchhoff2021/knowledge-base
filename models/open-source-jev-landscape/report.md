# 开源 Jev 决策模型全景报告：有哪些模型、各自怎么做（2026-10-26 更新版）

> **调研问题**：现有的开源 Jev 决策模型都有哪些？它们各自的底座、数据、损失、训练方式和适用场景是什么？

> **诚实性声明**：本报告全部关键事实来自**一手资料**——各模型 Hugging Face 模型卡（2026-10-26 逐一打开）、GitHub 仓库、本地 jev-trainer 代码与实测，以及 Typed Decisions 数据集卡。模型卡中自报的指标只代表其作者口径，引用时均标注；社区模型数量庞大（HF 上 "jev" 相关条目超过 50 个），本报告聚焦有完整训练说明、可复现或被榜单收录的 16 个（系列）代表，其余量化版本（GGUF/ONNX/CoreML/MLX）仅作生态观察。

📅 **2026-10-26** ｜ **时间锚点**：上游 `malevrigns/agent-jev` 396★（最近 push 2026-10-23）；Typed Decisions 数据集卡榜单截至本日版本 ｜ **分类**：models / agents

> 📌 与 2026-10-20 版《开源 Jev 模型训练方法全景报告》的关系：本文以**模型目录**为主线（有哪些、怎么做），新增 10-20 之后出现或被核实的 **OpenJev、Jev-Omni、AutoTrust-JEV-27B v1.1、Turion Jev 31B、Bonsai-JEV、Kev、Nimble、APUS Reflex、wolf-defender** 等条目；训练范式细节可与上一版互参。

---

## 目录

1. [背景：什么是 Jev，开源生态为何一月内爆发](#1-背景什么是-jev开源生态为何一月内爆发)
2. [统一契约：所有开源 Jev 模型共享什么](#2-统一契约所有开源-jev-模型共享什么)
3. [开源模型总览：七大家族、16 个代表](#3-开源模型总览七大家族16-个代表)
4. [家族一 · 软标签 SFT 派：AgentJev-0.6B](#4-家族一--软标签-sft-派agentjev-06b)
5. [家族二 · RLCD 校准强化学习派：Laya](#5-家族二--rlcd-校准强化学习派laya)
6. [家族三 · 教师分布蒸馏派：AutoTrust JEV](#6-家族三--教师分布蒸馏派autotrust-jev)
7. [家族四 · 多模态 Agent 决策派：OpenJev / Jev-Omni](#7-家族四--多模态-agent-决策派openjev--jev-omni)
8. [家族五 · 全参微调决策派：Turion / AutoTrust v1.1](#8-家族五--全参微调决策派turion--autotrust-v11)
9. [家族六 · LoRA 通用决策派：Kev / Nimble / APUS Reflex](#9-家族六--lora-通用决策派kev--nimble--apus-reflex)
10. [家族七 · 小型编码器与教学派：DeBERTa / Bekko / minimind-jev](#10-家族七--小型编码器与教学派deberta--bekko--minimind-jev)
11. [真实结果领域续训与安全专家：coding-gates / wolf-defender](#11-真实结果领域续训与安全专家coding-gates--wolf-defender)
12. [训练方法学归纳：五个共同设计决策](#12-训练方法学归纳五个共同设计决策)
13. [评测纪律：榜单、天花板与不可比性](#13-评测纪律榜单天花板与不可比性)
14. [选型建议与落地 Checklist](#14-选型建议与落地-checklist)
15. [风险、常见误解与注意事项](#15-风险常见误解与注意事项)
16. [参考资料](#16-参考资料)

---

## 1. 背景：什么是 Jev，开源生态为何一月内爆发

### 1.1 Jev：System One 决策模型

TypeSafe AI（创始人 Diogo Almeida，前 OpenAI 研究员）2026 年 9 月发布闭源模型 **Jev**，把这类模型命名为 **System One Model**——借《思考，快与慢》的 System 1：不写作、不聊天，只在一次前向中给出预设候选上的**校准概率分布**。三种原语（HF 数据集沿用的字段名）：

| 原语 | 问什么 | 输出 |
|---|---|---|
| `noul` | 命题是否成立（yes/no） | 两端概率、P(yes)、value |
| `choice` | 给定候选集合选哪个 | 每候选概率、value、top_probability、margin |
| `score` | 有序量表上的等级 | 每级概率、level（argmax）、score = Σ i·p_i（期望等级） |

### 1.2 开源为何一个月内爆发

1. **门槛低**：单镜决策就是"给定状态 + 候选，输出一组分数"，0.6B–4B 的小模型 + 单卡即可训练；
2. **有公开地基**：Typed Decisions 数据集（1200 训练/400 测试案例、教师软概率标签）2026-09 开源后，任何人都能在同一地基上开工；
3. **Agent 刚需**：浏览器/计算机 Agent 的"下一步点哪、是否完成"需要 20–200ms 级判断，大模型逐 token 生成答案再解析又慢又不稳；
4. **闭源教师可蒸馏**：Jev API 本身就是开放端点，社区用其输出分布构建蒸馏语料，复刻出 KL 差异极小的学生。

到 2026-10-26，HF 上带 "jev" 标签的模型条目超过 50 个，从 17M 到 27B、从纯文本到文/图/音/视频四模态、从云端到 8GB 显卡与手机端（GGUF/CoreML/ONNX/MLX）全覆盖。

---

## 2. 统一契约：所有开源 Jev 模型共享什么

### 2.1 输入输出

```
输入：
  state    任意非结构化状态（文本/diff/日志/DOM/截图）
  question 自然语言描述的判定问题 + 判定标准
  options  候选集合（noul 为隐含的 yes/no）
输出：
  每个选项一个概率（Σ p = 1）、argmax 选项、期望分（score）
```

关键特征：**标签随请求到达**（request-time labels）——换任务只需改 JSON，不需要重训模型、不需要固定标签集。

### 2.2 两种"读取（readout）"实现

所有开源模型都把决策读成"某些位置上的 logits 归一化"，但实现分两派：

| | Token-code 读取（LM 派，主流） | 置换等变候选头（集合编码器派） |
|---|---|---|
| 做法 | 每个选项绑定一个 token 码（A–Z，再两字母码），读提示结束后这些码的 log-prob，在选项集合内归一化 | 每个候选独立编码 → 无位置编码的集合 Transformer/池化 → 每候选标量分 → softmax |
| 代表 | JEV-27B、Kev、Nimble、Turion、APUS、OpenJev | AgentJev-0.6B、Laya（双向编码器） |
| 优点 | 复用 LM 头、零额外结构、选项可多达 255+ | 天然置换等变、候选数任意、不依赖 token 化技巧 |
| 代价 | 选项顺序会轻微影响答案（需 order-averaging 修补） | 需要额外头部结构与训练 |

### 2.3 置换等变性是硬指标

重排选项不应改变答案。Token-code 派靠工艺逼近：Turion、APUS 等对"正向 + 倒置"两次打分取平均（order averaging）；OpenJev 报告重排翻转率从调优前 **18.5% 降到 2.3%**；JEV-27B 为 7.4%（教师 7.0%）。

<figure>
<iframe src="diagrams/2-readout-architectures.html" height="840" loading="lazy"></iframe>
</figure>

> 🖼️ **配图 2（HTML 版查看）**：两种读取架构对比 —— `diagrams/2-readout-architectures.html`

### 2.4 共同地基数据集

- **LocalLLaMA/typed-decisions**：noul/choice/score，1200 train / 400 test（2000 decisions），教师软概率标签；数据集卡维护**零样本**与**train 上拟合**两张榜单（明确标注不可比）。
- 教师自一致天花板 **0.735**：新教师样本对其余样本构建的 gold 的一致率；分数远超 0.735 意味着在学教师的怪癖而非任务本身。

---

## 3. 开源模型总览：七大家族、16 个代表

| # | 模型（系列） | 团队/作者 | 底座 | 规模 | 训练方式 | License |
|---|---|---|---|---|---|---|
| 1 | **AgentJev-0.6B** | malevrigns | Qwen3-0.6B（去 LM 头） | ~0.6B | 软标签 SFT（soft CE + 0.1·Brier），全参 bf16 | Apache-2.0 |
| 2 | **Laya** / laya-typed-decisions | ConvAI Innovations | ModernBERT-large | 421M | RLCD：proper scoring rule 奖励的 RL，并联 soft CE | Apache-2.0 |
| 3 | **JEV-9B / JEV-27B / JEV-35B** | AutoTrust (autotrust) | Qwen3.5/3.8、MoE | 9–35B | BoE：冻结大模型 + 108.9M 可拆卸决策块，教师分布蒸馏 | Apache-2.0 |
| 4 | **OpenJev** | openjev | Qwen 系（VL） | 27B 级 | 文本 + DOM + 截图多模态决策微调，FP8 vLLM | CC BY-NC 4.0 |
| 5 | **Jev-Omni** | akhilaaa3 | Gemma 4 12B IT | 12B | 3 万题多模态（文/图/音/视频）SFT | Apache-2.0 |
| 6 | **AutoTrust-JEV-27B v1.1** | AutoTrust | 27B 决策模型 | 27B | Decision Index 37 基准全量评测定位的全参微调 | Apache-2.0 |
| 7 | **Turion Jev 31B** | Turion AI | Gemma 4 31B IT | 31B | 全参微调 + token-code 读取 + order averaging | Apache-2.0 |
| 8 | **Bonsai-JEV** | Dake Bakery | Ternary Bonsai 2（三值 27B） | 27B（3-bit） | 复用 JEV 配方的 LoRA + 24 槽决策头，8GB 可跑 | Apache-2.0 |
| 9 | **Kev（0.8B/4B/8B/9B）** | jaredpalmer | Qwen3.5（Base） | 0.8–9B | 12 个公开数据集上的 LoRA 决策训练 | Apache-2.0 |
| 10 | **Bespoke-Nimble-9B（v2）** | Bespoke Labs | Qwen3.5-9B | 9B | 上下文接地的 LoRA 结构化预测适配器 | Apache-2.0 |
| 11 | **APUS Reflex（4B/9B/35B-A3B）** | APUS AI Lab | Qwen3.5 | 4–35B | 可选计算深度 + 候选感知输出投影的联合后训练 | MIT（自有部分） |
| 12 | **open-jev-deberta-v3-large** | Koto Labs | DeBERTa-v3-large | ~440M | 公开语料自建训练，复现 Jev 之"形" | Apache-2.0 |
| 13 | **Bekko System One v0** | hotchpotch | Ettin reranker（ModernBERT 系） | 17M/68M/400M | 共享前缀编码器 + 任务头，小型化探索 | 开源 |
| 14 | **minimind-jev v0.1** | cnfeichen | bge-small-zh（冻结）+ 3 层 MLP | ~0.66M 头 | 中文 RAG 重排教学项目，3 分钟可训完 | MIT |
| 15 | **coding-gates 续训模型** | malevrigns | AgentJev-0.6B | 0.6B | 2100 次真实测试运行（退出码核对）领域续训 | Apache-2.0 |
| 16 | **wolf-defender-jev** | Patronus Studio | mmBERT-small | 小型 | 提示注入检测专用微调（15 语言） | MIT |

量化/转换生态（非独立模型）：`ggml-org/OpenJev-GGUF`、`ggml-org/Kev-4B-GGUF`、`mys/laya-GGUF`、`aac6fef/laya-coreml/mlx`、`ningsuhen/laya-onnx`、`onnx-community/...`、`openjev/OpenJev-Flash-9B-GGUF` 等——让上述模型可跑在 llama.cpp / Ollama / Core ML / ONNX / 端侧。

<figure>
<iframe src="diagrams/1-family-map.html" height="880" loading="lazy"></iframe>
</figure>

> 🖼️ **配图 1（HTML 版查看）**：开源 Jev 七大家族生态图 —— `diagrams/1-family-map.html`

另有 NeoHorse-Jev-4B（TokenRhythm）、Open-Jev-9B、Darwin-27B-ZTC（零训练成本零权重先验派）、Quyet-1.0-Large、pplx-decider、vllm-sr d3 系列、Cloudflare clef 系列、Liquid AI d1 系列等被 Typed Decisions 榜单收录的开放权重通用模型，第 13 章统一列出。

---

## 4. 家族一 · 软标签 SFT 派：AgentJev-0.6B

- **模型/仓库**：`malevrigns/agent-jev` 中的 AgentJev，HF 上亦有派生条目（如 IamBusy/OpenJev-0.6B）。
- **架构**：Qwen3-0.6B 去掉语言模型头，骨干输出 1024 维 → Linear 降到 256 → **2 层无位置编码的集合 Transformer**（候选之间只做集合内交互）→ 残差 + RMSNorm/SiLU → 每个候选一个标量分。noul/choice/score **共用同一个头，不按任务类型分头**。
- **标签**：Typed Decisions 的教师**软概率分布**（不是 hard label）。
- **损失**：

```
L = soft_CE(p, q_teacher) + 0.1 · Brier(p, q_teacher)
soft_CE = − Σ_i q_i · log p_i
Brier   = Σ_i (p_i − q_i)^2
```

- **训练配置**：案例级哈希切分 960/120/120 + 400 test（跨分割零重叠）；600 步全参 bf16（非 LoRA）；LLRD 差异化学习率（骨干 1e-5、头 1e-4）；顺序锁死：dev 选 checkpoint → 独立校准集拟合每原语温度 → 才碰 test。
- **结果**：phase4 续训 **79.25%**；本地从 Qwen3-0.6B 底座直训 **78.05%**（GB10 约 49 分钟、34GB 显存）。
- **适用**：想学习"判别式候选头"工程、追求最低复现成本的人；是后续 coding-gates、TreeEncoder 的底座。

---

## 5. 家族二 · RLCD 校准强化学习派：Laya

- **模型**：`convaiinnovations/laya-typed-decisions`（167 likes），GitHub NandhaKishorM/laya。
- **底座**：ModernBERT-large（**421M**，双向编码器），延迟 33–42ms。
- **动机**：argmax 对了但概率不可信——SFT 只学选项、不学置信度。
- **RLCD 算法**：

```
探索：  logits 加零均值 Gaussian 噪声 → 采样候选分布
奖励：  r = log-score + spherical-score（score 题再加 RPS）
         —— 均为严格适当评分规则（proper scoring rule），
            诚实报告概率是唯一最优策略
优化：  REINFORCE + 组均值 baseline，并联 soft CE 损失
教师/标签：Typed Decisions 教师分布
```

- **结果（作者口径）**：76.6–77% TD，2×T4 训练 4–5 小时（Kaggle 可复现）。
- **官方承认的缺陷**：ECE **0.213** 仍偏过度自信；发布权重的温度在**训练集**拟合（issue #186 温度泄漏）——这是新手上 RLCD 最容易踩的两个坑。
- **适用**：关注"概率校准"和小模型 RL 工艺；421M + CPU 可跑，是私有部署门槛最低的严肃实现之一。

---

## 6. 家族三 · 教师分布蒸馏派：AutoTrust JEV

AutoTrust（HF 组织 autotrust）是目前产品线最完整的开源家族。

### 6.1 Blocks of Experts（BoE）配方

```
不是微调一个单体模型，而是：
  System 2 块：冻结的强预训练模型（如 Qwen3.8-27B），逐字节不变
  System 1 块：108.9M 可训练参数（仅占骨干 0.4%）的可拆卸专家块
             —— 作为 lm_head 上的 LoRA / 决策头存在
训练：对 TypeSafe Jev 教师的输出分布做知识蒸馏
打包：一套权重、一个 vLLM 引擎，按请求路由 System 1/2
```

- **蒸馏语料**：Apache-2.0 的 `SargeDev/jev-distill-corpus-v3`——约 64 万行教师分布（29,955 行 test_set_30k 中 25,376 行是 Jev 1.13 分布）。
- **JEV-27B 关键指标（模型卡，对教师 test 集）**：

| 指标 | 值 |
|---|---|
| 对教师分布的 mean KL | **≈0.017**（约需 60 次决策才能收集 1 nat 的差异证据） |
| noul AUROC / Brier | 0.996 / 0.0013 |
| score 期望 MAE（0–5） | 0.098 |
| choice top-1 一致率 | 0.903（高置信目标行 0.958） |
| ECE（15 bins，温度后） | **0.0009** |
| System 2 HumanEval（适配器关） | 0.780 |

- **训练成本**：108.9M 参数，单卡 B200 约 **9.2 小时**、0.98 epoch。
- **六组公开基准（AutoTrust 自测，2026-09-27）**：JEV-27B 六组均值 84.07%，比其自测的 TypeSafe Jev 1.13（83.85%）高 0.22 点；开放权重可复现、最多 256 选项、256K 上下文（20/20 命中藏在 250K token 中的句子）。
- **家族成员**：JEV-9B（第一代）、JEV-27B（本文）、JEV-27B-VL（加视觉）、JEV-35B（MoE）、GEV-26B-Decide；另有演示 demo news.guru.so。
- **适用**：想要"一套权重同时要决策和生成"、追求与闭源 Jev KL 级一致的团队；注意模型卡指标是**对教师的拟合度**，不等于人类 gold 上的绝对能力（见第 13 章）。

---

## 7. 家族四 · 多模态 Agent 决策派：OpenJev / Jev-Omni

### 7.1 OpenJev（openjev/openjev，172 likes）

- **定位**：开放权重决策模型，能对**文本、网页、截图**做类型化决策；每问一次前向，**最多 52 个选项**。
- **性能（模型卡，10,000 个文本问题 / 34 个公开源，所有模型同题同序）**：

| 模型 | 准确率 |
|---|---|
| Jev（托管 API） | 85.4% |
| **OpenJev** | **84.0%** |
| 同底座未微调 | 80.4% |
| Nimble 9B（开放） | 75.7% |

- **Agent 决策（held-out 测试集，微调前后）**：截图选下一步 76.5%→**88.0%**；未见过网站上选下一步 68.5%→**87.4%**；整任务 MiniWoB 完成 39 个，与托管 Jev 持平。
- **稳定性**：选项重排翻转率 18.5%→**2.3%**。
- **速度（单 H100、FP8）**：短文本决策约 **80ms**；网页决策（约 1460 prompt token、23 选项）约 **210ms**；截图桌面决策 176ms；同页缓存再问 118–257ms；单卡并发 8 时每秒 7 个不同页面请求。
- **服务方式**：vLLM（FP8、prefix caching、max-model-len 16384、`--limit-mm-per-prompt image:1`）+ 自带 helper/shim 决策 API（固定 READOUT 温度 0.85、noul T=1.829074）。
- **License：CC BY-NC 4.0（非商用）**——商用集成需特别注意。
- **适用**：浏览器/计算机 Agent、需要"看懂 DOM + 截图"的场景；是非商用研究与自用 Agent 的最强开放权重选择之一。

### 7.2 Jev-Omni（akhilaaa3/Jev-Omni，434 likes）

- **底座**：Gemma 4 12B IT；**3 万题微调**。
- **模态**：文本 + 图像 + 音频（≤30 秒）+ 视频（16 帧）四模态统一决策分类器，无生成解释。
- **成绩（作者口径）**：DecisionBench Medium **87.57%**、JevBench **86.15%**、MMAU 63.10%、MVBench 53.10%。
- **成本叙事**：无输出 token，按 Gemma 3 12B 输入价（$0.05/M）计；FP32 约 50GB，BF16 autocast。
- **适用**：需要单一模型覆盖四类媒体的审核/路由/判断；对标 Inkling（975B/41B 激活）等巨型多模态系统。

---

## 8. 家族五 · 全参微调决策派：Turion / AutoTrust v1.1

### 8.1 Turion Jev 31B

- **底座/方式**：Gemma 4 31B IT，全参微调（Turion 训练平台）。
- **读取工艺**：每选项一个单 token 码（A–Z 再两字母码），最多 **255 选项一次读完**；按问题类型的温度（choice 1.18、yes/no 1.18、score 1.12）；**order averaging**（选项倒置再打一次、两次平均）；**read twice**（≤12,000 字符的输入在 prompt 里重复一遍）。
- **结果**：Decision Index 0.3 公共分 **66.16**；140,178 个可评分请求；单卡 RTX PRO 6000 Blackwell（FP8）中位延迟 **116ms**。
- **适用**：偏好"全参 + 成熟工程读取"而非 LoRA 的团队。

### 8.2 AutoTrust-JEV-27B v1.1

- 在 JEV-27B 基础上面向 **Decision Index 0.3** 做的迭代版：公共指数 **67.31**（37 个基准、140,178 请求，本地官方评测套件自测、待维护者验证）。
- 强项：工具选择（BFCL 96.25）、意图识别（CLINC150+OOS 97.06）、银行意图 93.63；Tools & Automation 区 85.19。
- **适用**：要在统一指数上与各类系统横向定位、且以工具/自动化决策为主的场景。

---

## 9. 家族六 · LoRA 通用决策派：Kev / Nimble / APUS Reflex

### 9.1 Kev 系列（jaredpalmer）

- **成员**：kev-0.8b、kev-4b（159 likes）、kev-8b、kev-9，全部为 Qwen3.5 上的 LoRA；ggml-org 提供官方 GGUF。
- **训练数据（12 个公开集）**：Banking77、BoolQ、AG News、MultiNLI、SST5、Yelp、TREC、DBPedia、Amazon Reviews、IMDB、CommitPackFT、Aegis 安全集、金融投诉等。
- **指标（kev-4b，模型卡）**：OOD 锁定测试 0.838 / Brier 0.224；14 个 held-out 公开集 breadth-v1 测试。
- **特点**：纯公开数据集训练，透明度高；定位是"校准的多选题 LoRA"。

### 9.2 Bespoke-Nimble-9B（bespokelabs，244 likes）

- Qwen3.5-9B 的 **LoRA 适配器（约 165MB）**，上下文接地的 choice/boolean/rubric score；直接对允许的答案 token 打分，不生成推理。
- 8192 上下文、最多 255 选项/字段，T=1.0；含精确 prompt builder 与推理代码；GitHub bespokelabsai/nimble。
- 在 OpenJev 万题同场测试中为 **75.7%**（四家同条件对比里的开放模型基线）。

### 9.3 APUS Reflex（apus-ailab）

- **最大特色：可选计算深度**。4B / 9B / 35B-A3B 三档，应用按 `effort=low/high` 选择：

```
low 路径： 早退出口 + 只对候选 label 算分的候选感知输出投影
high 路径：完整深度
联合后训练：两条路径同时学参考决策，
           完整路径给短路径提供分布级信号（避免早期出口未经训练）
```

- **结果（模型卡）**：冻结 80 题发布面板上 35B-A3B **88.75%**、9B 85.0%、4B 82.5%（Jev API 82.5%）；9B 优化服务中位 HTTP 延迟 **25.58ms**。
- **适用**：对延迟敏感、想在同一模型上按难度分配算力的 Agent 服务。

---

## 10. 家族七 · 小型编码器与教学派：DeBERTa / Bekko / minimind-jev
### 10.1 open-jev-deberta-v3-large（Koto Labs，74 likes）

- DeBERTa-v3-large 上独立复现 Jev 之"**形**"：choice（最多 255）/ score（2–10 级）/ noul，一次前向、结构化输出错误率为 0。
- **不用 TypeSafe 任何数据或代码**，公开 gold 标签自评；含语料构建器与消融（GitHub kotoba-lang/typed-decisions）；ONNX 社区版可高效部署。
- 意义：证明"双向大编码器 + 自造语料"也能独立长出该形态，不必依赖闭源蒸馏。

### 10.2 Bekko System One v0（hotchpotch，日本）

- **17M / 68M / 400M** 三档（17M 版 ONNX 仅约 29MB），基于 Ettin reranker，共享前缀编码（指令与状态只编码一次、跨候选复用）+ 任务头。
- **诚实的 v0 定位**：在 S1MB 泛化基准上远落后 Jev 1.13，且训练数据包含部分评测同族数据集，强分数不代表泛化——卡片明确标注。

### 10.3 minimind-jev v0.1（cnfeichen，2026-10-21）

- **教学项目**：冻结 bge-small-zh-v1.5 编码器 + 从零训练 3 层 MLP 头（512→256→256→1，约 **66 万参数**），做中文 RAG 重排的 Noul 型判断 P(段落回答问题)；约 **3 分钟**可训完。
- 意义：当前最小、最低成本理解 Jev 决策头机理的入口；指标按 jev-rag-zh-benchmark 协议（nDCG/MRR，bootstrap CI）。

---

## 11. 真实结果领域续训与安全专家：coding-gates / wolf-defender

### 11.1 coding-gates 续训模型（malevrigns，已并入 main）

- **动机**：教师标签是"读"出来的，不会看测试真实输出；代码门禁这种领域需要**真实结果标签**。
- **数据工程**：**2100 次真实测试运行**按退出码自动核对 + 约 1800 行运行摘要；中英/倒置问法增强；vitest/node 整组作为 OOD 隔离；第一次失败实验（只增强不控选点）如实记录。
- **受约束的 checkpoint 选择**：仅当 `TD dev CE ≤ step0 + 0.01` 才接受，防止"获得门禁能力、毁掉通用决策"。
- **结果**：门禁测试 **100%**、TD **80.85%**。
- **意义**：唯一不靠教师标签、靠环境真实执行结果训练的开源路线，与 RLVR（可执行验证奖励）思想同源。

### 11.2 wolf-defender-jev（Patronus Studio，2026-10-24，最新条目）

- 基于 **mmBERT-small** 微调的**提示注入 / 间接提示注入检测**专用决策模型，MIT、覆盖 **15 种语言**。
- 指标族：accuracy、ROC-AUC、balanced accuracy、recall/F1、误报率；定位为 Agent 工具安全与 DLP 护栏。
- 意义：Jev 形态开始向**垂直安全专家**细分——通用决策头之外出现按风险类别定制的小模型。

---

## 12. 训练方法学归纳：五个共同设计决策

| 决策 | 选项 | 开源社区的收敛点 |
|---|---|---|
| **标签从哪来** | 教师软概率 / 真实执行结果 / 公开数据集 hard label | 教师软概率最主流（JEV/OpenJev/AgentJev）；高风险或专业领域走真实结果（coding-gates） |
| **损失/奖励** | soft CE、Brier、KL、proper scoring rule、RPS | 分布匹配（KL/soft CE）打底；要校准就上 proper scoring rule（log/spherical/RPS）；Brier 作辅助项 |
| **骨干是否训练** | 全参 / LoRA / 冻结 | 大底座（9–31B）几乎全用 LoRA/小块（0.4% 参数）；≤4B 可全参 bf16 |
| **读取结构** | token-code LM logits / 置换等变集合头 | 复用 LM 头最省事（主流）；追求严格等变用集合 Transformer 头 |
| **校准** | 训后温度 / 训练中奖励塑造 / order averaging | 独立校准集 + 顺序锁死是底线；RL 可在训练中获得校准；选项倒置平均消除顺序偏差 |

共同的工程纪律：**先做切分再训练**（案例级哈希切分、ID+内容双重查重）、**选项集合内重归一化**、**order averaging 验证翻转率**、**选点与校准顺序锁死**、低精度（FP8/三值）降本 + prefix caching 提吞吐、关键失败只回退重做单环节。

<figure>
<iframe src="diagrams/3-training-lineage.html" height="860" loading="lazy"></iframe>
</figure>

> 🖼️ **配图 3（HTML 版查看）**：训练方法谱系（标签 × 损失 × 方式） —— `diagrams/3-training-lineage.html`

---

## 13. 评测纪律：榜单、天花板与不可比性

### 13.1 两套榜单不可混为一谈

Typed Decisions 数据集卡明确分两表：
- **零样本通用模型表**（从未见过该工作流）：截至 2026-10-26，开放权重/开放服务前列包括 Quyet-1.0-Large **0.791**、meraGPT Decider 1 与 Baseten Mammoth 1 0.768、Amos 1 medium **0.7645**、Amos large 0.761、pplx-decider 0.753、d3 0.749、d3-mini 0.743、clef 0.728、**openjev 0.728**、TypeSafe Jev 1.13 为 0.727 等。
- **train 上拟合/微调表**：Decider-4B **0.8205** 等——**与零样本表不可比**。

### 13.2 三条读榜铁律

1. **0.735 天花板**：教师自一致率；远超该值的跑分要怀疑"学到教师怪癖"。逐题天花板从 0.560（agent_trace/urgency）到 0.937 不等。
2. **对教师 KL ≠ 真实能力**：JEV-27B 对教师 KL≈0.017、ECE 0.0009 是"复刻度"；同一模型在零样本人类 gold 榜单上是另一套口径（其零样本表条目 0.495 即不同读取协议所得）。引用任何数字前先看清"对谁测、谁的协议"。
3. **新基准带来新坐标系**：Decision Index 0.3（37 基准、按机会水平调整的 skill 分，v1.1 为 67.31、Turion 66.16）、S1MB（专测跨任务泛化）、JevBench、DecisionBench、Six-group（六组等权）——各榜单任务集与评分不同，不能跨榜排名。

### 13.3 污染与纪律

数据集卡讨论区记录过污染事故（#25）与第三方跑分（#37、#38）；Bekko 等卡片主动声明"训练数据含评测同族集"。**切分纪律（案例级哈希、ID+内容双查、整组 OOD）是这个赛道的信任基础。**

---

## 14. 选型建议与落地 Checklist

### 14.1 按诉求直达

| 诉求 | 推荐 |
---|---
| 浏览器/计算机 Agent（看图/DOM） | **OpenJev**（自用/研究；注意 NC 许可）、APUS Reflex 9B（低延迟） |
| 要"决策 + 生成"一套权重 | **JEV-27B / v1.1**（BoE，适配器开关切 System 1/2） |
| 四模态（文/图/音/视频）统一审核 | **Jev-Omni** |
| 8GB 显卡/端侧私有化 | **Bonsai-JEV**（三值 27B）、Bekko 17M、minimind-jev |
| 纯公开数据、最透明可学 | **Kev-4B**、**open-jev-deberta-v3-large** |
| 最小成本理解机理/教学 | **minimind-jev**（3 分钟训完）、AgentJev-0.6B |
| 安全护栏/提示注入检测 | **wolf-defender-jev** |
| 按难度动态分配算力 | **APUS Reflex（effort low/high）** |

### 14.2 Checklist

```
□ 数据是否案例级切分、ID+内容双重查重？
□ 标签是否为软概率分布？hard label 时是否补 Brier/RPS？
□ 大底座是否只训 LoRA/小块（<1% 参数）？
□ 读取是否在选项集合内重归一化？
□ 是否做 order averaging / 验证重排翻转率（目标 <5%）？
□ 校准温度是否只在独立校准集拟合（严禁碰 train/test）？
□ 选 checkpoint 是否在 test 之前锁死？
□ License 是否允许商用（OpenJev 为 CC BY-NC）？
□ 是否用真实环境结果验证关键领域，而非只信教师？
```

---

## 15. 风险、常见误解与注意事项

- ❌ "Jev 就是个分类器，没什么技术"：它是 request-time 标签 + 校准概率 + 置换等变 + Agent 级延迟的**系统工程**；Karpathy 评价其处在大模型帕累托曲线上一个被长期低估的位置。
- ❌ 把"对教师 KL 小"当"真实可靠"：教师本身有自一致上限（0.735）和怪癖，蒸馏会忠实复制错误（JEV 卡自承"fidelity extends to the teacher's mistakes"）。
- ❌ 跨榜单比排名、把零样本和 train-fitted 分数直接比较。
- ❌ 忽视许可证：**OpenJev CC BY-NC 4.0 禁商用**；教学/二创模型另有条款。
- ⚠️ 同质化与刷分：调输出长度、措辞、温度都可能"刷"出虚高指标；垂直安全（wolf-defender）之外，误报率/漏报率需按业务单独评估。
- ⚠️ 生态噪声：HF 上大量条目是同名量化/小说玩梗项目（如 JevK5、Jeva-4B novel），选型以"有训练说明 + 榜单收录 + 可复现脚本"为准。
- 隐私与合规：不用真人/明星参考图；AI 生成内容按规标识；多语言安全模型需按地区复核。

---

## 16. 参考资料

### 16.1 数据集与上游代码（一手）

1. LocalLLaMA/typed-decisions（数据集卡含两套榜单、0.735 天花板）— https://huggingface.co/datasets/LocalLLaMA/typed-decisions
2. malevrigns/agent-jev（396★，Apache-2.0；AgentJev、coding-gates、TreeEncoder）— https://github.com/malevrigns/agent-jev
3. 本地 jev-trainer 训练/服务代码与 2026-10-20 实测记录。

### 16.2 开源模型卡（2026-10-26 逐一核实）

1. openjev/openjev — https://huggingface.co/openjev/openjev
2. akhilaaa3/Jev-Omni — https://huggingface.co/akhilaaa3/Jev-Omni
3. autotrust/JEV-27B — https://huggingface.co/autotrust/JEV-27B
4. autotrust/AutoTrust-JEV-27B-v1.1 — https://huggingface.co/autotrust/AutoTrust-JEV-27B-v1.1
5. turion-ai/turion-jev-31b — https://huggingface.co/turion-ai/turion-jev-31b
6. dakebakery/Bonsai-JEV — https://huggingface.co/dakebakery/Bonsai-JEV
7. jaredpalmer/kev-4b（及 0.8b/8b/9b）— https://huggingface.co/jaredpalmer/kev-4b
8. bespokelabs/Bespoke-Nimble-9B — https://huggingface.co/bespokelabs/Bespoke-Nimble-9B
9. apus-ailab/APUS-Reflex（4B/9B/35B-A3B）— https://huggingface.co/apus-ailab/APUS-Reflex
10. com-kotobalabs/open-jev-deberta-v3-large — https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large
11. hotchpotch/bekko-system-one-v0-400m（17M/68M/400m）— https://huggingface.co/hotchpotch/bekko-system-one-v0-400m
12. cnfeichen/minimind-jev-v0.1 — https://huggingface.co/cnfeichen/minimind-jev-v0.1
13. patronus-studio/wolf-defender-jev — https://huggingface.co/patronus-studio/wolf-defender-jev
14. convaiinnovations/laya-typed-decisions — https://huggingface.co/convaiinnovations/laya-typed-decisions

### 16.3 榜单收录的其他开放权重模型

chinhnc/Quyet-1.0-Large；mukaddamzaid/amos-1-medium/large；perplexity-ai/pplx-decider-v1.1-27b；vllm-sr d3/d3-mini/d3-turbo/d3-flash/d3-nano/d3-lite/d3-edge；Cloudflare clef/clef-flash/clef-omni；FINAL-Bench/Darwin-27B-ZTC；LiquidAI d1-3B；TokenRhythm/NeoHorse-Jev-4B；roudone/Decider-4B；manjunathshiva/opendecider-small；autotrust JEV-9B/27B-VL/35B/GEV-26B-Decide；e12ex2/jev-vision-27b；ggml-org 量化系列（链接与分数以数据集卡 2026-10-26 版为准）。

### 16.4 社区解读（知乎等，作者口径，引用前复核）

黄浴《开源版 Jev 登上 Hugging Face 热榜》《Jev 系列解读：Laya 结构与 RLCD》；Jev 技术拆解（含 Kev/Nimble/APUS 生态）；CSDN《Jev/Laya 深度拆解与本地部署》；以及数据集卡讨论区 #17/#25/#31/#35/#36/#37/#38。
