# CPT 前沿技术报告：持续预训练（Continued Pre-Training）——让基础模型长出领域能力的第二生长曲线

> 报告日期：2026-11-12　·　主题消歧：本文的 CPT 指 **Continued / Continual Pre-Training（继续/持续预训练）**，即对一个已经完成大规模预训练的基座模型，使用新的无标注/弱标注语料继续做 **语言建模（LM）训练**。它不是医学里的 CPT，也不是电路设计里的 CPT。
>
> 证据口径：本文 arXiv 编号均通过 arXiv 官网/API 逐一核实；文中厂商或论文给出的性能数字均标注「论文/厂商口径」，未做独立复现。

## 一、摘要与 TL;DR

CPT（Continued Pre-Training）是 2024 年以后大模型从「一个通用模型打天下」走向「通用基座 + 领域专家 + 多语言/主权模型」的关键工序。它处在 **预训练（Pre-Training）与监督微调（SFT）之间**：不改变模型架构，用新语料继续做 next-token prediction，把领域知识、语言分布、长上下文能力、代码/数学/医学等专业分布「压」进已有权重。

核心判断：

1. **CPT 学的是「分布与知识」，SFT 学的是「格式与行为」。** 知识缺口用 CPT，行为/格式缺口用 SFT；顺序上推荐 CPT → SFT →（可选）RL。
2. CPT 最大的两个敌人是 **灾难性遗忘（Catastrophic Forgetting）** 与 **稳定性鸿沟（Stability Gap）**：模型在训练初期出现临时性能塌陷，以及领域注入后通用能力被冲掉。
3. 现代 CPT 的工程共识：**高质量领域语料 + 通用语料回放（replay）+ 重新衰减的学习率（LR restart/anneal）+ token 级数据选择**；学习率通常显著低于原始预训练峰值。
4. 2025–2026 年，「**Mid-training（中训）**」作为独立阶段被系统化：在预训练后、SFT 前集中注入推理、工具、数学、代码、多语言等能力，并出现专门 survey（arXiv:2510.06826）。
5. 对 JEV（评判模型）而言，若评判知识与领域语料缺失，CPT 比单纯堆 SFT 更根本；但 choice/nlu/score 的**行为格式**仍需 SFT/多任务头完成。

## 二、背景：为什么需要 CPT

### 2.1 预训练的假设与现实缺口

大规模预训练得到的基座模型在分布上近似「互联网均值」：通用、宽覆盖、但在任一垂直领域都不是最深。现实需求却高度倾斜：

- **领域纵深**：医疗、金融、法律、工业、代码、数学等，术语密度、推理范式、文本结构与通用网页差异巨大。
- **语言/主权**：日语、中文方言、小语种，通用语料中 token 占比低，模型「会但不精」。
- **时间迁移**：基座知识有截止日期，新事件、新 API、新规范需要持续注入。
- **能力扩展**：更长上下文（如 Code Llama 的 16k→100k）、新模态、新工具调用格式。

重新从头预训练成本极高（万亿 token、千卡月级），而 CPT 复用已有权重，只需 **10^9 ~ 10^12 量级 token** 即可获得显著领域增益，性价比成为它流行的根本原因。

### 2.2 术语消歧：CPT / DAPT / TAPT / Mid-training

这些概念在文献中高度重叠，必须厘清：

| 术语 | 全称 | 含义 | 出处 |
|---|---|---|---|
| **CPT** | Continued / Continual Pre-Training | 对已预训练模型继续做 LM 训练的统称 | 业界通用 |
| **DAPT** | Domain-Adaptive Pre-Training | 用领域语料做 CPT，目标是领域适配 | Gururangan et al., 2020（arXiv:2004.10964） |
| **TAPT** | Task-Adaptive Pre-Training | 用具体任务的无标注语料继续训练 | 同上 |
| **Mid-training** | Mid-Training | 预训练后、SFT 前的独立「能力注入」阶段（推理/工具/数学/代码/多语言） | 2025 后系统化，survey：arXiv:2510.06826 |
| **CFT** | Continued Fine-Tuning | 部分厂商对 CPT 的别名，仍以 LM loss 为主 | Together AI 等 |

> 注意拼写差异：**Continued Pre-Training（继续预训练，偏工程）** 与 **Continual Pre-Training（持续预训练，偏持续学习视角）** 中文都常译「持续预训练」，本文统一用 CPT。

### 2.3 CPT 与 SFT 的本质区别

这是最容易混淆、也最影响项目成败的一点：

| 维度 | CPT | SFT |
|---|---|---|
| 数据 | 无标注/弱标注自然语料（文档、代码、书籍） | 「输入→理想输出」成对样本 |
| 目标 | 继续最小化语言建模 loss（next token） | 在答案片段上最小化监督 loss |
| 学到什么 | 知识、术语、分布、风格、领域结构 | 指令遵循、输出格式、对话/工具行为 |
| 是否改变模型「知道什么」 | 是（注入/更新知识） | 有限（主要是激发与格式化） |
| 典型数据量 | 10^9–10^12 token | 10^4–10^7 条样本 |
| 位置 | 预训练之后、SFT 之前 | CPT 之后 |
| 风险 | 遗忘、通用能力退化、对齐损坏 | 过拟合、格式僵化、泛化下降 |

经验法则：

> **模型「不知道」→ 用 CPT；模型「知道但不按要求做/不说」→ 用 SFT。**
> 若基座在领域文本上读不懂术语、生成不出正确结构，先 CPT；若它能答但格式乱、不听话，SFT 即可。

## 三、CPT 在模型生命周期中的位置

现代开源模型的典型训练流水线如下（图 1 可视化）：

```text
[原始预训练]                 [CPT / Mid-training]           [SFT]              [RL / 对齐]
万亿级通用 token   →    领域/语言/长上下文/推理语料   →   指令-回答对    →   偏好优化/RLVR
通用基座 L_base          领域/中训模型 L_cpt             指令模型 L_sft       对齐模型 L_rl
                          · 注入知识与分布                · 格式化行为        · 风格/价值观
                          · 扩展上下文                   · 激发能力          · 可验证奖励
                          · 保持通用回放                  · 多任务头
```

关键边界：

- CPT **不改变架构**（embedding/层数/头结构不变），但常扩展 **RoPE 基频/上下文长度**。
- CPT 之后必须重新评估是否需要 **二次 SFT**：领域 CPT 可能让已对齐模型「回退」到基座分布（见 §9）。
- 在 JEV 场景，CPT 后通常再接 choice/nlu/score 的多任务 SFT，而不是指望 CPT 直接学会输出判别格式。

## 四、核心原理：CPT 为什么有效

### 4.1 分布迁移视角

把文本建模看作对 token 分布的估计。预训练得到分布 p_base(x)，而目标领域的真实分布是 p_dom(x)。CPT 用领域语料把模型从 p_base 向 p_dom 拉动：在参数空间中，CPT 等价于以 p_base 参数为初始化，对 p_dom 的负对数似然继续优化。因为 p_base 已经包含通用语言结构，这种优化比从随机初始化快得多——这是 CPT 高性价比的理论基础。

### 4.2 知识注入与「世界模型」更新

Transformer 的 FFN 被广泛认为以键值记忆形式存储事实。CPT 通过持续呈现领域共现与推理模式，更新这些记忆：不仅是术语，还包括「什么导致什么」「什么是正确步骤」等结构化关系。这也是 CPT 能提升领域**推理**而非仅风格的原因。

### 4.3 适应而非记住

Gururangan 等人的经典结论（arXiv:2004.10964）：领域自适应预训练（DAPT）和任务自适应预训练（TAPT）在多类 NLP 任务上持续带来增益，且 DAPT 与 TAPT 可叠加。该工作把「继续预训练」从经验技巧变成可复现的方法范式，是 CPT 的奠基性文献。

### 4.4 为什么不能只靠 prompt/RAG 解决一切

RAG 把知识放在检索端，不改权重，适合**事实查询型**任务；但：

- 检索无法替代对术语、文体、长程结构的**内化**；
- 高并发下检索延迟与召回不稳；
- 复杂领域推理要求模型本身「会说领域语言」。

CPT 与 RAG 是互补关系：**CPT 内化分布与核心知识，RAG 补充精确、时效、可溯源事实**。生产系统常二者并用。

## 五、CPT 技术方案设计（端到端 Pipeline）

完整工程流水线分为六段（图 2 可视化）：数据采集 → 清洗/去重/过滤 → 质量选择与配比 → 打包/分词 → 训练 → 评估。

### 5.1 数据采集

典型来源：

- 领域文档：论文、专利、病历、财报、法规、工单、内部 Wiki（注意权限与脱敏）；
- 代码：仓库、commit、issue、文档（Code Llama 路线）；
- 书籍、教材、课程、题库（推理/数学）；
- 目标语言网页与平行文本；
- 合成语料：用强模型改写、问答化、指令化（注意合成误差累积）。

### 5.2 清洗、去重与过滤

- **语言识别与格式还原**：PDF/HTML 解析、去模板、去导航噪声；
- **去重**：MinHash/SimHash 文档级 + 精确/近重复，防止记忆污染和评估泄漏；
- **过滤**：规则（长度、符号比、困惑度）+ 模型质量分类器；
- **去污染**：对评测集（MMLU、GSM8K、领域 benchmark）做 n-gram 匹配剔除；
- **PII 与机密脱敏**：姓名、证件、账号、内部域名/IP、密钥，企业场景强制环节。

数据管线可参考现代开放数据栈：**FineWeb（arXiv:2406.17557）** 的 CommonCrawl 精炼流程、**DCLM / DataComp-LM（arXiv:2406.11794）** 的数据配方检索框架。

### 5.3 质量选择：token 也分「值得学」与「不值得学」

RHO-1（arXiv:2404.07965）提出 **Selective Language Modeling（SLM）**：用参考模型（更强教师或同模型自身）评估每个文档的损失，只在「模型当前不会、但教师认为重要」的高价值 token 上加权训练。其直觉是：语料中大量 token 是模型已掌握或低价值的，均匀训练浪费预算。后续 ESLM（arXiv:2505.19893）进一步做风险厌恶式选择，降低被异常/噪声样本带偏的概率。

### 5.4 数据配比：领域注入与通用回放

这是 CPT 成败的核心旋钮。设一个训练 batch 中：

- 领域语料占比 α（注入新知识）
- 通用语料回放占比 1−α（保住旧能力）

实践中 α 常见在 **0.3–0.9**：

- α 过高 → 领域强但通用/对齐能力明显退化；
- α 过低 → 领域注入慢、token 预算浪费。

通用回放语料应覆盖原始预训练的主要面（网页、书籍、代码、对话），而不是单一来源。配比需用「领域指标 + 通用指标 + 对齐指标」三组评测联合标定，不能只看领域 loss。

### 5.5 Token 预算

经验上领域 CPT 的预算通常为：

- 轻量适配（语言风格/术语）：10^8–10^9 token；
- 深度领域专家：10^10–10^11 token；
- 大规模中训/能力扩展：10^11–10^12+ token。

预算应服务于「边际收益」：当领域评测曲线趋平、而通用指标开始下滑时，即应停止或降低 α。

### 5.6 训练方式

| 方式 | 说明 | 适用 |
|---|---|---|
| 全参数继续训练 | 全部权重更新，能力上限最高 | 数据充足、追求专家模型 |
| LoRA / QLoRA CPT | 低秩适配，成本低、易回滚 | 资源受限、多领域快速适配 |
| 适配器并行 | 多个领域适配器按需加载 | 多租户、领域切换 |
| 分层/冻结 | 冻结底层、只训顶层或部分层 | 保守适配、降低遗忘 |

注意：**LoRA 用于 CPT 时 rank 通常要比行为微调更大**（领域知识是高容量信息），小 rank 容易只学到风格、学不进知识。

### 5.7 学习率与调度

CPT 几乎不沿用原始预训练的峰值 LR，常见做法：

- **LR 重启 + 余弦衰减**：以远低于原始峰值的 LR（常见为峰值的 1/10–1/3）重新衰减；
- **WSD（Warmup-Stable-Decay）**：MiniCPM（arXiv:2404.06395）推广的调度，长平台期利于持续注入，末尾短衰减随时产出模型，天然适配「持续」场景；
- **Annealing（退火）**：在训练末期把 LR 压低并混入高价值/目标数据。OLMo 2（arXiv:2501.00656）与 SmolLM2（arXiv:2502.02737）都把退火作为关键设计：在退火阶段集中喂高质量、高相关数据，单位 token 收益最高。

图 4 对比三种调度。核心经验：**学习率过大是早期崩溃和遗忘的首要元凶；宁可小 LR 多训，不要大 LR 硬冲。**

### 5.8 长上下文扩展

CPT 常同时承担上下文扩展（Code Llama 从 Llama 2 的 4k 扩展到 16k/64k/100k）。两条腿：

- **位置编码调整**：RoPE 基频放大（PI/NTK 思路）、位置插值；
- **长序列语料训练**：让模型真正见过长依赖，仅改位置编码不够。

长上下文 CPT 需配合通用回放与长度渐进（curriculum），否则短文本能力与位置外推都会受损。

## 六、损失函数设计

### 6.1 标准语言建模损失

对序列 x = (x1,…,x_T)：

```text
L_LM = -(1/T) Σ_t log p_θ(x_t | x_<t)
```

CPT 主体仍是该交叉熵，与原始预训练同构，区别只在数据分布与 LR。

### 6.2 混合数据的加权损失

设 batch 来自领域 D 与通用 G：

```text
L_mix = α · L_D(θ) + (1 − α) · L_G(θ)
```

可进一步按数据源/质量分桶加权 w_i：

```text
L = Σ_i w_i · L_i，  Σ_i w_i = 1
```

### 6.3 Selective LM（token 级加权）

RHO-1 式：用教师/参考模型定义 token 重要度 s_t，仅对高价值 token 加权：

```text
L_SLM = -(1/|S|) Σ_{t ∈ S} s_t · log p_θ(x_t | x_<t)
```

其中 S 是被选中的高价值 token 集合。实现上是「采样 + token 权重」，而非改变网络结构。

### 6.4 防遗忘的 KL 锚定

为抑制输出分布漂移，可加基座模型的 KL 正则：

```text
L_KL = β · KL( p_θ(·|x) || p_base(·|x) )

L_total = L_mix + L_KL
```

- β 小：保通用但可能限制领域注入；
- β 大：稳但学不动。
KL 锚定对「已对齐模型 CPT」尤其有用（防止对齐被冲掉），代价是需要额外跑一次 base 前向。

### 6.5 多任务/多目标（面向 JEV）

若 CPT 阶段已带部分监督头：

```text
L = L_LM + λ_c · L_choice_CE + λ_n · L_nlu_BCE + λ_s · L_score_MSE
```

但更推荐把这些项放到 **CPT 之后的 SFT 阶段**，CPT 保持纯 LM 目标，避免目标互相污染、调参复杂。

## 七、伪代码

### 7.1 全参数 CPT（PyTorch 风格，伪代码）

```python
# 伪代码：领域持续预训练
from torch.optim import AdamW

base_model = load_pretrained("base-checkpoint")   # 已预训练基座
tokenizer  = load_tokenizer("base-checkpoint")

dom_corpus = load_domain_documents("domain/")     # 高质量领域语料
gen_corpus = load_general_replay("replay/")       # 通用回放语料
alpha      = 0.7                                   # 领域占比
lr_peak    = base_pretrain_peak_lr * 0.1           # 显著低于原始峰值

optim = AdamW(base_model.parameters(), lr=lr_peak, betas=(0.9, 0.95),
              weight_decay=0.1)
sched = cosine_restart_schedule(optim, warmup_ratio=0.03,
                                total_steps=len(dom_corpus) // batch_size)

for step, batch in enumerate(mixture_loader(dom_corpus, gen_corpus, alpha)):
    out = base_model(input_ids=batch.input_ids, labels=batch.input_ids)
    loss_dom, loss_gen = out.domain_loss, out.general_loss
    loss = alpha * loss_dom + (1 - alpha) * loss_gen
    loss.backward()
    clip_grad_norm_(base_model.parameters(), 1.0)
    optim.step(); sched.step(); optim.zero_grad()

    if step % eval_every == 0:
        metrics = evaluate(base_model, domain_bench + general_bench + alignment_bench)
        if metrics.general_regressed or metrics.domain_saturated:
            break
```

### 7.2 KL 锚定与 token 选择（要点伪代码）

```python
# KL 锚定
with torch.no_grad():
    p_base = base_model_logprobs(batch.input_ids)
p_theta = model_logprobs(batch.input_ids)
loss = lm_loss + beta * kl_divergence(p_theta, p_base)

# RHO-1 式 token 选择
with torch.no_grad():
    teacher_loss = teacher.per_token_nll(batch.input_ids)
    student_loss = base_model.per_token_nll(batch.input_ids)
score = teacher_loss - student_loss          # 教师会、学生不会 → 高价值
mask = topk_tokens(score, k=keep_ratio)
loss_slm = masked_lm_loss(batch.input_ids, token_mask=mask)
```

## 八、关键挑战一：灾难性遗忘与稳定性鸿沟

### 8.1 灾难性遗忘（Catastrophic Forgetting）

持续在新分布上训练，神经网络会快速覆盖旧知识，表现为：

- 通用 benchmark 下降；
- 多语言模型单语 CPT 后其他语言退化；
- 已对齐模型 CPT 后指令遵循、安全拒答能力「回退」。

专门研究：**Examining Forgetting in Continual Pre-training of Aligned LLMs（arXiv:2401.03129）** 系统考察了已对齐模型 CPT 时的遗忘现象，是该问题的重要锚点。

### 8.2 稳定性鸿沟（Stability Gap）

研究（arXiv:2406.14833，ACL 2025 长文）发现：CPT 开始后模型性能往往**先明显下降、再恢复并超越**，这段临时塌陷被称为 Stability Gap。它与学习率、数据顺序相关；通过缓解该鸿沟，CPT 可以更高效、更稳定。工程含义是：**CPT 初期的指标下滑不一定要立刻放弃，但必须监控其是否能恢复。**

### 8.3 缓解遗忘的方法谱系

| 方法 | 机制 | 代表工作 |
|---|---|---|
| 通用回放 Replay | 混训旧分布，保持覆盖面 | 业界共识；Reuse, Don't Retrain（arXiv:2407.07263） |
| 知识蒸馏 / KL 锚定 | 约束输出不偏离基座 | §6.4 |
| 间隔复习 Spaced Repetition | 按遗忘曲线安排回放节奏 | arXiv:2608.17530 |
| 可回收训练 Recyclable Tuning | 设计可复用、低破坏的更新方式 | arXiv:2305.08702 |
| 无缝打包 Seamless Data Packing | 优化数据打包，降低分布突变 | arXiv:2505.22018 |
| 小 LR + 退火 | 降低更新冲击 | OLMo 2 / SmolLM2 |
| 参数高效适配 | LoRA 等限制可改容量 | 多领域实践 |

「**Reuse, Don't Retrain**」（arXiv:2407.07263）给出一套以复用基座为核心、减少重复训练成本的 CPT 配方，是工程实践的高价值参考。

## 九、关键挑战二：评估、对齐与数据

### 9.1 三组评测缺一不可

- **领域指标**：领域 QA、术语、专业推理、业务指标；
- **通用指标**：MMLU/GPQA/通用 NLU/多语言，防止遗忘；
- **对齐/安全指标**：指令遵循、拒答、有用性、格式，防止对齐损坏。

只看训练 loss 或单一领域指标，是 CPT 最常见的翻车方式。

### 9.2 已对齐模型为何更脆弱

对齐（SFT/RLHF）是在基座之上较「薄」的一层行为塑造。用大 LR、纯领域语料继续 LM 训练，会把分布拉回基座/领域，等于部分「抹掉」对齐。对策：小 LR、通用+对话回放、KL 锚定，CPT 后再做一轮恢复性 SFT。

### 9.3 数据质量比 token 数量更重要

现代开放训练栈反复验证：去重、去污染、教育/高质量过滤、退火期集中高价值数据，可以用更少 token 得到更强模型（OLMo 2、SmolLM2 的数据中心训练思路）。低质网页硬堆，常常既不涨领域、又伤通用。

## 十、演进谱系

| 阶段 | 时间 | 代表 | 主要贡献 |
|---|---|---|---|
| 奠基 | 2020 | Don't Stop Pretraining（2004.10964） | 提出并验证 DAPT/TAPT |
| 探索 | 2023 | Recyclable Tuning（2305.08702）、Efficient CPT（2311.08545） | 把 CPT 用于 LLM，强调效率 |
| 规模化 | 2023–2024 | Code Llama（2308.12950）、AdaptLLM（2309.09530）、日/金融 CPT（2404.17790/2404.10555） | 代码/领域/长上下文大规模 CPT |
| 机理化 | 2024 | 遗忘研究（2401.03129）、Stability Gap（2406.14833）、RHO-1（2404.07965）、Reuse（2407.07263） | 搞清楚为何崩、怎么选 token |
| 数据中心/中训 | 2025 | OLMo 2（2501.00656）、SmolLM2（2502.02737）、ESLM（2505.19893）、Seamless Packing（2505.22018）、Mid-training survey（2510.06826） | 退火、token 选择、中训独立成阶段 |
| 前沿 | 2026 | Spaced Repetition（2608.17530）、DACP（2510.05858）、主权/受监管行业 CPT（NVIDIA Domyn 等） | 持续学习节奏化、产业化落地 |

工业界实践也已成熟：Databricks 的 CPT 数据集刻画、Meta 的 LLM 适配方法、AMD ROCm 多语言 CPT Playbook、NVIDIA DGX Cloud + Domyn 的主权 AI CPT、Together AI 的 continued fine-tuning，均给出从数据到评测的完整路径。

## 十一、CPT / SFT / Mid-training / RL 对比总表

| 维度 | Pre-Training | CPT / Mid-training | SFT | RL（RLHF/RLVR） |
|---|---|---|---|---|
| 数据 | 通用无标注 | 领域/能力无标注 | 指令-回答对 | 偏好/奖励信号 |
| 主损失 | LM CE | LM CE（可加权/KL/SLM） | 答案片段 CE | 策略梯度/偏好优化 |
| 目标 | 通用能力 | 知识/分布/能力扩展 | 行为格式化 | 价值观/风格/可验证能力 |
| 数据规模 | 10^12+ token | 10^9–10^12 token | 10^4–10^7 条 | 10^4–10^6 偏好 |
| 遗忘风险 | — | 高 | 中 | 中 |

## 十二、工程落地 Checklist

数据：

- [ ] 领域语料来源合法、权限清晰、PII/机密已脱敏；
- [ ] 完成解析、文档/近重复去重、规则+质量分类器过滤；
- [ ] 对评测集做去污染；
- [ ] 准备覆盖全面的通用回放语料；
- [ ] 用 RHO-1/ESLM 思路做 token/文档级质量选择。

训练：

- [ ] 明确 α（领域占比）并联合三组评测标定；
- [ ] LR 峰值取原始峰值的小比例，配 warmup 与衰减；
- [ ] 评估使用 cosine-restart / WSD / annealing 中哪种调度；
- [ ] 长上下文扩展同步调整 RoPE 并提供长序列语料；
- [ ] 必要时加 KL 锚定或 LoRA；
- [ ] 设置保存与回滚点（CPT 初期可能有 Stability Gap）。

评估与交付：

- [ ] 领域 + 通用 + 对齐三组评测全程跟踪；
- [ ] 关注初期临时塌陷能否恢复；
- [ ] CPT 后评估是否需要恢复性/领域 SFT；
- [ ] 记录数据配方、token 预算、超参，保证可复现；
- [ ] 与 RAG 的分工明确（内化 vs 溯源）。

### 12.1 一份可直接参考的起步配方

```text
阶段一：1–3 × 10^9 高质量领域 token，α=0.7，LR=峰值×0.1，cosine restart
阶段二：观察三组评测；通用退化则 α→0.5 并加大回放
阶段三：末期 10%–20% 步数退火，集中喂最高价值数据
阶段四：CPT 完成后做领域/多任务 SFT，再评估是否需要偏好优化
```

## 十三、CPT 与 JEV 的关系

结合评判模型（choice / nlu / score）：

1. 若基座缺乏评判所需的**领域知识与文本理解力**（读不懂对话、识别不出语义错误），先 CPT 内化；
2. CPT 不直接教会「输出 A/B、输出多维 nlu 标签、输出标量分数」——这些**行为格式**由 SFT + 多任务头完成；
3. CPT 阶段保持纯 LM 目标更干净，多任务判别损失放到 SFT；
4. CPT 后务必复测 choice/nlu/score 三头与对齐表现，必要时用恢复性 SFT 找回行为。

即：**CPT 决定 JEV「懂不懂」，SFT/分类头决定 JEV「怎么判、怎么说」。**

## 十四、2026 年前沿方向

1. **Mid-training 独立化与标准化**：推理、工具使用、数学、代码作为中训能力模块被系统组织（survey：2510.06826），IBM 等研究指出中训对后续推理能力至关重要。
2. **持续学习节奏化**：间隔复习（2608.17530）等把「何时回放」做成可优化对象，逼近终身学习。
3. **token 级预算优化**：SLM/ESLM 与风险厌恶选择，让每 token 训练 ROI 更高。
4. **主权 AI 与受监管行业 CPT**：在医疗、金融、政务等域，私有化 CPT + 强审计成为主流交付形态。
5. **CPT 与 RL/推理的耦合**：中训注入推理数据，再经 RLVR 放大，形成「中训→RL」新流水线。
6. **数据-模型联合检索**：沿 DCLM 思路自动搜索最优数据配方，CPT 从手工配比走向自动化。

## 十五、结论

CPT 是复用基座、低成本注入领域与能力的「第二生长曲线」，其有效性来自以已训练权重为初始化的分布迁移，其风险集中在遗忘与稳定性鸿沟。现代实践已收敛出清晰范式：

> **高质量领域语料 + 通用回放 + 小学习率与退火 + token 级选择 + 三组评测**，并且严格遵守 **CPT 学知识、SFT 学行为** 的分工。

对 JEV 与多数企业场景，建议先用中小规模 CPT 验证领域增益，再接多任务 SFT，并用 RAG 补足可溯源事实——而非一上来重训或仅靠 prompt 兜底。

## 参考资料

### 原始出处（奠基与机理）

1. Don't Stop Pretraining: Adapt Language Models to Domains and Tasks（2020）— https://arxiv.org/abs/2004.10964
2. Code Llama: Open Foundation Models for Code（2023）— https://arxiv.org/abs/2308.12950
3. Adapting Large Language Models to Domains via Reading Comprehension（AdaptLLM, 2023）— https://arxiv.org/abs/2309.09530
4. Examining Forgetting in Continual Pre-training of Aligned Large Language Models（2024）— https://arxiv.org/abs/2401.03129
5. Efficient Continual Pre-training by Mitigating the Stability Gap（2024/ACL 2025）— https://arxiv.org/abs/2406.14833
6. Rho-1: Not All Tokens Are What You Need（2024）— https://arxiv.org/abs/2404.07965
7. Reuse, Don't Retrain: A Recipe for Continued Pretraining of Language Models（2024）— https://arxiv.org/abs/2407.07263
8. Recyclable Tuning for Continual Pre-training（2023）— https://arxiv.org/abs/2305.08702
9. Efficient Continual Pre-training for Building Domain Specific LLMs（2023）— https://arxiv.org/abs/2311.08545

### 数据栈、调度与中训

10. 2 OLMo 2 Furious（2025）— https://arxiv.org/abs/2501.00656
11. SmolLM2: When Smol Goes Big（2025）— https://arxiv.org/abs/2502.02737
12. ESLM: Risk-Averse Selective Language Modeling（2025）— https://arxiv.org/abs/2505.19893
13. Improving Continual Pre-training Through Seamless Data Packing（2025）— https://arxiv.org/abs/2505.22018
14. MiniCPM: Scalable Training Strategies（WSD 调度，2024）— https://arxiv.org/abs/2404.06395
15. DataComp-LM / DCLM（2024）— https://arxiv.org/abs/2406.11794
16. The FineWeb Datasets（2024）— https://arxiv.org/abs/2406.17557
17. Mid-Training of Large Language Models: A Survey（2025）— https://arxiv.org/abs/2510.06826
18. When to Review: Spaced Repetition for Continual Pre-Training（2026）— https://arxiv.org/abs/2608.17530
19. DACP: Domain-Adaptive Continual Pre-Training（2025）— https://arxiv.org/abs/2510.05858

### 综述

20. Continual Learning for Large Language Models: A Survey（2024）— https://arxiv.org/abs/2402.01364
21. Continual Learning of Large Language Models: A Comprehensive Survey（CSUR 2025）— https://arxiv.org/abs/2404.16789
22. The Llama 3 Herd of Models（2024）— https://arxiv.org/abs/2407.21783

### 工业界实践

23. Databricks：Characterizing Datasets and Building Better Models with CPT — https://www.databricks.com/blog/characterizing-datasets-and-building-better-models-continued-pre-training
24. Meta：Methods for adapting large language models — https://ai.meta.com/blog/adapting-large-language-models-llms
25. AMD ROCm：A Practical Playbook for Language-Specific CPT — https://rocm.blogs.amd.com/artificial-intelligence/multilingual-continued-pretraining/README.html
26. NVIDIA：Continued Pretraining for Sovereign AI and Regulated Industries（Domyn）— https://developer.nvidia.com/blog/continued-pretraining-of-state-of-the-art-llms-for-sovereign-ai-and-regulated-industries-with-domyn-and-nvidia-dgx-cloud
27. Together AI：Continued Fine-tuning of LLMs — https://www.together.ai/blog/continued-fine-tuning
28. IBM Research：Mid-training for better AI reasoning — https://research.ibm.com/blog/mid-training-for-better-ai-reasoning
