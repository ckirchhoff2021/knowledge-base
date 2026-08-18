# 大模型后训练技术调研报告：OPD 与 OPSD

> **调研主题**：On-Policy Distillation（OPD，在线策略蒸馏）与 On-Policy Self-Distillation（OPSD，在线策略自蒸馏）
> **调研来源**：知乎开放平台检索（20+ 篇深度文章）+ 原始论文/技术博客交叉验证
> **报告日期**：2026-08-18
> **配套材料**：架构图 3 张（`diagrams/` 目录，浏览器直接打开）

---

## 一、背景：后训练的三元困境

### 1.1 从"预训练 → SFT → RL"流水线说起

大模型能力来自三阶段叠加：**预训练**教语言与世界知识，**中期训练**注入领域知识，**后训练**激发目标行为（指令遵循、推理、对话）。后训练长期由两条路线主导：

| 路线 | 监督信号 | 采样来源 | 核心缺陷 |
|---|---|---|---|
| **SFT / 离线蒸馏** | 固定轨迹上的 hard label / 教师 logits（逐 token 稠密） | off-policy（外部数据） | **暴露偏差**：训练见教师前缀，推理见自己前缀，分布失配；上限被教师轨迹锁死 |
| **RLVR / GRPO / PPO** | 结果级标量奖励（对/错、单元测试） | on-policy（学生自己 rollout） | **信号稀疏**：每回合固定信息量，与 token 数无关；credit assignment 瓶颈——知道答错，不知道错在哪一步 |

两条路线在「**采样相关性**」和「**监督密度**」两个维度上恰好互补，又各有硬伤：

- **SFT 的暴露偏差（exposure bias）**：训练时第 t 步上下文是数据中的正确前缀，推理时却是模型自己生成的前缀。一旦某步生成训练集罕见的 token，后续所有条件分布都进入未覆盖状态，误差沿序列累积。
- **RL 的稀疏奖励**：RLVR 把整条答案的 advantage 均摊到每个 token，但一条推理轨迹里只有少数关键 token（数字、符号、逻辑转折词）真正决定对错，大量"载体 token"（标点、连词）不该收到梯度，均摊稀释了有效信号。

### 1.2 OPD 的定位

**OPD（On-Policy Distillation）= 学生自己采样（on-policy 相关性）+ 教师逐 token 分布监督（dense supervision）**，两头好处都占。用一个下棋类比（Thinking Machines Lab）：

| 训练方式 | 类比 | 反馈密度 | 贴合学生状态 |
|---|---|---|---|
| RL | 自己下棋没人教，赢/输一局才给一个反馈 | 稀疏（每局 O(1) bit） | ✅ |
| SFT / 离线蒸馏 | 看大师下棋，每步都精彩 | 密集 | ❌ 大师走的是新手遇不到的局面 |
| **OPD** | **自己下棋，引擎给每步打分（blunder/brilliant）** | **密集（每 token）** | ✅ |

一条 1000-token 轨迹在 RL 里只有 1 个序列奖励，在 OPD 里却能产生**数百到上千个教师 next-token 信号**。

### 1.3 关键出处

- **GKD**（Agarwal et al., 2024）与 **MiniLLM**（Gu et al., ICLR 2024）：OPD 的学术源头。GKD 把离线蒸馏问题重新表述为 interactive imitation learning（DAgger 思想），让学生访问自己的状态、教师在学生状态上标注；MiniLLM 强调 reverse-KL。
- **Thinking Machines Lab 博客《On-Policy Distillation》**（Kevin Lu et al., 2025-10-27，DOI:10.64434/tml.20251026）：OPD 被引用最广的"标准参考"，用 Tinker 复现 Qwen3 技术报告的 OPD recipe。**结论：用 RL 约 1/10 的算力，AIME'24 反超 6.8 个点**。博客明确受 Qwen 团队启发（全文提及 "Qwen" 38 次）。
- **工业标配**：Qwen3、GLM-5、MiMo-v2-flash、DeepSeek-V4 等技术报告均使用 OPD（或其多教师变体 MOPD）做后训练收尾。
- **OPSD 开山**：*Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models*（arXiv:2601.18734，UCLA/HKU/Meta，2026-01），首次系统化"特权信息自蒸馏"范式。

---

## 二、OPD 原理详解

### 2.1 形式化定义

语言模型学习条件分布 πθ(y|x) = ∏_t πθ(y_t | x, y_<t)。给定 prompt x：

1. **学生采样**：ŷ = (y₁,…,y_T) ~ πθ(·|x)（on-policy rollout）；
2. **教师打分**：教师 ν（冻结）在学生轨迹的每个前缀 y_<t 上输出 next-token 分布 q_t = ν(·|x, y_<t)；
3. **优化**：最小化学生轨迹上的分布距离。

**标准目标（reverse-KL）**：

```
L_OPD(θ) = E_{ŷ~πθ(·|x)} [ Σ_t  KL( πθ(·|x,y_<t) ‖ ν(·|x,y_<t) ) ]
```

reverse-KL 可以写成学生分布上的期望：

```
KL(πθ‖ν) = E_{y~πθ} [ log πθ(y) − log ν(y) ]
```

这正是 Monte Carlo 可估计的形式——学生采样出一个 token y_t，`log πθ(y_t) − log ν(y_t)` 就是对该位置 KL 的一次随机估计。这是 sampled-token OPD 在理论上成立的根基。

**为什么是 reverse-KL 而不是 forward-KL？**
- Forward KL（mode-covering）要求学生覆盖教师所有模式，等价于在教师分布上采样做期望——需要教师自己采样或拿到教师完整/top-k 分布才能构造无偏估计；sampled-token 场景下只有学生 token 对应的教师 log-prob，强行替代是有偏的。所以 **forward-KL 很难做出合理的 sampled-token 版本**，工业上只能 full-vocab 或 top-k。
- Reverse KL（mode-seeking）天然在学生分布上求期望，与 on-policy 采样自洽；且学生不会去拟合教师那些自己根本到不了的分布区域，避免被拉向遥远外部分布。

### 2.2 两种优化协议：GKD 路线 vs PG-OPD 路线

知乎文章（模力AILab、jamin）的关键澄清：**OPD 是一种采样与监督协议，KL、直接反传、policy gradient 才是具体的优化协议**。把教师意见变成训练信号有两条主路线：

**路线 A：GKD（广义知识蒸馏）式——分布对齐**

教师告诉学生"下一个 token 的完整分布应该是什么样"，学生对齐教师返回的候选分布：

```
L_GKD = E_{y_<t ~ 混合轨迹} [ D( πθ(·|y_<t) ‖ ν(·|y_<t) ) ]
D ∈ { forward-KL, reverse-KL, JSD }
轨迹来源：教师固定轨迹与学生 rollout 按 λ 混合（λ=1 为严格 on-policy）
```

**路线 B：PG-OPD（策略梯度式）——把师生分歧当 advantage**

教师只评价学生已采样出的 token，构造 token 级信号：

```
A_t = log ν(y_t | x, y_<t) − log πθ(y_t | x, y_<t)
```

- A_t > 0：教师比学生更认可该 token → 提高其概率；
- A_t < 0：学生自信过头而教师不认可 → 抑制。

然后走 PPO/GRPO 式的 importance-sampling 策略梯度。例：学生对"检查"给 0.2、教师给 0.6 → 正信号；学生给 0.5、教师只给 0.1 → 负信号。**工程上，这相当于在带 KL 正则的 RL 脚本上改一行——把正则项的 reference model 换成 teacher。**

### 2.3 监督粒度的三个变体

| 变体 | 计算范围 | 成本 | 特点 |
|---|---|---|---|
| **sampled-token** | 只看学生采样到的那 1 个 token | 最省 | 业界最常用；LLM 分布尖锐，高概率 token 被反复采到，长期信息量接近 top-k |
| **top-k** | 每位置对 top-16 token 子集算 KL | 中 | Rethinking OPD（清华 THUNLP）主实验用；注意未归一化会引入梯度偏差（见 §4.2） |
| **full-vocabulary** | 每位置整词表算 KL | 最贵 O(B·T·V) | 监督最完整；OPSD 论文实验显示优于 sampled-token（见 §3.4） |

### 2.4 为什么 OPD 能"学到真东西"：机制解释

综合多篇知乎解读（欠阿贝尔两块钱、jamin、秋水黑刀）：

1. **纠正发生在学生的错误上**。传统蒸馏在教师轨迹上训练，学生的错误不一定是教师的错误，监督覆盖不到学生真实会走错的地方。OPD 让教师针对学生自己的路径给指导——在错误前缀上，教师可能提高"检查展开""这里应为 2x−6"等**修正方向**的概率。学生学到的不只是复述标准答案，还包括"按自己的方式走到这里以后，怎样回到合理方向"。
2. **KL 匹配 ≠ 单序列复刻**。教师 logits 分布里包含风格、不确定性、多种后续可能、推理结构的信息。匹配分布是在**重塑整个采样行为**，而非复刻一条序列。
3. **on-policy 约束保护学生**。最优策略集合存在无数可行分布，RL/OPD 都只在当前采样邻域内寻优，自动选择与原始模型 KL 距离最小的解。所以即使教师有退化/偏科，学生也不会完全复刻教师的缺陷分布——**教师告诉学生往哪走，但学生自己的 on-policy 数据决定了从哪里出发、在哪里接受纠正**。
4. **RL 是在策略空间做搜索，OPD 是学习的捷径**（Thinking Machines 博客）：RL 不靠海量梯度更新，而是在已有权重上靠采样打磨策略；一旦好策略被找到，蒸馏就是低成本复制。博客还展示 OPD 用于**持续学习**——用"更早的自己"当教师，恢复被后续微调破坏的行为，且遗忘远小于 SFT。

---

## 三、OPSD 原理详解

### 3.1 核心思想：信息不对称的自蒸馏

OPSD（arXiv:2601.18734）把 OPD 的外部教师替换为**"持有特权信息（Privileged Information, PI）的自己"**，在单一模型内部构建师生角色：

- **教师策略** πθ(·| x, **PI**, y_<t)：同一组权重，上下文里额外注入参考答案（ground truth）或验证过的推理轨迹 → "开卷的自己"；
- **学生策略** πθ(·| x, y_<t)：同权重、不看答案，仅凭题目作答 → "闭卷的自己"；
- 训练时学生 on-policy 采样轨迹，教师在学生前缀上给分布，最小化两者距离。**教师分支 stop-gradient，且教师固定为初始策略**（隐式正则化，防止过度偏离）。

一句话：**用"带答案的自己"训练"不带答案的自己"**。推理时学生条件与训练时完全一致，特权信息不泄漏到部署。

### 3.2 与 OPD 的关系及消歧

- OPSD 本质是 OPD 框架（Agarwal et al. 2024）的一个实例化：**teacher 不是独立大模型，而是同 θ + PI**。
- **同名消歧**（重要）：π-Distill 论文中也有叫 "OPSD" 的模块，但其 PI 来自 frontier action trajectory（强模型的动作轨迹），而 Self-Distilled Reasoner 的 PI 来自 ground truth answer。两者目标相同（reverse-KL、on-policy、教师 stop-gradient），PI 来源不同，引用时需注意。
- 相比外部教师 OPD，OPSD 的优势：① 无需更大的教师模型，省在线推理成本；② 免除"师生必须共享词表才能对齐 logits"的约束（自己和自己天然同词表）；③ 师生同源只差一份答案，监督信号"难度恰好"。

### 3.3 目标函数

```
L_OPSD(θ) = E_{ŷ~πθ(·|x)} [ Σ_t  D( πθ(·|x, y_<t)  ‖  sg[ πθ(·|x, PI, y_<t) ] ) ]

D ∈ { forward-KL, reverse-KL, JSD }，sg = stop-gradient
```

### 3.4 关键实验结论（论文，Qwen3 系，OpenThoughts 数学子集训练，AIME24/25 + HMMT25 均分）

| 模型 | Base | SFT | GRPO | **OPSD** |
|---|---|---|---|---|
| Qwen3-8B | 61.8 | 59.8 | 64.0 | **64.8** |
| Qwen3-4B | 61.2 | 58.6 | 62.7 | **63.6** |
| Qwen3-1.7B | 37.1 | 35.8 | 37.7 | **43.4** |

要点：

- **三个尺度均优于 SFT，匹配或超过 GRPO**；1.7B 小模型提升最猛（+6.3）。
- **SFT 反而退化**：OpenThoughts 参考解风格简洁，直接模仿会缩短推理长度、损害泛化；OPSD 不模仿那条轨迹，只把它转成 dense supervision。
- **full-vocab > sampled-token**（Qwen3-4B, 2048 长度）：AIME25/HMMT25 分别 84.1/60.0 vs 82.1/57.3——教师全分布包含对"其他可能下一步"的偏好。
- **token 效率约为 GRPO 的 8–12 倍**；实现简单，无需奖励模型/价值函数。
- **局限**：① 模型容量是门槛（论文结论：过小的模型"理性化"能力不足，自蒸馏收益受限）；② 未显式利用答案正确性验证信号；③ 问题超出模型理解阈值时教师失效；④ full-vocab 峰值显存更高。
- 生成长度消融：4096 不稳定优于 1024——早期 token 更像"推理分支点"，前缀变长后教师的边际学习价值下降。

---

## 四、Loss 函数全景

### 4.1 基础散度家族

设学生分布 p = πθ(·|y_<t)，教师分布 q = ν(·|y_<t)，词表 V：

```
Forward-KL:   KL(q‖p) = Σ_v q(v) log [ q(v)/p(v) ]     # mode-covering，需教师分布采样
Reverse-KL:   KL(p‖q) = Σ_v p(v) log [ p(v)/q(v) ]     # mode-seeking，学生分布上可 MC 估计
JSD:          ½KL(p‖m) + ½KL(q‖m),  m = (p+q)/2        # 对称、有界，GKD 推荐默认
```

**GKD（Agarwal et al.）**：D 可在 FKL/RKL/JSD 中切换，轨迹按 λ 混合。**MiniLLM**：序列级 reverse-KL，用策略梯度 + teacher-mixed sampling 优化。**DistiLLM**：skew-KL / skew-RKL + 学生输出 replay。

### 4.2 TML / PG-OPD 的 token 级 surrogate

把师生 log-prob 差当 dense advantage，走 importance-sampling 策略梯度：

```
A_t = log ν(y_t|y_<t) − log πθ_old(y_t|y_<t)          # token 级 advantage
ρ_t = πθ(y_t|y_<t) / πθ_old(y_t|y_<t)                  # importance ratio

L = − E [ Σ_t min( ρ_t·A_t, clip(ρ_t, 1−ε, 1+ε)·A_t ) ]
```

（Rethinking OPD / REOPOLD 两篇反复引用的 "Lu et al., 2025" 即此 recipe；等价于 KL 正则 RL 中把 reference 换成 teacher。）

### 4.3 已知病理与对应的 loss 修复（重要）

来自《The Many Faces of On-Policy Distillation》（arXiv:2605.11182）、《Demystifying On-Policy Distillation》（arXiv:2607.13399，港中文+腾讯 AI Lab）、《Rethinking OPD》（清华 THUNLP，arXiv:2604.13016）：

| 病理 | 机理 | 修复 |
|---|---|---|
| **学生前缀扭曲教师分布** | 学生早期写歪，教师在歪前缀上被迫给"Wait/But"式纠偏 token，与正常推理语义冲突，学生学到支离破碎的思维 | SFT 冷启动：先在教师轨迹上短期 SFT，拉近初始分布再上 OPD |
| **Top-K reverse-KL 梯度偏差** | 截断未重归一化 → 隐藏偏差项 → 梯度不稳、模型崩溃（重复/乱码） | stop-gradient 或重归一化变体 |
| **OPSD 特权信息归纳困境** | 实例特定 PI（某题的具体解法）测试时不可得，强行聚合学出"平庸共识" | 教师适配（选分布与学生接近的教师）；RLVR 适配教师 |
| **更强教师反而更差** | OPD 是"探索催化剂"不是"能力灌注器"：加速学生翻出本就够得着的路径，但不抬高能力上限；师生差距过大时逐 token 打分起负作用 | 硬裁剪（hard clip）+ 对数压缩（log-compression）两个零开销信号调控（该文方案：4B 教师反超 30B 教师的 SOTA） |
| **改长度刷分后门** | 目标函数结构性漏洞：调整输出长度即可刷分 | 同上调控手段 |
| **OPSD 风格偏移（RLCSD, 智源+清华）** | "特权诱导的风格偏移"：拿到答案后模型倾向更短更直接的表达，信号被风格 token 主导而非关键推理 token | 对比正确提示与错误提示下的师生分布差异，抵消"加提示本身"引起的风格偏移 |
| **OPSD 死记硬背（Purified OPSD, arXiv:2607.02234, 浙大+通义）** | 长 CoT 模型上：批改信号被参考答案在方向和幅度上双重主导——Qwen3-8B 反思词全面崩塌（wait 27K→10K），R1-Distill-7B 单一 "Wait" 畸形爆炸（34K→83K） | 用"只看答案不看题"的对照教师分离死记信号，PMI 提取可迁移残差作为蒸馏目标 |
| **轨迹绑死（Rubric-SD）** | 监督绑定单条参考路径：学生稍偏离参考轨迹就被鼓励全局重写，推理链反复重算、又长又冗余 | 用 rubric（"好答案应满足的条件"）替代单条参考解作为教师条件 |
| **输出空间信号饱和（OPRD, arXiv:2606.06021）** | 所有 OPD 变体都在 LM head 之后算损失，教师中间层信息浪费；训练后期梯度信噪比崩塌，准确率平台期 | 把蒸馏搬到隐藏状态空间：同 rollout 上对中间层做 MSE 对齐（训练快 1.44×，显存省 54%） |

### 4.4 RL 与 OPD 的数学统一

多篇知乎综述（jamin、Leezhou）指出：G-OPD/ExOPD 证明 **OPD 与 KL 正则化 RL 可严格等价**——teacher log-ratio `log ν(y_t) − log πref(y_t)` 可视为隐式奖励，用 λ 调节其权重，λ>1 即做 reward extrapolation。这解释了"OPD 是蒸馏还是 RL"之争：**形式上是知识蒸馏（教师分布为目标），运行上是带 dense token 级奖励的 on-policy RL**。

---

## 五、伪代码

### 5.1 OPD（sampled-token，策略梯度形式）

```python
def opd_train(student, teacher, dataset, optimizer, steps):
    """
    On-Policy Distillation (Thinking Machines recipe, sampled-token PG 形式)
    student: 可训练学生模型 πθ
    teacher: 冻结教师模型 ν（白盒，需返回 token log-prob）
    """
    teacher.eval()
    for step in range(steps):
        # ---- 阶段 1: rollout（学生自己采样，on-policy）----
        batch = []
        for x in dataset.sample_prompts(batch_size=B):
            y = student.generate(x, sampling=True, max_len=T)   # ŷ ~ πθ(·|x)
            batch.append((x, y))

        # ---- 阶段 2: 教师逐 token 打分（在学生轨迹的每个前缀上）----
        for (x, y) in batch:
            with torch.no_grad():
                # 教师读取学生前缀，给出每个位置采样 token 的 log-prob
                logp_teacher = teacher.logprob(x, y)            # [T]  log ν(y_t|x,y_<t)
            logp_student_old = student.logprob(x, y).detach()   # [T]  采样时策略

            # token 级 dense advantage：教师相对学生的认可差
            A = logp_teacher - logp_student_old                 # [T]
            # 可选：hard clip / log-compression 防师生差距过大（Demystifying OPD）
            A = clip_signal(A, bound=CLIP_BOUND)

            # ---- 阶段 3: importance-sampling 策略梯度更新 ----
            logp_student = student.logprob(x, y)                # 当前策略
            ratio = torch.exp(logp_student - logp_student_old)  # ρ_t
            surr1 = ratio * A
            surr2 = torch.clamp(ratio, 1 - EPS, 1 + EPS) * A
            loss = -torch.minimum(surr1, surr2).sum()           # PPO-clip 风格
            loss.backward()

        optimizer.step(); optimizer.zero_grad()
```

### 5.2 OPD（full-vocab / top-k，GKD 分布对齐形式）

```python
def gkd_opd_step(student, teacher, x, lam=1.0, divergence="rkl", topk=None):
    """
    GKD 式 OPD：轨迹按 λ 混合，分布距离可选 FKL/RKL/JSD
    lam=1.0 → 严格 on-policy（学生轨迹）
    """
    # 轨迹来源：λ 概率学生自己生成，1-λ 概率用教师固定轨迹
    y = student.generate(x) if random.random() < lam else teacher.generate(x)

    p_logits = student.forward(x, y[:, :-1])          # 学生：每个位置的下一 token 分布
    with torch.no_grad():
        q_logits = teacher.forward(x, y[:, :-1])      # 教师：同前缀上的分布

    if topk:  # 只保留教师 top-k 支撑集；注意重归一化，否则梯度有偏
        idx = q_logits.topk(topk).indices
        p = student_dist(p_logits).restrict(idx, renormalize=True)
        q = teacher_dist(q_logits).restrict(idx, renormalize=True)
    else:
        p, q = softmax(p_logits), softmax(q_logits)   # full-vocab，O(B·T·V) 显存

    if divergence == "rkl":
        loss = (p * (p.log() - q.log())).sum(-1)      # KL(p‖q)
    elif divergence == "fkl":
        loss = (q * (q.log() - p.log())).sum(-1)      # KL(q‖p)
    else:  # jsd
        m = 0.5 * (p + q)
        loss = 0.5*(p*(p.log()-m.log())).sum(-1) + 0.5*(q*(q.log()-m.log())).sum(-1)
    return loss.mean()
```

### 5.3 OPSD（Self-Distilled Reasoner）

```python
def opsd_train(model, dataset, optimizer, steps):
    """
    OPSD: 同一模型两种条件 —— 教师看答案（开卷），学生只看题（闭卷）
    model: 唯一模型 πθ，教师分支 stop-gradient
    teacher_policy 固定为初始策略（隐式正则化）
    """
    teacher_ref = snapshot(model)                      # 冻结的初始策略（教师）
    for step in range(steps):
        for (x, answer_gt) in dataset.sample(batch_size=B):
            # ---- ① 学生闭卷采样轨迹（on-policy）----
            y = model.generate(x, sampling=True, max_len=T)    # ŷ ~ πθ(·|x)

            # ---- ② 构造特权上下文 ----
            # 教师输入 = 题目 + 参考答案（ground truth / 验证过的轨迹）
            teacher_ctx = f"Question: {x}\nReference answer: {answer_gt}\n"

            # ---- ③ 在学生轨迹的每个前缀上取两个分布 ----
            p_logits = model.forward(x, y[:, :-1])             # 学生：闭卷
            with torch.no_grad():
                q_logits = teacher_ref.forward(teacher_ctx, y[:, :-1])  # 教师：开卷

            # ---- ④ 最小化分布距离（论文主实验用 full-vocab）----
            loss = divergence_loss(p_logits, q_logits, kind="fkl")  # FKL/RKL/JSD
            loss.backward()
        optimizer.step(); optimizer.zero_grad()
```

### 5.4 一个直观的对照（SDPO 变体：特权信息 = 环境反馈）

知乎笔记（小何同学冷泡茶）给出的 SDPO 伪代码展示了 OPSD 思想的泛化——PI 不一定是参考答案，也可以是**环境返回的报错信息**：

```python
# 学生答错时：
attempt = model.sample(question)
success, feedback = env.evaluate(question, attempt)      # 代码评测器/单元测试
if not success:
    teacher_input = f"Question: {question}\nYour attempt: {attempt}\n" \
                    f"Feedback: {feedback}\nCorrect solution:"
    for t in range(len(attempt)):
        student_dist = model.forward(question, attempt[:t])
        teacher_dist = model.forward(teacher_input, attempt[:t])  # "自省教师"
        loss += KL(teacher_dist, student_dist)           # 定位关键错误 token
```

---

## 六、OPD 方法演进谱系

| 阶段 | 时间 | 代表工作 | 解决的问题 | 核心改进 |
|---|---|---|---|---|
| 奠基 | 2023–2024 | **GKD、MiniLLM** | 离线蒸馏的 exposure bias | 学生生成轨迹，教师在学生访问的状态上监督；MiniLLM 强调 reverse-KL |
| 工程化基线 | 2025.10 | **Thinking Machines OPD、Qwen3 实践** | RL 稀疏、SFT off-policy | sampled-token + reverse-KL surrogate，可复现 recipe，成本 ~RL 的 1/10 |
| 去除外部教师 | 2026.01 | **OPSD / Self-Distilled Reasoner** | 大教师在线推理昂贵、要求同词表 | 教师 = 同模型 + 特权信息 |
| 反馈解释错误 | 2026.01 | **SDPO** | RLVR 终局 0/1 奖励无法定位错误 token | 教师条件化环境反馈/成功 sibling rollout |
| 按样本路由 | 2026.04 | **SRPO** | SDPO 在正确样本上路径歧义、后期坍缩 | 正确轨迹走 GRPO，错误轨迹走 SDPO，按教师熵动态加权 |
| 方向/强度解耦 | 2026.04 | **RLSD** | 特权教师直接决定梯度方向 → 信息泄漏 | verifier 决定更新正负，teacher 概率比只调幅度 |
| 离线化 | 2026.04 | **Lightning OPD** | 在线教师服务的算力/系统开销 | teacher consistency 下缓存教师 log-prob |
| 上下文自蒸馏 | 2026.04 | **OPSDL** | 长上下文无关信息干扰推理 | PI 从"答案"换成"相关短上下文" |
| 多教师合并 | 2026 | **MiMo-MOPD** | 多领域专家合并回统一模型 | 领域路由 + RKL + outcome RL 混合 |
| 机理与修复 | 2026 | **Rethinking OPD、REOPOLD、OPRD、Purified OPSD、RLCSD、d-OPSD** | 更强教师反而更差、风格偏移、死记硬背、平台期 | 成功条件刻画、信号调控、表示空间蒸馏、PMI 提纯、对比信号、扩散语言模型适配 |

**工业实践认知**（知乎一线训练者 Cyril-KI、姜富春等）：OPD 常放在后训练**最后阶段**做"能力合并与收尾"——RL 分别训练数学/代码/Agent 专家（可以偏科），再用 OPD 把专家能力迁回基础模型；老师可以偏科，学生不必继承老师的副作用。

---

## 七、结论与选型建议

### 7.1 结论

1. **OPD 不是新算法，而是"采样协议 × 监督协议"的组合**：on-policy 采样解决暴露偏差，教师 token 级分布解决信号稀疏。它在形式上是蒸馏、运行上是带 dense 奖励的 RL，与 KL 正则化 RL 可数学统一。
2. **OPSD 是 OPD 的去外部教师版本**：用信息不对称（开卷/闭卷）构造自教师，免词表约束、省推理成本，在 ≤8B 规模匹配或超过 GRPO 且 token 效率 8–12×；但受模型容量与 PI 质量双重约束。
3. **OPD 是"探索催化剂"而非"能力灌注器"**：加速学生找到够得着的路径，不抬高能力上限。"更强教师→更好学生"不成立，师生分布相容性比教师绝对强度更重要。
4. **稠密信号是双刃剑**：标点、风格、长度都会进入目标，需 clip、mask、loss 系数、对比信号、PMI 提纯等手段控制模仿强度；长 CoT 模型的反思/自纠错行为尤其脆弱。
5. **适用性分界**：数学/代码有可验证奖励，RL 信号可靠；创作/知识类任务难写低偏差 reward，教师蒸馏（OPD/OPSD）更实用。理想方法应同时具备蒸馏的稠密 token 信号 + RLVR 的低偏差目标 + on-policy 数据——目前尚无标准答案。

### 7.2 工程落地 checklist

- [ ] 先 SFT 冷启动拉近师生分布，再上 OPD（防前缀扭曲）
- [ ] sampled-token 起步（性价比最高），追求上限再上 top-k / full-vocab
- [ ] top-k 必须重归一化或加 stop-gradient
- [ ] 对 advantage 信号做 hard clip / log-compression（防强教师负迁移、防刷长度）
- [ ] OPSD 教师固定为初始策略 + stop-gradient
- [ ] 监控生成中认知标记词（wait/maybe 等）数量，防反思行为崩塌
- [ ] 复用现有 PPO/GRPO 框架：rollout 组件不动，只替换奖励来源为教师 log-prob 差

### 7.3 注意事项

- **本报告中的 arXiv 编号、实验数据均来自知乎文章转述**（多数文章注明了论文出处），建议引用前按 §八 的原始链接二次核对；个别 2026 年论文编号格式特殊，未能全部独立验证，已如实标注来源。
- OPSD 存在同名歧义（Self-Distilled Reasoner vs π-Distill 的 OPSD 模块），引用时务必带上 arXiv 号或 PI 来源说明。
- Thinking Machines 博客是工程博客而非同行评审论文，但其 recipe 被学术界反复引用为事实基线。

---

## 八、参考资料

### 原始出处
1. Kevin Lu et al., *On-Policy Distillation*, Thinking Machines Lab Blog, 2025-10-27 — thinkingmachines.ai/blog/on-policy-distillation/ （代码：Tinker cookbook, recipes/distillation）
2. Agarwal et al., *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD)*, 2024
3. Gu et al., *MiniLLM: Knowledge Distillation of Large Language Models*, ICLR 2024
4. *Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models (OPSD)*, arXiv:2601.18734

### 机理 / 病理研究
5. 清华 THUNLP, *Rethinking On-Policy Distillation of LLMs: Phenomenology, Mechanism, and Recipe*, arXiv:2604.13016, ICML 2026 FoGen Workshop（代码 thunlp/OPD）
6. 港中文 + 腾讯 AI Lab, *Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations*, arXiv:2607.13399
7. *The Many Faces of On-Policy Distillation: Pitfalls, Mechanisms, and Fixes*, arXiv:2605.11182
8. *OPRD: On-Policy Representation Distillation*, arXiv:2606.06021
9. 浙大 + 阿里通义等, *Purified OPSD: On-Policy Self-Distillation Without Losing How to Think*, arXiv:2607.02234
10. 智源 + 清华, *RLCSD: Reinforcement Learning with Contrastive On-Policy Self-Distillation*

### 知乎主要参考文章（检索时间 2026-08-18）
11. 姚远（哈工大博士）《LLM后训练｜从监督信号的视角理解 SFT、RL 到 OPD 与 OPSD》— zhuanlan.zhihu.com/p/2065369932192338649
12. jamin《OPD系列：方法演进与SFT/RL的数学边界》— zhuanlan.zhihu.com/p/2069531934863122559
13. Cyril-KI《On-Policy Distillation(OPD)：一些工程实践与训练认知》— zhuanlan.zhihu.com/p/2070927698294317227
14. 模力AILab《On-Policy Distillation(OPD)：为什么大模型后训练要在学生自己的轨迹上蒸馏?》— zhuanlan.zhihu.com/p/2069863120466715393
15. Ferry《On-Policy Distillation：Policy Gradient 视角下的 OPD》— zhuanlan.zhihu.com/p/2058368749707776341
16. 欠阿贝尔两块钱《从分布视角解释LLM后训练：SFT、RL、OPD的差异》— zhuanlan.zhihu.com/p/2070328173166866733
17. wuxiaojun（粤港澳大湾区数字经济研究院）《强化学习小白理解GRPO(四)：SFT、RL 和 OPD 原来是一家人》— zhuanlan.zhihu.com/p/2072657717714499210
18. zklink《Rethinking On-Policy Distillation of LLMs 解读》— zhuanlan.zhihu.com/p/2067769832960008241
19. 秋水黑刀《OPRD：把OPD蒸馏拉回到表示空间》— zhuanlan.zhihu.com/p/2062916803983102197
20. 机器之心《2026开年关键词：Self-Distillation，大模型真正走向「持续学习」》— zhuanlan.zhihu.com/p/2004574638794089414
21. CVHub《OPSD：让同一个模型用"带答案的自己"训练"不带答案的自己"》— zhuanlan.zhihu.com/p/2051773072194135172
22. Frog《On-Policy Self-Distillation 202603（OPSD 论文精读）》— zhuanlan.zhihu.com/p/2014642833177466144
23. Leezhou《LLM 后训练蒸馏技术演进梳理：自蒸馏与RL的融合》— zhuanlan.zhihu.com/p/2067197522330776365
24. 《On-Policy Distillation 方法演进》— zhuanlan.zhihu.com/p/2062918960367120976
25. 小何同学冷泡茶《大模型在线蒸馏｜Notes》（含 SDPO 伪代码）— zhuanlan.zhihu.com/p/2023189391883936284
26. 姜富春《OPD(On-Policy Distillation)》（TML 博客全文翻译+注解）— zhuanlan.zhihu.com/p/2045821729470260443
27. 落蓝飞雪《[OPD系列] OPD最初标注参考文献 Thinking Machines Lab出品》— zhuanlan.zhihu.com/p/2058190705093238808
28. 《π-Distill：用 privileged information 做联合蒸馏》— zhuanlan.zhihu.com/p/2045079309610701829
29. 机器之心《你的自教师模型还在用参考解吗？d-OPSD》— zhuanlan.zhihu.com/p/2058624019235076091
30. 算法大作手《推理大模型后训练方案：Rubric引导的自蒸馏》— zhuanlan.zhihu.com/p/2051329130172507642
