# OPD（在线策略蒸馏）技术完全指南
## 问题背景
大语言模型后训练阶段长期存在两类核心痛点，始终没有完美的解决方案：
1. **离线方法的暴露偏差（Exposure Bias）**：SFT、离线知识蒸馏、DPO等基于固定数据集训练的方法，训练时模型总是在完美的前缀（人工标注/教师生成的正确上下文）上学习，而推理时必须基于自己可能生成错误的前缀继续预测，训练和推理分布严重不一致，导致长程生成、多步工具调用、Agent交互场景下误差指数级累积。
2. **在线RL的稀疏奖励与信用分配问题**：PPO、GRPO等在线强化学习方法虽然解决了分布偏移问题，但奖励信号通常是稀疏的（只有任务最终成功/失败的标量奖励），长程多步任务中无法定位到底哪一步决策出了问题，训练不稳定、样本利用率低、容易出现奖励黑客现象。

OPD（On-Policy Distillation，在线策略蒸馏，部分文献也称Online Preference Distillation在线偏好蒸馏）是2023年提出、2026年成为工业界主流的大模型后训练技术，完美融合了知识蒸馏的密集Token级监督和在线RL的同策略分布对齐优势，被DeepSeek-V4等大模型作为核心后训练技术，尤其在Agent、代码生成、数学推理等长程交互场景效果远超传统方法。

本文将从数学原理、架构设计、算法细节、工程实现、落地路径、调参指南全方面详细讲解OPD技术，内容包含完整的数学推导、架构图、可运行伪代码、工业界落地经验。

---
## 核心分析：数学原理与本质
### 1. 形式化定义
OPD的核心目标是：**在学生模型自身生成的轨迹分布上，最小化学生策略与教师策略的KL散度**。
#### 符号定义
| 符号 | 定义 |
|------|------|
| $x$ | 用户输入prompt/初始状态 |
| $y = (y_1, y_2, ..., y_T)$ | 长度为T的生成轨迹/序列 |
| $\pi_\theta(y|x)$ | 学生模型当前策略，参数为$\theta$ |
| $\pi_T(y|x)$ | 教师模型策略（可以是强模型、奖励模型、规则验证器、学生EMA版本） |
| $s_t = (x, y_{<t})$ | 第t步的上下文状态，包含输入和前t-1步生成的前缀 |
| $V$ | 词表大小 |
| $\pi_\theta(v|s_t)$ | 学生模型在状态$s_t$下生成token $v$的概率 |
| $\pi_T(v|s_t)$ | 教师模型在状态$s_t$下生成token $v$的概率 |

### 2. 目标函数的三种形式
根据计算KL散度的词表范围和方向，OPD有三种主流的目标函数形式：
#### 2.1 Full-Vocabulary OPD（全词表OPD）
对词表所有token计算KL散度，是理论最完整、效果最好但计算成本最高的形式：
$$
\mathcal{L}_{OPD}^{full}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ \sum_{t=1}^T D_{*}\left( \pi_\theta(\cdot|s_t) \parallel \pi_T(\cdot|s_t) \right) \right]
$$
其中$D_*$代表散度函数，根据KL方向分为三种：
##### （1）Forward KL（前向KL）
$$
D_{FKL}(p \parallel q) = \sum_{v \in V} p(v) \log \frac{p(v)}{q(v)}
$$
- 期望权重来自教师分布$p=\pi_T, q=\pi_\theta$
- 特性：Mode-Covering（模式覆盖），严厉惩罚"教师有概率但学生概率为0"的token，强制学生覆盖教师所有合理输出模式
- 优点：多样性好，不会遗漏教师的知识
- 缺点：当学生容量小于教师时，容易过度分散概率，生成内容平庸保守
- 适用场景：多专家合并、创造性生成任务
##### （2）Reverse KL（反向KL，工业界默认选择）
$$
D_{RKL}(p \parallel q) = \sum_{v \in V} p(v) \log \frac{p(v)}{q(v)}
$$
- 期望权重来自学生分布$p=\pi_\theta, q=\pi_T$
- 特性：Mode-Seeking（模式寻找），严厉惩罚"学生有高概率但教师概率极低"的token，强制学生只在教师认可的范围内生成
- 优点：精度高、稳定性好，不会生成教师认为错误的内容，训练不易崩溃
- 缺点：容易丢失多样性，极端情况会出现模式崩溃
- 适用场景：代码生成、数学推理、Agent工具调用等对正确性要求高的任务（工业界90%以上场景选择RKL）
##### （3）Generalized JSD（广义Jensen-Shannon散度）
$$
D_{JSD(\alpha)}(p \parallel q) = \alpha D_{FKL}(p \parallel \alpha p + (1-\alpha)q) + (1-\alpha) D_{RKL}(q \parallel \alpha p + (1-\alpha)q)
$$
- $\alpha \in [0,1]$：平衡系数，$\alpha=0$等价于RKL，$\alpha=1$等价于FKL，$\alpha=0.5$是标准JSD
- 特性：在Mode-Covering和Mode-Seeking之间灵活平衡
- 适用场景：需要同时兼顾正确性和多样性的场景
#### 2.2 Top-K OPD（Top-K截断OPD，工业界性价比首选）
全词表OPD需要计算教师所有V个token的logits，当词表大小为15万时，显存和计算成本很高，因此工业界通常采用Top-K截断：只对教师和学生概率最高的K个token计算KL散度，其余token的概率归一化后合并为一个"其他"类：
$$
\mathcal{L}_{OPD}^{topk}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ \sum_{t=1}^T D_{*}\left( \text{TopK}(\pi_\theta(\cdot|s_t), K) \parallel \text{TopK}(\pi_T(\cdot|s_t), K) \right) \right]
$$
- K通常取20/50/100，计算成本仅为全词表的1/1000~1/3000
- 效果损失小于2%，是目前工业界落地的首选方案
- 注意：K不能太小，否则会丢失有价值的长尾token信号
#### 2.3 Sampled-Token OPD（采样Token OPD，轻量版）
最轻量化的形式，只对学生当前采样出来的那一个token计算对数概率差，不需要计算整个词表的分布：
$$
\mathcal{L}_{OPD}^{sample}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ \sum_{t=1}^T \left( \log \pi_\theta(y_t|s_t) - \log \pi_T(y_t|s_t) \right) \right]
$$
- 计算成本最低，只需要教师输出单个token的对数概率
- 缺点：梯度方差大，信号稀疏，需要更大的batch size
- 适用场景：超大规模模型训练、算力极度受限的场景
### 3. 梯度推导（以Reverse KL为例）
我们对Reverse KL损失求梯度，可以得到OPD更新的本质：
首先展开单步RKL：
$$
D_{RKL}(\pi_\theta \parallel \pi_T) = \sum_v \pi_\theta(v|s_t) \log \pi_\theta(v|s_t) - \sum_v \pi_\theta(v|s_t) \log \pi_T(v|s_t)
$$
对参数$\theta$求梯度：
$$
\nabla_\theta D_{RKL} = \mathbb{E}_{v \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(v|s_t) \cdot \left( \log \frac{\pi_\theta(v|s_t)}{\pi_T(v|s_t)} + 1 \right) \right]
$$
令优势函数$A_t(v) = \log \pi_T(v|s_t) - \log \pi_\theta(v|s_t)$，则梯度可以写为策略梯度的形式：
$$
\nabla_\theta \mathcal{L}_{OPD} = -\mathbb{E} \left[ \sum_{t=1}^T \nabla_\theta \log \pi_\theta(y_t|s_t) \cdot A_t(y_t) \right]
$$
**核心发现**：OPD本质上是一个特殊的RL算法，其奖励信号就是教师模型在每一步给的Token级对数概率差$A_t(v)$：
- 如果学生采样的token $y_t$ 教师认为概率高（$A_t(y_t) > 0$），则提高该token的生成概率
- 如果学生采样的token $y_t$ 教师认为概率低（$A_t(y_t) < 0$），则降低该token的生成概率
- 这完美解决了传统RL的信用分配问题：每一个token都有即时的密集奖励，不需要等整个序列结束才能计算奖励
### 4. 与其他方法的数学本质区别
![OPD与其他后训练方法对比图](./assets/opd_comparison.jpg)
图2：OPD与其他后训练方法特性对比
| 方法 | 目标函数 | 数据分布 | 监督粒度 | 分布偏移 | 训练稳定性 |
|------|---------|---------|---------|----------|------------|
| SFT | 最小化$- \log \pi_\theta(y^*_t|s_t^*)$ | 教师/人工生成的固定前缀$s_t^*$ | Token硬标签 | 高 | 极高 |
| 离线KD | 最小化$D(\pi_T(\cdot|s_t^*) \parallel \pi_\theta(\cdot|s_t^*))$ | 教师/人工生成的固定前缀$s_t^*$ | Token分布级 | 高 | 高 |
| DPO | 最大化偏好对相对概率 | 离线偏好数据集 | 序列级偏好 | 中 | 高 |
| PPO/GRPO | 最大化$\mathbb{E}[\sum_t R_t] - \beta D_{KL}$ | 学生自生成轨迹 | 稀疏标量奖励 | 无 | 中 |
| OPD | 最小化$\mathbb{E}_{y \sim \pi_\theta}[\sum_t D(\pi_\theta \parallel \pi_T)]$ | 学生自生成轨迹 | Token分布级密集奖励 | 无 | 高 |
**OPD的核心优势**：既避免了离线方法的分布偏移，又解决了在线RL的奖励稀疏问题。
### 5. 为什么OPD训练效率极高？（EffOPD理论证明）
2026年EffOPD论文通过数学证明和实验验证了OPD训练高效的根本原因：
1. **低秩更新特性**：OPD的Hessian矩阵天然具有低秩结构，99%以上的参数更新能量集中在前1%的奇异方向上
2. **早期锁定效应**：训练开始几百步内，模型就会锁定正确的低秩更新方向，后续训练沿着这个方向稳定收敛，不会像RL那样剧烈震荡
3. **梯度一致性**：Token级密集监督使得梯度方向非常稳定，不需要像PPO那样做重要性采样裁剪、价值函数估计等复杂操作，训练稳定性大幅提升。
---
## 系统架构与核心流程
### 1. 整体架构图
![OPD在线策略蒸馏系统架构图](./assets/opd_architecture.jpg)
图1：OPD三层系统架构
```mermaid
flowchart LR
    subgraph 数据采样层
        A[Prompt数据集D] --> B[学生模型并行Rollout<br>生成多轮交互轨迹]
        B --> C{环境交互<br>工具调用/代码执行/搜索}
        C -->|环境反馈| B
        B --> D[完整学生轨迹<br>(s_1,y_1,s_2,y_2,...,s_T,y_T)]
    end
    subgraph 教师监督层
        D --> E[教师模型前向<br>在每个学生前缀s_t上输出logits]
        E --> F[Top-K截断/归一化<br>得到教师分布π_T(·|s_t)]
    end
    subgraph 训练更新层
        D --> G[学生模型前向<br>在每个前缀s_t上输出logits]
        F --> H[损失计算<br>Reverse KL/Top-K/采样Token]
        G --> H
        H --> I[反向传播更新学生参数θ]
        I -->|周期性更新| J[教师模型更新<br>EMA/硬拷贝/升级强模型]
        J --> E
    end
    I --> B
```

### 2. 完整训练流程（工业界标准实现）
![OPD训练流程图](./assets/opd_workflow.jpg)
图3：OPD标准训练流程
1. **初始化阶段**
   - 加载预训练学生模型，完成SFT warmup（必须先有基础能力，否则OPD训练不稳定）
   - 加载教师模型（通常量化为4bit/8bit节省显存，不保存梯度）
   - 初始化环境（工具沙箱、代码执行环境、搜索API等Agent交互环境）
   - 配置训练超参数（学习率、batch size、KL系数、Top-K大小等）
2. **采样阶段**
   - 从Prompt池中采样一批输入$x$
   - 学生模型与环境交互，自回归生成完整轨迹，记录每一步的状态$s_t$和采样的token $y_t$
   - 对长轨迹做截断/过滤，剔除无效样本（如重复生成、格式错误）
3. **教师监督阶段**
   - 教师模型在每一个学生生成的前缀$s_t$上做前向传播，输出logits
   - 对教师logits做Top-K截断，归一化得到概率分布
   - （可选）对特殊token（如padding、工具返回结果）做mask，不计算损失
4. **损失计算阶段**
   - 学生模型在相同的前缀$s_t$上做前向传播，输出logits
   - 计算每一步的RKL/FKL/JSD损失
   - （可选）加入步级权重：长轨迹后半段权重适当降低，避免后期误差累积
   - （可选）加入熵正则：鼓励适当探索，避免模式崩溃
5. **更新阶段**
   - 反向传播计算梯度，做梯度裁剪防止梯度爆炸
   - 优化器（通常AdamW）更新学生模型参数
   - 周期性（如每100步）用学生模型EMA权重更新教师模型，或升级到更强的教师版本
6. **评估阶段**
   - 周期性在验证集上评测任务成功率、准确率等指标
   - 早停：当验证集指标不再提升时停止训练，防止过拟合
---
## 算法伪代码与可运行实现
### 1. 最简版本OPD伪代码（100行以内可实现）
```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
# -------------------------- 初始化 --------------------------
# 加载模型：学生模型fp16/bf16，教师模型4bit量化节省显存
student = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
teacher = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",  # 教师用更大的模型
    load_in_4bit=True,
    device_map="auto"
)
teacher.eval()  # 教师模型不需要梯度
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.8B-Instruct")
optimizer = AdamW(student.parameters(), lr=2e-5, weight_decay=0.01)
config = {
    "max_seq_len": 2048,
    "top_k": 50,  # Top-K OPD的K值
    "kl_type": "rkl",  # 默认用Reverse KL
    "grad_clip": 1.0,
    "entropy_coeff": 0.01,  # 熵正则系数，避免模式崩溃
}
# -------------------------- 核心函数 --------------------------
def get_topk_dist(logits, k=50):
    """获取Top-K截断后的概率分布"""
    topk_logits, topk_indices = torch.topk(logits, k=k, dim=-1)
    topk_probs = F.softmax(topk_logits, dim=-1)
    # 构造稀疏分布：不在Top-K中的token概率为0
    full_probs = torch.zeros_like(logits).scatter_(-1, topk_indices, topk_probs)
    return full_probs, topk_indices, topk_probs
def opd_loss(student_logits, teacher_logits, mask=None):
    """计算OPD的Reverse KL损失"""
    # 对齐长度：学生/教师logits都是[bsz, seq_len, vocab_size]
    seq_len = min(student_logits.shape[1], teacher_logits.shape[1])
    student_logits = student_logits[:, :seq_len, :]
    teacher_logits = teacher_logits[:, :seq_len, :]
    if mask is not None:
        mask = mask[:, :seq_len].unsqueeze(-1)
    # 计算Top-K分布
    student_probs, _, _ = get_topk_dist(student_logits, config["top_k"])
    teacher_probs, _, _ = get_topk_dist(teacher_logits, config["top_k"])
    # 计算Reverse KL: sum_v p(v) log(p(v)/q(v)), p=student, q=teacher
    # 加epsilon防止log(0)
    eps = 1e-8
    rkl = student_probs * (torch.log(student_probs + eps) - torch.log(teacher_probs + eps))
    rkl = rkl.sum(dim=-1)
    # 熵正则：鼓励探索，防止模式崩溃
    entropy = -(student_probs * torch.log(student_probs + eps)).sum(dim=-1)
    loss = rkl - config["entropy_coeff"] * entropy
    # mask掉不需要计算损失的位置（如padding、tool返回）
    if mask is not None:
        loss = (loss * mask).sum() / mask.sum()
    else:
        loss = loss.mean()
    return loss, rkl.mean(), entropy.mean()
@torch.no_grad()
def rollout(prompt, env, max_steps=10):
    """学生模型与环境交互生成轨迹"""
    messages = [{"role": "user", "content": prompt}]
    done = False
    steps = 0
    trajectory = []
    while not done and steps < max_steps:
        # 学生模型生成下一步动作
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(student.device)
        outputs = student.generate(
            inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response})
        trajectory.append(("assistant", response))
        # 环境执行动作，获取反馈
        obs, done = env.step(response)
        if not done:
            messages.append({"role": "tool", "content": obs})
            trajectory.append(("tool", obs))
        steps += 1
    # 构造模型输入，准备计算logits
    full_inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", padding=True
    ).to(student.device)
    # 生成mask：tool返回内容、user输入不计算损失，只有assistant生成内容计算
    mask = torch.zeros_like(full_inputs)
    # 这里省略mask构造的细节，实际实现需要根据chat template标记assistant token位置
    return full_inputs, mask
# -------------------------- 训练循环 --------------------------
def train_step(prompts, env):
    student.train()
    total_loss = 0.0
    all_inputs = []
    all_masks = []
    # 1. 采样轨迹
    for prompt in prompts:
        inputs, mask = rollout(prompt, env)
        all_inputs.append(inputs)
        all_masks.append(mask)
    # padding到相同长度
    inputs = torch.nn.utils.rnn.pad_sequence(
        [i[0] for i in all_inputs], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    masks = torch.nn.utils.rnn.pad_sequence(
        [m[0] for m in all_masks], batch_first=True, padding_value=0
    )
    # 2. 学生前向
    student_outputs = student(inputs)
    student_logits = student_outputs.logits[:, :-1, :].contiguous()  # 移位，预测下一个token
    # 3. 教师前向（no_grad）
    with torch.no_grad():
        teacher_outputs = teacher(inputs)
        teacher_logits = teacher_outputs.logits[:, :-1, :].contiguous()
    # 4. 计算损失
    loss, rkl, entropy = opd_loss(student_logits, teacher_logits, masks[:, 1:])
    # 5. 反向传播更新
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), config["grad_clip"])
    optimizer.step()
    return loss.item(), rkl.item(), entropy.item()
```
### 2. 工业级实现优化点
#### （1）显存优化
- 教师模型4bit/8bit量化，不存储梯度
- 采用梯度检查点（Gradient Checkpointing），减少学生模型显存占用30%+
- 采用FlashAttention-2/3加速注意力计算，减少显存碎片
- 离线缓存教师logits（Lightning OPD方案）：对于不需要实时更新教师的场景，提前生成所有轨迹和教师logits，训练时不需要跑教师前向，速度提升10倍
#### （2）训练稳定性优化
- **步级归一化**：对每一步的KL损失做归一化，防止长序列梯度爆炸，StepOPSD论文证明这可以将训练稳定性提升40%
- **Prefix Drift Gate**：当学生生成的前缀偏离教师分布太远时（KL超过阈值），降低该步的损失权重，避免教师在坏前缀上给出不可靠的监督信号
- **EMA教师更新**：不直接硬拷贝学生参数到教师，而是用指数移动平均更新教师参数：$\theta_T = \tau \theta_T + (1-\tau)\theta_\theta$，$\tau$通常取0.99/0.999，大幅提升训练稳定性
- **特殊Token Mask**：对padding、工具返回结果、思考标签等不需要学习的token做mask，不计算损失
#### （3）效率优化
- 多环境并行采样：128-256个环境并行生成轨迹，提升采样吞吐量
- 序列打包（Sequence Packing）：将多个短轨迹打包到同一个batch，提升GPU利用率
- 分布式训练：采用FSDP/DeepSpeed ZeRO-3做分布式训练，支持70B+大模型训练
---
## 落地实现路径
### 阶段一：最小版本验证（1-2天）
**目标**：跑通OPD核心流程，验证在小模型上的效果
1. 基于HuggingFace Transformers实现最简版Top-K Reverse KL OPD
2. 用单卡GPU在0.5B/1.8B模型上做测试，任务选择简单的单轮数学/代码任务
3. 对比SFT基线，验证OPD在分布外测试集上的效果提升
**避坑点**：
- 必须先做SFT warmup，从base模型直接开始OPD训练大概率会崩
- 学习率设为SFT的1/2~1/5，太大容易发散
- 初始Top-K设为50，熵正则系数0.01
### 阶段二：工程优化适配Agent场景（1-2周）
**目标**：支持多轮工具调用、长程Agent任务训练
1. 集成Agent环境（代码沙箱、搜索API、文件操作工具等）
2. 实现多轮对话mask逻辑，只对assistant生成的内容计算损失，工具返回结果自动mask
3. 加入步级归一化、EMA教师、Prefix Drift Gate等稳定性优化
4. 实现多环境并行采样，提升训练效率
5. 在工具调用基准（如ToolBench、ALFWorld）上测试效果
**避坑点**：
- 长轨迹（>10步）必须加位置权重，后期步的权重适当降低，避免误差累积
- 教师必须和学生使用相同的chat template，否则会出现分布不匹配
- 工具调用格式必须严格对齐，否则模型容易学不会正确的工具调用语法
### 阶段三：工业级落地（2-4周）
**目标**：支持大模型训练、多教师合并、生产环境部署
1. 集成分布式训练框架（FSDP/DeepSpeed、Megatron-LM）
2. 支持多教师OPD：不同领域的专家教师，根据路由策略选择对应教师监督，适合多能力模型合并
3. 实现Lightning OPD离线缓存方案，降低训练成本
4. 加入完善的监控指标：每步KL、熵、教师-学生token重叠率、任务成功率等
5. 自动化流水线：数据采样→训练→评估→模型导出全链路自动化
**避坑点**：
- 全词表OPD在大模型上成本过高，优先用Top-K（K=50/100）方案，性价比最高
- 多教师场景下，按轨迹路由教师比按token路由教师更稳定，实现成本更低
- 训练过程中必须持续监控验证集任务成功率，KL损失下降不代表效果提升
### 开源实现参考
| 框架 | OPD支持情况 | 地址 |
|------|------------|------|
|  verl（字节跳动） | 原生支持Full/Top-K/Sampled OPD，支持分布式训练 | https://github.com/volcengine/verl |
| ms-swift（魔搭） | 支持GKD/OPD训练，开箱即用 | https://github.com/modelscope/ms-swift |
| OpenClaw-RL | Agent场景OPD实现，支持多环境交互 | https://github.com/openclaw/OpenClaw-RL |
| StepOPSD | 步骤级OPD官方实现，支持多步Agent | 论文arXiv:2605.27140 |
---
## 调参指南与常见问题
### 1. 超参数选择表
| 超参数 | 推荐值 | 调整方向 |
|--------|--------|---------|
| 学习率 | 1e-5 ~ 5e-5 | 小模型用大学习率，大模型用小学习率；训练不稳定调小 |
| Batch size | 32 ~ 256 | 越大梯度越稳定，根据显存调整，显存不够用梯度累积 |
| Top-K | 50 ~ 100 | 任务越复杂K越大，计算资源不足调小到20 |
| KL类型 | Reverse KL | 默认RKL，需要多样性时用JSD(α=0.3)，多专家合并用FKL |
| 熵正则系数 | 0.001 ~ 0.05 | 模式崩溃/输出单调查大，输出太乱/有错误调小 |
| EMA教师系数τ | 0.99 ~ 0.999 | 训练不稳定调大（接近1），希望教师更新快调小 |
| 梯度裁剪 | 0.5 ~ 1.0 | 训练不稳定/梯度爆炸调小 |
| SFT warmup步数 | 100 ~ 500步 | 从base模型开始训练时warmup步数要足够，从SFT模型开始可以减少 |
### 2. 常见问题排查
| 问题现象 | 原因 | 解决方案 |
|----------|------|---------|
| 训练loss震荡不收敛 | 学习率太大/教师模型太弱/没有SFT warmup | 降低学习率；换更强的教师；增加SFT warmup步数 |
| 生成内容重复/模式崩溃 | 熵正则太小/长时间训练/RKL太强 | 增大熵正则系数；加入FKL混合；早停；降低训练轮数 |
| 输出存在明显错误/幻觉 | 教师在坏前缀上给出错误监督/K值太小 | 增大Top-K；加入Prefix Drift Gate；提升教师模型能力 |
| 长任务效果差/误差累积 | 长序列梯度爆炸/后期步权重太高 | 加入步级归一化；对后半段步降权；分段训练 |
| 训练速度太慢 | 教师前向占比太高/采样效率低 | 用量化教师；离线缓存教师logits；增加并行环境数；用Top-K替代全词表 |
| 工具调用格式错误 | 没有对工具调用mask/教师模板不匹配 | 正确实现assistant mask；确保学生和教师用相同的chat template |
---
## 前沿变体与未来方向
1. **StepOPSD（步骤级OPD）**：将轨迹拆解为以动作为中心的步骤片段，利用后见之明（最终结果）给每个步骤重新打分，解决长程Agent任务的细粒度信用分配问题，在ALFWorld等多步任务上效果提升15%+
2. **OPRD（表示空间OPD）**：不监督输出层logits，改为监督中间层隐藏状态，跨架构/跨分词器场景下也能蒸馏，显存占用更低，训练更快
3. **Direct OPD/弱到强OPD**：打破教师必须强于学生的限制，弱模型也可以作为教师监督强模型，甚至模型可以自我蒸馏持续进化
4. **Lightning OPD**：将在线OPD离线化，缓存教师logits，训练速度提升10倍，精度损失小于2%，适合工业界大规模落地
5. **GRM+OPD（DeepSeek-V4方案）**：用生成式奖励模型（输出结构化评价文本而非标量分数）作为教师，同时学习输出分布和评价能力，在Agent场景下效果显著优于传统标量奖励RL
---
## 参考资料
1. MiniLLM: On-Policy Distillation of Large Language Models (arXiv:2306.08543, 2023)
2. On-policy Distillation of Language Models: Learning from Self-Generated Mistakes (ICLR 2024)
3. StepOPSD: Step-Aware Online Preference Distillation for Agent Reinforcement Learning (arXiv:2605.27140, 2026)
4. EffOPD: Efficient On-Policy Distillation via Early Low-Rank Lock-in (2026)
5. OPRD: On-Policy Representation Distillation (2026)
6. DeepSeek-V4 Technical Report (2026)
7. Verl Documentation: On-Policy Distillation (字节跳动, 2026)
8. ModelScope Swift: GKD/OPD Trainer Documentation (阿里魔搭, 2026)
9. 从SFT/RL到OPD: 给LLM Post-training初学者的直觉指南（知乎@Chase, 2026）
10. On-Policy Distillation原理、损失函数与工程实现详解（知乎@Orient, 2026）
11. OPD深度解析: 从数学推导到DeepSeek V4、SWIFT与verl实践（知乎@banana, 2026）
12. 三万字长文精讲2026上半年的On-Policy Distillation（知乎@Lei Tian, 2026）
