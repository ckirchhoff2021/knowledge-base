# OPD（在线策略蒸馏/在线偏好蒸馏）技术调研报告
## 问题背景
大语言模型后训练阶段，传统的SFT监督微调、DPO离线偏好优化、PPO/GRPO在线强化学习等方法存在各自的瓶颈：
1. **离线方法（SFT/离线KD/DPO）**：基于固定数据集训练，存在exposure bias（暴露偏差）问题，训练和推理分布不一致，泛化性差，学生只能学到教师在固定数据上的知识，无法应对动态交互场景
2. **在线RL方法（PPO/GRPO）**：奖励信号稀疏（尤其是长程Agent任务），存在严重的信用分配问题，训练不稳定，样本利用率低
3. **传统离线知识蒸馏**：学生在教师预先生成的静态数据上学习，分布偏移严重，"纸上谈兵"，无法在真实交互场景中有效纠错

OPD（On-Policy Distillation/Online Preference Distillation，在线策略蒸馏/在线偏好蒸馏）作为2026年大模型后训练领域的热点技术，融合了在线RL的分布对齐优势和知识蒸馏的密集监督效率，尤其在Agent强化学习场景取得了显著效果，DeepSeek-V4等主流大模型已将OPD作为核心后训练技术。

---
## 核心分析
### 1. OPD核心定义
OPD是一种**同策略（On-Policy）**的蒸馏技术，核心思想是：**让学生模型自己生成交互轨迹（Rollout），教师模型在学生生成的轨迹上实时提供逐Token的分布级监督，通过KL散度损失让学生对齐教师的输出分布**。
> 核心公式（Reverse KL散度损失）：
> $$L_{OPD} = \mathbb{E}_{s_t \sim \pi_\theta} \left[ D_{KL}\left( \pi_\theta(\cdot|s_t) \parallel \pi_{teacher}(\cdot|s_t) \right) \right]$$
> 其中：
> - $s_t$：学生模型在第t步的上下文（用户输入+学生之前生成的前缀）
> - $\pi_\theta$：学生模型当前策略
> - $\pi_{teacher}$：教师模型策略（可以是强模型、生成式奖励模型GRM、规则验证器、甚至学生自身的历史版本）
> - 采用Reverse KL（学生分布为权重），强制学生只在教师有概率的区域分配质量，避免生成分布外的错误动作。

### 2. 核心流程
标准OPD训练流程分为4步：
1. **学生采样（Student Rollout）**：学生模型与环境（工具、沙箱、API等）交互，自主生成完整的交互轨迹（多轮对话、工具调用、推理过程）
2. **教师前向（Teacher Forward）**：教师模型在学生轨迹的每一步前缀上做前向传播，输出全词表的logits分布
3. **学生前向（Student Forward）**：学生模型在相同前缀上输出自己的logits分布
4. **蒸馏更新（KL Distillation）**：计算两个分布的KL散度作为损失，反向传播更新学生模型参数

### 3. 关键技术特性
| 特性 | 说明 |
|------|------|
| **同策略无偏移** | 训练数据完全来自学生当前策略的生成分布，完全消除exposure bias，训练和推理分布完全一致，泛化性强 |
| **密集Token级监督** | 教师提供逐Token的分布级监督，而不是稀疏的标量奖励，完美解决长程Agent任务的信用分配问题，每一步的错误都能被精准纠正 |
| **无奖励模型依赖** | 不需要训练单独的奖励模型，避免奖励黑客（Reward Hacking）问题 |
| **训练稳定性高** | KL散度损失天然限制策略更新幅度，不会像PPO那样容易出现策略崩溃，训练过程平滑 |

### 4. 与其他后训练方法的对比
| 对比维度 | OPD | PPO | DPO | 离线知识蒸馏 | SFT |
|---------|-----|-----|-----|-------------|-----|
| 样本来源 | On-Policy（学生自生成轨迹） | On-Policy（策略自生成） | Mix-Policy（自生成+偏好样本） | Off-Policy（教师预生成固定数据） | Off-Policy（人工标注数据） |
| 监督信号 | 逐Token分布级密集监督 | 稀疏标量奖励 | 成对偏好排序 | 逐Token分布级密集监督 | 硬标签单点监督 |
| 损失核心 | Reverse KL散度 | 裁剪优势函数+KL正则 | 偏好对比损失 | Forward KL/MSE | 交叉熵 |
| 分布偏移 | 无 | 无 | 中等 | 严重 | 严重 |
| 信用分配 | 完美解决（逐Token监督） | 困难（稀疏奖励） | 中等（序列级偏好） | 无（静态数据） | 无（静态数据） |
| 训练稳定性 | 高 | 中（易发散） | 高 | 高 | 高 |
| 计算成本 | 中（需在线跑教师前向） | 高（需采样+奖励模型+价值模型） | 中 | 低（静态数据训练） | 低 |
| 泛化性 | 极佳 | 好 | 中 | 差 | 差 |
| 适用场景 | Agent长程交互、复杂推理 | 游戏、简单决策场景 | 通用偏好对齐 | 小模型能力继承 | 基础能力注入 |

---
## 关键技术点和结论
### 1. 主流OPD变体
| 变体名称 | 核心创新 | 效果 |
|---------|---------|------|
| **OPD+** | 重新设计优势项，显式构造奖励项提升教师到学生的能力迁移效率 | 能力迁移效率提升30%以上，相同训练步数下学生性能更接近教师 |
| **Direct OPD（直接OPD）** | 打破"教师必须强于学生"的假设，让弱模型在自身rollout上蒸馏强模型，无需每轮迭代重新生成数据 | 弱到强泛化场景有效，可作为模型持续自我改进的机制 |
| **StepOPSD（步骤级在线偏好蒸馏）** | 将整条轨迹拆解为以动作为中心的Step片段，用后见之明（知道最终结果）重新给每个Step打分，将Token级对数概率差转化为优势加权项，加入步级归一化避免方差爆炸 | 完美解决多轮Agent任务的信用分配问题，在ALFWorld、Search-QA等任务上超越标准OPD 15%+ |
| **Lightning OPD** | 将在线蒸馏过程离线化，提前缓存教师在学生轨迹上的分布，训练时无需实时跑教师前向 | 训练速度提升5-10倍，精度损失小于2% |
| **GRM+OPD（DeepSeek-V4方案）** | 用生成式奖励模型（Generative Reward Model，输出结构化评价文本而不是标量分数）作为教师，在OPD过程中同时学习教师的评价能力和输出分布 | 在Agent场景下显著提升模型的工具使用合理性和任务成功率，同时让模型思维链更符合人类工程师的行为模式 |
| **Weak-to-Strong OPD** | 扩展教师范围到弱教师、同质教师，系统研究"什么样的教师、以什么方式、在哪些轨迹上监督"最有效 | 发现即使是弱教师在学生自生成轨迹上的监督，也能显著提升学生能力 |

### 2. 落地优势（尤其Agent场景）
1. **长程任务表现优异**：逐Token监督解决了传统RL在多步工具调用、代码生成、搜索总结等长程Agent任务中的信用分配难题，OpenClaw-RL实验显示OPD在16步交互任务上的成功率从Binary RL的0.23提升到0.72，结合Binary RL可达到0.81
2. **多环境适配**：天然支持Terminal/GUI/SWE/Tool-call等多种Agent环境，不需要为每个环境单独设计奖励函数
3. **训练成本可控**：不需要大量人工偏好标注，训练稳定性远高于PPO/GRPO，样本利用率是传统RL的3-5倍
4. **风格对齐能力强**：OPD可以对齐教师的输出风格、思维模式，DeepSeek-V4用OPD对齐人类工程师的行为模式后，Agent思维链会自然出现符合人类工作习惯的表达（如任务间隙的吐槽、进度说明等）

### 3. 核心结论
1. OPD是当前大模型后训练，尤其是Agent强化学习领域最具落地价值的技术方向，兼顾了在线RL的泛化性和蒸馏方法的训练效率
2. OPD的核心价值在于解决了传统方法的两大痛点：**暴露偏差**和**信用分配问题**
3. 生成式奖励模型（GRM）+ OPD的组合，将成为未来Agent后训练的主流范式，替代传统的PPO/GRPO+标量奖励模型方案
4. 自蒸馏（教师是学生自身历史版本）方向的OPD（如R-Zero、Absolute Zero）为大模型零数据自我进化提供了可行路径，但仍存在奖励黑客和迭代崩溃的稳定性问题待解决

---
## 落地方案
### 1. OPD训练最小实现方案
#### 环境依赖
- PyTorch 2.0+
- Transformers 4.40+
- Accelerate
- 教师模型（可选择同规模更强的检查点、生成式奖励模型、或者强API模型）
- Agent环境（工具沙箱、代码执行环境、搜索API等）

#### 核心代码逻辑
```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载学生和教师模型
student = AutoModelForCausalLM.from_pretrained("student_model_path", device_map="auto")
teacher = AutoModelForCausalLM.from_pretrained("teacher_model_path", device_map="auto", torch_dtype=torch.bfloat16)
teacher.eval()  # 教师模型不需要梯度
tokenizer = AutoTokenizer.from_pretrained("student_model_path")

optimizer = torch.optim.AdamW(student.parameters(), lr=2e-5)

def opd_train_step(prompt, env):
    """
    单步OPD训练
    prompt: 初始用户指令
    env: Agent环境，支持step()方法执行动作返回反馈
    """
    student.train()
    total_loss = 0.0
    context = [{"role": "user", "content": prompt}]
    done = False
    
    while not done:
        # 1. 学生模型生成下一步动作
        inputs = tokenizer.apply_chat_template(context, return_tensors="pt").to(student.device)
        with torch.no_grad():
            student_logits = student(inputs).logits[:, -1, :]
            action = torch.argmax(student_logits, dim=-1)
            action_text = tokenizer.decode(action)
        
        # 2. 环境执行动作，获取反馈
        obs, done = env.step(action_text)
        context.append({"role": "assistant", "content": action_text})
        if not done:
            context.append({"role": "tool", "content": obs})
        
        # 3. 教师模型在当前上下文上的分布
        inputs_with_action = tokenizer.apply_chat_template(context, return_tensors="pt").to(student.device)
        with torch.no_grad():
            teacher_logits = teacher(inputs_with_action[:, :-1]).logits[:, -1, :]
            teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # 4. 计算Reverse KL损失：D_KL(student || teacher)
        student_logits = student(inputs_with_action[:, :-1]).logits[:, -1, :]
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        
        # 5. 反向传播更新
        total_loss += kl_loss.item()
        kl_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(context)
```

#### 训练流程建议
1. 先完成SFT基础训练，让学生模型具备基础的指令遵循和工具调用能力
2. 先用小规模数据做OPD warmup，再逐步扩大并行环境数量提升训练效率
3. 建议加入步级归一化：对每一步的KL损失做归一化，避免长序列梯度爆炸
4. 定期在验证集上评测任务成功率，早停避免过拟合教师的错误
5. 如果计算资源有限，可以采用Lightning OPD方案：先批量生成学生轨迹，缓存教师所有步的logits，再离线训练，不需要训练时同时跑两个模型

### 2. 工程优化点
- **显存优化**：教师模型用4bit/8bit量化加载，不需要存储梯度，可大幅降低显存占用
- **并行训练**：采用多环境并行采样（建议32-128个环境并行），提升采样效率
- **教师选择策略**：
  - 初期训练用强模型作为教师，快速提升基础能力
  - 中后期可以切换到生成式奖励模型/规则验证器作为教师，提升任务成功率
  - 训练后期可以用学生自身的EMA（指数移动平均）版本作为教师，做自蒸馏提升稳定性
- **KL方向选择**：初期用Reverse KL避免学生生成错误内容，后期可以混合少量Forward KL提升多样性

---
## 注意事项
1. **教师上限问题**：OPD中学生的性能上限由教师决定，如果教师本身存在错误，学生也会学习到这些错误，需要定期更新教师模型
2. **计算成本**：训练过程中需要同时跑学生和教师两个模型的前向，相比SFT/DPO训练成本更高，建议用量化+离线缓存logits的方式降低成本
3. **分布保守性**：纯Reverse KL训练会让学生过于保守，不敢尝试教师没有出现过的正确路径，建议适当加入探索噪声或混合少量RL损失提升探索能力
4. **Agent环境适配**：OPD要求环境反馈准确，如果环境本身存在错误（如工具返回错误结果），会导致教师给出错误监督，需要做好环境容错
5. **自蒸馏稳定性**：使用学生自身作为教师的自蒸馏方案，需要加入多样性约束和定期验证，避免迭代过程中出现能力下降（迭代崩溃）和奖励黑客问题
6. **评估指标**：OPD训练过程中不能只看KL损失下降，必须同时评测实际任务成功率，KL损失低不代表实际任务效果好，可能是学生完全复制了教师的输出但没有学到真实能力。

---
## 参考资料
1. StepOPSD: Step-Aware Online Preference Distillation for Agent Reinforcement Learning（arXiv:2605.27140）
2. On-Policy Distillation: Policy Gradient视角下的OPD（知乎@Ferry）
3. OpenClaw-RL: 当强化学习长出"龙虾触角"（知乎@AI大排档）
4. 浅谈DeepSeek-V4的OPD（知乎@小拆）
5. OPD前沿论文精选:从理论到实践的全景解读（知乎@扎西得嘞）
6. 最近比较火的OPD方向总结（知乎@zklink）
7. OpenClaw-RL源码阅读笔记: On-Policy Distillation（知乎@罗西的思考）
8. 从DPO到GRPO:偏好优化与强化优化的边界（知乎@智能计算系统手记）
