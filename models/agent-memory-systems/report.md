# Agent Memory 系统设计与开源商业实现技术报告：从 Hermes、Mem0 到 Codex 的长期记忆架构

> **报告日期**：2026-10-26　**快照口径**：本文涉及的开源仓库行为、文档与源码均为 2026-10-26 实测；商业产品的功能边界变化快，引用前请重新核对官方文档。
>
> **TL;DR**：Agent Memory 已从「把历史塞进 Prompt」的上下文技巧，演进为独立的基础设施层。本文以四个代表性切片剖析这一层：**Hermes**（轻量 Markdown 文件记忆 + Provider 插件生态）、**Mem0**（生产级事实抽取与多信号检索的开源/托管双轨）、**OpenAI Codex**（Rollout Extraction → Global Consolidation 两阶段后台记忆流水线），以及 Claude Code、Letta、Devin/Windsurf、Honcho、Supermemory、Zep 等商业级模块。核心结论：记忆系统的真正分水岭不在于「用不用向量库」，而在于三个架构决策——**写时抽取还是读时重判、谁来做 consolidation（合并去重消解矛盾）、记忆是否可审计可删除**。选型上，个人本地 Coding Agent 用文件/Git 记忆即可，跨用户产品优先 Mem0/Honcho 类服务，强时序与合规场景选 Zep 式双时态图谱，长生命周期自治 Agent 才需要 MemFS + Dreaming 式重型架构。

## 一、为什么 Agent 需要一层独立的记忆

### 1.1 无状态 LLM 的三个断层

主流大模型在推理时是**无状态（stateless）**的：每次 API 调用只看到本次请求里携带的 token，权重不会因为对话而更新。当模型被包进一个 Agent（规划 + 工具调用 + 多轮执行的循环）后，无状态性造成三个断层：

1. **会话断层（Session Reset）**：进程退出、上下文压缩、窗口滑动之后，上一段经历对模型不可见。用户上周调试过的环境约束、明确否决过的方案，本周要重新讲一遍。
2. **知识断层（Knowledge Gap）**：模型权重只反映预训练截止前的公共知识，而 Agent 面对的是私有仓库、个人偏好、本次任务的中间产物——这些永远不会出现在权重里。
3. **学习断层（Learning Leak）**：Agent 在失败中获得的教训（「这个仓库必须先跑 codegen」「这个用户只接受精简表格」）无法沉淀，同类型错误在第 N 个任务里重演。

记忆层的目标就是补上这三个断层，让 Agent 的有效经验跨 **turn（轮次）→ session（会话）→ user（用户）→ workspace/project（工作区）** 四个尺度持续累积。

### 1.2 Memory ≠ RAG：边界到底在哪里

工程上最常见的误解是把 Memory 等同于 RAG。二者共享「检索后注入」的机制，但关注点正交：

| 维度 | RAG | Agent Memory |
|---|---|---|
| 数据来源 | 预存在的外部语料（文档、网页、代码） | Agent 自己的交互轨迹与派生结论 |
| 写入主体 | 离线/批量灌入，与 Agent 运行解耦 | Agent 运行时持续产生，写路径是系统的一部分 |
| 时间属性 | 基本是静态知识 | 强时序：会更新、会失效、会被删除 |
| 典型条目 | 「SDK v3 的鉴权头叫 X-Key」 | 「这个项目上周从 v2 迁到了 v3，旧写法会报错」 |
| 核心难题 | 索引质量、召回率、chunk 策略 | 抽取质量、矛盾消解、遗忘与隐私、范围隔离 |

一个实用判断：**RAG 回答「世界知道什么」，Memory 回答「我们之间发生过什么、我因此学到了什么」**。生产系统里两者并存且共用检索栈，但写路径和治理策略必须分开设计。

### 1.3 从上下文技巧到独立基础设施

记忆方案的演进大致经历了四代，驱动力始终是「上下文窗口不够 → 窗口变大后问题仍未消失」：

1. **全量历史（Full History）**：把所有对话拼进 prompt。实现成本为零，但 token 成本二次增长，且 LLM 存在「中间迷失」（lost-in-the-middle），长历史里的关键信息反而被忽略。
2. **摘要记忆（Summary Memory）**：用 LLM 把旧对话压缩成 running summary。LangChain 的 `ConversationSummaryMemory` 是代表。问题是摘要有损、不可追溯、旧错误会被反复复读。
3. **检索式外部记忆（Retrieval Memory）**：把经历切块存入向量库，按当前 query 检索 top-k 注入。记忆容量与窗口解耦，但「切块的原文」噪声大、实体关系断裂。
4. **抽取-巩固式记忆（Extract & Consolidate）**：写入时由模型把轨迹抽取成原子化事实/笔记，读出时多信号检索；后台还有 consolidation 环节负责合并、去重、消解矛盾。Mem0、A-MEM、Codex、Letta Dreaming 均属此代——这也是 2025–2026 年商业系统收敛的主流形态。

值得强调：即便上下文窗口做到 1M token，第四代仍不可替代。长上下文解决的是「装得下」，记忆层解决的是「信噪比、可更新、可审计、跨会话隔离」——把 1M token 的旧轨迹原样塞给模型，既贵也会稀释注意力。

## 二、记忆分类与生命周期：所有系统的统一坐标系

### 2.1 四类记忆：工作 / 情景 / 语义 / 程序

认知科学的四分法对系统设计最有指导意义，综述 arXiv:2404.13501 与后续记忆综述均沿用：

- **工作记忆 Working**：当前上下文窗口本身——系统指令、工具 schema、最近若干轮。载体是 KV-Cache，成熟度最高。
- **情景记忆 Episodic**：「我经历过什么」——带时间、任务、结果标记的事件序列。Generative Agents（arXiv:2304.03442）、EM-LLM（arXiv:2407.09450）、A-MEM（arXiv:2502.12110）是代表。
- **语义记忆 Semantic**：从经历中抽象出、与具体事件解耦的事实、偏好、实体关系。Mem0 的 facts、Zep 的时序知识图谱是代表。
- **程序记忆 Procedural**：「我知道怎么做」——工作流、工具使用习惯、评审规范。形态上常常不是记忆条目，而是 Skills、自改 system prompt、工具使用先验。成熟度最低、长期复利最高。

另外两个正交轴同样重要：**形式轴**（参数记忆 / 激活记忆 / 外部文本-向量-图谱记忆）与**动态轴**（只读 / 可更新 / 可遗忘）。任何一条记忆都能在三个轴上定位，例如「外部 × 情景 × 可演化」就是 A-MEM 式笔记。

### 2.2 记忆生命周期七环节

把四类记忆放进时间维度，所有记忆系统——无论文件式还是服务式——都是同一条流水线的不同裁剪：

```text
capture → extract → store → retrieve → inject → consolidate → forget
 捕获       抽取      存储      检索       注入       巩固        遗忘
```

1. **捕获 Capture**：记录原始轨迹（消息、工具调用、产物路径）。关键决策是全量留痕还是只留模型判定相关的片段。
2. **抽取 Extract**：把轨迹变成记忆候选。可由 Agent 主动调用工具写（Hermes），也可由后台模型异步抽取（Codex Phase 1、Mem0 add）。抽取时机与质量是第一大架构分水岭。
3. **存储 Store**：Markdown 文件、SQLite、向量索引、知识图谱、bitemporal 表通常**多写并存**——原文留档、索引用向量、关系用图。
4. **检索 Retrieve**：向量语义召回只是一路；生产系统普遍叠加 BM25 关键词、实体链接、时间过滤、权限过滤后做融合（fusion / rerank）。
5. **注入 Inject**：决定把哪些记忆放进 prompt、放在什么位置、以什么格式。冻结快照（Hermes）保护 prefix cache，动态注入更灵活但破坏缓存。
6. **巩固 Consolidate**：合并重复、消解矛盾（新事实覆盖旧事实）、把情景抽象为语义、把语义沉淀为程序。这是第二大分水岭，决定记忆会不会越用越脏。
7. **遗忘 Forget**：TTL、容量上限、用户删除权（GDPR）、敏感凭据清除。遗忘不是可选项，是隐私安全与信噪比的共同要求。

### 2.3 范围隔离：多租户记忆的主键设计

记忆一旦跨用户共享存储，**范围（scoping）就是安全边界**。Mem0 的主键设计值得直接参考：

- `user_id`：记忆属于谁；
- `agent_id`：哪个 Agent 写入/使用（同一用户可有多个性格、职能不同的 Agent）；
- `run_id`：哪次执行（调试、回放、临时隔离用）；
- `metadata`：`category`、`app`、自定义标签，以及时间戳与来源指针。

文件式系统用路径实现同样的隔离：`~/.claude/projects/<project>/`、`~/.hermes/memories/`、`.devin/rules/`。**无论哪种形态，检索时必须把 scope 作为硬过滤条件而非 metadata 软匹配**——多租户记忆泄漏几乎都源于此。

<figure>
<iframe src="diagrams/1-memory-lifecycle.html"></iframe>
<figcaption>图 1：Agent Memory 的四类记忆、七环节生命周期与范围主键（2026-10-26）</figcaption>
</figure>

## 三、Hermes：轻量文件记忆 + Provider 插件架构

Hermes（Nous Research 的本地 Agent，本机安装于 `~/.hermes/hermes-agent`）的记忆设计代表了「**极简内建 + 可插拔外接**」路线：内建记忆刻意做小做透明，重型能力交给外部 Provider 插件。

### 3.1 两个内建文件：MEMORY.md 与 USER.md

内建记忆只有两个纯文本文件，位于 `~/.hermes/memories/`：

| 文件 | 内容 | 字符上限（默认） |
|---|---|---|
| `MEMORY.md` | Agent 的个人笔记：环境事实、约定、经验教训 | 2200 chars（`memory_char_limit`） |
| `USER.md` | 用户画像：偏好、沟通风格、长期背景 | 1375 chars（`user_char_limit`） |

这两个数字在 `tools/memory_tool.py` 与 `agent/agent_init.py` 中均可查到，且可通过配置覆盖。两个文件按裸 `§` 分隔符切成卡片，由 `MemoryManager` 统一管理。

### 3.2 冻结快照模式：为 prefix cache 而设计

Hermes 文档明确记载了一个反直觉但工程上极重要的决策：

- 会话启动时，两个文件的内容被渲染成 system prompt 中的**冻结快照（frozen snapshot）**；
- 会话进行中 Agent 用 `memory` 工具写入的内容**立即落盘，但不改变当前 system prompt**——要等下次会话才出现在注入块里；
- 工具返回值始终展示实时内容，所以 Agent 当前轮就能看到自己刚写了什么。

原因是 **prefix cache**：现代推理引擎对 system prompt 前缀做 KV 缓存，中途改 system prompt 会让整段缓存失效，延迟与成本同时上升。冻结快照用「下轮生效」换取整轮缓存命中，是本地 Agent 高性价比的关键设计。

### 3.3 写入工具与「容量满了让 Agent 自己收拾」

Agent 通过 `memory` 工具操作记忆，动作只有三个：`add`、`replace`、`remove`。

- `replace/remove` 用**唯一子串匹配**定位卡片，避免按编号编辑带来的错位；
- 写入前有**精确去重**与**安全扫描**：防提示注入、防凭据外泄、防不可见 Unicode；
- 容量超限时工具**不做自动压缩**，而是返回错误，让 Agent 在同一轮自行合并/删除旧卡片后重试。

最后一点是典型的「把智能留给模型、把机制留给代码」：自动压缩需要额外 LLM 调用且可能误删，而让 Agent 在收到容量错误后自主决定「哪条旧记忆已经过时」，既省一次后台任务，决策也与当前任务语境对齐。

### 3.4 session_search：独立于记忆的原文回放

Hermes 把「记忆」与「会话检索」明确分成两套能力：

- 历史会话存于 **SQLite（FTS5 全文索引）**；
- `session_search` 返回的是**原文片段**，不经 LLM 总结；
- 这对应认知轴上的情景记忆原文回放，而 MEMORY.md 里存的是提炼后的语义/程序记忆。

「原文走 FTS、提炼走文件」的双轨设计避免了摘要不可追溯的通病——任何记忆决策都能回到原始会话核对。

### 3.5 外部 Provider：一个生态位，八个生命周期 Hook

`MemoryManager` 强制**最多一个外部 Provider**，理由很直白：防止工具 schema 膨胀和多个记忆后端互相冲突。文档列出的内建适配包括 **Honcho、OpenViking、Mem0、Hindsight、Holographic、RetainDB、ByteRover、Supermemory**。外部 Provider 与内建记忆并存，而非替换。

`agent/memory_provider.py` 中的 `MemoryProvider` ABC 定义了 Provider 可挂载的完整生命周期：

| Hook | 时机与职责 |
|---|---|
| `initialize` / `shutdown` | 建连、资源预热与优雅退出 |
| `system_prompt_block` | 提供静态 system prompt 注入块 |
| `prefetch` / `queue_prefetch` | 每轮前的（后台）召回 |
| `sync_turn` | 每轮后的异步写入 |
| `on_session_end` | 会话结束时的整段抽取 |
| `on_memory_write` | 内建记忆写入的镜像（双写） |
| `on_pre_compress` | 上下文压缩前的检查点抽取 |
| `on_delegation` / `on_session_switch` 等 | 子任务观察、会话 ID 轮换等运行时事件 |

其中 `on_pre_compress` 有一套 v2 **fail-closed checkpoint 合同**：Provider 成功返回意味着持久化检查点已提交；若配置为严格模式，抽取失败会**阻止有损压缩继续进行**。这解决了「先压缩、再抽取、抽取失败 → 信息永久丢失」的竞态——记忆系统的可靠性合同比功能多寡更值得关注。

**Hermes 小结**：用 ~3.5K 字符的两个文件覆盖 80% 的个人记忆需求，把成本、透明度、缓存友好做到极致；向量库、图谱、用户建模等重能力通过单一 Provider 插槽获得。它是「记忆层不必重，但边界必须清晰」的范本。

## 四、Mem0：生产级事实抽取与多信号检索

Mem0（arXiv:2504.19413）定位是「LLM agents 的 memory layer」，提供开源 Python 包与托管平台双轨，是 2025 年后产品化记忆最常被直接集成的方案。

### 4.1 双入口与 CRUD 表面

- 开源自建：`from mem0 import Memory`；
- 托管平台：`from mem0 import MemoryClient`。

核心 API 是一套显式 CRUD：`add`、`search`、`get_all`、`get`、`update`、`delete`、`delete_all`。显式 CRUD 的意义是让记忆成为可被业务代码直接治理的对象——可以后台批量导出、按法规删除、给运营台展示，而不是藏在 Agent 黑箱里的隐式状态。

### 4.2 存「抽取后的事实」而非逐字轨迹

Mem0 默认存的是从对话中**抽取后的 memories**（原子化事实，如「用户对花生过敏」「项目使用 pnpm 而非 npm」），而不是逐字 transcript。需要保存原文时可在 `add(..., infer=False)` 显式指定。范围通过 `user_id` / `agent_id` / `run_id` 与 `metadata`（category、app、自定义字段）表达，与第 2.3 节的主键模型一致。

### 4.3 2026 新算法：ADD-only、实体链接与多信号检索

根据其官方 README 与文档（2026-10-26 实测），当前算法有四个要点：

1. **Single-pass ADD-only extraction**：单趟扫描、只做新增式抽取，降低写路径延迟与 token 消耗，把矛盾消解更多放到检索/更新阶段；
2. **Agent-generated facts are first-class**：Agent 自己声明的事实与对话中隐式抽取出的事实同等入库，鼓励 Agent 主动 `add`；
3. **Entity linking**：记忆条目与实体挂钩，同一实体的多条事实可被聚合，而不只是孤立的文本块；
4. **Multi-signal retrieval**：检索同时跑**语义向量 + BM25 关键词 + 实体匹配**，并带**时序推理**（解析「上周」「在那之后」之类的时间约束）。

README 给出的基准口径包括：LoCoMo 92.5（约 7.0K tokens、p50 0.88s）、LongMemEval 94.4（约 6.8K tokens、p50 1.09s）、BEAM 1M 64.1、BEAM 10M 48.6。⚠️ **这些数字明确标注来自 Managed Platform，包含专有优化；开源自建只能期待架构方向相似，不能外推同等分数与延迟。** 报告引用此类厂商数字时必须保留这层限定。

### 4.4 OSS 与 Managed 的真实差距

集成 Mem0 时应按两条曲线规划：

- **开源版**：自带抽取 prompt、可插拔 LLM/embedder、多种向量后端，适合内网与数据不出域场景，但抽取策略、融合权重、图谱化程度是基础版，效果依赖自备模型质量；
- **托管版**：开箱即用的多信号融合、实体与时序推理、账号体系与可观测性，按用量计费，代价是数据出域与供应商锁定。

**Mem0 小结**：它把「抽取-存储-检索」产品化成了一个薄薄的客户端调用，是「想让产品明天就有跨会话记忆」的最短路径；但要清醒地区分 OSS 能力与平台口径的跑分，把它当作**可替换的记忆服务**而非黑箱魔法。

## 五、Codex：Rollout Extraction → Global Consolidation 两阶段流水线

OpenAI Codex 的记忆系统是「编码 Agent 如何从自己的 rollout 中学习」的一手样本。其源码（2026-10-26 拉取自 `github.com/openai/codex`）将可复用逻辑拆为两个 crate，运行时编排留在 `codex-core`：

- `codex-rs/memories/read`（`codex-memories-read`）：读路径——记忆型 developer instruction 注入、记忆引用解析、读取用量遥测；
- `codex-rs/memories/write`（`codex-memories-write`）：写路径——两阶段 prompt 渲染、文件系统产物、workspace diff、扩展资源剪枝。

### 5.1 触发条件：root、非临时、非子 Agent

据 `codex-rs/memories/README.md`，流水线在 root session 启动时异步触发，且必须同时满足：会话**非 ephemeral**、记忆特性开启、**不是 sub-agent session**、state DB 可用。这四个条件划出了一条清晰边界：只从「真实、持久、顶层」的工作经历中学习，避免临时会话噪声与子 Agent 递归写记忆。

### 5.2 Phase 1：Per-thread 的 Rollout 抽取

Phase 1 从 state DB 中选取近期合格 rollout，逐条抽取结构化记忆。合格规则（startup claim rules）要求 rollout：

- 来自允许的交互式会话来源；
- 在配置的年龄窗口内；
- 已空闲足够久（避免总结仍在进行的新鲜 rollout）；
- 未被其他在途 worker 占用；
- 在启动扫描/claim 上限内（每次启动的工作量有界）。

执行流程：claim 一组有界任务 → 过滤出与记忆相关的 response items → **并行调用模型（有并发上限）** → 期望结构化输出三字段：

- `raw_memory`：详细的原始记忆；
- `rollout_summary`：紧凑摘要；
- 可选 `rollout_slug`：稳定标识。

生成结果经**密钥脱敏（secrets redaction）**后写回 state DB。任务三态收口：`succeeded`（产出记忆）、`succeeded_no_output`（有效但无值得记的内容）、`failed`（带退避重试，防止空转 hot-loop）。lease/claim 机制保证多个 worker、多次启动不会重复处理同一 rollout。

### 5.3 Phase 2：全局锁 + Git 基线 + 专职 consolidation Agent

Phase 2 把散落的 stage-1 产物巩固为文件系统记忆，是整套设计最有工程含量的部分：

1. **单全局锁**：触碰 memories root 前先 claim 一个全局 phase-2 lock，任何时刻只有一个 consolidation 在检查或修改工作区；
2. **Top-N 选择**：从 state DB 按规则挑选 stage-1 输出——`last_usage` 超出 `max_unused_days` 窗口的丢弃；从未使用过的回退按 `generated_at` 取（保证新鲜记忆有入选机会）；排序先看 `usage_count` 再看最近 `last_usage/generated_at`。**用「被读路径引用的次数」给记忆投票，没人用过的旧记忆自然淘汰**；
3. **产物同步**：`raw_memories.md`（按 thread-id 稳定升序合并）与 `rollout_summaries/`（每个 rollout 一个摘要文件），并剪枝不再入选的摘要与超过保留窗口的扩展资源文件——剪枝也进 diff；
4. **Git 基线**：memories root 本身是 git 仓库（`~/.codex/memories/.git`），每次成功巩固后重置基线；
5. **生成 workspace diff**：把上次成功基线到当前工作区的 git 风格差异写入 `phase2_workspace_diff.md`；无变化则任务直接成功退出；
6. **专职 consolidation sub-agent**：有变化时启动内部子 Agent，喂给它 diff 路径，并以严苛沙箱运行——**无审批、无网络、仅本地写权限、禁用 collab（防止递归 delegation）**；父进程在其运行期间持续 heartbeat 全局租约；
7. **成功后重置基线并删除 diff 文件**——删除发生在 reset 之前，保证被删内容既不留在 prompt 产物里，也不作为不可达 git 对象残留。

### 5.4 设计解读：把记忆当「带审计的异步编译」

Codex 的选择与 Hermes 形成鲜明对照：

- Hermes 的记忆是**同步写、下一轮生效、模型亲手写**的小文件；
- Codex 的记忆是**异步、两级缓冲（DB → 文件）、后台模型抽取、子 Agent 巩固**的编译流水线。

它解决了三个规模化问题：高频交互下写路径不能阻塞主循环（异步 + 并发）、多窗口/多启动并发写冲突（lease + 全局锁）、长期运行后记忆腐化（按使用度淘汰 + 每次巩固可 diff、可回滚的 Git 基线）。「diff 驱动 consolidation」尤其值得借鉴：巩固 Agent 只处理变化量而非全量记忆，token 成本可控，且每一次记忆变更都有版本化审计。

<figure>
<iframe src="diagrams/2-hermes-codex-pipeline.html"></iframe>
<figcaption>图 2：Hermes 冻结快照式记忆 vs Codex 两阶段抽取-巩固流水线对比</figcaption>
</figure>

## 六、商业级与编码 Agent 记忆模块横评

### 6.1 Claude Code：CLAUDE.md/AGENTS.md + Auto Memory

Claude Code 的记忆分两层。第一层是**手动规则文件** `CLAUDE.md` / `AGENTS.md`（项目级、用户级），随会话加载；第二层是 **Auto Memory**，自动生成四类记忆：

- `user`：用户的长期偏好与背景；
- `feedback`：用户对具体做法的正负反馈（「别再用 X」「提交前先跑测试」）；
- `project`：本项目的约定与结构事实；
- `reference`：外部资源指针（文档地址、内部 wiki）。

存储于 `~/.claude/projects/<project>/memory/`，其中 `MEMORY.md` 是**索引文件**，topic files 按需读取；启动时只读索引（前 200 行或 25KB），具体话题不默认加载——这是控制注入成本的标准手法。系统明确会**跳过可直接从代码或 git 历史推导的信息**，避免记忆与仓库重复。记忆为 **machine-local**，不跨机器、不上云。

### 6.2 Letta（MemGPT）：MemFS 文件系统 + Dreaming 后台巩固

Letta 源自 MemGPT（arXiv:2310.08560），核心理念是把 LLM 当操作系统、把记忆当文件系统：

- **MemFS**：git-backed 的记忆文件系统。root 层文件直接进入 system prompt，带 `MEMORY.md` 的目录按需分页读取，模拟「主存-外存」分层；
- **Dreaming**：空闲时后台 subagents 回顾交互、合并更新记忆，与人类睡眠期记忆巩固类比——这正是 consolidation 环节的产品化；
- 编码场景由 Letta Code harness 承载；云记忆按仓库配额计费（Free 5 MiB/repo，Pro 50 MiB/repo）。

### 6.3 Devin / Windsurf Cascade：Memories、Rules 与 Skills 的三分

Devin Desktop（Cascade）把三类知识刻意分开：

- **Memories**：自动生成、可在设置中管理，本地位于 `~/.codeium/windsurf/memories/`；
- **Rules**：手动规则——global `global_rules.md`（约 6000 chars），workspace 级 `.devin/rules/*.md`（每文件约 12000 chars；旧路径 `.windsurf/rules/` 仍兼容）；`AGENTS.md` 走同一规则引擎，root 文件始终生效、子目录按 glob 自动加载；
- **Skills**：值得注意的产品转向——**新版 Devin Local agent 不再持久化 memories，官方建议把需要长期保留的内容迁移为 Skills**。

最后一点是一个行业信号：当「记忆」与「能力/程序记忆」需要取舍时，把可复用的做法沉淀为显式、可版本管理、可分发的 Skills，比让自动记忆默默膨胀更可控——程序记忆正在从记忆层独立出来。

### 6.4 记忆即服务：Honcho、Supermemory、Zep

- **Honcho**：开源 + 云双轨，自我定位「open-source memory for AI agents」，支持 self-host。其建模单元是 workspace 中的 **peers**：一个 user peer 与多个 AI peer，每个 AI peer 对同一用户独立建模（coder profile 与 writer profile 看到的是不同侧面），并可控制 observation 方向。这种「多 Agent 对同一用户分别画像」是多 Agent 产品的有用原语。
- **Supermemory**：Memory / continual learning API，提供 plugin 与 MCP 接入。概念管线为 raw data → learner model → vector-graph DB → context injection。⚠️ 其 v3/v4 API 已弃用、**2026-12-31 停服**，新集成应直接使用 v5——商业记忆 API 的版本寿命是选型时必须计入的迁移成本。
- **Zep / Graphiti**：Zep（arXiv:2501.13956）是**时序知识图谱架构**，把记忆组织为带有效时间的实体-关系网络，支持事实失效、矛盾处理与时间推理（Graphiti 为其开源图谱引擎）。适合「用户状态持续变化、历史必须可追溯」的 CRM、医疗、合规场景。

### 6.5 横向对比总表

| 系统 | 存储形态 | 写入方式 | 检索方式 | 时序/失效 | 审计/版本 | 跨机器 | 程序记忆 | 运维成本 | 最适场景 |
|---|---|---|---|---|---|---|---|---|---|
| **Hermes** | 2 个 MD + SQLite FTS | Agent 工具同步写 | 启动冻结注入 + FTS 原文 | 靠手动替换 | 文件可 diff（弱） | 否 | MEMORY 笔记 | 极低 | 个人本地 Agent |
| **Mem0 OSS** | 向量库 + 事实条目 | `add`/异步抽取 | 向量+BM25+实体 | 部分支持 | CRUD 可导出 | 自建可 | 弱 | 中 | 内网产品化 |
| **Mem0 Cloud** | 托管多索引 | 单趟 ADD-only | 多信号融合+时序 | 支持 | 平台侧 | ✔ | 弱 | 低（付费） | 快速上线 |
| **Codex** | Git 文件 + state DB | 后台两阶段 | 引用驱动注入 | usage/TTL | ✔ Git 基线+diff | 否（本地） | raw/summary | 中高 | 编码 Agent |
| **Claude Code** | MD 索引+话题文件 | Auto 四类 | 索引+按需读 | 手动管理 | 本地文件 | 否 | feedback 类 | 极低 | 编码助手 |
| **Letta** | MemFS（Git）+状态 | 工具写+Dreaming | 分层文件读取 | 支持 | ✔ Git | 云配额 | 自改 core | 中高 | 长生命周期 Agent |
| **Devin/Windsurf** | memories + rules | 自动+手动 | 规则引擎/Glob | rules 覆盖 | 文件 | 部分 | Skills | 低 | 编码/团队规范 |
| **Honcho** | 云/自建 peer 模型 | 观察式建模 | per-peer 画像 | 支持 | 平台/自管 | ✔ | 弱 | 中 | 多 Agent 用户建模 |
| **Supermemory** | vector-graph 服务 | learner pipeline | 注入式 API | 支持 | 平台侧 | ✔ | 弱 | 低（付费） | 插件/MCP 接入 |
| **Zep/Graphiti** | 时序知识图谱 | 抽取入图 | 图遍历+时间 | ✔ 双时态 | ✔ 可追溯 | ✔ | 弱 | 中高 | CRM/合规/医疗 |

<figure>
<iframe src="diagrams/3-vendor-selection.html"></iframe>
<figcaption>图 3：记忆系统定位地图（存储形态 × 巩固方式）与按场景的选型路径</figcaption>
</figure>

## 七、参考架构：一个生产级 Agent Memory Service

综合上述系统，给出一个可裁剪的生产级参考架构。原则：**写路径异步、读路径同步、所有写入可追溯、scope 贯穿全链路**。

```text
                         ┌──────────────────────── Policy / Privacy ───────────────────────┐
                         │ 凭据脱敏 · 注入检测 · scope ACL · 删除权/TTL · 审计日志          │
                         └──────────────────────────────────┬───────────────────────────────┘
 Agent runtime ──turn──▶ Ingest Gateway ──▶ Raw Store(S3/对象存储, 不可变原文 + 来源指针)
  (tool calls,            (鉴权/限流/         │
   artifacts)              scope 注入)        ▼
                                      Extractor Workers (队列, 并发有界, 幂等 claim)
                                   事实/事件/偏好/程序 四类候选 + 置信度 + 证据指针
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              ▼                             ▼                             ▼
        Fact/Semantic Store          Episodic Timeline             Procedural Library
        (Postgres + 实体表)          (bitemporal: valid/asserted)  (Skills / Prompt 模板, 走 Git)
              │                             │                             │
              ▼                             ▼                             ▼
        Vector Index ──┐         BM25/FTS ──┤                    版本发布 / 评审门禁
        Graph Store  ──┴──▶ Retrieval Fusion (RRF/rerank, 时间+scope 硬过滤) ──▶ top-k + 引用
                                                                              │
                                              Context Assembler ◀────────────┘
                                   (按位置: 用户/项目/事件/技能分块, 预算分配, 引用编号)
                                              │
                                              ▼
                                         Agent Prompt

   后台: Consolidation Jobs ─ 合并/去重/矛盾消解 (diff 驱动, 产出 PR 式变更, 可回滚)
   控制台: Memory Console ─ 查看/编辑/删除/导出, 用户可见可改 (记忆透明度)
```

关键设计决策逐条说明：

1. **Raw Store 不可变、与派生记忆分离**：任何抽取结果都带证据指针指回原文，错误抽取可溯源、可重放（Hermes 的 FTS 原文、Codex 的 raw_memories 是同一思想的轻量版）。
2. **Extractor 走队列、幂等 claim**：对齐 Codex Phase 1 的 lease 模型，worker 扩缩容不产生重复抽取；输出按四类记忆分流而非一张大表。
3. **多写索引 + 融合检索**：向量管语义、BM25 管精确词、图管多跳关系、bitemporal 管时间；检索时 scope 与时间是**硬过滤**，相关性只决定排序。
4. **Context Assembler 管预算与引用**：按记忆类型分块、分配 token 预算，每条注入带编号引用——读路径遥测（哪条记忆被实际用到）回流为 consolidation 的淘汰信号，闭环对齐 Codex 的 `usage_count`。
5. **Consolidation 产出「PR」而非直接改库**：diff 驱动、评审或门禁通过后提交，Git 式版本历史；程序记忆（Skills）走独立发布流。
6. **Memory Console 是合规组件不是赠品**：用户能查看、修改、删除、导出关于自己的记忆，这是删除权与记忆中毒后人工干预的唯一现实通道。

最小可行裁剪：个人/小团队可把 Raw Store 换成 SQLite、去掉 Graph、用 Markdown + Git 承载事实与 Skills（≈ Hermes + Claude Code 形态）；面向多用户 SaaS 时再把 Extractor、索引、Console 服务化（≈ Mem0 Cloud / Honcho 形态）；强时序行业最后叠加 bitemporal 图谱（≈ Zep 形态）。

## 八、选型建议：按诉求直达

| 你的诉求 | 推荐方案 | 理由 / 注意 |
|---|---|---|
| 本地 Coding Agent，零成本起步 | Hermes 内建 / Claude Code 式 MD 记忆 | 冻结快照保缓存、透明可手改；上限小但够用 |
| 给现有 Agent 加跨会话记忆，最快上线 | Mem0 Cloud / Supermemory | 一个客户端 + API key；先确认数据出域合规，锁定 API 版本 |
| 数据必须内网、不出域 | Mem0 OSS 自建 / Honcho self-host | 抽取与 embedder 全部可换私有模型；预留运维 |
| 用户状态频繁变化、需追溯历史 | Zep / Graphiti 时序图谱 | 双时态支持失效与矛盾；图谱维护成本高 |
| 多个 Agent 服务同一用户 | Honcho per-peer 建模 | 各 Agent 独立画像，避免互相污染 |
| Agent 需长期自主、从经验复利 | Letta MemFS + Dreaming / 自研巩固 | 重架构，先确认巩固质量可观测 |
| 团队编码规范沉淀 | Devin Rules + AGENTS.md + Skills | 优先程序化、可 Git 管理的知识，少依赖自动记忆 |
| 从 rollout 中自动学习（编码/工具型） | 借鉴 Codex 两阶段 | lease 抽取 + 全局锁 + diff 巩固，沙箱隔离子 Agent |

一条总原则：**记忆的默认形态应该尽量轻、尽量显式、尽量可删；重型记忆（图谱、自主巩固）只在确有其时序或自治需求时引入。** 记忆系统最常见的失败不是记不住，而是记住了错误、过时、不该记的东西且无法清理。

## 九、风险与开放问题

1. **错误抽取累积（Extraction Drift）**：一次误抽会在后续每轮被反复注入，甚至被 consolidation 二次确认形成「事实」闭环。对策：证据指针、置信度、来源冲突检测、定期重抽。
2. **记忆中毒与提示注入（Memory Poisoning）**：外部内容（网页、邮件、issue）中夹带的指令一旦被存入记忆，就从「数据」升级为每次启动的「指令」，攻击面显著放大。对策：写入侧注入扫描（Hermes 内建）、记忆内容以数据角色注入并加隔离标记、高权限记忆只接受可信来源。
3. **凭据与敏感信息泄漏**：轨迹里天然包含密钥、token、个人身份信息。Codex 在 Phase 1 强制 secrets redaction；自建系统应把脱敏放在抽取之前、并对 Raw Store 加密与限权。
4. **多租户范围泄漏**：见 2.3，scope 必须是硬过滤；每次新索引、新缓存通道上线都要重审 scope 是否随检索传递。
5. **Stale Context 与行为固化**：过时记忆让 Agent 抱着旧习惯工作；删除权之外需要 TTL、矛盾检测与「新证据优先」的注入排序。
6. **评测不可比**：LoCoMo、LongMemEval、BEAM 等基准各有侧重，厂商分数多来自带专有优化的托管版本。开源对比应在同一模型、同一 embedding、同一硬件下自测，勿直接引用厂商口径。
7. **程序记忆的归属未收敛**：Devin 把长期知识导向 Skills、Letta 允许自改 core、Claude Code 用 feedback 类记忆——「教训」到底进记忆、进技能还是进训练数据，行业尚无定论，是值得持续跟踪的开放方向。

## 十、参考资料

### 一手源码与官方文档（2026-10-26 实测）

1. Hermes Agent（Nous Research）本地源码：`~/.hermes/hermes-agent`——`agent/memory_manager.py`、`agent/memory_provider.py`、`tools/memory_tool.py`、`agent/agent_init.py`
2. Hermes 记忆文档：`website/docs/user-guide/features/memory.md`、`memory-providers.md`、`developer-guide/memory-provider-plugin.md`
3. OpenAI Codex 源码：https://github.com/openai/codex —— `codex-rs/memories/README.md`、`codex-rs/memories/read|write/`、`codex-rs/core/src/memories/`
4. Mem0 GitHub：https://github.com/mem0ai/mem0（README 实测）；文档：https://docs.mem0.ai/llms.txt、`/core-concepts/memory-types.md`
5. Claude Code Memory 文档：https://code.claude.com/docs/en/memory
6. Devin Desktop Cascade Memories：https://docs.devin.ai/desktop/cascade/memories.md
7. Letta：https://github.com/letta-ai/letta 与 Letta Code https://github.com/letta-ai/letta-code
8. Honcho 文档：https://docs.honcho.dev/v3/
9. Supermemory 文档：https://docs.supermemory.ai/

### 论文（arXiv 编号经 arXiv API 逐一核实）

10. A Survey on the Memory Mechanism of LLM based Agents, 2024 — arXiv:2404.13501
11. MemGPT: Towards LLMs as Operating Systems, 2023 — arXiv:2310.08560
12. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory, 2025 — arXiv:2504.19413
13. A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts, 2024 — arXiv:2402.09727
14. Human-inspired Episodic Memory for Infinite Context LLMs (EM-LLM), 2024 — arXiv:2407.09450
15. HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs, 2024 — arXiv:2405.14831
16. Generative Agents: Interactive Simulacra of Human Behavior, 2023 — arXiv:2304.03442
17. Zep: A Temporal Knowledge Graph Architecture for Agent Memory, 2025 — arXiv:2501.13956
18. MemOS: An Operating System for Memory-Augmented Generation, 2025 — arXiv:2505.22101
19. A-MEM: Agentic Memory for LLM Agents, 2025 — arXiv:2502.12110

### 延伸阅读（知识库内相关报告）

20. 《智能体记忆（Agent Memory）前沿技术报告》，knowledge-base `models/agent-memory/`，2026-10-13 —— 偏学术谱系与评测，可与本文的系统实现视角互补
21. 《智能体互联协议栈 MCP/A2A/AGENTS.md》，`models/agent-interop/`，2026-10-16
22. 《Agentic RL 智能体强化学习》，`models/agentic-rl/`，2026-10-10 —— 记忆之外的另一条「经验内化」路线（更新权重）

---

*本报告基于 2026-10-26 的开源仓库快照与已核实论文撰写；商业产品功能与 API 版本变化快，引用前请重新核对。*
