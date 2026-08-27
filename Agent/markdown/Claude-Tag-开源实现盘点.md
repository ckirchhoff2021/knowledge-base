# Claude Tag 开源实现盘点

> 整理日期：2026-08-21（所有 GitHub 数据均为当日实测）
> 关联报告：`Claude-Tag-技术调研报告.md`（同目录，2026-08-18）

## 1. 背景：Claude Tag 与开源复刻潮

Claude Tag 是 Anthropic 于 2026-06-23 发布的"常驻 AI 同事"：住在 Slack 频道里、
@ 即委派任务、持续学习组织上下文、异步自主执行。但它是**闭源、付费
（Enterprise/Team 订阅）、仅支持 Claude、仅 Slack、云托管**的。

发布后迅速催生了一批开源复刻项目，核心诉求一致：**自托管、不锁模型、数据不出内网**。

## 2. Zilliz Open Tag 的去向（上次报告遗留问题）

上次调研报告中提到的 Zilliz "Open Tag"（两天手搓的开源版 Claude Tag），
知乎原文中 GitHub 地址被截断。本次实测查证结果：

| URL | 状态 |
|---|---|
| github.com/zilliztech/open-tag | 404 |
| github.com/zilliztech/open-claude-tag | 404 |
| github.com/zilliztech/opentag | 404 |
| github.com/zilliztech/mfs | ✅ 存活（121★，最后推送 2026-07-31） |

**结论（推断，无官方公告佐证）**：Open Tag 没有公开删除记录，更可能是被收编——
它本就是为展示 MFS 做的传播向 demo，使命完成后下线，团队聚焦底层引擎
**MFS（Multi-source File-like Search）**：一个"上下文挽具"（context harness），
把代码、记忆、技能、文档、消息等所有数据源统一成 AI agent 的工作区，
基于 Milvus 检索，提供 Rust CLI（crates.io: mfs-cli）+ Python SDK（≥3.10）。
想要"Zilliz 味"的技术沉淀，看 MFS。

## 3. 现存实现总览（按星标排序）

| 项目 | Stars | 许可证 | 定位一句话 | 通道 | 模型 |
|---|---|---|---|---|---|
| paradigmxyz/centaur | 1167 | — | "Claude Tag, but open source and on steroids"，K8s 沙箱执行 | Slack | — |
| CopilotKit/OpenTag | 1107 | MIT | Channels SDK 起步应用，含生成式 UI | Slack + Teams | AG-UI agent |
| linxidnju/OpenTag | 501 | Apache-2.0 | 通道原生 multiplayer AI 网关，审计+审批 | Slack | 可插拔 runtime |
| openma-ai/open-managed-agents | 243 | Apache-2.0 | Claude Managed Agents API 的开源实现（底座） | API | Claude 兼容 |
| fancyboi999/open-tag | 164 | — | 自托管 Slack 风格工作区，多 agent 引擎 | 自有工作区 | Claude Code/Codex/Copilot |
| AgentBull/ankole | 50 | — | "AI Workforce OS"，agent 当自主劳动力 | — | — |
| zilliztech/mfs | 121 | — | 上下文/记忆引擎（非完整 Tag，是底座） | — | — |
| anthropics/claude-tag-plugins | 43 | — | Anthropic 官方插件仓库 | Slack | Claude |
| duyet/oma | 5 | — | Open Managed Agents，drop-in 兼容 | — | — |
| liliang-cn/tagit | 4 | — | 自托管 Claude Tag | — | — |

### 3.1 paradigmxyz/centaur（推荐首选）

https://github.com/paradigmxyz/centaur

paradigm 出品的"团队共享 AI agent"基础设施。核心特性：

- **Slack 原生会话**：@ 机器人，进度与最终答案回到 thread 里
- **真实执行环境**：每个会话跑在隔离的 Kubernetes 沙箱中，内置
  shell、workspace、git、Python、Node.js、Bun 等开发工具
- 本地部署用轻量 k3s 集群即可，不需要完整生产 K8s

星标最多、更新最活跃（实测当日仍在推送），架构上最接近 Claude Tag 的
"Channel Session + 沙箱执行"模式。

### 3.2 CopilotKit/OpenTag

https://github.com/CopilotKit/OpenTag

CopilotKit 团队的 **Channels SDK 官方起步应用**——"把散件组装成你真正会部署的东西"。

- 支持 Slack 和 **Microsoft Teams**（少有的双通道）
- 演示场景：扔进去一个电子表格 → 回来原生 Slack 图表、带审批门的
  Linear issue、带引用的调研简报，全部在 thread 内交付
- 含**生成式 UI**（不只是文本回复）
- 配套 Channels SDK（github.com/CopilotKit/channels-sdk）可把任何 AG-UI
  agent 接进 Slack/Teams

### 3.3 linxidnju/OpenTag

https://github.com/linxidnju/OpenTag

"通道原生 multiplayer AI 网关，共享、可审计的团队知识"。

- Slack-first：@OpenTag 进线程，团队在同一线程讨论任务
- **本地执行**、进度可见、**危险操作需审批**、完整审计轨迹
- agent runtime 可插拔（Node ≥20.11.0），自我标注 MVP 状态
- 文档齐全：User Guide / SECURITY.md / CONTRIBUTING.md

安全治理特性（审批+审计）是它区别于其他项目的卖点。

### 3.4 openma-ai/open-managed-agents（底座型）

https://github.com/openma-ai/open-managed-agents · https://openma.dev

**Claude Managed Agents API 的开源实现**，是搭 Claude Tag 式 agent 的地基
而非成品：

- Anthropic API 兼容（可换端点直接接入现有 Claude 工作流）
- TypeScript 5.8，Cloudflare Workers 部署
- 定位"自托管 Claude Tag 式 agent 的 foundation"

适合想从 API 层自己控制一切的团队；同类的还有 duyet/oma（drop-in 兼容）。

### 3.5 fancyboi999/open-tag（不锁模型）

https://github.com/fancyboi999/open-tag

"人机一体的开源工作区"——Claude Tag 的自托管替代：

- **不锁 Claude**：同时接 Claude Code、Codex、GitHub Copilot
- Slack 风格协作层：频道共享上下文、委派真实任务、看实时进度
- agent 的记忆与工作区全部留在**你自己控制的基础设施**上

对已有多个 coding agent 工具的团队最实用。

### 3.6 其他

- **AgentBull/ankole**（50★）：开源 AI Workforce OS，把 agent 组织成自主劳动力执行业务职能
- **anthropics/claude-tag-plugins**（43★）：Anthropic 官方插件仓库，配合官方 Claude Tag 使用
- **DanielLi202/hermes-tag**（3★）：基于 Hermes 的小实现，@ 后拉取上下文执行
- **Octember/earshot**（3★）：homebrew Claude Tag + reference spec

## 4. 飞书 / Lark 生态（重点）

Claude Tag 官方仅支持 Slack，飞书用户需要移植版：

### 4.1 aws-samples/sample-claude-tag-in-lark（官方示例，最规范）

https://github.com/aws-samples/sample-claude-tag-in-lark

AWS 官方出品的**飞书版 Claude Tag**：

- 基于 **Amazon Bedrock AgentCore + Claude Agent SDK**——
  **不需要 Claude Enterprise 订阅**
- 模型后端三选一（`MODEL_BACKEND`）：LiteLLM 网关（默认）/
  Bedrock Invoke API 直连 / Bedrock Mantle 端点
- 核心能力：
  1. @ 委派：群里 @bot，自动拆任务、调工具/技能，CardKit 实时流式更新
     （显示当前在跑哪个工具）
  2. **每频道自进化记忆**：主动 curate 自己的长期记忆 + 技能循环
- ⚠️ AWS 声明：示例代码，勿直接用于生产账号

### 4.2 jiangdaxia-AI/open_claude_tag_lark（功能最完整）

https://github.com/jiangdaxia-AI/open_claude_tag_lark

"飞书群里的 AI 数字员工团队"：

- **一个群 = 一支团队**：产品、研发、调度等角色各司其职
- 一句话触发：自动拆任务、排队执行、交付文档
- 共享记忆，永不忘事
- Python 3.11+，飞书原生，MIT 许可
- ⚠️ 个人项目（1★），生产前必须通读代码，重点审查凭据存储与任务状态持久化

### 4.3 RCliang/open-claude-tag-feishu

https://github.com/RCliang/open-claude-tag-feishu

频道原生的共享 AI 同事 + 飞书集成版（0★，更早期，仅参考）。

## 5. 选型建议

| 你的诉求 | 推荐 |
|---|---|
| 最完整的 Claude Tag 复刻 | **paradigmxyz/centaur**（K8s 沙箱最成熟） |
| 多 agent 引擎共存、自托管工作区 | **fancyboi999/open-tag** |
| 需要 Teams 支持 / 生成式 UI | **CopilotKit/OpenTag** |
| 重安全治理（审批+审计） | **linxidnju/OpenTag** |
| 要 API 兼容底座，自己搭上层 | **openma-ai/open-managed-agents** |
| 飞书落地 | **aws-samples/sample-claude-tag-in-lark**（规范）或 jiangdaxia-AI 版（功能多需审代码） |
| 上下文/记忆引擎单独集成 | **zilliztech/mfs** |

## 6. 风险与注意事项

1. **凭据安全是第一风险**：Claude Tag 式 agent 需要大量组织凭据
   （Slack/飞书 token、代码仓库、数据库）。Claude Tag 官方用
   **Agent Proxy 凭据隔离**（凭据不进沙箱，只在出站边界注入），
   多数开源项目没有这层——**部署前务必审查每个项目的凭据流向**。
2. **低星项目质量无保障**：100★ 以下多为个人作品，无 CI、无审计，
   生产使用前需完整代码审查。
3. **生态变动快**：本盘点为 2026-08-21 快照，Open Tag 类项目更新/下线频繁
   （Zilliz Open Tag 即为先例），引用前先验证仓库存活。
4. **合规**：Claude Tag 是 Anthropic 产品名，开源复刻多为独立实现，
   商用时注意商标与许可条款。

## 7. 信息来源

- GitHub REST API 搜索（query: open-claude-tag / "claude tag" / open-tag），2026-08-21
- 各项目 README（raw.githubusercontent.com 抓取），2026-08-21
- 前序调研：`Claude-Tag-技术调研报告.md`（Anthropic 官方公告已交叉验证）
