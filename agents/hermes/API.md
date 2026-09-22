---
tags: [AI/Agent, Hermes, 工具, 远程访问, OpenAI-API]
created: 2026-08-24
source: Hermes 源码 gateway/platforms/api_server.py + 官方文档 + 实践验证
---

# 本地访问远端 Hermes Agent 完整指南

> **一句话总结**：想让本地代码/IDE 调远端 **Hermes Agent 完整能力**（含工具调用、skills、记忆、会话持久化），开启 `API_SERVER_ENABLED=true` 后 `hermes gateway start`，本地用 OpenAI SDK 指过去就行；如果只想调模型本身，用 `hermes proxy`。
>
> 核心坑：`hermes proxy` 只透传 LLM，**不跑 Agent 循环**（无 tools/memory/skills）；`hermes serve` 是桌面 app 专用的 WebSocket/JSON-RPC 协议，不适合代码调用；真正给代码/前端用的是 **API Server adapter**（OpenAI 兼容）。

相关文档：

---

## 1. 四种"访问 Hermes"方式的区别

很多人搞不清 `proxy` / `serve` / `dashboard` / `api_server` 的区别，一张表讲清楚：

| 命令/方式 | 暴露的能力 | LLM 透传 | 工具/skills | 长期记忆 | 会话持久 | 协议 | 典型场景 |
|-----------|----------|---------|------------|---------|---------|------|---------|
| `hermes proxy start` | 纯模型代理 | ✅ | ❌ | ❌ | ❌ | OpenAI `/v1/chat/completions` | 本地 IDE/脚本用远端 OAuth 模型 |
| **`API_SERVER_ENABLED=true` + `hermes gateway run`** | **完整 Agent** | ✅ | **✅** | **✅** | **✅** | OpenAI `/v1/chat/completions` + `/api/sessions` + `/v1/runs` SSE | **代码调用远端 Hermes Agent（本文重点）** |
| `hermes serve` | Headless 后端 | ✅ | ✅ | ✅ | ✅ | JSON-RPC over WebSocket（私有协议） | Hermes Desktop / IDE 插件连接 |
| `hermes dashboard` | Web 管理面板 + 内嵌聊天 | ✅ | ✅ | ✅ | ✅ | 浏览器页面 + 内部 API | 浏览器里管理/聊天 |

> 只要你的需求是"写代码让远端 Hermes 干活（能调 terminal、读写文件、用 skills、记东西）"，必须用 **API Server**，不是 proxy、不是 serve、不是 dashboard。

---

## 2. 远端配置：启用 API Server

### 2.1 环境变量

编辑远端的 `~/.hermes/.env`，加入：

```bash
# 启用 OpenAI 兼容 API Server
API_SERVER_ENABLED=true

# 必填：Bearer token 认证。没有这个 server 会直接拒绝启动（安全审计强制）
# 即使绑定 127.0.0.1 也必须设 key。
API_SERVER_KEY=***$(openssl rand -hex 32)

# 端口，默认 8642
API_SERVER_PORT=8642

# 绑定地址：
# - 只用 SSH 隧道 → 127.0.0.1（推荐，最安全）
# - 内网多机直连 → 0.0.0.0 或内网 IP
# - 不建议直接公网裸奔，至少套 HTTPS
API_SERVER_HOST=127.0.0.1

# 可选：对外宣告的模型名，默认是 profile 名（默认 profile 就是 "hermes-agent"）
# API_SERVER_MODEL_NAME=hermes-agent
```

关键强制规则（源码 `security_audit_startup.py` 里写死）：

- 启用 API Server **必须**设 `API_SERVER_KEY`，否则 server 启动时拒绝，并有明确错误提示："terminal-capable agent execution. Set a strong API_SERVER_KEY."
- 即使 bind 127.0.0.1 也要 key，防止本地其他用户/恶意进程无认证调用
- Key 通过 `Authorization: Bearer ***` 头校验

### 2.2 启动 Gateway

API Server 是 gateway 的一个 platform adapter，**不是独立进程**，启动 gateway 时自动加载：

```bash
# 前台启动（调试用）
hermes gateway run

# 后台常驻（推荐，安装成系统服务）
hermes gateway install   # 注册成 launchd（macOS）/ systemd（Linux）服务
hermes gateway start
hermes gateway status    # 确认运行，日志里应出现 "API server listening on 127.0.0.1:8642"
```

查看日志排查：

```bash
hermes logs -f            # tail -f gateway 日志
# 或直接看日志文件
tail -f ~/.hermes/logs/gateway.log
```

---

## 3. 本地访问：SSH 隧道（推荐）

**不要**把 API Server 直接暴露公网（即使有 key，也是多一层风险）。最稳妥的方式是 SSH 端口转发：

```bash
# 一次性后台隧道（-f 后台、-N 不执行远端命令）
ssh -fN -L 8642:127.0.0.1:8642 your-user@your-server-ip

# 验证连通
curl -s http://127.0.0.1:8642/v1/models \
  -H "Authorization: Bearer sk-你的KEY"
```

> 如果必须公网暴露，前面套 Nginx/Caddy 做 TLS 终结 + IP 白名单，或用 Tailscale/WireGuard 组网。

---

## 4. 代码调用（Python）

### 4.1 最简 stateless 调用（OpenAI SDK）

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8642/v1",
    api_key="sk-你的KEY",
)

resp = client.chat.completions.create(
    model="hermes-agent",       # 或 /v1/models 返回的模型名
    messages=[
        {"role": "user", "content": "帮我列一下当前目录的文件"},
    ],
    stream=True,                # 支持流式
)

for chunk in resp:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

Hermes Agent 会在背后跑完整的工具调用循环（terminal/file/browser/...），最终返回结果。流式输出里你会看到工具调用、tool_result、最终回答依次流出。

### 4.2 带会话持久化（跨调用保持记忆和上下文）

默认每次调用是新会话。要让 Hermes **记住你**（调用 memory/长期记忆、保持多轮上下文），传 `X-Hermes-Session-Id` header：

```python
import uuid
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8642/v1", api_key="sk-你的KEY")
session_id = str(uuid.uuid4())
headers = {"X-Hermes-Session-Id": session_id}

# 第一轮：告诉 Hermes 记住信息
r1 = client.chat.completions.create(
    model="hermes-agent",
    messages=[{"role": "user", "content": "记住我叫小明，主要写 Python"}],
    extra_headers=headers,
)
print(r1.choices[0].message.content)

# 第二轮：用同一个 session_id，Hermes 会记住
r2 = client.chat.completions.create(
    model="hermes-agent",
    messages=[{"role": "user", "content": "我叫什么？我用什么语言？"}],
    extra_headers=headers,
)
print(r2.choices[0].message.content)   # 应能答出"小明"和"Python"
```

`X-Hermes-Session-Key` 头用于控制**长期记忆 scope**（记忆绑定到哪个 key 下），不传则绑定到 session。

### 4.3 REST 方式管理 Sessions（/api/sessions）

```python
import requests, json

BASE = "http://127.0.0.1:8642"
H = {"Authorization": "Bearer sk-你的KEY", "Content-Type": "application/json"}

# 列出现有 sessions
print(requests.get(f"{BASE}/api/sessions", headers=H).json())

# 创建一个空 session
sid = requests.post(f"{BASE}/api/sessions", headers=H, json={}).json()["session_id"]

# 往 session 发消息（非流式）
r = requests.post(
    f"{BASE}/api/sessions/{sid}/chat",
    headers=H,
    json={"message": "现在时间？"},
)
print(r.json())

# 流式
r = requests.post(
    f"{BASE}/api/sessions/{sid}/chat/stream",
    headers=H,
    json={"message": "列一下文件"},
    stream=True,
)
for line in r.iter_lines():
    if line:
        print(line.decode())

# 看历史
print(requests.get(f"{BASE}/api/sessions/{sid}/messages", headers=H).json())

# 删除 session
requests.delete(f"{BASE}/api/sessions/{sid}", headers=H)
```

### 4.4 异步 run 模式（长时间任务 + 审批）

对于可能跑很久的任务，或需要中途人工审批（工具调用危险命令）的场景，用 `/v1/runs`：

```python
import requests, time

BASE = "http://127.0.0.1:8642/v1"
H = {"Authorization": "Bearer sk-你的KEY", "Content-Type": "application/json"}

# 启动 run，立即返回 run_id（HTTP 202）
run = requests.post(f"{BASE}/runs", headers=H, json={
    "model": "hermes-agent",
    "messages": [{"role": "user", "content": "部署一下我的项目"}],
}).json()
run_id = run["id"]

# SSE 订阅事件（tool_call、approval_required、completed、error）
with requests.get(f"{BASE}/runs/{run_id}/events", headers=H, stream=True) as es:
    for line in es.iter_lines():
        if line.startswith(b"data: "):
            evt = json.loads(line[6:])
            print(evt)
            if evt.get("type") == "approval_required":
                # 人工审批
                requests.post(f"{BASE}/runs/{run_id}/approval", headers=H,
                              json={"decision": "once"})
            if evt.get("type") in ("completed", "error"):
                break
```

---

## 5. 常见坑点

### 5.1 用 `hermes proxy` 发现 Hermes 不调工具、不记东西

正常的——`proxy` 只是个裸 LLM 反向代理（给你复用 OAuth 凭据用），**根本不跑 Agent 循环**。想用完整 Agent 必须用 API Server。

### 5.2 启动报 "Set a strong API_SERVER_KEY"

API Server 必须设 key，哪怕 bind 127.0.0.1。设一个就行：

```bash
echo "API_SERVER_KEY=***$(openssl rand -hex 32)" >> ~/.hermes/.env
hermes gateway restart
```

### 5.3 外网能不能直接访问 8642？

可以（`API_SERVER_HOST=0.0.0.0`），但不建议裸奔：
- HTTP 明文，Bearer token 会被窃听
- 没有速率限制
- 一旦 key 泄露，攻击者可以通过 terminal tool 完全控制服务器

推荐做法（按安全性从高到低）：
1. **SSH 隧道**（最简单，零配置安全）
2. **Tailscale/WireGuard** 组网，只在 VPN 内访问
3. **Nginx/Caddy 反代 + TLS + Basic Auth + IP 白名单**

### 5.4 多 profile 怎么访问？

如果开启了 `gateway.multiplex_profiles: true`，子 profile 通过 URL 前缀访问：

```
POST http://127.0.0.1:8642/p/<profile-name>/v1/chat/completions
```

子 profile 可以设不同的 `API_SERVER_PORT` 避免冲突。

### 5.5 本地也想用 Open WebUI/LobeChat 接远端 Hermes

完全可以——任何 OpenAI 兼容前端（Open WebUI、LobeChat、LibreChat、AnythingLLM、NextChat、ChatBox 等）都支持，填：

- **API Base URL**: `http://127.0.0.1:8642/v1`
- **API Key**: 你的 `API_SERVER_KEY`
- **Model**: `hermes-agent`（或 `API_SERVER_MODEL_NAME` 设的值）

### 5.6 想让另一个 Hermes 把工作转发到远端

用 `GATEWAY_PROXY_URL` / `GATEWAY_PROXY_KEY`（proxy 模式）：

```bash
# 本地 hermes 的 .env
GATEWAY_PROXY_URL=http://remote-host:8642
GATEWAY_PROXY_KEY=sk-远端的KEY
```

本地 Hermes 的 gateway 只做平台 IO（Telegram/Discord/...），所有 agent 计算在远端跑。适合 Docker E2EE 容器或瘦客户端场景。

---

## 6. 完整端点列表（远端 Hermes API Server 暴露）

摘自源码 `gateway/platforms/api_server.py` docstring：

| 方法 | 路径 | 用途 |
|------|------|------|
| POST | `/v1/chat/completions` | OpenAI Chat Completions（无状态/有状态 via header） |
| POST | `/v1/responses` | OpenAI Responses API（stateful via `previous_response_id`） |
| GET  | `/v1/responses/{id}` | 获取已存响应 |
| DELETE | `/v1/responses/{id}` | 删除响应 |
| GET  | `/v1/models` | 列出可用模型 |
| GET  | `/v1/capabilities` | 机器可读能力描述 |
| GET  | `/api/sessions` | 列 sessions |
| POST | `/api/sessions` | 新建空 session |
| GET/PATCH/DELETE | `/api/sessions/{sid}` | 读/改/删 session |
| GET  | `/api/sessions/{sid}/messages` | 读消息历史 |
| POST | `/api/sessions/{sid}/fork` | fork 一个 session |
| POST | `/api/sessions/{sid}/chat` | 向 session 发消息 |
| POST | `/api/sessions/{sid}/chat/stream` | 流式版本 |
| POST | `/v1/runs` | 启动异步 run（返回 run_id，202）|
| GET  | `/v1/runs/{rid}` | 查 run 状态 |
| GET  | `/v1/runs/{rid}/events` | SSE 流事件 |
| POST | `/v1/runs/{rid}/approval` | 审批待决操作 |
| POST | `/v1/runs/{rid}/stop` | 中断 run |
| GET  | `/health` | 健康检查 |
| GET  | `/health/detailed` | 详细状态 |

---

## 7. 快速 Checklist

- [ ] 远端 `~/.hermes/.env` 设了 `API_SERVER_ENABLED=true`
- [ ] 设了强 `API_SERVER_KEY`（`openssl rand -hex 32` 生成）
- [ ] `API_SERVER_HOST=127.0.0.1`（SSH 隧道场景）或内网 IP
- [ ] 启动 gateway：`hermes gateway start`
- [ ] `curl /v1/models` 能拿到模型列表（先本地测，再隧道测）
- [ ] 本地 `ssh -fN -L 8642:127.0.0.1:8642 your-server`
- [ ] 代码里 `base_url=http://127.0.0.1:8642/v1`，`api_key` 是 `API_SERVER_KEY`
- [ ] 要记忆就用 `X-Hermes-Session-Id` header 或 `/api/sessions` API

---

## 参考

- 源码：`~/.hermes/hermes-agent/gateway/platforms/api_server.py`
- 安全审计：`~/.hermes/hermes-agent/hermes_cli/security_audit_startup.py`
- 配置默认值：`~/.hermes/hermes-agent/hermes_cli/config_defaults.py`（搜 `API_SERVER_`）
- 官方文档：https://hermes-agent.nousresearch.com/docs/
