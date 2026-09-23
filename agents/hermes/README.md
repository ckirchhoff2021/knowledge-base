# Hermes Agent API 指南

> 本地代码/IDE 如何调用远端 Hermes Agent 的完整能力（工具、skills、记忆、会话持久化）。

## 文档

### [本地访问远端 Hermes Agent 完整指南](API.md)

📄 [Markdown 源文件](API.md) ｜ 📅 2026-08-24 ｜ 来源：Hermes 源码 + 官方文档 + 实践验证

一张表厘清 `proxy` / `serve` / `dashboard` / `api_server` 四种访问方式：`hermes proxy` 只透传 LLM 不跑 Agent 循环；给代码用的是 OpenAI 兼容的 **API Server**（`/v1/chat/completions` + `/api/sessions` + SSE `/v1/runs`）。含配置步骤与常见坑。

`Hermes` `OpenAI 兼容 API` `远程访问` `Agent Gateway`
