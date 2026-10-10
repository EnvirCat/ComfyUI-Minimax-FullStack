# API prompt backend / API 提示词接入

Version / 版本：2.9.0。**API is for prompt/visual design, not hosted image/video generation.** / API 负责文案与可选视觉分析，图片和 H3 视频仍由本地 ComfyUI 生成。

## Setup / 配置入口

Full Stack: **Settings → Prompt model · Local / API**. Image helper: the same named button. / 主面板设置页与生图挂载节点都有同名按钮。

1. Select API, a provider, Base URL and a model ID actually available to your account. Provider names do not grant model access. / 选供应商、地址、账号可用模型 ID，不自动猜测权限。
2. Enter a key and Save. Blank keeps the existing key for that exact provider/endpoint. You may instead specify a key environment-variable name under Advanced; it takes precedence. / 留空保留此接口已有密钥，环境变量优先。
3. **Save & test text connection** performs one small text request, with no image upload or queued video. It can incur a small provider charge. / 测试仅文本，可能有少量费用，不提交生图／视频。
4. Choose the reference-analysis path below, then enable video prompt optimization if you want the API to design video prompts. / 文案模型选择与“启用视频提示词优化”是不同开关。

## Providers & compatibility / 供应商与兼容范围

The transport uses **OpenAI-compatible Chat Completions** (`/chat/completions`), accepting SSE streaming or complete JSON responses. It is not a native Responses, Anthropic Messages, media-generation or arbitrary API adapter.

| Preset / 预设 | Default Base URL | Primary setup reference / 官方说明 |
| --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) |
| Qwen / DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` | [Compatibility and regional endpoints](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope) |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | [OpenAI-compatible endpoint](https://ai.google.dev/gemini-api/docs/openai) |
| Moonshot / Kimi | `https://api.moonshot.ai/v1` | [Kimi developer documentation](https://platform.kimi.ai/docs/overview) |
| OpenRouter | `https://openrouter.ai/api/v1` | [API overview](https://openrouter.ai/docs/api_reference/overview) |
| LM Studio | `http://127.0.0.1:1234/v1` | [OpenAI compatibility](https://lmstudio.ai/docs/developer/openai-compat) |
| Ollama | `http://127.0.0.1:11434/v1` | [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility) |
| Custom / 自定义 | Fill in a compatible endpoint / 自填 | HTTPS required except loopback HTTP / 除回环外必须 HTTPS |

Addresses are editable. DashScope/Kimi keys, regions and workspace-specific addresses must match; use the provider's current documentation rather than assuming one regional key works everywhere. Model IDs are intentionally not hardcoded as universal defaults. DeepSeek is not a preset and its previously supplied key is not bundled or used for this release.

地址可编辑，区域／工作空间与密钥必须配套；模型 ID 按账号权限填写。没有 DeepSeek 预设，本次发布不使用或携带此前测试密钥。

## References / 素材怎么分析

- **API vision off + local visual analysis on**: local GGUF/mmproj examines references; the API receives text observations, authoritative labels and the intent, not the image bytes. A configured local vision model is still required for visual tasks.
- **API vision on**: explicitly authorizes compressed reference images / sampled video frames to be sent to the configured API. Use a genuinely vision-capable model. Image long edge is capped at 768 for prompt analysis; this is not a visual-quality or full-video-understanding guarantee.
- **Both off**: text-only tasks work; tasks needing visual analysis fail with a configuration message rather than pretending the API saw the references.

纯文本 API 不会看图。不开 API 看图时，参考分析需要本地视觉模型；本地分析的文字仍会发到 API。开启 API 看图是上传授权，需选择真正支持视觉的模型。两项都关时，需看图的任务会明确报配置问题。视频分析仅抽帧，不是观看完整运动；该分析流程也不做音频听写。

## Advanced / 高级设置

- Output budget: Auto uses `max_completion_tokens` for OpenAI and `max_tokens` for other compatible providers; override for a gateway that expects a different field.
- Temperature: automatically omitted for recognized reasoning-model names that do not support it; disable manually if your selected model rejects it. Reasoning/thinking budgets are provider-specific and are not universally toggled here.
- Timeout: configurable total 10–600 seconds, default 180; individual socket inactivity timeout is capped at 60 seconds. A silent stalled response can therefore fail earlier than the total budget.
- Network/auth failures are **not retried automatically**, avoiding duplicate billed requests. An invalid complete H3 prompt can receive one format-repair request, which can incur another charge.
- Hidden reasoning is not used as the output prompt. Truncated/incomplete responses are rejected; H3 structure, Chinese descriptions and actual reference labels are validated before use.

供应商的思考模式并没有统一协议，本插件不承诺通用开关。网络／鉴权失败不自动重发；完整文案格式错误最多再请求一次修复。输出预算不足、流中断会明确失败，不把残缺文案静默用于生视频。

## Privacy / 隐私

- On Windows, saved keys are encrypted with Windows DPAPI under `ComfyUI/user/h3_fullstack/credentials/`, bound to the current Windows user and scoped to provider + exact normalized endpoint. A copied workflow does not include them.
- On other platforms, use an environment variable; no plaintext cross-platform key store is supplied. Keep the variable out of committed scripts and workflow JSON.
- No password value is returned by the status route. API configuration accepts only nonsecret fields; literal `api_key` workflow entries are rejected.
- Remote requests do not follow redirects with the Authorization header. Remote error bodies are not echoed. Credential/test mutations require a local loopback request and same-origin JSON.
- Intent, prompts and local visual descriptions **are transmitted** when using an API. Remote visual uploads require the separate vision checkbox. Provider retention, account access and logging policies remain the provider's responsibility.
- Local output reports, diagnostics and workflow paths may contain private content. Review before sharing; never upload credential files, environment files or raw user diagnostics with a bug report.

Windows 密钥由 DPAPI 加密，仅当前系统用户可用，按供应商／接口绑定；其它平台使用环境变量。本地凭据接口不返回密钥原文，不跟随携带授权信息的远端重定向。API 模式会发送文案和视觉观察文字；图片上传另行授权。工作流、输出报告和诊断也可能包含私人内容，发布前务必脱敏。

This release's automated API tests use a local fake HTTP server. No real paid provider or remote vision model was exercised for this release; compatibility with your chosen account/model must be tested separately. / 本次 API 验收使用本地模拟 HTTP 服务，未调用真实付费供应商／远端视觉模型；请用自己的账号与模型测试连接。
