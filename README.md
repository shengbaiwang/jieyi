# 介译（Jieyi）

面向人文社科长文本的可追溯、多模型翻译工作流。目前是一个可运行的工程内核：它把长文分成具有稳定身份的结构化段落，通过可替换的模型适配器生成候选，逐段保存检查点，并将机器译文、人工确认、质量问题和审计记录分别保存。

本项目采用独立核心，而不是合并 TranslateBooksWithLLMs（TBL）和 Supervertaler 的代码。这样既保留长文流水线与专业译者工作台的设计优点，也把许可证、UI 框架和供应商 SDK 隔离在核心之外。

## 图形工作台

在 macOS 中双击项目根目录的 `启动介译.command`。启动器会同时运行本地 API 与图形界面，并打开 `http://localhost:3000/`。

- 在左侧选择“模型配置”，可添加多个独立连接，并把草译、审校分别绑定到不同连接和模型。Kimi Coding、GLM、OpenRouter、DeepSeek、本地模型等使用显式端点，不再猜测 `/v1` 或 `/v4`；
- 每个连接的云端密钥使用独立的 macOS 钥匙串条目，不会写入配置文件。后台进程无法取得钥匙串授权时会降级为本次运行会话，并在界面提示；
- 在左侧选择“导入书籍”，拖入 EPUB、TXT、MD 或 Markdown 文件，确认项目名称、书名、原文/目标语言和翻译风格，检查分段预览后点击“导入并创建项目”；
- 导入完成后会直接打开真实书稿；书库、目录、全文搜索、分段对照、人工草稿、确认状态、术语、质量问题和双语导出都读取本机数据库。长书按页加载，阅读位置会自动记忆；
- 书库支持一键导出译后书籍：EPUB 会把译文写回原书 XHTML，并保留封面、目录、CSS、字体和图片；TXT/Markdown 直接导出译文文件；
- 导入时可从“学术严谨、文学自然、通俗易读、忠实直译”中选择风格并继续自定义，导入后可在“翻译设置”中随时调整；
- 每本书可单独选择草译与审校连接、模型和计算档位；任务会固化这些选择，因此多本书可以同时用不同模型翻译；
- 导入的正文、分段和项目设置保存在本机 `jieyi.db`，模型的非敏感设置保存在 `jieyi.settings.json`。

OpenRouter 使用 `https://openrouter.ai/api/v1`，模型填写完整 ID（例如 `anthropic/...` 或 `openai/...`）。工作台每次运行 8 个段落，完成后可继续下一批，以便在发送长书内容前控制进度与费用。

## 当前已实现

- TXT/Markdown 图形化导入，识别标题、正文、块引用和脚注块；
- EPUB 2/3 图形化、CLI 与原始字节 API 导入：原始 ZIP、OPF、manifest、spine、导航、XHTML、CSS、封面、图片、SVG 和字体全部保留，并识别 EPUB 2/3 封面与 fixed-layout 元数据；
- 内容寻址的稳定段落键，插入无关段落不会改变已有段落身份；
- SQLite 持久化，逐段断点续跑；
- `TranslationProvider` 窄接口，可接不同云模型和本地模型；
- OpenAI-compatible 适配器，可连接云 API、Ollama、LM Studio 或 vLLM；
- 项目风格、局部上下文和按需术语注入；
- 引文、脚注、URL、DOI、代码与标签的严格占位符保护，损坏时聚焦重试；
- 人工确认自动沉淀项目 TM，后续采用精确优先、模糊其次的译例检索；
- 全文候选术语发现：C-value 与翻译风险集成排序、低频显著概念保留、原文偏移证据、可选模型义项/译法建议、人工批准和既有译文回查；
- 分布式全文取样，为风格和其他全书分析提供稳定入口；
- API/CLI 提示词预览，预览与真实模型请求共用同一个构建器；
- 草译、条件审校和人工确认三种不同状态；
- 单段隔离的持续任务队列：默认从 3 路并发安全起步，连续成功后最多扩至 5 路，调用异常时自动收缩；
- 段落级结构或内容验收失败会隔离为红色质量问题，不写入无效译文，也不阻断全书其余段落；
- 脚注、批准术语和禁用译法 QA；
- 候选、人工决定、问题和审计事件分别存储；
- 无第三方依赖的 CLI，以及可选的 FastAPI 外壳。

术语发现的算法、证据不变量、成本控制和评估方法见 [全书候选术语发现与审核](docs/TERMINOLOGY_DISCOVERY.md)。

## 快速验证

核心运行只需要 Python 3.11+；完整测试（含 HTTP API）安装开发扩展：

```bash
uv sync --extra api --extra dev
uv run pytest
```

安装为本地命令：

```bash
uv sync
uv run jieyi --db demo.db init-db
```

创建项目并导入示例文本：

```bash
uv run jieyi --db demo.db project-create \
  --name "社会理论选读" --source-lang en --target-lang zh-CN

uv run jieyi --db demo.db document-import \
  --project <上一步的项目 ID> --file examples/sample.md
```

EPUB 可直接从 CLI 导入，书名默认读取包内 Dublin Core 元数据：

```bash
uv run jieyi --db demo.db document-import \
  --project <项目 ID> --file "/绝对路径/book.epub"
```

EPUB 同时进入既有 Segment/术语/审校/QA 流程，并保存 SourceAtom 到 spine、DOM 元素和文本节点的稳定映射。阅读模式在禁用脚本与外部网络的 sandbox iframe 中呈现原 XHTML，提供“原版、仅译文、双语对照”三种模式；译文以原 XHTML 为模板回填，因此继承标题、字体、颜色、缩进、粗斜体、脚注、链接、表格和图片结构。

加入经人工批准的术语约束。`--alias` 声明会主动命中的同义词或变体；
`--context`、`--sense` 和 `--disambiguation` 用于区分同形词的不同义项：

```bash
uv run jieyi --db demo.db term-add \
  --project <项目 ID> --source agency --target 能动性 \
  --alias "agentic capacity" --context action --context autonomy \
  --sense "行动主体的能力" --disambiguation "与 structure 对举时采用此义项" \
  --rationale "与 structure/结构 的概念对举保持一致" \
  --forbid 代理性
```

已批准条目不是查询参考：系统会在每段原文中按词界主动识别规范源词和别名，
把唯一匹配的义项作为强制约束写入生成提示词，并在草译、模型校对、人工保存及
人工确认后重新检查。若多个义项都可能成立且语境关键词不足，系统不会强制套用
冲突译法，而会生成“术语未消歧”警告供模型或人工复核。

先用内置 `echo` 供应商验证完整流程，不会访问网络：

```bash
uv run jieyi --db demo.db job-create \
  --document <文档 ID> --draft-provider echo --draft-model dry-run \
  --review-policy never

uv run jieyi --db demo.db job-run --job <任务 ID>
```

执行前检查实际模型消息、术语、TM 和受保护片段：

```bash
uv run jieyi --db demo.db prompt-preview \
  --job <任务 ID> --segment <段落 ID>
```

任务中断后再次执行同一条 `job-run` 命令，会从 `next_ordinal` 继续。机器译文不会自动获得“人工确认”状态：

```bash
uv run jieyi --db demo.db segment-confirm \
  --segment <段落 ID> --translation "人工确认后的译文" \
  --rationale "核对了原著术语用法"
```

## 连接真实模型

连接任意 OpenAI-compatible 服务时，可以使用服务器环境变量，或在仅监听本机回环地址的图形工作台中保存设置。面向远程部署时应关闭本地设置接口，并通过部署环境注入固定端点，避免把模型网关变成 SSRF 通道。

```bash
export JIEYI_OPENAI_API_KEY="..."

uv run jieyi --db demo.db job-run --job <任务 ID> \
  --base-url https://provider.example/v1
```

创建任务时使用的供应商名称应为 `openai-compatible`。本地无密钥服务可以不设置密钥。

图形工作台使用版本化的多连接配置。旧版单连接及 `version: 2` 配置会在读取时自动迁移；保存后写为 `version: 3`。每个 Profile 明确保存 `base_url`、`chat_path`、`models_path` 和 `protocol`，草译与审校通过 `profile:<id>` 注册名独立调用。

支持的协议包括：`chat_completions`（OpenAI-compatible，默认）、`responses`（OpenAI Responses）、`anthropic_messages`（Anthropic Messages）和 `gemini_generate_content`（Gemini 原生接口）。

模型计算不再直接保存供应商的 `none / low / high / xhigh` 等档位。用户只选择“节约、均衡、性能”策略，任务保存这一稳定意图；调用时由适配器按具体模型映射到其原生 `reasoning_effort` 或 `thinking` 控制。已知模型使用各自能力表，未知 OpenAI-compatible 模型从可移植候选值开始，收到明确的不支持错误后自动尝试下一档或回落到模型默认值，并按“连接实例 + 模型 + 策略”缓存成功映射。不提供原生推理控制的模型仍可运行，但其隐藏推理 token 无法由客户端精确约束。

除钥匙串外，也可按 Profile ID 注入环境变量。变量名规则是 `JIEYI_API_KEY_<PROFILE_ID>`，连字符会转换为下划线并转为大写，例如 `kimi-coding` 对应：

```bash
export JIEYI_API_KEY_KIMI_CODING="..."
```

## HTTP API

```bash
uv sync --extra api
JIEYI_DB=demo.db uv run uvicorn jieyi.api.app:create_app --factory --reload
```

打开 `http://127.0.0.1:8000/docs` 查看交互式接口。长任务在当前 MVP 中可以由 `/jobs/{id}/run` 运行；正式部署更适合由独立 worker 调用同一个 `TranslationEngine`，HTTP 层只负责创建任务和查询状态。

上传 EPUB 时直接发送文件字节，无需 multipart：

```bash
curl -X POST \
  -H "Content-Type: application/epub+zip" \
  --data-binary @"/绝对路径/book.epub" \
  "http://127.0.0.1:8000/projects/<项目ID>/documents/epub"
```

图形工作台支持直接拖放 EPUB：选择文件后读取书名、书脊、封面、资源数和 fixed-layout 信息并生成分段预览。普通流式 EPUB 会保留原书视觉结构，但译文长度变化会重新换行和分页；双语模式会增加页面高度。fixed-layout EPUB 可在“忠实排版”（译文溢出区可滚动）与“舒适阅读”（解除绝对定位并重排）之间切换。

已存在但缺少原书资源的 EPUB 可用同一份原文件补全；服务端先校验 SHA-256，且不改写已有 Segment 译文、人工确认、决定或审计记录：

```bash
curl -X PUT \
  -H "Content-Type: application/epub+zip" \
  --data-binary @"/绝对路径/book.epub" \
  "http://127.0.0.1:8000/documents/<文档ID>/epub/source"
```

阅读器清单位于 `/documents/{id}/epub`，原始文件可从 `/documents/{id}/epub/original` 下载；spine 渲染端点只接受 `original`、`translated`、`bilingual` 和 `faithful`、`comfort` 的受控组合。



## 目录边界

```text
src/jieyi/
├── domain/       # 纯领域模型和供应商端口
├── ingestion/    # 文本解析与稳定分段
├── context/      # 有预算的上下文编译
├── providers/    # 模型供应商适配器
├── quality/      # 确定性 QA
├── persistence/  # SQLite 实现
├── workflow/     # 可恢复翻译用例
├── api/          # 可选 FastAPI 外壳
└── cli.py        # 命令行入口
```

更详细的边界、状态机和扩展顺序见 [架构说明](docs/ARCHITECTURE.md) 与 [工程决策](docs/DECISIONS.md)。

## 许可证状态

项目尚未替所有者选择发布许可证。在做出开源或商业化决定前，不应加入 TBL 的 AGPL-3.0 源码。当前实现从零编写；参考项目及边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
