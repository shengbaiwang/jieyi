# 架构说明

## 目标

介译的核心对象不是“某次模型调用”，而是一个可以中断、校订、复现和追责的翻译工程。领域层不依赖 FastAPI、供应商 SDK 或文件格式库。

依赖方向固定为：

```text
API / CLI
    ↓
workflow
    ↓
domain ← context / quality / ingestion
    ↑
persistence / providers（实现领域端口）
```

`domain` 不允许反向导入外层模块。接入 TBL、Supervertaler、DOCX、消息队列或新的数据库时，只新增适配器。

## 规范文档模型

所有输入先转成 `Document + Segment[]`。`Segment.stable_key` 由以下内容计算：

- 块类型；
- 标题路径；
- 规范化后的源文内容；
- 同一标题下重复块的序号。

它不依赖全局段落位置，因此在文档中插入无关段落后，其他块的稳定键保持不变。实际数据库 ID 还带有文档命名空间，防止跨文档冲突。

下一阶段的“重新导入”应以 `stable_key` 对齐新旧版本，迁移人工确认与译者笔记；对内容发生变化的段落只创建待复核版本，不静默复用旧译。

### EPUB 导入适配器

EPUB 适配器不把 ZIP 解压到磁盘，而是：

1. 检查成员数量、单文件/总解压大小、压缩率、重复路径和路径穿越；
2. 拒绝 XML DTD 与实体声明，并仅展开安全的内置 HTML 命名实体；
3. 从 `META-INF/container.xml` 定位 OPF package，解析 manifest、完整 spine、EPUB 2 NCX / EPUB 3 nav、page progression 和 rendition metadata；
4. 原始 EPUB 字节与每个成员（OPF、XHTML、CSS、封面、图片、SVG、字体等）一起写入 `epub_packages` / `epub_resources`；EPUB 3 `cover-image`、EPUB 2 `meta name=cover` 及 guide cover 都会解析；
5. 可翻译文本仍经 `SourceAtom → ParsedBlock → Segment` 进入既有术语、审校、QA、TM 和人工确认流程；
6. `epub_atoms` 与 `epub_text_nodes` 保存 Segment/SourceAtom 到 spine index、DOM path、`element.text` / `element.tail` slot 的稳定映射；
7. 多 SourceAtom Segment 的模型输入使用受保护的 atom 与内联标签边界，模型必须按原顺序逐 atom 返回；恢复时同时验证 placeholder 和 atom ID/顺序；
8. 阅读器以原 XHTML 为模板产生原版、仅译文和双语视图。内容通过无脚本 sandbox iframe 呈现，服务端删除脚本、事件属性、表单及危险 URL，CSS/SVG 只允许访问同一本 EPUB 的白名单资源；
9. 同一项目再次提供相同 SHA-256 的 EPUB 时只补全/刷新包资源与映射，不新建重复文档，也不改写 Segment 译文、人工决定和确认记录。

原始 ZIP 始终可原样取回。当前“模板式回填”用于阅读，不生成新的可下载翻译 EPUB；流式书会因译文长度重排，双语会增加高度。fixed-layout 提供忠实排版溢出保护和舒适阅读重排覆盖层。

## 翻译状态

```text
source
  ↓ 模型生成
machine_translated
  ↓ 人工明确确认
human_confirmed
```

机器审校不会产生 `human_confirmed`。每次模型输出写入 `candidates`，当前机器稿写入段落；人工确认同时写入 `decisions` 和 `audit_events`。重跑模型不能覆盖 `accepted_translation`。

## 任务状态与恢复

```text
pending → running → completed
             ├──→ paused → running
             └──→ failed → running
```

每处理完一个段落，系统按以下顺序提交：

1. 保存模型候选；
2. 运行确定性 QA；
3. 必要时调用审校模型并保存第二候选；
4. 保存最终机器稿与问题；
5. 更新任务 `next_ordinal`。

因此进程在下一段失败时，上一段仍然是完整检查点。生产 worker 可以直接重试相同任务。

优化执行器始终保持“一段一次模型请求”，避免跨段串译；调度器使用持续补位队列，
从任务配置的安全并发数起步，在连续成功后逐级增加到上限，任何调用异常都会立即收缩。
这只改变请求调度，不改变提示词、模型参数或译文验收规则。

内容拒绝、输出预算耗尽以及占位符或 EPUB SourceAtom 最终修复失败等可确定为单段局部的
错误，会记录 `translation_deferred` 硬错误并隔离该段。无效输出不会写入当前译文，后续
审校任务可从原文重新处理；其余段落继续运行。无法确定为局部错误的连接中断、服务商
故障等仍使任务失败并保留断点，防止把系统性故障误判为大量坏段。

## 上下文编译

首版上下文包含：

- 项目语种、领域、引文策略和风格指南；
- 当前标题路径；
- 当前段落实际出现的批准术语及同义词；
- 基于显式语境关键词选定的义项，以及无法可靠消歧的候选义项；
- 指定半径内的前后原文；
- 前文已有的人工确认或机器译文。

编译结果受到字符预算约束，并作为普通字符串传入供应商。后续可以增加章节摘要、概念关系和语义检索，但不能让供应商直接访问数据库。

术语由 `jieyi.terminology` 统一解析，标准翻译、并发翻译、提示词预览和 QA
不得各自实现子串匹配。唯一义项以 mandatory constraint 形式注入；同一源词
命中多个义项时，以语境关键词择优，不能唯一确定则只提示候选并产生 QA 警告。

### 受保护片段

模型调用前，系统将以下内容替换成顺序编号的占位符：

- Markdown 行内/块代码；
- HTML/XML 标签；
- 脚注标记；
- URL 和 DOI；
- 年份型括号引文和数字型方括号引用。

恢复时同时校验数量、唯一性、未知占位符和顺序。第一次输出损坏时，系统保存原候选并调用同一供应商执行一次 `repair`；修复仍失败则任务进入 `failed`，不会推进段落检查点。

## 翻译记忆

只有 `human_confirmed` 才会写入项目 TM。相同规范化源文更新当前译例，历史修改仍保留在 `decisions`。检索顺序为：

1. 规范化后的精确匹配；
2. 长度上界预筛；
3. 标准库 `SequenceMatcher` 模糊匹配；
4. 按相似度取项目配方规定的前 N 条。

TM 只作为参考上下文，不自动产生人工确认。后续数据量增大时，可以把候选预取或索引实现替换掉，而不改变 `ContextCompiler` 的语义。

## 提示词可见性

`jieyi.prompting.build_messages()` 是 OpenAI-compatible 调用与 prompt preview 的共同来源，避免“界面预览”和“实际请求”漂移。预览同时列出被保护的原文片段与相关术语，但不会发起模型请求。

## 分布式取样

长文档预分析不应只看开头。取样器在总字符预算内，从首部、中部和尾部等距选取窗口，并把窗口边缘吸附到空白字符。当前 API 用它提供文档样本；下一阶段的术语/风格提取必须复用此入口。

## 质量检查快照

质量问题描述的是“当前可见译文”的状态，而不是某次任务永久有效的结论：

- 检测结果记录检测器版本和当前译文哈希；
- 模型生成、人工保存、人工确认和术语新增后都重新检测；
- 新结果将旧结果标记为已解决，历史记录继续保留用于审计；
- 查询只返回未解决且译文哈希仍匹配的结果，避免并发或中断造成陈旧告警；
- 检测器版本升级时全库重扫一次，成功后记录版本检查点。

结构性不变量（引用、脚注、批准术语）使用 `error`；数字检测先把月份、日期、年代、
世纪、百分比、数量单位、时间、章节坐标和中文数字转换成类型化语义事实，再检查原文
事实是否仍存在于译文。脚注段不运行通用数字规则，书目坐标由结构化规则负责。无法
可靠解释的数字差异不进入用户问题列表。界面只把 `error` 计入红色待处理数，`warning`
明确标为不阻断流程的建议复核；`on_issue` 自动审校也只由 `error` 触发。

## 模型路由

`TranslationRecipe` 将角色和供应商分开：

- `draft`：主译；
- `reviewer`：独立审校；
- `review_policy`：`never`、`on_issue` 或 `all`。

当前 `on_issue` 由确定性 QA 触发。下一阶段可以增加风险分类器和模型分歧，但不应默认让所有模型处理所有段落。

### 跨模型计算策略

`TranslationRecipe` 保存 `economy / balanced / performance` 用户意图，同时保留旧任务的具体 effort 字段用于读取兼容。`domain.reasoning` 负责把意图映射成模型原生能力：

- 已知模型按自身支持的离散等级选择低位、中位或高位；
- 未知兼容模型使用按策略排序的候选值，不假定所有模型具有相同等级；
- 只支持开关的模型映射为 `thinking` 布尔值；
- 没有原生控制时省略参数，使用模型默认行为；
- 适配器只在服务端明确返回“不支持 reasoning effort”的 400 时降级，并缓存成功结果。

因此 UI、任务恢复和业务规则不依赖任何供应商的等级命名；供应商差异只存在于纯领域映射和请求适配器中。

## 扩展点

优先按以下方式扩展：

- 文件格式：新增 ingestion adapter，仍输出 `Segment[]`；
- 模型：实现 `TranslationProvider`；
- TBL：作为长文档格式或作业执行适配器，不进入领域层；
- Supervertaler：通过 TMX/TBX/XLIFF 或明确 API 交换，不共享其 PyQt UI 状态；
- 数据库：实现与 `SQLiteStore` 同等语义的仓储；
- worker：调用 `TranslationEngine.run()`，不复制工作流。

## 尚未实现的生产能力

- 登录、项目权限和 API 密钥加密；
- 跨进程任务租约、心跳和并发锁；
- 文档重新导入与稳定键对齐；
- 自动术语/概念抽取及人工审核界面；
- 语义 TM 和章节/全书知识包；
- 大规模 TM 的 FTS/向量候选预取；
- 更完整的 CAT 能力（候选版本对比、批注、快捷键自定义与多人协作）；
- 模型响应的结构化 schema 和更细的成本统计；
- DOCX、TEI、XLIFF/TMX/TBX；
- 保留 XHTML/CSS/图片/目录映射的 EPUB round-trip 导出；
- 数据库迁移版本管理和 PostgreSQL 实现。
