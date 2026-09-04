# 全书候选术语发现与审核

## 目标与边界

本子系统发现“如果误译、多译或跨章节漂移，会影响全书理解”的候选概念。它不是关键词云，
也不会把统计或模型输出直接变成翻译约束。任何候选都必须满足两个条件：源词能回指书中
精确字符区间；人工明确批准对应义项。批准之前，候选不会进入翻译提示词或确定性质量检查。

“全面”在这里指高召回、可检查和跨语言稳健，而不是承诺零漏检。系统宁可把有依据的边缘项
放入有限复核池，也不允许模型脱离原文凭空补词。

## 分层数据模型

| 层 | 含义 | 允许的关系 |
|---|---|---|
| `term_lexeme_candidates` | 书中实际出现的规范词形及全部观察到的表面形式 | NFKC、大小写和空白规范化；只做保守屈折词形族归并，不合并近义词 |
| `term_candidate_senses` | 一个词形可能表达的概念/义项假设 | 同一词形可以有多个义项，各自拥有译法、证据和审核状态 |
| `term_candidate_evidence` | 原文段落、精确偏移、引文、章节位置 | 每条证据都能由 `source_text[start:end] == source_form` 复验 |
| `terms` | 人工批准后的翻译约束 | 只有人工批准操作能从候选创建该记录 |

屈折归并只把同一词的保守变体放进一个词形族，例如德语 `Werk / Werke / Werkes`、法语
`lieu / lieux`。系统仍保存每个实际词形及其原始偏移；不会把 `lieu` 与 `lieue`、德语
`Masse` 与 `Maß` 这类不同词合并。模型可以建议拆分义项，但只能引用已提供的候选 ID
和证据 ID；不存在的 ID、无证据的义项和模型自造源词会被丢弃。

## Evidence-Grounded Glossary Mining v3.2（严格本地准入）

当前实现是“多语种语言学召回 + 全书统计与结构排序 + 分类型复核池 + LLM 证据裁决 +
人工批准”的混合架构。LLM 不是自由生成术语，而是在本地已验证的候选及少量证据摘录上
做 keep/drop 与译法建议；系统不会把整本书作为术语发现提示发送给外部模型。

1. **全文覆盖与语言纠错**：每个 Segment 都参与覆盖统计。系统用正文功能词和字符分布
   校验 source_lang；当正文强烈反驳项目元数据时，以正文检测结果为准，并把
   `language_profile` 写入运行记录。
2. **多语种边界**：拉丁字母文本先按 Unicode 单词、标点、引号、句界和语言缩合切分，再
   生成 2–5 词候选。当前为英语、德语、法语、西班牙语、意大利语、葡萄牙语配置功能词、
   限定词、连接词、并列词、定义提示和名词派生后缀；中日韩文本使用独立字符 n-gram
   通道。法语 `l’/d’/qu’` 等缩合不会把句法碎片粘成术语。
3. **保守形态族**：候选统计前归并复数、格变化等高置信屈折变体，避免一个概念被不同词形
   分散排名。归并规则按语言限制，保留 `ß` 等正字法差异以及所有原文表面形式，不做
   激进词干化、词元猜测或跨词近义合并。
4. **结构信号**：除普通 n-gram 外，单独记录定义、引语、标题、句末修辞边界、连接词短语、
   并列概念、德语名词短语、专名和连字符/缩写风险。这让低频但显式界定的概念、多词概念
   和翻译时容易漂移的词形拥有独立召回路径。
5. **元数据隔离**：ISBN、版权、电子书格式、目录、同作者书目和编辑说明按正文与
   `heading_path` 联合识别，不进入术语排序。
6. **统计术语性**：C-value 抑制被更长固定短语吸收的子串；短语 association 使用预建
   单词索引比较短语频率与组成词频率。频率、全书分布、定义/标题/引语、边界、上下文多样性、
   正字法、名词派生形态、短语结构和既有译文暴露共同排序。核心计算对候选量近似线性，
   不再逐候选重复扫描全部单词候选。
7. **分类型配额与交错排序**：concept、named_entity、lexical_risk 分开；复核池再分为
   核心单词、派生名词、普通短语、显著短语、并列概念、词形风险和专名七条通道。每条通道
   独立排名后按相对位置交错，防止高频普通词、标题专名或大量短语垄断前部模型预算。
   非德语文本把更多容量留给多词概念；德语则提高复合词、短名词和显著概念对的容量。
8. **严格本地准入与自适应预算**：预算根据正文词数（CJK 按字符单元）而不是段落数量计算，
   本地复核池为 25–70 项，并受 `max_candidates` 上限约束。分道配额不再为了凑数回填：
   单词必须同时具备复现、跨段、词形术语性和上下文证据；普通短语必须有名词短语结构；
   四词以上非显式短语、低频专名、通用叙述名词、同频嵌套残片和疑似断行词形会在本地淘汰。
   显式定义可越过通用名词表。复核池仍是“待甄别候选库存”，不是有效术语数量。
9. **有界、精确证据**：统计读取全部 occurrence；每项最多保存 6 条跨全文分散证据，
   每条均可复验精确偏移。模型默认最多复核交错排序后的 200 项，每批 4 项，只接收这些
   候选卡片和证据摘录。
10. **四重精度门**：模型只有在 `stable_concept`、`book_significant`、
    `consistency_needed`、`specialized_usage` 四项全部为真且置信度不低于 0.75 时，
    才能建议保留。高频、首字母大写、出现在标题中或仅与主题相关，都不能单独构成保留理由；
    任一条件不成立或不确定时一律建议舍弃。
11. **可恢复裁决**：模型必须对每个候选 ID 明确 keep/drop，drop 也要引用有效证据。
    漏答、格式无效或输出截断时只重试缺项，批次逐步缩小至单项；连接异常时停止本次复核，
    已完成判断与本地候选继续保留。
12. **人工控制与回查**：模型判断始终是建议。人工批准与创建 TermEntry 在同一事务完成，
    随后扫描既有译文、重建术语 QA，并把影响清单与决策写入审计事件。

## 当前风险分

所有分量先归一化到 0–1：

~~~text
risk = 0.17*CValue
     + 0.14*logFrequency
     + 0.10*bookDispersion
     + 0.10*explicitEvidence
     + 0.06*boundaryConfidence
     + 0.08*phraseAssociation
     + 0.05*multiword
     + 0.07*lexicalSpecificity
     + 0.04*contextDiversity
     + 0.06*orthographicTermhood
     + 0.07*nominalTermhood
     + 0.04*phraseStructure
     + 0.02*translatedExposure
~~~

这只是进入分类型通道前的全局风险分。每条通道还用与其任务匹配的二级分数，例如显著短语
更重视定义/引语并兼顾频率和分布，派生单词更重视名词后缀，词形风险更重视频率与边界。
专名有轻微降权并受独立配额限制。任何分数都不代表“术语为真”的概率，也不能覆盖原文
存在性校验。

## 模型配置

在“模型配置 → 任务模型”中独立设置术语发现的连接、模型和计算模式。候选筛选、义项和
译法建议使用这项绑定；修改草译模型不会改变它。留空时只运行本地扫描。旧配置首次加载时
复制原草译绑定作为初始值，保存后独立持久化。历史运行保留最初模型，续跑模型另记在
coverage 中；切换模型只影响未完成项。批准后的术语语境核验使用草译模型。

## 成本、覆盖与失败语义

- 初次本地阶段扫描全文，候选与证据在首次模型请求前一同落库。扫描指纹包含原文、配置和
  算法版本；v2、v3.0、v3.1 结果不会被误当成 v3.2 结果续跑。
- 默认上限为本地 70 项、模型 70 项；实际本地数量可更少。每项模型只看到最多 6 条
  有界原文证据。仅本地模式不产生模型请求，其结果明确标作“尚非术语的待甄别候选”。
- 每次模型回复在同一事务保存新判断及累计 prompt/completion/reasoning token、费用、
  调用数和无效提案数。缺失项从持久化候选重新计算，费用只按新回复累加。
- 复核不完整时运行状态为 `partial`。点击“继续复核 N 项”沿用同一
  run/candidate/evidence ID，仅提交尚未得到有效判断的候选；重复点击由数据库原子占用去重。
- 浏览器请求断开不会取消后台复核；应用重启后已保存的扫描恢复为可续跑状态。已完成判断
  以及人工批准、驳回、编辑、撤销批准的记录都受保护。
- 模型 drop 项从默认人工队列折叠，但证据、理由和运行历史全部保留。没有有效结果的候选
  不会被静默当成保留或舍弃。
- 诊断仅保存候选 ID、尝试序号、输出上限、finish_reason 和错误类别，不保存完整模型回复。

## API

- `POST /documents/{id}/term-discovery-runs`：按当前算法版本复用兼容扫描，否则新建全文扫描，
  可选模型复核。
- `POST /documents/{id}/term-discovery-runs/{run_id}/retry`：仅复核兼容运行中的未完成项；
  空请求体沿用已保存模型，可指定 provider/model/compute_mode；不扫描全文。
- `GET /documents/{id}/term-discovery-runs`：运行历史、覆盖和费用。
- `GET /documents/{id}/term-candidates`：最新运行或指定运行的词形、义项与证据。
- `PATCH /term-candidate-senses/{id}`：人工编辑或驳回，保留审计。
- `POST /term-candidate-senses/{id}/approve`：人工批准、创建约束并回查现有译文。

## 回归验证与评估

自动回归覆盖以下不依赖具体书句的行为：

- 德语复合词、概念对、`Hier und Jetzt` 式并列结构，以及 `Werk/Werke/Werkes` 词形族；
- 法语 `lieu/lieux` 与 `lieue/lieues` 不误合并，并过滤常见助动词噪声；
- 英语抽象名词后缀、连字符词、多词概念和显式定义；
- 西班牙语、意大利语、葡萄牙语定义提示；
- CJK 候选生成、证据偏移可复验、同一 occurrence 多信号但只计数一次；
- 自适应预算只取决于语料规模，不取决于段落切分数量。

上线语料仍应人工构建按义项标注的 gold set，至少跟踪复核池 Recall（另报低频召回）、
模型 keep Precision（精度优先）、drop 误杀率、四重精度门拒绝率、词形误合并率、
证据可复验率（目标 100%）、人工分钟、
批准后译名不一致残留，以及每万原文字符的模型 token 和费用。

## 后续算法升级路线

v3.2 的零外部依赖核心适合作为稳定、可解释基线。具备本地语言模型或解析器时，可在不替换
证据门的前提下增加：

1. Universal Dependencies 名词短语模式，进一步提升多词候选边界；
2. 专域语料与通用语料的 contrastive keyness，降低一般叙述词；
3. occurrence 级上下文向量的本地聚类，再做跨词形概念软聚类，以表达多义与近义；
4. 基于人工批准/驳回历史的学习排序和不确定性采样；
5. 从已确认双语段落做词对齐，直接估计同一源概念的译名熵。

## 研究依据

- Pollak et al., 2016, [A Hybrid Approach to Term Extraction](https://aclanthology.org/L16-1081/)
- Kliegr et al., 2016, [Semantic Textual Similarity for Terminology Extraction](https://aclanthology.org/L16-1294/)
- Brank et al., 2020, [Human-in-the-Loop Entity Linking](https://aclanthology.org/2020.acl-main.624/)
- Universal Dependencies, [French tokenization rules](https://universaldependencies.org/fr/tokenization.html)
- Šajatović et al., 2019, [Evaluating Automatic Term Extraction Methods on Individual Documents](https://aclanthology.org/W19-5118/)
- Marciniak et al., 2023, [TermoUD — a language-independent terminology extraction tool](https://aclanthology.org/2023.eacl-demo.21/)
- Liétard et al., 2024, [To Word Senses and Beyond: Inducing Concepts with Contextualized Language Models](https://arxiv.org/abs/2406.20054)
- Kim et al., 2024, [Efficient Terminology Integration for LLM-based Translation in Specialized Domains](https://aclanthology.org/2024.wmt-1.51/)
