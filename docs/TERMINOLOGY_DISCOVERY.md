# 全书候选术语发现与审核

## 目标与边界

本子系统发现“如果误译、多译或跨章节漂移，会影响全书理解”的候选概念。它不是关键词云，
也不会把模型输出直接变成约束。任何候选都必须满足两个条件：源词能回指书中精确字符区间；
人工明确批准对应义项。批准之前，候选不会进入翻译提示词或确定性质量检查。

## 分层数据模型

| 层 | 含义 | 允许的关系 |
|---|---|---|
| `term_lexeme_candidates` | 书中实际出现的规范词形及大小写表面形式 | 只做 NFKC、大小写和空白规范化；不自动词干化或合并近义词 |
| `term_candidate_senses` | 一个词形可能表达的概念/义项假设 | 同一词形可以有多个义项，各自拥有译法、证据和审核状态 |
| `term_candidate_evidence` | 原文段落、精确偏移、引文、章节位置 | 每条证据都能由 `source_text[start:end] == source_form` 复验 |
| `terms` | 人工批准后的翻译约束 | 只有人工批准操作能从候选创建该记录 |

这一区分避免把词形、概念和义项压成一个字符串。模型可以建议拆分义项，但只能引用已提供的
候选 ID 和证据 ID；不存在的 ID、无证据的义项和模型自造源词会被丢弃。

## Evidence-Grounded Glossary Mining v2

当前实现是“语言学候选生成 + 全书统计排序 + LLM 逐项裁决 + 人工批准”的混合架构。LLM
不是自由生成术语，而是在可复验的本地召回池上做有证据的 keep/drop 与译法建议。

1. **全文覆盖与语言纠错**：每个 Segment 都参与覆盖统计。系统用多语言功能词分布校验
   source_lang；当正文证据强烈反驳项目元数据时，以正文检测结果为准，并将
   language_profile 写入运行记录。
2. **语言边界优先**：在统计 n-gram 之前先按空白、标点、引号和语言缩合规则切边界。法语
   l’/d’/qu’ 等缩合被分离；逗号、括号和不配对引号不能把句子碎片拼成候选。
3. **元数据隔离**：ISBN、版权、电子书格式、目录、同作者书目和编辑说明按正文与
   heading_path 联合识别，不进入术语排序。
4. **类型分离**：候选显式标记为 concept、named_entity 或 lexical_risk。词形仍与
   term_candidate_senses 分层，避免把专名、普通词和概念义项错误合并。
5. **术语性与单元性联合排序**：C-value 抑制嵌套子串；association 分量比较短语频率与其
   组成词频率，降低偶然共现的普通句法片段。频率、全书分布、显式定义/标题/引语、边界
   置信度、词数和已有译文暴露共同决定风险排序。
6. **动态审核预算**：max_candidates 是安全上限，不是填满目标。实际本地复核池按书长动态
   限制在 12–40 项，并对普通单词和专名分别设预算；这避免长书因候选空间增大而机械填满
   300 项，同时保留引语、定义和低频重要词的通道。
7. **有界、精确证据**：统计读取全部 occurrence；每项只保存固定数量的分散证据，且每条都能
   复验 source_text[start_offset:end_offset] == source_form。
8. **LLM 逐项裁决**：每批最多 8 项，提示词要求对每个候选 ID 明确 keep/drop，drop 也必须
   引用有效证据。模型漏答时只对漏项重试一次；运行记录保存 model_decisions、
   missing_model_decisions、model_kept 和 model_omitted。
9. **审核队列与审计分离**：默认人工队列只显示模型建议保留和未完成复核的项；模型略过项仍
   可展开检查，绝不删除。模型判断始终是建议，任何候选仍须人工批准后才进入翻译约束。
10. **批准后回查**：批准与创建 TermEntry 在同一事务中完成，随后扫描既有译文、重建术语
    QA，并把影响清单与决策写入审计事件。

## 当前风险分

所有分量先归一化到 0–1：

~~~text
risk = 0.24*CValue
     + 0.17*logFrequency
     + 0.13*bookDispersion
     + 0.16*explicitEvidence
     + 0.10*boundaryConfidence
     + 0.12*phraseAssociation
     + 0.04*multiword
     + 0.04*translatedExposure
~~~

专名另有轻微降权，普通单词、专名和总队列分别受预算约束。该分数仅用于本地复核池排序，
不代表“术语为真”的概率；模型置信度独立保存，也不能覆盖原文存在性校验。

## 模型配置

在“模型配置 → 任务模型”中独立设置术语发现的连接、模型和计算模式。候选筛选、义项和
译法建议使用这项绑定；修改草译模型不会改变它。留空时只运行本地扫描。
旧配置首次加载时复制原草译绑定作为初始值，保存后独立持久化。术语库显示下一次扫描
使用的模型；历史运行仍保留当时的模型。批准后的术语语境核验使用草译模型。

## 成本、覆盖与失败语义

- 本地阶段始终扫描全文；动态预算只限制复核池，不改变覆盖计数。
- 默认上限为本地 40 项、模型 40 项、每批 8 项；实际池通常小于上限。
- 每次运行保存 prompt/completion/reasoning token、费用、模型调用数、无效提案和逐项决策覆盖。
- 模型失败或两轮后仍漏答时，漏项保持“尚未模型复核”，不会被静默当成保留或舍弃。
- 模型 drop 项从默认人工队列折叠，但证据、理由和运行历史全部保留。
- 输入指纹包含原文哈希、发现配置和当前可见译文状态，便于判断运行是否陈旧。

## API

- `POST /documents/{id}/term-discovery-runs`：全文扫描；可选模型复核。
- `GET /documents/{id}/term-discovery-runs`：运行历史、覆盖和费用。
- `GET /documents/{id}/term-candidates`：最新运行或指定运行的词形、义项与证据。
- `PATCH /term-candidate-senses/{id}`：人工编辑或驳回，保留审计。
- `POST /term-candidate-senses/{id}/approve`：人工批准、创建约束并回查现有译文。

## 评估建议

不能只报告候选数量。上线语料应人工构建按义项标注的小型 gold set，并至少跟踪：

- 动态复核池的全书 gold 术语 Recall，另报低频（1–2 次）召回率；
- 模型 keep 队列 Precision、drop 误杀率和每批准一条所需人工分钟；
- 词形合并错误率、同形异义错误合并率；
- 原文证据可复验率（目标 100%）；
- 批准术语在既有译文中的不一致发现数与修复后残留数；
- 逐项模型决策覆盖率、每万原文字符的模型 token 和费用。

## 后续算法升级路线

当前零外部依赖算法适合作为稳定基线。具备相应语言模型或解析器时，可增加而不替换证据门：

1. Universal Dependencies 名词短语模式，用语言学边界提升多词候选精度；
2. 专域语料与通用语料的 contrastive keyness，降低一般叙述词；
3. occurrence 级上下文向量的局部聚类，再做跨词形全局概念聚类，以软聚类表达多义与近义；
4. 基于人工批准/驳回历史的学习排序和不确定性采样；
5. 从已确认双语段落做词对齐，直接估计同一源概念的译名熵，而非仅用“已有译文暴露度”代理。

## 研究依据

- Pollak et al., 2016, [A Hybrid Approach to Term Extraction](https://aclanthology.org/L16-1081/)
- Kliegr et al., 2016, [Semantic Textual Similarity for Terminology Extraction](https://aclanthology.org/L16-1294/)
- Brank et al., 2020, [Human-in-the-Loop Entity Linking](https://aclanthology.org/2020.acl-main.624/)
- Universal Dependencies, [French tokenization rules](https://universaldependencies.org/fr/tokenization.html)
- Šajatović et al., 2019, [Evaluating Automatic Term Extraction Methods on Individual Documents](https://aclanthology.org/W19-5118/)
- Marciniak et al., 2023, [TermoUD — a language-independent terminology extraction tool](https://aclanthology.org/2023.eacl-demo.21/)
- Liétard et al., 2024, [To Word Senses and Beyond: Inducing Concepts with Contextualized Language Models](https://arxiv.org/abs/2406.20054)
- Kim et al., 2024, [Efficient Terminology Integration for LLM-based Translation in Specialized Domains](https://aclanthology.org/2024.wmt-1.51/)

