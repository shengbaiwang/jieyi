# EPUB 结构重排

## 目标

EPUB 中的 XHTML 块边界不一定是语义边界。出版工具可能把一个标题或段落拆成多个
`div`/`span`，也可能用大量短块表达诗行、清单或实验性节奏。导入器因此不能以
“短文本”作为合并条件。

当前实现采用两阶段模型：

```text
XHTML + CSS
    ↓
SourceAtom（原始结构与来源定位）
    ↓
ParsedBlock / Segment（保守重建的翻译单元）
```

## SourceAtom

每个原子保存：

- spine 文件路径和 DOM 路径；
- 最近的语义容器；
- 原始标签、class、样式签名和顺序；
- 标题、正文、块引文、脚注、列表项、表格单元、题注或诗行类型；
- 分页前置标记和出版方续段提示。

混合块内容按直接文本、块级子元素和 child tail 顺序扫描，避免父节点文本因内部包含
`p` 而丢失。外部 CSS、XHTML 内嵌 CSS 和元素 `style` 共同参与常用
tag/class/id/后代选择器的显示类型判断；`display:none` 和
`visibility:hidden` 内容不进入正文。

## 边界策略

结构约束先于语言启发：

- 同一语义容器内被 CSS 拆开的显示块高置信度合并；
- 脚注、列表项、表格单元和诗行默认不跨容器合并；
- 不同语义类型不合并；
- 相邻短标题可以组成复合标题，但仍保留所有来源原子；
- 跨容器正文只有在分页、出版方 continuation class、续接标点、未闭合括号、
  小写续句等证据达到保守阈值时才合并；
- 普通相同 class/style 的短块不足以触发合并，避免破坏实验性文本。

每个边界都产生 action、score、confidence 和 reasons。合并后的 Segment 保存所有
成员原子路径、最低置信度、合并理由和 `segmenter_version`。

## 持久化与兼容

`segments` 表增加：

- `source_refs_json`；
- `segmentation_confidence`；
- `segmentation_reason`；
- `segmenter_version`。

迁移对已有数据库使用非空默认值，TXT/Markdown 导入也通过 ParsedBlock 默认值保持
兼容。API 和 CLI 的 dataclass 序列化会自动暴露这些诊断字段。

## 资源与阅读边界

当前实现保存原始 EPUB、完整成员资源、spine、封面和 DOM/TextNode 映射，并以原 XHTML
模板生成阅读视图。仅译文模式替换对应 atom 的文字与安全内联片段；双语模式在对应
结构位置插入译文，不把正文统一重建为 `h2/p`。

阅读渲染不是像素级 EPUB 回写：流式版面允许因译文长度重新换行和分页，双语会增加
页面高度。fixed-layout 的“忠实排版”保留原坐标并为译文提供溢出滚动，“舒适阅读”
通过高优先级样式解除常见绝对定位。图片型 EPUB 仍需要后续 OCR；复杂出版 CSS 的
computed layout 证据也可继续补充 SourceAtom，但不得绕过现有结构硬约束与安全清洗。
