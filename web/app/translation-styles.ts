export const STYLE_PRESETS = [
  { id: "academic", label: "学术严谨", note: "概念稳定，论证清楚", guide: "忠实原意，保持严谨、克制的学术表达；统一核心概念译法，保留引文、脚注、专名及论证层次。" },
  { id: "literary", label: "文学自然", note: "保留声调与节奏", guide: "准确传达原意与人物声调，译文自然流畅并保留文学节奏、意象与修辞；避免生硬直译。" },
  { id: "popular", label: "通俗易读", note: "清晰、顺畅、少术语", guide: "在不损失关键信息的前提下使用清晰、自然、易读的现代语言；必要术语首次出现时给出简短说明。" },
  { id: "faithful", label: "忠实直译", note: "贴近句法与措辞", guide: "尽量贴近原文句法、措辞和段落结构，不擅自增删或改写；歧义处保留原文的开放性。" },
] as const;
