"use client";

import { useEffect, useState } from "react";

type Mode = "original" | "translated" | "bilingual";

function PageImage({ url, label }: { url: string; label: string }) {
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  return <figure className="pdf-layout-sheet" aria-busy={!ready && !failed}>
    <figcaption>{label}</figcaption>
    {!ready && !failed && <p role="status">正在准备页面…</p>}
    {failed && <p role="alert">页面加载失败，请重新切换页码或下载 PDF 查看。</p>}
    <img src={url} alt={label} onLoad={() => setReady(true)} onError={() => setFailed(true)} />
  </figure>;
}

export function PdfLayoutReader({ apiBase, documentId, page, mode, onPage }: {
  apiBase: string; documentId: string; page: number; mode: Mode; onPage: (page: number) => void;
}) {
  const [zoomed, setZoomed] = useState(false);
  const [total, setTotal] = useState(0);
  const [input, setInput] = useState(String(page));
  const [notes, setNotes] = useState<{ ordinal: number; translation: string }[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBase}/documents/${documentId}/pdf`, { signal: controller.signal })
      .then(async (response) => { if (!response.ok) throw new Error("无法读取 PDF 页码"); return response.json(); })
      .then((result) => setTotal(result.pages.length))
      .catch((caught) => { if (!controller.signal.aborted) setError(caught.message); });
    return () => controller.abort();
  }, [apiBase, documentId]);
  useEffect(() => {
    if (mode === "original") return;
    const controller = new AbortController();
    fetch(`${apiBase}/documents/${documentId}/pdf/pages/${page}/translation-notes`, { signal: controller.signal })
      .then(async (response) => { const data = await response.json(); if (!response.ok) throw new Error(data.detail || "译文排版失败"); return data; })
      .then((result) => setNotes(result.overflow))
      .catch((caught) => { if (!controller.signal.aborted) setError(caught.message); });
    return () => controller.abort();
  }, [apiBase, documentId, page, mode]);
  function jump(value: number) {
    if (Number.isInteger(value) && value >= 1 && value <= total) onPage(value);
    else setInput(String(page));
  }
  const prefix = `${apiBase}/documents/${documentId}/pdf/pages/${page}?width=1200`;
  return <div className="pdf-layout-reader">
    <div className="pdf-page-controls">
      <button disabled={page <= 1} onClick={() => jump(page - 1)} aria-label="上一页">‹</button>
      <form onSubmit={(event) => { event.preventDefault(); jump(Number(input)); }}><label>PDF 第 <input aria-label="阅读页码" type="number" min={1} max={total || 1} value={input} onChange={(event) => setInput(event.target.value)} onBlur={() => jump(Number(input))} /> / {total || "…"} 页</label></form>
      <button disabled={!total || page >= total} onClick={() => jump(page + 1)} aria-label="下一页">›</button>
      <span>保留原页版式与图片 · 未译内容显示原文</span>
      <button onClick={() => setZoomed(!zoomed)} aria-pressed={zoomed}>{zoomed ? "适应宽度" : "放大阅读"}</button>
      <a href={mode === "original" ? `${apiBase}/documents/${documentId}/pdf/original` : `${apiBase}/documents/${documentId}/export?format=book&bilingual=${mode === "bilingual"}`}>{mode === "original" ? "下载原 PDF" : "导出 PDF"}</a>
    </div>
    {error && <p role="alert" className="pdf-layout-error">{error}</p>}
    <div className="pdf-layout-scroll"><div className={`pdf-layout-pages ${mode === "bilingual" ? "paired" : ""} ${zoomed ? "zoomed" : ""}`}>
      {mode !== "translated" && <PageImage key={`${page}-original`} url={`${prefix}&mode=original`} label={`原文 · 第 ${page} 页`} />}
      {mode !== "original" && <PageImage key={`${page}-translated`} url={`${prefix}&mode=translated`} label={`译文 · 第 ${page} 页`} />}
    </div>
    </div>
    {mode !== "original" && notes.length > 0 && <aside className="pdf-overflow-notes"><h3>本页译文续读</h3><p>以下段落无法安全放入原区域，原页已完整保留。导出 PDF 时附在译文续页。</p>{notes.map((note) => <section key={note.ordinal}><small>第 {note.ordinal + 1} 段</small><p>{note.translation}</p></section>)}</aside>}
  </div>;
}
