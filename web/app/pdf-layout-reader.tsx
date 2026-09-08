"use client";

import { useEffect, useRef, useState } from "react";

type Mode = "original" | "translated" | "bilingual";
type PdfPage = { number: number; label: string };

function PageImage({ url, label }: { url: string; label: string }) {
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  return <figure className="pdf-layout-sheet" aria-busy={!ready && !failed}>
    <figcaption>{label}</figcaption>
    {!ready && !failed && <p role="status">正在准备页面…</p>}
    {failed && <p role="alert">页面加载失败，请重新切换页码或下载 PDF 查看。</p>}
    <img src={url} alt={label} loading="lazy" onLoad={() => setReady(true)} onError={() => setFailed(true)} />
  </figure>;
}

export function PdfLayoutReader({ apiBase, documentId, page, mode, onPage }: {
  apiBase: string; documentId: string; page: number; mode: Mode; onPage: (page: number) => void;
}) {
  const [zoomed, setZoomed] = useState(false);
  const [pages, setPages] = useState<PdfPage[]>([]);
  const [input, setInput] = useState(String(page));
  const [error, setError] = useState("");
  const [activePage, setActivePage] = useState(page);
  const pageNodes = useRef(new Map<number, HTMLElement>());
  const activePageRef = useRef(page);
  const hasInitialPosition = useRef(false);
  const onPageRef = useRef(onPage);

  useEffect(() => { onPageRef.current = onPage; }, [onPage]);
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBase}/documents/${documentId}/pdf`, { signal: controller.signal })
      .then(async (response) => { if (!response.ok) throw new Error("无法读取 PDF 页码"); return response.json() as Promise<{ pages: PdfPage[] }>; })
      .then((result) => setPages(result.pages))
      .catch((caught) => { if (!controller.signal.aborted) setError(caught.message); });
    return () => controller.abort();
  }, [apiBase, documentId]);

  useEffect(() => {
    if (!pages.length) return;
    const frame = requestAnimationFrame(() => {
      setInput(String(page));
      if (!hasInitialPosition.current || page !== activePageRef.current) {
        hasInitialPosition.current = true;
        activePageRef.current = page;
        setActivePage(page);
        pageNodes.current.get(page)?.scrollIntoView({ block: "start", behavior: "auto" });
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [page, pages.length]);

  useEffect(() => {
    if (!pages.length || typeof IntersectionObserver === "undefined") return;
    const root = pageNodes.current.get(pages[0].number)?.closest<HTMLElement>(".reader-view") || null;
    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((left, right) => right.intersectionRatio - left.intersectionRatio)[0];
      if (!visible) return;
      const next = Number((visible.target as HTMLElement).dataset.pdfPage);
      if (!Number.isInteger(next) || next === activePageRef.current) return;
      activePageRef.current = next;
      setActivePage(next);
      setInput(String(next));
      onPageRef.current(next);
    }, { root, rootMargin: "-72px 0px -45%", threshold: [0, 0.15, 0.5] });
    for (const item of pageNodes.current.values()) observer.observe(item);
    return () => observer.disconnect();
  }, [pages]);

  function jump(value: number) {
    if (Number.isInteger(value) && value >= 1 && value <= pages.length) {
      activePageRef.current = value;
      setActivePage(value);
      setInput(String(value));
      onPageRef.current(value);
      pageNodes.current.get(value)?.scrollIntoView({ block: "start", behavior: "smooth" });
    } else setInput(String(page));
  }

  return <div className="pdf-layout-reader">
    <div className="pdf-page-controls">
      <button disabled={activePage <= 1} onClick={() => jump(activePage - 1)} aria-label="上一页">‹</button>
      <form onSubmit={(event) => { event.preventDefault(); jump(Number(input)); }}><label>PDF 第 <input aria-label="阅读页码" type="number" min={1} max={pages.length || 1} value={input} onChange={(event) => setInput(event.target.value)} onBlur={() => jump(Number(input))} /> / {pages.length || "…"} 页</label></form>
      <button disabled={!pages.length || activePage >= pages.length} onClick={() => jump(activePage + 1)} aria-label="下一页">›</button>
      <span>连续滚动阅读 · 保留原页版式与图片</span>
      <button onClick={() => setZoomed(!zoomed)} aria-pressed={zoomed}>{zoomed ? "适应宽度" : "放大阅读"}</button>
      <a href={mode === "original" ? `${apiBase}/documents/${documentId}/pdf/original` : `${apiBase}/documents/${documentId}/export?format=book&bilingual=${mode === "bilingual"}`}>{mode === "original" ? "下载原 PDF" : "导出 PDF"}</a>
    </div>
    {error && <p role="alert" className="pdf-layout-error">{error}</p>}
    <div className="pdf-layout-scroll"><div className={`pdf-layout-pages ${mode === "bilingual" ? "paired" : ""} ${zoomed ? "zoomed" : ""}`}>
      {pages.map((item) => <section key={item.number} data-pdf-page={item.number} ref={(node) => { if (node) pageNodes.current.set(item.number, node); else pageNodes.current.delete(item.number); }} className="pdf-layout-page-row">
        {mode !== "translated" && <PageImage url={`${apiBase}/documents/${documentId}/pdf/pages/${item.number}?width=1200&mode=original`} label={`原文 · 第 ${item.number} 页${item.label !== String(item.number) ? `（书内 ${item.label}）` : ""}`} />}
        {mode !== "original" && <PageImage url={`${apiBase}/documents/${documentId}/pdf/pages/${item.number}?width=1200&mode=translated`} label={`译文 · 第 ${item.number} 页${item.label !== String(item.number) ? `（书内 ${item.label}）` : ""}`} />}
      </section>)}
    </div></div>
  </div>;
}
