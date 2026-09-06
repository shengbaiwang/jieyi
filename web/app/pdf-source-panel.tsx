"use client";

import { useEffect, useRef, useState } from "react";

type PdfManifest = {
  pages: { number: number; label: string; text_chars: number }[];
  warnings: string[];
};

export function pdfPages(refs?: string[]): number[] {
  return [...new Set((refs || []).filter((ref) => /^pdf:page:\d+$/.test(ref))
    .map((ref) => Number(ref.split(":")[2])))];
}

export function PdfSourcePanel({ apiBase, documentId, title, initialPage, onClose }: {
  apiBase: string; documentId: string; title: string; initialPage: number; onClose: () => void;
}) {
  const [manifest, setManifest] = useState<PdfManifest | null>(null);
  const [page, setPage] = useState(initialPage);
  const [pageInput, setPageInput] = useState(String(initialPage));
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState("");
  const [zoom, setZoom] = useState(false);
  const close = useRef<HTMLButtonElement>(null);
  const view = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    close.current?.focus();
    return () => previous?.focus();
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBase}/documents/${documentId}/pdf`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("无法读取 PDF 页码，请关闭后重试。");
        return response.json() as Promise<PdfManifest>;
      }).then(setManifest).catch((caught) => {
        if (!controller.signal.aborted) setError(caught.message);
      });
    return () => controller.abort();
  }, [apiBase, documentId]);

  function go(next: number) {
    if (!Number.isInteger(next) || next < 1 || (manifest && next > manifest.pages.length)) {
      setPageInput(String(page));
      return;
    }
    if (next !== page) { setLoaded(false); setError(""); }
    setPage(next);
    setPageInput(String(next));
    view.current?.scrollTo({ top: 0 });
  }
  const current = manifest?.pages[page - 1];
  return <div className="pdf-source-overlay" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    {/* Modal keyboard handling keeps focus inside and supports Escape/page navigation. */}
    {/* eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions */}
    <div role="dialog" tabIndex={-1} aria-modal="true" aria-label="PDF 原页核对" className="pdf-source-panel" onKeyDown={(event) => {
      if (event.key === "Escape") { event.stopPropagation(); onClose(); }
      if (event.key === "Tab") {
        const elements = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('button:not(:disabled), input, a[href]'));
        const first = elements[0], last = elements.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
      if ((event.target as HTMLElement).tagName !== "INPUT" && manifest) {
        if (event.key === "ArrowLeft") { event.preventDefault(); go(page - 1); }
        if (event.key === "ArrowRight") { event.preventDefault(); go(page + 1); }
      }
    }}>
      <header><div><span>PDF 原页 · 核对排版与注释</span><h2>{title}</h2></div><button ref={close} onClick={onClose} aria-label="关闭 PDF 原页">×</button></header>
      <div className="pdf-page-controls">
        <button aria-label="PDF 上一页" disabled={page <= 1} onClick={() => go(page - 1)}>‹</button>
        <form onSubmit={(event) => { event.preventDefault(); go(Number(pageInput)); }}><label>第 <input aria-label="PDF 页码" type="number" min={1} max={manifest?.pages.length} value={pageInput} onChange={(event) => setPageInput(event.target.value)} onBlur={() => go(Number(pageInput))} /> / {manifest?.pages.length || "…"} 页</label></form>
        <button aria-label="PDF 下一页" disabled={!manifest || page >= manifest.pages.length} onClick={() => go(page + 1)}>›</button>
        <button className="pdf-zoom" onClick={() => setZoom(!zoom)}>{zoom ? "适合宽度" : "放大"}</button>
        <a href={`${apiBase}/documents/${documentId}/pdf/original`} download>下载原 PDF</a>
      </div>
      {current && current.label !== String(page) && <div className="pdf-page-label">书内页码 {current.label} · 文件第 {page} 页</div>}
      {current && current.text_chars < 20 && <div className="pdf-page-label">本页文字较少或没有文字层，图片文字尚未识别。</div>}
      <div className="pdf-page-image" ref={view} aria-busy={!loaded && !error}>
        {!loaded && !error && <p role="status">正在加载原页…</p>}
        {error && <p role="alert">{error}</p>}
        <img key={page} className={zoom ? "zoomed" : ""} alt={`${title} · PDF 第 ${page} 页`}
          src={`${apiBase}/documents/${documentId}/pdf/pages/${page}?width=1600`}
          onLoad={() => setLoaded(true)} onError={() => { setError("原页加载失败，可下载原 PDF 查看。"); setLoaded(true); }} />
      </div>
      <footer>原页保留图表与排版；关闭后继续当前段的翻译与校对。</footer>
    </div>
  </div>;
}
