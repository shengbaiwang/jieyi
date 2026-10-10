"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import { pdfRangeRects, pdfTextMatches, resolvePdfHighlight, type PdfHighlight, type PdfLocation, type PdfMode, type PdfTextPage } from "./pdf-reading";

type SelectionPart = { page: number; mode: PdfMode; start: number; end: number; quote: string };
export type PdfSelection = SelectionPart & { parts?: SelectionPart[] };
const rectStyle = ([x, y, width, height]: number[]): CSSProperties => ({ left: `${x * 100}%`, top: `${y * 100}%`, width: `${width * 100}%`, height: `${height * 100}%` });

export function PdfPageSheet({ apiBase, documentId, number, label, mode, width, height, highlights, query, target, onSelect, onLink, onText }: {
  apiBase: string; documentId: string; number: number; label: string; mode: PdfMode; width: number; height: number;
  highlights: PdfHighlight[]; query: string; target?: { start: number; end: number; quote: string };
  onSelect: (selection: PdfSelection | null) => void; onLink: (location: PdfLocation) => void;
  onText: (page: PdfTextPage) => void;
}) {
  const figure = useRef<HTMLElement>(null);
  const layer = useRef<HTMLDivElement>(null);
  const [near, setNear] = useState(false);
  const [textPage, setTextPage] = useState<PdfTextPage | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const onTextRef = useRef(onText);
  useEffect(() => { onTextRef.current = onText; }, [onText]);
  useEffect(() => {
    const node = figure.current;
    if (!node) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((item) => item.isIntersecting)) { setNear(true); observer.disconnect(); }
    }, { root: node.closest(".reader-view"), rootMargin: "900px" });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!near) return;
    const controller = new AbortController();
    fetch(`${apiBase}/documents/${documentId}/pdf/pages/${number}/text?mode=${mode}`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("文字层加载失败，可重试或下载 PDF。");
        return response.json() as Promise<PdfTextPage>;
      }).then((result) => { setTextPage(result); onTextRef.current(result); })
      .catch((caught) => { if (!controller.signal.aborted) setError(caught.message); });
    return () => controller.abort();
  }, [apiBase, documentId, mode, near, number, retry]);
  useEffect(() => {
    const node = layer.current;
    if (!node || !textPage) return;
    const resize = () => {
      const renderedWidth = node.getBoundingClientRect().width;
      for (const line of node.querySelectorAll<HTMLElement>(".pdf-text-line")) {
        line.style.fontSize = `${Number(line.dataset.fontRatio) * renderedWidth}px`;
        const inner = line.firstElementChild as HTMLElement;
        inner.style.transform = "none";
        const actual = inner.getBoundingClientRect().width;
        if (actual > 0) inner.style.transform = `scaleX(${line.getBoundingClientRect().width / actual})`;
      }
    };
    const observer = new ResizeObserver(resize);
    observer.observe(node);
    resize();
    return () => observer.disconnect();
  }, [textPage]);

  function selectText() {
    const selection = window.getSelection();
    if (!selection?.rangeCount || selection.isCollapsed || !textPage) { onSelect(null); return; }
    const range = selection.getRangeAt(0);
    if (!layer.current?.contains(range.startContainer) || !layer.current.contains(range.endContainer)) {
      const root = layer.current?.closest(".pdf-layout-reader");
      if (!root?.contains(range.startContainer) || !root.contains(range.endContainer)) { onSelect(null); return; }
      const parts: SelectionPart[] = [];
      for (const textLayer of root.querySelectorAll<HTMLElement>(".pdf-text-layer")) {
        const selected = Array.from(textLayer.querySelectorAll<HTMLElement>(".pdf-text-line"))
          .filter((line) => range.intersectsNode(line));
        if (!selected.length) continue;
        const first = selected[0], last = selected.at(-1)!;
        const start = first.contains(range.startContainer) ? offset(range.startContainer, range.startOffset) : Number(first.dataset.start);
        const end = last.contains(range.endContainer) ? offset(range.endContainer, range.endOffset) : Number(last.dataset.end);
        if (start === null || end === null || start >= end) continue;
        const quote = selected.map((line) => (line.firstElementChild?.textContent || "").slice(Math.max(0, start - Number(line.dataset.start)), Math.min(Number(line.dataset.end), end) - Number(line.dataset.start))).join("\n");
        if (quote.trim()) parts.push({ page: Number(textLayer.dataset.pdfTextPage), mode: textLayer.dataset.pdfTextMode as PdfMode, start, end, quote });
      }
      if (parts.length) onSelect({ ...parts[0], quote: parts.map((part) => part.quote).join("\n"), parts });
      else onSelect(null);
      return;
    }
    function offset(node: Node, at: number) {
      const line = (node.nodeType === Node.ELEMENT_NODE ? node as HTMLElement : node.parentElement)?.closest<HTMLElement>(".pdf-text-line");
      if (!line) return null;
      const before = document.createRange();
      before.selectNodeContents(line);
      before.setEnd(node, at);
      return Number(line.dataset.start) + before.toString().length;
    }
    const start = offset(range.startContainer, range.startOffset), end = offset(range.endContainer, range.endOffset);
    if (start === null || end === null || start >= end) return;
    const quote = textPage.text.slice(start, end);
    if (quote.trim()) onSelect({ page: number, mode, start, end, quote });
  }
  const matches = textPage ? pdfTextMatches(textPage.text, query) : [];
  const targetRange = textPage && target ? resolvePdfHighlight(textPage.text, target) : null;
  const marks = textPage ? highlights.flatMap((mark) => {
    const range = resolvePdfHighlight(textPage.text, mark);
    return range ? pdfRangeRects(textPage.lines, range).map((rect) => ({ rect, color: mark.color, id: mark.id })) : [];
  }) : [];
  const pageLabel = `${mode === "original" ? "原文" : "译文"} · 第 ${number} 页${label !== String(number) ? `（书内 ${label}）` : ""}`;

  return <figure ref={figure} className="pdf-layout-sheet" aria-label={pageLabel}>
    <figcaption>{pageLabel}</figcaption>
    <div className="pdf-page-surface" style={{ aspectRatio: `${textPage?.width || width} / ${textPage?.height || height}` }}>
      {near && <img src={`${apiBase}/documents/${documentId}/pdf/pages/${number}?width=1200&mode=${mode}&retry=${retry}`} alt={pageLabel} draggable={false}
        onLoad={() => setLoaded(true)} onError={() => setError("页面加载失败，可重试或下载 PDF。")} />}
      {!loaded && !error && <p className="pdf-sheet-status" role="status">正在准备页面…</p>}
      {textPage && <>
        <div className="pdf-mark-layer" aria-hidden="true">
          {marks.map((mark, index) => <span key={`${mark.id}-${index}`} className={`pdf-highlight-${mark.color}`} style={rectStyle(mark.rect)} />)}
          {matches.flatMap((range) => pdfRangeRects(textPage.lines, range)).map((rect, index) => <span key={`search-${index}`} className="pdf-search-mark" style={rectStyle(rect)} />)}
          {targetRange && pdfRangeRects(textPage.lines, targetRange).map((rect, index) => <span key={`target-${index}`} className="pdf-target-mark" style={rectStyle(rect)} />)}
        </div>
        {/* Native selectable PDF text needs pointer and keyboard selection listeners. */}
        {/* eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions */}
        <div ref={layer} className="pdf-text-layer" data-pdf-text-page={number} data-pdf-text-mode={mode} onPointerUp={selectText} onKeyUp={selectText} role="group" aria-label={`${pageLabel} · 可选文字`}>
          {textPage.lines.map((line, index) => <span key={index} className="pdf-text-line" data-start={line.start} data-end={line.end} data-font-ratio={line.rect[3] * textPage.height / textPage.width}
            style={rectStyle(line.rect)}><span>{line.text}</span>{"\n"}</span>)}
        </div>
        {textPage.links.map((link, index) => <button key={`link-${index}`} className="pdf-internal-link" style={rectStyle(link.rect)}
          aria-label={`跳至 PDF 第 ${link.page} 页，支持返回正文`} title={`第 ${link.page} 页 · 内部链接`}
          onClick={() => onLink(link)} />)}
        {textPage.annotations.map((annotation, index) => <button key={`annotation-${index}`} className="pdf-annotation-pin" style={{ left: `${annotation.rect[0] * 100}%`, top: `${annotation.rect[1] * 100}%` }}
          aria-label={`查看注释：${annotation.text.slice(0, 50)}`} title={annotation.text} onClick={() => onLink({ page: number, offset: annotation.rect[1] })}>✎</button>)}
      </>}
    </div>
    {error && <p role="alert" className="pdf-sheet-error">{error} <button onClick={() => { setError(""); setRetry((value) => value + 1); }}>重试</button></p>}
    {textPage && !textPage.text.trim() && <p className="pdf-text-unavailable">本页没有可选文字；扫描图片需要先进行 OCR。</p>}
  </figure>;
}
