"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { PdfPageSheet, type PdfSelection } from "./pdf-page-sheet";
import { emptyPdfReadingState, exportPdfNotes, parsePdfReadingState, pdfReadingKey, resolvePdfHighlight, visiblePdfChapters,
  type PdfChapter, type PdfHighlight, type PdfLocation, type PdfMode, type PdfReadingState, type PdfSearchHit, type PdfTextPage } from "./pdf-reading";

type Mode = PdfMode | "bilingual";
type PdfPage = { number: number; label: string; width: number; height: number };
type Tab = "chapters" | "search" | "bookmarks" | "annotations" | "notes";
const tabs: { id: Tab; label: string }[] = [{ id: "chapters", label: "目录" }, { id: "search", label: "搜索" }, { id: "bookmarks", label: "书签" }, { id: "annotations", label: "注释" }, { id: "notes", label: "笔记" }];

export function PdfLayoutReader({ apiBase, documentId, title, page, mode, chapters, footnotes, onPage, onMode, onChapter }: {
  apiBase: string; documentId: string; title: string; page: number; mode: Mode; chapters: PdfChapter[];
  footnotes: { id: string; page: number; text: string }[];
  onPage: (page: number) => void; onMode: (mode: Mode) => void; onChapter: (id: string) => void;
}) {
  const [pages, setPages] = useState<PdfPage[]>([]);
  const [input, setInput] = useState(String(page));
  const [activePage, setActivePage] = useState(page);
  const [zoomed, setZoomed] = useState(false);
  const [panelOpen, setPanelOpen] = useState(true);
  const [tab, setTab] = useState<Tab>("chapters");
  const [collapsed, setCollapsed] = useState(new Set<string>());
  const [chosenChapter, setChosenChapter] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const [storageError, setStorageError] = useState("");
  const [reading, setReading] = useState<PdfReadingState>(emptyPdfReadingState);
  const readingRef = useRef(reading);
  const hydrated = useRef(false);
  const storageHealthy = useRef(true);
  const [selection, setSelection] = useState<PdfSelection | null>(null);
  const [noteDraft, setNoteDraft] = useState("");
  const [editingNoteDraft, setEditingNoteDraft] = useState("");
  const [color, setColor] = useState<PdfHighlight["color"]>("yellow");
  const [editingNote, setEditingNote] = useState<string | null>(null);
  const [bookmarkDraft, setBookmarkDraft] = useState("");
  const [texts, setTexts] = useState<Record<string, PdfTextPage>>({});
  const [query, setQuery] = useState("");
  const [searchedQuery, setSearchedQuery] = useState("");
  const [searchMode, setSearchMode] = useState<PdfMode>(mode === "translated" ? "translated" : "original");
  const [searchedMode, setSearchedMode] = useState<PdfMode>("original");
  const [results, setResults] = useState<PdfSearchHit[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [searching, setSearching] = useState(false);
  const searchController = useRef<AbortController | null>(null);
  const searchField = useRef<HTMLInputElement>(null);
  const [target, setTarget] = useState<(PdfSelection & { locate: boolean }) | null>(null);
  const [history, setHistory] = useState<PdfLocation[]>([]);
  const historyRef = useRef<PdfLocation[]>([]);
  const pageNodes = useRef(new Map<number, HTMLElement>());
  const container = useRef<HTMLDivElement>(null);
  const location = useRef<PdfLocation>({ page, offset: 0 });
  const anchor = useRef<PdfLocation | null>(null);
  const initialized = useRef(false);
  const onPageRef = useRef(onPage);
  useEffect(() => { onPageRef.current = onPage; }, [onPage]);

  const save = useCallback((value: PdfReadingState) => {
    const changed = readingRef.current.bookmarks !== value.bookmarks || readingRef.current.highlights !== value.highlights;
    readingRef.current = value;
    if (changed) setReading(value);
    if (!hydrated.current || !storageHealthy.current) return;
    try { localStorage.setItem(pdfReadingKey(documentId), JSON.stringify(value)); }
    catch { setStorageError("浏览器未能保存阅读记录，请先导出笔记，书签和续读记录可能无法保留。"); }
  }, [documentId]);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBase}/documents/${documentId}/pdf`, { signal: controller.signal })
      .then(async (response) => { if (!response.ok) throw new Error("无法读取 PDF 页码，请重新打开阅读模式。"); return response.json() as Promise<{ pages: PdfPage[] }>; })
      .then((result) => {
        let saved = emptyPdfReadingState();
        try { saved = parsePdfReadingState(localStorage.getItem(pdfReadingKey(documentId))); }
        catch { storageHealthy.current = false; setStorageError("已有阅读记录暂时无法读取，已保留原记录；本次新增笔记可先导出。"); }
        hydrated.current = true;
        readingRef.current = saved;
        setReading(saved);
        setPages(result.pages);
        const start = Math.max(1, Math.min(page, result.pages.length));
        location.current = { page: start, offset: saved.location.page === start ? saved.location.offset : 0 };
        anchor.current = location.current;
        setActivePage(start);
        setInput(String(start));
        onPageRef.current(start);
      }).catch((caught) => { if (!controller.signal.aborted) setError(caught.message); });
    return () => { controller.abort(); searchController.current?.abort(); };
  }, [apiBase, documentId]); // eslint-disable-line react-hooks/exhaustive-deps

  const inset = useCallback(() => {
    const root = container.current?.closest<HTMLElement>(".reader-view");
    return (root?.querySelector<HTMLElement>(".reader-toolbar")?.offsetHeight || 0)
      + (container.current?.querySelector<HTMLElement>(".pdf-reader-tools")?.offsetHeight || 0)
      + (window.matchMedia("(max-width: 760px)").matches ? container.current?.querySelector<HTMLElement>(".pdf-reading-panel")?.offsetHeight || 0 : 0) + 12;
  }, []);
  const align = useCallback(() => {
    const destination = anchor.current;
    const root = container.current?.closest<HTMLElement>(".reader-view");
    const row = destination && pageNodes.current.get(destination.page);
    if (!root || !row || !destination) return;
    root.scrollTo({ top: Math.max(0, root.scrollTop + row.getBoundingClientRect().top
      - root.getBoundingClientRect().top + row.getBoundingClientRect().height * destination.offset - inset()), behavior: "instant" });
  }, [inset]);

  const currentLocation = useCallback((): PdfLocation => {
    const root = container.current?.closest<HTMLElement>(".reader-view");
    if (!root || !pages.length) return location.current;
    const line = root.getBoundingClientRect().top + inset();
    // Binary search avoids measuring thousands of sheets on each scroll frame.
    let low = 0, high = pages.length - 1;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      const rect = pageNodes.current.get(pages[middle].number)?.getBoundingClientRect();
      if (rect && rect.bottom <= line) low = middle + 1; else high = middle;
    }
    const number = pages[low].number, rect = pageNodes.current.get(number)?.getBoundingClientRect();
    return { page: number, offset: rect ? Math.max(0, Math.min(1, (line - rect.top) / rect.height)) : 0 };
  }, [inset, pages]);

  const go = useCallback((destination: PdfLocation, remember = true) => {
    if (!pages.length || !Number.isInteger(destination.page) || destination.page < 1 || destination.page > pages.length) {
      setStatus("页码已超出本书范围。"); return;
    }
    if (remember) {
      historyRef.current = [...historyRef.current, currentLocation()].slice(-50);
      setHistory(historyRef.current);
    }
    location.current = destination;
    anchor.current = destination;
    setActivePage(destination.page);
    setInput(String(destination.page));
    onPageRef.current(destination.page);
    save({ ...readingRef.current, location: destination });
    align();
  }, [align, currentLocation, pages.length, save]);

  useEffect(() => {
    if (!pages.length) return;
    if (!initialized.current) { initialized.current = true; align(); return; }
    if (page !== location.current.page) go({ page, offset: 0 });
  }, [align, go, page, pages.length]);

  useEffect(() => {
    const root = container.current?.closest<HTMLElement>(".reader-view");
    if (!root || !pages.length) return;
    let frame = 0, timer = 0;
    const persist = () => save({ ...readingRef.current, location: location.current });
    const track = () => {
      frame = 0;
      if (anchor.current) { align(); return; }
      const next = currentLocation();
      const changed = next.page !== location.current.page;
      location.current = next;
      if (changed) {
        setActivePage(next.page); setInput(String(next.page)); setChosenChapter(null); onPageRef.current(next.page);
      }
      window.clearTimeout(timer); timer = window.setTimeout(persist, 200);
    };
    const schedule = () => { if (!frame) frame = requestAnimationFrame(track); };
    const release = () => { anchor.current = null; schedule(); };
    const keydown = (event: KeyboardEvent) => {
      const editable = (event.target as HTMLElement).closest("input, textarea, select, [contenteditable=true]");
      if (!editable && ["ArrowDown", "ArrowUp", "PageDown", "PageUp", "Home", "End", " "].includes(event.key)) release();
    };
    const pointerdown = (event: PointerEvent) => { if (!(event.target as HTMLElement).closest(".pdf-reader-tools, .pdf-reading-panel")) release(); };
    const updateInsets = () => {
      const toolbarHeight = root.querySelector<HTMLElement>(".reader-toolbar")?.offsetHeight || 0;
      const toolsHeight = container.current?.querySelector<HTMLElement>(".pdf-reader-tools")?.offsetHeight || 0;
      container.current?.style.setProperty("--pdf-toolbar-top", `${toolbarHeight}px`);
      container.current?.style.setProperty("--pdf-panel-top", `${toolbarHeight + toolsHeight}px`);
      if (anchor.current) align(); else schedule();
    };
    const resize = new ResizeObserver(updateInsets);
    resize.observe(root);
    const toolbar = root.querySelector(".reader-toolbar"), tools = container.current?.querySelector(".pdf-reader-tools"), panel = container.current?.querySelector(".pdf-reading-panel");
    if (toolbar) resize.observe(toolbar);
    if (tools) resize.observe(tools);
    if (panel) resize.observe(panel);
    updateInsets();
    for (const row of pageNodes.current.values()) resize.observe(row);
    root.addEventListener("scroll", schedule, { passive: true });
    root.addEventListener("wheel", release, { passive: true });
    root.addEventListener("touchstart", release, { passive: true });
    root.addEventListener("pointerdown", pointerdown);
    root.addEventListener("keydown", keydown);
    window.addEventListener("pagehide", persist);
    return () => {
      persist(); cancelAnimationFrame(frame); window.clearTimeout(timer); resize.disconnect();
      root.removeEventListener("scroll", schedule); root.removeEventListener("wheel", release);
      root.removeEventListener("touchstart", release); root.removeEventListener("pointerdown", pointerdown);
      root.removeEventListener("keydown", keydown); window.removeEventListener("pagehide", persist);
    };
  }, [align, currentLocation, pages, panelOpen, save]);

  useEffect(() => {
    const focusSearch = () => {
      setPanelOpen(true); setTab("search");
      window.setTimeout(() => searchField.current?.focus(), 0);
    };
    const openSearch = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "f") {
        event.preventDefault(); focusSearch();
      }
      if (event.key === "Escape") { setSelection(null); setEditingNote(null); }
    };
    window.addEventListener("keydown", openSearch);
    window.addEventListener("jieyi-pdf-search", focusSearch);
    return () => { window.removeEventListener("keydown", openSearch); window.removeEventListener("jieyi-pdf-search", focusSearch); };
  }, []);

  useEffect(() => {
    if (!target?.locate) return;
    const text = texts[`${target.page}-${target.mode}`];
    if (!text || (mode !== "bilingual" && mode !== target.mode)) return;
    const range = resolvePdfHighlight(text.text, target);
    const line = range && text.lines.find((line) => line.end > range.start);
    const frame = requestAnimationFrame(() => {
      if (line) go({ page: target.page, offset: line.rect[1] }, false);
      else setStatus("已显示所在页；原文或译文已变化，无法唯一定位这段选文。");
      setTarget({ ...target, locate: false });
    });
    return () => cancelAnimationFrame(frame);
  }, [go, mode, target, texts]);

  async function search(event?: FormEvent, nextOffset = 0) {
    event?.preventDefault();
    const term = nextOffset ? searchedQuery : query.trim(), variant = nextOffset ? searchedMode : searchMode;
    if (!term) return;
    searchController.current?.abort();
    const controller = new AbortController(); searchController.current = controller;
    setSearching(true); setError("");
    if (!nextOffset) { setResults([]); setTotal(null); setTarget(null); }
    try {
      const response = await fetch(`${apiBase}/documents/${documentId}/pdf/search?q=${encodeURIComponent(term)}&mode=${variant}&offset=${nextOffset}`, { signal: controller.signal });
      if (!response.ok) throw new Error("全文搜索失败，请重试。");
      const value = await response.json() as { results: PdfSearchHit[]; total: number };
      if (controller.signal.aborted) return;
      setSearchedQuery(term); setSearchedMode(variant); setResults((current) => nextOffset ? [...current, ...value.results] : value.results); setTotal(value.total);
    } catch (caught) { if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : "搜索失败"); }
    finally { if (!controller.signal.aborted) setSearching(false); }
  }
  function locate(item: PdfSelection) {
    setSelection(null); setTarget({ ...item, locate: true });
    if (mode !== "bilingual" && mode !== item.mode) onMode(item.mode);
    go({ page: item.page, offset: 0 });
  }
  function addBookmark() {
    const here = currentLocation();
    save({ ...readingRef.current, bookmarks: [...readingRef.current.bookmarks, { ...here,
      id: crypto.randomUUID(), label: bookmarkDraft.trim() || `第 ${here.page} 页`, createdAt: new Date().toISOString() }] });
    setBookmarkDraft(""); setStatus("已保存书签，可在书签列表重命名或删除。");
  }
  function addHighlight(withNote: boolean) {
    if (!selection) return;
    const marks: PdfHighlight[] = (selection.parts || [selection]).map((part) => ({ page: part.page, mode: part.mode, start: part.start, end: part.end, quote: part.quote,
      id: crypto.randomUUID(), color, note: withNote ? noteDraft.trim() : "", createdAt: new Date().toISOString() }));
    save({ ...readingRef.current, highlights: [...readingRef.current.highlights, ...marks] });
    setSelection(null); setNoteDraft(""); window.getSelection()?.removeAllRanges();
    if (withNote) { setTab("notes"); setPanelOpen(true); }
    setStatus(withNote ? "高亮和笔记已保存。" : "高亮已保存。选中内容可在笔记汇总查看。");
  }
  async function copySelection() {
    if (!selection) return;
    try { await navigator.clipboard.writeText(selection.quote); setStatus("选文已复制。"); }
    catch { setStatus("复制未完成，请在页面重新选文后使用 Ctrl/Cmd+C。"); }
  }
  function exportNotes() {
    const url = URL.createObjectURL(new Blob([exportPdfNotes(title, readingRef.current.highlights)], { type: "text/markdown;charset=utf-8" }));
    const link = document.createElement("a"); link.href = url; link.download = `${title.replace(/[\\/:*?"<>|]/g, "_")}-阅读笔记.md`; link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function back() {
    const previous = historyRef.current.at(-1);
    if (!previous) return;
    historyRef.current = historyRef.current.slice(0, -1); setHistory(historyRef.current); setTarget(null);
    go(previous, false);
  }
  const captureText = useCallback((text: PdfTextPage) => setTexts((current) => ({ ...current, [`${text.page}-${text.mode}`]: text })), []);
  const visibleChapters = visiblePdfChapters(chapters, collapsed);
  const activeChapter = chosenChapter || [...chapters].reverse().find((item) => item.page <= activePage)?.id;
  const annotations = useMemo(() => Object.values(texts).filter((item) => item.mode === "original")
    .flatMap((item) => item.annotations.map((annotation, index) => ({ ...annotation, page: item.page, id: `${item.page}-${index}` }))).sort((a, b) => a.page - b.page), [texts]);
  const notes = [...reading.highlights].sort((a, b) => a.page - b.page || a.start - b.start);
  const progress = pages.length ? Math.round(activePage / pages.length * 100) : 0;

  return <div ref={container} className={`pdf-layout-reader ${panelOpen ? "with-panel" : ""}`} onCopy={(event) => {
    const native = window.getSelection();
    if (!selection || !native?.rangeCount || native.isCollapsed) return;
    const range = native.getRangeAt(0);
    const textNode = range.startContainer.parentElement?.closest(".pdf-text-layer");
    if (textNode && container.current?.contains(textNode)) { event.clipboardData.setData("text/plain", selection.quote); event.preventDefault(); }
  }}>
    <div className="pdf-reader-tools">
      <div className="pdf-page-controls">
        <button onClick={() => { anchor.current = currentLocation(); setPanelOpen(!panelOpen); }} aria-expanded={panelOpen} aria-controls="pdf-reading-panel">{panelOpen ? "收起工具" : "阅读工具"}</button>
        <button disabled={!history.length} onClick={back} title="返回搜索、注释或书签跳转前的位置">↶ 返回正文</button>
        <button disabled={activePage <= 1} onClick={() => { setTarget(null); go({ page: activePage - 1, offset: 0 }); }} aria-label="上一页">‹</button>
        <form onSubmit={(event) => { event.preventDefault(); const next = Number(input); if (Number.isInteger(next) && next >= 1 && next <= pages.length) { setTarget(null); go({ page: next, offset: 0 }); } else { setInput(String(activePage)); setStatus("请输入有效页码。"); } }}>
          <label>第 <input aria-label="阅读页码" type="number" min={1} max={pages.length || 1} value={input} onChange={(event) => setInput(event.target.value)} /> / {pages.length || "…"} 页</label>
        </form>
        <button disabled={!pages.length || activePage >= pages.length} onClick={() => { setTarget(null); go({ page: activePage + 1, offset: 0 }); }} aria-label="下一页">›</button>
        <span>{progress}% · 自动续读</span>
        <button onClick={addBookmark} disabled={!pages.length}>＋ 书签</button>
        <button onClick={() => { anchor.current = currentLocation(); setZoomed(!zoomed); }} aria-pressed={zoomed}>{zoomed ? "适应宽度" : "放大阅读"}</button>
        <a href={mode === "original" ? `${apiBase}/documents/${documentId}/pdf/original` : `${apiBase}/documents/${documentId}/export?format=book&bilingual=${mode === "bilingual"}`}>{mode === "original" ? "下载原 PDF" : "导出 PDF"}</a>
      </div>
      {selection && <div className="pdf-selection-tools" role="group" aria-label="选文操作">
        <span>已选 {selection.quote.length} 字 · 第 {selection.page} 页</span><button onClick={() => void copySelection()}>复制选文</button>
        <label>颜色 <select aria-label="高亮颜色" value={color} onChange={(event) => setColor(event.target.value as PdfHighlight["color"])}><option value="yellow">黄色</option><option value="green">绿色</option><option value="blue">蓝色</option></select></label>
        <button onClick={() => addHighlight(false)}>高亮</button><input aria-label="选文笔记" placeholder="为选文写一条笔记…" value={noteDraft} onChange={(event) => setNoteDraft(event.target.value)} />
        <button onClick={() => addHighlight(true)}>保存笔记</button><button onClick={() => { setSelection(null); setNoteDraft(""); }} aria-label="关闭选文操作">×</button>
      </div>}
      {status && <p className="pdf-reader-status" role="status">{status}</p>}
      {storageError && <p className="pdf-layout-error" role="alert">{storageError}</p>}
      {error && <p className="pdf-layout-error" role="alert">{error}</p>}
    </div>
    <div className="pdf-reader-body">
      {panelOpen && <aside id="pdf-reading-panel" className="pdf-reading-panel" aria-label="PDF 阅读工具">
        <div className="pdf-reading-tabs" role="tablist" aria-label="阅读工具">
          {tabs.map((item) => <button key={item.id} role="tab" id={`pdf-tab-${item.id}`} aria-controls={`pdf-panel-${item.id}`} aria-selected={tab === item.id}
            onClick={() => setTab(item.id)}>{item.label}{item.id === "notes" && reading.highlights.length ? ` ${reading.highlights.length}` : ""}</button>)}
        </div>
        <div className="pdf-reading-panel-content" role="tabpanel" id={`pdf-panel-${tab}`} aria-labelledby={`pdf-tab-${tab}`}>
          {tab === "chapters" && <>
            <div className="pdf-panel-heading"><strong>章节目录</strong><button onClick={() => setCollapsed(collapsed.size ? new Set() : new Set(chapters.filter((item, index) => (chapters[index + 1]?.level ?? -1) > item.level).map((item) => item.id)))}>{collapsed.size ? "全部展开" : "全部折叠"}</button></div>
            {!chapters.length && <p className="pdf-panel-empty">本书没有章节目录，可使用页码或全文搜索定位。</p>}
            <nav aria-label="PDF 章节目录">{visibleChapters.map((chapter) => {
              const index = chapters.findIndex((item) => item.id === chapter.id), hasChildren = (chapters[index + 1]?.level ?? -1) > chapter.level;
              return <div key={chapter.id} className={`pdf-chapter-row ${activeChapter === chapter.id ? "active" : ""}`} style={{ paddingLeft: `${Math.min(chapter.level, 8) * 14}px` }}>
                {hasChildren ? <button className="pdf-chapter-toggle" aria-label={`${collapsed.has(chapter.id) ? "展开" : "折叠"}${chapter.title}`} aria-expanded={!collapsed.has(chapter.id)} onClick={() => setCollapsed((current) => { const next = new Set(current); if (next.has(chapter.id)) next.delete(chapter.id); else next.add(chapter.id); return next; })}>{collapsed.has(chapter.id) ? "▸" : "▾"}</button> : <span className="pdf-chapter-spacer" />}
                <button aria-current={activeChapter === chapter.id ? "location" : undefined} onClick={() => { setTarget(null); go({ page: chapter.page, offset: 0 }); setChosenChapter(chapter.id); onChapter(chapter.id); }}><strong>{chapter.title}</strong><small>第 {chapter.page} 页</small></button>
              </div>;
            })}</nav>
          </>}
          {tab === "search" && <>
            <form className="pdf-search-form" onSubmit={(event) => void search(event)}>
              <label>全文搜索<input ref={searchField} aria-label="PDF 全文搜索" placeholder="输入关键词 · Ctrl/Cmd+F" value={query} onChange={(event) => setQuery(event.target.value)} /></label>
              <div><select aria-label="搜索范围" value={searchMode} onChange={(event) => setSearchMode(event.target.value as PdfMode)}><option value="original">原文</option><option value="translated">译文（含未译原文）</option></select><button disabled={!query.trim() || searching} type="submit">{searching ? "搜索中…" : "搜索"}</button></div>
            </form>
            <p className="pdf-panel-hint">{total === null ? "搜索整本书，点击结果回到原页。扫描图片中的文字须先进行 OCR。" : `找到 ${total} 处，已显示 ${results.length} 处`}</p>
            {total === 0 && <p className="pdf-panel-empty">没有找到匹配文字。</p>}
            <ol className="pdf-search-results">{results.map((item, index) => <li key={index}><button className={target?.page === item.page && target.start === item.start && target.mode === item.mode ? "active" : ""}
              onClick={() => locate(item)}><strong>第 {item.page} 页 · {item.mode === "original" ? "原文" : "译文"}</strong><span>{item.snippet}</span></button></li>)}</ol>
            {total !== null && results.length < total && <button disabled={searching} onClick={() => void search(undefined, results.length)}>{searching ? "搜索中…" : "加载更多结果"}</button>}
          </>}
          {tab === "bookmarks" && <>
            <div className="pdf-panel-heading"><strong>个人书签 · {reading.bookmarks.length}</strong></div>
            <form className="pdf-bookmark-form" onSubmit={(event) => { event.preventDefault(); addBookmark(); }}><input aria-label="新书签名称" value={bookmarkDraft} onChange={(event) => setBookmarkDraft(event.target.value)} placeholder={`第 ${activePage} 页的书签名称`} /><button disabled={!pages.length}>添加</button></form>
            {!reading.bookmarks.length && <p className="pdf-panel-empty">在读到的位置添加书签，之后可一键回到这一页的具体位置。</p>}
            {[...reading.bookmarks].sort((a, b) => a.page - b.page || a.offset - b.offset).map((item) => <div className="pdf-saved-card" key={item.id}>
              <input aria-label={`重命名第 ${item.page} 页书签`} value={item.label} onChange={(event) => save({ ...readingRef.current, bookmarks: readingRef.current.bookmarks.map((bookmark) => bookmark.id === item.id ? { ...bookmark, label: event.target.value } : bookmark) })} />
              <div><button onClick={() => { setTarget(null); go(item); }}>第 {item.page} 页 ↗</button><button aria-label={`删除书签 ${item.label}`} onClick={() => save({ ...readingRef.current, bookmarks: readingRef.current.bookmarks.filter((bookmark) => bookmark.id !== item.id) })}>删除</button></div>
            </div>)}
            <p className="pdf-panel-hint">阅读位置、书签和笔记保存在当前浏览器。再次打开本书会恢复阅读位置。</p>
          </>}
          {tab === "annotations" && <>
            <div className="pdf-panel-heading"><strong>原书注释与脚注</strong></div>
            <p className="pdf-panel-hint">点击原页内部链接或下方注释跳转；用“返回正文”回到跳转前的位置。原页批注随页面加载。</p>
            {!footnotes.length && !annotations.length && <p className="pdf-panel-empty">尚未发现原书批注或脚注。带内部链接的注释可直接点击原页跳转。</p>}
            {footnotes.map((item) => <button className="pdf-annotation-card" key={item.id} onClick={() => locate({ page: item.page, mode: "original", quote: item.text, start: -1, end: -1 })}><strong>第 {item.page} 页 · 脚注</strong><span>{item.text}</span></button>)}
            {annotations.map((item) => <button className="pdf-annotation-card" key={item.id} onClick={() => { if (mode === "translated") onMode("original"); go({ page: item.page, offset: item.rect[1] }); }}><strong>第 {item.page} 页 · 原页批注</strong><span>{item.text}</span></button>)}
          </>}
          {tab === "notes" && <>
            <div className="pdf-panel-heading"><strong>高亮与笔记 · {notes.length}</strong><button disabled={!notes.length} onClick={exportNotes}>导出 Markdown</button></div>
            {!notes.length && <p className="pdf-panel-empty">在页面选中文字后添加高亮或笔记。汇总按页码排列，每条都可跳回选文。</p>}
            {notes.map((item) => {
              const text = texts[`${item.page}-${item.mode}`], missing = text && !resolvePdfHighlight(text.text, item);
              return <article className={`pdf-saved-card pdf-note-card pdf-note-${item.color}`} key={item.id}>
                <button className="pdf-note-location" onClick={() => locate(item)}>第 {item.page} 页 · {item.mode === "original" ? "原文" : "译文"} ↗</button><blockquote>{item.quote}</blockquote>
                {missing && <p className="pdf-panel-hint">正文已变化，选文已保留，请在原页核对。</p>}
                {editingNote === item.id ? <form onSubmit={(event) => { event.preventDefault(); save({ ...readingRef.current, highlights: readingRef.current.highlights.map((mark) => mark.id === item.id ? { ...mark, note: editingNoteDraft.trim() } : mark) }); setEditingNote(null); setEditingNoteDraft(""); }}><textarea aria-label="编辑笔记" value={editingNoteDraft} onChange={(event) => setEditingNoteDraft(event.target.value)} /><div><button type="submit">保存</button><button type="button" onClick={() => setEditingNote(null)}>取消</button></div></form> : <>{item.note && <p className="pdf-note-text">{item.note}</p>}<div><button onClick={() => { setEditingNote(item.id); setEditingNoteDraft(item.note); }}>{item.note ? "编辑笔记" : "添加笔记"}</button><button aria-label={`删除第 ${item.page} 页高亮与笔记`} onClick={() => save({ ...readingRef.current, highlights: readingRef.current.highlights.filter((mark) => mark.id !== item.id) })}>删除</button></div></>}
              </article>;
            })}
          </>}
        </div>
      </aside>}
      <div className="pdf-layout-scroll"><div className={`pdf-layout-pages ${mode === "bilingual" ? "paired" : ""} ${zoomed ? "zoomed" : ""}`}>
        {pages.map((item) => <section key={item.number} data-pdf-page={item.number} ref={(node) => { if (node) pageNodes.current.set(item.number, node); else pageNodes.current.delete(item.number); }} className="pdf-layout-page-row">
          {(["original", "translated"] as const).filter((variant) => mode === "bilingual" || variant === mode).map((variant) => <PdfPageSheet key={`${item.number}-${variant}`} apiBase={apiBase} documentId={documentId} number={item.number} label={item.label} mode={variant} width={item.width || 600} height={item.height || 800}
            highlights={reading.highlights.filter((mark) => mark.page === item.number && mark.mode === variant)} query={searchedMode === variant ? searchedQuery : ""}
            target={target?.page === item.number && target.mode === variant ? target : undefined} onSelect={(value) => { if (value) anchor.current = currentLocation(); setSelection(value); setNoteDraft(""); }}
            onLink={(destination) => { setTab("annotations"); setPanelOpen(true); setTarget(null); go(destination); }} onText={captureText} />)}
        </section>)}
      </div></div>
    </div>
  </div>;
}
