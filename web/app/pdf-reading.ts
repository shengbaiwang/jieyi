export type PdfMode = "original" | "translated";
export type PdfLocation = { page: number; offset: number };
export type PdfBookmark = PdfLocation & { id: string; label: string; createdAt: string };
export type PdfHighlight = {
  id: string; page: number; mode: PdfMode; start: number; end: number; quote: string;
  color: "yellow" | "green" | "blue"; note: string; createdAt: string;
};
export type PdfReadingState = {
  version: 1; location: PdfLocation; bookmarks: PdfBookmark[]; highlights: PdfHighlight[];
};
export type PdfTextLine = { text: string; start: number; end: number; rect: [number, number, number, number] };
export type PdfTextPage = {
  page: number; mode: PdfMode; width: number; height: number; text: string; lines: PdfTextLine[];
  links: (PdfLocation & { rect: [number, number, number, number] })[];
  annotations: { text: string; rect: [number, number, number, number] }[];
};
export type PdfChapter = { id: string; title: string; level: number; page: number };
export type PdfSearchHit = { page: number; mode: PdfMode; quote: string; snippet: string; start: number; end: number };
export const pdfReadingKey = (documentId: string) => `jieyi.pdf.reader.${documentId}`;
export const emptyPdfReadingState = (): PdfReadingState => ({ version: 1, location: { page: 1, offset: 0 }, bookmarks: [], highlights: [] });

export function validPdfLocation(value: unknown): value is PdfLocation {
  if (!value || typeof value !== "object") return false;
  const item = value as PdfLocation;
  return Number.isInteger(item.page) && item.page >= 1 && Number.isFinite(item.offset) && item.offset >= 0 && item.offset <= 1;
}

export function parsePdfReadingState(raw: string | null): PdfReadingState {
  if (!raw) return emptyPdfReadingState();
  const value = JSON.parse(raw) as PdfReadingState;
  if (value?.version !== 1 || !validPdfLocation(value.location)
    || !Array.isArray(value.bookmarks) || !Array.isArray(value.highlights)) throw new Error("阅读记录格式无效");
  return {
    version: 1, location: value.location,
    bookmarks: value.bookmarks.filter((item) => validPdfLocation(item) && typeof item.id === "string"
      && typeof item.label === "string" && typeof item.createdAt === "string"),
    highlights: value.highlights.filter((item) => item && typeof item.id === "string"
      && Number.isInteger(item.page) && item.page >= 1 && ["original", "translated"].includes(item.mode)
      && Number.isInteger(item.start) && item.start >= 0 && Number.isInteger(item.end) && item.end > item.start
      && typeof item.quote === "string" && item.quote.length > 0 && typeof item.note === "string"
      && ["yellow", "green", "blue"].includes(item.color) && typeof item.createdAt === "string"),
  };
}

export function visiblePdfChapters(chapters: PdfChapter[], collapsed: Set<string>) {
  let hiddenBelow: number | null = null;
  return chapters.filter((chapter) => {
    if (hiddenBelow !== null && chapter.level > hiddenBelow) return false;
    hiddenBelow = collapsed.has(chapter.id) ? chapter.level : null;
    return true;
  });
}

// Literal matching tolerates PDF line breaks, preserving offsets into the text layer.
export function pdfTextMatches(text: string, query: string): { start: number; end: number }[] {
  const parts = query.trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return [];
  const pattern = new RegExp(parts.map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\s+"), "giu");
  return [...text.matchAll(pattern)].map((match) => ({ start: match.index!, end: match.index! + match[0].length }));
}

export function resolvePdfHighlight(text: string, mark: Pick<PdfHighlight, "start" | "end" | "quote">) {
  if (text.slice(mark.start, mark.end) === mark.quote) return { start: mark.start, end: mark.end };
  // Translations can change after a mark was made. Only relocate unambiguous quotes.
  const matches = pdfTextMatches(text, mark.quote);
  return matches.length === 1 ? matches[0] : null;
}

export function pdfRangeRects(lines: PdfTextLine[], range: { start: number; end: number }) {
  return lines.flatMap((line) => {
    const start = Math.max(range.start, line.start), end = Math.min(range.end, line.end);
    if (start >= end || !line.text.length) return [];
    const [x, y, width, height] = line.rect;
    return [[x + width * (start - line.start) / line.text.length, y,
      width * (end - start) / line.text.length, height] as [number, number, number, number]];
  });
}

export function exportPdfNotes(title: string, highlights: PdfHighlight[]) {
  const ordered = [...highlights].sort((a, b) => a.page - b.page || a.start - b.start);
  return `# ${title} · 阅读笔记\n\n` + ordered.map((item) =>
    `## 第 ${item.page} 页 · ${item.mode === "original" ? "原文" : "译文"}\n\n`
    + item.quote.split(/\r?\n/).map((line) => `> ${line}`).join("\n")
    + (item.note ? `\n\n${item.note}` : "") + "\n").join("\n");
}
