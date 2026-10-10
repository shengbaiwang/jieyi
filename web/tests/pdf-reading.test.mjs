import assert from "node:assert/strict";
import test from "node:test";
import { emptyPdfReadingState, exportPdfNotes, parsePdfReadingState, pdfRangeRects, pdfReadingKey, pdfTextMatches, resolvePdfHighlight, visiblePdfChapters } from "../app/pdf-reading.ts";

const mark = { id: "note", page: 2, mode: "original", start: 6, end: 11, quote: "trade", color: "yellow", note: "Check this claim.", createdAt: "2026-10-10T00:00:00Z" };

test("reading state preserves page offsets, personal bookmarks and notes per book", () => {
  const state = { ...emptyPdfReadingState(), location: { page: 3, offset: 0.42 }, bookmarks: [{ id: "bookmark", page: 3, offset: 0.42, label: "Return here", createdAt: mark.createdAt }], highlights: [mark] };
  assert.deepEqual(parsePdfReadingState(JSON.stringify(state)), state);
  assert.notEqual(pdfReadingKey("first"), pdfReadingKey("second"));
  assert.deepEqual(parsePdfReadingState(null), emptyPdfReadingState());
  assert.throws(() => parsePdfReadingState("invalid"));
  assert.throws(() => parsePdfReadingState(JSON.stringify({ ...state, version: 99 })));
  assert.throws(() => parsePdfReadingState(JSON.stringify({ ...state, location: { page: 0, offset: -1 } })));
  assert.equal(parsePdfReadingState(JSON.stringify({ ...state, highlights: [mark, { ...mark, end: -1 }, { ...mark, mode: "script" }] })).highlights.length, 1);
});

test("collapsing a parent hides all descendants and preserves following siblings", () => {
  const chapters = [{ id: "a", title: "A", level: 0, page: 1 }, { id: "b", title: "B", level: 1, page: 1 }, { id: "c", title: "C", level: 2, page: 2 }, { id: "d", title: "D", level: 0, page: 3 }];
  assert.deepEqual(visiblePdfChapters(chapters, new Set(["a", "b"])).map(item => item.id), ["a", "d"]);
  assert.deepEqual(visiblePdfChapters(chapters, new Set(["b"])).map(item => item.id), ["a", "b", "d"]);
  assert.deepEqual(visiblePdfChapters(chapters, new Set()).map(item => item.id), ["a", "b", "c", "d"]);
});

test("search is literal, case insensitive and tolerates PDF line breaks", () => {
  assert.deepEqual(pdfTextMatches("Trade\nacross trade across", "TRADE across"), [{ start: 0, end: 12 }, { start: 13, end: 25 }]);
  assert.deepEqual(pdfTextMatches("A .* literal [word]", ".*"), [{ start: 2, end: 4 }]);
  assert.deepEqual(pdfTextMatches("A [word]", "[word]"), [{ start: 2, end: 8 }]);
  assert.deepEqual(pdfTextMatches("trade", "  "), []);
  assert.deepEqual(pdfTextMatches("😀 trade", "trade"), [{ start: 3, end: 8 }]);
});

test("highlights relocate only when changed text has one unambiguous quote", () => {
  assert.deepEqual(resolvePdfHighlight("Coast trade", mark), { start: 6, end: 11 });
  assert.deepEqual(resolvePdfHighlight("New coastal trade", mark), { start: 12, end: 17 });
  assert.equal(resolvePdfHighlight("trade and trade", { ...mark, start: 99, end: 104 }), null);
  assert.equal(resolvePdfHighlight("This translation changed", mark), null);
});

test("a selection covering two lines creates separate page-relative highlight boxes", () => {
  const lines = [{ text: "first", start: 0, end: 5, rect: [0.1, 0.1, 0.5, 0.03] }, { text: "second", start: 6, end: 12, rect: [0.1, 0.2, 0.6, 0.03] }];
  const rects = pdfRangeRects(lines, { start: 2, end: 9 });
  assert.equal(rects.length, 2);
  assert.ok(Math.abs(rects[0][0] - 0.3) < 1e-9);
  assert.ok(Math.abs(rects[1][2] - 0.3) < 1e-9);
  assert.deepEqual(pdfRangeRects(lines, { start: 5, end: 6 }), []);
});

test("notes export preserves quotes and multiline notes in page order", () => {
  const markdown = exportPdfNotes("My book", [{ ...mark, page: 8 }, { ...mark, page: 1, quote: "line one\nline two", note: "Thought one\nThought two" }]);
  assert.ok(markdown.indexOf("第 1 页") < markdown.indexOf("第 8 页"));
  assert.match(markdown, /> line one\n> line two/);
  assert.match(markdown, /Thought one\nThought two/);
});
