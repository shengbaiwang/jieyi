import assert from "node:assert/strict";
import test from "node:test";
import { activeChapter, toggleChapter, normalizeSegmentRanges } from "../app/chapter-navigation.ts";

const chapters = [
  { id: "index", title: "Index", level: 0, start_ordinal: 2980, end_ordinal: 2995 },
  ...["F", "G", "H", "I"].map(id => ({ id, title: id, level: 1, start_ordinal: 2980, end_ordinal: 2995 })),
  { id: "J", title: "J", level: 1, start_ordinal: 2996, end_ordinal: 3003 },
];

test("same-position directory clicks select exactly one entry, including its parent", () => {
  for (const clicked of chapters) {
    const selected = activeChapter(chapters, clicked.start_ordinal, clicked.id);
    assert.deepEqual(chapters.filter(item => item.id === selected?.id), [clicked]);
  }
});

test("position tracking preserves the clicked sibling and resolves new boundaries", () => {
  assert.equal(activeChapter(chapters, 2985, "H").id, "H");
  assert.equal(activeChapter(chapters, 2996, "H").id, "J");
  assert.equal(activeChapter(chapters, 2980).id, "F");
  assert.equal(activeChapter(chapters, 2979, "H"), null);
  assert.equal(activeChapter([], 0), null);
  assert.equal(activeChapter(chapters, 2980, "other-document:H").id, "F");
});

test("independent IDs survive refresh, duplicate labels and structural ordinal shifts", () => {
  const refreshed = JSON.parse(JSON.stringify(chapters));
  refreshed.find(item => item.id === "G").title = "F";
  for (const item of refreshed) { item.start_ordinal += 2; item.end_ordinal += 2; }
  assert.equal(activeChapter(refreshed, 2982, "G").id, "G");
});

test("checkbox toggles never select or deselect same-position siblings", () => {
  let ids = [];
  ids = toggleChapter(ids, "G");
  assert.deepEqual(chapters.filter(item => ids.includes(item.id)).map(item => item.id), ["G"]);
  ids = toggleChapter(ids, "H");
  ids = toggleChapter(ids, "G");
  assert.deepEqual(ids, ["H"]);
  ids = toggleChapter(chapters.map(item => item.id), "G");
  assert.deepEqual(ids, ["index", "F", "H", "I", "J"]);
});

test("separately selected overlapping chapters translate each segment only once", () => {
  const selected = chapters.filter(item => ["index", "G", "H", "J"].includes(item.id));
  const ranges = normalizeSegmentRanges(selected.map(item => [item.start_ordinal, item.end_ordinal]));
  assert.deepEqual(ranges, [[2980, 3003]]);
  assert.equal(ranges.reduce((sum, [start, end]) => sum + end - start + 1, 0), 24);
});
