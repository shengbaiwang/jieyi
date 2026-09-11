import assert from "node:assert/strict";
import test from "node:test";
import { createTextHistory, recordText, travelText, sourceSelectionOffsets, joinSoftLines } from "../app/source-editing.ts";

const snapshot = (text, start = text.length, end = start) => ({ text, start, end });

test("undo restores the replaced selection and redo restores its replacement", () => {
  const before = snapshot("First wrong sentence", 6, 11);
  const after = snapshot("First correct sentence", 13);
  const edited = recordText(createTextHistory(before.text), before, after);
  const undone = travelText(edited, "undo");
  assert.deepEqual(undone.present, before);
  assert.deepEqual(travelText(undone, "redo").present, after);
});

test("continuous typing is one undo step; a new edit after undo discards redo", () => {
  let history = createTextHistory("Original");
  history = recordText(history, snapshot("Original"), snapshot("Original a"));
  history = recordText(history, snapshot("Original a"), snapshot("Original abc"), true);
  assert.equal(history.past.length, 1);
  history = travelText(history, "undo");
  assert.equal(history.present.text, "Original");
  history = recordText(history, history.present, snapshot("Original new"));
  assert.equal(history.future.length, 0);
  assert.equal(travelText(history, "redo").present.text, "Original new");
});

test("history boundaries are harmless and retain only the latest 100 edits", () => {
  let history = createTextHistory("0");
  assert.equal(travelText(history, "undo"), history);
  assert.equal(travelText(history, "redo"), history);
  for (let index = 1; index <= 120; index++) history = recordText(history, history.present, snapshot(String(index)));
  assert.equal(history.past.length, 100);
  for (let index = 0; index < 100; index++) history = travelText(history, "undo");
  assert.equal(history.present.text, "20");
});

test("line cleanup joins extraction wraps while preserving paragraph boundaries", () => {
  assert.equal(joinSoftLines("First line\n continued.\n\nSecond\nparagraph.\n \nLast."), "First line continued.\n\nSecond paragraph.\n \nLast.");
  assert.equal(joinSoftLines("One\r\ntwo\r\n\r\nThree"), "One two\n\nThree");
  assert.equal(joinSoftLines("Already one paragraph."), "Already one paragraph.");
});

test("split offsets match Python code points for emoji and astral characters", () => {
  const text = "📖 前言\n\n𠮷野家在这里。\n尾声";
  const start = text.indexOf("𠮷");
  const end = text.indexOf("。") + 1;
  const offsets = sourceSelectionOffsets(text, start, end);
  assert.equal(Array.from(text).slice(offsets.start, offsets.end).join(""), "𠮷野家在这里。");
  assert.deepEqual(sourceSelectionOffsets("plain text", 6, 10), { start: 6, end: 10 });
});
