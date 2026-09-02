import assert from "node:assert/strict";
import test from "node:test";
import { createReaderNavigation, ordinalAtOffset } from "../app/reader-navigation.ts";

// Geometry-only unit fixture: iframe measurements arrive independently of clicks.
function fixture(mode = "epub") {
  const win = new EventTarget();
  const frames = [];
  const sections = [];
  const positions = [];
  const pending = [];
  let unavailable = 0;
  let toolbarHeight = 80;
  let tick = 0;
  const tasks = new Map();
  win.requestAnimationFrame = fn => { tasks.set(++tick, fn); return tick; };
  win.cancelAnimationFrame = id => tasks.delete(id);
  const flush = () => { for (const [id, fn] of [...tasks]) { tasks.delete(id); fn(); } };
  const previousObserver = globalThis.ResizeObserver;
  let resized;
  globalThis.ResizeObserver = class {
    constructor(fn) { resized = fn; }
    observe() {}
    disconnect() {}
  };
  const root = new EventTarget();
  root.ownerDocument = { defaultView: win };
  root.scrollTop = 0;
  root.getBoundingClientRect = () => ({ top: 100 });
  root.scrollTo = ({ top }) => { root.scrollTop = top; };
  for (let index = 0; index < 2; index++) {
    const frame = { dataset: { spineIndex: String(index) }, loading: "lazy", style: { height: "96px" },
      contentWindow: { postMessage() {} } };
    const section = { dataset: { readerOrdinal: String(index * 50), readerPage: String(index + 1) },
      classList: { contains: () => mode === "fixed" },
      querySelector: () => mode === "plain" ? null : frame,
      getBoundingClientRect: () => {
        const top = 100 + 100 + (index ? Number.parseFloat(frames[0].style.height) : 0) - root.scrollTop;
        return { top, bottom: top + Number.parseFloat(frame.style.height) };
      } };
    frame.closest = () => section;
    frames.push(frame);
    sections.push(section);
  }
  root.querySelectorAll = selector => selector.startsWith("iframe") ? (mode === "plain" ? [] : frames) : sections;
  root.querySelector = selector => {
    if (selector === ".reader-toolbar") return { getBoundingClientRect: () => ({ height: toolbarHeight }) };
    const match = selector.match(/(?:spine-|segment-|index=")(\d+)/);
    return match ? (selector.startsWith("iframe") ? (mode === "plain" ? [] : frames) : sections)[selector.includes("segment-") ? Number(match[1]) / 50 : Number(match[1])] : null;
  };
  const nav = createReaderNavigation(root, {
    documentId: "book", targets: mode === "plain" ? {} : { 0: { spineIndex: 0, pageNumber: 1 },
      50: { spineIndex: 1, pageNumber: 2 }, 60: { spineIndex: 1, pageNumber: 2 } },
    onPosition: (ordinal, page) => positions.push([ordinal, page]),
    onHeight() {}, onPending: value => pending.push(value), onUnavailable: () => unavailable++,
  });
  const message = (index, data, source = frames[index].contentWindow) => {
    const event = new Event("message");
    Object.assign(event, { source, data: { documentId: "book", spineIndex: index, ...data } });
    win.dispatchEvent(event);
    flush();
  };
  return { nav, root, frames, positions, pending, message, flush,
    resizeToolbar(height) { toolbarHeight = height; resized(); flush(); },
    unavailable: () => unavailable,
    close() { nav.destroy(); globalThis.ResizeObserver = previousObserver; } };
}

test("directory loads distant pages and stays aligned through delayed page and toolbar resizing", () => {
  const f = fixture();
  try {
    f.nav.goTo(50);
    assert.equal(f.frames[1].loading, "eager");
    assert.equal(f.pending.at(-1), true);
    f.message(1, { type: "jy-epub-resize", height: 3000, locations: [{ ordinal: 50, top: 400 }] });
    assert.equal(f.root.scrollTop, 504);
    assert.equal(f.pending.at(-1), false);
    f.message(0, { type: "jy-epub-resize", height: 1800, locations: [{ ordinal: 0, top: 0 }] });
    assert.equal(f.root.scrollTop, 2208);
    f.resizeToolbar(120);
    assert.equal(f.root.scrollTop, 2168);
    assert.deepEqual(f.positions.at(-1), [50, 2]);
  } finally { f.close(); }
});

test("last click wins and user scrolling releases the navigation anchor", () => {
  const f = fixture();
  try {
    f.nav.goTo(50);
    f.nav.goTo(60);
    f.message(1, { type: "jy-epub-resize", height: 3000,
      locations: [{ ordinal: 50, top: 200 }, { ordinal: 60, top: 900 }] });
    const top = f.root.scrollTop;
    f.message(1, { type: "jy-epub-location", segmentOrdinal: 50, top: 200 });
    assert.equal(f.root.scrollTop, top);
    f.message(1, { type: "jy-epub-location", segmentOrdinal: 60, top: 12 }, {});
    assert.equal(f.root.scrollTop, top, "unrelated window messages are ignored");
    f.message(1, { type: "jy-epub-interact" });
    f.root.scrollTop = 400;
    f.root.dispatchEvent(new Event("scroll"));
    f.flush();
    assert.deepEqual(f.positions.at(-1), [50, 2]);
    f.message(1, { type: "jy-epub-location", segmentOrdinal: 60, top: 900 });
    assert.equal(f.root.scrollTop, 400);
    f.nav.destroy();
    f.message(1, { type: "jy-epub-resize", height: 5000 });
    assert.equal(f.frames[1].style.height, "3000px");
  } finally { f.close(); }
});

test("missing anchors fall back to their page and clear the loading indicator", () => {
  const f = fixture();
  try {
    f.nav.goTo(50);
    f.message(1, { type: "jy-epub-location", segmentOrdinal: 50, found: false, top: null });
    assert.equal(f.pending.at(-1), false);
    assert.equal(f.unavailable(), 1);
    assert.equal(f.root.scrollTop, 104);
  } finally { f.close(); }
});

test("scroll tracking distinguishes multiple chapters in one spine", () => {
  const entries = [{ ordinal: 4, top: 100 }, { ordinal: 18, top: 850 }, { ordinal: 33, top: 1700 }];
  assert.equal(ordinalAtOffset(entries, 50), 4);
  assert.equal(ordinalAtOffset(entries, 900), 18);
  assert.equal(ordinalAtOffset(entries, 2000), 33);
  assert.equal(ordinalAtOffset([], 0), undefined);
});

for (const mode of ["plain", "fixed"]) {
  test(`${mode} reading jumps without waiting for iframe position messages`, () => {
    const f = fixture(mode);
    try {
      f.nav.goTo(50);
      assert.equal(f.root.scrollTop, 104);
      assert.equal(f.pending.at(-1), false);
      assert.equal(f.positions.at(-1)[0], 50);
      f.resizeToolbar(120);
      assert.equal(f.root.scrollTop, 64);
    } finally { f.close(); }
  });
}
