import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("https://jieyi.example/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the Jieyi translation workbench", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>介译 · 翻译工作台<\/title>/);
  assert.match(html, /我的书库/);
  // The sidebar splits into a global group (always rendered) and a per-book
  // group (rendered only once a document is open, so absent from the empty SSR).
  assert.match(html, /<nav aria-label="全局导航">[\s\S]*书库 <b>[\s\S]*导入书籍<\/button>[\s\S]*模型配置<\/button>/);
  assert.doesNotMatch(html, /<nav aria-label="当前书">/);
  assert.match(html, /m10.7 3.1 2.55-.55/);
  assert.doesNotMatch(html, /阅读与翻译|阅读模式/);
  assert.doesNotMatch(html, /[①②③]/);
  assert.doesNotMatch(html, /规训的历史时刻/);
  assert.match(html, /https:\/\/jieyi\.openai\.site\/og\.png/);
  assert.doesNotMatch(html, /codex-preview|SkeletonPreview|react-loading-skeleton/);
});


test("night mode inherits theme text and forbids hard-coded black text", async () => {
  const [globalCss, setupCss, setupSource] = await Promise.all([
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../app/setup-panels.css", import.meta.url), "utf8"),
    readFile(new URL("../app/setup-panels.tsx", import.meta.url), "utf8"),
  ]);
  const css = globalCss + "\n" + setupCss;

  assert.match(globalCss, /\.desktop-shell\s*\{[^}]*color:\s*var\(--text\)/s);
  assert.match(setupCss, /\.import-card input, \.import-card select \{ height: 34px; min-height: 34px; \}/);
  assert.match(setupCss, /\.import-card > label \+ label, \.import-card > \.two-fields \+ label, \.import-card > label \+ \.two-fields/);
  assert.doesNotMatch(setupSource, /className="import-steps"|导入原文，介译会保留标题结构并生成稳定段落。/);
  assert.doesNotMatch(css, /(?:^|[;{]\s*)color\s*:\s*(?:black|#0{3}(?:0{3})?\b|rgb\(\s*0\s*,\s*0\s*,\s*0\s*\))/im);
});

test("reader opens directly, locates exact EPUB chapters, and lazy-loads pages", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const openReader = source.slice(source.indexOf("async function openReader"), source.indexOf("function jumpToReaderChapter"));
  assert.doesNotMatch(openReader, /openDocument\(/);
  assert.match(source, /manifest\.segment_locations/);
  assert.match(source, /jumpToReaderChapter\(chapter\)/);
  assert.match(source, /createReaderNavigation/);
  assert.match(source, /id=\{`reader-spine-\$\{item\.spine_index\}`\}/);
  assert.match(source, /readerNavigation.current\?\.goTo/);
  assert.match(source, /readerCurrentPage/);
  assert.match(source, /loading=\{index < 2 \? "eager" : "lazy"\}/);
});
