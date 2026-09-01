import assert from "node:assert/strict";
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
  assert.match(html, /阅读与翻译/);
  assert.match(html, /阅读模式/);
  assert.match(html, /术语库/);
  assert.match(html, /质量检查/);
  assert.doesNotMatch(html, /规训的历史时刻/);
  assert.match(html, /https:\/\/jieyi\.openai\.site\/og\.png/);
  assert.doesNotMatch(html, /codex-preview|SkeletonPreview|react-loading-skeleton/);
});
