import assert from "node:assert/strict";
import test from "node:test";
import { endpointPreview, modelSearch, selectedModels, validBaseUrl } from "../app/model-config.ts";

test("legacy bound IDs stay available without adding all provider presets", () => {
  assert.deepEqual(selectedModels(null, ["legacy-model", "legacy-model", ""]), ["legacy-model"]);
  assert.deepEqual(selectedModels([], []), []);
  assert.deepEqual(selectedModels(["a", " b ", "a"], ["c", ""]), ["a", "b", "c"]);
});

test("model search handles namespaces, case and multiple search terms", () => {
  const models = ["vendor/Model-Flash", "vendor/Model-Pro", "other/Model-Flash"];
  assert.deepEqual(modelSearch(models, " VENDOR flash "), ["vendor/Model-Flash"]);
  assert.deepEqual(modelSearch(models, "missing"), []);
  assert.deepEqual(modelSearch(models, ""), models);
});

test("request address preview matches explicit paths without inserting a version", () => {
  assert.equal(endpointPreview(" https://example.com/api/v4/ ", "/chat/completions"), "https://example.com/api/v4/chat/completions");
  assert.equal(endpointPreview("https://example.com/v1", "https://relay.example/messages/"), "https://relay.example/messages");
  assert.equal(validBaseUrl("localhost:11434"), false);
  assert.equal(validBaseUrl("file:///settings"), false);
  assert.equal(validBaseUrl("http://localhost:11434/v1"), true);
});
