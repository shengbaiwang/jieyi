import assert from "node:assert/strict";
import test from "node:test";

import {
  availableCompareModels,
  blindCandidateLabel,
  blindCandidateLabels,
  formatCompareUsage,
  groupCandidatesByModel,
  modelOptionKey,
  shuffleCompareItems,
  toggleModelKey,
} from "../app/model-compare.ts";

test("groups candidates by model and keeps the newest first", () => {
  const groups = groupCandidatesByModel([
    {
      id: "c1",
      job_id: "j1",
      stage: "draft",
      provider: "profile:a",
      model: "glm",
      text: "旧稿",
      prompt_tokens: 1,
      completion_tokens: 2,
      cost_usd: 0,
      created_at: "2026-01-01T00:00:00Z",
    },
    {
      id: "c2",
      job_id: "j2",
      stage: "compare",
      provider: "profile:a",
      model: "glm",
      text: "新稿",
      prompt_tokens: 1,
      completion_tokens: 2,
      cost_usd: 0,
      created_at: "2026-01-02T00:00:00Z",
    },
    {
      id: "c3",
      job_id: "j2",
      stage: "compare",
      provider: "profile:b",
      model: "kimi",
      text: "另一模型",
      prompt_tokens: 1,
      completion_tokens: 2,
      cost_usd: 0,
      created_at: "2026-01-03T00:00:00Z",
    },
  ]);
  assert.equal(groups.length, 2);
  const glm = groups.find((item) => item.model === "glm");
  assert.ok(glm);
  assert.equal(glm.latest?.text, "新稿");
  assert.equal(glm.candidates.length, 2);
});

test("toggles model keys without duplicates", () => {
  const key = modelOptionKey("profile:a", "glm");
  const next = toggleModelKey([key], key);
  assert.deepEqual(next, []);
  assert.deepEqual(toggleModelKey([], key), [key]);
});

test("lists models from profiles and draft binding", () => {
  const options = availableCompareModels({
    profiles: [
      { id: "a", name: "A", provider_type: "openai-compatible", base_url: "https://a", selected_models: ["m1", "m2"] },
      { id: "b", name: "B", provider_type: "custom", base_url: "https://b", selected_models: ["m3"] },
    ],
    presets: [{ id: "openai-compatible", default_models: ["preset"] }],
    draft_profile_id: "a",
    draft_model: "m1",
  });
  assert.equal(options.length, 3);
  assert.ok(options.some((item) => item.provider === "profile:a" && item.model === "m1"));
  assert.ok(options.some((item) => item.provider === "profile:b" && item.model === "m3"));
});

test("formats usage including reasoning tokens", () => {
  const line = formatCompareUsage({
    prompt_tokens: 12,
    completion_tokens: 8,
    reasoning_tokens: 4,
    cost_usd: 0.000123,
  });
  assert.match(line, /输入 12/);
  assert.match(line, /推理 4/);
  assert.match(line, /\$0\.000123/);
});

test("blind labels never leak provider or model names", () => {
  assert.equal(blindCandidateLabel(0), "译文 A");
  assert.equal(blindCandidateLabel(1), "译文 B");
  assert.deepEqual(blindCandidateLabels(3), ["译文 A", "译文 B", "译文 C"]);
  for (const label of blindCandidateLabels(4)) {
    assert.doesNotMatch(label, /glm|kimi|openai|profile:/i);
  }
});

test("shuffles compare columns so model slots are not fixed", () => {
  const items = ["glm", "kimi", "deepseek"];
  const shuffled = shuffleCompareItems(items, () => 0);
  assert.equal(shuffled.length, 3);
  assert.deepEqual([...shuffled].sort(), [...items].sort());
  // high random always swaps with self → identity; low random moves first slot
  assert.deepEqual(shuffleCompareItems(items, () => 0.99), items);
  assert.notDeepEqual(shuffleCompareItems(items, () => 0), items);
  // same call is a permutation, not a reorder of labels only
  assert.equal(new Set(shuffleCompareItems(items, () => 0.3)).size, 3);
});
