import assert from "node:assert/strict";
import test from "node:test";
import { createSubmissionGate } from "../app/submission-gate.ts";

test("same-tick clicks cannot pass asynchronous configuration preflight twice", async () => {
  const gate = createSubmissionGate();
  let unblock;
  const settings = new Promise(resolve => { unblock = resolve; });
  let requests = 0;
  async function start() {
    const release = gate.acquire();
    if (!release) return;
    try { await settings; requests++; }
    finally { release(); }
  }
  const first = start();
  await start();
  assert.equal(requests, 0);
  unblock();
  await first;
  assert.equal(requests, 1);
  await start();
  assert.equal(requests, 2);
});

test("failures release the gate and an old cleanup cannot release a new submission", () => {
  const gate = createSubmissionGate();
  const first = gate.acquire();
  try { throw new Error("settings unavailable"); }
  catch { /* a later explicit attempt should remain possible */ }
  finally { first(); }
  const second = gate.acquire();
  assert.ok(second);
  first();
  assert.equal(gate.acquire(), null);
  second();
  assert.ok(gate.acquire());
});
