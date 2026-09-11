export type TextSnapshot = { text: string; start: number; end: number };
export type TextHistory = { past: TextSnapshot[]; present: TextSnapshot; future: TextSnapshot[] };

export function createTextHistory(text: string): TextHistory {
  return { past: [], present: { text, start: 0, end: 0 }, future: [] };
}

export function recordText(history: TextHistory, before: TextSnapshot, after: TextSnapshot, group = false): TextHistory {
  if (before.text === after.text) return history;
  return { past: group && history.past.length ? history.past : [...history.past, before].slice(-100), present: after, future: [] };
}

export function travelText(history: TextHistory, direction: "undo" | "redo"): TextHistory {
  if (direction === "undo") {
    const previous = history.past.at(-1);
    return previous ? { past: history.past.slice(0, -1), present: previous, future: [history.present, ...history.future] } : history;
  }
  const next = history.future[0];
  return next ? { past: [...history.past, history.present], present: next, future: history.future.slice(1) } : history;
}

// Textareas count UTF-16 units; the Python API counts Unicode code points.
export function sourceSelectionOffsets(text: string, start: number, end: number) {
  return { start: Array.from(text.slice(0, start)).length, end: Array.from(text.slice(0, end)).length };
}

export function joinSoftLines(text: string) {
  return text.replace(/\r\n?/g, "\n").split(/(\n[\t ]*\n+)/).map((part, index) => index % 2 ? part : part.replace(/[\t ]*\n[\t ]*/g, " ")).join("");
}
